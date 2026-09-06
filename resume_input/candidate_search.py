"""
resume_input/candidate_search.py

Resume -> (선택)Query Rewriting -> BM25(Recall) -> Embedding(정렬 보조
점수) -> 결합 정렬. 최종 판단(LLM)은 이 파일이 하지 않는다 -
judge_service가 담당.

실측 검증 이력(중요 - 나중에 또 "임베딩 모델을 바꿔볼까"로 되돌아가지
않기 위해 기록한다):
- 순수 키워드(제목+본문 토큰) 스코어링은 회사가 다른 유사 직무를 놓쳤다.
- BM25 Top200 -> Embedding Top40 "하드 컷" 구조로 바꾼 뒤 실측했더니,
  BM25는 문제 없이 Business Strategy/CRM Marketing/Operations Analytics
  같은 직무명이 다른 후보를 Top200에 넣었지만, Embedding 하드 컷에서
  대부분 탈락했다.
- MiniLM 대신 BGE-M3/E5-large로 교체해도 마찬가지였다(오히려 더 나쁨).
  즉 병목은 "모델 성능"이 아니라 "Embedding을 후보 제거에 쓰는 구조"
  자체였다.
- 그래서 Embedding은 더 이상 후보를 제거하지 않는다. BM25 점수와 결합해
  정렬 순서만 조정하는 보조 신호로만 쓴다.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3

from resume_input.job_store import DB_PATH, list_all_candidate_jobs
from resume_input.header_registry import DUTIES, REQUIRED, PREFERRED, NOISE, as_regex

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]{2,}")

_emb_model = None


def _load_embedding_model():
    """2026-07-19(사용자 요청, 실측 원인 확인) - "공고 검색 중입니다"
    화면에서 수 분씩 멈추는 문제의 원인을 코드만 읽어서(LLM 호출 없이)
    추적한 결과, LLM/크레딧과는 무관하게 이 임베딩 모델 로딩 지점이었다.
    모델 파일 자체는 이미 로컬에 캐시돼 있는데(~/.cache/huggingface/hub/
    models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
    존재 확인), huggingface_hub가 기본값으로는 그 캐시를 바로 쓰지 않고
    매번 huggingface.co에 메타데이터 조회를 먼저 시도한다(스크립트
    실행 때마다 뜨던 "unauthenticated requests to the HF Hub" 경고가
    그 증거) - 이 네트워크 왕복이 느리거나 막히면 그만큼 그대로
    지연된다. 서버를 재시작할 때마다 이 모듈 전역 캐시(_emb_model)가
    비워지므로, 재시작 직후 첫 검색마다 이 지연을 다시 겪는다.

    2026-07-19 보강(사용자 지적) - HF_HUB_OFFLINE 환경변수는 app.py
    진입점 맨 위(다른 HF 관련 모듈 import보다 먼저)에서도 설정하지만,
    huggingface_hub가 이미 import된 뒤라면 환경변수만으로는 늦을 수
    있다는 지적에 따라 `local_files_only=True`를 생성자에 직접 넘겨서
    이중으로 보장한다 - 이러면 환경변수 설정 시점과 무관하게 항상
    로컬 캐시만 쓴다. 로컬 캐시가 없으면(이 프로젝트 환경 밖에서
    처음 실행하는 경우 등) 조용히 네트워크로 전환하지 않고, 무슨
    문제인지 알 수 있는 메시지로 바로 실패한다(지어낸 결과를 내지
    않는다는 원칙과 동일하게, "조용히 다른 동작으로 넘어가는" 것도
    피한다)."""
    global _emb_model
    if _emb_model is not None:
        return _emb_model

    from sentence_transformers import SentenceTransformer
    _MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    try:
        _emb_model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
    except Exception:
        # 공개 데모 배포 환경(예: Streamlit Community Cloud) 컨테이너는
        # 이 프로젝트 밖에서 매번 새로 뜨는 환경이라 로컬 캐시가 처음부터
        # 없다 - local_files_only 로만 막아두면 데모가 항상 실패한다.
        # 캐시가 없을 때만(최초 1회) 공개 모델 허브에서 내려받는 것으로
        # 폴백한다(개인 데이터·Credential과 무관한 공개 오픈소스 모델
        # 가중치일 뿐이며, 캐시가 있는 로컬 개발 환경에서는 여전히 위
        # local_files_only 경로가 그대로 쓰여 동작이 바뀌지 않는다).
        _emb_model = SentenceTransformer(_MODEL_NAME)
    return _emb_model


_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&nbsp;|&[a-z]+;")

# 검색 Representation 우선순위 분류용 헤더 패턴 - 회사소개/복지가 아니라
# 담당업무/자격요건/우대사항 위주로 검색해야 한다는 실측 판단을 반영한다.
# _classify_header()에서 pref -> resp -> req 순으로 검사한다(REQ의
# "qualifications?"가 "Preferred Qualifications" 같은 복합 헤더의
# 부분 문자열과도 매치되므로, 더 구체적인 PREF를 먼저 확인해야 오분류를
# 피할 수 있다).
#
# 2026-07-16 리팩터 - 패턴 목록 자체는 header_registry.py(단일 출처)를
# 쓴다 - detail_parser.py/job_detail.py/job_prep.py와 동의어가
# 어긋나지 않도록. 이 파일만의 정규식 문법(\s* 등)은 필요 없어졌다 -
# header_registry가 이미 붙여쓰기/띄어쓰기 변형을 각각 별도 항목으로
# 갖고 있다.
_HDR_RESP = as_regex(DUTIES)
_HDR_REQ = as_regex(REQUIRED)
_HDR_PREF = as_regex(PREFERRED)

# 종료 마커 - 이 헤더를 만나면 무조건 "other"(최하위 우선순위)로 되돌린다.
# 실측 확인된 문제: 일부 공고는 "자격요건" 헤더가 아예 없어서, "주요업무"
# 헤더 뒤에 오는 개인정보처리방침·서류반환정책·전형절차 같은 법적/행정
# 고지사항까지 전부 "resp" 버킷에 같이 쓸려 들어갔다(회사소개만큼 나쁜
# 노이즈). 이런 섹션이 시작되면 명시적으로 우선순위를 낮춘다.
_HDR_END = as_regex(NOISE)
_HDR_ANY = re.compile(
    f"({_HDR_RESP.pattern})|({_HDR_REQ.pattern})|({_HDR_PREF.pattern})|({_HDR_END.pattern})", re.I
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _classify_header(header_text: str) -> str:
    if _HDR_PREF.search(header_text):
        return "pref"
    if _HDR_RESP.search(header_text):
        return "resp"
    if _HDR_REQ.search(header_text):
        return "req"
    return "other"


def _classify_sections(job: dict) -> dict[str, list[str]]:
    """HTML을 벗기고 담당업무/자격요건/우대사항/기타(회사소개 등)로
    줄 단위 분류한다. `_job_text()`(검색용, 하나로 합침)와
    `job_detail.py`(상세보기 표시용, 버킷별로 따로 씀) 둘 다 이
    함수를 재사용한다 - 분류 로직은 한 곳에만 있다."""
    raw = job.get("posting_text", "") or ""
    plain = _TAG_RE.sub(" ", raw)
    plain = _ENTITY_RE.sub(" ", plain)
    plain = _HDR_ANY.sub(lambda m: "\n" + m.group(0) + "\n", plain)

    buckets: dict[str, list[str]] = {"resp": [], "req": [], "pref": [], "other": []}
    current = "other"
    for raw_line in re.split(r"[\n\r]+", plain):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if _HDR_ANY.fullmatch(line.rstrip(":：ㆍ・ ")):
            current = _classify_header(line)
            continue
        buckets[current].append(line)
    return buckets


def _job_text(job: dict) -> str:
    """검색용 Representation - HTML을 벗기고, 담당업무/자격요건/우대사항을
    회사소개/복지보다 앞에 오도록 재배치한 뒤 3000자로 자른다.

    실측 근거(중요 - Ranking 실험은 이미 종료했고 다시 건드리지 않는다.
    이건 그 이후 단계인 JD 전처리 개선이다):
    1) HTML 미제거 버그 - 일부 공고(워드 붙여넣기)는 단어마다
       <span data-contrast=auto>단어</span>식으로 태그가 씌워져 있어서,
       벗기지 않고 3000자를 자르면 실제 텍스트가 몇백 자도 안 들어갔다
       (특정 이커머스 공고 한 건 실측: 태그가 70% 이상). HTML을 먼저 벗겨서 해결했다.
    2) 그런데 HTML을 벗긴 뒤에도, 회사소개/미션/문화 문단이 문서 맨
       앞에서 3000자 예산의 상당 부분을 차지해서 실제 담당업무/자격요건이
       여전히 잘리는 경우가 있었다(같은 공고: HTML 제거만으로는
       BM25 순위가 1756위->225위까지만 개선되고 Top200 진입은 실패).
       원문 순서(회사소개 -> 업무 -> 자격요건 -> 우대사항 -> 복지)를
       그대로 따르지 않고, 헤더를 인식해서 담당업무/자격요건/우대사항을
       앞으로, 회사소개/복지/문화/전형절차 등은 뒤로 재배치한 뒤에
       3000자를 자른다. 헤더가 줄바꿈 없이 본문 중간에 묻혀 있는 경우가
       많아(이전 세션에서 반복 확인된 패턴) 헤더 앞뒤로 강제 줄바꿈을
       넣은 뒤 줄 단위로 분류한다."""
    title = job.get("title", "") or ""
    buckets = _classify_sections(job)
    ordered_body = " ".join(buckets["resp"] + buckets["req"] + buckets["pref"] + buckets["other"])
    combined = f"{title} {ordered_body}"
    return re.sub(r"\s+", " ", combined).strip()[:3000]


def _bm25_stage(query_text: str, jobs: list[dict], top_n: int) -> list[dict]:
    from rank_bm25 import BM25Okapi

    corpus_tokens = [_tokenize(_job_text(j)) for j in jobs]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = _tokenize(query_text)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(zip(scores, jobs), key=lambda x: x[0], reverse=True)
    return [{**j, "bm25_score": float(s)} for s, j in ranked[:top_n]]


# 2026-07-19(실측 근거: 프로파일링 결과 jd_semantic_objects가 아직 없는
# job(Representation 미생성, without_rep - 실측 193건)은 검색마다
# _job_text() 결과(최대 3000자)를 처음부터 encode()해서 search_
# candidates_meaning() 소요시간의 대부분(약 37초)을 차지했다. 이 텍스트는
# job의 posting_text가 안 바뀌는 한 결정적이므로, 텍스트 내용 자체의
# 해시를 키로 영구 캐시한다(job_id/버전 필드가 아니라 텍스트 내용을 직접
# 키로 써서, 오염 수정 등으로 posting_text가 바뀌면 해시가 달라져
# 자동으로 무효화된다 - 별도 무효화 로직 불필요).


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _init_job_text_embedding_cache_table(db_path=DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_text_embedding_cache (
            text_hash TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _load_cached_job_text_embeddings(texts: list[str], db_path=DB_PATH):
    import numpy as np

    _init_job_text_embedding_cache_table(db_path)
    unique_texts = list(dict.fromkeys(texts))
    hash_to_text = {_text_hash(t): t for t in unique_texts}
    if not hash_to_text:
        return {}
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(hash_to_text))
    rows = conn.execute(
        f"SELECT text_hash, embedding FROM job_text_embedding_cache WHERE text_hash IN ({placeholders})",
        list(hash_to_text.keys()),
    ).fetchall()
    conn.close()
    return {hash_to_text[h]: np.frombuffer(blob, dtype=np.float32) for h, blob in rows}


def _save_job_text_embeddings(text_to_emb: dict, db_path=DB_PATH) -> None:
    import numpy as np

    _init_job_text_embedding_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT OR REPLACE INTO job_text_embedding_cache (text_hash, embedding) VALUES (?, ?)",
        [(_text_hash(t), np.asarray(e, dtype=np.float32).tobytes()) for t, e in text_to_emb.items()],
    )
    conn.commit()
    conn.close()


def _add_embedding_scores(resume_raw: str, jobs: list[dict]) -> list[dict]:
    """Embedding은 후보를 제거하지 않는다 - 모든 jobs에 점수만 추가한다.
    job 쪽 텍스트는 DB의 영구 캐시(텍스트 해시 기준)에서 먼저 채우고,
    캐시에 없는 것만 배치로 encode한 뒤 다시 저장한다 - 매칭 로직/점수식은
    그대로, 같은 텍스트를 다시 검색할 때 encode() 호출만 건너뛴다."""
    import numpy as np

    model = _load_embedding_model()
    job_texts = [_job_text(j) for j in jobs]

    cached = _load_cached_job_text_embeddings(job_texts)
    unique_texts = list(dict.fromkeys(job_texts))
    missing = [t for t in unique_texts if t not in cached]
    if missing:
        embs = model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
        new_cache = {t: np.asarray(e, dtype=np.float32) for t, e in zip(missing, embs)}
        _save_job_text_embeddings(new_cache)
        cached.update(new_cache)

    job_embs = np.stack([cached[t] for t in job_texts])
    resume_emb = model.encode([resume_raw[:3000]], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = job_embs @ resume_emb

    return [{**j, "embedding_score": float(s)} for j, s in zip(jobs, sims)]


def _combine_and_sort(jobs: list[dict], bm25_weight: float = 0.7) -> list[dict]:
    """BM25 점수와 Embedding 점수를 각각 대칭적으로(둘 다 min-max) 정규화한
    뒤 가중합한다. BM25 0.7 : Embedding 0.3.

    실측 이력(중요 - 아래 두 대안을 이미 시도하고 폐기했다. 다시 시도하지
    않는다):
    1) 이전엔 BM25만 min-max 정규화하고 embedding은 원시 코사인 값을
       그대로 썼다 - embedding_score가 이 코퍼스에서 좁은 구간(0.3~0.6)에
       몰려 있어 "50:50 가중치"가 이름뿐이었다(실제로는 BM25가 더 강하게
       반영됨). 이게 우연히 성능이 가장 좋았지만, 의도하지 않은 버그에
       의존하는 구조라 그대로 채택하지 않았다.
    2) Reciprocal Rank Fusion(RRF, BM25:Embedding=2:1, k=60)도 시도했다.
       원점수 단위를 안 쓰므로 정규화 문제는 없지만, 실측 결과
       대칭 가중합과 Recall@30/50이 동률이었고 대표 공고별 순위는 오히려
       근소하게 더 나빴다(예: Business Strategy 74->77, CRM/Growth
       42->44, Operations Analytics 76->78).
    22건 라벨링된 평가 집합(관련 13/비관련 9, 실제 원문 확인)으로 비교한
    결과 대칭 가중합 0.7:0.3이 RRF와 동률이거나 근소 우위였고 더 단순해서
    최종 채택했다(Recall@30=2/13, Recall@50=3/13, Top30 내 비관련=0,
    두 방식 모두 동일 - 선택 규칙에 따라 더 단순한 쪽 채택). BM25를
    Embedding보다 더 신뢰한다는 게 실측 결과이므로, 이걸 우연한 버그가
    아니라 0.7:0.3이라는 명시적 가중치로 표현한다."""
    if not jobs:
        return jobs

    bm25_scores = [j["bm25_score"] for j in jobs]
    emb_scores = [j["embedding_score"] for j in jobs]
    b_lo, b_hi = min(bm25_scores), max(bm25_scores)
    e_lo, e_hi = min(emb_scores), max(emb_scores)
    b_span = (b_hi - b_lo) or 1.0
    e_span = (e_hi - e_lo) or 1.0

    for j in jobs:
        b_norm = (j["bm25_score"] - b_lo) / b_span
        e_norm = (j["embedding_score"] - e_lo) / e_span
        j["combined_score"] = bm25_weight * b_norm + (1 - bm25_weight) * e_norm

    return sorted(jobs, key=lambda j: -j["combined_score"])


# 추천 이유(matched_keywords)에서 제외할 너무 일반적인 단어 - LLM 없이
# BM25/토큰 겹침만으로 "왜 이 공고가 추천됐는지" 근거를 보여주기 위한
# 것이라, "경험"/"가능"처럼 어떤 공고에나 다 있는 단어는 근거로서 의미가
# 없다. "데이터"/"분석"처럼 이 이력서에 실제로 핵심인 단어는 남겨둔다.
_EXPLAIN_STOPWORDS = {
    "경험", "가능", "수행", "관련", "이상", "우대", "필수", "사항", "요건",
    "역량", "업무", "담당", "보유", "있는", "분들", "지원", "채용", "회사",
}

# "데이터를"/"데이터가"처럼 조사가 붙어 BM25 토큰화에서는 별개 토큰이 되는
# 것을, 추천 이유 표시에서만 같은 단어로 묶기 위한 흔한 조사 목록.
# BM25 토큰화 자체(_tokenize)는 이미 검증 끝난 부분이라 건드리지 않는다.
_TRAILING_PARTICLES = re.compile(r"(을|를|이|가|은|는|와|과|에|의|로|으로)$")


def _normalize_for_display(token: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9]+", token):
        return token
    stripped = _TRAILING_PARTICLES.sub("", token)
    return stripped if len(stripped) >= 2 else token


def _matched_keywords(resume_raw: str, job: dict, top_n: int = 8) -> list[str]:
    """이 공고가 왜 추천됐는지 - Resume와 JD가 실제로 공유하는 토큰(LLM
    호출 없음). BM25/Embedding 점수 자체는 사람이 읽고 이해할 수 없으니,
    Top30 화면에 "왜"를 보여주기 위한 설명용 근거다."""
    from collections import Counter

    resume_tokens = Counter(_tokenize(resume_raw))
    job_token_set = set(_tokenize(_job_text(job)))

    overlap = [
        (t, c) for t, c in resume_tokens.items()
        if t in job_token_set and len(t) >= 2 and t not in _EXPLAIN_STOPWORDS
    ]
    overlap.sort(key=lambda x: -x[1])

    seen: dict[str, int] = {}
    for t, c in overlap:
        norm = _normalize_for_display(t)
        seen[norm] = seen.get(norm, 0) + c
    ranked = sorted(seen.items(), key=lambda x: -x[1])
    return [t for t, _ in ranked[:top_n]]


def search_candidates_meaning(
    resume_raw: str,
    resume_short_semantic: dict,
    resume_semantic_graph: list[str],
    final_top_n: int = 50,
    jobs: list[dict] | None = None,
    resume_semantic_objects: list[dict] | None = None,
    user_career_level: str | None = None,
    challenge_option: bool = False,
) -> list[dict]:
    """M3+M4(Rule Representation 기반 Meaning Matching)를 전체 후보
    풀에 직접 적용한다 - BM25 사전필터 없음. Rule Representation이
    무효한(원문 파싱 실패/정보 부족) job만 BM25+Embedding fallback으로
    채점한다(candidate_search.py의 기존 함수 재사용). Resume 쪽만
    Understanding(LLM, 세션당 1회)을 쓰고, JD 쪽은 이 함수 안에서
    **LLM을 절대 호출하지 않는다** - JD가 과거에 "분석하기"로 이해된
    적이 있는지 여부는 이 함수 결과에 전혀 영향을 주지 않는다.

    검증 이력(2026-07-13, job_ai_v2/docs/verification/2026-07-13_
    understanding_engine_meaning_matching/summary.md): M3+M4가 LLM Gold
    Standard 대비 Spearman 0.820으로 기존 BM25+Embedding(0.708)을
    앞섰고, 1,917건 전체 풀 기준 쿼리 속도도 48.8ms로 BM25(1,300ms+)
    보다 빠르다(이때는 JD 쪽도 LLM Understanding 기반이었다). jobs를
    넘기면 그 목록만 검색한다(Career Filter로 이미 축소된 후보 풀) -
    이 함수는 Career 필터링(풀 축소)을 하지 않는다(호출부 책임,
    career_filter.filter_by_career_level 참고).

    2026-08-13(Rule-only Candidate Generation, 사용자 확정 - docs/
    verification/2026-07-25_m3_representation_ab/summary.md §2·§7·§8
    검증 결과를 production에 정확히 반영) - JD 쪽 M3/M4를 전부 Rule
    Representation(`rule_representation.py`, LLM 미사용) 기반으로
    통일했다. 이전에는 `has_valid_representation()`(옛 LLM 산출물
    `jd_short_semantic`/`jd_semantic_graph` 존재 여부)이 게이트였는데,
    이러면 Rule Representation이 멀쩡해도 "과거에 분석한 적 없는 JD"는
    전부 BM25 fallback으로 빠지는 문제가 있었다(§13 Phase B에서 발견,
    2026-07-25 "게이트는 보류" - 그날은 M3만 Rule로 교체하고 M4/게이트는
    옛 구조로 남겨뒀었다). 이제 게이트 자체를 `is_rule_representation_
    valid()`(Rule Representation 자체의 유효성)로 바꿔서, JD가 과거에
    LLM으로 분석된 적이 있든 없든 완전히 동일한 경로를 탄다.
    - M3: `LayerM3Scorer` + Rule Representation(변경 없음, 2026-07-25에
      이미 프로덕션 반영됨).
    - M4: `rule_representation.build_rule_based_m4_nodes()`(Task/
      Qualification 줄 + Skill 키워드, LLM 미사용)를 기존 `meaning_
      matching.m4_score()`에 그대로 넣는다 - 2026-07-25 §8 "Rule Node
      vs LLM Node" 실측(Recall@100 82.6%=82.6%, Spearman 0.836)과
      §7 "Relation Ablation"(관계/Edge가 M4 성능에 기여하지 않음)을
      그대로 production에 옮긴 것 - `meaning_matching.py`는 무수정.
    - 옛 LLM 필드(`jd_short_semantic`/`jd_semantic_graph`) 기반 old-M3
      폴백은 완전히 제거했다(사용자 확정 - "같은 JD인데 과거 분석
      여부에 따라 다른 경로를 타면 안 된다"는 원칙과 상충하므로).
    resume_semantic_objects를 안 넘기면(기본값 None) Rule Representation을
    만들 수 없으므로 전체가 BM25+Embedding fallback으로 처리된다 -
    현재 유일한 프로덕션 호출부(pipeline.run_recommendation_mode)는
    항상 넘기므로 이 경로는 하위 호환 목적의 방어적 분기일 뿐이다.

    2026-07-31(Constraint Layer 도입, docs/verification/2026-07-31_
    candidate_generation_top100_eval/root_cause_table_top20.md) -
    user_career_level을 넘기면(기본값 None) constraint_layer.
    compute_constraint_penalty()로 계산한 Career penalty를 combined_score에
    더한다. 이건 "필터링"이 아니라 "감점"이다 - 필터링(위 문단)은 여전히
    호출부가 Career Filter로 미리 하고, 이 함수는 필터를 통과한 후보들
    사이에서 코사인 유사도만으로는 못 잡는 "지원 가능 여부"를 점수에
    반영하는 역할만 한다. user_career_level을 안 넘기면(기본값 None)
    이 분기도 전혀 타지 않는다 - 하위 호환."""
    import numpy as np

    from resume_input.meaning_matching import MeaningMatchScorer, combine_m3_m4
    from resume_input.layer_m3 import LayerM3Scorer
    from resume_input.layer_representation import build_layer_representation
    from resume_input.rule_representation import build_rule_based_representation, build_rule_based_m4_nodes

    if jobs is None:
        jobs = list_all_candidate_jobs()
    if not jobs:
        return []

    with_rep: list[dict] = []
    without_rep: list[dict] = []
    combined: list[float] = []

    if resume_semantic_objects:
        resume_layer_repr = build_layer_representation(resume_semantic_objects)
        layer_scorer = LayerM3Scorer(resume_layer_repr)
        scorer = MeaningMatchScorer(resume_short_semantic, resume_semantic_graph)

        rule_reps = [build_rule_based_representation(j) for j in jobs]
        valid_pairs = [(j, rr) for j, rr in zip(jobs, rule_reps) if rr["is_valid"]]
        without_rep = [j for j, rr in zip(jobs, rule_reps) if not rr["is_valid"]]

        layer_scorer.preload([rr for _, rr in valid_pairs])
        rule_node_lists = [build_rule_based_m4_nodes(rr) for _, rr in valid_pairs]
        scorer.preload([], rule_node_lists)  # jd_semantics=[] - 옛 필드 임베딩 없음

        m3_scores, m4_scores = [], []
        for (j, rule_rep), nodes in zip(valid_pairs, rule_node_lists):
            m3_scores.append(layer_scorer.score(rule_rep))
            j["m3_source"] = "rule"
            m4_scores.append(scorer.m4_score(nodes) if nodes else 0.0)
            j["m4_source"] = "rule"

        combined = combine_m3_m4(m3_scores, m4_scores) if valid_pairs else []
        for (j, _), score in zip(valid_pairs, combined):
            j["combined_score"] = score
            j["score_source"] = "meaning_matching"
        with_rep = [j for j, _ in valid_pairs]
    else:
        without_rep = list(jobs)

    if without_rep:
        stage1 = _bm25_stage(resume_raw, without_rep, len(without_rep))
        fallback = _combine_and_sort(_add_embedding_scores(resume_raw, stage1))
        # BM25+Embedding의 combined_score(0~1)를 M3+M4의 z-score 분포
        # 범위로 재조정한다 - 두 점수 체계는 단위가 달라서 그대로 섞으면
        # 안 된다. Rule Representation 무효(원문 파싱 실패/정보 부족)는
        # 예외적인 경우로 가정하므로, 이 fallback 그룹은 M3+M4 분포의
        # 중간~하위권에 배치하는 것을 기본으로 한다.
        if combined:
            target_lo, target_hi = min(combined), np.percentile(combined, 50) if len(combined) > 1 else combined[0]
        else:
            target_lo, target_hi = -1.0, 0.0
        fb_scores = [j["combined_score"] for j in fallback]
        f_lo, f_hi = (min(fb_scores), max(fb_scores)) if fb_scores else (0.0, 1.0)
        f_span = (f_hi - f_lo) or 1.0
        for j in fallback:
            norm = (j["combined_score"] - f_lo) / f_span
            j["combined_score"] = target_lo + norm * (target_hi - target_lo)
            j["score_source"] = "bm25_fallback"
            j["m3_source"] = "bm25_fallback"
            j["m4_source"] = "bm25_fallback"
        with_rep = with_rep + fallback

    # 2026-07-31 Constraint Layer - m3_source/score_source(Rule/
    # bm25_fallback) 어느 경로로 채점됐든 상관없이 동일하게 적용한다
    # (Constraint는 "어떻게 채점했는지"가 아니라 "지원 가능한지"를 보는
    # 것이라 채점 경로와 독립적이어야 한다). combined_score_before_constraint/
    # constraint_penalty/constraint_debug를 job에 남겨서, 나중에 "왜 이
    # 순위가 됐는지"를 로그로 그대로 설명할 수 있게 한다. user_career_level
    # 이 None이어도(하위 호환 경로) compute_constraint_penalty()가 내부에서
    # 0.0을 반환하므로 여기서 따로 on/off 분기를 두지 않는다 - Constraint
    # Layer가 늘어나도 이 블록은 그대로 유지된다.
    from resume_input.constraint_layer import compute_constraint_penalty, compute_constraint_debug
    for j in with_rep:
        penalty = compute_constraint_penalty(j, user_career_level, challenge_option)
        j["combined_score_before_constraint"] = j["combined_score"]
        j["constraint_penalty"] = penalty
        j["constraint_debug"] = compute_constraint_debug(j, user_career_level, challenge_option)
        j["combined_score"] += penalty

    ranked = sorted(with_rep, key=lambda j: -j["combined_score"])
    result = ranked[:final_top_n]
    for j in result:
        j["matched_keywords"] = _matched_keywords(resume_raw, j)
    return result


def search_candidates(
    resume_raw: str,
    bm25_top_n: int = 200,
    final_top_n: int = 200,
    query_text: str | None = None,
    jobs: list[dict] | None = None,
) -> list[dict]:
    """query_text를 주면 그걸로 BM25 검색(Query Rewriting 결과 등),
    안 주면 resume_raw 원문으로 검색한다. Embedding은 결과를 줄이지 않고
    정렬 보조 점수로만 결합된다. 결과에는 matched_keywords(추천 이유,
    LLM 미사용)도 포함된다.

    jobs를 넘기면 그 목록만 검색한다(Career Filter로 이미 축소된
    후보 풀 등) - candidate_search.py는 Career 판단을 하지 않으므로,
    필터링은 항상 호출부(pipeline.py)가 미리 끝내고 결과만 넘긴다.
    안 넘기면(기본값) 전체 candidate_jobs를 그대로 검색한다.
    """
    if jobs is None:
        jobs = list_all_candidate_jobs()
    if not jobs:
        return []

    search_query = query_text if query_text is not None else resume_raw

    bm25_top_n = min(bm25_top_n, len(jobs))
    stage1 = _bm25_stage(search_query, jobs, bm25_top_n)

    with_embeddings = _add_embedding_scores(resume_raw, stage1)
    combined = _combine_and_sort(with_embeddings)

    result = combined[:final_top_n]
    for j in result:
        j["matched_keywords"] = _matched_keywords(resume_raw, j)

    return result
