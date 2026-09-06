"""
resume_input/semantic_matching.py

Semantic Object 기반 Meaning Matching Engine(체크포인트 3, docs/
semantic_object_schema.md) - 기존 M3(Rule+Embedding)+M4(Graph) 방식
(`meaning_matching.py`)을 대체할 신규 엔진. 아직 파이프라인에 연결되지
않았다(체크포인트 5에서 `pipeline.py`가 이 모듈로 전환한다) - 지금은
독립적으로 존재하며, `meaning_matching.py`(옛 엔진)는 그대로 살아있고
계속 실제 추천에 쓰인다. 옛 파일은 체크포인트 7에서 제거한다.

옛 엔진과의 근본적 차이: M3+M4는 스칼라 점수(m3_score/m4_score) 두
개만 반환하고 "왜 매칭됐는지"를 버렸다. 이 엔진은 Resume/JD의 Semantic
Object를 레이어별로 비교해서 **Match Object**(어떤 신호로, 어느 필드가
겹쳐서, 왜 매칭/미매칭됐는지까지 구조화된 근거)를 만든다 - 점수는 그
결과의 부산물이다.

매칭 신호는 5가지(DIRECT/FRAME/EVIDENCE/ANCHOR/SEMANTIC)를 전부
계산한 뒤 최댓값을 match_type으로 채택한다(첫 신호에서 멈추지 않음).
체크포인트 0 실측(docs/semantic_object_schema.md)에서 "Python" vs
"Python 및 SQL 데이터 분석 역량"처럼 normalized_text만으로는 못 잡는
경우를 확인했다 - 그래서 FRAME 신호는 전체 frame 텍스트 비교뿐 아니라
frame.method 리스트 항목 각각을 상대 Object의 embedding_text와도
교차 비교한다.

Match Relation(Resume<->JD 교차 설명 체인)은 여기서 저장하지 않는다 -
`derive_match_relation()`으로 필요할 때(체크포인트 6, Presentation
Layer)만 그 자리에서 파생한다. Document-internal Relation + Match
Object만 있으면 항상 다시 만들 수 있는 데이터이기 때문(저장해두면
이력서가 바뀔 때마다 다시 계산해서 덮어써야 하는 불필요한 상태가
생긴다).
"""
from __future__ import annotations

import io
import sqlite3

import numpy as np

from resume_input.candidate_search import _load_embedding_model
from resume_input.job_store import DB_PATH

# 2026-07-17(사용자 확정, 성능 병목 2순위) - SemanticObjectMatcher는
# 요청 안에서는 재사용되지만(pipeline.py, 2026-07-17 앞선 수정), 검색을
# 다시 하면(다음 프로세스 실행/다른 요청) 같은 이력서라도 이력서 쪽
# 임베딩을 처음부터 다시 계산했다. resume_hash 기준으로 이력서 임베딩을
# DB에 영구 저장해서 다음 검색부터 재사용한다 - 값 자체는 전혀 안
# 바뀐다(임베딩은 결정적 함수라 같은 텍스트는 항상 같은 벡터).


def _init_resume_embedding_cache_table(db_path=DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_embedding_cache (
            resume_hash TEXT PRIMARY KEY,
            cache_blob BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _serialize_emb_cache(cache: dict[str, np.ndarray]) -> bytes:
    keys = list(cache.keys())
    embeddings = np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 384))
    buf = io.BytesIO()
    np.savez(buf, keys=np.array(keys, dtype=object), embeddings=embeddings)
    return buf.getvalue()


def _deserialize_emb_cache(blob: bytes) -> dict[str, np.ndarray]:
    buf = io.BytesIO(blob)
    data = np.load(buf, allow_pickle=True)
    keys = data["keys"]
    embeddings = data["embeddings"]
    return {str(k): embeddings[i] for i, k in enumerate(keys)}


def _load_resume_embedding_cache(resume_hash: str, db_path=DB_PATH) -> dict[str, np.ndarray]:
    _init_resume_embedding_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT cache_blob FROM resume_embedding_cache WHERE resume_hash = ?", (resume_hash,)
    ).fetchone()
    conn.close()
    if not row:
        return {}
    return _deserialize_emb_cache(row[0])


def _save_resume_embedding_cache(resume_hash: str, cache: dict[str, np.ndarray], db_path=DB_PATH) -> None:
    _init_resume_embedding_cache_table(db_path)
    blob = _serialize_emb_cache(cache)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO resume_embedding_cache (resume_hash, cache_blob) VALUES (?, ?)",
        (resume_hash, blob),
    )
    conn.commit()
    conn.close()


# 2026-07-19(실측 근거: 프로파일링 결과 검색 1회에 warm job 283건의 JD
# 텍스트를 매번 처음부터 encode()해서 약 180초 소요 - 이력서 쪽은 이미
# resume_hash 기준 영구 캐시가 있지만(위) JD 쪽은 그런 캐시가 없어서,
# 같은 JD를 재검색/다른 이력서로 검색할 때마다 매번 재계산했다. JD
# Semantic Object의 텍스트(embedding_text/frame/method/evidence)는
# representation_version이 그대로인 한 절대 안 바뀌므로(결정적 함수),
# job_id 기준으로 영구 캐시한다 - 첫 스캔만 비용을 내고 그 다음부터는
# (버전이 안 바뀌는 한) 어떤 이력서로 검색하든 재사용한다.
# representation_version을 같이 저장해서, JD Understanding이 재생성되면
# (버전 문자열이 바뀌면) 자동으로 캐시 미스 처리되고 새로 계산해
# 덮어쓴다 - 별도 무효화 로직 불필요.


def _init_jd_embedding_cache_table(db_path=DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jd_embedding_cache (
            job_id TEXT PRIMARY KEY,
            representation_version TEXT,
            cache_blob BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _load_jd_embedding_cache(job_id: str, representation_version: str | None, db_path=DB_PATH) -> dict[str, np.ndarray]:
    _init_jd_embedding_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT cache_blob FROM jd_embedding_cache WHERE job_id = ? AND representation_version = ?",
        (job_id, representation_version),
    ).fetchone()
    conn.close()
    if not row:
        return {}
    return _deserialize_emb_cache(row[0])


def _save_jd_embedding_cache(job_id: str, representation_version: str | None, cache: dict[str, np.ndarray], db_path=DB_PATH) -> None:
    _init_jd_embedding_cache_table(db_path)
    blob = _serialize_emb_cache(cache)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO jd_embedding_cache (job_id, representation_version, cache_blob) VALUES (?, ?, ?)",
        (job_id, representation_version, blob),
    )
    conn.commit()
    conn.close()


# 이 엔진의 채점 로직(신호 계산식/임계치)이 바뀌면 올린다(2026-07-16
# 추가) - match_validator.py의 Validator 캐시가 이 값을 캐시 키에
# 포함해서, 엔진 로직이 바뀌면(예: ANCHOR 캡 조정 같은 채점식 변경)
# 같은 이력서/JD 조합이어도 Validator를 자동으로 다시 부르게 한다.
MATCHING_VERSION = "v1-2026-07-16"

# status 판정 임계치(체크포인트 0/3에서 실측 확정 예정, 잠정값 -
# docs/semantic_object_schema.md 참고).
MATCH_THRESHOLD = 0.75
CANDIDATE_THRESHOLD = 0.60

# Ranking Score 계산용 JD importance 가중치(잠정값, 실측으로 조정).
_JD_IMPORTANCE_WEIGHT = {"critical": 4.0, "core": 2.0, "normal": 1.5, "low": 1.0}

# Resume/JD 공통 레이어만 매칭 대상 - "culture"는 JD 전용이라(Resume에
# 대응하는 레이어가 없음) 매칭하지 않는다(③번 화면에서 JD 단독으로만
# 보여준다).
_MATCHABLE_LAYERS = ("problem", "thinking", "task", "skill", "qualification")

# Qualification Evidence Type Gate(_score_pair)와 validate_matches_rule_based
# (품질 검증)가 공유하는 "교육 이수/자격 취득" 동사 목록 - 실무 경력
# (experience)과 구분하는 기준(2026-07-17 감사 근거는 _score_pair 주석 참고).
_CREDENTIAL_ACTIONS = ("수료", "취득", "합격", "이수", "획득")


def _embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    model = _load_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _frame_text(frame: dict | None) -> str:
    frame = frame or {}
    parts = [frame.get("action", ""), frame.get("object", ""), frame.get("goal", ""),
              frame.get("domain", ""), frame.get("outcome", "")]
    parts += [m for m in (frame.get("method") or []) if m]
    return " ".join(p for p in parts if p)


def _obj_id(obj: dict) -> str:
    """Object의 참조용 id. understanding.py가 새로 생성한 Object는
    `id`가 있지만(레이어별 순번), 이번 backfill 이전에 저장된 Object는
    없을 수 있어 normalized_text로 대체한다(둘 다 문서 내에서는
    충분히 안정적으로 유일하다)."""
    return obj.get("id") or obj.get("normalized_text") or ""


class SemanticObjectMatcher:
    """이력서 1건에 대해 여러 JD를 반복 비교할 때 이력서 쪽 임베딩을
    한 번만 계산해서 재사용한다(옛 MeaningMatchScorer와 같은 목적).
    JD 쪽 텍스트는 match_jd_object() 호출 시점에 필요한 것만 추가로
    임베딩한다(캐시 재사용)."""

    def __init__(self, resume_objects: list[dict], resume_hash: str | None = None):
        """`resume_hash`를 넘기면(2026-07-17, 사용자 확정) 이력서 쪽
        임베딩을 DB에서 먼저 불러오고(있으면), 이번에 새로 계산한 것만
        추가로 저장한다 - 같은 이력서로 다시 검색하면 이 이력서의
        임베딩을 처음부터 다시 계산하지 않는다. JD 쪽 텍스트(아래
        match_jd_object에서 그때그때 추가되는 것)는 여기 저장하지 않는다
        - resume_objects에서 나온 텍스트만 저장 대상이다(안 넘기면
        기존과 동일하게 매번 새로 계산, 하위호환)."""
        self.resume_objects = resume_objects
        self._emb_cache: dict[str, np.ndarray] = {}
        self.by_layer: dict[str, list[dict]] = {}
        for obj in resume_objects:
            self.by_layer.setdefault(obj.get("layer", ""), []).append(obj)

        loaded_keys: set[str] = set()
        if resume_hash:
            cached = _load_resume_embedding_cache(resume_hash)
            if cached:
                self._emb_cache.update(cached)
                loaded_keys = set(cached.keys())

        self._preload(self._all_texts(resume_objects))

        if resume_hash and set(self._emb_cache.keys()) - loaded_keys:
            _save_resume_embedding_cache(resume_hash, self._emb_cache)

    @staticmethod
    def _all_texts(objects: list[dict]) -> list[str]:
        texts: list[str] = []
        for obj in objects:
            texts.append(obj.get("embedding_text") or obj.get("normalized_text") or "")
            frame = obj.get("frame") or {}
            texts.append(_frame_text(frame))
            # 2026-07-19(실측 근거: 프로파일링 결과 이 5개 개별 필드가
            # 여기 빠져 있어서 _frame_field_diff()의 필드별 _cos() 비교가
            # 매번 텍스트 1개짜리 encode()를 새로 호출했다 - 이력서 쪽은
            # 최초 1건 처리 시 44.8초, JD 쪽은 매 job마다 0.3~0.7초씩
            # 누적되던 원인. 여기 포함시켜서 다른 텍스트들과 함께 한
            # 번에 배치로 preload되게 한다 - 비교 로직/점수식은 그대로,
            # 캐시 커버리지만 넓어진다.
            for field in ("action", "object", "goal", "domain", "outcome"):
                texts.append(frame.get(field, ""))
            texts.extend(frame.get("method") or [])
            for ev in obj.get("expected_evidence") or []:
                if ev.get("value"):
                    texts.append(ev["value"])
        return [t for t in texts if t]

    def _preload(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self._emb_cache]
        if not missing:
            return
        embs = _embed(missing)
        for t, e in zip(missing, embs):
            self._emb_cache[t] = e

    def preload_jd_for_job(self, job_id: str | None, representation_version: str | None, jd_objects: list[dict]) -> None:
        """JD 1건의 Object 전체 텍스트를 DB의 영구 캐시(job_id +
        representation_version 기준)에서 먼저 채워보고, 캐시가 없으면
        (첫 스캔이거나 버전이 바뀌었으면) 한 번에 배치로 embed한 뒤 이
        job의 몫만 다시 DB에 저장한다. job_id가 없으면(테스트 등) DB
        캐시 없이 기존과 동일하게 그때그때 embed한다.

        캐시가 있어도 이번에 필요한 텍스트를 전부 커버하는지는 보장 못
        한다(예: _all_texts()가 나중에 필드를 더 추가하도록 바뀌면 그
        전에 저장된 캐시는 새 필드가 빠진 채로 남는다) - `_preload`가
        빠진 텍스트만 골라 채우므로 항상 호출하고, 새로 채워진 게
        있으면(= 기존 캐시가 불완전했으면) 완전해진 값으로 다시
        저장한다(자가 치유, 2026-07-19)."""
        texts = self._all_texts(jd_objects)
        if not texts:
            return
        if not job_id:
            self._preload(texts)
            return
        cached = _load_jd_embedding_cache(job_id, representation_version)
        if cached:
            self._emb_cache.update(cached)
        unique_texts = list(dict.fromkeys(texts))
        was_complete = cached and all(t in cached for t in unique_texts)
        if was_complete:
            return
        self._preload(texts)
        job_cache = {t: self._emb_cache[t] for t in unique_texts if t in self._emb_cache}
        _save_jd_embedding_cache(job_id, representation_version, job_cache)

    def _emb(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(384)
        if text not in self._emb_cache:
            self._preload([text])
        return self._emb_cache[text]

    def _cos(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return float(np.dot(self._emb(a), self._emb(b)))

    def _score_pair(self, jd_obj: dict, r_obj: dict) -> dict[str, float]:
        jd_text = jd_obj.get("embedding_text") or jd_obj.get("normalized_text") or ""
        r_text = r_obj.get("embedding_text") or r_obj.get("normalized_text") or ""
        jd_frame_text = _frame_text(jd_obj.get("frame"))
        r_frame_text = _frame_text(r_obj.get("frame"))
        jd_methods = [m for m in (jd_obj.get("frame") or {}).get("method") or [] if m]
        r_methods = [m for m in (r_obj.get("frame") or {}).get("method") or [] if m]
        jd_expected = [ev["value"] for ev in jd_obj.get("expected_evidence") or [] if ev.get("value")]

        direct = 1.0 if jd_obj.get("concept_id") and jd_obj.get("concept_id") == r_obj.get("concept_id") else 0.0

        # ANCHOR는 5개 신호 중 가장 약한 신호(우선순위 DIRECT>FRAME>
        # EVIDENCE>ANCHOR>SEMANTIC)라, 겹치기만 하면 1.0(=MATCH_THRESHOLD
        # 0.75 이상, 자동 "match" 확정)을 주면 안 된다 - 실측(2026-07-16,
        # 특정 이커머스 카탈로그 품질 케이스)에서 ANCHOR 단독으로 1.0이 나와
        # 검증 없이 "match"로 확정돼버리는 과대평가를 확인했다. CANDIDATE_
        # THRESHOLD(0.60)는 넘지만 MATCH_THRESHOLD(0.75)는 못 넘는
        # 값으로 고정해서, Anchor만으로는 항상 "candidate"(Top30 검증
        # 대상)에 머물게 한다.
        # 2026-07-17 "직접 관계만 허용" 실험(docs/verification/2026-07-17_
        # anchor_hub_audit/) - 제3의 공유 이웃(hub)만으로 ANCHOR를 주는
        # 문제를 막으려고 시도했으나, 64건 전수 재검증 결과 Problem/Task
        # 레이어의 정상 TP까지 함께 제거됨을 확인(Problem TP 6->3,
        # Task TP 3->1) - 유사도 전이 차단 자체는 유효하지만 레이어
        # 전체에 획일 적용하면 Recall이 죽는다는 게 실측으로 반증됨.
        # 원래 로직(교집합 기반)으로 되돌린다 - 수정은 레이어별 정책
        # 감사 이후에 다시 검토한다.
        jd_anchor = set(jd_obj.get("matching_anchor") or [])
        r_anchor = set(r_obj.get("matching_anchor") or [])
        anchor = 0.65 if (jd_anchor & r_anchor) else 0.0

        semantic = self._cos(jd_text, r_text)

        # FRAME: 전체 frame 텍스트 유사도 + method 리스트 교차 비교(체크
        # 포인트 0 실측 근거 - "Python" vs "Python 및 SQL 데이터 분석
        # 역량"처럼 normalized_text 비교만으로는 못 잡는 케이스가 있음).
        frame_candidates = [self._cos(jd_frame_text, r_frame_text)]
        for m in jd_methods:
            frame_candidates.append(self._cos(m, r_text))
        for m in r_methods:
            frame_candidates.append(self._cos(m, jd_text))
        frame = max(frame_candidates) if frame_candidates else 0.0

        evidence_candidates = [self._cos(v, r_text) for v in jd_expected]
        evidence = max(evidence_candidates) if evidence_candidates else 0.0

        # Domain/Object Gate(2026-07-17, 사용자 확정 - 감사 근거:
        # docs/verification/2026-07-17_matching_engine_understanding_audit/).
        # object와 domain이 둘 다 비어있으면(핵심 대상·업무 영역 자체가
        # 다르면) FRAME/EVIDENCE/SEMANTIC은 candidate조차 확정할 근거가
        # 못 된다 - 실측(특정 제조업 구매 공고 task_001 "원가 절감 전략" vs
        # Resume "프로모션 퍼널 분석"): object/domain 둘 다 unmatched인데도
        # FRAME이 action/goal 통짜 cosine만으로 0.657까지 올라가 candidate가
        # 됐다. DIRECT(concept_id 완전 일치)와 ANCHOR(taxonomy 기반, 이미
        # 0.65로 캡되어 match까지는 못 감)는 이 게이트 대상이 아니다 - 이미
        # 다른 방식으로 과대평가를 막고 있는 신호라서 건드리지 않는다.
        matched_fields, missing_fields = self._frame_field_diff(jd_obj, r_obj)
        # 게이트 확대(2026-07-17, 사용자 확정 - 감사 근거: docs/verification/
        # 2026-07-17_gate_widen_audit/, warm job 428건 전수 시뮬레이션).
        # AND(둘 다 missing)만 캡하던 걸 problem/thinking/task/qualification/
        # culture는 OR(하나만 missing)로 넓힌다 - 실측: task 레이어에서
        # action/object/goal이 전부 일반적인 문구("고객","구축","분석")라
        # 코사인이 우연히 맞고, domain만이 유일하게 실제 업무 영역 차이를
        # 정확히 구분하는 사례가 15건 이상 확인됨(이력서 "고객 프로모션
        # 퍼널 분석" task가 엔터테인먼트/게임/핀테크 등 무관한 업종 공고에
        # 반복적으로 강한 매칭). qualification도 "공인노무사 자격증" vs
        # "ADsP 합격" 같은 명백한 오탐이 AND 게이트로는 안 걸리는 걸 확인.
        # skill 레이어만 예외로 AND를 유지한다 - 실측(특정 DB전문기업 SQL/특정 이커머스 Python):
        # skill의 domain 필드가 종종 "RDBMS"/"Technical Stack"처럼 실제
        # 의미 없는 섹션 헤더 텍스트로 채워져서(Understanding 쪽 문제,
        # 별도 프롬프트 수정 진행 중) domain만 missing으로 걸리는 경우가
        # 흔한데, 그 경우도 대부분 실제로는 올바른 스킬 매칭이었다 - OR로
        # 넓히면 skill 레이어의 진짜 매칭까지 같이 캡돼버린다.
        if jd_obj.get("layer") == "skill":
            gate_fires = "object" in missing_fields and "domain" in missing_fields
        else:
            gate_fires = "object" in missing_fields or "domain" in missing_fields
        if gate_fires:
            cap = CANDIDATE_THRESHOLD - 0.01
            frame = min(frame, cap)
            evidence = min(evidence, cap)
            semantic = min(semantic, cap)

        # Qualification Evidence Type Gate(2026-07-17, 사용자 확정 - 감사
        # 근거: docs/verification/2026-07-17_qualification_evidence_type_
        # audit/). qualification 레이어 실측: JD expected_evidence 224건
        # 중 173건(77%)이 type="experience"(실무 경력)를 요구하는데, 이
        # 이력서의 qualification Object 2개("데이터 분석가 과정 수료",
        # "ADsP 합격")는 전부 교육 이수/자격 취득이지 실무 경력이 아니다 -
        # 이건 임베딩 유사도로 판단할 문제가 아니라 애초에 다른 종류의
        # 증거다(교육·자격증이 아무리 관련 주제여도 "경력"을 증명하지
        # 못한다). DIRECT는 대상이 아니다(concept_id 완전 일치는 이미
        # 신뢰할 수 있는 신호).
        #
        # 버그 수정(2026-07-17, 특정 스타트업 Data Scientist 공고 실측): 이 게이트가
        # expected_evidence.type == "experience"만 검사했는데, 스키마상
        # type은 project/metric/experience/tool 4종류다. "project"도
        # experience와 마찬가지로 실무 프로젝트 수행을 요구하는 것이지
        # 교육 이수로 충족되지 않는다(실측: 해당 공고 "불균형/시계열 데이터
        # 문제 해결 경험" 객체가 type="project"라서, 이력서의 "데이터
        # 분석가 과정 수료"(수료=credential)와 매칭됐는데도 게이트가
        # 안 걸려서 anchor 0.65가 그대로 critical 가중치에 들어갔다).
        # 정책 변경이 아니라 원래 의도(교육·자격 이수가 실무 증거를
        # 대체할 수 없다)를 스키마 전체에 맞게 반영하는 구현 누락 수정.
        if jd_obj.get("layer") == "qualification":
            jd_requires_practical_evidence = any(
                ev.get("type") in ("experience", "project")
                for ev in (jd_obj.get("expected_evidence") or [])
            )
            r_action = (r_obj.get("frame") or {}).get("action", "")
            r_is_credential = any(v in r_action for v in _CREDENTIAL_ACTIONS)
            if jd_requires_practical_evidence and r_is_credential:
                cap = CANDIDATE_THRESHOLD - 0.01
                frame = min(frame, cap)
                evidence = min(evidence, cap)
                semantic = min(semantic, cap)
                anchor = min(anchor, cap)

        return {"direct": direct, "frame": frame, "evidence": evidence, "anchor": anchor, "semantic": semantic}

    def _frame_field_diff(self, jd_obj: dict, r_obj: dict, field_threshold: float = 0.5) -> tuple[list[str], list[str]]:
        """frame의 개별 필드(action/object/goal/domain/outcome)가 실질적으로
        겹치는지 확인해서 matched_fields/missing_fields를 만든다(설명
        생성용 - Match Object의 strength_reason/gap_reason 근거)."""
        jd_frame = jd_obj.get("frame") or {}
        r_frame = r_obj.get("frame") or {}
        matched, missing = [], []
        for field in ("action", "object", "goal", "domain", "outcome"):
            jd_val, r_val = jd_frame.get(field, ""), r_frame.get(field, "")
            if not jd_val:
                continue
            if r_val and self._cos(jd_val, r_val) >= field_threshold:
                matched.append(field)
            else:
                missing.append(field)
        jd_methods = set(jd_frame.get("method") or [])
        r_methods = set(r_frame.get("method") or [])
        if jd_methods:
            if jd_methods & r_methods:
                matched.append("method")
            else:
                # method는 완전 문자열 일치가 아니어도(예: JD "Python/SQL"
                # 묶음 vs Resume 개별 스킬) 교차 임베딩으로 이미 FRAME
                # 점수에 반영되므로, 텍스트 일치 실패만으로 missing 단정하지
                # 않는다 - 개별 cosine 재확인.
                any_hit = any(self._cos(m, jd_m) >= field_threshold for m in r_methods for jd_m in jd_methods)
                (matched if any_hit else missing).append("method")
        return matched, missing

    def _reasons(self, match_type: str, matched_fields: list[str], jd_obj: dict, r_obj: dict) -> tuple[list[str], list[str]]:
        strength = []
        if match_type == "DIRECT":
            strength.append(f"동일 개념({jd_obj.get('normalized_text','')})")
        elif matched_fields:
            strength.append("일치 항목: " + ", ".join(matched_fields) +
                             f"({jd_obj.get('normalized_text','')} <-> {r_obj.get('normalized_text','')})")
        gap = []
        return strength, gap

    def match_jd_object(self, jd_obj: dict) -> dict:
        """JD Semantic Object 1개를 이 이력서의 같은 레이어 Object 전체와
        비교한다. 반환: Match Object(status="match"|"candidate",
        matched_resume_objects=[...]) 또는 Missing Object
        (status 없이 reason만)."""
        layer = jd_obj.get("layer", "")
        candidates = self.by_layer.get(layer, [])
        self._preload(self._all_texts([jd_obj]))

        matched = []
        for r_obj in candidates:
            scores = self._score_pair(jd_obj, r_obj)
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            if confidence < CANDIDATE_THRESHOLD:
                continue
            matched_fields, missing_fields = self._frame_field_diff(jd_obj, r_obj)
            strength_reason, gap_reason = self._reasons(best_type.upper(), matched_fields, jd_obj, r_obj)
            if missing_fields:
                gap_reason.append("부족 항목: " + ", ".join(missing_fields))
            matched.append({
                "resume_object": _obj_id(r_obj),
                "match_type": best_type.upper(),
                "signal_scores": scores,
                "matched_fields": matched_fields,
                "missing_fields": missing_fields,
                "strength_reason": strength_reason,
                "gap_reason": gap_reason,
                "engine_confidence": confidence,
            })

        matched.sort(key=lambda m: -m["engine_confidence"])
        jd_id = _obj_id(jd_obj)
        if not matched:
            return {"jd_object": jd_id, "reason": "no evidence"}

        status = "match" if matched[0]["engine_confidence"] >= MATCH_THRESHOLD else "candidate"
        return {"jd_object": jd_id, "status": status, "matched_resume_objects": matched}


# 2026-07-20(사용자 확정 - 실측 근거: Top50 레이어 감사, docs/verification/
# 2026-07-20_ranking_layer_weight_fix/). ranking_score가 레이어 구분 없이
# JD Object 개수 × importance만으로 가중평균되던 구조라, Skill Object가
# (a) critical로 여러 개 쪼개져 있고 (b) DIRECT/EVIDENCE로 confidence
# 1.0이 쉽게 나오는 특성 때문에, critical Task가 전부 미스여도 skill
# 2~3개만 맞으면 상위권까지 올라가는 편향이 실측으로 확인됐다(Top50 중
# 44%가 이 패턴, 상위 10위 중 8개). 손계산 검증: 특정 스타트업 데이터엔지니어
# 사례(task 5개 전부 미스, skill 3개 매칭)가 기존 공식으로 정확히
# ranking_score=0.319 나옴 - 공식 자체는 버그 없이 의도대로 작동했지만
# "의도"에 레이어 균형이 없었던 게 문제.
#
# "이 사람이 실제로 이 일을 해봤는가"(Task/Problem)가 "이 도구를 아는가"
# (Skill)보다 직무 적합도의 핵심 신호라는 게 이 프로젝트의 반복된 결론
# (Understanding Engine 감사, FP 감사 등)이라, 레이어별로 먼저 정규화한
# 뒤(레이어 안의 Object 개수가 그 레이어의 비중을 부풀리지 않도록) 고정
# 비율로 합친다. Task/Problem을 가장 무겁게, Skill/Qualification/Culture는
# "필요하지만 충분하지 않은" 보조 신호로 낮게 둔다. 값 자체는 잠정치 -
# 이번 실측(Top50 재검증, docs/verification/2026-07-20_ranking_layer_
# weight_fix/)에서 44%->? 로 실제 개선되는지 확인 후 조정 여지 있음.
_LAYER_RANKING_WEIGHT = {
    "problem": 0.25,
    "task": 0.30,
    "thinking": 0.15,
    "skill": 0.15,
    "qualification": 0.10,
    "culture": 0.05,
}


def compute_coverage_and_ranking(jd_objects: list[dict], match_results: list[dict]) -> dict:
    """Coverage(레이어별 importance tier - 표시/진단용, 안 바뀜)와 Ranking
    Score(정렬용)를 계산한다.

    Ranking Score(2026-07-20 재설계) - 레이어별로 먼저 importance 가중
    평균을 구하고(레이어 안에서는 기존과 동일한 계산), 그 레이어 점수들을
    _LAYER_RANKING_WEIGHT 고정 비율로 합친다 - "이 레이어에 Object가
    몇 개 있는지"가 그 레이어의 전체 영향력을 더 이상 좌우하지 않는다
    (Skill Object를 5개로 쪼개도 Skill 레이어 전체 비중은 여전히 0.15).
    이 JD에 없는 레이어는 분모(가중치 합)에서도 제외해서, 레이어 구성이
    JD마다 달라도(예: culture Object가 없는 JD) 점수 비교 가능성을
    유지한다."""
    by_jd_id = {r["jd_object"]: r for r in match_results}

    counts: dict[str, dict[str, list[int]]] = {}
    layer_weighted_sum: dict[str, float] = {}
    layer_weight_total: dict[str, float] = {}

    for jd_obj in jd_objects:
        jid = _obj_id(jd_obj)
        layer = jd_obj.get("layer", "")
        importance = jd_obj.get("importance", "normal")
        weight = _JD_IMPORTANCE_WEIGHT.get(importance, 1.0)

        result = by_jd_id.get(jid)
        best_conf = 0.0
        if result and result.get("matched_resume_objects"):
            best_conf = max(m["engine_confidence"] for m in result["matched_resume_objects"])

        layer_weighted_sum[layer] = layer_weighted_sum.get(layer, 0.0) + weight * best_conf
        layer_weight_total[layer] = layer_weight_total.get(layer, 0.0) + weight

        tier_key = f"{importance}_coverage"
        bucket = counts.setdefault(layer, {}).setdefault(tier_key, [0, 0])
        bucket[1] += 1
        if best_conf >= MATCH_THRESHOLD:
            bucket[0] += 1

    coverage = {
        layer: {tier: (c[0] / c[1] if c[1] else 0.0) for tier, c in tiers.items()}
        for layer, tiers in counts.items()
    }

    layer_scores = {
        layer: (layer_weighted_sum[layer] / layer_weight_total[layer])
        for layer in layer_weight_total if layer_weight_total[layer] > 0
    }
    layer_weight_sum = sum(_LAYER_RANKING_WEIGHT.get(layer, 0.0) for layer in layer_scores)
    if layer_weight_sum > 0:
        ranking_score = sum(
            layer_scores[layer] * _LAYER_RANKING_WEIGHT.get(layer, 0.0) for layer in layer_scores
        ) / layer_weight_sum
    else:
        ranking_score = 0.0
    return {"coverage": coverage, "ranking_score": ranking_score}


def match_resume_to_jd(resume_objects: list[dict], jd_objects: list[dict], matcher: "SemanticObjectMatcher | None" = None) -> dict:
    """이력서 1건 vs JD 1건의 전체 매칭 결과. Match Object 리스트(매칭
    레이어만 - culture 등 대응 레이어 없는 JD Object는 자동으로 Missing
    처리됨) + Coverage + Ranking Score를 반환한다. 저장하지 않는다 -
    매 요청마다 계산(옛 M3+M4와 같은 원칙).

    2026-07-17(사용자 확정) - `matcher`를 넘기면 그 인스턴스를 그대로
    재사용한다(이력서 쪽 임베딩을 다시 계산하지 않음) - 호출부가 같은
    이력서로 JD를 여러 건 반복 비교할 때(pipeline.py의 Top50 후보
    스캔) 매번 `SemanticObjectMatcher(resume_objects)`를 새로 만들면
    이력서 임베딩을 매 job마다 처음부터 다시 계산하게 된다(실측: 후보군
    150~400건까지 늘어난 뒤로 이 재계산 비용이 커짐). 안 넘기면(기본값)
    기존과 동일하게 이 호출 안에서 1회 생성한다 - 하위 호환, 결과값은
    matcher를 넘기든 안 넘기든 동일하다."""
    if matcher is None:
        matcher = SemanticObjectMatcher(resume_objects)
    match_results = [matcher.match_jd_object(jd_obj) for jd_obj in jd_objects]
    agg = compute_coverage_and_ranking(jd_objects, match_results)
    return {"matches": match_results, **agg}


def _rule_quality(jd_obj: dict, best: dict, resume_by_id: dict) -> tuple[str, str]:
    """Match 1건(가장 확신도 높은 matched_resume_objects[0])을 이미 계산된
    필드만으로 valid/weak/review 중 하나로 분류한다. 새 LLM/임베딩 호출
    없음 - match_type/matched_fields/missing_fields/engine_confidence/
    expected_evidence/frame.action처럼 Engine이 이미 만든 값만 본다."""
    match_type = best.get("match_type", "")
    matched_fields = best.get("matched_fields") or []
    missing_fields = best.get("missing_fields") or []
    confidence = best.get("engine_confidence", 0.0)

    # DIRECT(concept_id 완전 일치)인데 핵심 대상(object)/영역(domain)
    # 필드가 안 겹치면 - 같은 개념이라는 판정과 실제 frame 내용이 어긋난다.
    if match_type == "DIRECT" and ("object" in missing_fields or "domain" in missing_fields):
        return "review", "동일 개념(DIRECT)으로 판정됐지만 핵심 대상(object)/영역(domain)이 이력서와 다릅니다."

    # Qualification Evidence Type Gate가 이미 confidence를 캡하지만, DIRECT는
    # 그 게이트 대상이 아니므로(_score_pair 주석 참고) 여기서 별도로 잡는다.
    # 버그 수정(2026-07-17, _score_pair와 동일 - type="project"도
    # 실무 증거를 요구하므로 검사 대상에 포함해야 한다).
    if jd_obj.get("layer") == "qualification":
        jd_requires_practical_evidence = any(
            ev.get("type") in ("experience", "project")
            for ev in (jd_obj.get("expected_evidence") or [])
        )
        r_obj = resume_by_id.get(best.get("resume_object"), {})
        r_action = (r_obj.get("frame") or {}).get("action", "")
        if jd_requires_practical_evidence and any(v in r_action for v in _CREDENTIAL_ACTIONS):
            return "review", "요구사항은 실무 경력/프로젝트인데 이력서 근거는 교육 이수/자격 취득입니다."

    # Domain/Object Gate는 FRAME/EVIDENCE/SEMANTIC만 캡하고 ANCHOR는 대상이
    # 아니다(_score_pair 주석 참고) - ANCHOR가 그 빈틈으로 candidate가 된
    # 경우를 잡는다.
    if match_type == "ANCHOR" and "object" in missing_fields and "domain" in missing_fields:
        return "review", "간접 연결(ANCHOR)만 있고 핵심 대상(object)/영역(domain)은 겹치지 않습니다."

    if len(missing_fields) >= 3:
        return "weak", "부족 항목이 많습니다(" + ", ".join(missing_fields) + ")."

    # ANCHOR는 캡(0.65) 때문에 MATCH_THRESHOLD를 못 넘지만, SEMANTIC(순수
    # 임베딩 유사도)은 넘을 수 있다 - 겹치는 세부 필드 없이 confidence만
    # 높은 경우를 잡는다.
    if confidence >= MATCH_THRESHOLD and match_type in ("SEMANTIC", "ANCHOR") and len(matched_fields) <= 1:
        return "weak", "신뢰도는 높지만 실제로 겹치는 세부 항목이 적습니다."

    return "valid", ""


def validate_matches_rule_based(resume_objects: list[dict], jd_objects: list[dict], match_results: list[dict]) -> dict[str, dict]:
    """Top30 LLM Validator(match_validator.py)를 대체하는 Rule 기반 품질
    검증(2026-07-17, 사용자 확정). 이미 계산된 Match Object 필드만 보고
    각 매칭에 "valid"/"weak"/"review" 상태 + 근거 문장을 매긴다 - 새
    LLM 호출도, 새 임베딩 호출도 없다(모두 기존 필드 조회/비교뿐). 반환:
    {jd_object_id: {"quality": "valid"|"weak"|"review", "quality_reason": str}}.
    Missing Object(매칭 자체가 없는 것)는 검증 대상이 아니다 - 검증할
    매칭이 없기 때문(match_validator.py와 동일한 원칙)."""
    resume_by_id = {(obj.get("id") or obj.get("normalized_text")): obj for obj in resume_objects}
    jd_by_id = {_obj_id(obj): obj for obj in jd_objects}

    result: dict[str, dict] = {}
    for m in match_results:
        if not m.get("matched_resume_objects"):
            continue
        jd_obj = jd_by_id.get(m["jd_object"], {})
        best = m["matched_resume_objects"][0]
        quality, reason = _rule_quality(jd_obj, best, resume_by_id)
        result[m["jd_object"]] = {"quality": quality, "quality_reason": reason}
    return result


def derive_match_relation(match_object: dict, resume_objects: list[dict], resume_relations: list[dict]) -> dict | None:
    """Match Object(JD Object 1개에 대한 매칭 결과)와 이력서의 Document-
    internal Relation을 조합해서 Resume<->JD 교차 설명 체인을 그 자리
    에서 만든다(저장하지 않음, 새 LLM 호출 없음 - 체크포인트 6
    Presentation Layer에서 실제로 쓰인다). 가장 확신도 높은 매칭
    (matched_resume_objects[0])을 시작점으로, 이력서 내부 relation을
    따라가며 체인을 확장한다.

    주의: relations는 normalized_text로 노드를 참조하지만(understanding.py
    프롬프트 스키마), Match Object의 resume_object는 id(또는 id가 없는
    옛 데이터는 normalized_text)를 쓴다 - 둘 다로 조회 가능하도록
    id<->normalized_text 매핑을 먼저 만든다."""
    if not match_object.get("matched_resume_objects"):
        return None
    root_id = match_object["matched_resume_objects"][0]["resume_object"]

    id_to_text = {_obj_id(o): (o.get("normalized_text") or _obj_id(o)) for o in resume_objects}
    text_to_id = {v: k for k, v in id_to_text.items()}

    root_text = id_to_text.get(root_id, root_id)
    next_of = {r["from"]: r["to"] for r in resume_relations if r.get("from") and r.get("to")}

    chain_text, cur, seen = [root_text], root_text, {root_text}
    while cur in next_of and next_of[cur] not in seen:
        cur = next_of[cur]
        chain_text.append(cur)
        seen.add(cur)

    chain = [text_to_id.get(t, t) for t in chain_text]
    return {"jd_object": match_object["jd_object"], "resume_chain": chain, "relation": "satisfies"}
