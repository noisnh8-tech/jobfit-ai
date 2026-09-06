"""
resume_input/match_validator.py

Top30 Validator(체크포인트 4, docs/semantic_object_schema.md) -
Semantic Matching Engine(semantic_matching.py)이 만든 Match Object를
LLM이 검증만 한다. 옛 meaning_judgment.py처럼 이력서/JD 원문을 통째로
다시 읽어 처음부터 판단하지 않는다 - Engine이 이미 계산한 Match Object
(engine_confidence/match_type/matched_fields 등)를 입력으로 받아
"이 판단이 타당한가"만 확인한다.

LLM이 할 수 있는 것: relation 확인(direct/related/weak/none),
llm_validation_score 부여(engine_confidence와 별도 저장, 절대 덮어쓰지
않음), judgement(충족/부분충족/보완필요)+reasoning 부여, 동의하지 않으면
counter_evidence/disagreement_reason 기록.

LLM이 절대 하면 안 되는 것: 새 Match 생성, 새 의미 추론, Engine Match
Object 내용(jd_object/resume_object/match_type/engine_confidence) 수정
·삭제, 새 Evidence 생성(제공된 Evidence Context 밖의 것 인용 금지).

`validate_matches()`의 반환값은 ranking_score와 완전히 분리된 별도
dict다 - 이 모듈은 semantic_matching.py가 만든 Match Object나
ranking_score를 절대 변경하지 않는다(체크포인트 3 Acceptance ④
"Ranking 불변" 원칙).

아직 파이프라인에 연결되지 않았다(체크포인트 5에서 pipeline.py가
연결한다) - 지금은 독립 검증 대상. 옛 meaning_judgment.py는 그대로
살아있고 계속 실제 추천에 쓰인다(체크포인트 7에서 제거).

캐시(2026-07-16 추가): 지금까지 이 모듈은 검색할 때마다 Top30 전부를
매번 다시 LLM으로 검증하고 있었다(캐시 없음) - 같은 이력서로 같은
공고를 내일 다시 봐도 Validator가 또 호출됐다. `(resume_hash, job_id,
jd_hash, matching_version, validator_version)` 조합으로 캐시한다.
`jd_hash`는 이 job에 실제로 넘어온 `jd_objects`(Semantic Object 리스트)
내용의 해시다 - JD가 재생성돼 내용이 바뀌면(예: posting_hash 변경으로
JD Understanding이 재생성된 경우) 자동으로 캐시 미스가 나서 다시
검증한다. `matching_version`(semantic_matching.py)이 바뀌면(엔진
채점 로직 변경) 같은 이력서/JD 조합이어도 다시 검증한다. `validator_
version`은 이 모듈의 프롬프트가 바뀌면 올린다 - job_service.py/
cover_letter.py와 동일한 관례(resume_hash + 도메인 키 + 버전 컬럼).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from resume_input.llm_client import call_llm
from resume_input.semantic_matching import MATCHING_VERSION

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db

# 이 모듈의 검증 프롬프트/로직이 바뀌면 올린다(2026-07-16 추가) -
# validator_cache 캐시 무효화 기준.
VALIDATOR_VERSION = "v1-2026-07-16"

_VALIDATION_PROMPT = """당신은 이미 계산된 매칭 결과를 검증하는 역할만
합니다. 스스로 새로운 매칭을 만들거나 새로운 의미를 추론하지 마세요 -
아래 각 항목에 대해 "이 매칭이 실제로 타당한가"만 판단하세요.

절대 금지 사항:
- 새로운 매칭(Match)을 만들지 마세요 - 아래 목록에 없는 항목을 추가로
  판단하지 마세요.
- 제공된 [근거 문맥] 밖의 내용을 인용하지 마세요(이력서에 없는 경험을
  있다고 지어내지 마세요).
- Engine이 이미 정한 판단(match_type/engine_confidence)을 바꾸려 하지
  마세요 - 당신은 그 판단에 대한 독립적인 검증 의견만 냅니다.

각 항목에 대해 아래를 판단하세요:
- relation: "direct"(직접 관련) / "related"(간접 관련) / "weak"(약한
  관련, 표면적 구조만 비슷할 뿐 내용은 다름) / "none"(관련 없음)
- llm_validation_score: 0~1 사이 당신의 독립적인 확신도(Engine의
  engine_confidence와 별개 값입니다 - 둘 다 따로 저장됩니다)
- judgement: "충족" / "부분충족" / "보완필요" 중 하나
- reasoning: 1~2문장 근거(반드시 [근거 문맥]에 실제로 있는 내용만 인용)
- counter_evidence: **relation이 "weak" 또는 "none"이면 반드시 채우세요**
  (relation이 "direct"/"related"면 비워도 됩니다) - Engine의 판단
  (matched_fields 등)에 왜 동의하지 않는지 구체적 근거를 배열로 1개
  이상 쓰세요. reasoning과 내용이 겹쳐도 괜찮습니다 - 이 필드가
  나중에 "Engine과 LLM이 왜 갈렸는지" 추적하는 공식 기록입니다.
- disagreement_reason: counter_evidence를 채웠으면 그걸 1문장으로
  요약하세요(비우지 마세요). counter_evidence가 빈 배열일 때만
  `""`로 둡니다.

[검증할 항목 목록]
{items_block}

전체를 JSON 배열로만 답하세요(다른 설명 없이):
[
  {{"candidate_id": "<그대로>", "relation": "...", "llm_validation_score": 0.0, "judgement": "...", "reasoning": "...", "counter_evidence": [], "disagreement_reason": ""}}
]
"""

_ITEM_BLOCK = """
--- candidate_id={candidate_id} ---
[JD 요구사항] {jd_point} (중요도: {importance}, 유형: {requirement_type})
[이력서에서 찾은 근거] {resume_point}
[Engine 판단] match_type={match_type}, engine_confidence={engine_confidence}
[일치 항목] {matched_fields} / [부족 항목] {missing_fields}
[근거 문맥](이력서 원문 중 해당 부분 - 이 안에 있는 내용만 인용 가능)
{evidence_context}
"""


def extract_evidence_context(source_text: str, raw_text: str, window: int = 300) -> str:
    """source_text가 원문(raw_text) 안에서 있던 위치를 찾아 앞뒤 문맥을
    슬라이싱한다(순수 문자열 탐색 - 새 LLM 호출 아님, job_prep.py의
    _extract_block()류 패턴과 동일한 원칙). 못 찾으면 source_text
    자체를 그대로 반환한다(최소한의 근거는 유지)."""
    if not source_text or not raw_text:
        return source_text or ""
    idx = raw_text.find(source_text)
    if idx == -1:
        return source_text
    start = max(0, idx - window)
    end = min(len(raw_text), idx + len(source_text) + window)
    return raw_text[start:end]


def _parse_json(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _resume_hash(resume_raw: str) -> str:
    return hashlib.sha256(resume_raw.encode("utf-8")).hexdigest()[:16]


def _jd_hash(jd_objects: list[dict]) -> str:
    """이 job에 넘어온 jd_objects(Semantic Object 리스트) 내용의 해시 -
    JD가 재생성돼 내용이 바뀌면 자동으로 캐시 미스가 나게 한다."""
    payload = json.dumps(jd_objects, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _init_cache_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validator_cache (
            resume_hash TEXT NOT NULL,
            job_id TEXT NOT NULL,
            jd_hash TEXT NOT NULL,
            matching_version TEXT NOT NULL,
            validator_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (resume_hash, job_id, jd_hash, matching_version, validator_version)
        )
        """
    )
    conn.commit()
    conn.close()


def _build_items(resume_by_id: dict, jd_objects: list[dict], match_results: list[dict], resume_raw: str) -> list[dict]:
    by_jd_id = {m["jd_object"]: m for m in match_results}
    items = []
    for jd_obj in jd_objects:
        jd_id = jd_obj.get("id") or jd_obj.get("normalized_text")
        match = by_jd_id.get(jd_id)
        if not match or not match.get("matched_resume_objects"):
            continue  # Missing Object는 검증 대상 아님 - 애초에 매칭이 없다.
        best = match["matched_resume_objects"][0]
        r_obj = resume_by_id.get(best["resume_object"], {})
        source_text = (r_obj.get("evidence") or {}).get("source_text", "")
        items.append({
            "candidate_id": match["jd_object"],
            "jd_point": jd_obj.get("meaning") or jd_obj.get("normalized_text", ""),
            "importance": jd_obj.get("importance", ""),
            "requirement_type": jd_obj.get("requirement_type", ""),
            "resume_point": r_obj.get("meaning") or r_obj.get("normalized_text", ""),
            "match_type": best["match_type"],
            "engine_confidence": best["engine_confidence"],
            "matched_fields": best.get("matched_fields", []),
            "missing_fields": best.get("missing_fields", []),
            "evidence_context": extract_evidence_context(source_text, resume_raw),
        })
    return items


def validate_matches(
    resume_objects: list[dict], jd_objects: list[dict], match_results: list[dict],
    resume_raw: str, job_id: str = "", provider: str = "gemini", db_path: Path = DB_PATH,
) -> dict[str, dict]:
    """Match Object들을 LLM으로 검증한다. 반환:
    {jd_object_id: {relation, llm_validation_score, judgement, reasoning,
    counter_evidence, disagreement_reason}}. Missing Object(매칭 자체가
    없는 것)는 검증하지 않는다 - 검증할 매칭이 없기 때문. 이 dict는
    match_results/ranking_score를 전혀 건드리지 않는다(호출부가 화면
    표시용으로만 별도 사용 - 체크포인트 6).

    캐시(2026-07-16 추가): `job_id`를 넘기면 (resume_hash, job_id,
    jd_hash, matching_version, validator_version) 조합으로 캐시를
    조회/저장한다 - 같은 이력서로 같은 공고를 다시 보면(내일 재검색
    등) LLM을 다시 부르지 않는다. `job_id`를 안 넘기면(하위 호환)
    캐시 없이 항상 새로 검증한다."""
    resume_by_id = {(obj.get("id") or obj.get("normalized_text")): obj for obj in resume_objects}

    items = _build_items(resume_by_id, jd_objects, match_results, resume_raw)
    if not items:
        return {}

    r_hash = _resume_hash(resume_raw) if job_id else ""
    j_hash = _jd_hash(jd_objects) if job_id else ""

    if job_id:
        _init_cache_table(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT result_json FROM validator_cache WHERE resume_hash=? AND job_id=? AND jd_hash=? "
            "AND matching_version=? AND validator_version=?",
            (r_hash, job_id, j_hash, MATCHING_VERSION, VALIDATOR_VERSION),
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])

    blocks = "\n".join(_ITEM_BLOCK.format(**it) for it in items)
    prompt = _VALIDATION_PROMPT.format(items_block=blocks)
    max_tokens = min(8192, 400 * len(items) + 400)

    raw = call_llm(prompt, provider=provider, max_tokens=max_tokens)
    try:
        parsed = _parse_json(raw)
    except json.JSONDecodeError:
        raw = call_llm(prompt + "\n\n(JSON 배열로만, 간결하게 다시 답하세요.)", provider=provider, max_tokens=max_tokens)
        parsed = _parse_json(raw)

    result = {str(entry.get("candidate_id")): entry for entry in parsed}

    if job_id:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT OR REPLACE INTO validator_cache "
            "(resume_hash, job_id, jd_hash, matching_version, validator_version, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r_hash, job_id, j_hash, MATCHING_VERSION, VALIDATOR_VERSION, json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()

    return result
