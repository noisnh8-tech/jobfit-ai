"""
resume_input/link_engine.py

Semantic Linking Engine(semantic-link-v1, 2026-08-03 스키마 확정) - Resume
Understanding과 JD Understanding이 만든 Semantic Object 전체를 입력받아,
JD Semantic Object 하나하나에 대해 Resume의 어떤 근거가 대응하는지
연결(Link)만 하는 단일 LLM 호출 엔진.

역할 경계(SYSTEM_DESIGN.md §6 "각 서비스 단계는 하나의 책임만 가진다"):
이 엔진은 "무엇이 연결되고, 무엇이 부족하며, 어떤 근거가 있는가"까지만
반환한다. 점수 계산·지원 여부 판단·이력서 문장 생성은 하지 않는다 -
Judge/Resume Customizing/Checklist는 이 결과만 소비하는 별도 Rule
엔진(LLM 호출 없음)이 담당한다. 설계 확정 근거·논의 과정 전체는
docs/design/semantic_linking_schema_v1.md 참고.

이전 버전(v2-2026-07-23-single-call, git 히스토리 참고)은 flat item-level
list + gap_type + match_strength(primary/secondary/supporting) 스키마
였다. 이번 스키마는 그걸 완전히 대체한다(필드 재사용 없음 - JD Object
단위 links 배열 + relation_reason_type 5종 고정 enum + missing(타입 있는
배열) + improvement(단일 객체, rewrite/genuine_gap 배타) +
qualification_validation(Rule 선계산, LLM은 참고만)).

구현 범위(2026-08-03): Semantic Linking까지만. qualification_validation
계산 함수/프롬프트 조립/출력 검증까지 전부 구현했고 LLM 호출 코드
(run_semantic_linking)도 완성했지만, 이 커밋 시점까지는 아직 실행(첫
LLM 호출)하지 않았다 - Step 5(5건 스키마 검증)를 사용자 확인 후 실행한다.
pipeline.py/app.py에는 아직 연결하지 않았다 - 기존 Rule 엔진
(semantic_matching.py)이 계속 프로덕션 경로로 살아있다.
"""
from __future__ import annotations

import json

from resume_input.career_filter import compute_career_status
from resume_input.llm_client import DEFAULT_PROVIDER, call_llm

LINK_SCHEMA_VERSION = "semantic-link-v1"

_MATCHABLE_LAYERS = ("problem", "thinking", "task", "skill", "qualification")
_JD_ONLY_LAYERS = ("culture",)
_JD_LAYERS = _MATCHABLE_LAYERS + _JD_ONLY_LAYERS

# JD 레이어별로 연결 가능한 Resume 레이어 화이트리스트. culture만 예외로
# Resume에 대응 레이어가 없어 thinking/task를 교차로 허용한다(2026-07-23
# 구버전에서 이미 확정된 화이트리스트 - 스키마가 바뀌어도 이 규칙 자체는
# 그대로 승계한다).
_CROSS_LAYER_ALLOWED: dict[str, tuple[str, ...]] = {
    "problem": ("problem",),
    "thinking": ("thinking",),
    "task": ("task",),
    "skill": ("skill",),
    "qualification": ("qualification",),
    "culture": ("thinking", "task"),
}

_RELATION_VALUES = ("match", "partial_match", "no_match")
_ALLOWED_RELATION_STRENGTH: dict[str, tuple[str, ...]] = {
    "match": ("strong", "moderate"),
    "partial_match": ("strong", "moderate", "weak"),
    "no_match": ("none",),
}
_REASON_TYPE_VALUES = (
    "scope_gap", "experience_gap", "domain_gap", "skill_gap", "evidence_gap",
    # responsibility_gap(2026-08-03 추가, 사용자 확정) - "팀 KPI vs 전사 KPI"처럼
    # 도메인도 경험 부족도 아니라 책임/권한 범위 자체가 커진 경우. scope_gap에
    # 넣을 수도 있었지만, 실측(5건 테스트)에서 "범위" 표현이 자주 나와서
    # 별도 카테고리로 분리해야 Judge가 구분해서 쓸 수 있다는 판단.
    "responsibility_gap",
)
_IMPROVEMENT_TYPES = ("rewrite", "genuine_gap")
_IMPROVEMENT_ACTIONS = ("reposition", "rephrase", "add_emphasis")

# matched_dimensions(2026-08-03 추가, 사용자 확정) - relation을 "왜" 그렇게
# 정했는지 구조화된 근거로 남긴다. LLM의 내부 비교 과정(action/goal/object/
# scope 각각 same/similar/different) 자체를 출력시키지 않는다 - 그건 CoT를
# 그대로 노출하는 것이라 프롬프트에서는 "내부적으로 비교만 하고 결과는
# 요약해서 담아라"로 지시한다. 여기 담기는 건 "이 축은 일치한다고 판단한
# 것들의 목록"뿐이다(예: action/goal만 일치, scope는 불일치 -> ["action",
# "goal"]). Judge/Resume Customizing이 reason(자유 서술)보다 이 필드를
# 더 안정적으로 재사용할 수 있다.
_MATCHED_DIMENSION_VALUES = ("action", "goal", "object", "scope")


def _obj_id(obj: dict) -> str:
    return obj.get("id") or obj.get("normalized_text") or ""


def _by_layers(objects: list[dict], layers: tuple[str, ...]) -> list[dict]:
    return [o for o in objects if o.get("layer") in layers]


# ── Qualification Validation(Rule 선계산) ───────────────────────────
# LLM에게 "자격 충족 여부"를 스스로 판단시키지 않는다(2026-08-03, 사용자
# 확정) - 신입인데 "3년 경력 있음" 같은 임의 판단을 막기 위해, 이미
# 저장된 구조화 데이터만으로 계산해서 프롬프트에 고정 사실로 넣고 LLM은
# 인용만 하게 한다. education/certificate/language는 키워드 스캔(v1,
# 잠정치 - Step 5(5건 검증)에서 오탐/누락 여부를 같이 확인한다).
# experience는 새 판단 로직을 만들지 않고 candidate_search.py의
# Candidate Generation 단계가 이미 쓰는 career_filter.compute_career_
# status()를 그대로 재사용한다(constraint_layer.py와 동일 원칙 - 검증된
# 로직 중복 구현 금지).
_EDUCATION_KEYWORDS = ("학사", "석사", "박사", "전공", "졸업", "학위", "대학교", "대학원", "수료")
_CERTIFICATE_KEYWORDS = ("자격증", "합격", "취득", "인증", "ADsP", "ADP", "SQLD", "컴퓨터활용능력")
_LANGUAGE_KEYWORDS = ("TOEIC", "토익", "OPIc", "오픽", "TOEFL", "토플", "IELTS", "JLPT", "HSK", "어학")


def _qualification_text(resume_objects: list[dict]) -> str:
    parts: list[str] = []
    for obj in resume_objects:
        if obj.get("layer") != "qualification":
            continue
        parts.append(obj.get("normalized_text", ""))
        parts.append(obj.get("meaning", ""))
        parts.extend(obj.get("concepts") or [])
    return " ".join(p for p in parts if p)


def compute_qualification_validation(
    resume_objects: list[dict],
    job: dict,
    user_career_level: str | None,
    challenge_option: bool = False,
) -> dict:
    """Resume qualification 레이어 Object + career_filter 기존 로직만으로
    계산하는 Rule 기반 사실. "수료"는 education/certificate 경계가
    모호할 수 있음(교육과정 수료가 자격증은 아니지만 학위도 아님) - v1
    한계로 남겨두고 Step 5에서 실제 사례로 확인한다."""
    text = _qualification_text(resume_objects)
    career_status = (
        compute_career_status(user_career_level, job.get("career_level"), challenge_option)
        if user_career_level else None
    )
    return {
        "education": any(k in text for k in _EDUCATION_KEYWORDS),
        "certificate": any(k in text for k in _CERTIFICATE_KEYWORDS),
        "language": any(k in text for k in _LANGUAGE_KEYWORDS),
        "experience": career_status in ("적합", "확인필요") if career_status else False,
    }


# ── Prompt Builder ──────────────────────────────────────────────────
def _build_allowed_pairs_text() -> str:
    lines = []
    for jd_layer in _MATCHABLE_LAYERS:
        lines.append(f"- JD {jd_layer} -> Resume [{jd_layer}] 항목만 연결 가능")
    return "\n".join(lines)


def _format_frame(frame: dict | None) -> str:
    frame = frame or {}
    method = ", ".join(frame.get("method") or [])
    return (
        f"action={frame.get('action', '')!r} object={frame.get('object', '')!r} "
        f"goal={frame.get('goal', '')!r} domain={frame.get('domain', '')!r} "
        f"outcome={frame.get('outcome', '')!r} method=[{method}]"
    )


def _format_jd_objects(jd_objects: list[dict]) -> str:
    """JD Understanding이 만든 필드 전체(frame/importance/expected_evidence/
    concepts까지)를 프롬프트에 넣는다 - 지금 semantic_matching.py의
    cosine 매칭은 사실상 normalized_text/frame 일부만 보고 embedding_text
    코사인으로 판단하는데, 여기서는 애써 구조화한 정보(persona_tags,
    expected_evidence 등)를 버리지 않고 전부 판단 근거로 쓰는 게 이번
    설계의 핵심이다(2026-08-03 논의)."""
    lines = []
    for o in jd_objects:
        ev = (o.get("evidence") or {}).get("source_text", "")
        expected = "; ".join(
            f"{e.get('type')}:{e.get('value')}" for e in (o.get("expected_evidence") or [])
        )
        lines.append(
            f"- id={_obj_id(o)} layer={o.get('layer')} importance={o.get('importance', '')} "
            f"requirement_type={o.get('requirement_type', '')}\n"
            f"  normalized_text={o.get('normalized_text', '')!r}\n"
            f"  meaning={o.get('meaning', '')!r}\n"
            f"  frame: {_format_frame(o.get('frame'))}\n"
            f"  expected_evidence: {expected or '(없음)'}\n"
            f"  concepts: {', '.join(o.get('concepts') or [])}\n"
            f"  evidence.source_text={ev!r}"
        )
    return "\n".join(lines) if lines else "(없음)"


def _format_resume_objects(resume_objects: list[dict]) -> str:
    lines = []
    for o in resume_objects:
        ev = o.get("evidence") or {}
        lines.append(
            f"- id={_obj_id(o)} layer={o.get('layer')} importance={o.get('importance', '')}\n"
            f"  normalized_text={o.get('normalized_text', '')!r}\n"
            f"  meaning={o.get('meaning', '')!r}\n"
            f"  frame: {_format_frame(o.get('frame'))}\n"
            f"  persona_tags: {', '.join(o.get('persona_tags') or [])}\n"
            f"  concepts: {', '.join(o.get('concepts') or [])}\n"
            f"  evidence: project={ev.get('project', '')!r} section={ev.get('section', '')!r} "
            f"source_text={ev.get('source_text', '')!r}"
        )
    return "\n".join(lines) if lines else "(없음)"


_LINK_PROMPT = """당신은 채용공고(JD)와 이력서(Resume)를 의미 단위로 이미 구조화한 결과를
받아, JD의 개별 요구사항(Semantic Object) 하나하나에 대해 Resume의 어떤
근거가 대응하는지 연결하는 역할만 합니다.

당신은 지원 여부를 판단하지 않습니다. 점수를 계산하지 않습니다. 이력서
문장을 새로 쓰지 않습니다. 당신의 역할은 "JD Semantic Object 하나하나가,
Resume의 어떤 Semantic Object로 근거를 확인할 수 있는가"를 연결하는
것뿐입니다.

[레이어 연결 규칙 - 반드시 지키세요]
{allowed_pairs}
Culture만 예외적으로 Resume의 thinking/task와 연결 가능합니다(Resume에
culture 레이어가 없기 때문 - 조직문화/인재상 요구는 지원자의 사고방식·
실제 행동으로 드러납니다).

[이미 Rule로 확인된 사실 - qualification 레이어 판단 시 반드시 이 값을
그대로 인용하고, 스스로 다시 판단하지 마세요]
{qualification_validation_json}
(education=학위/전공 근거 존재, certificate=자격증 존재, language=어학
성적 존재, experience=JD가 요구하는 연차 수준 충족 여부 - 전부 Rule이
이미 계산한 사실입니다. 예: JD가 실무 경력을 요구하는데 experience=false면
Resume에 교육이수/자격증만 있어도 relation="match"를 주면 안 됩니다.)

[JD Semantic Object 전체 - 총 {jd_count}개, 각각 정확히 1개의 link를
반환하세요]
{jd_items}

[Resume Semantic Object 전체]
{resume_items}

[판단 순서 - 반드시 이 순서로]
1. JD Object의 요구 의미를 확인한다(normalized_text/meaning/frame/
   expected_evidence를 전부 본다 - frame만 보거나 표면 단어만 보지
   않는다).
2. 레이어 연결 규칙에서 허용된 Resume 레이어 안에서만 근거를 찾는다.
3. 가장 직접적인 근거만 선택한다(여러 개면 matched_resume_objects에
   전부 담아도 되지만, 억지로 관련 없는 항목을 끼워넣지 않는다).
4. relation을 정하기 전에, 다음 4가지 축을 내부적으로 비교하세요(이
   비교 과정 자체는 출력하지 않습니다 - 아래 matched_dimensions에는
   "일치한다고 판단한 축의 이름만" 요약해서 담습니다):
   - action(핵심 행위) - 실제로 하는 일이 같은가?
   - goal(목적) - 왜 그 일을 하는가가 같은가?
   - object(대상) - 무엇을 대상으로 하는가가 같은가?
   - scope(범위) - 규모/권한/환경이 같은가? (예: 팀 단위 vs 전사 단위)
   relation을 정한다: "match"(네 축이 전부 또는 대부분 일치, 핵심
   행위·대상·목적이 명확히 대응) / "partial_match"(action/goal은
   일치하나 object나 scope 등 일부가 다름) / "no_match"(대부분
   불일치하거나 근거 자체가 없음).
   matched_dimensions에는 "일치한다"고 판단한 축 이름만 배열로 담습니다
   (예: action/goal만 일치, object/scope는 불일치 -> ["action","goal"]).
   match면 보통 4개 다 담기고, no_match면 보통 빈 배열입니다.
5. match_strength를 정한다(relation과 반드시 아래 조합만 허용):
   - match: strong(양쪽 근거가 구체적이고 대응이 명확) 또는 moderate
   - partial_match: strong, moderate, 또는 weak(추론 비중이 크고 근거
     부족)
   - no_match: none 고정
   match_strength는 당신의 확신도가 아니라 "JD·Resume 양쪽 근거가 얼마나
   구체적인가"입니다.
6. relation이 match가 아니면 relation_reason_type을 다음 6개 중 정확히
   하나로 채웁니다(match면 채우지 않습니다): scope_gap(다루는 대상/방식의
   범위만 다름) / responsibility_gap(팀 단위 KPI vs 전사 단위 KPI처럼
   책임·권한의 규모 자체가 다름 - scope_gap과 헷갈리면, "같은 일을 더
   넓은 범위에서 하는가"면 responsibility_gap, "비슷한 일이지만 대상만
   다른가"면 scope_gap) / experience_gap(실무 경험 부족) / domain_gap
   (업무 영역이 다름) / skill_gap(요구 기술 자체가 없음) / evidence_gap
   (관련 있는 Resume Object가 존재하지만, 그 근거만으로는 JD가 기대하는
   수준을 충분히 증명하지 못함 - **이 경우 matched_resume_objects에 그
   관련 Resume Object를 반드시 채우세요**, 최소 이렇게 판단한 근거가
   있어야 합니다). 관련 있어 보이는 Resume Object를 하나도 찾지
   못했다면 evidence_gap을 쓰지 마세요 - 그때는 domain_gap/skill_gap/
   experience_gap 중 가장 맞는 것을 쓰고 matched_resume_objects는
   빈 배열로 둡니다("관련 있어 보인다"는 추측만으로 근거 없이 evidence_
   gap을 고르지 마세요 - Resume에 실제로 존재하는 관련 항목이 있을
   때만 evidence_gap입니다).
7. missing을 도출한다(relation이 match가 아닐 때). 각 항목은
   {{"type": "experience|skill|qualification|domain", "text": "..."}}
   형태입니다.
8. relation이 match가 아니면 improvement를 정확히 1개만 반환합니다(배열
   아님). 자연어 설명(reason/suggestion)은 만들지 마세요 - Resume
   Customizing이 action 값만으로 Rule Template 문장을 만듭니다. 두
   종류 중 하나입니다:
   - {{"type": "rewrite", "action": "reposition|rephrase|add_emphasis",
     "target_resume_object_id": "..."}} - 실제 경험은 있으나 표현이
     약한 경우만. target_resume_object_id는 반드시 matched_resume_
     objects 안의 id여야 합니다.
   - {{"type": "genuine_gap"}} - 경험 자체가 없는 경우. 이 경우
     target_resume_object_id와 action은 절대 채우지 마세요.
   relation이 match면 improvement는 null입니다.

[절대 금지]
- Resume/JD에 없는 내용을 지어내지 마세요. matched_resume_objects의
  evidence_text는 반드시 해당 Resume Object의 evidence.source_text
  그대로여야 합니다.
- 종합 점수, 지원 추천 여부, 자기소개서 문장, 자유 서술형 설명(reason/
  suggestion)을 만들지 마세요 - relation/relation_reason_type/
  matched_dimensions만으로 "왜"를 설명하고, 나머지는 요청한 범주값
  필드만 채웁니다.

[출력 형식 - JSON만, 다른 설명 없이]
{{
  "link_version": "semantic-link-v1",
  "links": [
    {{
      "layer": "problem|thinking|task|skill|qualification|culture",
      "jd_object_id": "...",
      "jd_requirement": "...",
      "relation": "match|partial_match|no_match",
      "match_strength": "strong|moderate|weak|none",
      "relation_reason_type": "scope_gap|responsibility_gap|experience_gap|domain_gap|skill_gap|evidence_gap 또는 null",
      "matched_dimensions": ["action", "goal", "object", "scope" 중 일치하는 것만],
      "matched_resume_objects": [
        {{"resume_object_id": "...", "evidence_text": "...", "source_project": "...", "source_section": "..."}}
      ],
      "missing": [{{"type": "...", "text": "..."}}],
      "improvement": {{"type": "rewrite|genuine_gap", "action": "...또는 null", "target_resume_object_id": "...또는 null"}} 또는 null
    }}
  ]
}}
"""


def build_link_prompt(jd_objects: list[dict], resume_objects: list[dict], qualification_validation: dict) -> str:
    jd_items = _by_layers(jd_objects, _JD_LAYERS)
    resume_items = _by_layers(resume_objects, _MATCHABLE_LAYERS)
    return _LINK_PROMPT.format(
        jd_count=len(jd_items),
        allowed_pairs=_build_allowed_pairs_text(),
        qualification_validation_json=json.dumps(qualification_validation, ensure_ascii=False),
        resume_items=_format_resume_objects(resume_items),
        jd_items=_format_jd_objects(jd_items),
    )


def parse_link_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── Output 검증/정리(이중 방어 - 프롬프트 지시를 어겨도 코드가 강제) ──
# 2026-08-03(사용자 확정) - 이 함수가 곧 사용자가 말한 "Link Validator"
# 단계다(Semantic Link -> Rule Validation -> Judge). 여기서 걸러진 결과만
# judge_engine.py가 받는다 - LLM 출력이 이상해도 Judge는 항상 스키마가
# 보장된 깨끗한 데이터만 받는다.
def _validate_and_clean_links(
    parsed: dict,
    jd_objects: list[dict],
    resume_objects: list[dict],
    qualification_validation: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """반환: (정리된 links, issues). issues는 스키마 밖 진단 정보다 -
    warnings 필드를 스키마에서 뺀 이유(Link의 책임이 아님)와 같은 이유로
    최종 결과물에 넣지 않고, Step 5 검증/로그 전용으로만 별도 반환한다."""
    issues: list[str] = []
    valid_jd_ids = {_obj_id(o) for o in jd_objects}
    jd_by_id = {_obj_id(o): o for o in jd_objects}
    resume_by_id = {_obj_id(o): o for o in resume_objects}
    seen_jd_ids: set[str] = set()

    cleaned: list[dict] = []
    for link in parsed.get("links") or []:
        jid = link.get("jd_object_id")
        if jid not in valid_jd_ids:
            issues.append(f"존재하지 않는 jd_object_id 무시: {jid}")
            continue
        if jid in seen_jd_ids:
            issues.append(f"jd_object_id 중복, 첫 항목만 사용: {jid}")
            continue
        seen_jd_ids.add(jid)

        # importance(2026-08-03 추가) - LLM에게 안 묻는다. JD Object 자신의
        # importance를 그대로 복사한다(judge_engine.py가 critical 요구사항을
        # 가려내려면 이 값이 필요한데, Link가 스스로 판단하는 값이 아니라
        # Understanding 단계에서 이미 정해진 값을 그대로 들고 오는 것뿐이다 -
        # "Judge는 Link 결과만 소비한다"는 원칙을 지키면서도 Judge가 Understanding
        # 데이터를 다시 열어보지 않아도 되게 하기 위함).
        link["importance"] = jd_by_id.get(jid, {}).get("importance", "normal")
        # concepts(2026-08-03 추가) - importance와 같은 이유. Judge가 "같은
        # 근본 원인이 여러 레이어에서 중복 감점되는지"(예: task의 Airflow
        # 요구와 skill의 Airflow 요구)를 판단하려면 JD Object의 통제 어휘
        # (concepts)가 필요하다 - Judge가 스스로 재해석하는 게 아니라
        # Understanding이 이미 뽑아둔 값을 그대로 복사해오는 것뿐이다.
        link["concepts"] = jd_by_id.get(jid, {}).get("concepts") or []

        relation = link.get("relation")
        if relation not in _RELATION_VALUES:
            issues.append(f"{jid}: 알 수 없는 relation={relation!r}, no_match로 강제")
            relation = "no_match"
        link["relation"] = relation

        strength = link.get("match_strength")
        allowed_strength = _ALLOWED_RELATION_STRENGTH[relation]
        if strength not in allowed_strength:
            issues.append(f"{jid}: relation={relation}에 허용되지 않는 match_strength={strength!r}, {allowed_strength[0]}로 강제")
            strength = allowed_strength[0]
        link["match_strength"] = strength

        reason_type = link.get("relation_reason_type")
        if relation == "match":
            if reason_type is not None:
                issues.append(f"{jid}: relation=match인데 relation_reason_type이 채워짐, null로 강제")
            link["relation_reason_type"] = None
        else:
            if reason_type not in _REASON_TYPE_VALUES:
                issues.append(f"{jid}: relation={relation}인데 relation_reason_type={reason_type!r} 무효, evidence_gap으로 강제")
                reason_type = "evidence_gap"
            link["relation_reason_type"] = reason_type

        matched_dims = link.get("matched_dimensions")
        if not isinstance(matched_dims, list):
            issues.append(f"{jid}: matched_dimensions가 리스트가 아님({matched_dims!r}), 빈 리스트로 강제")
            matched_dims = []
        cleaned_dims = sorted({d for d in matched_dims if d in _MATCHED_DIMENSION_VALUES})
        if len(cleaned_dims) != len(matched_dims):
            issues.append(f"{jid}: matched_dimensions에 무효 값 포함, 유효한 것만 남김: {cleaned_dims}")
        link["matched_dimensions"] = cleaned_dims

        layer = link.get("layer")
        cleaned_matches = []
        for m in link.get("matched_resume_objects") or []:
            rid = m.get("resume_object_id")
            r_obj = resume_by_id.get(rid)
            if r_obj is None:
                issues.append(f"{jid}: 존재하지 않는 resume_object_id 무시: {rid}")
                continue
            if r_obj.get("layer") not in _CROSS_LAYER_ALLOWED.get(layer, ()):
                issues.append(f"{jid}: 허용되지 않는 레이어 연결({layer}->{r_obj.get('layer')}) 무시: {rid}")
                continue
            ev = r_obj.get("evidence") or {}
            # evidence_text/source_project/source_section은 LLM 출력을
            # 안 믿고 Resume Object 자신의 evidence 필드로 항상 덮어쓴다
            # (재작성 방지 - 근거 재인용이지 재해석이 아니라는 원칙).
            cleaned_matches.append({
                "resume_object_id": rid,
                "evidence_text": ev.get("source_text", ""),
                "source_project": ev.get("project", ""),
                "source_section": ev.get("section", ""),
            })
        link["matched_resume_objects"] = cleaned_matches

        # evidence_gap 방어(2026-08-14, 사용자 확정) - evidence_gap의 정의는
        # "관련 Resume Object는 있지만 근거가 부족하다"이므로, matched_resume_
        # objects가 비어있으면 정의상 모순이다(관련 항목이 있다면서 하나도
        # 안 채운 것). 프롬프트에서 이미 금지했지만, 이중 방어로 코드에서도
        # 강제 하향한다 - "Resume에서 후보를 못 찾음"과 "실제 경험이 없음"은
        # 다른 사실이므로, 이 다운그레이드가 genuine_gap을 의미하지는 않는다
        # (그건 Analysis 화면이 matched_resume_objects 유무로 별도 판단한다).
        if link["relation"] == "no_match" and link["relation_reason_type"] == "evidence_gap" and not cleaned_matches:
            issues.append(f"{jid}: evidence_gap인데 matched_resume_objects가 비어있음, experience_gap으로 강제 하향")
            link["relation_reason_type"] = "experience_gap"

        # 위 evidence_gap 방어의 반대 방향(2026-08-14, 사용자 확정, 30건
        # 검증 중 실측 발견 - wd/280517 qualification_001: relation_reason_
        # type=domain_gap인데 matched_resume_objects가 채워져 있었다).
        # 정의상 "matched_resume_objects가 있다"는 "관련 Resume Object를
        # 찾았다"는 뜻이고, 그건 evidence_gap의 정의 자체다 - domain_gap/
        # skill_gap/experience_gap은 프롬프트 지시상 "관련 항목을 하나도
        # 못 찾았을 때"만 써야 한다(있는데도 이 reason_type을 쓰면 Analysis
        # 화면의 5-state 분류가 "C(근거는 있으나 부족)"를 "D(근거 자체를
        # 못 찾음)"로 잘못 내려버린다 - selection_engine.py의 protected
        # 판정 기준(matched_resume_objects 유무)과 어긋나는 원인이 바로
        # 이 불일치였다). 양방향 모두 강제해서 "matched_resume_objects
        # 비어있음 <-> reason_type=evidence_gap이 아님"이 항상 같이
        # 가도록 만든다.
        if link["relation"] == "no_match" and link["relation_reason_type"] != "evidence_gap" and cleaned_matches:
            issues.append(
                f"{jid}: relation_reason_type={link['relation_reason_type']!r}인데 "
                f"matched_resume_objects가 채워져 있음, evidence_gap으로 강제 상향"
            )
            link["relation_reason_type"] = "evidence_gap"

        # Qualification Validation 일치성 검증(2026-08-03, 사용자 확정) -
        # LLM이 qualification_validation(Rule 선계산)을 무시하고 스스로
        # match를 줬는지 사후 검증한다. JD가 실무 경력(expected_evidence.
        # type in experience/project)을 요구하는데 Rule이 experience=False
        # 라고 이미 확인했다면, LLM이 match를 줬어도 인정하지 않는다(옛
        # semantic_matching.py의 Qualification Evidence Type Gate와 동일
        # 원칙 - 사실 검증이 유사도 판단보다 우선한다).
        if layer == "qualification" and qualification_validation is not None:
            jd_obj_full = jd_by_id.get(jid, {})
            requires_experience = any(
                ev.get("type") in ("experience", "project")
                for ev in (jd_obj_full.get("expected_evidence") or [])
            )
            if requires_experience and not qualification_validation.get("experience", True) and relation == "match":
                issues.append(f"{jid}: qualification_validation.experience=False인데 relation=match, partial_match로 강제 하향")
                link["relation"] = relation = "partial_match"
                link["match_strength"] = "moderate"
                link["relation_reason_type"] = "experience_gap"
                # 강제 하향 직후에는 LLM이 원래 match로 알고 improvement를
                # 안 만들었을 수 있다(match면 improvement 없음이 정상) -
                # 아래 improvement 블록이 "relation != match인데 improvement
                # 없음" 이슈를 내지 않도록, 이 하향의 원인을 그대로 담은
                # 기본 improvement를 여기서 채워둔다.
                link["improvement"] = {"type": "genuine_gap", "action": None, "target_resume_object_id": None}

        if relation != "no_match" and not cleaned_matches:
            issues.append(f"{jid}: relation={relation}인데 유효한 matched_resume_objects가 없어 no_match로 강제")
            link["relation"] = relation = "no_match"
            link["match_strength"] = "none"
            link["relation_reason_type"] = "evidence_gap"

        improvement = link.get("improvement")
        if link["relation"] == "match":
            if improvement:
                issues.append(f"{jid}: relation=match인데 improvement가 채워짐, null로 강제")
            link["improvement"] = None
        elif not improvement:
            issues.append(f"{jid}: relation={link['relation']}인데 improvement 없음")
            link["improvement"] = None
        else:
            itype = improvement.get("type")
            if itype not in _IMPROVEMENT_TYPES:
                issues.append(f"{jid}: 알 수 없는 improvement.type={itype!r}, 항목 제거")
                improvement = None
            elif itype == "rewrite" and link["relation"] == "no_match":
                # no_match인데 rewrite는 논리 모순이다(2026-08-03, 사용자
                # 확정 Link Validator 규칙) - rewrite는 "실제 경험은 있으나
                # 표현이 약함"을 뜻하는데, no_match는 애초에 대응할 만한
                # 근거가 없다는 뜻이라 재배치/재서술로 고칠 대상 자체가
                # 없다. genuine_gap으로 강제 전환한다.
                issues.append(f"{jid}: relation=no_match인데 improvement.type=rewrite, genuine_gap으로 강제 전환")
                improvement = {"type": "genuine_gap", "action": None, "target_resume_object_id": None}
            elif itype == "rewrite":
                if improvement.get("action") not in _IMPROVEMENT_ACTIONS:
                    issues.append(f"{jid}: rewrite인데 action={improvement.get('action')!r} 무효, rephrase로 강제")
                    improvement["action"] = "rephrase"
                target = improvement.get("target_resume_object_id")
                matched_ids = {m["resume_object_id"] for m in cleaned_matches}
                if not target or target not in matched_ids:
                    issues.append(f"{jid}: rewrite의 target_resume_object_id={target!r}가 matched_resume_objects 밖, genuine_gap으로 강제 전환")
                    improvement = {"type": "genuine_gap", "action": None, "target_resume_object_id": None}
            else:  # genuine_gap
                if improvement.get("target_resume_object_id"):
                    issues.append(f"{jid}: genuine_gap인데 target_resume_object_id 존재, 제거")
                improvement["target_resume_object_id"] = None
                improvement["action"] = None
            link["improvement"] = improvement

        if layer != "qualification":
            link.pop("qualification_validation", None)
        # reason/suggestion은 스키마에서 뺐다(2026-08-03, 사용자 확정 -
        # 자연어 설명은 Resume Customizing이 action으로 Rule Template을
        # 만든다) - LLM이 프롬프트를 무시하고 만들어도 저장하지 않는다.
        link.pop("reason", None)

        cleaned.append(link)

    missing_jd_ids = valid_jd_ids - seen_jd_ids
    for jid in missing_jd_ids:
        issues.append(f"LLM이 누락한 jd_object_id: {jid} (수동 no_match로 보강)")
        obj = jd_by_id[jid]
        cleaned.append({
            "layer": obj.get("layer"), "jd_object_id": jid,
            "jd_requirement": obj.get("normalized_text", ""),
            "importance": obj.get("importance", "normal"),
            "relation": "no_match", "match_strength": "none", "relation_reason_type": "evidence_gap",
            "matched_dimensions": [],
            "matched_resume_objects": [],
            "missing": [], "improvement": None,
        })

    return cleaned, issues


def _compute_unmatched_resume_objects(resume_objects: list[dict], links: list[dict]) -> list[str]:
    """LLM에게 다시 묻지 않고 links 결과에서 코드로 파생시킨다(같은
    사실을 두 번 답하게 하면 서로 어긋날 위험만 생긴다는 원칙 - 구버전
    link_engine.py의 derive_unused_resume_items()와 동일)."""
    used = {m["resume_object_id"] for link in links for m in link.get("matched_resume_objects") or []}
    all_ids = {_obj_id(o) for o in resume_objects if o.get("layer") in _MATCHABLE_LAYERS}
    return sorted(all_ids - used)


def run_semantic_linking(
    resume_objects: list[dict],
    jd_objects: list[dict],
    job: dict,
    user_career_level: str | None = None,
    challenge_option: bool = False,
    provider: str = DEFAULT_PROVIDER,
) -> tuple[dict, list[str]]:
    """Resume 1건 + JD 1건 -> LLM 1회 호출 -> Semantic Link Output Schema
    v1. 반환값은 (link_result, issues) - issues는 스키마 밖 진단 정보
    (Step 5 검증/로그 전용, 화면에 노출하지 않는다).

    2026-08-03 시점: 이 함수는 아직 실행(첫 LLM 호출)하지 않았다 - 설계
    문서(docs/design/semantic_linking_schema_v1.md) 확정 직후 구현만
    완료한 상태다. 실행은 Step 5(5건 스키마 검증)에서 사용자 확인 후
    시작한다."""
    qualification_validation = compute_qualification_validation(
        resume_objects, job, user_career_level, challenge_option,
    )
    prompt = build_link_prompt(jd_objects, resume_objects, qualification_validation)

    raw = call_llm(prompt, provider=provider, max_tokens=8192)
    try:
        parsed = parse_link_output(raw)
    except json.JSONDecodeError:
        raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=8192)
        parsed = parse_link_output(raw)

    links, issues = _validate_and_clean_links(parsed, jd_objects, resume_objects, qualification_validation)
    result = {
        "link_version": LINK_SCHEMA_VERSION,
        "links": links,
        "unmatched_resume_objects": _compute_unmatched_resume_objects(resume_objects, links),
    }
    return result, issues


# ── Link Debug View(Step 5 검증 전용, 저장 스키마 변경 없음) ──────────
def format_link_debug_rows(link_result: dict, jd_objects: list[dict]) -> list[dict]:
    """Semantic Link 결과를 사람이 검토하기 쉬운 표(행 리스트)로 변환한다.
    JD Requirement/Relation/Match Strength/Matched Evidence/Missing/
    Improvement를 한 행에 모은다 - Step 5(5건 검증)에서 이 표를 보고
    "근거를 지어내지 않았는가/rewrite와 genuine_gap을 올바르게
    구분하는가"를 확인한다. 순수 표시용 파생 함수 - 저장 스키마는 그대로."""
    jd_req_by_id = {_obj_id(o): o.get("normalized_text", "") for o in jd_objects}
    rows = []
    for link in link_result.get("links") or []:
        evidence_text = " / ".join(
            f"[{m['resume_object_id']}] {m['evidence_text']}"
            for m in link.get("matched_resume_objects") or []
        ) or "(없음)"
        missing_text = "; ".join(
            f"{m['type']}:{m['text']}" for m in link.get("missing") or []
        ) or "(없음)"
        improvement = link.get("improvement")
        if improvement:
            improvement_text = f"[{improvement['type']}/{improvement.get('action') or '-'}] target={improvement.get('target_resume_object_id') or '-'}"
        elif link.get("relation") == "match":
            improvement_text = "(없음 - match, 개선 불필요)"
        else:
            improvement_text = "(없음 - 응답 누락/미판단)"
        rows.append({
            "layer": link.get("layer"),
            "jd_object_id": link.get("jd_object_id"),
            "jd_requirement": link.get("jd_requirement") or jd_req_by_id.get(link.get("jd_object_id"), ""),
            "relation": link.get("relation"),
            "match_strength": link.get("match_strength"),
            "relation_reason_type": link.get("relation_reason_type"),
            "matched_dimensions": ", ".join(link.get("matched_dimensions") or []) or "(없음)",
            "matched_evidence": evidence_text,
            "missing": missing_text,
            "improvement": improvement_text,
        })
    return rows


# ── 기존 Rule 엔진(semantic_matching.py) 처리 방침 ──────────────────
# 2026-08-03(현재 상태) - pipeline.py의 ensure_job_semantic_match()는
# 여전히 semantic_matching.match_resume_to_jd()(cosine+gate)를 그대로
# 호출한다. 이 파일은 아직 어디에도 연결되지 않은 독립 모듈이다. 교체
# 시점은 Step 5(5건 스키마 검증) 통과 이후, Step 6(Judge를 Rule 기반으로
# 재작성해서 이 Link 결과만 소비하게 만드는 단계)에서 결정한다 - 지금은
# 검증 전이라 기존 Rule 엔진을 빼지 않는다.
