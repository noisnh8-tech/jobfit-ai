"""
resume_input/judge_engine.py

Judge Rule Engine(2026-08-14, v5 재설계) - Semantic Link 결과
(semantic-link-v1)만 입력받아 지원/보류/비추천을 판단한다. LLM 호출 없음.
숫자 점수를 전혀 쓰지 않는다.

v4에서 v5로 바꾼 이유(사용자 확정, 2026-08-14): v4는 qualification
레이어에서 "경력 연차/학위/자격증/어학" 같은 Hard Eligibility 항목을
가려내는 데까진 맞았지만, 그 항목의 충족 여부 판단을 여전히 Semantic
Linking(Link의 relation/relation_reason_type)에 맡기고 있었다. 실측
(특정 헬스케어 공고 케이스)에서 "석사 이상 학위"가 이력서의 "부트캠프 수료"와
"교육을 받았다"는 의미적 유사성만으로 partial_match가 되는 걸 확인했다
- 원본 이력서엔 실제로 "전문학사"가 있는데(Resume Understanding
semantic_objects에서는 통째로 누락돼 있었음), Link는 학위 수준을 비교한
게 아니라 "교육"이라는 개념적 유사성만 봤다. Hard Eligibility는 "의미가
비슷한가"가 아니라 "조건을 충족하는가"를 봐야 하는 별개의 판단이라
Semantic Linking 자체가 맞지 않는 도구였다(사용자 확정).

그래서 v5는 Hard Eligibility로 분류된 항목에 한해 Link의 relation을
아예 참고하지 않고, `resume_facts.py`(학력/자격증/경력 factual value,
LLM 없음)와 `eligibility_compare.py`(deterministic 조건 비교)로 직접
판단한다. 일반 qualification(competency성, 80%)과 problem/task/skill/
thinking/culture는 기존 Semantic Linking 그대로 - 이 부분은 건드리지
않았다.

**3-state 규칙(핵심)**: 충족/미충족/확인불가를 끝까지 구분한다.
    미충족   -> 비추천(가장 강한 판단, 확정된 것만)
    확인불가 -> 보류("필수 조건 확인 필요") - "지원"으로 자동 확정되지
        않는다. 이력서에 정보가 없다는 사실을 실제 미충족이라는 사실로
        바꾸지 않는다(absence of evidence ≠ evidence of absence 원칙 -
        D/E 구분과 동일한 원칙을 Hard Eligibility에도 그대로 적용한 것).
    충족     -> Gate 통과, 기존 구조적 gap 판단으로 진행.
"A 또는 B" 복합 조건은 한쪽만 검사하므로(완전한 parser는 범위 밖),
그 결과가 "미충족"이면 확정하지 않고 "확인불가"로 낮춘다(false rejection
방지 - eligibility_compare.py 참고).

**점수는 이 파일의 책임이 아니다.** 점수(z-score 결합, importance 가중치
등)는 Candidate Generation(목록, `candidate_search.py`/`meaning_matching.py`/
`constraint_layer.py`)의 랭킹 목적으로만 존재한다 - "1000건 -> Top30 -> Top5"
처럼 좁혀나가는 단계다. "분석하기"를 누른 이후(이 파일이 담당하는 영역)는
Rank가 아니라 개별 JD Requirement 하나하나에 대한 근거 있는 판단이 목적이라,
그걸 다시 숫자 하나로 뭉개면 이미 Link가 만들어둔 relation/reason_type/
matched_dimensions 같은 풍부한 정보가 오히려 줄어든다(사용자 확정). Hard
Eligibility 판별도 "몇 개 이상이면 보류" 같은 임계값을 새로 만들지 않았다.

v5 구조:
1. **Hard Eligibility 판별**: qualification 레이어 항목 중 JD 원문에
   경력 연차/학위/자격증/어학 점수/출장·운전 등 객관적 조건 신호가 있는
   것만 후보로 삼는다(정규식, 새 LLM 분류 아님 - 117개 전수조사로 검증됨).
2. **Hard Eligibility 비교**: eligibility_compare.compare()로 각 후보를
   충족/미충족/확인불가로 판정한다. 미충족이 1건이라도 있으면 "비추천".
   아니고 확인불가가 1건이라도 있으면 "보류"(필수 조건 확인 필요).
3. **구조적 gap 판단**: 위에서 전부 통과(충족)했거나 Hard Eligibility
   후보가 아예 없으면, problem/task/skill/thinking/culture + 일반
   qualification(competency성)에 대해 기존과 동일하게 "구조적으로 못
   고치는" 이유(domain_gap/skill_gap/experience_gap/responsibility_gap)의
   no_match가 있으면 "보류", 없으면 "지원".
4. `decision_reason`은 relation/relation_reason_type을 문장 틀에 채운
   것뿐이다(숫자 아님). `required_actions`는 improvement.type="rewrite"인
   항목만 모은 것 - "이력서를 이렇게 고치면 된다"는 실행 가능한 항목만
   보여주고, genuine_gap(고칠 수 없는 것)은 `structural_gaps`로 따로 둔다.

기존 judge_service.py(LLM 1회)와의 관계, checklist_semantic 범위 밖인 점은
이전 버전과 동일하다.
"""
from __future__ import annotations

import re

from resume_input import eligibility_compare

_STRUCTURAL_REASON_TYPES = (
    "domain_gap", "skill_gap", "experience_gap", "responsibility_gap",
    # 2026-08-31 - quick_analysis(경량 분석)가 competency성 자격요건 미충족을
    # qualification_gap 으로 보낸다(Hard Eligibility 정규식에 안 걸리는 것).
    "qualification_gap",
)

_RELATION_LABEL = {"match": "충족", "partial_match": "일부 충족", "no_match": "미충족"}
_REASON_TYPE_LABEL = {
    "scope_gap": "다루는 범위만 다름", "responsibility_gap": "책임/권한 범위가 다름",
    "experience_gap": "실무 경험 부족", "domain_gap": "업무 영역이 다름",
    "skill_gap": "요구 기술 자체가 없음", "evidence_gap": "관련 근거가 이력서에 불충분",
}

# 2026-08-14(Hard Eligibility 판별, 사용자 확정) - JD 원문(evidence.source_
# text)에 이미 있는 문구만 본다. 117개 qualification 항목 전수 조사에서
# 24개(20%)를 가려냈고 전량 수작업 검증했다(오탐 0건) - 새 LLM 분류/새
# 필드 없이 정규식만으로 충분히 안정적이었다.
_HARD_ELIGIBILITY_PATTERN = re.compile(
    r"\d+\s*(년|개월|년차)|학위|석사|박사|학사|자격증|면허|점\s*이상"
    r"|TOEIC|OPIc|JLPT|JPT|TOEFL|IELTS|만\s*\d+세"
    r"|출장|운전|결격사유|병역|국적|(?<!소)비자|근무지|교대\s*근무|신원조회|채용\s*결격|현장\s*근무"
)


def _is_hard_eligibility_text(*texts: str) -> bool:
    return bool(_HARD_ELIGIBILITY_PATTERN.search(" ".join(t or "" for t in texts)))


def _classify_hard_eligibility(
    links: list[dict], jd_text_by_id: dict[str, str], resume_facts: dict | None,
) -> tuple[list[dict], list[dict], set[str]]:
    """qualification 레이어의 Hard Eligibility 후보를 전부 골라
    eligibility_compare.compare()로 판정한다. resume_facts가 없으면
    (호출부가 안 넘겼으면) 전부 "확인불가"로 취급한다 - 정보가 없다고
    미충족으로 확정하지 않는다는 원칙을 여기서도 지킨다.
    반환: (미충족 목록, 확인불가 목록, Hard Eligibility로 판정된 jd_object_id 집합)."""
    resume_facts = resume_facts or {}
    blocking, unknown, hard_elig_ids = [], [], set()
    for l in links:
        if l.get("layer") != "qualification":
            continue
        combined = f"{l.get('jd_requirement', '')} {jd_text_by_id.get(l.get('jd_object_id'), '')}"
        if not _is_hard_eligibility_text(combined):
            continue
        hard_elig_ids.add(l.get("jd_object_id"))
        result = eligibility_compare.compare(combined, resume_facts)
        if result["status"] == "미충족":
            blocking.append({**l, "_elig_reason": result["reason"]})
        elif result["status"] == "확인불가":
            unknown.append({**l, "_elig_reason": result["reason"]})
        # "충족"은 Gate를 통과한 것 - blocking/unknown 어디에도 안 들어가고,
        # 아래에서 구조적 gap 판단 대상(qualification 레이어)에서도 제외된다.
    return blocking, unknown, hard_elig_ids


def _structural_gaps(links: list[dict]) -> list[dict]:
    """problem/task/skill/thinking/culture/qualification 전 레이어에서,
    이력서 표현을 고쳐서 해결되지 않는 종류의 no_match만 모은다
    (evidence_gap은 제외 - 그건 "근거 문장만 보강하면 되는" 경우라
    구조적 결함이 아니다). qualification도 포함한다 - Hard Eligibility로
    판정돼 이미 factual하게 처리된 항목은 judge()가 호출 전에
    `remaining`에서 걸러서 넘기므로, 여기 들어오는 qualification은 전부
    competency성(비Hard Eligibility) 항목뿐이다.

    버그 수정 이력(2026-08-14) - v4에서 이 함수가 "qualification의
    competency성 항목도 여기 합류한다"고 문서에 적어놓고 실제 튜플에는
    "qualification"을 빠뜨려서, 일반 역량 qualification 항목이 구조적
    gap 판단에 전혀 반영되지 않던 게 최종 통합 중 발견됐다."""
    return [
        l for l in links
        if l.get("layer") in ("problem", "task", "skill", "thinking", "culture", "qualification")
        and l.get("relation") == "no_match"
        and l.get("relation_reason_type") in _STRUCTURAL_REASON_TYPES
    ]


def _decision_reason_line(link: dict) -> str:
    relation = link.get("relation")
    label = _RELATION_LABEL.get(relation, relation)
    if relation == "match":
        return f"[{link.get('layer')}] {link.get('jd_requirement')}: {label}."
    reason_type_label = _REASON_TYPE_LABEL.get(link.get("relation_reason_type"), "")
    return f"[{link.get('layer')}] {link.get('jd_requirement')}: {label}({reason_type_label})."


def judge(
    link_result: dict,
    jd_semantic_objects: list[dict] | None = None,
    resume_facts: dict | None = None,
) -> dict:
    """Semantic Link 결과(semantic-link-v1) 하나를 규칙으로 판단한다.
    숫자 점수 없음 - relation/relation_reason_type 값만으로 결정한다.
    반환의 "decision" 값(지원/보류/비추천)은 기존 judge_service.judge()와
    호환되며 app.py의 DECISION_TONE이 그대로 이 값을 쓴다.

    jd_semantic_objects(선택, job["jd_semantic_objects"]를 이미 파싱한
    리스트)는 Hard Eligibility 판별에만 쓴다 - link["jd_requirement"]는
    정규화된 짧은 문장이라 "5년 이상" 같은 문턱값이 원문(evidence.
    source_text)에만 있고 정규화 문장엔 없는 경우가 있었다(2026-08-14
    실측).

    resume_facts(선택, resume_facts.extract_resume_facts()의 반환값)는
    Hard Eligibility 항목의 충족 여부를 Semantic Linking 대신 factual
    value로 비교하는 데 쓴다(v5 - 모듈 docstring 참고). 안 넘기면 Hard
    Eligibility 후보 전부가 "확인불가"로 처리된다(정보 없음을 미충족으로
    확정하지 않는다는 원칙 - 새 LLM 호출은 없다).

    2026-08-16(사용자 확정 - "Semantic Linking(관련성)과 Hard Eligibility
    (사실 조건 충족)는 서로 다른 질문") - 반환값에 `hard_eligibility`
    ({jd_object_id: "미충족"|"확인불가"|"충족"})를 추가한다. _classify_
    hard_eligibility()가 이미 계산해 갖고 있던 blocking/unknown/
    hard_elig_ids를 버리지 않고 조회용 dict 하나로 모아 넘기는 것뿐이다
    - 판별 규칙(_classify_hard_eligibility/eligibility_compare) 자체는
    전혀 건드리지 않는다. analysis_engine.build_analysis()가 이 값으로
    "이 qualification 항목은 Semantic Linking의 relation과 무관하게
    사실 조건을 충족했는지"를 확인해서 최종 state를 정할 때 쓴다(그
    반대로 Linking의 relation 자체를 여기서 바꾸지 않는다)."""
    links = link_result.get("links") or []
    jd_text_by_id = {
        (o.get("id") or o.get("normalized_text")): (o.get("evidence") or {}).get("source_text", "")
        for o in (jd_semantic_objects or [])
    }

    blocking, unknown, hard_elig_ids = _classify_hard_eligibility(links, jd_text_by_id, resume_facts)
    blocking_ids = {b.get("jd_object_id") for b in blocking}
    unknown_ids = {u.get("jd_object_id") for u in unknown}
    hard_eligibility = {
        oid: ("미충족" if oid in blocking_ids else "확인불가" if oid in unknown_ids else "충족")
        for oid in hard_elig_ids
    }

    if blocking:
        return {
            "decision": "비추천",
            "decision_reason": [
                f"지원 자격 조건 미충족: {b['jd_requirement']} ({b['_elig_reason']})." for b in blocking
            ],
            "blocking_requirements": blocking,
            "required_actions": [],
            "structural_gaps": [],
            "hard_eligibility": hard_eligibility,
        }

    if unknown:
        # 2026-08-14(사용자 확정) - Hard Eligibility를 확인할 수 없는데
        # "지원"으로 자동 확정하지 않는다. 다른 구조적 gap이 하나도 없어도
        # "보류"로 남긴다 - 객관적 필수조건 충족 여부가 아직 안 확인됐다는
        # 뜻이라, 결과가 없다고 결과가 좋다고 넘겨짚지 않는다.
        return {
            "decision": "보류",
            "decision_reason": [
                f"필수 조건 확인 필요: {u['jd_requirement']} ({u['_elig_reason']})." for u in unknown
            ],
            "blocking_requirements": [],
            "required_actions": [],
            "structural_gaps": [],
            "hard_eligibility": hard_eligibility,
        }

    # Hard Eligibility 후보였던 qualification 항목(전부 충족 판정)은
    # 구조적 gap 판단에서 제외한다 - 이미 factual하게 통과했으므로
    # Link의 relation으로 다시 걸러지지 않는다.
    remaining = [l for l in links if l.get("jd_object_id") not in hard_elig_ids]
    structural = _structural_gaps(remaining)
    decision = "보류" if structural else "지원"

    required_actions = [
        {
            "jd_object_id": link.get("jd_object_id"), "layer": link.get("layer"),
            "action": link["improvement"]["action"],
            "target_resume_object_id": link["improvement"]["target_resume_object_id"],
        }
        for link in links
        if link.get("improvement") and link["improvement"].get("type") == "rewrite"
    ]
    structural_gap_notes = [
        {"jd_object_id": l.get("jd_object_id"), "layer": l.get("layer"), "jd_requirement": l.get("jd_requirement"), "relation_reason_type": l.get("relation_reason_type")}
        for l in structural
    ]

    decision_reason = [_decision_reason_line(l) for l in links if l.get("relation") != "match"]
    if not decision_reason:
        decision_reason = ["Problem/Thinking/Task/Skill/Qualification/Culture 전 항목이 match 또는 개선 가능한 partial_match."]

    return {
        "decision": decision,
        "decision_reason": decision_reason,
        "blocking_requirements": [],
        "required_actions": required_actions,
        "structural_gaps": structural_gap_notes,
        "hard_eligibility": hard_eligibility,
    }
