# -*- coding: utf-8 -*-
"""
resume_input/analysis_engine.py

Analysis Screen(2026-08-14, 사용자 확정) - Semantic Linking 결과(semantic-
link-v1)와 judge_engine.judge()의 출력을 그대로 소비해서, "지원할 가치가
있는가"를 사용자가 확인할 수 있는 화면 단위로 정리한다. LLM 호출 없음,
새 의미 판단/점수/threshold 없음 - link_engine.py가 이미 계산한 relation/
relation_reason_type/matched_resume_objects만으로 아래 5-state Evidence
Contract에 분류한다.

Evidence Contract(공고 전체에서 합의된 5-state, Analysis/Customization/
Cover Letter Source가 전부 이 상태를 그대로 재사용한다 - 화면마다 "증거가
있는가"를 다시 판단하지 않는다):
  A. Supported      - relation=match. 직접 근거 있음.
  B. Transferable    - relation=partial_match. 전이 가능한 근거 있음(범위/
     책임 등 일부만 다름).
  C. Evidence Gap    - relation=no_match, relation_reason_type=evidence_gap.
     관련 Resume Object는 있지만(matched_resume_objects 존재) 그 근거만으로는
     JD가 기대하는 수준을 증명하기엔 부족.
  D. Unverified       - relation=no_match, matched_resume_objects=[]. Resume에서
     관련 근거를 하나도 찾지 못했다는 뜻일 뿐, "경험이 없다"고 확정한 것이
     아니다. 사용자가 직접 "경험 없음"을 확인하기 전까지는 절대 E로 승격하지
     않는다(2026-08-13, 사용자 확정 원칙 - genuine_gap 오분류 방지).
  E. Genuine Gap      - 사용자가 명시적으로 "이 항목은 실제로 경험이 없다"를
     확인한 뒤에만 존재하는 상태. 이 파일은 이 상태를 자동으로 만들지
     않는다 - confirm_genuine_gap()을 통해 호출자(UI)가 명시적으로 승격시킬
     때만 생긴다. 이 모듈 혼자서는 D만 만들 수 있고 E는 절대 만들 수 없다.

이 5-state는 judge_engine.py의 decision(지원/보류/비추천) 로직을 그대로
따른다 - 새 규칙을 얹지 않는다. judge_engine.judge()가 이미 결정한 것을
사용자가 이해할 수 있는 근거 단위로 "설명"만 하는 것이 이 파일의 책임이다.
"""
from __future__ import annotations

_STATE_LABEL = {
    "A": "충족 (Supported)",
    "B": "전이 가능 (Transferable)",
    "C": "근거 부족 (Evidence Gap)",
    "D": "미확인 (Unverified)",
    "E": "실제 경험 없음 (Genuine Gap, 사용자 확인됨)",
}

_REASON_TYPE_LABEL = {
    "scope_gap": "다루는 범위만 다름",
    "responsibility_gap": "책임/권한 범위가 다름",
    "experience_gap": "실무 경험 부족(이력서에서 확인 안 됨)",
    "domain_gap": "업무 영역이 다름",
    "skill_gap": "요구 기술 자체가 없음",
    "evidence_gap": "관련 근거는 있으나 수준 증명 부족",
}

_LAYER_LABEL = {
    "problem": "Problem", "thinking": "Thinking", "task": "Task",
    "skill": "Skill", "qualification": "Qualification", "culture": "Culture",
}


def _classify_state(link: dict) -> str:
    """relation/relation_reason_type/matched_resume_objects만으로 A~D를
    결정한다(E는 여기서 절대 만들지 않음 - 위 docstring 참고)."""
    relation = link.get("relation")
    if relation == "match":
        return "A"
    if relation == "partial_match":
        return "B"
    # relation == "no_match"
    matched = link.get("matched_resume_objects") or []
    if matched and link.get("relation_reason_type") == "evidence_gap":
        return "C"
    return "D"


def _why(link: dict, state: str) -> str:
    layer = _LAYER_LABEL.get(link.get("layer"), link.get("layer"))
    if state == "A":
        dims = ", ".join(link.get("matched_dimensions") or []) or "핵심 축"
        return f"[{layer}] Resume 근거가 JD 요구와 직접 대응합니다({dims} 일치)."
    if state == "B":
        dims = ", ".join(link.get("matched_dimensions") or []) or "일부"
        return f"[{layer}] {dims} 축은 일치하지만, 범위/방식 일부가 다릅니다."
    reason_label = _REASON_TYPE_LABEL.get(link.get("relation_reason_type"), "")
    if state == "C":
        return f"[{layer}] 관련 경험은 있으나, 근거만으로는 요구 수준을 증명하기 부족합니다({reason_label})."
    if state == "D":
        return f"[{layer}] 현재 이력서에서 관련 근거를 찾지 못했습니다({reason_label}). 실제 경험이 없다는 뜻은 아닙니다 - 확인이 필요합니다."
    return f"[{layer}] 사용자가 실제 경험 없음을 확인했습니다."


def build_analysis(link_result: dict, judge_result: dict) -> dict:
    """link_engine.run_semantic_linking()의 link_result와 judge_engine.
    judge()의 judge_result를 그대로 받아 Evidence Contract 5-state로
    재분류한 Analysis 화면 데이터를 만든다. 새로 계산하는 값 없음 - 이미
    있는 필드를 상태값(state)이라는 이름으로 재라벨링만 한다.

    2026-08-16(사용자 확정 - "관련성(relation)과 지원 자격 충족
    (eligibility)은 다른 질문") - 실측으로 확인된 문제: Hard Eligibility
    qualification 항목(예: "학사 이상")이 judge_engine에서는 사실
    기반으로 "미충족"(요구=학사, 보유=전문학사)이라고 정확히 판정됐는데,
    같은 항목의 Semantic Linking relation은 "관련 있어 보인다"는 이유로
    "match"가 나올 수 있고, 이 모듈이 relation만 보고 state를 매기면
    그 항목이 "A(잘 맞는 부분)"으로 노출되는 모순이 실제 5건 중 3건에서
    재현됐다.

    수정 원칙(중요, 반드시 지킬 것):
    - link["relation"]은 절대 바꾸지 않는다 - items[i]["relation"]에는
      원본 relation을 그대로 넣는다(위 사례라면 여전히 "match"로 보임 -
      Semantic Linking이 "이 항목과 관련된 근거가 있다"고 본 것 자체는
      맞을 수 있다). 바뀌는 건 오직 "state"(=최종 화면 분류)뿐이다.
    - eligibility="미충족"/"확인불가"인 항목만 state를 "D"로 강제한다
      (D=지원 전 확인 필요 - Linking이 뭐라 했든 "잘 맞는 부분"에 들어갈
      수 없다).
    - eligibility="충족"인 항목은 **건드리지 않는다**(state를 A로
      끌어올리지 않는다) - "자격 조건 통과"가 "직무 적합성의 강점"을
      뜻하지는 않기 때문이다(2026-08-16 사용자 지적). Eligibility의
      역할은 오직 "잘못된 긍정(false positive) relation을 제어"하는
      것으로 한정한다.
    - Hard Eligibility 대상이 아닌 항목(judge_result["hard_eligibility"]에
      키가 없는 항목)은 이 로직을 전혀 타지 않는다 - 기존 relation→state
      변환 그대로 유지된다(회귀 없음).
    - 디버깅 추적을 위해 각 item에 hard_eligibility_status를 그대로
      남긴다("Semantic Link: match / Eligibility: 미충족 / Final
      State: D"를 그대로 재구성할 수 있게)."""
    links = link_result.get("links") or []
    hard_eligibility = judge_result.get("hard_eligibility") or {}
    items = []
    for link in links:
        state = _classify_state(link)
        elig_status = hard_eligibility.get(link.get("jd_object_id"))
        if elig_status in ("미충족", "확인불가"):
            state = "D"
        items.append({
            "jd_object_id": link.get("jd_object_id"),
            "layer": link.get("layer"),
            "jd_requirement": link.get("jd_requirement"),
            "importance": link.get("importance"),
            "state": state,
            "state_label": _STATE_LABEL[state],
            "relation": link.get("relation"),
            "relation_reason_type": link.get("relation_reason_type"),
            "matched_resume_objects": link.get("matched_resume_objects") or [],
            "missing": link.get("missing") or [],
            "improvement": link.get("improvement"),
            "why": _why(link, state),
            "hard_eligibility_status": elig_status,
            # UI가 "경험 없음" 확인 버튼을 눌렀는지 여부. 이 모듈은 항상
            # False로 채운다 - True로 바꾸는 유일한 통로는 confirm_genuine_gap().
            "user_confirmed_no_experience": False,
        })

    by_state: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for it in items:
        by_state[it["state"]].append(it)

    return {
        "decision": judge_result.get("decision"),
        "decision_reason": judge_result.get("decision_reason"),
        "items": items,
        "by_state": by_state,
        "counts": {k: len(v) for k, v in by_state.items()},
        "required_actions": judge_result.get("required_actions") or [],
        "structural_gaps": judge_result.get("structural_gaps") or [],
        # 2026-08-14(S1D 판단결과-우선 재설계) - judge_engine.judge()가
        # 이미 계산한 "게이트를 막은 실제 항목"을 그대로 통과시킨다(새
        # 판단 없음, required_actions/structural_gaps와 같은 패턴).
        # UI가 "왜 비추천/보류인지"를 결과 맨 위에서 바로 보여주려면
        # 이 값이 필요한데, 지금까지는 build_analysis가 이 필드를
        # 버리고 있었다.
        "blocking_requirements": judge_result.get("blocking_requirements") or [],
    }


def confirm_genuine_gap(analysis: dict, jd_object_id: str) -> dict:
    """사용자가 "이 항목은 실제로 경험이 없다"를 명시적으로 확인했을 때만
    호출한다(UI 버튼 클릭 시점). 이 함수를 거치지 않고는 D가 E로 바뀌는
    경로가 이 코드베이스 어디에도 없다 - absence-of-evidence를 absence-of-
    experience로 자동 확정하지 않는다는 원칙을 코드로 강제한다.

    새 analysis dict를 반환한다(원본을 변형하지 않음)."""
    import copy
    out = copy.deepcopy(analysis)
    for it in out["items"]:
        if it["jd_object_id"] == jd_object_id and it["state"] == "D":
            it["state"] = "E"
            it["state_label"] = _STATE_LABEL["E"]
            it["user_confirmed_no_experience"] = True
            it["why"] = _why(it, "E")
    out["by_state"] = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for it in out["items"]:
        out["by_state"][it["state"]].append(it)
    out["counts"] = {k: len(v) for k, v in out["by_state"].items()}
    return out
