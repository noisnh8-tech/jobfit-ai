# -*- coding: utf-8 -*-
"""
resume_input/composition_engine.py

Customization - Composition(2026-08-14, docs/verification/2026-08-13_
list_benchmark_30/composition_engine_draft.py에서 프로덕션으로 승격,
동시에 최종 지시 Section 4-5에 맞춰 한 가지를 되돌렸다 - 아래 v1->v2
변경 참고). 결정론적 로직, 새 LLM/새 유사도/새 점수 없음. Selection 결과
+ 기존 Resume semantic_objects의 frame 필드만 재조립한다.

v1(draft) -> v2(2026-08-14, 사용자 지시 반영) 변경:
- v1은 result_candidates[0]을 "대표 성과(primary_result)"로, 나머지를
  supporting_results로 구분해서 반환했다. 이건 "이 성과가 covered
  requirement를 실제로 증명하는가"라는 질문에 위치 기반 추정으로 답한
  것이라, 최종 지시(Section 4-5)가 명시적으로 금지한 "새 판단"에 해당할
  위험이 있다 - 원 이력서 작성자가 정해둔 순서일 뿐 이 공고와의 관련성
  근거는 아니기 때문이다.
- v2는 대표 성과를 고르지 않는다. result_candidates를 순서 그대로 보존한
  "함께 보여줄 성과" 후보 리스트로만 넘긴다 - 화면에서 "이 중 자소서/
  강조에 쓸 성과를 고르세요"로 보여주고 최종 선택은 사용자(또는 Cover
  Letter Source 단계에서 이미 링크된 evidence)에게 맡긴다. 이 한계는
  README성 문서가 아니라 여기 코드 주석과 최종 보고서에 명시한다.

원칙(2026-08-14, 사용자 확정):
- "이 성과가 covered requirement를 증명하는가"를 새 점수/새 유사도로
  판정하지 않는다. 현재 데이터(성과 bullet과 특정 Problem/Task 객체를
  잇는 필드 없음)로는 결정론적으로 답할 수 없으므로, 순서를 강제로
  해석하지 않고 후보 그대로 넘긴다.
- 핵심 메시지는 새로 생성하지 않고, selected_resume_objects의 기존
  frame.action/object/outcome을 layer 순서(problem->thinking->task)로
  이어붙이기만 한다(재조립, 새 판단 아님).
- Skills/Qualifications 순서, Protected는 Selection 결과를 그대로 통과.
"""
from __future__ import annotations

_LAYER_ORDER = {"problem": 0, "thinking": 1, "task": 2}


def _core_message(project_entry: dict, resume_objects_by_text: dict) -> str:
    """selected_resume_objects(normalized_text 리스트)를 원본 layer 순서로
    복원해 frame.action+object -> outcome을 이어붙인다. 새 텍스트 생성
    없음 - 기존 frame 필드 재조립만."""
    objs = []
    for text in project_entry["selected_resume_objects"]:
        o = resume_objects_by_text.get(text)
        if o:
            objs.append(o)
    objs.sort(key=lambda o: _LAYER_ORDER.get(o.get("layer"), 9))

    parts = []
    for o in objs:
        frame = o.get("frame") or {}
        action = frame.get("action", "")
        obj = frame.get("object", "")
        parts.append(f"{obj} {action}".strip())
    if objs:
        last_outcome = (objs[-1].get("frame") or {}).get("outcome")
        if last_outcome:
            parts.append(last_outcome)
    return " → ".join(p for p in parts if p)


def build_composition(selection: dict, resume_semantic_objects: list[dict]) -> dict:
    resume_objects_by_text = {o["normalized_text"]: o for o in resume_semantic_objects}

    composed_projects = []
    for proj in selection["projects"]:
        core_message = _core_message(proj, resume_objects_by_text)

        role_label = "핵심 경험" if proj["rank_within_tier"] == 1 and proj["priority"] == "최우선" else (
            "보조 경험" if proj["priority"] in ("최우선", "보조") else "참고 경험"
        )

        composed_projects.append({
            "project": proj["project"],
            "order": proj["rank_within_tier"],
            "role_label": role_label,
            "core_message": core_message,
            "emphasize": proj["selected_resume_objects"],
            # v2(2026-08-14): 대표/보조 구분 없음 - 원문 순서 그대로인
            # "함께 보여줄 성과" 후보. 어떤 성과가 이 프로젝트의 covered_
            # jd_requirements를 실제로 증명하는지는 결정론적으로 판정할
            # 데이터가 없어 여기서 순서를 재해석하지 않는다(사용자 확인
            # 필요 - 최종 보고서 "미해결 한계" 참고).
            "result_candidates": proj["result_candidates"],
            "deemphasize": proj["deemphasize_candidates"],
        })

    composed_projects.sort(key=lambda p: p["order"])

    return {
        "projects": composed_projects,
        "skills_order": [s["text"] for s in selection["supporting_skills"]],
        "qualifications_order": [q["text"] for q in selection["supporting_qualifications"]],
        "do_not_generate": [
            {"requirement": p["jd_requirement"], "reason": "현재 이력서에서 확인되지 않음 - 사용자 확인 전 생성 금지"}
            for p in selection["protected"]
        ],
    }
