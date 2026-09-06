# -*- coding: utf-8 -*-
"""
resume_input/selection_engine.py

Customization - Selection(2026-08-14, docs/verification/2026-08-13_
list_benchmark_30/selection_engine_draft.py에서 프로덕션으로 승격 -
로직 변경 없음, Microsoft 예시로 이미 검증됨). 결정론적 로직, 새 LLM/새
판단 없음. 이미 계산된 Semantic Linking 결과(links)와 Resume
semantic_objects, 그리고 기존 customizer.py의 project/subsection 파싱만
소비한다.

원칙(2026-08-14, 사용자 확정):
- 새로운 의미 판단/유사도 계산/LLM 호출/임의 가중합 점수 없음.
- Project 우선순위는 raw link count가 아니라 distinct JD requirement
  coverage로 판단.
- importance -> relation -> match_strength 순 lexicographic 서열만
  쓴다(가중합 아님). tier 경계(최우선/보조/낮은우선순위)도 새 임계값을
  만들지 않는다 - "critical/core 요구를 하나라도 커버하는가"라는 기존
  의미 그대로 유지하고, 같은 tier 안의 순서는 rank_within_tier(서수,
  점수 아님)로만 노출한다. 화면에서 이 tier를 그대로 보여줄 필요는 없고
  Composition이 배열 순서+coverage로 "핵심 경험/보조 경험" 같은 자연어
  역할로 번역한다.
- unlinked = 낮은 우선순위 후보일 뿐, 삭제/무관 판정 아님.
- matched_resume_objects=[]인 no_match 요구(Unverified)는 protected로만
  전달 - 여기서 새 경험을 생성하지 않는다.
- 성과(4) 성과)는 customizer.py 파싱을 재사용해 후보로만 복원한다
  (최종 선택은 이 단계 책임 아님 - Composition의 몫).
- Skill과 Qualification은 동일 원칙(순서만 조정, 삭제 없음) - project
  목록과 별도로 supporting_skills/supporting_qualifications로 뺀다.
"""
from __future__ import annotations

from resume_input.customizer import _split_projects, split_project_subsections, split_bullets

_IMPORTANCE_RANK = {"critical": 0, "core": 1, "major": 2, "minor": 3}
_STRENGTH_RANK = {"strong": 0, "moderate": 1, "weak": 2, "none": 3}


def _importance_rank(v):
    return _IMPORTANCE_RANK.get(v, 9)


def _strength_rank(v):
    return _STRENGTH_RANK.get(v, 9)


def build_selection(resume_raw: str, resume_semantic_objects: list[dict], links: list[dict]) -> dict:
    obj_by_id = {o["id"]: o for o in resume_semantic_objects}

    # ── 1) 각 link을 project(=matched resume object의 evidence.project)별로 귀속 ──
    project_coverage: dict[str, list[dict]] = {}
    protected: list[dict] = []
    linked_object_ids: set[str] = set()  # skill/qualification 순서 조정용(레이어 무관 전체)

    for link in links:
        relation = link.get("relation")
        matched = link.get("matched_resume_objects") or []
        if relation == "no_match" and not matched:
            protected.append({
                "jd_object_id": link["jd_object_id"],
                "jd_requirement": link.get("jd_requirement"),
                "layer": link.get("layer"),
                "importance": link.get("importance"),
                "relation_reason_type": link.get("relation_reason_type"),
                "reason": "Resume에서 관련 근거를 확인할 수 없음(Unverified) - 사용자 확인 전까지 생성 금지",
            })
            continue
        if not matched:
            continue

        connection_type = "직접 대응" if relation == "match" else "전이 가능한 경험"
        resume_ids = [m["resume_object_id"] for m in matched]
        linked_object_ids.update(resume_ids)

        projects_touched = set()
        for rid in resume_ids:
            robj = obj_by_id.get(rid)
            if not robj:
                continue
            proj = (robj.get("evidence") or {}).get("project") or "(project 없음)"
            projects_touched.add(proj)
        for proj in projects_touched:
            project_coverage.setdefault(proj, []).append({
                "jd_object_id": link["jd_object_id"],
                "jd_requirement": link.get("jd_requirement"),
                "importance": link.get("importance"),
                "relation": relation,
                "match_strength": link.get("match_strength"),
                "connection_type": connection_type,
                "resume_object_ids": [rid for rid in resume_ids if obj_by_id.get(rid, {}).get("evidence", {}).get("project") == proj],
            })

    # ── 2) 전체 Resume project 목록(evidence.project 기준) ──
    all_projects: dict[str, list[dict]] = {}
    for o in resume_semantic_objects:
        proj = (o.get("evidence") or {}).get("project") or "(project 없음)"
        all_projects.setdefault(proj, []).append(o)

    # 순수 skill/qualification 컨테이너(problem/thinking/task가 하나도 없는 project)는
    # "경험 project"가 아니므로 아래 5)의 projects 목록에서 제외하고, 4)에서
    # supporting_skills/supporting_qualifications로 별도 처리한다.
    def _is_experience_project(proj: str) -> bool:
        layers = {o.get("layer") for o in all_projects[proj]}
        return bool(layers & {"problem", "thinking", "task"})

    # ── 3) project별 우선순위 산정(lexicographic, 가중합 없음) - 경험 project만 ──
    experience_projects = [p for p in all_projects if _is_experience_project(p)]

    def priority_key(proj: str):
        entries = project_coverage.get(proj, [])
        critical_covered = len({e["jd_object_id"] for e in entries if _importance_rank(e["importance"]) == 0})
        core_covered = len({e["jd_object_id"] for e in entries if _importance_rank(e["importance"]) == 1})
        best_strength = min((_strength_rank(e["match_strength"]) for e in entries), default=9)
        return (-critical_covered, -core_covered, best_strength)

    def tier_of(proj: str) -> str:
        entries = project_coverage.get(proj, [])
        if not entries:
            return "낮은 우선순위"
        has_critical_or_core = any(_importance_rank(e["importance"]) <= 1 for e in entries)
        return "최우선" if has_critical_or_core else "보조"

    ordered_projects = sorted(experience_projects, key=priority_key)

    # tier 안에서의 순번(rank_within_tier) - 새 점수 아니라 이미 정해진 배열 순서의 서수 표기
    tier_counters: dict[str, int] = {}
    rank_within_tier: dict[str, int] = {}
    for proj in ordered_projects:
        t = tier_of(proj)
        tier_counters[t] = tier_counters.get(t, 0) + 1
        rank_within_tier[proj] = tier_counters[t]

    # ── 4) skill/qualification은 project 목록과 별도, 순서만 조정(삭제 없음) ──
    def _supporting_list(layer_name: str) -> list[dict]:
        objs = [o for o in resume_semantic_objects if o.get("layer") == layer_name]
        linked = [o for o in objs if o["id"] in linked_object_ids]
        unlinked = [o for o in objs if o["id"] not in linked_object_ids]
        return (
            [{"text": o["normalized_text"], "linked": True} for o in linked] +
            [{"text": o["normalized_text"], "linked": False} for o in unlinked]
        )

    supporting_skills = _supporting_list("skill")
    supporting_qualifications = _supporting_list("qualification")

    # ── 5) 경험 project별 결과 조립 ──
    _, blocks, _ = _split_projects(resume_raw)
    block_by_title = {title: block for title, block in blocks}

    result_projects = []
    for proj in ordered_projects:
        entries = project_coverage.get(proj, [])
        covered_reqs = sorted({(e["jd_object_id"], e["jd_requirement"], e["importance"], e["connection_type"]) for e in entries},
                               key=lambda x: _importance_rank(x[2]))
        selected_ids = sorted({rid for e in entries for rid in e["resume_object_ids"]
                                if obj_by_id.get(rid, {}).get("layer") in ("problem", "thinking", "task")})
        all_ids_in_project = [o["id"] for o in all_projects[proj] if o.get("layer") in ("problem", "thinking", "task")]
        deemphasize = [oid for oid in all_ids_in_project if oid not in selected_ids]

        result_candidates = []
        block = block_by_title.get(proj)
        if block:
            for sec_title, sec_body in split_project_subsections(block):
                if "성과" in sec_title:
                    result_candidates = split_bullets(sec_body) or [sec_body.strip()]

        why_parts = [f"{req}({imp}, {ct})" for _, req, imp, ct in covered_reqs]
        result_projects.append({
            "project": proj,
            "priority": tier_of(proj),
            "rank_within_tier": rank_within_tier[proj],
            "connection_type": sorted({e["connection_type"] for e in entries}) if entries else [],
            "why_selected": " / ".join(why_parts) if why_parts else "이번 공고 요구사항과 직접 연결되지 않음",
            "covered_jd_requirements": [{"jd_object_id": r[0], "jd_requirement": r[1], "importance": r[2]} for r in covered_reqs],
            "selected_resume_objects": [obj_by_id[i]["normalized_text"] for i in selected_ids],
            "result_candidates": result_candidates,
            "deemphasize_candidates": [obj_by_id[i]["normalized_text"] for i in deemphasize],
        })

    return {
        "projects": result_projects,
        "supporting_skills": supporting_skills,
        "supporting_qualifications": supporting_qualifications,
        "protected": protected,
    }
