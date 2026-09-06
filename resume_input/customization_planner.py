"""
resume_input/customization_planner.py

Customization Planner(2026-08-16, 사용자 확정) - analysis_engine.build_analysis()
의 5-state(A~E) Evidence Contract 결과만 입력으로 쓴다. 새 LLM 호출 없음,
새 점수/가중치 없음. link_engine.py가 만드는 개별 `improvement` 제안
(resume_customizing.py v4가 쓰던 것)은 이 파일에서 쓰지 않는다 - Hard
Eligibility 반영, 5-state, JD layer 필드가 전부 analysis_engine 결과에만
있고 raw link_result에는 없기 때문이다(이번 세션 조사로 확인).

이 세션에서 실측 검증한 3가지 원칙을 그대로 코드화한다:

1. **레이어 라우팅** - 슬롯마다 관련 있는 JD layer만 본다.
   ② 프로젝트 순서: problem/thinking/task만(skill/culture/qualification 제외 -
   Class101 사례에서 culture 항목 하나가 집계에 섞이면서 "변경"이 잘못
   나온 걸 실측으로 확인, 라우팅 적용 후 "유지"로 정정됨).
   ③ 기술 순서: skill만.
   ④+⑤ bullet 강조: problem/thinking/task(주 기준) + culture(보조 기준,
   같은 프로젝트 안에서만).
   ① 자기소개: problem/thinking/task/culture(skill 제외 - SQL 같은 스킬
   레이어 항목 때문에 "변경 후보"가 잘못 뜨는 걸 실측으로 확인).
   qualification은 전 슬롯에서 제외 - Hard Eligibility/판단 전용 레이어.

2. **최소 변경(강/중/약 사전식 비교)** - 항상 "현재 1번째"와 "최선의
   대안"만 비교한다. 절대 위치 규칙(예: "3번째까지는 유지") 없음 - 사용자가
   "3/7이면 이미 잘 보임"류 절대 위치 판단을 명시적으로 반려했다. 대안이
   현재보다 명백히 우세할 때만(강한 근거 등급이 더 높거나, 같은 등급에서
   culture가 보조로 붙을 때만) 변경한다.

3. **culture는 보조 신호일 뿐, 별도로 bullet을 옮기지 않는다** - ⑤가
   "강조가 필요하다"고 판단해도 그 자체로 순서를 바꾸지 않고, ④의 직무
   비교가 동률일 때만 tie-break로 참여한다(사용자 확정 - ④/⑤가 서로 다른
   bullet을 동시에 1번으로 보내려 하는 충돌을 원천 차단).

이 파일이 만들지 않는 것:
- ⑥ 표현 치환: customization_rules.evaluate_terms()(Rule5, APPROVED_TERM_MAP
  기반)가 이미 별도 신호(jd_summary.resume_customization_targets)로 안전하게
  동작 중 - 이번 세션에서 노이즈가 확인된 건 이 파일이 아니라 investigation
  전용 스크립트(verify_all_slots.py의 decide_expression)였다. 중복 재구현
  하지 않는다.
- 자기소개 문장 자동 생성: v3->v4에서 이미 제거된 범위(resume_customizing.py
  참고) - 이 파일도 "무엇이 안 드러나는지"만 알려주고 문장은 짓지 않는다.
"""
from __future__ import annotations

_TIER_RANK = {"A": 3, "B": 2, "C": 1}  # 강/중/약 - 새 숫자 점수 아님, 기존 state 순서 그대로
_TIER_LABEL = {"A": "강", "B": "중", "C": "약"}
_IMPORTANCE_RANK = {"critical": 2, "core": 1, "normal": 0.5}  # culture 보조 기준 전용

_PROJECT_ORDER_LAYERS = {"problem", "thinking", "task"}
_BULLET_JOB_LAYERS = {"problem", "thinking", "task"}


# ── ② 프로젝트 순서 ───────────────────────────────────────────────────

def decide_project_order(analysis: dict, project_titles: list[str]) -> dict:
    """현재 1번째 프로젝트 vs 최선의 대안만 비교(사전식: 강 -> 중).
    culture/skill/qualification은 집계에서 제외한다(라우팅, 실측 검증됨)."""
    if len(project_titles) < 2:
        return {"decision": "유지", "reason": "프로젝트가 1개 이하", "project_order": None}

    counts = {p: {"강": 0, "중": 0, "약": 0} for p in project_titles}
    for it in analysis.get("items") or []:
        if it.get("layer") not in _PROJECT_ORDER_LAYERS:
            continue
        tier = _TIER_RANK.get(it.get("state"))
        if not tier:
            continue
        label = _TIER_LABEL[it["state"]]
        for m in it.get("matched_resume_objects") or []:
            proj = m.get("source_project")
            if proj in counts:
                counts[proj][label] += 1

    current_top = project_titles[0]
    cur_tuple = (counts[current_top]["강"], counts[current_top]["중"])
    best_other, best_tuple = None, None
    for p in project_titles[1:]:
        t = (counts[p]["강"], counts[p]["중"])
        if best_tuple is None or t > best_tuple:
            best_other, best_tuple = p, t

    if best_tuple is None or best_tuple <= cur_tuple:
        return {"decision": "유지", "reason": "현재 1번째 프로젝트보다 명백히 강한 대안 없음",
                "project_order": None, "counts": counts}

    new_order = [best_other] + [p for p in project_titles if p != best_other]
    return {
        "decision": "변경",
        "reason": f"{best_other}(강{counts[best_other]['강']}·중{counts[best_other]['중']})가 "
                  f"{current_top}(강{counts[current_top]['강']}·중{counts[current_top]['중']})보다 "
                  "직무 관련 근거가 명백히 강함",
        "project_order": new_order, "counts": counts,
    }


# ── ③ 기술 순서 ───────────────────────────────────────────────────────

def decide_skill_order(analysis: dict, skill_lines: list[str], resume_semantic_objects: list[dict]) -> dict:
    """skill 레이어에서 A/B로 연결된 실제 보유 기술만 앞으로. 새 우선순위
    점수 없음 - '연결됨/안됨' 이분류 + 원래 상대 순서 유지."""
    skill_lines = [ln for ln in skill_lines if ln.strip()]
    if len(skill_lines) < 2:
        return {"decision": "유지", "reason": "기술 목록이 1개 이하", "skill_order": None}

    skill_objs = [o for o in resume_semantic_objects if o.get("layer") == "skill"]
    line_by_object_id = {}
    for o in skill_objs:
        src = (o.get("evidence") or {}).get("source_text", "")
        oid = o.get("id")
        if not src or not oid:
            continue
        line = next((ln for ln in skill_lines if src in ln or ln in src), None)
        if line:
            line_by_object_id[oid] = line

    matched_lines: set[str] = set()
    matched_requirements: list[str] = []
    for it in analysis.get("items") or []:
        if it.get("layer") != "skill" or it.get("state") not in ("A", "B"):
            continue
        hit = False
        for m in it.get("matched_resume_objects") or []:
            line = line_by_object_id.get(m.get("resume_object_id"))
            if line:
                matched_lines.add(line)
                hit = True
        if hit:
            matched_requirements.append(it["jd_requirement"])

    if not matched_lines:
        return {"decision": "유지", "reason": "JD가 요구하는 기술과 A/B로 연결된 실제 보유 기술 없음",
                "skill_order": None}

    ordered = [ln for ln in skill_lines if ln in matched_lines] + [ln for ln in skill_lines if ln not in matched_lines]
    if ordered == skill_lines:
        return {"decision": "유지", "reason": "이미 관련 기술이 앞쪽", "skill_order": None}

    names = [ln.split(":", 1)[0].strip() for ln in ordered]
    return {
        "decision": "변경",
        "reason": f"JD가 요구하는 기술({', '.join(matched_requirements)})을 앞으로 이동",
        "skill_order": names,
    }


# ── ④+⑤ bullet 강조(직무 근거 주, culture 보조) ────────────────────────

_MATCHABLE_LAYERS = ("problem", "thinking", "task", "skill", "qualification")


def _project_sections_with_bullets(resume_raw: str, project_id: str, resume_semantic_objects: list[dict]) -> list[dict]:
    """resume_apply_engine.apply_commands()의 move_bullet이 실제로 실행
    가능한 단위(같은 서브섹션 내부)로만 bullet을 묶는다.

    2026-08-16(버그 수정, 실사용 다건 점검 중 Microsoft 공고 사례로 실측
    확인) - customizer.build_structured_resume()는 프로젝트 전체("해결
    과정 및 역할" + "성과" 등 모든 서브섹션)를 b1..bN 전역 번호로 매긴다.
    이걸 그대로 bullet 비교 단위로 쓰면 "성과 섹션의 bullet을 해결 과정
    섹션보다 앞으로 옮겨라" 같은, 애초에 실행 불가능한 판단이 나올 수
    있다 - customizer._replace_subsection_bullets()는 서브섹션 하나만
    교체하므로 서로 다른 서브섹션 사이의 재배치를 지원하지 않는다. 그
    결과 "카드는 변경인데 실제 이력서는 그대로"인 조용한 실패가 났다.
    그래서 판단 자체를 서브섹션 단위로 쪼갠다 - Apply Engine이 실제로
    할 수 있는 일과 Planner가 판단하는 대상을 일치시키는 것뿐, 새로운
    규칙이나 우선순위를 추가하는 게 아니다."""
    from resume_input import customizer

    _, blocks, _ = customizer._split_projects(resume_raw)
    block = next((b for t, b in blocks if t == project_id), None)
    if not block:
        return []
    matchable = [o for o in resume_semantic_objects if o.get("layer") in _MATCHABLE_LAYERS]

    def _find_refs(text: str) -> list[str]:
        return [
            o.get("id") for o in matchable
            if (o.get("evidence") or {}).get("source_text") and o["evidence"]["source_text"] in text
        ]

    sections = []
    for sec_title, sec_body in customizer.split_project_subsections(block):
        bullets = customizer.split_bullets(sec_body)
        if len(bullets) < 2:
            continue  # resume_apply_engine도 2개 미만이면 재배치를 안 함(동일 기준 재사용, 새 규칙 아님)
        items = [
            {"bullet_id": f"{sec_title}#{i}", "text": text, "refs": _find_refs(text)}
            for i, text in enumerate(bullets)
        ]
        sections.append({"section_title": sec_title, "bullets": items})
    return sections


def _project_bullet_signals(analysis: dict, project_id: str, bullets: list[dict]) -> list[dict]:
    job_signal: dict[str, tuple[int, str, str]] = {}
    culture_signal: dict[str, tuple[float, str, str, str]] = {}

    for it in analysis.get("items") or []:
        layer = it.get("layer")
        state = it.get("state")
        for m in it.get("matched_resume_objects") or []:
            if m.get("source_project") != project_id:
                continue
            rid = m.get("resume_object_id")
            if layer in _BULLET_JOB_LAYERS and state in _TIER_RANK:
                rank = _TIER_RANK[state]
                prev = job_signal.get(rid)
                if prev is None or rank > prev[0]:
                    job_signal[rid] = (rank, _TIER_LABEL[state], it["jd_requirement"])
            elif layer == "culture" and state in ("A", "B"):
                imp = it.get("importance")
                rank = _IMPORTANCE_RANK.get(imp, 0)
                prev = culture_signal.get(rid)
                if prev is None or rank > prev[0]:
                    culture_signal[rid] = (rank, imp, it["jd_requirement"], state)

    signals = []
    for idx, b in enumerate(bullets):
        refs = b.get("refs", [])
        best_job = (0, "없음", None)
        best_job_ref = None
        for ref in refs:
            if ref in job_signal and job_signal[ref][0] > best_job[0]:
                best_job = job_signal[ref]
                best_job_ref = ref
        best_culture = (0, None, None, None)
        best_culture_ref = None
        for ref in refs:
            if ref in culture_signal and culture_signal[ref][0] > best_culture[0]:
                best_culture = culture_signal[ref]
                best_culture_ref = ref
        signals.append({
            "bullet_id": b["bullet_id"], "position": idx,
            # 2026-08-16(버그 수정, 실사용 검증 중 실측 확인) - 이 bullet을
            # move_bullet Command로 옮길 때는 반드시 이 job_tier_rank를 만든
            # 바로 그 resume_object_id(best_job_ref)를 써야 한다. bullet의
            # refs[0](임의 첫 항목)을 쓰면, 그 bullet에 여러 근거가 달려
            # 있을 때 직무 판단과 무관한 객체(예: skill 레이어)의
            # evidence.source_text로 검색하게 되어 resume_apply_engine이
            # 실제 불릿을 못 찾고 조용히 건너뛴다 - "카드는 변경인데 실제
            # 이력서는 안 바뀜" 불일치가 이렇게 발생했다(Microsoft 공고
            # 사례로 재현).
            "job_tier_rank": best_job[0], "job_tier_label": best_job[1], "job_requirement": best_job[2],
            "job_tier_ref": best_job_ref,
            "culture_rank": best_culture[0], "culture_importance": best_culture[1],
            "culture_requirement": best_culture[2], "culture_ref": best_culture_ref,
        })
    return signals


def decide_bullet_order(signals: list[dict]) -> dict:
    """현재 1번째 bullet vs 최선의 대안만 비교. 직무 근거가 동률일 때만
    culture를 보조 기준으로 쓴다. 절대 위치 규칙 없음(사용자 명시적 반려).

    `driven_by`("job"|"culture"|None) - 실제 순서는 하나의 결정(④+⑤ 통합,
    사용자 확정 - 서로 다른 bullet을 각자 1번으로 보내려는 충돌 방지)이지만,
    화면에 "핵심 경험·성과"/"인재상·업무방식"을 별도 줄로 보여줄 때 이
    변경이 어느 근거로 일어났는지 구분하는 용도(새 판단 아님, 이미 계산된
    비교 결과를 어느 카드에 돌릴지 표시만 다르게 함)."""
    if not signals:
        return {"decision": "유지", "reason": "bullet 없음", "new_order": None, "driven_by": None}

    current_top = signals[0]
    best = current_top
    for s in signals[1:]:
        if s["job_tier_rank"] > best["job_tier_rank"]:
            best = s
        elif s["job_tier_rank"] == best["job_tier_rank"] and s["culture_rank"] > best["culture_rank"]:
            best = s

    if best["bullet_id"] == current_top["bullet_id"]:
        return {"decision": "유지", "reason": "현재 1번째 bullet이 이미 최선", "new_order": None, "driven_by": None}

    if best["job_tier_rank"] > current_top["job_tier_rank"]:
        reason = (f"{best['bullet_id']}({best['job_tier_label']}·{best['job_requirement']})가 "
                  f"현재 1번째({current_top['job_tier_label']})보다 직무 근거가 명백히 강함")
        driven_by = "job"
        moved_ref = best["job_tier_ref"]
    elif best["job_tier_rank"] == current_top["job_tier_rank"] and best["culture_rank"] > current_top["culture_rank"]:
        reason = (f"직무 근거는 동률({best['job_tier_label']})이지만 {best['bullet_id']}가 "
                  f"{best['culture_importance']} 인재상({best['culture_requirement']})까지 보조로 연결됨")
        driven_by = "culture"
        moved_ref = best["culture_ref"]
    else:
        return {"decision": "유지", "reason": "직무 근거가 현재 1번째보다 명백히 우세하지 않음", "new_order": None, "driven_by": None}

    if not moved_ref:
        # 방어적 폴백(정상적으로는 도달하지 않음 - driven_by가 job/culture로
        # 정해졌다는 건 그 신호를 만든 ref가 반드시 있었다는 뜻). 혹시라도
        # 없으면 근거 없는 Command를 만들지 않고 유지로 안전하게 되돌린다 -
        # "카드는 변경인데 실제로는 안 바뀜" 불일치를 여기서도 다시 막는다.
        return {"decision": "유지", "reason": "이동 근거 resume_object를 특정할 수 없어 유지", "new_order": None, "driven_by": None}

    new_order = [best["bullet_id"]] + [s["bullet_id"] for s in signals if s["bullet_id"] != best["bullet_id"]]
    return {
        "decision": "변경", "reason": reason, "new_order": new_order,
        "moved_bullet": best["bullet_id"], "moved_ref": moved_ref, "driven_by": driven_by,
    }


def culture_evidence_summary(analysis: dict) -> dict:
    """⑤ 화면 표시 전용 - 실제 순서 변경과 무관하게, culture 레이어에서
    A/B로 연결된 실제 근거가 있는지만 요약한다(있음/근거 없음 이분류,
    새 점수 없음)."""
    items = []
    for it in analysis.get("items") or []:
        if it.get("layer") != "culture":
            continue
        if it.get("state") in ("A", "B") and it.get("matched_resume_objects"):
            items.append(it["jd_requirement"])
    return {"has_evidence": bool(items), "requirements": items}


# ── ① 자기소개(2026-08-16 재설계, 사용자 확정) ─────────────────────────
# 2026-08-16 2차 수정(사용자 확정, Microsoft 사례로 문제 제기) - 1차
# 버전은 이 culture 항목이 Semantic Linking에서 A/B(=이미 이 회사 수준을
# "충족"한다고 판정됨)일 때만 후보로 삼았다. 그런데 Microsoft의 "성장
# 마인드셋 및 포용적 리더십"처럼 judge_engine이 "미충족(실무 경험 부족)"
# 으로 본 항목도, 사실은 "이 회사가 요구하는 수준의 리더십 경험"이
# 없다는 뜻이지 "성장/학습 성향을 보여줄 근거가 이력서 어디에도 없다"는
# 뜻이 아니다. ①의 질문은 "지원 자격을 충족하는가"(Judge/Hard
# Eligibility의 몫)가 아니라 "이 인재상 중 실제 경험에서 꺼내 강조할 수
# 있는 부분이 있는가"라는 다른 질문이다 - 그래서 Linking의 A/B/C/D
# 판정을 그대로 재사용하지 않는다.
#
# 이 함수는 여전히 "어느 인재상을 볼지"만 고른다(critical>core>normal,
# 기존 importance 등급 그대로 - 새 숫자 점수 아님. 동점이면 하나를
# 임의로 고르지 않고 유지). "그 인재상 문장 안에 여러 특성이 섞여
# 있을 수 있고, 그중 일부만 근거가 있을 수 있다"는 분해 + 이력서 전체
# Resume Object에서 그 근거를 찾는 일은 결정론적 규칙으로 하지 않는다
# (자연어 문장을 특성 단위로 쪼개는 건 규칙으로 안전하게 할 수 없다는
# 게 이미 이 세션에서 반복 확인됨) - rewrite_engine.generate_headline_
# proposal()이 LLM 1회 호출로 "분해 -> 이력서 전체에서 근거 확인 ->
# 있으면 그 부분만 반영해 최소 수정, 없으면 원문 유지"까지 한 번에
# 하고, 인용한 근거가 실제 이력서 문장에 있는지는 그 함수가 사실 검증
# (존재 여부만, 새 유사도 아님)으로 막는다.

_HEADLINE_IMPORTANCE_RANK = {"critical": 3, "core": 2, "normal": 1}


def decide_headline(jd_objs: list[dict], current_headline: str) -> dict:
    culture_objs = [o for o in jd_objs if o.get("layer") == "culture"]
    if not culture_objs:
        return {"decision": "유지", "reason": "이 공고에 인재상(culture) 항목이 없음", "current_headline": current_headline}

    best_rank = max(_HEADLINE_IMPORTANCE_RANK.get(o.get("importance"), 0) for o in culture_objs)
    top = [o for o in culture_objs if _HEADLINE_IMPORTANCE_RANK.get(o.get("importance"), 0) == best_rank]
    if len(top) > 1:
        tied = ", ".join(o.get("normalized_text", "") for o in top)
        return {
            "decision": "유지",
            "reason": f"같은 우선순위의 인재상이 여럿이라({tied}) 하나를 임의로 고르지 않음",
            "current_headline": current_headline,
        }

    winner = top[0]
    return {
        "decision": "평가 필요",
        "reason": "이 공고가 가장 강조하는 인재상 중 실제 경험으로 뒷받침되는 부분이 있는지 확인이 필요함(LLM 1회)",
        "jd_culture": winner.get("normalized_text"),
        "importance": winner.get("importance"),
        "current_headline": current_headline,
    }


# ── 통합 진입점 ───────────────────────────────────────────────────────

def build_resume_commands(
    analysis: dict, jd_objs: list[dict], resume_raw: str, resume_semantic_objects: list[dict],
) -> tuple[list[dict], dict]:
    """resume_apply_engine.apply_commands()가 그대로 실행할 수 있는 Command
    목록(②③④+⑤)과, 자동 적용 대상이 아닌 ①의 판단 결과(notes)를 함께
    반환한다. 새 Command 스키마를 만들지 않는다 - 기존 move_project/
    move_bullet에 reorder_skills 하나만 추가한다(apply_commands()도 같이
    확장, customizer._reorder_skills()는 이미 있던 함수 재사용)."""
    from resume_input import customizer

    commands: list[dict] = []
    notes: dict = {}

    _, project_blocks, _ = customizer._split_projects(resume_raw)
    project_titles = [t for t, _ in project_blocks]

    # ②
    project_decision = decide_project_order(analysis, project_titles)
    notes["project_order"] = project_decision
    if project_decision["decision"] == "변경":
        for i, title in enumerate(project_decision["project_order"]):
            commands.append({"type": "move_project", "project_id": title, "priority": i + 1})

    # ③
    _, skill_lines, _ = customizer._split_skill_lines(resume_raw)
    skill_decision = decide_skill_order(analysis, skill_lines, resume_semantic_objects)
    notes["skill_order"] = skill_decision
    if skill_decision["decision"] == "변경":
        commands.append({"type": "reorder_skills", "skill_order": skill_decision["skill_order"], "priority": 1})

    # ④+⑤ (서브섹션별로 최대 1건 - 최소 변경 원칙. Apply Engine이 실제로
    # 재배치를 실행할 수 있는 단위(같은 서브섹션 내부)와 판단 단위를
    # 맞춘다 - _project_sections_with_bullets() docstring 참고)
    bullet_decisions = []
    for proj_id in project_titles:
        for section in _project_sections_with_bullets(resume_raw, proj_id, resume_semantic_objects):
            signals = _project_bullet_signals(analysis, proj_id, section["bullets"])
            decision = decide_bullet_order(signals)
            bullet_decisions.append({
                "project_id": proj_id, "section_title": section["section_title"], **decision,
            })
            if decision["decision"] == "변경":
                # move_bullet 커맨드는 resume_object_id를 요구한다(resume_apply_engine이
                # evidence.source_text로 실제 불릿을 재탐색) - 반드시 이 순서
                # 변경 판단을 만든 바로 그 resume_object(moved_ref)를 써야 한다.
                commands.append({
                    "type": "move_bullet", "project_id": proj_id,
                    "bullet_id": decision["moved_ref"], "priority": 1,
                })
    notes["bullet_order"] = bullet_decisions
    notes["culture_evidence"] = culture_evidence_summary(analysis)

    # ①(자동 적용 없음 - notes만)
    current_headline = customizer.extract_current_headline(resume_raw)
    notes["headline"] = decide_headline(jd_objs, current_headline)

    return commands, notes
