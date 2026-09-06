"""
resume_input/customization_rules.py

Rule Executor v1(2026-07-21, docs/architecture_rule_executor_v1.md 참고)
- LLM 호출 0회. JD Understanding이 이미 만든 resume_customization_
targets(project_priority/skill_priority/headline_focus/expression_
focus/experience_focus/avoid_changes)와 Resume Understanding의
Object별 concepts만 입력으로 쓴다.

판단하지 않는다 - 오직:
- concepts 집합의 교집합 개수(overlap_count) 계산 + 정렬(동점이면
  원래 순서 유지) - 가중치/threshold/레이어 우선순위 없음
- 실제 스킬명 문자열 일치 여부(Rule2 - 스킬은 의미 비교 대상이 아님)
- concepts 교집합 존재 여부(있다/없다)

임베딩/코사인/threshold/의미 유사도 계산은 어디에도 없다. Rule5(표현)
는 APPROVED_TERM_MAP을 그 자리에서 확인해서 change/keep을 즉시
반환한다 - Rule Executor가 최종 판단까지 끝낸다(후처리 단계 없음).

이 파일 이전 버전(문자열 포함 검사 기반 Planner, matching_anchor/
persona_tags 실험 등)의 배경은 docs/verification/2026-07-21_
customization_rule_engine/summary.md와 docs/verification/2026-07-21_
planner_ab_test/summary.md에 남아 있다 - 다시 실험하지 않는다.
"""
from __future__ import annotations

import json

from resume_input import customizer

_CRITICAL_CORE = ("critical", "core")

# 승인된 동일 의미 용어 사전 - 이 쌍 안에 있는 것만 치환 대상이다.
# key: 이력서 쪽에 최종적으로 남길 표현, value: 이 표현과 동일 의미로
# 취급하는 JD 쪽 표현들.
APPROVED_TERM_MAP: dict[str, list[str]] = {
    "데이터 시각화": ["시각화", "시각화 대시보드 제작"],
    "대시보드 구축": ["대시보드 제작", "대시보드 기획"],
    "고객 세분화": ["고객 유형 분류", "행동 기반 고객 분류"],
    "리텐션 분석": ["고객 유지 분석", "재구매 분석"],
    "실험 설계": ["가설 수립 및 검증"],
    "지표 설계": ["분석 지표 정의"],
    "운영 효율화": ["운영 개선", "업무 효율 개선"],
}

REASON_TEMPLATES = {
    "CURRENTLY_OPTIMAL": "현재 {item_name}이 공고의 핵심 요구와 이미 일치하거나 변경할 실익이 없어 유지합니다.",
    "NO_JD_TARGET": "공고에서 이 판단에 쓸 만한 target이 없어 {item_name}을(를) 유지합니다.",
    "PROJECT_ORDER_CHANGED": "공고와 개념이 더 많이 겹치는 프로젝트 순서로 변경합니다: {order}.",
    "SKILL_ORDER_CHANGED": "공고가 우선순위로 요구하는 기술({skills})을 앞으로 이동합니다.",
    "BULLET_REORDER": "공고와 연결되는 '{section_title}' 내용({concepts})이 뒤쪽에 있어 앞으로 이동합니다.",
    "HEADLINE_FIT": "현재 자기소개가 공고 핵심 개념({concepts})과 이미 연결됩니다.",
    "HEADLINE_MANUAL_REVIEW": "현재 자기소개가 공고 핵심 개념({concepts})과 연결되지 않습니다. 새 문장을 자동 생성하지 않으므로 직접 확인이 필요합니다.",
    "NO_CONCEPT_MATCH": "공고 용어와 연결되는 표현이 없어 변경하지 않았습니다.",
    "APPROVED_TERM_ALIGNMENT": "이력서의 '{resume_term}' 표현이 공고의 '{jd_term}'과 같은 의미로 승인되어 있어 표현을 통일합니다.",
    "EXPRESSION_HIGHLIGHTED": "다음 표현이 공고 핵심 개념과 연결됩니다: {source_texts}",
}


def _reason(code: str, **kwargs) -> str:
    return REASON_TEMPLATES[code].format(**kwargs)


def _decision(item: str, status: str, reason_code: str, reason: str,
              evidence: dict | None = None, before=None, after=None) -> dict:
    return {
        "item": item, "status": status, "reason_code": reason_code, "reason": reason,
        "evidence": evidence or {}, "before": before, "after": after,
    }


def _parse_jd_targets(job: dict) -> dict:
    """job["jd_summary"](JD Understanding 생성 시 이미 저장된 자유 JSON)
    안의 resume_customization_targets 키를 읽는다. 새 DB 컬럼을 만들지
    않고 기존 jd_summary 컬럼을 재사용한다(understanding.py 참고)."""
    try:
        summary = json.loads(job.get("jd_summary") or "{}")
    except (TypeError, ValueError):
        return {}
    return summary.get("resume_customization_targets") or {}


def _concept_set(items) -> set[str]:
    """target 리스트([{"concept":.., "reason":..}, ...] 또는 문자열
    리스트인 skill_priority 둘 다 처리)에서 concept 문자열 집합만
    뽑는다."""
    out: set[str] = set()
    for it in items or []:
        c = (it.get("concept") if isinstance(it, dict) else it) or ""
        c = c.strip()
        if c:
            out.add(c)
    return out


def _norm_ws(text: str) -> str:
    """공백(줄바꿈 포함)과 불릿 기호만 지운다 - 원본 PDF의 줄바꿈과
    evidence.source_text(공백으로 이어붙임)가 표기만 다르고 내용은
    같은 경우를 같다고 보기 위함. 이건 "JD 요구사항과 이력서를
    매칭"하는 게 아니라, 이미 관련 있다고 확정된 Resume Object가
    이력서 원문의 어느 불릿에 대응하는지 찾는 순수 내부 매핑이다
    (Rule의 판단 대상이 아님)."""
    import re
    return re.sub(r"[\s•]+", "", text or "")


# ── Rule 1 — Project Order ──────────────────────────────────────────

_NON_PROJECT_SECTIONS = {"보유 기술", "자격증", "대외활동", "", None}


def _project_concepts(resume_objects: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for o in resume_objects:
        proj = (o.get("evidence") or {}).get("project")
        if proj in _NON_PROJECT_SECTIONS:
            continue
        out.setdefault(proj, set()).update(c.strip() for c in (o.get("concepts") or []) if c and c.strip())
    return out


def _evaluate_project_order(jd_targets: dict, resume_objects: list[dict], resume_raw: str) -> dict:
    item = "이력서 구조 최적화"
    _, blocks, _ = customizer._split_projects(resume_raw)
    titles = [t for t, _ in blocks]
    if len(titles) < 2:
        return _decision(item, "keep", "CURRENTLY_OPTIMAL", _reason("CURRENTLY_OPTIMAL", item_name="프로젝트 순서"))

    target = _concept_set(jd_targets.get("project_priority"))
    if not target:
        return _decision(item, "keep", "NO_JD_TARGET", _reason("NO_JD_TARGET", item_name="프로젝트 순서"))

    proj_concepts = _project_concepts(resume_objects)
    # overlap_count: 사실값(교집합 개수)일 뿐 가중치/점수가 아니다 -
    # 가중치·threshold·레이어 우선순위는 추가하지 않는다(사용자 확정).
    overlap_count = {t: len(proj_concepts.get(t, set()) & target) for t in titles}
    ordered = sorted(titles, key=lambda t: -overlap_count[t])  # 안정 정렬 - 동점이면 원래 순서 유지

    if ordered == titles:
        return _decision(item, "keep", "CURRENTLY_OPTIMAL", _reason("CURRENTLY_OPTIMAL", item_name="프로젝트 순서"))

    overlaps = {t: sorted(proj_concepts.get(t, set()) & target) for t in titles}
    return _decision(
        item, "change", "PROJECT_ORDER_CHANGED",
        _reason("PROJECT_ORDER_CHANGED", order=", ".join(ordered)),
        evidence={"overlap_count": overlap_count, "overlaps": overlaps},
        before=titles, after={"project_order": ordered},
    )


# ── Rule 2 — Skill Order (실제 스킬명, concepts 관여 없음) ────────────

def _evaluate_skill_order(jd_targets: dict, resume_raw: str) -> dict:
    item = "이력서 구조 최적화"
    _, lines, _ = customizer._split_skill_lines(resume_raw)
    current_names = [ln.split(":", 1)[0].strip() for ln in lines if ln.strip()]
    if not current_names:
        return _decision(item, "keep", "CURRENTLY_OPTIMAL", _reason("CURRENTLY_OPTIMAL", item_name="기술 순서"))

    skill_priority = [s for s in (jd_targets.get("skill_priority") or []) if s]
    if not skill_priority:
        return _decision(item, "keep", "NO_JD_TARGET", _reason("NO_JD_TARGET", item_name="기술 순서"))

    target_names: list[str] = []
    for skill in skill_priority:
        matched = next(
            (c for c in current_names if skill.lower() in c.lower() or c.lower() in skill.lower()),
            None,
        )
        if matched and matched not in target_names:
            target_names.append(matched)

    if not target_names:
        return _decision(item, "keep", "NO_JD_TARGET", _reason("NO_JD_TARGET", item_name="기술 순서"))

    full_order = target_names + [c for c in current_names if c not in target_names]
    if full_order == current_names:
        return _decision(item, "keep", "CURRENTLY_OPTIMAL", _reason("CURRENTLY_OPTIMAL", item_name="기술 순서"))

    return _decision(
        item, "change", "SKILL_ORDER_CHANGED",
        _reason("SKILL_ORDER_CHANGED", skills=", ".join(target_names)),
        evidence={"matched_skills": target_names},
        before=current_names, after={"skill_order": full_order},
    )


def evaluate_structure(job: dict, resume_objects: list[dict], resume_raw: str) -> dict:
    """① 이력서 구조 최적화 - Rule1(프로젝트) + Rule2(스킬). 둘 중
    하나라도 change면 항목 전체를 change로, 아니면 keep."""
    jd_targets = _parse_jd_targets(job)
    skill_d = _evaluate_skill_order(jd_targets, resume_raw)
    project_d = _evaluate_project_order(jd_targets, resume_objects, resume_raw)

    if skill_d["status"] != "change" and project_d["status"] != "change":
        return _decision("이력서 구조 최적화", "keep", "CURRENTLY_OPTIMAL",
                          _reason("CURRENTLY_OPTIMAL", item_name="이력서 구조"))

    reasons = [d["reason"] for d in (skill_d, project_d) if d["status"] == "change"]
    after = {}
    if skill_d["status"] == "change":
        after.update(skill_d["after"])
    if project_d["status"] == "change":
        after.update(project_d["after"])
    return _decision(
        "이력서 구조 최적화", "change", "STRUCTURE_CHANGE", " ".join(reasons),
        evidence={"skill": skill_d, "project": project_d},
        before={"skill": skill_d.get("before"), "project": project_d.get("before")},
        after=after,
    )


# ── Rule 3 — Experience(핵심 경험 및 어필 전략) - bullet_reorders ────

_EMPHASIS_SECTIONS = ("해결 과정 및 역할", "성과")


def _bullet_matches_object(bullet: str, source_text: str) -> bool:
    """이 불릿이 이 Resume Object의 evidence.source_text에 대응하는가
    (같은 문서 내부 매핑 - JD 판단 아님)."""
    if not source_text:
        return False
    b, s = _norm_ws(bullet), _norm_ws(source_text)
    if not b or not s:
        return False
    if b in s or s in b:
        return True
    return any(_norm_ws(line) and _norm_ws(line) in b for line in source_text.split("\n"))


def _evaluate_bullet_reorders(jd_targets: dict, resume_objects: list[dict], resume_raw: str) -> tuple[list[dict], list[str]]:
    target = _concept_set(jd_targets.get("experience_focus"))
    if not target:
        return [], []

    relevant_objects = [o for o in resume_objects if _concept_set(o.get("concepts")) & target]
    if not relevant_objects:
        return [], []

    _, blocks, _ = customizer._split_projects(resume_raw)
    reorders: list[dict] = []
    reasons: list[str] = []
    for title, block in blocks:
        project_objects = [o for o in relevant_objects if (o.get("evidence") or {}).get("project") == title]
        if not project_objects:
            continue
        for section_title, section_body in customizer.split_project_subsections(block):
            if section_title not in _EMPHASIS_SECTIONS:
                continue
            bullets = customizer.split_bullets(section_body)
            if len(bullets) < 2:
                continue
            relevant_bullets = [
                b for b in bullets
                if any(_bullet_matches_object(b, (o.get("evidence") or {}).get("source_text", "")) for o in project_objects)
            ]
            if not relevant_bullets or bullets[0] in relevant_bullets:
                continue  # 이미 앞쪽이거나 관련 불릿 없음
            new_order = relevant_bullets + [b for b in bullets if b not in relevant_bullets]
            if new_order == bullets:
                continue
            reorders.append({"project_title": title, "section_title": section_title, "new_order": new_order})
            matched_concepts = sorted({c for o in project_objects for c in _concept_set(o.get("concepts")) & target})
            reasons.append(_reason(
                "BULLET_REORDER", section_title=f"{title} - {section_title}",
                concepts=", ".join(matched_concepts),
            ))
    return reorders, reasons


def evaluate_experience(job: dict, resume_objects: list[dict], resume_raw: str) -> dict:
    item = "핵심 경험 및 어필 전략"
    jd_targets = _parse_jd_targets(job)
    reorders, reasons = _evaluate_bullet_reorders(jd_targets, resume_objects, resume_raw)

    if reorders:
        return _decision(
            item, "change", "BULLET_REORDER", " ".join(reasons),
            evidence={"bullet_reorders": [{"project_title": r["project_title"], "section_title": r["section_title"]} for r in reorders]},
            before=None, after={"bullet_reorders": reorders},
        )

    if not _concept_set(jd_targets.get("experience_focus")):
        return _decision(item, "keep", "NO_JD_TARGET", _reason("NO_JD_TARGET", item_name="경험/성과 강조"))
    return _decision(item, "keep", "CURRENTLY_OPTIMAL", _reason("CURRENTLY_OPTIMAL", item_name="경험/성과 강조"))


# ── Rule 4 — Headline Fit(자기소개 최적화) - 후보 선택 없음 ──────────

def evaluate_headline(job: dict, resume_raw: str) -> dict:
    item = "자기소개 최적화"
    jd_targets = _parse_jd_targets(job)
    target = _concept_set(jd_targets.get("headline_focus"))
    current_headline = customizer.extract_current_headline(resume_raw) or ""

    if not target or not current_headline:
        return _decision(item, "keep", "NO_JD_TARGET", _reason("NO_JD_TARGET", item_name="자기소개"))

    headline_norm = current_headline.replace(" ", "").lower()
    hits = sorted(c for c in target if c.replace(" ", "").lower() in headline_norm)

    if hits:
        return _decision(
            item, "keep", "HEADLINE_FIT", _reason("HEADLINE_FIT", concepts=", ".join(hits)),
            evidence={"matched_concepts": hits}, before=current_headline,
        )
    return _decision(
        item, "manual_review", "HEADLINE_MANUAL_REVIEW",
        _reason("HEADLINE_MANUAL_REVIEW", concepts=", ".join(sorted(target))),
        evidence={"jd_headline_focus": sorted(target)}, before=current_headline,
    )


# ── Rule 5 — Expression(표현 및 JD 용어 최적화) ──────────────────────

def evaluate_terms(job: dict, resume_objects: list[dict], resume_raw: str) -> dict:
    """concepts로 "무엇이 JD와 연결되는가"를 찾은 뒤, APPROVED_TERM_MAP
    으로 승인된 치환이 가능한지 그 자리에서 확인해서 change/keep을
    즉시 반환한다(후처리 단계 없음 - Rule Executor가 최종 판단까지
    끝낸다). 승인된 치환이 없으면 텍스트는 안 바꾸고 강조 근거(reason)
    만 남긴다."""
    item = "표현 및 JD 용어 최적화"
    jd_targets = _parse_jd_targets(job)
    target = _concept_set(jd_targets.get("expression_focus"))
    if not target:
        return _decision(item, "keep", "NO_JD_TARGET", _reason("NO_JD_TARGET", item_name="표현"))

    hits = []
    for o in resume_objects:
        overlap = _concept_set(o.get("concepts")) & target
        if not overlap:
            continue
        source_text = (o.get("evidence") or {}).get("source_text", "")
        if not source_text:
            continue
        hits.append({"source_text": source_text, "matched_concepts": sorted(overlap)})

    if not hits:
        return _decision(item, "keep", "NO_CONCEPT_MATCH", _reason("NO_CONCEPT_MATCH"))

    replacements: list[dict] = []
    for hit in hits:
        source_text = hit["source_text"]
        for resume_term, jd_synonyms in APPROVED_TERM_MAP.items():
            if resume_term in resume_raw:
                continue  # 이미 승인된 표현을 쓰고 있으면 손 안 댐
            matched_synonym = next((s for s in jd_synonyms if s in source_text), None)
            if matched_synonym:
                replacements.append({"before": matched_synonym, "after": resume_term, "source_text": source_text})
                break

    if replacements:
        return _decision(
            item, "change", "APPROVED_TERM_ALIGNMENT",
            " ".join(_reason("APPROVED_TERM_ALIGNMENT", resume_term=r["before"], jd_term=r["after"]) for r in replacements),
            evidence={"replacements": replacements},
            before=[r["before"] for r in replacements], after={"replacements": replacements},
        )

    return _decision(
        item, "keep", "EXPRESSION_HIGHLIGHTED",
        _reason("EXPRESSION_HIGHLIGHTED", source_texts="; ".join(f"'{h['source_text']}'" for h in hits)),
        evidence={"expression_hits": hits},
    )


# ── 통합 진입점 ───────────────────────────────────────────────────────

def evaluate(job: dict, resume_semantic_objects: list[dict], resume_raw: str) -> list[dict]:
    """4개 항목 전부 판단해서 반환한다. LLM 호출 없음, 임베딩/코사인/
    threshold 없음 - jd_summary.resume_customization_targets와
    resume_semantic_objects의 concepts 조회 + 교집합 개수/존재 여부
    확인뿐이다. 모든 항목의 change/keep/manual_review가 여기서 최종
    확정된다(후처리 단계 없음)."""
    return [
        evaluate_structure(job, resume_semantic_objects, resume_raw),
        evaluate_experience(job, resume_semantic_objects, resume_raw),
        evaluate_terms(job, resume_semantic_objects, resume_raw),
        evaluate_headline(job, resume_raw),
    ]
