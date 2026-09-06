"""
resume_input/presentation_layer.py

Presentation Layer(체크포인트 6, docs/semantic_object_schema.md) -
Engine(link_engine.py + judge_engine.py) 결과를 화면용 데이터로
**번역만** 한다. Semantic Object/Link/Judge 결과를 수정하거나 새로운
판단을 추가하지 않는다(Presentation != Explanation Engine) - 여기서
하는 일은 Engine JSON -> 화면용 데이터 변환, 표시 순서 결정, 한국어
라벨 변환, 카드/표 데이터 구성뿐이다. **새 LLM 호출 없음.**

2026-08-13(사용자 지시, 실제 화면 버그 수정) - `job["_semantic_match"]`가
옛 cosine 엔진(semantic_matching.py, Match Object: status/engine_confidence/
match_type/matched_fields/missing_fields/ranking_score/coverage)에서
`link_engine.py`(Link: relation/match_strength/relation_reason_type/
matched_dimensions/missing/improvement)로 완전히 교체(2026-08-04~06)된
뒤에도 이 파일 일부 함수가 옛 필드를 계속 읽고 있었다 - 옛 필드가 전부
빠져있으니 크래시는 안 나지만(.get() 기본값으로 조용히 통과) 화면에
잘못된/텅 빈 값이 나오는 조용한 버그였다. 실제로 app.py가 호출하는
`build_section1_view`/`build_section2_cards`만 이번에 새 스키마로
고쳤다 - `build_section2_view`/`build_strengths_gaps_view`/
`build_gaps_split_view`는 app.py 어디서도 더 이상 호출되지 않는 죽은
코드라(2026-08-13 확인) 이번 수정 대상이 아니다.

Top30 LLM Validator(match_validator.py)는 2026-07-17 자동 실행 경로에서
제거됐다(사용자 확정) - 순위/설명 모두 Engine 기준 하나로만 통일한다.
Top1~50 전체가 ②번 카드에서 동일한 규칙(relation/match_strength/
matched_dimensions)으로 표시된다 - "1~30위는 LLM 검증, 31~50위는 Engine
판단"처럼 job 순위에 따라 표시 기준이 갈리던 구조를 없앴다(같은 화면
안에서 기준이 섞여 있으면 사용자가 더 헷갈린다는 게 이유).
`job["_match_quality"]`(Rule 기반 품질 검증, pipeline.py의
`_apply_match_quality_check()`)는 진단용 데이터일 뿐 이 화면이
소비하지 않는다.

`job["_semantic_match"]`(pipeline.py의 `_compute_semantic_match()`가
붙여준 것 - `{"link_version", "links", "unmatched_resume_objects"}`,
호환을 위해 `matches`/`jd_object`/`resume_object` 별칭도 같이 채워짐)가
없는 job은 `has_semantic_data()`가 False를 반환한다 - 호출부(app.py)가
그 경우 기존 화면(build_jd_detail_view 등)으로 폴백해야 한다(이 파일은
그 폴백을 만들지 않는다 - 옛 코드를 그대로 재사용하는 게 호출부 책임).
"""
from __future__ import annotations

import json
import re

from datetime import date as _date

from resume_input import judge_engine
from resume_input.semantic_matching import MATCH_THRESHOLD
from resume_input.job_detail import (
    culture_highlights as _culture_highlights,
    extract_display_sections as _extract_display_sections,
    format_company_intro as _format_company_intro,
    infer_industry as _infer_industry,
)
from resume_input.job_prep import (
    extract_deadline as _extract_deadline,
    extract_job_info as _extract_job_info,
    parse_deadline_date as _parse_deadline_date,
)
from resume_input.rule_representation import build_rule_based_representation
from resume_input.job_store import get_company_research

# ── Insight Model 공통 어휘 (엔진 용어 -> 한국어, 화면 노출 문자열에는
# critical/core/DIRECT/FRAME/engine_confidence 같은 엔진 원문이 절대
# 남으면 안 된다 - 2026-07-16 재설계) ────────────────────────────────
_FIELD_LABEL = {
    "action": "수행 방식", "object": "대상", "goal": "목적",
    "domain": "분야", "outcome": "성과", "method": "방법",
    "scope": "범위",  # link_engine.py matched_dimensions 전용(action/goal/object/scope)
}

_CONFIDENCE_LEVEL_LABEL = {
    "very_high": "매우 높음",
    "high": "높음",
    "medium": "보통 · 검토 필요",
    "missing": "근거 부족",
}
_CONFIDENCE_LEVEL_TONE = {
    "very_high": "green",
    "high": "green",
    "medium": "orange",
    "missing": "gray",
}

_IMPORTANCE_RANK = {"critical": 0, "core": 1, "normal": 2, "low": 3}


def _importance_rank(jd_obj: dict) -> int:
    """정렬 전용 - 화면에 숫자/영단어를 노출하지 않는다."""
    return _IMPORTANCE_RANK.get(jd_obj.get("importance"), 2)


def _confidence_level(engine_confidence: float) -> str:
    """engine_confidence(이미 계산된 값)를 기존 엔진 상수(MATCH_THRESHOLD
    =0.75)로만 구간화한다 - 새 임계값을 만들지 않는다. 이 함수가
    호출되는 시점엔 이미 CANDIDATE_THRESHOLD(0.60)는 통과한 상태다
    (semantic_matching.match_jd_object가 그 밑은 애초에 후보에서
    제외함)."""
    if engine_confidence >= 0.85:
        return "very_high"
    if engine_confidence >= MATCH_THRESHOLD:
        return "high"
    return "medium"


def has_semantic_data(job: dict) -> bool:
    """이 job이 새 스키마(Semantic Object/Match Object)로 화면을 만들
    수 있는지 - False면 호출부가 옛 화면으로 폴백해야 한다."""
    return bool(job.get("_semantic_match")) and bool(job.get("jd_semantic_objects"))


def _jd_objects(job: dict) -> list[dict]:
    raw = job.get("jd_semantic_objects")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _jd_summary(job: dict) -> dict:
    """JD Summary(Job Overview - job_summary/purpose/problem/thinking/
    environment). understanding.py가 JD Understanding 생성 시 이미
    LLM 1회로 만들어 `jd_summary` 컬럼에 저장해두는 값이다(체크포인트
    2) - 여기서 새로 만들지 않고 그대로 파싱만 한다. 지금까지 어디서도
    읽히지 않고 있었다(2026-07-16 실측 확인 - LLM 호출은 했는데 화면에
    한 번도 안 쓰인 상태였음)."""
    raw = job.get("jd_summary")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _obj_label(obj: dict) -> str:
    return obj.get("meaning") or obj.get("normalized_text") or ""


def _short_label(obj: dict) -> str:
    """짧은 개념 라벨 - `meaning`(LLM이 쓴 전체 문장) 대신
    `normalized_text`(이미 LLM이 정규화한 짧은 표현)를 우선 사용한다.
    배지/행동 목록처럼 한 줄로 짧게 보여줘야 하는 자리에 쓴다 - 전체
    문장을 넣으면 배지가 깨지거나 "다음 행동"이 문장이 돼 버린다."""
    return obj.get("normalized_text") or obj.get("meaning") or ""


def _josa(word: str, if_batchim: str, if_no_batchim: str) -> str:
    """받침 유무에 따라 조사(을/를, 이/가 등)를 고른다 - 동적으로 조립한
    문장이 "경험를 준비하면 좋습니다" 같은 어색한 한국어가 되지 않게
    한다. 한글이 아니거나 빈 문자열이면 받침 있는 쪽(더 흔함)으로
    처리한다."""
    if not word:
        return if_batchim
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return if_batchim if (code - 0xAC00) % 28 != 0 else if_no_batchim
    return if_batchim


def build_section1_view(job: dict, resume_facts: dict | None = None) -> dict:
    """① 추천 결과 - "지원해도 되는가? 왜? 어느 정도 자신 있는가?
    무엇이 부족한가?" 4가지 질문에만 답한다(UI Model, 2단계). 전부
    이미 Engine이 계산한 Link/Judge/importance의 집계일 뿐 새 판단이
    아니다(LLM 호출 없음) - 다만 화면에 보이는 문자열에는 "critical"/
    "core"/"relation" 같은 엔진 용어를 절대 노출하지 않는다(2026-07-16
    재설계 - 첫 라운드에서 bullet에 그대로 남아있던 걸 발견해 제거).

    2026-08-13 - headline은 이제 죽은 필드였던 ranking_score(옛
    cosine 엔진 전용, link_engine.py에는 애초에 없는 필드라 항상
    None이었다) 대신 judge_engine.judge()의 실제 지원 판단(지원/보류/
    비추천)을 그대로 문장으로 옮긴다 - 새 판단을 만드는 게 아니라 이미
    있는 Judge Rule 결과(LLM 호출 없음, 순수 함수)를 그대로 재사용."""
    match_result = job.get("_semantic_match") or {}
    matches = match_result.get("links", [])
    matched_n = sum(1 for m in matches if m.get("matched_resume_objects"))

    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}
    total_key = matched_key = 0
    missing_key_labels: list[str] = []
    for m in matches:
        jd_obj = jd_by_id.get(m["jd_object_id"], {})
        if jd_obj.get("importance") not in ("critical", "core"):
            continue
        total_key += 1
        if m.get("matched_resume_objects"):
            matched_key += 1
        else:
            missing_key_labels.append(_short_label(jd_obj))

    # headline(3초 티어) - "지원해도 되는가?"의 결론 한 줄.
    _HEADLINE_BY_DECISION = {
        "지원": ("지원 경쟁력이 높은 편입니다.", "green"),
        "보류": ("지원 경쟁력이 보통 수준입니다.", "orange"),
        "비추천": ("지원 경쟁력이 낮은 편입니다.", "gray"),
    }
    if matches:
        decision = judge_engine.judge(match_result, _jd_objects(job), resume_facts).get("decision")
        headline, headline_tone = _HEADLINE_BY_DECISION.get(decision, ("", ""))
    else:
        headline, headline_tone = "", ""

    # bullets(10초 티어) - "왜?"의 근거 한 줄.
    bullets: list[str] = []
    if total_key:
        if matched_key == total_key:
            bullets.append("핵심 요구사항을 모두 충족합니다.")
        else:
            ratio = matched_key / total_key
            level = "대부분" if ratio >= 0.6 else "일부만"
            bullets.append(f"핵심 요구사항 {matched_key}/{total_key}개를 {level} 충족합니다.")

    # action_items(다음 행동) - "무엇이 부족한가?"에서 끝내지 않고
    # "무엇을 하면 되는가?"로 마무리한다. 설명 문장이 아니라 짧은
    # 행동 목록.
    action_items = [f"{label} 경험 강조" for label in missing_key_labels[:3]]

    return {
        "headline": headline,
        "headline_tone": headline_tone,
        "matched_count": matched_n,
        "total_count": len(matches),
        "summary_bullets": bullets,
        "action_items": action_items,
    }


def build_strengths_gaps_view(job: dict, max_n: int = 5) -> tuple[list[str], list[str]]:
    """강점/보완점 태그 - Match Object의 `status`만으로 만든다(2026-07-16
    추가, 옛 matched_keywords 키워드 겹침 기반을 대체). status="match"인
    JD Object는 강점, "candidate"(Top30 검증 대상) 또는 Missing Object
    (매칭 자체가 없음)는 보완점으로 분류한다 - 둘 다 이미 Engine이 낸
    판단이지 새로 만드는 게 아니다."""
    match_result = job.get("_semantic_match") or {}
    matches = match_result.get("matches", [])
    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}

    strengths, gaps = [], []
    for m in matches:
        jd_obj = jd_by_id.get(m["jd_object"], {})
        label = _short_label(jd_obj)
        if not label:
            continue
        if m.get("status") == "match":
            strengths.append(label)
        else:
            gaps.append(label)
    return strengths[:max_n], gaps[:max_n]


def _resume_evidence_label(r_obj: dict) -> str:
    """이력서 근거를 사람이 읽는 한 구절로 - evidence.project가 있으면
    "OO 프로젝트"를, 없으면 이미 있는 짧은 라벨을 쓴다. normalized_text
    원문을 그대로 노출하지 않는다."""
    project = (r_obj.get("evidence") or {}).get("project")
    if project:
        return f"{project} 프로젝트"
    return _obj_label(r_obj) or "관련 경험"


def _insight_match(best: dict, r_obj: dict) -> dict:
    """Insight 계층(1단계) - Match Object 1건만 보고 엔진 용어가 전혀
    없는 한국어 판단을 만든다. 반환: {"level": "very_high"|"high"|
    "medium", "reason": str, "todo": str}. Top1~50 전체가 이 규칙
    하나로만 판단된다(2026-07-17, LLM Validator 자동 실행 제거 - 사용자
    확정). 아직 배지 색 등 표시 스타일은 모른다(UI Model인
    build_section2_view의 역할). Missing Object는 이 함수를 거치지
    않는다(`_insight_missing` 참고)."""
    resume_evidence = _resume_evidence_label(r_obj)
    match_type = best.get("match_type", "")
    matched_fields = best.get("matched_fields") or []
    missing_fields = best.get("missing_fields") or []

    confidence = best.get("engine_confidence", 0.0)
    level = _confidence_level(confidence)
    if match_type == "DIRECT":
        josa = _josa(resume_evidence, "이", "가")
        reason = f"{resume_evidence}{josa} 이 요구사항과 동일한 개념으로 확인됩니다."
    elif matched_fields:
        labels = ", ".join(_FIELD_LABEL.get(f, f) for f in matched_fields)
        josa = _josa(labels, "이", "가")
        reason = f"{resume_evidence} 경험에서 {labels}{josa} 일치합니다."
    elif match_type == "EVIDENCE":
        reason = f"{resume_evidence} 경험이 이 요구사항이 기대하는 근거와 일치합니다."
    else:
        reason = f"{resume_evidence} 경험이 의미상 관련이 있습니다."

    todo = ""
    if missing_fields:
        labels = ", ".join(_FIELD_LABEL.get(f, f) for f in missing_fields)
        todo = f"다만 {labels}에 대한 근거는 부족합니다."
    return {"level": level, "reason": reason, "todo": todo}


def _insight_missing(jd_obj: dict) -> dict:
    """Missing Object(이력서에서 이 요구사항의 근거를 못 찾음)의 Insight."""
    expected = jd_obj.get("expected_evidence") or []
    values = [e.get("value") for e in expected if e.get("value")]
    if values:
        joined = ", ".join(values[:2])
        josa = _josa(values[:2][-1], "을", "를")
        todo = f"{joined}{josa} 준비하면 좋습니다."
    else:
        todo = "관련 경험을 보완하면 좋습니다."
    return {
        "level": "missing",
        "reason": "이력서에서 이 요구사항과 관련된 경험을 찾지 못했습니다.",
        "todo": todo,
    }


def build_section2_view(job: dict, resume_objects: list[dict], resume_relations: list[dict]) -> list[dict]:
    """② 매칭 카드 - UI Model(2단계). `_insight_match`/`_insight_missing`
    (Insight, 1단계)의 결과를 배지 톤·카드 정렬 순서로만 변환한다.
    "출처"/"Verified Match"/"Engine Match" 같은 파이프라인 단계명은
    화면에 노출하지 않는다. Top1~50 전체가 Engine Match Object
    (engine_confidence/matched_fields/missing_fields) 기준 하나로만
    표시된다(2026-07-17, LLM Validator 자동 실행 제거 - 사용자 확정)."""
    match_result = job.get("_semantic_match") or {}
    matches = match_result.get("matches", [])
    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}
    resume_by_id = {(o.get("id") or o.get("normalized_text")): o for o in resume_objects}

    rows = []
    for m in matches:
        jd_obj = jd_by_id.get(m["jd_object"], {})
        jd_point = _obj_label(jd_obj)

        if not m.get("matched_resume_objects"):
            insight = _insight_missing(jd_obj)
            rows.append({
                "jd_point": jd_point,
                "resume_point": "이력서에서 관련 근거를 찾지 못함",
                "confidence_label": _CONFIDENCE_LEVEL_LABEL[insight["level"]],
                "confidence_tone": _CONFIDENCE_LEVEL_TONE[insight["level"]],
                "confidence_pct": 0,
                "reason": insight["reason"],
                "todo": insight["todo"],
                "_importance_rank": _importance_rank(jd_obj),
            })
            continue

        best = m["matched_resume_objects"][0]
        r_obj = resume_by_id.get(best["resume_object"], {})
        insight = _insight_match(best, r_obj)

        rows.append({
            "jd_point": jd_point,
            "resume_point": _obj_label(r_obj) or "관련 경험",
            "confidence_label": _CONFIDENCE_LEVEL_LABEL[insight["level"]],
            "confidence_tone": _CONFIDENCE_LEVEL_TONE[insight["level"]],
            "confidence_pct": round(best.get("engine_confidence", 0.0) * 100),
            "reason": insight["reason"],
            "todo": insight["todo"],
            "_importance_rank": _importance_rank(jd_obj),
        })

    rows.sort(key=lambda r: r["_importance_rank"])
    return rows



def build_section3_view(job: dict) -> dict:
    """③ "이 회사는 무슨 문제를 해결하려고 나를 뽑는가?"에 답한다
    (Semantic Object를 설명하는 화면이 아니다). JD Summary(문서 요약,
    LLM이 이미 만들어 `jd_summary` 컬럼에 저장해둔 값 - 새 LLM 호출
    없음) 다음에 problem/task/(skill+qualification 합쳐 dedup)/culture
    4그룹만 짧은 라벨로 보여준다. **thinking 레이어는 목록에서 제외**
    한다 - JD Summary의 "중요하게 보는 사고방식" 문단이 이미 같은
    내용을 서술하고 있어서, 그대로 다시 나열하면 같은 이야기를 두 번
    하게 된다(사용자가 지적한 반복의 핵심 원인)."""
    summary = _jd_summary(job)

    by_layer: dict[str, list[dict]] = {}
    for obj in _jd_objects(job):
        by_layer.setdefault(obj.get("layer", ""), []).append(obj)

    sections = []

    problem_objs = by_layer.get("problem") or []
    problem_labels = [_short_label(o) for o in problem_objs if _short_label(o)]
    if problem_labels:
        sections.append({"label": "핵심 목표", "items": problem_labels})

    task_objs = by_layer.get("task") or []
    task_labels = [_short_label(o) for o in task_objs if _short_label(o)]
    if task_labels:
        sections.append({"label": "가장 중요한 업무", "items": task_labels})

    skill_qual_objs = (by_layer.get("skill") or []) + (by_layer.get("qualification") or [])
    seen: set[str] = set()
    skill_labels: list[str] = []
    for o in skill_qual_objs:
        label = _short_label(o)
        if label and label not in seen:
            seen.add(label)
            skill_labels.append(label)
    if skill_labels:
        sections.append({"label": "중요 역량", "items": skill_labels})

    culture_objs = by_layer.get("culture") or []
    culture_labels = [_short_label(o) for o in culture_objs if _short_label(o)]
    if culture_labels:
        sections.append({"label": "조직 문화", "items": culture_labels})

    return {"summary": summary, "sections": sections}


# ── STEP3 "공고 분석" 화면 v2(2026-08-14, 사용자 확정 - 레퍼런스 이미지
# 기준 재설계) - "1. 공고 한눈에 보기"(JD Understanding 6-layer 그대로)
# + "2. 매칭 결과"(Semantic Link + Judge 결과 그대로)만 쓴다. 새 카테고리
# (Product Analytics/검색·플랫폼/업무 성격 같은 임의 그룹)를 만들지
# 않는다 - jd_semantic_objects의 layer 필드(problem/task/skill/
# qualification/thinking/culture, 6종 고정)를 그대로 6개 카드로 나눈다.
_JD_OVERVIEW_LAYERS = (
    ("problem", "해결해야 하는 문제", "Problem"),
    ("task", "주요 업무", "Task"),
    ("skill", "요구 역량", "Skill"),
    ("qualification", "자격 요건", "Qualification"),
    ("thinking", "일하는 방식", "Thinking"),
    ("culture", "조직/협업 관련 요구", "Culture"),
)


def build_jd_overview_view(job: dict, max_n: int = 4) -> list[dict]:
    """"1. 공고 한눈에 보기" - jd_semantic_objects를 layer별로만 묶어서
    보여준다(6종 고정, 새 카테고리 없음). 값이 없는 layer는 아예
    리스트에서 뺀다(호출부가 "빈 카드"를 그리지 않게 - 사용자 확정).
    항목이 중요도(_importance_rank) 순으로 상위 max_n개까지만, 있는
    만큼만(강제로 채우지 않음) - LIST 화면의 build_rule_task_view와
    같은 원칙."""
    objs = _jd_objects(job)
    by_layer: dict[str, list[dict]] = {}
    for obj in objs:
        by_layer.setdefault(obj.get("layer", ""), []).append(obj)

    out = []
    for layer_key, label_kor, label_eng in _JD_OVERVIEW_LAYERS:
        layer_objs = sorted(by_layer.get(layer_key) or [], key=_importance_rank)
        items = [lbl for o in layer_objs[:max_n] if (lbl := _short_label(o))]
        if not items:
            continue
        out.append({"layer": layer_key, "label_kor": label_kor, "label_eng": label_eng, "items": items})
    return out


def build_evidence_detail_view(job: dict, jd_object_id: str) -> dict | None:
    """"근거 보기"(2. 매칭 결과) 상세 패널 - JD evidence 원문, Resume
    evidence 원문, 연결된 요소(matched_dimensions), 판단(relation),
    판단 이유를 전부 이미 있는 값에서 그대로 가져온다. 새로 만드는
    문장/판단 없음.

    JD evidence 원문은 jd_semantic_objects의 evidence.source_text(Link가
    아니라 원래 Understanding 단계 Object 자체에만 있다 - link_engine.py의
    jd_requirement는 normalized_text라 원문이 아니다). Resume evidence는
    job["_semantic_match"]["links"]의 matched_resume_objects[].evidence_text
    (link_engine.py가 이미 resume_object.evidence.source_text에서
    옮겨둔 값)를 그대로 쓴다 - Unverified(matched_resume_objects=[])
    항목은 여기서도 빈 리스트만 나온다(Resume 근거를 지어내지 않음)."""
    link_result = job.get("_semantic_match") or {}
    links = link_result.get("links") or []
    link = next((l for l in links if l.get("jd_object_id") == jd_object_id), None)
    if link is None:
        return None

    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}
    jd_obj = jd_by_id.get(jd_object_id, {})
    jd_evidence_text = (jd_obj.get("evidence") or {}).get("source_text") or link.get("jd_requirement") or ""

    resume_evidence = [
        {"project": m.get("source_project", ""), "text": m.get("evidence_text", "")}
        for m in (link.get("matched_resume_objects") or [])
    ]

    return {
        "jd_requirement": link.get("jd_requirement", ""),
        "jd_evidence_text": jd_evidence_text,
        "resume_evidence": resume_evidence,
        "matched_dimensions": link.get("matched_dimensions") or [],
        "relation": link.get("relation"),
        "missing": link.get("missing") or [],
    }


# ── STEP3 "공고 분석" v3(2026-08-15, 사용자 확정 - 레퍼런스 이미지
# 285ffd77 기준 재설계) - "역할 → 판단 → 공고 요약 → 나와의 적합성 →
# (숨김) 상세 근거 → 지원 전략 → 다음 행동" 순서로, 화면에 보이는 모든
# 문장을 "무슨 데이터를 어떤 규칙으로 가공했는지" 추적 가능하게 만든다.
# 새 LLM 호출 없음 - jd_summary/jd_semantic_objects/analysis_engine
# Evidence Contract가 이미 계산해둔 값만 고르고 모을 뿐, 새 문장을
# 생성하지 않는다. importance("critical"/"core") 같은 엔진 영단어는
# 화면에 그대로 노출하지 않는다(기존 _importance_rank 원칙 유지) -
# 한국어 배지("필수"/"중요")로만 변환해서 보여준다.

_IMPORTANCE_BADGE_KOR = {"critical": "필수", "core": "중요", "normal": "일반", "low": "선택"}


def importance_badge_kor(importance: str | None) -> str:
    """app.py는 Engine 결과를 직접 읽지 않는다(체크포인트 6) - importance
    raw value를 이 함수로만 한국어 배지로 바꾼다."""
    return _IMPORTANCE_BADGE_KOR.get(importance, "일반")


def connected_fields_text(matched_dimensions: list[str], max_n: int = 2) -> str:
    """app.py는 matched_dimensions raw value(action/goal/object/scope 등
    link_engine.py 내부 축 이름)를 직접 화면에 노출하지 않는다 - 기존
    _FIELD_LABEL 어휘 변환을 그대로 재사용한다(새 매핑 없음)."""
    return _connected_fields_text(matched_dimensions, max_n)


def build_role_intro(job: dict) -> str:
    """헤더 아래 1줄 역할 소개 - jd_summary.job_summary(Understanding
    단계에서 이미 LLM 1회로 만들어 DB에 저장해둔 값, understanding.py
    참고)를 그대로 옮긴다. 새 문장을 만들지 않는다 - 값이 없으면 빈
    문자열(호출부가 그 줄 자체를 안 그림)."""
    summary = _jd_summary(job)
    return summary.get("job_summary") or summary.get("purpose") or ""


def _normalize_dedup_key(text: str) -> str:
    """"제품 및 사업 전략"과 "제품/사업 전략"처럼 접속사·구두점만 다른
    사실상 동일한 라벨을 하나로 합치기 위한 비교키(2026-08-16, 사용자
    지적 - "의미가 겹치는 표현을 그대로 여러 개 보여주지 않는다"). 화면에
    보여줄 문자열 자체는 바꾸지 않는다 - 중복 판정에만 쓰고, 먼저 나온
    표기를 그대로 채택한다(새 표준 표기를 만들지 않음).
    2026-08-16 추가 - "·"(U+00B7 MIDDLE DOT) 외에 한글 문서에 흔한
    "ㆍ"(U+318D 반각 가운뎃점)도 같이 걸러낸다 - LLM이 둘 중 아무거나
    써도 같은 라벨로 합쳐지게 한다(실측 - 두 유니코드가 섞여 나와서
    "제품 및 사업 전략"/"제품ㆍ사업 전략"이 안 합쳐지던 경우가 있었다)."""
    return re.sub(r"[\s/·ㆍ,및]+", "", text)


def _dedup_ranked(order: list[str], count: dict[str, int]) -> list[str]:
    seen_keys: set[str] = set()
    out: list[str] = []
    for label in sorted(order, key=lambda d: (-count[d], order.index(d))):
        key = _normalize_dedup_key(label)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(label)
    return out


def build_overview_v3(job: dict, task_max: int = 4, competency_max: int = 5, domain_max: int = 5) -> dict:
    """"공고 한눈에 보기" 3열 - 전부 jd_semantic_objects에 이미 있는
    필드를 고르거나 모으기만 한다(새 문장 없음):
    - 핵심 업무: task 레이어 항목의 normalized_text, importance 순.
    - 중요 역량: task 레이어를 제외한 나머지 레이어(thinking/skill/
      qualification/culture) 항목의 normalized_text, importance 순.
      2026-08-16(사용자 지적) - 기존엔 importance="critical"이면 레이어
      무관하게 넣어서 task 레이어의 critical 항목이 핵심 업무/중요
      역량 양쪽에 중복 표시되는 버그가 있었다 - 레이어를 배타적으로
      나눠서 고쳤다(task→핵심 업무, 나머지→중요 역량, 한 Object가 두
      칸에 겹치지 않는다).
    - 주요 업무 영역: 각 Object가 Understanding 단계에서 이미 갖고
      있는 frame.domain(예: "제품/사업 전략", "그로스 분석" - LLM이
      원래 그 항목을 분류해둔 실제 업무 영역명)을 빈도순으로 모은다.
      2026-08-16(사용자 지적) - 기존엔 concepts 필드("가설"/"방향성"
      같은 짧은 개념어 파편)를 썼는데 "분석 영역"이라 부르기엔 너무
      추상적이었다 - frame.domain이 훨씬 실제 업무 영역에 가깝다.
      culture 레이어는 업무 영역이 아니라 태도/문화라서 제외한다.
      2026-08-16 두 번째 수정(사용자 지적 - "제품 및 사업 전략"과
      "제품/사업 전략"처럼 의미가 겹치는 표현을 하나로 normalize") -
      _dedup_ranked()로 접속사/구두점만 다른 사실상 동일 라벨을 합친다.
      2026-08-16 네 번째 수정(사용자 지적 - "가설 기반 사고는 업무
      영역이라기보다 역량에 가깝다") - thinking/qualification 레이어도
      culture와 같은 이유로 제외한다: 이 레이어들의 frame.domain은
      LLM이 "그 역량이 어떤 분야 지식인지"를 채운 값이라, task/skill/
      problem 레이어의 frame.domain(실제 업무 맥락)과 성격이 달라
      "업무 영역" 카드에 섞으면 "가설 기반 사고" 같은 역량 문구가
      업무 영역인 것처럼 보였다.
    - 키워드: 전 항목의 concepts 필드를 빈도순으로 모은다. 2026-08-16
      세 번째 수정(사용자 지적 - "위 카드에서 이미 충분히 전달된 일반
      단어를 무조건 반복하지 않는다") - 핵심 업무/중요 역량/주요 업무
      영역에 이미 나온 문자열과 정확히 같은 키워드는 제외하고, 최대
      8개까지 보여준다(기존 6개 → 새 스펙 "6~8개").
    - 각 열의 개수 상한을 열마다 다르게 받는다(기존엔 max_each 하나를
      세 열에 공용으로 썼다 - 2026-08-16 사용자 스펙: 핵심 업무 4 /
      중요 역량 5 / 주요 업무 영역 5)."""
    objs = _jd_objects(job)

    tasks = [
        lbl for o in sorted((o for o in objs if o.get("layer") == "task"), key=_importance_rank)
        if (lbl := _short_label(o))
    ][:task_max]

    competency_layers = ("thinking", "skill", "qualification", "culture")
    key_competencies = [
        lbl for o in sorted((o for o in objs if o.get("layer") in competency_layers), key=_importance_rank)
        if (lbl := _short_label(o))
    ][:competency_max]

    domain_order: list[str] = []
    domain_count: dict[str, int] = {}
    for o in objs:
        if o.get("layer") in ("culture", "thinking", "qualification"):
            continue
        d = (o.get("frame") or {}).get("domain")
        if not d:
            continue
        if d not in domain_count:
            domain_order.append(d)
        domain_count[d] = domain_count.get(d, 0) + 1
    domains_ranked = _dedup_ranked(domain_order, domain_count)[:domain_max]

    concept_order: list[str] = []
    concept_count: dict[str, int] = {}
    for o in objs:
        for c in (o.get("concepts") or []):
            if not c:
                continue
            if c not in concept_count:
                concept_order.append(c)
            concept_count[c] = concept_count.get(c, 0) + 1
    already_shown = {_normalize_dedup_key(t) for t in (tasks + key_competencies + domains_ranked)}
    concepts_ranked = [
        c for c in _dedup_ranked(concept_order, concept_count)
        if _normalize_dedup_key(c) not in already_shown
    ]

    return {
        "tasks": tasks,
        "key_competencies": key_competencies,
        "focus_areas": domains_ranked,
        "keywords": concepts_ranked[:8],
    }


def build_domain_meta(job: dict) -> str:
    """헤더 아래 "도메인" 메타 한 줄 - 회사가 속한 산업(예: "교육
    플랫폼 · 에듀테크"). 새 LLM 호출을 하지 않는다 - 기존
    build_company_info_view()가 이미 하는 순서(1. company_research
    테이블의 검증된 리서치 값 → 2. job_detail.infer_industry()의
    키워드 규칙 기반 폴백, 둘 다 LLM 없음)를 그대로 재사용만 한다.
    둘 다 없으면 빈 문자열(호출부가 그 줄 자체를 안 그림 - "정보
    없음"을 화면에 노출하지 않는다, 2026-08-16 사용자 지시 - "이미
    검증된 데이터가 있는지 먼저 확인해서 그 값만 표시")."""
    info = build_company_info_view(job)
    industry = info.get("industry") or ""
    return "" if industry == "정보 없음" else industry


def _josa_eun_neun(word: str) -> str:
    return _josa(word, "은", "는")


_ELIG_REASON_RE = re.compile(r"요구=(.*?),\s*보유=(.*)")


def _split_elig_reason(reason: str) -> tuple[str, str]:
    """eligibility_compare.compare()가 이미 만들어둔 "요구=X, 보유=Y"
    형식 문자열(judge_engine._classify_hard_eligibility()가 blocking
    item에 `_elig_reason`으로 그대로 붙여준다)을 두 값으로 나눈다 - 새
    비교를 하지 않고 이미 계산된 문자열을 화면 두 칸(요구/이력서 확인)에
    나눠 꽂기만 한다. 패턴이 안 맞으면(예: "복합(또는) 조건" 안내문)
    통째로 첫 칸에 넣고 둘째 칸은 비운다."""
    m = _ELIG_REASON_RE.match(reason or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return reason or "", ""


_DECISION_ACTION = {
    "지원": "우선 지원",
    "보류": "우선순위 낮춰 지원",
    "비추천": "지원 우선순위에서 제외",
}


def build_decision_banner(analysis: dict) -> dict:
    """"지원 판단" 배너 - decision은 judge_engine.judge()가 이미 정한
    값 그대로(여기서 다시 계산하지 않음). 2026-08-16(사용자 스펙 -
    "결정에 가장 큰 영향을 준 이유 1개만 보여준다") - 이전엔 걸린 항목을
    최대 2개까지 나열했는데, 화면 하나에 이유를 하나만 남기고 나머지는
    "나와의 적합성"/"상세 근거"에서 보여준다(반복 제거).

    2026-08-16 두 번째 수정(사용자 지적 - "판정 → 이유 → 행동으로 끝나야
    빨리 읽힌다") - reason은 한 문장으로 압축하고("직무 핵심과 관련된
    경험은 있으나, {A}의 근거가 충분치 않다"처럼 강점 항목 원문을 다시
    풀어 쓰지 않는다 - 어차피 "나와의 적합성"에 나온다), action은
    decision 값에 대해 고정된 3개 라벨(_DECISION_ACTION)만 쓴다 - 매번
    새 문장을 조립하지 않는다(Judge가 이미 정한 decision을 그대로
    라벨링만 함, 새 판단 아님).

    Hard Eligibility(blocking)가 있으면 일반 구조적 gap보다 항상
    우선한다(사용자 스펙 5번 - "Hard Eligibility 미충족이면 일반적인
    적합성 부족보다 우선"). 이 경우 reason 문장 대신 요구/이력서 확인
    값을 나눈 hardEligibility 목록을 반환해서, 화면이 "요구 학력: 석사
    이상 / 이력서 확인: 전문학사" 같은 비교 표를 그리게 한다 - 새 비교를
    하지 않고 eligibility_compare.compare()가 이미 계산해둔 reason
    문자열을 두 칸으로 나눠 꽂기만 한다(_split_elig_reason)."""
    decision = analysis.get("decision")
    blocking = analysis.get("blocking_requirements") or []
    strong = analysis["by_state"]["A"]
    risk = sorted(analysis["by_state"]["D"], key=_importance_rank)

    hard_eligibility = []
    if decision == "비추천" and blocking:
        for b in blocking:
            required, actual = _split_elig_reason(b.get("_elig_reason", ""))
            hard_eligibility.append({
                "requirement": b.get("jd_requirement", ""),
                "required": required,
                "actual": actual,
            })
        reason = "필수 지원 조건을 충족하지 못했습니다."
    elif decision == "보류" and risk:
        top = risk[0]["jd_requirement"]
        lead = "핵심 요구사항은 연결되지만" if strong else "직무 핵심과 관련된 경험은 있으나"
        reason = f"{lead}, {top} 등 일부 역량의 직접 근거가 부족합니다."
    elif decision == "보류":
        # structural gap도 hard eligibility도 없는 "보류"는 judge_engine의
        # unknown(Hard Eligibility 확인불가) 경로뿐이다 - 그 경우
        # judge_result.decision_reason[0]을 그대로 쓴다(새 문장 없음).
        reason = (analysis.get("decision_reason") or ["필수 조건을 아직 확인하지 못했습니다."])[0]
    elif decision == "지원":
        reason = "구조적으로 지원을 막는 문제가 확인되지 않았습니다."
    else:
        reason = "판단 근거를 계산하지 못했습니다."

    action = _DECISION_ACTION.get(decision, "")
    return {"decision": decision, "reason": reason, "action": action, "hardEligibility": hard_eligibility}


def build_fit_summary_v3(analysis: dict) -> dict:
    """"나와의 적합성" 3열 - 이미 있는 5-state Evidence Contract의
    A(강점)/B+C(일부 연결)/D(핵심 리스크)를 그대로 옮긴다(E는 여기
    없음 - "확인되지 않음"과 "사용자가 확정한 실제 공백"을 다시 섞지
    않기 위해 E는 상세 근거 표에서만 별도로 보여준다). intro 한 줄도
    실제 항목 텍스트만 이어붙인다.

    risk만 importance 순으로 정렬해서 반환한다(2026-08-16, 사용자
    지적 - "핵심 리스크 카드에 중요하지 않은 항목까지 다 올라온다".
    카드에 몇 개까지 보여줄지는 화면(컴포넌트)이 정하고, 여기서는
    "무엇을 먼저 보여줄 후보로 삼을지"만 정한다 - 전체 목록/개수는
    그대로 유지해 상세 근거에서는 계속 전부 확인할 수 있다)."""
    strong = analysis["by_state"]["A"]
    partial = analysis["by_state"]["B"] + analysis["by_state"]["C"]
    risk = sorted(analysis["by_state"]["D"], key=_importance_rank)

    if risk and strong:
        s = strong[0]["jd_requirement"]
        r = risk[0]["jd_requirement"]
        intro = f"{s} {_josa_eun_neun(s)} 잘 맞지만, {r} {_josa_eun_neun(r)} 아직 확인되지 않았습니다."
    elif risk:
        r = risk[0]["jd_requirement"]
        intro = f"{r} {_josa_eun_neun(r)} 아직 확인되지 않았습니다."
    elif strong:
        intro = "핵심 요구사항과 대부분 연결됩니다."
    else:
        intro = "연결 근거를 계산하지 못했습니다."

    return {
        "intro": intro,
        "strong": strong, "strong_count": len(strong),
        "partial": partial, "partial_count": len(partial),
        "risk": risk, "risk_count": len(risk),
        "total_count": len(strong) + len(partial) + len(risk),
    }



def build_detail_rows_v3(job: dict, analysis: dict) -> dict:
    """"상세 근거 보기" 표 - build_fit_summary_v3와 같은 3그룹(A/B+C/D,
    E 제외)을 그대로 재사용해서 그룹당 번호가 매겨진 행을 만든다. 행마다
    build_evidence_detail_view()로 JD 원문/이력서 근거/연결된 요소/부족한
    부분을 그대로 채운다 - 새 판단 없음. app.py의 Custom Component(CCv2)가
    이 값을 JSON으로 그대로 받아 그린다(체크포인트 6 - app.py가 Engine
    결과를 직접 읽지 않는다는 원칙을 CCv2 페이로드에도 유지)."""
    groups = {
        "strong": analysis["by_state"]["A"],
        "partial": analysis["by_state"]["B"] + analysis["by_state"]["C"],
        "risk": analysis["by_state"]["D"],
    }
    out: dict[str, list[dict]] = {}
    for key, items in groups.items():
        rows = []
        for i, it in enumerate(items, 1):
            jd_object_id = it["jd_object_id"]
            detail = build_evidence_detail_view(job, jd_object_id) or {}
            resume_evidence = detail.get("resume_evidence") or []
            missing = detail.get("missing") or []
            matched_dims = [
                _FIELD_LABEL.get(f, f) for f in (detail.get("matched_dimensions") or [])
            ]
            rows.append({
                "id": f"{key}-{jd_object_id}",
                "n": i,
                "title": it["jd_requirement"],
                "importanceBadge": importance_badge_kor(it.get("importance")),
                "evidenceLabel": f"근거 {len(resume_evidence)}건" if resume_evidence else "근거 없음",
                "matchedText": ", ".join(matched_dims[:2]) if matched_dims else "-",
                "missingText": missing[0].get("text", "") if missing else "-",
                "jdEvidence": detail.get("jd_evidence_text", ""),
                "resumeEvidence": resume_evidence,
                "matchedDims": matched_dims,
                "missingList": missing,
            })
        out[key] = rows
    return out


# ── Composition Rule 1~5 (2026-07-18, 사용자 요청) ────────────────────
# "이미 있는 Semantic Object를 조합하는 규칙만 만들면 된다 - LLM을 또
# 붙일 필요 없다"는 사용자 판단에 따라, JD Semantic Object(problem/
# task/thinking/domain)만으로 화면용 문장/목록/비율을 조립한다. 전부
# 순수 함수이고 새 LLM 호출도, 새 판단(누가 적합한지 등)도 없다 -
# 이미 Understanding/Matching이 만든 값을 문장 템플릿으로 재구성만
# 한다. UI(app.py)에는 아직 연결하지 않는다(사용자 지시 - "이 다섯
# 개가 완성되면, 그다음에 UI를 입히는 게 맞습니다").

_LAYER_KOR_LABEL = {
    "problem": "문제", "task": "업무", "skill": "기술",
    "qualification": "자격", "thinking": "사고방식", "culture": "조직문화",
}


def _dominant_domain(objs: list[dict]) -> str | None:
    """여러 Semantic Object의 frame.domain 중 가장 대표적인 값 하나를
    고른다(Rule 2 - "도메인은 이미 있음, 그대로 표시"). 빈도수가 가장
    높은 domain을 쓰고, 동률이면 problem > task > skill > qualification
    > culture > thinking 순으로 우선한다(problem이 보통 업종/도메인을
    가장 잘 요약한다)."""
    layer_priority = {"problem": 0, "task": 1, "skill": 2, "qualification": 3, "culture": 4, "thinking": 5}
    counts: dict[str, int] = {}
    best_layer: dict[str, int] = {}
    for o in objs:
        domain = ((o.get("frame") or {}).get("domain") or "").strip()
        if not domain:
            continue
        counts[domain] = counts.get(domain, 0) + 1
        prio = layer_priority.get(o.get("layer", ""), 9)
        if domain not in best_layer or prio < best_layer[domain]:
            best_layer[domain] = prio
    if not counts:
        return None
    return max(counts, key=lambda d: (counts[d], -best_layer[d]))


def build_domain_view(job: dict) -> str | None:
    """Rule 2 - 이 공고의 대표 도메인 하나."""
    return _dominant_domain(_jd_objects(job))


def build_jd_one_liner(job: dict) -> str | None:
    """Rule 1 - "{도메인}에서 {Problem}을 위해 {Task}를 수행하는 역할"
    템플릿으로 AI 공고 한줄 요약을 만든다. Problem/Task 둘 다 없으면
    지어내지 않고 None."""
    objs = _jd_objects(job)
    if not objs:
        return None
    problems = sorted((o for o in objs if o.get("layer") == "problem"), key=_importance_rank)
    tasks = sorted((o for o in objs if o.get("layer") == "task"), key=_importance_rank)
    domain = _dominant_domain(objs)
    problem_text = _short_label(problems[0]) if problems else None
    task_text = _short_label(tasks[0]) if tasks else None

    if problem_text and task_text:
        body = f"{problem_text}{_josa(problem_text, '을', '를')} 위해 {task_text}{_josa(task_text, '을', '를')} 수행하는 역할"
    elif task_text:
        body = f"{task_text}{_josa(task_text, '을', '를')} 수행하는 역할"
    elif problem_text:
        body = f"{problem_text}{_josa(problem_text, '을', '를')} 해결하는 역할"
    else:
        return None
    return f"{domain}에서 {body}" if domain else body


def build_core_tasks_view(job: dict, max_n: int = 3) -> list[str]:
    """Rule 3 - Task 레이어 상위 max_n개를 짧은 라벨로."""
    objs = _jd_objects(job)
    tasks = sorted((o for o in objs if o.get("layer") == "task"), key=_importance_rank)
    return [label for o in tasks[:max_n] if (label := _short_label(o))]


def build_skill_tags_view(job: dict, max_n: int = 4) -> list[str]:
    """Rule 3 확장(2026-07-24, 공고 카드 "필수 기술" 태그용) - Skill
    레이어 상위 max_n개를 짧은 라벨로. build_core_tasks_view와 같은
    패턴이고 layer만 다르다."""
    objs = _jd_objects(job)
    skills = sorted((o for o in objs if o.get("layer") == "skill"), key=_importance_rank)
    return [label for o in skills[:max_n] if (label := _short_label(o))]


def _thinking_clause(obj: dict) -> str | None:
    """Thinking Object 1개를 "{대상}을(를) {행위}" 형태의 짧은 절로
    바꾼다(frame.action/object가 없으면 normalized_text로 대체)."""
    frame = obj.get("frame") or {}
    action = (frame.get("action") or "").strip()
    target = (frame.get("object") or "").strip()
    if action and target:
        return f"{target}{_josa(target, '을', '를')} {action}"
    return _short_label(obj) or None


def build_persona_view(job: dict, max_n: int = 2) -> str | None:
    """Rule 4 - Thinking Pattern -> "이런 사람을 찾습니다" 문장. 상위
    max_n개 Thinking Object를 "~하고 ~하는 사람"으로 잇는다. 재료가
    없으면 None(지어내지 않음)."""
    objs = _jd_objects(job)
    thinking = sorted((o for o in objs if o.get("layer") == "thinking"), key=_importance_rank)
    clauses = [c for o in thinking[:max_n] if (c := _thinking_clause(o))]
    if not clauses:
        return None
    return "하고 ".join(clauses) + "하는 사람"


def build_requirement_summary_view(job: dict, max_n: int = 2) -> str | None:
    """Rule 4 확장(2026-07-24, 공고 카드 "핵심 요구"용) - Thinking
    Object가 있으면 build_persona_view를 그대로 쓴다. Thinking이 없고
    Problem만 있으면 상위 1개로 "~을(를) 해결할 수 있는 사람" 문장을
    만든다(build_jd_one_liner의 problem/task 폴백과 같은 패턴). 둘 다
    없으면 None(지어내지 않음)."""
    persona = build_persona_view(job, max_n=max_n)
    if persona:
        return persona
    objs = _jd_objects(job)
    problems = sorted((o for o in objs if o.get("layer") == "problem"), key=_importance_rank)
    label = _short_label(problems[0]) if problems else None
    if not label:
        return None
    return f"{label}{_josa(label, '을', '를')} 해결할 수 있는 사람"


def build_deadline_view(job: dict, today: _date | None = None) -> dict:
    """"마감일" meta 값(2026-07-25 재설계, 사용자 확정) - "마감일"과
    "D-Day"를 별도 칸 두 개로 나눠 같은 정보를 중복 표시하던 것을
    "D-5(2026.07.08)" 한 줄로 합친다. {"text", "tone"} 반환. 규칙(전부
    이미 있는 extract_deadline/parse_deadline_date 조합, 새 판단 없음):
    - 실제 달력 날짜를 파싱할 수 있으면 "D-{n}(YYYY.MM.DD)"(지났으면
      "마감(YYYY.MM.DD)").
    - "상시채용"/"수시모집"처럼 원문에 실제로 있는, 날짜 없는 문구는
      그 문구 그대로 표시(정보 없음으로 뭉개지 않는다 - 사용자 확정).
    - 원문에 마감 관련 문구 자체가 없으면 "정보 없음"(진짜 정보가 없을
      때만)."""
    text = _extract_deadline(job.get("posting_text", "") or "")
    if not text:
        return {"text": "정보 없음", "tone": "gray", "badge": None, "date": None}
    d = _parse_deadline_date(text)
    if d is None:
        return {"text": text, "tone": "gray", "badge": None, "date": None}
    today = today or _date.today()
    days_left = (d - today).days
    date_str = d.strftime("%Y.%m.%d")
    # badge/date(2026-08-14 추가) - "text"를 그대로 유지한 채(하위 호환),
    # 목록 카드가 "D-n 배지"와 "마감 YYYY.MM.DD" 날짜를 별도 칸에 나눠
    # 보여줄 수 있도록 이미 계산된 같은 값(days_left/date_str)만 추가로
    # 쪼개서 반환한다 - 임계값/톤 판단 자체는 그대로다(새 Rule 아님).
    if days_left < 0:
        return {"text": f"마감({date_str})", "tone": "gray", "badge": None, "date": date_str}
    if days_left == 0:
        return {"text": f"오늘 마감({date_str})", "tone": "red", "badge": "오늘 마감", "date": date_str}
    if days_left <= 3:
        return {"text": f"D-{days_left}({date_str})", "tone": "red", "badge": f"D-{days_left}", "date": date_str}
    if days_left <= 7:
        return {"text": f"D-{days_left}({date_str})", "tone": "orange", "badge": f"D-{days_left}", "date": date_str}
    if days_left <= 30:
        return {"text": f"D-{days_left}({date_str})", "tone": "yellow", "badge": f"D-{days_left}", "date": date_str}
    return {"text": f"D-{days_left}({date_str})", "tone": "gray", "badge": f"D-{days_left}", "date": date_str}


# 대한민국 17개 광역 시/도 - 실제 행정구역 목록(추측/축약 아님).
_SIDO_NAMES = (
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
)
_SIDO_RE = re.compile("(" + "|".join(_SIDO_NAMES) + ")(?:특별시|광역시|특별자치시|특별자치도|도)?")
_DISTRICT_RE = re.compile(r"([가-힣]{1,4})(?:구|군|시)(?![가-힣])")


def normalize_location(location: str) -> str:
    """지역 정규화(2026-07-24) - "서울특별시 강남구 테헤란로107길 6" 같은
    도로명 주소를 "서울 강남"처럼 시/도 + 구/군/시 단위로만 줄인다.
    실제 candidate_jobs.location 샘플(도로명 주소/짧은 지명/영문 도시명/
    "재택"·"Remote" 등)을 확인해서 만든 규칙이다 - 구/군/시 표기가 없는
    지명("판교", "Seoul" 등)이나 한글이 아닌 값은 원문을 그대로 쓴다
    (모르는 지명을 지어내 축약하지 않는다). 여러 지역이 세미콜론/쉼표로
    나열된 값은 첫 구간만 보여준다."""
    text = (location or "").strip()
    if not text:
        return ""
    sido_match = _SIDO_RE.search(text)
    sido_name = sido_match.group(1) if sido_match else None
    rest = text[sido_match.end():] if sido_match else text
    district_match = _DISTRICT_RE.search(rest)
    if sido_name and district_match:
        return f"{sido_name} {district_match.group(1)}"
    if district_match:
        return district_match.group(1)
    if sido_name:
        return sido_name
    return re.split(r"[,;]", text)[0].strip()


def normalize_employment_type(job: dict) -> str:
    """고용형태(2026-07-24) - job_prep.extract_job_info()가 이미
    정규식으로 뽑아둔 "근무 형태"(정규직/계약직/인턴/파견직)를 그대로
    쓴다. 새 정규식을 만들지 않는다. 못 찾으면 "정보 없음"."""
    info = _extract_job_info(job.get("posting_text", "") or "")
    return info.get("근무 형태") or "정보 없음"


def build_meta_view(job: dict) -> dict:
    """공고 카드 Meta 5종(2026-07-25 재설계) - 고용형태/경력/지역/등록일/
    마감일(D-Day+날짜 통합). 전부 이미 있는 필드·함수 조합이고 새 판단은
    없다."""
    return {
        "employment_type": normalize_employment_type(job),
        "career": job.get("career_level") or "경력 무관",
        "location": normalize_location(job.get("location") or "") or "지역 미표기",
        "posted": (job.get("created_at") or "")[:10] or "등록일 정보 없음",
        "deadline": build_deadline_view(job),
    }


def _as_list(value) -> list[str]:
    """company_research 필드가 list/str/None 중 무엇이든 화면용
    list[str]로 통일한다(배치마다 형태가 살짝 다름 - 예: "미조사"라는
    문자열 하나, 빈 리스트, 실제 항목 리스트)."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    text = str(value).strip()
    if text in ("미조사", "정보 없음", ""):
        return []
    return [text]


def _hiring_values_keywords(raw) -> list[str]:
    """강조가치관 필드는 배치마다 키 이름이 조금씩 다르다(키워드/
    반복키워드/단일샘플) - 실제 반복 키워드만 뽑아 화면용 리스트로
    통일한다. 표본 수·집계상태 같은 메타 정보는 여기서 버린다(호출부가
    "n건 공고 기반"처럼 별도로 보여줄 때는 raw["표본"]을 직접 쓴다)."""
    if not isinstance(raw, dict):
        return _as_list(raw)
    for key in ("키워드", "반복키워드"):
        if raw.get(key):
            return [str(v) for v in raw[key] if v]
    if raw.get("단일샘플"):
        return [str(raw["단일샘플"])]
    return []


def build_company_info_view(job: dict) -> dict:
    """"회사 알아보기" 모달용 데이터(2026-07-25 재설계, 사용자 확정 -
    company_research 테이블 연결). 2026-07-24엔 "외부 Company Research는
    만들지 않는다"고 확정했었으나, 같은 날 이후 별도로 반자동 리서치
    (WebSearch + jd_semantic_objects culture-layer DB 집계, LLM 호출
    없음)로 company_research 테이블(Top100 기준 83개사)을 만들면서 그
    결정을 대체한다.

    company_research에 해당 회사가 있으면 그 데이터를 그대로 쓰고,
    없으면(아직 리서치 안 된 회사 - 전체 후보 pool 714개사 중 다수)
    기존 JD-only 방식(회사소개/산업/인재상 추출)으로 자연스럽게
    폴백한다 - 화면이 깨지지 않게. 정보가 없는 항목은 "정보 없음"으로
    채운다(추측하지 않음)."""
    company = job.get("company") or ""
    research = get_company_research(company)

    sections = _extract_display_sections(job)
    industry_fallback = job.get("industry") or _infer_industry(sections.get("other", ""))

    if research:
        return {
            "has_research": True,
            "company_size": research.get("회사규모") or "정보 없음",
            "founded_year": research.get("설립연도") or "정보 없음",
            "industry": research.get("산업") or industry_fallback or "정보 없음",
            "location": research.get("위치") or job.get("location") or "정보 없음",
            "intro": research.get("한줄소개") or _format_company_intro(job) or "정보 없음",
            "products_services": _as_list(research.get("주요서비스")),
            "business_model": research.get("비즈니스모델") or "정보 없음",
            "culture": _as_list(research.get("조직문화")) or _culture_highlights(job),
            "work_style": _as_list(research.get("일하는방식")),
            "benefits": _as_list(research.get("주요복지")),
            "hiring_values": _hiring_values_keywords(research.get("강조가치관")),
            "highlights": _as_list(research.get("주요이슈")),
            "quick_summary": _as_list(research.get("3줄요약")),
            "sources": _as_list(research.get("출처")),
            "updated_at": research.get("최종업데이트") or "",
            "job_url": job.get("url") or None,
        }

    # 폴백(company_research 없는 회사) - 예전 JD-only 뷰와 동일한 필드만
    # 채우고, 나머지(주요서비스/비즈니스모델/일하는방식/강조가치관/
    # 주요이슈/3줄요약)는 "아직 리서치 안 됨"을 뜻하는 빈 값으로 둔다.
    return {
        "has_research": False,
        "company_size": "정보 없음",
        "founded_year": "정보 없음",
        "industry": industry_fallback or "정보 없음",
        "location": job.get("location") or "정보 없음",
        "intro": _format_company_intro(job) or "정보 없음",
        "products_services": [],
        "business_model": "정보 없음",
        "culture": _culture_highlights(job),
        "work_style": [],
        "benefits": [],
        "hiring_values": [],
        "highlights": [],
        "quick_summary": [],
        "sources": [],
        "updated_at": "",
        "job_url": job.get("url") or None,
    }


_JD_BULLET_MARKER_RE = re.compile(r"^[\s•▪●\-\*■◆①-⑨]+|^\d+[\.\)]\s*")


_JD_DECORATIVE_BRACKET_RE = re.compile(r"[＜＞<>]")


def _split_jd_bullets(text: str) -> list[str]:
    """Rule Representation 텍스트(task/qualification 원문 블록)를 불릿
    단위 문장으로 나눈다. LLM 미사용 - 순수 문자열 처리.

    2026-07-25 - "＜ OOO가 담당할 업무에요 ＞" 같은 장식용 괄호 안내
    문구가 실제 업무 원본 텍스트 안에 섞여 있는 경우가 있다(Header
    Detector가 진짜 헤더로 인식하지 않는 문장 - Known Limitation과
    같은 계열). 실제 불릿(마커로 시작)이 아니면서 이런 괄호를 포함한
    줄은 안내 문구로 보고 제외한다 - 새로운 판단이 아니라 표시 단계의
    노이즈 필터일 뿐이다."""
    if not text:
        return []
    bullets = []
    for line in text.split("\n"):
        stripped = line.strip()
        is_marked_bullet = bool(_JD_BULLET_MARKER_RE.match(stripped))
        cleaned = _JD_BULLET_MARKER_RE.sub("", line).strip()
        if len(cleaned) < 4:
            continue
        if not is_marked_bullet and _JD_DECORATIVE_BRACKET_RE.search(cleaned):
            continue
        bullets.append(cleaned)
    return bullets


_TASK_TERM_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"cross-functional", "협업"),
    (r"stakeholders?", "이해관계자"),
    (r"business", "사업"),
    (r"sales", "영업"),
    (r"infra(structure)?", "인프라"),
    (r"pipeline", "파이프라인"),
    (r"dashboard", "대시보드"),
    (r"growth", "성장"),
    (r"insights?", "인사이트"),
    (r"report(ing)?", "리포트"),
    (r"process", "프로세스"),
    (r"data", "데이터"),
    (r"operations?", "운영"),
    (r"marketing", "마케팅"),
    (r"product", "프로덕트"),
    (r"customer", "고객"),
    (r"strategy", "전략"),
    (r"planning", "기획"),
    (r"quality", "품질"),
    (r"management", "관리"),
    (r"analysis", "분석"),
]
_TASK_TERM_RE = [(re.compile(rf"\b{pat}\b", re.IGNORECASE), repl) for pat, repl in _TASK_TERM_SUBSTITUTIONS]

_TASK_FILLER_WORDS = {
    "및", "등", "위한", "위해", "통한", "통해", "기반", "관련", "전반",
    "그리고", "또는", "혹은", "대한", "대해", "관하여", "수행", "따라",
}

# "Sales Business"/"Business Sales"처럼 붙어 나오는 중복 치환 결과를
# 하나로 합친다(둘 다 그대로 두면 "영업 사업"처럼 같은 뜻이 겹쳐 보임) -
# 새 판단이 아니라 순수 문자열 중복 제거.
_TASK_REDUNDANT_PAIR_RE = re.compile(r"(영업\s+사업|사업\s+영업)")

# 조사 제거 - 긴 조사(에서/으로)를 먼저 시도해야 짧은 조사(로/에)가
# 중간에 먼저 매치되는 걸 막는다. 2글자 이하 토큰은 건드리지 않는다
# (build_condensed_task_phrase에서 len(t) > 2로 가드 - "성과"처럼
# 조사로 착각하기 쉬운 2글자 명사를 지키기 위함). "은"/"는"은 목록에서
# 뺐다 - 토픽 조사보다 "지탱하는"/"만드는"처럼 동사 관형형 어미로 쓰인
# 경우가 더 많아서, 무조건 지우면 "지탱하"처럼 동사 어간이 잘려나갔다
# (2026-08-01, 실측 확인).
_TASK_TRAILING_PARTICLE_RE = re.compile(r"(에서|으로|을|를|이|가|의|로|에|과|와)$")

_TASK_SENTENCE_ENDING_RE = re.compile(
    r"(합니다|한다|해요|됩니다|된다|하며|하고|하여|해서|드립니다|해주세요|드려요|함)$"
)

_TASK_CLAUSE_SPLIT_RE = re.compile(r"[,;·.\n]")


def _condense_task_phrase(text: str, max_len: int = 18) -> str:
    """공고 카드 "주요 직무" 표시 전용 축약(2026-08-01, 사용자 확정 -
    "목록에서 JD 원문 문장을 그대로 보여주지 않는다, 3초 안에 읽히는
    동사+명사 수준으로"). LLM 없이 문자열 치환/정리만 한다 - Rule
    Representation의 task 원문 자체(rep["task"])는 바꾸지 않는다(M3
    채점 등 다른 소비처는 원문을 그대로 쓴다) - 이 함수는 카드에 보여줄
    표시값만 따로 만든다.

    한계(사용자와 합의된 부분, 2026-08-01) - "Business Data Infra
    고도화" -> "데이터 인프라 관리" 같은 진짜 의미 재구성(어떤 단어가
    핵심이고 어떤 수식어를 버릴지 판단)은 LLM 없이는 못 한다. 여기서는
    기계적으로 가능한 선까지만 한다: (1) 흔한 영어 JD 용어를 한국어로
    치환, (2) 접속어/조사 제거, (3) 첫 절만 사용, (4) 문장 어미를
    명사형으로 정리, (5) 길이 컷. 원문과 100% 같은 결과를 보장하지
    않는다 - 그래도 원문 문장 전체를 그대로 보여주는 것보다는 짧다."""
    if not text:
        return ""
    s = text.strip()
    for pattern, repl in _TASK_TERM_RE:
        s = pattern.sub(repl, s)
    s = _TASK_REDUNDANT_PAIR_RE.sub("영업", s)
    # 괄호 안 예시/부연설명("장기(전략) 클라이언트", "카테고리(식품, 리빙
    # 등)")은 짧은 표시 문구에서 군더더기다 - 통째로 지운다. 길이 컷이
    # 괄호를 반쯤 자르고 남은 짝 없는 괄호도 마저 지운다(2026-08-01,
    # 실측 확인 - "카테고리(식품"처럼 닫는 괄호 없이 잘린 조각이 남았었다).
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[()]", " ", s)
    clause = _TASK_CLAUSE_SPLIT_RE.split(s, maxsplit=1)[0].strip()
    if len(clause) < 4:
        clause = s
    tokens = [t for t in clause.split() if t not in _TASK_FILLER_WORDS]
    # 영어 용어 치환 후 원문에 이미 있던 한국어 단어와 겹치는 경우가
    # 있다(예: "Cross-functional 협업" -> "협업 협업") - 바로 옆에 같은
    # 토큰이 연속되면 하나만 남긴다.
    tokens = [t for i, t in enumerate(tokens) if i == 0 or t != tokens[i - 1]]
    tokens = [_TASK_TRAILING_PARTICLE_RE.sub("", t) if len(t) > 2 else t for t in tokens]
    tokens = [t for t in tokens if t]
    condensed = " ".join(tokens)
    condensed = _TASK_SENTENCE_ENDING_RE.sub("", condensed).strip()
    if condensed.endswith("하") and len(condensed) > 1:
        condensed = condensed[:-1]
    condensed = re.sub(r"\s+", " ", condensed).strip()
    if len(condensed) < 2:
        condensed = re.sub(r"\s+", " ", text.strip())
    if len(condensed) > max_len:
        # 한국어(SOV - 동사가 문장 끝)와 영어(SVO - 동사가 앞쪽)는 핵심이
        # 놓이는 위치가 반대다. 한글이 있으면 뒤(핵심 동사/명사, 예:
        # "관리"/"구축"/"분석")를 남기고 앞의 수식어부터 자르고
        # (2026-08-01, 실측 확인 - 앞에서부터 글자 수로 자르면 "동사+
        # 명사로 끝나야 한다"는 원칙이 깨졌다), 한글이 없는 영문 JD
        # (예: 특정 게임사 등 해외 사업부 공고)는 반대로 앞을 남기고 뒤를
        # 자른다 - 그대로 뒤를 남기면 "portfolio is large"처럼 의미
        # 없는 문장 조각만 남는다(2026-08-01, 실측 확인).
        words = condensed.split()
        has_hangul = bool(re.search(r"[가-힣]", condensed))
        if has_hangul:
            while len(words) > 1 and len(" ".join(words)) > max_len:
                words.pop(0)
        else:
            while len(words) > 1 and len(" ".join(words)) > max_len:
                words.pop()
        condensed = " ".join(words)
        if len(condensed) > max_len:
            condensed = condensed[:max_len].strip() if has_hangul else condensed[-max_len:].strip()
    return condensed.strip()


def build_rule_task_view(job: dict, max_n: int = 3) -> tuple[list[str], int]:
    """공고 카드 "주요 업무"(목록 전용) - Rule Representation
    (rule_representation.build_rule_based_representation)의
    task_lines 중 task_signal=True인 줄만 쓴다. LLM/jd_semantic_objects
    미사용 - 목록 단계는 크롤링 직후 바로 채워져야 한다는 원칙(사용자
    확정)에 따른 것. 분석 화면(S1D)의 build_core_tasks_view(jd_semantic_
    objects 기반)와는 별개 함수다 - 분석 화면은 여전히 LLM 결과를
    보여준다.

    2026-08-14(사용자 확정, 실측 검증 결과 - docs/verification/
    2026-08-14_list_rule_task_qual_check) - 이전엔 task_signal로
    거르지 않고 섹션 원문 줄을 그대로 가져다 _condense_task_phrase()로
    축약했는데, 그 결과 (1) 컬처/헤더 문구가 필터 없이 "주요 업무"에
    섞여 들어가고(예: 회사 철학 선언문이 업무처럼 보임), (2) 축약
    함수의 절단(clause-split+18자 컷)이 정상 문장까지 주어 없는 조각
    으로 망가뜨리는 문제(예: "execution", "수 있는 실행기 서비스
    제공")가 30건 실측에서 반복 확인됐다. task_signal(Frozen v1.0,
    F1 0.88)로 거르는 것만으로 두 문제 모두 해소됨을 30건으로 확인
    했다 - 새 Rule/새 LLM 없이 이미 계산되어 있던 필드를 켜기만 한
    것이다. 축약도 더 이상 하지 않는다 - signal을 통과한 원문 문장을
    그대로 반환하고, 길이 제한(짧게 보이게)은 화면(UI) 쪽 책임으로
    넘긴다(여기서 의미를 다시 판단해 잘라내지 않는다).

    3개를 채우려고 signal=False인 줄을 끌어오지 않는다 - 0~max_n개,
    있는 만큼만 반환한다(사용자 확정 - "없는 걸 채우지 않는다"). 반환
    값은 (표시할 문구, 남은 개수) - 남은 개수는 카드에서 안 써도 된다."""
    rep = build_rule_based_representation(job)
    seen: set[str] = set()
    filtered: list[str] = []
    for ln in rep["task_lines"]:
        if not ln["task_signal"]:
            continue
        # 화면이 자체 불릿(•)을 그리므로 원문 마커만 벗긴다(_split_jd_bullets와
        # 같은 정규식 재사용 - 새 정리 규칙 아님, 텍스트 내용 자체는 안 바꿈).
        text = _JD_BULLET_MARKER_RE.sub("", ln["text"]).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        filtered.append(text)
    return filtered[:max_n], max(0, len(filtered) - max_n)


def build_rule_skill_view(job: dict, max_n: int = 4) -> tuple[list[str], int]:
    """공고 카드 "필수 기술"(목록 전용) - Rule Representation의 skill만
    쓴다(콤마로 이어진 문자열을 태그 리스트로 분리). (표시할 태그,
    넘치는 개수) 튜플을 반환한다. LLM 미사용."""
    rep = build_rule_based_representation(job)
    skills = [s.strip() for s in rep["skill"].split(",") if s.strip()]
    return skills[:max_n], max(0, len(skills) - max_n)


def build_rule_requirement_view(job: dict, max_n: int = 3) -> tuple[list[str], int]:
    """공고 카드 "자격 요건"(목록 전용) - Rule Representation의
    qualification_lines 중 qualification_signal=True인 줄만 쓴다.
    LLM 미사용.

    2026-08-14(사용자 확정, 실측 검증 - build_rule_task_view와 동일한
    근거/같은 날 검증) - 필터 없이 원문 섹션을 그대로 쓰면 불릿 마커가
    없는 일부 공고에서 근무조건/급여(예: "월 급여 240만원")나 기술
    스택 나열까지 "자격 요건"처럼 통째로 섞여 들어가는 사례가 30건
    중 2건 확인됐다(플리토, S2W). qualification_signal(Frozen v1.0,
    F1 0.61)로 거르면 이런 사례는 안전하게 빈 결과로 줄어든다(잘못된
    내용을 보여주는 것보다 낫다). max_n을 3으로 올렸다(기존 2 -
    참조 레이아웃의 "주요 업무 3개/자격 요건 3개" 대칭 구성에 맞춤).
    3개를 채우려고 signal=False 줄을 끌어오지 않는다 - 0~max_n개,
    있는 만큼만."""
    rep = build_rule_based_representation(job)
    seen: set[str] = set()
    filtered: list[str] = []
    for ln in rep["qualification_lines"]:
        if not ln["qualification_signal"]:
            continue
        text = _JD_BULLET_MARKER_RE.sub("", ln["text"]).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        filtered.append(text)
    return filtered[:max_n], max(0, len(filtered) - max_n)


def build_job_card_view(job: dict) -> dict:
    """공고 목록 카드(render_s1) ViewModel(2026-07-24, 사용자 요청 -
    "render_s1은 출력만 한다"). render_job_card()는 이 함수가 반환하는
    값만 출력하고, job(candidate dict)을 직접 들여다보지 않는다 -
    나중에 카드 데이터가 늘어나도 이 함수만 고치면 된다.

    2026-07-25(사용자 확정) - task_summary/skill_summary/requirement_
    summary를 jd_semantic_objects(LLM) 대신 Rule Representation(LLM
    미사용)으로 채운다. 목록은 크롤링 직후 바로 완성되어야 하고, LLM은
    사용자가 [분석하기]를 누른 공고 1건에만 써야 한다는 원칙에 따른
    것 - "AI 분석 준비 중" 대기 상태 자체가 목록에서 없어진다.

    2026-08-14 세 번째 수정(사용자 확정, 레퍼런스 이미지 기준 재구성) -
    처음엔 task_summary/requirement_summary를 3개씩 대칭으로, 그 다음엔
    자격요건만 빼고 주요 업무 2개를 남겼었는데, task_signal/
    qualification_signal은 "이 문장이 업무/자격요건이냐"를 판별할 뿐
    목록에 맞는 짧은 요약을 만들어주지 않는다는 게 실측으로 반복
    확인됐다(그건 의역/요약 문제 - Rule이나 LLM을 새로 더 만들지 않기로
    확정). 최종적으로 주요 업무도 목록 카드에서 뺀다 - 카드는 로고 +
    회사/직무/메타/스킬(사실 정보)만 보여주고, 문장형 내용(업무/자격
    요건) 판단은 전부 [분석하기] 이후 LLM Understanding+Linking 화면의
    책임으로 넘긴다. build_rule_task_view() 자체는 지우지 않았다 -
    다른 화면이 필요하면 그대로 재사용 가능하다."""
    skill_summary, skill_overflow = build_rule_skill_view(job, max_n=4)
    return {
        "job_id": job.get("job_id"),
        "company": job.get("company") or "",
        "title": job.get("title") or "",
        "meta": build_meta_view(job),
        "career_status": job.get("career_status", "확인필요"),
        "skill_summary": skill_summary,
        "skill_overflow": skill_overflow,
        "url": job.get("url") or None,
    }


def build_match_reason_breakdown(job: dict) -> list[dict]:
    """"왜 추천됐는가"를 레이어별로 "이 레이어 요구사항 중 몇 %가
    match/partial_match로 충족됐는가"로 분해한다(2026-08-06, S3 크래시
    수정 겸 스키마 마이그레이션 - 사용자 확정).

    옛 구현은 semantic_matching의 ranking_score 공식(weight *
    engine_confidence의 가중 평균)을 레이어별로 쪼갠 것이었는데,
    Semantic Linking(v1) 스키마에는 engine_confidence 같은 연속값이
    아예 없다(judge_engine.py도 이미 숫자 점수를 버림 - PROJECT_
    HANDBOOK.md "설명되지 않는 숫자를 만들지 않는다" 원칙). 여기서
    새 숫자를 지어내는 대신, 바로 아래 "핵심 요구사항 충족률"과 같은
    카운트 방식(충족 개수/전체 개수)을 레이어 단위로만 쪼갠다 - 이미
    쓰고 있는 계산 방식의 재사용일 뿐 새 판단 기준이 아니다.
    `job["_semantic_match"]["links"]`가 없으면 빈 리스트."""
    sm = job.get("_semantic_match") or {}
    links = sm.get("links") or []
    if not links:
        return []

    layer_total: dict[str, int] = {}
    layer_matched: dict[str, int] = {}
    for link in links:
        layer = link.get("layer", "")
        layer_total[layer] = layer_total.get(layer, 0) + 1
        if link.get("relation") in ("match", "partial_match"):
            layer_matched[layer] = layer_matched.get(layer, 0) + 1

    breakdown = [
        {
            "layer": layer, "label": _LAYER_KOR_LABEL.get(layer, layer),
            "pct": round(layer_matched.get(layer, 0) / total * 100) if total else 0,
        }
        for layer, total in layer_total.items()
    ]
    breakdown.sort(key=lambda b: -b["pct"])
    return breakdown


def build_core_tasks_label(job: dict, max_n: int = 3) -> str | None:
    """Rule 3 결과를 "·"로 이어붙인 한 줄(① 섹션 "핵심 업무" 표시용)."""
    tasks = build_core_tasks_view(job, max_n=max_n)
    return " · ".join(tasks) if tasks else None


def build_gaps_split_view(job: dict, max_n: int = 6) -> tuple[list[str], list[str]]:
    """③ 보완하면 좋은 점 - 기존 build_strengths_gaps_view의 gaps를
    requirement_type으로 "필수 부족 역량"(required/responsibility)과
    "있으면 좋은 역량"(preferred)으로 나눈다. 새 판단이 아니라 이미
    Match Object가 정한 gap 목록을 jd_obj.requirement_type(이미 있는
    필드)으로 다시 나누기만 한다."""
    match_result = job.get("_semantic_match") or {}
    matches = match_result.get("matches", [])
    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}

    required_gaps, preferred_gaps = [], []
    for m in matches:
        if m.get("status") == "match":
            continue
        jd_obj = jd_by_id.get(m["jd_object"], {})
        label = _short_label(jd_obj)
        if not label:
            continue
        if jd_obj.get("requirement_type") == "preferred":
            preferred_gaps.append(label)
        else:
            required_gaps.append(label)
    return required_gaps[:max_n], preferred_gaps[:max_n]


# ── ② AI 의미 기반 매칭 - 카드 명세 (2026-07-19, 사용자 상세 명세) ────
# "JD 요구사항을 늘어놓는 게 아니라, 요구 하나와 그 요구에 연결된 내
# 경험 하나를 1:1로 보여주는 카드"를 만든다. 전부 이미 계산된 Link
# 필드의 조합/번역일 뿐 새 판단이 아니다(새 LLM 호출 없음).
_MATCH_LEVEL_LABEL = {
    "direct": "직접 일치", "strong": "강한 경험", "similar": "유사 경험",
    "partial": "부분 경험", "needs_evidence": "경험 확인 필요",
}
_MATCH_LEVEL_TONE = {
    "direct": "navy", "strong": "blue", "similar": "pink",
    "partial": "yellow", "needs_evidence": "red",
}


def _match_level(relation: str, match_strength: str) -> str:
    """relation/match_strength만으로 5단계 중 하나를 고른다(2026-08-13,
    link_engine.py 스키마 전환 - 옛 match_type/matched_fields 개수 기반
    규칙을 대체, 의도는 동일하게 유지: 확실한 일치일수록 상위 단계).
    이 함수는 matched_resume_objects가 있는 항목에만 호출되므로
    (build_section2_cards가 그 전에 필터링) relation은 match 또는
    partial_match만 온다 - needs_evidence는 방어적 기본값이다."""
    if relation == "match" and match_strength == "strong":
        return "direct"
    if relation == "match":
        return "strong"
    if relation == "partial_match" and match_strength in ("strong", "moderate"):
        return "similar"
    if relation == "partial_match" and match_strength == "weak":
        return "partial"
    return "needs_evidence"


def _jd_point_text(jd_obj: dict, max_len: int = 35) -> str:
    """"회사가 원하는 것" - frame.action+object를 우선 조합하고, 없으면
    normalized_text를 max_len자로 축약한다(사용자 지정 규칙 - 새로
    추론하지 않는다, 있는 필드를 그대로 잇거나 자를 뿐)."""
    frame = jd_obj.get("frame") or {}
    action, obj = (frame.get("action") or "").strip(), (frame.get("object") or "").strip()
    text = f"{obj} {action}".strip() if (action and obj) else _short_label(jd_obj)
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _resume_point(r_obj: dict) -> tuple[str, str]:
    """(프로젝트/출처 라벨, 수행 내용). qualification(자격/교육)
    Object는 project인 것처럼 보이면 안 된다(사용자 지정 규칙) - evidence.
    section(예: "대외활동", "자격증")을 라벨로 써서 실무 경험과 구분한다.
    skill Object는 "SQL한 경험"처럼 명사+"한"이 어색해지므로 "SQL 활용"
    형태로 조사를 붙여 반환한다(카드 설명 문장의 "~{content}한 경험"
    템플릿과 자연스럽게 맞물리도록)."""
    evidence = r_obj.get("evidence") or {}
    layer = r_obj.get("layer", "")
    if layer == "qualification":
        label = evidence.get("section") or "자격/교육"
    else:
        label = evidence.get("project") or evidence.get("section") or "이력서"
    content = _short_label(r_obj) or ""
    if layer == "skill" and content:
        content = f"{content}{_josa(content, '을', '를')} 활용"
    return label, content


def _connected_fields_text(matched_dimensions: list[str], max_n: int = 2) -> str:
    shown = matched_dimensions[:max_n]
    return ", ".join(_FIELD_LABEL.get(f, f) for f in shown)


def _card_description(jd_text: str, project: str, resume_text: str, field_text: str, level: str) -> str:
    """카드 설명 문장 - 사용자 지정 템플릿. 근거가 약하면(partial/
    needs_evidence) 반드시 약하게 쓴다 - "직접 연결됩니다" 같은 강한
    단정을 쓰지 않는다(특정 해외 브랜드류 FP를 강한 매칭처럼 설명하지 않기 위함)."""
    jd_josa = _josa(jd_text, "을", "를")
    if level in ("direct", "strong", "similar"):
        tail = f"{field_text} 측면에서 연결됩니다." if field_text else "의미상 연결됩니다."
        return f"이 공고는 {jd_text}{jd_josa} 요구합니다. 이력서의 {project}에서 {resume_text}한 경험이 {tail}"
    if level == "partial":
        tail = f"{field_text}" if field_text else "전반적인 맥락"
        return f"이 공고는 {jd_text}{jd_josa} 요구합니다. 이력서의 {project}에서 {resume_text}한 경험이 {tail} 측면에서만 부분적으로 연결됩니다."
    return f"이 공고는 {jd_text}{jd_josa} 요구합니다. 이력서의 {project}에 관련 내용({resume_text})이 있지만, 구체적으로 연결되는 부분은 확인되지 않았습니다."


def build_section2_cards(job: dict, resume_objects: list[dict]) -> dict:
    """② AI 의미 기반 매칭 카드. `_semantic_match.links` 중 실제로
    연결된(matched_resume_objects가 있는) JD Object만 대상으로 한다 -
    연결이 아예 없는 요구사항은 ③(build_gaps_detail_view)의 몫이다.
    우선순위 점수로 정렬해 기본 노출 5개 + 나머지("더보기")로 나눈다.
    반환: {"default": [...], "more": [...]} - 각 원소는 jd_point/
    resume_project/resume_point/connected_fields/level/level_label/
    level_tone/description.

    2026-08-13 - relation/match_strength/matched_dimensions(link_engine.py
    실제 스키마)로 옛 match_type/matched_fields/engine_confidence 기반
    로직을 대체했다(옛 필드는 새 스키마에 아예 없어 항상 빈 값/0으로
    읽혀서, 이 함수가 모든 카드를 최하위 등급("경험 확인 필요", 빨강)
    으로 표시하던 조용한 버그가 있었다)."""
    match_result = job.get("_semantic_match") or {}
    matches = match_result.get("links", [])
    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}
    resume_by_id = {(o.get("id") or o.get("normalized_text")): o for o in resume_objects}

    cards = []
    seen_skill_text: set[str] = set()
    seen_pt_resume: set[str] = set()
    for m in matches:
        if not m.get("matched_resume_objects"):
            continue
        jd_obj = jd_by_id.get(m["jd_object_id"], {})
        best = m["matched_resume_objects"][0]
        r_obj = resume_by_id.get(best.get("resume_object_id"), {})
        matched_dimensions = m.get("matched_dimensions") or []
        layer = jd_obj.get("layer", "")
        relation = m.get("relation", "")
        match_strength = m.get("match_strength", "")

        level = _match_level(relation, match_strength)
        jd_text = _jd_point_text(jd_obj)
        project, resume_text = _resume_point(r_obj)
        field_text = _connected_fields_text(matched_dimensions)
        description = _card_description(jd_text, project, resume_text, field_text, level)

        jd_has_evidence = bool((jd_obj.get("evidence") or {}).get("source_text"))
        resume_has_evidence = bool((r_obj.get("evidence") or {}).get("source_text"))

        # 제외 규칙(사용자 지정) - 완전히 지우지 않고 "더보기"로 보낸다.
        is_dup = False
        if layer == "skill":
            key = jd_obj.get("normalized_text", "")
            is_dup = key in seen_skill_text
            seen_skill_text.add(key)
        elif layer in ("problem", "thinking"):
            key = best.get("resume_object_id", "")
            is_dup = bool(key) and key in seen_pt_resume
            if key:
                seen_pt_resume.add(key)
        excluded = (
            (not matched_dimensions and match_strength == "weak")
            or not (jd_has_evidence and resume_has_evidence)
            or is_dup
        )

        # 노출 우선순위(사용자 지정) - critical 필수 > Skill 강한 일치 >
        # Task 강한 근거 > Problem/Thinking 실제 축 겹침 > 우대사항.
        priority = 0.0
        if jd_obj.get("importance") == "critical":
            priority += 100
        if layer == "skill" and relation == "match" and match_strength == "strong":
            priority += 50
        if layer == "task" and level in ("direct", "strong", "similar"):
            priority += 40
        if layer in ("problem", "thinking") and matched_dimensions:
            priority += 30
        if jd_obj.get("requirement_type") == "preferred":
            priority += 10
        priority += {"strong": 2, "moderate": 1, "weak": 0}.get(match_strength, 0)  # 동률 완화용

        cards.append({
            "jd_point": jd_text,
            "resume_project": project,
            "resume_point": resume_text,
            "connected_fields": field_text,
            "level": level,
            "level_label": _MATCH_LEVEL_LABEL[level],
            "level_tone": _MATCH_LEVEL_TONE[level],
            "description": description,
            "_priority": priority,
            "_excluded": excluded,
        })

    cards.sort(key=lambda c: -c["_priority"])
    default_cards = [c for c in cards if not c["_excluded"]][:5]
    default_ids = {id(c) for c in default_cards}
    more_cards = [c for c in cards if id(c) not in default_ids]
    for c in cards:
        c.pop("_excluded", None)
        c.pop("_priority", None)
    return {"default": default_cards, "more": more_cards}


# ── ③ 보완하면 좋은 점 - 상세 카드 (2026-07-19, 사용자 상세 명세) ─────

def _requires_practical_evidence(jd_obj: dict) -> bool:
    expected = jd_obj.get("expected_evidence") or []
    return any(e.get("type") in ("experience", "project") for e in expected)


def build_gaps_detail_view(job: dict, max_n: int = 4) -> dict:
    """③ 보완하면 좋은 점 - 상세 카드. "②는 연결된 근거만, ③은 연결
    되지 않은 요구만 보여준다"(사용자 지정 원칙) - matched_resume_
    objects가 하나라도 있으면(status가 "match"든 약한 "candidate"든)
    이미 ②(build_section2_cards)에 카드로 나오므로 ③에서는 완전히
    제외한다(중복 방지, 실측으로 확인한 버그: 처음엔 status!="match"만
    걸렀더니 candidate 항목이 ②③ 양쪽에 다 나왔음). ③은 matched_
    resume_objects가 아예 없는(연결 자체가 없는) JD Object만 다룬다.

    판정은 실무 증거(expected_evidence type=experience/project)를
    요구하는지로만 가른다 - 요구하면 "미충족", 아니면(도구/지표 수준
    요구라 키워드 인식 문제일 수도 있음) "확인 필요"로 단정을 낮춘다.
    "현재 이력서"는 항상 "관련 근거를 확인하지 못함"이라고만 쓴다 -
    "경험 없음"으로 단정하지 않는다(사용자 지정 규칙)."""
    match_result = job.get("_semantic_match") or {}
    matches = match_result.get("matches", [])
    jd_by_id = {(o.get("id") or o.get("normalized_text")): o for o in _jd_objects(job)}

    required, preferred = [], []
    for m in matches:
        if m.get("matched_resume_objects"):
            continue  # 연결된 근거가 있으면 ②의 몫 - ③에서 제외
        jd_obj = jd_by_id.get(m["jd_object"], {})
        title = _short_label(jd_obj)
        if not title:
            continue
        ask = jd_obj.get("normalized_text") or (jd_obj.get("evidence") or {}).get("source_text") or title
        verdict = "미충족" if _requires_practical_evidence(jd_obj) else "확인 필요"

        card = {
            "title": title,
            "ask": ask,
            "current": "관련 근거를 확인하지 못함",
            "importance": "우대" if jd_obj.get("requirement_type") == "preferred" else "필수",
            "verdict": verdict,
        }
        if jd_obj.get("requirement_type") == "preferred":
            preferred.append(card)
        else:
            required.append(card)

    return {
        "required": required[:max_n], "required_more": required[max_n:],
        "preferred": preferred[:max_n], "preferred_more": preferred[max_n:],
    }


# ── ④ 공고 상세 정보 - 우대사항 분류 / 자격요건 불릿 분리 / 복지 분류
# (2026-07-19, 사용자 상세 명세) ────────────────────────────────────
_LANGUAGE_KEYWORDS = ["영어", "중국어", "일본어", "스페인어", "프랑스어", "독일어", "베트남어"]

_WELFARE_CATEGORY_KEYWORDS = {
    "근무환경": ["유연근무", "재택", "원격", "자율출퇴근", "자율 출퇴근", "선택적 근로"],
    "성장지원": ["교육비", "도서", "세미나", "컨퍼런스", "자기계발", "학회"],
    "생활지원": ["식대", "간식", "건강검진", "경조사", "통근", "주차"],
    "휴가": ["리프레시", "생일휴가", "연차", "안식", "휴가"],
    "보상": ["성과급", "스톡옵션", "인센티브", "상여금", "포상"],
}


def build_preferred_classified_view(job: dict) -> dict:
    """우대사항을 기술/경험/도메인/언어로 분류한다. requirement_type=
    "preferred" JD Object를 층(layer)과 normalized_text 키워드로 대충
    나눈다 - 새 텍스트를 만들지 않고 이미 있는 Object를 재배열만 한다."""
    buckets: dict[str, list[str]] = {"기술": [], "경험": [], "도메인": [], "언어": []}
    seen: set[str] = set()
    for obj in _jd_objects(job):
        if obj.get("requirement_type") != "preferred":
            continue
        label = _short_label(obj)
        if not label or label in seen:
            continue
        seen.add(label)
        layer = obj.get("layer", "")
        if any(lang in label for lang in _LANGUAGE_KEYWORDS):
            buckets["언어"].append(label)
        elif layer == "skill":
            buckets["기술"].append(label)
        elif layer == "qualification":
            buckets["경험"].append(label)
        else:
            domain = (obj.get("frame") or {}).get("domain", "")
            buckets["도메인" if domain else "경험"].append(label)
    return {k: v for k, v in buckets.items() if v}


_BULLET_PREFIX_RE = re.compile(r"^[•●▪◦\-\*]\s*")
_QUAL_SPLIT_RE = re.compile(r"[,、;/]|(?:이며|하며|하고)\s")


def split_qualification_bullets(text: str, max_n: int = 6) -> list[str]:
    """자격요건 긴 문장을 의미 단위 불릿으로 쪼갠다(간단한 구분자 기반
    - 완벽한 문장 분석이 아니라 최선 노력 수준의 분리임을 명시). 문장을
    그대로 복붙하지 않는다는 사용자 지정 규칙에 대응.

    원문이 이미 줄바꿈/불릿마커("•" 등)로 나뉜 리스트면 그 구조를
    그대로 존중한다(줄 안에서 다시 쪼개지 않음) - "·"는 split 대상에서
    뺐다("데이터 추출·분석"처럼 한 항목 안의 복합명사 연결에도 흔히
    쓰여서, 분리하면 오히려 항목이 어색하게 잘린다는 걸 실측으로
    확인함). 줄바꿈이 없는 한 문단짜리 긴 문장일 때만 쉼표/접속어
    기준으로 최소한만 쪼갠다."""
    if not text:
        return []
    lines = [_BULLET_PREFIX_RE.sub("", ln).strip(" .") for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 2:
        return lines[:max_n]
    single = lines[0] if lines else text.strip()
    parts = [p.strip(" .") for p in _QUAL_SPLIT_RE.split(single) if p and p.strip(" .")]
    return parts[:max_n] if parts else [single[:80]]


def build_welfare_classified_view(welfare_text: str) -> dict:
    """복리후생 원문 블록을 근무환경/성장지원/생활지원/휴가/보상으로
    분류한다. 분류 안 되는 항목은 "기타"에 남긴다(지어내지 않음)."""
    if not welfare_text:
        return {}
    items = split_qualification_bullets(welfare_text, max_n=20)
    buckets: dict[str, list[str]] = {k: [] for k in _WELFARE_CATEGORY_KEYWORDS}
    etc: list[str] = []
    for item in items:
        matched_cat = None
        for cat, kws in _WELFARE_CATEGORY_KEYWORDS.items():
            if any(kw in item for kw in kws):
                matched_cat = cat
                break
        (buckets[matched_cat] if matched_cat else etc).append(item)
    result = {k: v for k, v in buckets.items() if v}
    if etc:
        result["기타"] = etc
    return result


# ── STEP1(이력서 업로드 화면) "내 이력서 한눈에 보기" ────────────────────
# 2026-07-23 추가 - Resume Understanding 결과(semantic_objects)만 사용한다.
# resume-side importance 어휘는 JD-side(critical/core/normal/low)와 달리
# core/major/minor다(understanding.py:515) - 그래서 _IMPORTANCE_RANK를
# 재사용하지 않고 별도 랭크 테이블을 둔다.
_RESUME_IMPORTANCE_RANK = {"core": 0, "major": 1, "minor": 2}


def _resume_importance_rank(obj: dict) -> int:
    return _RESUME_IMPORTANCE_RANK.get(obj.get("importance"), 1)


_SKILL_SYNTAX_FRAGMENT_RE = re.compile(r"(문|함수|절|구문|연산자)$")
_LEADING_TOOL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.#_-]*")


def _competency_clause(obj: dict) -> str | None:
    """핵심역량 문장 - Thinking Object의 frame.action/object를 "~하는
    역량" 틀로 조립한다(2026-07-23, 사용자 지적 반영). Problem 레이어의
    normalized_text를 그대로 쓰면 "전체 에어비앤비 숙소의 비활성 상태로
    인한 시장 지표 왜곡"처럼 프로젝트 문맥/문제 상황이 그대로 노출된다 -
    이건 "이 사람이 어떤 방식으로 문제를 해결하는가"라는 핵심역량의
    목적과 다르다. Thinking 레이어는 접근 방식/사고방식을 담고 있어
    action+object 조합이 곧 "역량 문장"이 된다(_thinking_clause와 동일한
    조립 방식 - 새 문장을 짓는 게 아니라 이미 있는 두 필드를 정해진
    틀에 넣을 뿐이다). action/object가 비어 있으면 normalized_text로
    폴백한다(그래도 Thinking 레이어 안에서만 고른다)."""
    clause = _thinking_clause(obj)
    if not clause:
        return None
    frame = obj.get("frame") or {}
    if frame.get("action") and frame.get("object"):
        # LLM 이 action 을 "전환하다"(동사 원형)로 뽑으면 "전환하다하는 역량"
        # 이 되므로, "~하는 역량" 틀에 넣기 전에 끝의 "하다"만 뗀다
        # ("전환하다" -> "전환하는 역량", 명사형 "전환"은 그대로).
        return f"{re.sub(r'하다$', '', clause)}하는 역량"
    return clause


def build_resume_summary_view(
    resume_objects: list[dict], max_competencies: int = 4, max_skills: int = 8, max_experiences: int = 3,
) -> dict:
    """STEP1 "내 이력서 한눈에 보기" 카드 3개(핵심 역량/주요 기술/대표
    경험) - Resume Understanding 결과만 쓴다. 새 문장을 생성하거나
    추론하지 않는다 - 이미 있는 구조화 필드(frame.action/object/method,
    normalized_text)를 정해진 틀로 조립하거나 그대로 뽑을 뿐이다
    (2026-07-23, 사용자 지적 반영 - 핵심역량에 Problem 레이어의 프로젝트
    문맥이 그대로 노출되고, 주요기술에 "WITH문"/"WINDOW 함수" 같은 SQL
    문법 조각이 섞여 나온 문제를 고쳤다).

    - 핵심 역량: thinking 레이어만 사용한다(problem/task 제외 - Problem은
      "무슨 문제가 있었는가"라는 프로젝트 문맥이고, 핵심역량이 보여줘야
      하는 건 "어떻게 접근하는 사람인가"이므로 접근방식을 담은 Thinking
      레이어만 쓴다). frame.action+object를 "~하는 역량" 틀로 조립
      (_competency_clause). importance 순, 중복 제거, 최대 3개.
    - 주요 기술: skill 레이어에서 두 가지를 합친다 - (1) normalized_text
      맨 앞의 영문 토큰("SQL을 이용한..." -> "SQL", "Python 기반..." ->
      "Python" - 일반 기술명은 보통 문장 맨 앞에 옴), (2) frame.method
      중 SQL 문법 조각이 아닌 것만(WITH문/WINDOW 함수처럼 "문/함수/절/
      구문/연산자"로 끝나는 토큰은 제외 - 이건 기술명이 아니라 SQL
      사용법이다). importance 순, 최대 8개.
    - 대표 경험: task 레이어의 normalized_text만 사용한다(프로젝트명
      evidence.project를 그대로 보여주지 않는다 - "숙소 운영 데이터
      분석 및 RevPAR 최적화"처럼 실제 수행한 일을 보여주라는 요구).
      importance 순, 중복 제거, 최대 3개.

    *_total 필드는 화면 상단 "핵심 역량 N개 · 기술 M개 · 경험 K개
    추출되었습니다" 완료 안내에 쓰는 실제 총량이다(표시 개수가 아니라
    잘라내기 전 전체 개수) - 지어낸 숫자가 아니라 실측치다.
    """
    if not resume_objects:
        return {
            "core_competencies": [], "core_competencies_total": 0,
            "skills": [], "skills_total": 0,
            "representative_experiences": [], "representative_experiences_total": 0,
        }

    def _dedup_labels(objs: list[dict]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for o in sorted(objs, key=_resume_importance_rank):
            label = _short_label(o)
            if label and label not in seen:
                out.append(label)
                seen.add(label)
        return out

    competencies: list[str] = []
    seen_comp: set[str] = set()
    for o in sorted((o for o in resume_objects if o.get("layer") == "thinking"), key=_resume_importance_rank):
        clause = _competency_clause(o)
        if clause and clause not in seen_comp:
            competencies.append(clause)
            seen_comp.add(clause)

    skill_objs = sorted(
        (o for o in resume_objects if o.get("layer") == "skill"),
        key=_resume_importance_rank,
    )
    skills: list[str] = []
    seen_skill: set[str] = set()

    def _add_skill(tok: str) -> None:
        tok = (tok or "").strip()
        if tok and tok not in seen_skill:
            skills.append(tok)
            seen_skill.add(tok)

    for o in skill_objs:
        leading = _LEADING_TOOL_TOKEN_RE.match(_short_label(o) or "")
        if leading:
            _add_skill(leading.group(0))
        for tok in (o.get("frame") or {}).get("method") or []:
            tok = (tok or "").strip()
            if tok and not _SKILL_SYNTAX_FRAGMENT_RE.search(tok):
                _add_skill(tok)

    representative_experiences = _dedup_labels([o for o in resume_objects if o.get("layer") == "task"])

    return {
        "core_competencies": competencies[:max_competencies],
        "core_competencies_total": len(competencies),
        "skills": skills[:max_skills],
        "skills_total": len(skills),
        "representative_experiences": representative_experiences[:max_experiences],
        "representative_experiences_total": len(representative_experiences),
    }
