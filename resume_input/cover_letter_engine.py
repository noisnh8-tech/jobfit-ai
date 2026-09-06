# -*- coding: utf-8 -*-
"""
resume_input/cover_letter_engine.py

Cover Letter Source(2026-08-14, 신규). 자소서 문항 하나("이런 문제를
해결한 경험을 설명하세요" 등)에 대해, 이미 계산된 Semantic Linking
결과만 근거 삼아 "이 문항에 쓸 수 있는 사실 재료(Source Pack)"를
조립하고, 그 재료 밖으로 나가지 않는 초안만 생성한다.

LLM은 정확히 2곳에서만 쓴다(사용자 사전 승인):
1. interpret_question() - 자유 서술형 질문 텍스트 하나만 보고, 이미
   존재하는 6개 Layer(problem/thinking/task/skill/qualification/culture)
   중 이 질문이 주로 묻는 Layer가 무엇인지 분류한다. 이력서/JD 어떤
   내용도 이 호출에 넣지 않는다(질문 텍스트만) - 그래서 이 단계에서
   사실을 지어낼 소재 자체가 없다. 새 분류 체계를 만들지 않고 기존
   Layer만 재사용한다.
2. generate_draft() - Source Pack(아래)에 이미 있는 필드만 입력으로
   준다. Resume/JD 원문을 통째로 주지 않는다.

2026-08-16(사용자 확정, S6 재설계) - 자소서 소재 화면은 "지원동기"와
"관련 직무 경험"을 처음부터 서로 다른 데이터로 분리해서 보여주기로
했다 - 둘 다 문항 텍스트를 입력받아 interpret_question()으로 Layer를
분류한 뒤 같은 build_source_pack() 경로로 만드니까 "지원동기" 질문에도
프로젝트 카드가 붙어 두 항목이 겹치는 문제가 있었다(사용자 지적).
- build_motivation_source(): 회사 이야기(jd_summary.problem/purpose +
  company_research의 "주요이슈" 보조)만. 이력서/프로젝트를 전혀 참조
  하지 않는다. LLM 호출 없음(둘 다 이미 만들어둔 값을 읽기만 함) -
  "작성 방향"도 고정 템플릿 문장이다(새 LLM 판단 추가 안 함).
  company_research는 "회사 알아보기" 모달(2026-07-25 완성, 현재 UI에서
  호출 안 되는 죽은 코드)이 쓰려고 WebSearch로 반자동 수집해둔 데이터 -
  jd_summary는 JD 원문(회사가 채용공고에 쓴 말)에서 나와 "24년 시리즈B
  200억 유치" 같은 구체적 외부 사실은 안 담기는데, 이런 사실이 지원동기를
  훨씬 구체적으로 만든다는 사용자 지적(2026-08-16)에 따라 추가했다.
  Top100 기준 83개사만 커버 - 없는 회사는 조용히 생략된다(폴백).
- build_related_experience_source(): 기존 build_source_pack()을 질문
  없이(고정 Layer - problem/thinking/task, customization_planner.py의
  ②프로젝트순서/④bullet강조와 동일 라우팅) 호출해서 1순위 프로젝트
  하나만 꺼낸다 - selection_engine.build_selection()이 이미 관련도
  순으로 정렬해두므로 새 랭킹 로직 없음.
이 두 함수는 interpret_question()/generate_draft()를 쓰지 않는다 -
소재까지만 만들고 초안(LLM 문장 생성)은 이번 화면 범위 밖(사용자 확정).
interpret_question()/generate_draft()/verify_draft() 자체는 지우지
않는다 - 문항 기반 자유 질의 흐름이 나중에 다시 필요해질 수 있어
(rewrite_engine.py를 지우지 않은 것과 같은 원칙), 지금 화면이 안 쓸
뿐이다.

Source Pack 조립(select_source/build_source_pack)은 100% 결정론적이다 -
새 검색/새 유사도 없이 selection_engine.build_selection()이 이미 만든
project별 그룹핑과 link_result를 그대로 재사용한다(Section 9 "새 엔진
전에 기존 코드부터" 원칙). Unverified(D)/Genuine Gap(E) 상태 항목은
Source Pack의 "행동/성과/판단접근" 같은 사실 재료에는 절대 들어가지
않는다 - analysis_engine의 5-state 분류를 그대로 재사용해서 걸러낸다
(여기서 새로 "증거가 있는가"를 판단하지 않는다). 그 항목들은 오직
"주장하면 안 되는 내용" 리스트에만 존재해서, 초안 생성 LLM이 실수로도
그걸 성과처럼 쓰지 않도록 명시적으로 알려준다.
"""
from __future__ import annotations

import json
import re

from resume_input import analysis_engine
from resume_input import generation_critic
from resume_input import selection_engine
from resume_input.job_store import get_company_research
from resume_input.llm_client import DEFAULT_PROVIDER, call_llm

_LAYERS = ("problem", "thinking", "task", "skill", "qualification", "culture")
_NUMBER_RE = re.compile(r"[₩$€¥]?\d[\d,\.]*\s*%?")


# ── 1) 질문 -> Evidence Need(기존 6-Layer만 재사용, 새 분류체계 없음) ──

_INTERPRET_PROMPT = """다음은 자기소개서 문항입니다. 이 질문이 아래 6개
카테고리(이미 정의된 채용 분석 체계입니다) 중 주로 무엇을 묻고 있는지
판단하세요. 여러 개를 물을 수도 있습니다.

- problem: 어떤 문제/상황을 마주했는가
- thinking: 어떻게 접근/판단했는가
- task: 실제로 무엇을 했는가(행동)
- skill: 어떤 기술/도구를 썼는가
- qualification: 자격/경력 조건에 관한 질문인가
- culture: 가치관/협업방식/조직적합성에 관한 질문인가

이력서나 채용공고 내용은 전혀 모른다고 가정하고, 오직 질문 문장
자체만으로 판단하세요(사실을 지어내지 마세요 - 이 단계는 분류만 합니다).

[질문]
{question}

JSON만 답하세요:
{{"layers": ["problem" 등 해당하는 것만, 최소 1개], "keywords": ["질문 문장에 실제로 있는 단어만, 새로 만들지 마세요"]}}
"""


def _parse_json_obj(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def interpret_question(question_text: str, provider: str = DEFAULT_PROVIDER) -> dict:
    """질문 텍스트만 LLM에 준다(Resume/JD 어느 쪽도 주지 않음 - 이 호출은
    사실을 지어낼 재료 자체가 없다)."""
    prompt = _INTERPRET_PROMPT.format(question=question_text)
    raw = call_llm(prompt, provider=provider, max_tokens=400)
    try:
        parsed = _parse_json_obj(raw)
    except json.JSONDecodeError:
        raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=400)
        parsed = _parse_json_obj(raw)

    layers = [l for l in (parsed.get("layers") or []) if l in _LAYERS]
    if not layers:
        layers = list(_LAYERS)
    return {"layers": layers, "keywords": parsed.get("keywords") or [], "question_text": question_text}


# ── 1.5) 지원동기 소재(회사 이야기만, 이력서/프로젝트 미참조) ─────────

_MOTIVATION_WRITING_DIRECTION = (
    "이 회사가 이 문제를 왜 해결하려 하는지 보고, 그 방향에 관심을 가진 "
    "이유와 여기서 일하고 싶은 이유를 중심으로 작성하세요."
)


def _research_highlights(company: str | None) -> list[str]:
    """company_research 테이블("회사 알아보기" 모달용으로 2026-07-25에
    WebSearch 기반 반자동 리서치로 이미 만들어둔 것, LLM 호출 없음 -
    job_store.get_company_research())의 "주요이슈"만 뽑는다. jd_summary.
    problem/purpose는 JD 원문(회사가 채용공고에 쓴 말)에서 나온 것이라
    "24년 시리즈B 200억 유치, 2년 연속 흑자" 같은 구체적 외부 사실은
    안 담긴다 - 이런 사실이 지원동기를 훨씬 구체적으로 만들어준다는
    사용자 지적(2026-08-16)에 따라 리서치가 있는 회사에 한해서만 보조로
    추가한다(Top100 기준 83개사만 커버 - 없으면 조용히 빈 리스트,
    화면이 깨지지 않게 이미 있는 폴백 패턴 그대로 재사용)."""
    if not company:
        return []
    research = get_company_research(company)
    if not research:
        return []
    raw = research.get("주요이슈")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw if v]
    text = str(raw).strip()
    return [] if text in ("", "미조사", "정보 없음") else [text]


def build_motivation_source(job: dict) -> dict:
    """지원동기 소재 - jd_summary(understanding.py가 JD Understanding
    생성 시 LLM 1회로 이미 만들어 `jd_summary` 컬럼에 저장해둔 값)의
    problem("이 직무가 해결하는 문제")/purpose("이 직무가 존재하는 목적")
    을 기본으로 하고, company_research(있는 회사만)의 "주요이슈"를 보조로
    더한다. 새 LLM 호출 없음(둘 다 이미 계산/조사된 값을 읽기만 함),
    이력서/프로젝트 내용을 전혀 참조하지 않는다(회사 이야기만 - "관련
    직무 경험"과 겹치지 않게, 2026-08-16 사용자 확정). environment는
    기본으로는 노출하지 않고 보조 정보로만 반환한다(화면이 필요할 때만
    접어서 보여줌)."""
    try:
        summary = json.loads(job.get("jd_summary") or "{}")
    except (TypeError, ValueError):
        summary = {}

    problem = (summary.get("problem") or "").strip()
    purpose = (summary.get("purpose") or "").strip()
    environment = (summary.get("environment") or "").strip()

    # 2026-08-31(경량 분석) - jd_summary(구 JD Understanding 산출물)는
    # quick_analysis 경로에서 더 이상 생성되지 않는다. problem/purpose 가
    # 비면 quick_analysis 의 role_context(purpose)/job_core 로 대체한다 -
    # 새 LLM 없음(분석 화면이 이미 만든 값 재사용), 이력서는 여전히 참조 안 함.
    if not (problem or purpose):
        qa = job.get("_quick_analysis") or {}
        rc = qa.get("role_context") or {}
        purpose = (rc.get("purpose") or "").strip()
        problem = (qa.get("job_core") or "").strip()
        if not environment:
            environment = (rc.get("domain") or "").strip()

    highlights = _research_highlights(job.get("company"))

    return {
        "problem": problem or None,
        "purpose": purpose or None,
        "environment": environment or None,
        "highlights": highlights,
        "writing_direction": _MOTIVATION_WRITING_DIRECTION,
        "has_content": bool(problem or purpose),
    }


# ── 2) Source Pack 조립(결정론적, selection_engine 재사용) ────────────

_REASON_TYPE_LABEL = {
    "scope_gap": "다루는 범위만 다름", "responsibility_gap": "책임/권한 범위가 다름",
    "experience_gap": "실무 경험 부족(이력서에서 확인 안 됨)", "domain_gap": "업무 영역이 다름",
    "skill_gap": "요구 기술 자체가 없음", "evidence_gap": "관련 근거는 있으나 수준 증명 부족",
}


def build_source_pack(
    evidence_need: dict, link_result: dict, judge_result: dict,
    resume_raw: str, resume_semantic_objects: list[dict],
) -> dict:
    """질문이 묻는 Layer(evidence_need["layers"])에 해당하는 project들만
    골라 Source Pack을 만든다. 새 그룹핑을 만들지 않고 selection_engine.
    build_selection()의 project 그룹핑(project당 covered_jd_requirements/
    selected_resume_objects/result_candidates)을 그대로 재사용한다."""
    analysis = analysis_engine.build_analysis(link_result, judge_result)
    usable_states = {"A", "B", "C"}  # D/E는 사실 재료로 쓰지 않음

    usable_jd_ids = {
        it["jd_object_id"] for it in analysis["items"]
        if it["state"] in usable_states and it["layer"] in evidence_need["layers"]
    }
    unusable_items = [
        it for it in analysis["items"]
        if it["state"] in ("D", "E") and it["layer"] in evidence_need["layers"]
    ]

    selection = selection_engine.build_selection(resume_raw, resume_semantic_objects, link_result.get("links") or [])

    packs = []
    for proj in selection["projects"]:
        covered = [r for r in proj["covered_jd_requirements"] if r["jd_object_id"] in usable_jd_ids]
        if not covered:
            continue

        obj_by_text = {o["normalized_text"]: o for o in resume_semantic_objects}
        proj_objects = [obj_by_text[t] for t in proj["selected_resume_objects"] if t in obj_by_text]

        my_situation = [o["normalized_text"] for o in proj_objects if o.get("layer") == "problem"]
        thinking = [o["normalized_text"] for o in proj_objects if o.get("layer") == "thinking"]
        action = [o["normalized_text"] for o in proj_objects if o.get("layer") == "task"]

        emphasizable = sorted({c for o in proj_objects for c in (o.get("concepts") or [])})

        # 이 project와 관련된 link 중 partial_match/evidence_gap(B/C)이면
        # "주의할 내용"으로 남긴다 - 성과인 것처럼 단정하지 말라는 신호.
        caution = []
        for it in analysis["items"]:
            if it["jd_object_id"] not in {r["jd_object_id"] for r in covered}:
                continue
            if it["state"] in ("B", "C"):
                caution.append({
                    "jd_requirement": it["jd_requirement"],
                    "reason": _REASON_TYPE_LABEL.get(it["relation_reason_type"], ""),
                })

        packs.append({
            "project": proj["project"],
            "company_requirement": [r["jd_requirement"] for r in covered],
            "my_situation": my_situation,
            "problem": my_situation,
            "thinking_approach": thinking,
            "action": action,
            "result_candidates": proj["result_candidates"],
            "job_connection": proj["connection_type"],
            "emphasizable": emphasizable,
            "caution": caution,
            "evidence": [
                {"text": o.get("evidence", {}).get("source_text", ""),
                 "project": o.get("evidence", {}).get("project", ""),
                 "section": o.get("evidence", {}).get("section", "")}
                for o in proj_objects
            ],
        })

    must_not_claim = [
        {"jd_requirement": it["jd_requirement"], "layer": it["layer"], "why": it["why"]}
        for it in unusable_items
    ]

    return {
        "question_text": evidence_need["question_text"],
        "projects": packs,
        "must_not_claim": must_not_claim,
    }


# ── 2.5) 관련 직무 경험 소재(질문 없이, 1순위 프로젝트 1개만) ──────────

_JOB_FIT_LAYERS = ("problem", "thinking", "task")


def build_related_experience_source(
    link_result: dict, judge_result: dict, resume_raw: str, resume_semantic_objects: list[dict],
    quick_analysis: dict | None = None,
) -> dict | None:
    """"관련 직무 경험" 소재 - build_source_pack()을 문항 없이(고정
    Layer) 호출해서 selection_engine이 이미 관련도 순으로 정렬해둔
    프로젝트 중 1순위 하나만 반환한다(2026-08-16 사용자 확정 - "지원동기"
    와 겹치지 않게 프로젝트 이야기는 여기서만 한다). 새 랭킹/새 판단
    없음 - build_source_pack()의 결과를 그대로 재사용.

    2026-08-31(경량 분석) - quick_analysis 경로에서는 link_result 가
    합성 resume_object_id 라 selection_engine 의 project 귀속이 안 된다.
    quick_analysis 가 넘어오면 그 결과(resume_order.project_priority +
    requirements)와 캐시된 resume_semantic_objects 로 1순위 프로젝트
    소재를 직접 조립한다(새 LLM 없음)."""
    if quick_analysis is not None:
        return _experience_source_from_quick(quick_analysis, resume_raw, resume_semantic_objects)
    evidence_need = {"layers": list(_JOB_FIT_LAYERS), "keywords": [], "question_text": ""}
    source_pack = build_source_pack(evidence_need, link_result, judge_result, resume_raw, resume_semantic_objects)
    if not source_pack["projects"]:
        return None
    return source_pack["projects"][0]


def _experience_source_from_quick(
    qa: dict, resume_raw: str, resume_semantic_objects: list[dict],
) -> dict | None:
    from resume_input.customizer import _split_projects, split_project_subsections, split_bullets

    project_priority = ((qa.get("resume_order") or {}).get("project_priority")) or []
    _, blocks, _ = _split_projects(resume_raw)
    block_by_title = {t: b for t, b in blocks}
    top = next((name for name in project_priority if name in block_by_title), None)
    if top is None:
        return None

    proj_objs = [
        o for o in resume_semantic_objects
        if (o.get("evidence") or {}).get("project") == top
    ]
    my_situation = [o["normalized_text"] for o in proj_objs if o.get("layer") == "problem"]
    thinking = [o["normalized_text"] for o in proj_objs if o.get("layer") == "thinking"]
    action = [o["normalized_text"] for o in proj_objs if o.get("layer") == "task"]
    if not (my_situation or thinking or action):
        return None

    result_candidates: list[str] = []
    for sec_title, sec_body in split_project_subsections(block_by_title[top]):
        if "성과" in sec_title:
            result_candidates = split_bullets(sec_body) or [sec_body.strip()]

    reqs = qa.get("requirements") or []
    company_requirement = [
        r["requirement"] for r in reqs if r.get("relation") in ("match", "partial")
    ][:5]
    caution = [
        {"jd_requirement": r["requirement"],
         "reason": _REASON_TYPE_LABEL.get((r.get("gap_type") or ""), "")}
        for r in reqs if r.get("relation") == "partial"
    ][:4]

    return {
        "project": top,
        "company_requirement": company_requirement,
        "my_situation": my_situation,
        "problem": my_situation,
        "thinking_approach": thinking,
        "action": action,
        "result_candidates": result_candidates,
        "job_connection": ["전이 가능한 경험"],
        "emphasizable": sorted({c for o in proj_objs for c in (o.get("concepts") or [])}),
        "caution": caution,
        "evidence": [
            {"text": (o.get("evidence") or {}).get("source_text", ""),
             "project": (o.get("evidence") or {}).get("project", ""),
             "section": (o.get("evidence") or {}).get("section", "")}
            for o in proj_objs
        ],
    }


# ── 3) 근거 있는(Source Pack 밖으로 안 나가는) 초안 생성 ───────────────

_DRAFT_PROMPT = """당신은 자기소개서 초안을 작성합니다. 아래 [사실 재료]
에 있는 내용만 사용하세요 - 여기 없는 경험/숫자/기술을 새로 지어내면
안 됩니다.

[문항]
{question}

[사실 재료 - 이 안에서만 작성하세요]
{source_pack_text}

[절대 하면 안 되는 것 - 이 요구사항들은 이력서에서 근거를 찾지 못했습니다.
경험한 것처럼 쓰면 안 됩니다]
{must_not_claim_text}

[작성 규칙]
- 사실 재료에 없는 숫자/성과/기술/조직명을 새로 쓰지 마세요.
- "주의할 내용"에 해당하는 부분은 확정적으로 단정하지 말고, 실제
  근거 수준에 맞게 표현하세요(과장 금지).
- 400자 내외.

JSON만 답하세요:
{{"draft": "..."}}
"""


def _format_source_pack_text(source_pack: dict) -> str:
    lines = []
    for p in source_pack["projects"]:
        lines.append(f"- 프로젝트: {p['project']}")
        lines.append(f"  회사요구: {'; '.join(p['company_requirement'])}")
        lines.append(f"  내상황/문제: {'; '.join(p['problem']) or '(없음)'}")
        lines.append(f"  판단접근: {'; '.join(p['thinking_approach']) or '(없음)'}")
        lines.append(f"  행동: {'; '.join(p['action']) or '(없음)'}")
        lines.append(f"  성과 후보: {' / '.join(p['result_candidates']) or '(없음)'}")
        lines.append(f"  강조 가능 키워드: {', '.join(p['emphasizable'])}")
        if p["caution"]:
            lines.append(f"  주의(단정 금지): {'; '.join(c['jd_requirement'] + '-' + c['reason'] for c in p['caution'])}")
    return "\n".join(lines) if lines else "(사용 가능한 재료 없음)"


def generate_draft(source_pack: dict, provider: str = DEFAULT_PROVIDER) -> dict:
    must_not_claim_text = "; ".join(m["jd_requirement"] for m in source_pack["must_not_claim"]) or "(없음)"
    prompt = _DRAFT_PROMPT.format(
        question=source_pack["question_text"],
        source_pack_text=_format_source_pack_text(source_pack),
        must_not_claim_text=must_not_claim_text,
    )
    raw = call_llm(prompt, provider=provider, max_tokens=1200)
    try:
        parsed = _parse_json_obj(raw)
    except json.JSONDecodeError:
        raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=1200)
        parsed = _parse_json_obj(raw)
    draft = parsed.get("draft", "") or ""
    verification = verify_draft(draft, source_pack)
    return {"draft": draft, "verification": verification}


# ── 4) 사후 사실 검증(새 유사도 점수 아님 - 존재/금지어 검사만) ───────

# ── 5) Composition(2026-08-16 2차 수정, Context Engineering 재설계,
# 사용자 확정) - Planner(관점/이야기 선택) → Writer(선택된 것만으로
# 작성) → Critic(억지연결/인과오류/과적재 검토) → Code Verifier ────────
# 1차 수정(같은 날)은 지원동기+경험 재료를 통째로 LLM 1회에 넣고 "소재+
# 초안"을 한 번에 뽑았다. 실사용 검증(docs/verification/2026-08-16_
# s4_s6_e2e_freeze/summary.md)에서 두 가지 문제가 나왔다:
# 1) 지원동기: "구독 모델 피벗 후 수익성 악화 → 2024년 흑자"라는 두
#    독립된 사실을 LLM이 "치밀한 데이터 전략이 있었기에 흑자가 가능했다"
#    는 근거 없는 인과관계로 이어붙였다.
# 2) 관련 직무 경험: 에어비앤비 프로젝트의 모든 사실(RevPAR/가격/예약률/
#    슈퍼호스트/즉시예약/대시보드/오차범위)을 한 문단에 다 욱여넣어
#    "하나의 이야기"가 아니라 "프로젝트 전체 요약"이 됐다.
#
# 원인은 같다 - LLM에게 재료 전체를 주고 "소재를 골라서 초안까지
# 써라"를 한 호출에 시켰다. 그래서 "무엇을 쓸지 고르는 것"(Planner)과
# "그것만으로 쓰는 것"(Writer)을 분리한다:
# - 지원동기: select_motivation_angle()이 회사/JD 사실 중 관점 하나와
#   그 관점에 쓸 사실만 고른다 → write_motivation_draft()는 그 관점과
#   고른 사실만 받는다(재료 전체를 다시 안 준다).
# - 관련 직무 경험: select_experience_story()가 1순위 프로젝트(이미
#   selection_engine이 골라둔 것) 안에서 "장면 하나"와 근거 2~4개만
#   고른다 → write_experience_draft()는 그 장면과 근거만 받는다.
# 그 뒤 generation_critic.py가 억지 인과/과적재 같은 확률적 문제를
# LLM으로 한 번 더 검토하고(핵심 조건 실패 시 REVISE, Writer 1회 재작성),
# 마지막으로 이 파일의 _verify_*(숫자/금지어 존재 검사, 그대로 유지)가
# 결정론적 사실 검증을 한다. 새 검색/새 랭킹/새 매칭 알고리즘을 추가한
# 게 아니다 - 이미 있던 LLM 호출 1개를 역할별로 쪼갠 것뿐이다.
#
# Cover Letter Writing Skill(사용자 확정 원칙, 두 Writer 프롬프트에
# 공통 반영): 실제 지원자가 쓰는 평이한 한국어 / 추상적 회사 칭찬
# 최소화 / JD 복붙 금지 / 문장 길이 획일화 금지 / 상투적 표현 최소화 /
# 사실→생각/판단→연결 구조 / 재료에 없는 사실 금지 / 회사 정보 과장
# 금지 / 불필요한 결론 문장 금지.

_MOTIVATION_ANGLE_PROMPT = """당신은 자기소개서 지원동기에서 잡을 관점
하나를 고르는 역할입니다. 아직 문장을 쓰지 마세요 - 관점과 그 관점에
쓸 사실만 고르세요.

[JD 사실 - 이 회사/직무가 채용공고에서 밝힌 것]
- 해결하려는 문제: {problem}
- 가려는 방향: {purpose}

[회사 사실 - 최근 동향(있을 때만 참고, 없으면 무시)]
{highlights}

위 사실만 갖고 지원동기로 쓸 관점 하나를 고르세요. 두 개 이상의 사실을
섞어 원인-결과 관계로 단정하지 마세요(예: "A했기 때문에 B했다") - 각
사실은 독립된 사실일 뿐입니다. 문제/방향이 둘 다 없으면 has_angle을
false로 하세요.

[출력 형식 - JSON만, 다른 설명 없이]
{{"has_angle": true/false,
  "angle": "이 회사에 지원하는 이유로 잡을 관점 1문장(없으면 null)",
  "company_facts": ["이 관점에 쓸 회사 사실만, 위 [회사 사실]에서 그대로 골라서"],
  "jd_facts": ["이 관점에 쓸 JD 사실만, 위 [JD 사실]에서 그대로 골라서"]}}
"""

_MOTIVATION_WRITE_PROMPT = """당신은 자기소개서 지원동기 문단을 쓰는
역할입니다. 아래 [사용 가능한 사실] 밖의 회사 정보나 사실 사이의
인과관계를 새로 만들지 마세요.

[관점]
{angle}

[사용 가능한 회사 사실 - 이것만 쓰세요]
{company_facts}

[사용 가능한 JD 사실 - 이것만 쓰세요]
{jd_facts}
{revision_note}
[글쓰기 원칙]
- 실제 지원자가 쓰는 평이한 한국어로 쓰세요. AI가 쓴 듯한 매끈한 문장을 피하세요.
- 회사에 대한 추상적인 칭찬("훌륭한 기업입니다", "업계를 선도하는")을 쓰지 마세요.
- 채용공고 문구를 그대로 복사하지 마세요 - 자기 언어로 바꿔 쓰세요.
- 모든 문장 길이를 비슷하게 만들지 마세요.
- "저는 ~라고 생각합니다" 같은 상투적 표현을 최소화하세요.
- 사실을 먼저 제시하고 → 그에 대한 생각/판단 → 지원 이유로 자연스럽게 연결하세요.
- 위 사실 중 서로 다른 두 개를 "A했기 때문에 B했다" 식으로 단정하지 마세요.
- 회사 정보를 과장하지 마세요.
- "이런 이유로 지원하게 되었습니다" 같은 뻔한 결론 문장을 반복하지 마세요.
- 300~400자.

[출력 형식 - JSON만, 다른 설명 없이]
{{"title": "이 소재를 한 줄로 요약",
  "material": "2~3문장의 소재 설명",
  "writing_point": "이 소재를 쓸 때 강조할 활용 포인트 한 줄",
  "draft": "300~400자 지원동기 문단"}}
"""

_EXPERIENCE_ANGLE_PROMPT = """당신은 자소서에 쓸 경험을 "하나의 장면"
으로 좁히는 역할입니다. 아직 문장을 쓰지 마세요 - 어떤 장면에 집중할지,
그 장면에 쓸 근거만 고르세요.

[프로젝트]
{project}

[이 프로젝트가 대응하는 공고 요구사항]
{company_requirement}

[상황/문제 후보]
{problem}

[판단/접근 후보]
{thinking}

[행동 후보]
{action}

[성과 후보]
{results}

이 프로젝트 안에는 여러 장면(이야기)이 섞여 있을 수 있습니다. 공고
요구사항과 가장 관련 있는 장면 하나만 고르고, 그 장면을 설명하는 데
필요한 근거만 2~4개 고르세요(위 후보 문장을 그대로 인용하세요 - 새로
쓰지 마세요). 모든 후보를 다 쓰려고 하지 마세요 - 이건 프로젝트 전체
요약이 아니라 한 장면의 이야기여야 합니다.

[출력 형식 - JSON만, 다른 설명 없이]
{{"story_angle": "이 경험에서 집중할 장면 1문장",
  "evidence": ["위 후보 중에서 그대로 고른 근거 2~4개"]}}
"""

_EXPERIENCE_WRITE_PROMPT = """당신은 자소서 관련 직무 경험 문단을 쓰는
역할입니다. 아래 [이야기]와 [근거] 밖의 사실을 새로 만들지 마세요 -
이 프로젝트에 있었을 다른 사건이나 수치를 추가하지 마세요.

[프로젝트]
{project}

[이 초안이 다뤄야 할 이야기]
{story_angle}

[사용 가능한 근거 - 이것만 쓰세요]
{evidence}

[주의 - 근거 부족, 확정적으로 경험한 것처럼 쓰지 말 것]
{caution}
{revision_note}
[글쓰기 원칙]
- 실제 지원자가 쓰는 평이한 한국어로 쓰세요.
- [이야기] 하나에 집중하세요 - 프로젝트에서 있었던 다른 일들을 나열하지 마세요.
- 위 [사용 가능한 근거]에 없는 숫자나 사건을 새로 지어내지 마세요.
- 모든 문장 길이를 비슷하게 만들지 마세요.
- 상황 → 판단/행동 → 결과 순서로 자연스럽게 쓰세요.
- 300~400자.

[출력 형식 - JSON만, 다른 설명 없이]
{{"title": "이 경험을 한 줄로 요약",
  "situation": "상황/문제 짧게",
  "action": "판단/행동 짧게",
  "result": "결과 짧게",
  "writing_point": "이 경험에서 살릴 부분 한 줄",
  "draft": "300~400자 직무경험 문단"}}
"""


def _fmt_list(items: list[str] | None) -> str:
    items = [i for i in (items or []) if i]
    return "; ".join(items) if items else "(없음)"


def _call_llm_json(prompt: str, provider: str, max_tokens: int) -> dict:
    """실패 시 예외를 던지지 않는다(2026-08-16, generation_critic.py와
    동일한 이유로 수정 - 특정 해외 테크기업 케이스에서 Critic 크래시 실측 확인 후,
    Planner/Writer 쪽도 같은 위험이 있어 함께 방어한다). 빈 dict를
    반환하면 각 호출부의 .get()이 falsy 값을 돌려줘 "찾지 못함"/"근거
    없음" 경로로 안전하게 빠진다 - 전체 생성 파이프라인이 죽지 않는다."""
    raw = call_llm(prompt, provider=provider, max_tokens=max_tokens)
    try:
        return _parse_json_obj(raw)
    except json.JSONDecodeError:
        pass
    try:
        raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=max_tokens)
        return _parse_json_obj(raw)
    except json.JSONDecodeError:
        return {}


# ── 지원동기: Planner → Writer → Critic → Code Verifier ────────────────

def select_motivation_angle(motivation_source: dict, provider: str = DEFAULT_PROVIDER) -> dict:
    """Planner(1단계) - 지원동기로 쓸 관점 하나와, 그 관점에 쓸 사실만
    고른다. 문장을 쓰지 않는다(Writer의 책임과 분리)."""
    prompt = _MOTIVATION_ANGLE_PROMPT.format(
        problem=motivation_source.get("problem") or "(없음)",
        purpose=motivation_source.get("purpose") or "(없음)",
        highlights=_fmt_list(motivation_source.get("highlights")),
    )
    parsed = _call_llm_json(prompt, provider, 500)
    return {
        "has_angle": bool(parsed.get("has_angle")) and bool(parsed.get("angle")),
        "angle": parsed.get("angle"),
        "company_facts": [f for f in (parsed.get("company_facts") or []) if f],
        "jd_facts": [f for f in (parsed.get("jd_facts") or []) if f],
    }


def write_motivation_draft(angle_result: dict, provider: str = DEFAULT_PROVIDER, revision_note: str = "") -> dict:
    """Writer(2단계) - Planner가 고른 angle/company_facts/jd_facts만
    받는다. motivation_source 원본(problem/purpose/highlights 전체)을
    다시 주지 않는다(Context Engineering - 필요한 것만)."""
    note = f"\n[이전 시도 실패 이유 - 이번엔 반드시 피하세요]\n{revision_note}\n" if revision_note else ""
    prompt = _MOTIVATION_WRITE_PROMPT.format(
        angle=angle_result.get("angle") or "",
        company_facts=_fmt_list(angle_result.get("company_facts")),
        jd_facts=_fmt_list(angle_result.get("jd_facts")),
        revision_note=note,
    )
    return _call_llm_json(prompt, provider, 900)


def _verify_motivation_draft(draft: str, angle_result: dict) -> dict:
    """Code Verifier - 초안의 숫자가 Planner가 고른 사실 안에 실제로
    있는가만 본다(새 유사도 점수 없음)."""
    failed = []
    material_text = " ".join(angle_result.get("company_facts", []) + angle_result.get("jd_facts", []))
    material_numbers = set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(material_text))
    draft_numbers = set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(draft or ""))
    invented = sorted(draft_numbers - material_numbers)
    if invented:
        failed.append({"check": "numbers_grounded", "detail": f"재료에 없는 숫자 사용: {invented}"})
    return {"passed": not failed, "failed_checks": failed}


def _build_motivation(motivation_source: dict | None, provider: str) -> dict | None:
    if not motivation_source or not motivation_source.get("has_content"):
        return None
    angle_result = select_motivation_angle(motivation_source, provider)
    if not angle_result["has_angle"]:
        return None

    revision_note = ""
    for attempt in range(2):  # 최초 시도 + 실패 시 1회만 재작성
        draft_out = write_motivation_draft(angle_result, provider, revision_note)
        draft = draft_out.get("draft") or ""
        code_check = _verify_motivation_draft(draft, angle_result)
        if not code_check["passed"]:
            if attempt == 0:
                revision_note = "; ".join(c["detail"] for c in code_check["failed_checks"])
                continue
            return None  # 2회 다 실패 - 근거 없는 숫자를 화면에 보여줄 수 없음

        critic = generation_critic.evaluate_motivation_draft(
            draft, angle_result.get("company_facts", []), angle_result.get("jd_facts", []), provider,
        )
        if critic["verdict"] == "PASS":
            return {**draft_out, "draft": draft}
        if attempt == 0:
            revision_note = critic.get("fail_reason") or "억지 인과관계나 일반적인 문장을 피할 것"
            continue
        return None  # Critic이 2회 다 거부 - 사용자에게 근거 없는 초안을 보여주지 않음
    return None


# ── 관련 직무 경험: Planner → Writer → Critic → Code Verifier ──────────

def select_experience_story(experience_source: dict, provider: str = DEFAULT_PROVIDER) -> dict:
    """Planner(1단계) - 1순위 프로젝트(build_related_experience_source가
    이미 골라둔 것) 안에서 장면 하나와 근거 2~4개만 고른다. 프로젝트
    전체를 다 쓰지 않는다(2026-08-16 실사용 확인된 "프로젝트 요약처럼
    됨" 문제의 직접적 원인 수정)."""
    problem = experience_source.get("problem") or experience_source.get("my_situation") or []
    thinking = experience_source.get("thinking_approach") or []
    action = experience_source.get("action") or []
    results = experience_source.get("result_candidates") or []
    prompt = _EXPERIENCE_ANGLE_PROMPT.format(
        project=experience_source.get("project") or "",
        company_requirement=_fmt_list(experience_source.get("company_requirement")),
        problem=_fmt_list(problem), thinking=_fmt_list(thinking),
        action=_fmt_list(action), results=_fmt_list(results),
    )
    parsed = _call_llm_json(prompt, provider, 600)
    all_candidates = set(problem) | set(thinking) | set(action) | set(results)
    evidence = [e for e in (parsed.get("evidence") or []) if e in all_candidates][:4]
    return {
        "has_story": bool(parsed.get("story_angle")) and bool(evidence),
        "story_angle": parsed.get("story_angle"),
        "evidence": evidence,
    }


def write_experience_draft(
    project: str, story_result: dict, caution: list[dict], provider: str = DEFAULT_PROVIDER, revision_note: str = "",
) -> dict:
    """Writer(2단계) - Planner가 고른 story_angle/evidence(2~4개)만
    받는다. 프로젝트의 전체 근거 목록을 다시 주지 않는다."""
    note = f"\n[이전 시도 실패 이유 - 이번엔 반드시 피하세요]\n{revision_note}\n" if revision_note else ""
    caution_text = "; ".join(f"{c['jd_requirement']}({c['reason']})" for c in caution) or "(없음)"
    prompt = _EXPERIENCE_WRITE_PROMPT.format(
        project=project, story_angle=story_result.get("story_angle") or "",
        evidence=_fmt_list(story_result.get("evidence")), caution=caution_text, revision_note=note,
    )
    return _call_llm_json(prompt, provider, 900)


def _verify_experience_draft(draft: str, story_result: dict, caution: list[dict]) -> dict:
    """Code Verifier - 숫자는 Planner가 고른 근거 안에 있는가, 근거
    부족 항목("주의")을 단정적으로 언급하지 않았는가만 본다."""
    failed = []
    material_text = " ".join(story_result.get("evidence", []))
    material_numbers = set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(material_text))
    draft_numbers = set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(draft or ""))
    invented = sorted(draft_numbers - material_numbers)
    if invented:
        failed.append({"check": "numbers_grounded", "detail": f"근거에 없는 숫자 사용: {invented}"})

    draft_norm = re.sub(r"\s+", "", draft or "")
    claimed_forbidden = [
        c["jd_requirement"] for c in caution
        if re.sub(r"\s+", "", c["jd_requirement"]) and re.sub(r"\s+", "", c["jd_requirement"]) in draft_norm
    ]
    if claimed_forbidden:
        failed.append({"check": "no_forbidden_claims", "detail": f"근거 부족한 내용을 단정적으로 언급함: {claimed_forbidden}"})
    return {"passed": not failed, "failed_checks": failed}


def _build_experience(experience_source: dict | None, provider: str) -> dict | None:
    if not experience_source:
        return None
    story_result = select_experience_story(experience_source, provider)
    if not story_result["has_story"]:
        return None

    project = experience_source.get("project") or ""
    caution = experience_source.get("caution") or []
    revision_note = ""
    for attempt in range(2):
        draft_out = write_experience_draft(project, story_result, caution, provider, revision_note)
        draft = draft_out.get("draft") or ""
        code_check = _verify_experience_draft(draft, story_result, caution)
        if not code_check["passed"]:
            if attempt == 0:
                revision_note = "; ".join(c["detail"] for c in code_check["failed_checks"])
                continue
            return None

        critic = generation_critic.evaluate_experience_draft(draft, story_result.get("story_angle") or "", story_result.get("evidence", []), provider)
        if critic["verdict"] == "PASS":
            return {**draft_out, "draft": draft}
        if attempt == 0:
            revision_note = critic.get("fail_reason") or "프로젝트 전체를 요약하지 말고 하나의 장면에만 집중할 것"
            continue
        return None
    return None


def generate_composition(motivation_source: dict, experience_source: dict | None, provider: str = DEFAULT_PROVIDER) -> dict:
    """motivation_source(build_motivation_source 반환값)와 experience_source
    (build_related_experience_source 반환값, 없으면 None)를 받아 지원동기/
    관련 직무 경험을 각각 Planner→Writer→Critic→Code Verifier 파이프라인
    으로 만든다. 두 재료 다 비어 있으면 LLM을 아예 호출하지 않는다(지어낼
    것도, 쓸 것도 없음). 재작성은 각 트랙마다 최대 1회 - 그래도 실패하면
    해당 트랙만 None으로 반환한다(근거 없는 초안을 화면에 보여주지
    않는 것이 우선, 다른 트랙에는 영향 없음)."""
    has_motivation = bool(motivation_source and motivation_source.get("has_content"))
    has_experience = bool(experience_source)
    if not has_motivation and not has_experience:
        return {"motivation": None, "experience": None, "verification": {"passed": True, "failed_checks": []}}

    motivation_out = _build_motivation(motivation_source, provider) if has_motivation else None
    experience_out = _build_experience(experience_source, provider) if has_experience else None

    return {
        "motivation": motivation_out,
        "experience": experience_out,
        "verification": {"passed": True, "failed_checks": []},
    }


def verify_draft(draft_text: str, source_pack: dict) -> dict:
    failed = []

    pack_text = " ".join(
        " ".join(p["my_situation"]) + " " + " ".join(p["thinking_approach"]) + " " +
        " ".join(p["action"]) + " " + " ".join(p["result_candidates"])
        for p in source_pack["projects"]
    )
    pack_numbers = set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(pack_text))
    draft_numbers = set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(draft_text))
    invented_numbers = sorted(draft_numbers - pack_numbers)
    if invented_numbers:
        failed.append({"check": "numbers_grounded", "detail": f"Source Pack에 없는 숫자 사용: {invented_numbers}"})

    draft_norm = re.sub(r"\s+", "", draft_text)
    claimed_forbidden = [
        m["jd_requirement"] for m in source_pack["must_not_claim"]
        if re.sub(r"\s+", "", m["jd_requirement"]) and re.sub(r"\s+", "", m["jd_requirement"]) in draft_norm
    ]
    if claimed_forbidden:
        failed.append({"check": "no_forbidden_claims", "detail": f"근거 없는 요구사항을 그대로 언급함: {claimed_forbidden}"})

    if not draft_text.strip():
        failed.append({"check": "not_empty", "detail": "초안이 비어 있음"})

    return {"passed": not failed, "failed_checks": failed}
