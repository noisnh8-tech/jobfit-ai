"""
resume_input/quick_analysis.py

경량 분석(2026-08-31, 사용자 확정) - "분석하기"의 3-LLM 파이프라인
(JD Understanding → Resume Understanding → Semantic Linking)을 **LLM 1회**
로 압축한다. 목적은 호출 수 축소가 아니라 "실사용에서 빠른 지원 판단".

**의미 기반 판단 원리는 유지한다:**
- JD 요구사항 ↔ 이력서 근거를 의미로 연결(단순 키워드 비교 아님)
- Domain / Context / 직접경험 vs 전이경험 / responsibility scope 구분
- gap 유형(domain/skill/experience/responsibility/qualification)을 명시
- Hard Eligibility(학위/연차/자격증/어학)는 여전히 Rule(judge_engine +
  eligibility_compare + resume_facts)이 최종 결정한다 - LLM 은 관계만 본다

**제거한 것(경량화):**
- JD 6-layer intermediate object 저장(problem/thinking/task/skill/
  qualification/culture 전체 frame/intent/expected_evidence/...)
- resume_customization_targets(experience_focus/headline_focus/
  expression_focus/avoid_changes), Link improvement(문장 rewrite 지시)
- persona_tags, relations graph
→ 이 모듈이 만드는 건 판단 + 화면에 필요한 최소 필드뿐.

입력: JD 원문 + **캐시된** resume_semantic_objects(이력서당 최초 1회 LLM,
   추천/선별에서도 재사용 - 유지) + resume_facts + qualification_validation(Rule).

judge_engine/analysis_engine/cover_letter_engine 는 기존 그대로 쓰기
위해 `to_link_result()`(link_engine 결과 호환) + `to_jd_objects()`
(Hard Eligibility 용 최소 JD object)로 변환해서 넘긴다.
"""
from __future__ import annotations

import json
import re
import time

from resume_input import customizer
from resume_input.llm_client import DEFAULT_PROVIDER, call_llm

SCHEMA_VERSION = "quick-analysis-v2"  # v2(2026-09-02): requirement 별 short_label 추가

# link_result 로 변환할 때 requirement.type -> link.layer 매핑.
# judge_engine._structural_gaps 는 layer ∈ (problem,task,skill,thinking,
# culture,qualification) 만 구조적 gap 후보로 본다. experience 는
# qualification 으로 보내 Hard Eligibility 정규식 경로를 타게 한다.
_TYPE_TO_LAYER = {
    "task": "task", "skill": "skill", "qualification": "qualification",
    "experience": "qualification", "responsibility": "task", "domain": "task",
}
_RELATION_MAP = {"match": "match", "partial": "partial_match", "no_match": "no_match"}
_VALID_GAP = {
    "domain_gap", "skill_gap", "experience_gap", "responsibility_gap", "qualification_gap",
}


_PROMPT = """당신은 채용공고(JD)와 지원자의 이력서 구조화 결과를 받아, "이 공고를
지원할 만한가"를 빠르게 판단할 수 있도록 **JD 요구사항 하나하나를 이력서
근거와 의미로 연결**합니다. 점수를 매기지 않고, 최종 지원/보류/미지원을
결정하지 않습니다(그건 규칙 엔진이 합니다). 당신의 일은 관계 판단입니다.

[매우 중요 - 단어가 겹친다는 이유만으로 match 를 주지 마세요]
같은 Task 인가 / 같은 Skill 인가 / **같은 Domain·업무 맥락인가** / 비슷한
문제를 해결했는가 / **책임 범위(scope)가 비슷한가** / **직접 실행 경험인가
아니면 분석·제안 경험인가** 를 각각 따져야 합니다.

예) JD "고객 데이터를 분석해 CRM 캠페인 전략 수립" vs 이력서 "Starbucks
고객 행동 분석 → 프로모션 운영 전략 제안":
  - 고객 행동 분석 + 분석→운영전략 연결 경험은 존재 → 관계 있음
  - 그러나 CRM 실무 도메인 직접 경험은 없음
  → relation="partial", gap_type="domain_gap"
  (Python/SQL 이 같다고 match 로 끝내면 안 됩니다.)

- Domain 이 달라도 문제 해결 방식·Task 가 실제로 전이 가능하면
  relation="partial" + 해당 gap_type 으로 표현하세요(무조건 no_match 아님).
- Domain mismatch 가 있다고 무조건 no_match 로 만들지 마세요 - 전이 가능한
  경험과 직접 경험을 구분하는 것이 목적입니다.

[이미 규칙으로 확인된 사실 - qualification/experience 판단 시 이 값을
그대로 존중하세요. 스스로 뒤집지 마세요]
{qualification_validation_json}
(education=학위/전공 근거 존재, certificate=자격증 존재, language=어학성적
존재, experience=JD 요구 연차 수준 충족 여부. 예: JD 가 실무 경력을 요구하는데
experience=false 면 교육이수/자격증만으로 relation="match" 를 주지 마세요.)

[회사] {company}
[직무] {title}
[채용공고 원문]
{posting_text}

[지원자 이력서 - 구조화된 경험/역량]
{resume_objects}

[지원자 이력서에 실제로 있는 프로젝트명]
{project_titles}
[지원자 이력서에 실제로 있는 기술]
{skill_names}

[지원 가능 경력 범위 vs 관련 경험 선호 - 절대 하나로 섞지 마세요]
"신입 가능 / 경력 무관 / N년 이하 / 신입 또는 N년 이하" 처럼 지원자가
충족하기 쉬운 **지원 가능 범위** 와, 같은 문장에 붙어 있는 "관련 경험
환영·우대·선호" 같은 **soft preference** 는 의미가 달라 requirement 를 나눠야 합니다.
  - 이력서가 그 경력 범위 안에 있으면(예: 이력서=신입, JD="신입 또는 1년 이하")
    그 범위 조건 자체는 relation="match", gap_type=null 로 두세요. "신입 또는
    1년 이하" 같은 조건을 domain_gap / experience_gap / responsibility_gap 으로
    만들면 안 됩니다.
  - 괄호 안의 "관련(광고/마케팅/세일즈 등) 경험 환영·우대" 는 별도 requirement
    (type="experience", 보통 relation="no_match"/"partial")로 빼거나, 실제
    담당업무 requirement 에서 gap 을 판단하세요.
  - "N년 이상 필수" 같은 **경력 하한** 조건은 이 규칙과 무관합니다 - 위
    qualification_validation 의 experience 값을 그대로 존중하세요.

[근거 원칙 - 반드시]
- resume_evidence 는 위 이력서 구조화 결과의 evidence.source_text 를 **그대로
  인용**하거나, 그 경험을 사실 그대로 요약합니다. 이력서에 없는 경험/숫자/
  기술/도메인/프로젝트 목적을 새로 만들거나 과장하지 마세요.
- CRM 직접 경험이 없으면 "CRM 캠페인 운영 경험" 이라고 쓰면 안 됩니다.
  "CRM 직접 경험은 없으나 고객 행동 분석·프로모션 전략 제안 경험이 연결됨"
  처럼 relation/gap 을 분리하세요.

[short_label - 화면에서 빠르게 훑기 위한 "표시용 이름표"]
- 각 requirement 에 대해, 그 요구사항과 **연결되는 지원자의 경험·역량**을
  3~6단어의 **짧은 명사구**로 씁니다(문장 금지). 예: "데이터 분석·대시보드 구축",
  "고객 행동 기반 문제 정의", "SQL·Python 분석 경험", "운영 데이터 기반 개선".
- resume_evidence 를 짧게 바꾼 이름표일 뿐입니다. requirement 와 resume_evidence 에
  없는 기술/도메인/성과/숫자를 새로 만들지 마세요.
- relation="no_match" 로 연결되는 경험이 없으면 short_label 은 null 로 둡니다.
- resume_evidence 는 그대로(판단 근거 확인용) 유지합니다 - short_label 로 대체하지 마세요.

[출력 - 아래 JSON 만, 다른 설명 없이]
{{
  "job_core": "이 직무가 무엇을 하는 역할인지 1~2문장(회사가 왜 뽑는지 + 핵심 업무)",
  "role_context": {{
    "purpose": "이 역할이 해결하려는 핵심 문제/목적 1문장",
    "domain": "업무 도메인(예: CRM/마케팅 분석, 제조 데이터, 커머스 그로스) 또는 null",
    "main_tasks": ["핵심 업무 2~4개(명사구)"],
    "important_capabilities": ["중요 역량/도구 2~5개"]
  }},
  "requirements": [
    {{
      "requirement": "JD 원문에 근거한 실제 요구사항(짧은 문장)",
      "type": "task | skill | qualification | experience | responsibility | domain",
      "relation": "match | partial | no_match",
      "gap_type": "domain_gap | skill_gap | experience_gap | responsibility_gap | qualification_gap | null",
      "short_label": "이 요구사항과 연결되는 지원자의 경험·역량을 3~6단어 명사구로(연결되는 경험이 없으면 null)",
      "resume_evidence": "이력서에서 연결되는 실제 근거(원문 인용/사실 요약) 또는 null"
    }}
  ],
  "resume_order": {{
    "project_priority": ["위 '실제로 있는 프로젝트명' 중에서만, 이 공고에 먼저 보여줄 순서"],
    "skill_priority": ["위 '실제로 있는 기술' 중에서만, 이 공고에 먼저 보여줄 순서"]
  }}
}}

requirements 는 JD 의 담당업무·필수자격·우대사항의 직무 관련 항목을 빠짐없이
커버하되(6~14개 정도), 의미가 같은 항목은 합치세요. 자격요건에 "N년", "학위",
"자격증", "어학점수" 같은 객관 조건이 있으면 그 문구를 requirement 에 그대로
남기고 type 을 "experience" 또는 "qualification" 으로 하세요.
"""


def _fmt_resume_objects(resume_objects: list[dict]) -> str:
    lines = []
    for o in resume_objects:
        if o.get("layer") not in ("problem", "thinking", "task", "skill", "qualification"):
            continue
        ev = o.get("evidence") or {}
        frame = o.get("frame") or {}
        lines.append(
            f"- [{o.get('layer')}] {o.get('normalized_text', '')}\n"
            f"  의미: {o.get('meaning', '')}\n"
            f"  도메인: {frame.get('domain', '') or '(미상)'} / 대상: {frame.get('object', '') or ''}\n"
            f"  프로젝트: {ev.get('project', '') or ''}\n"
            f"  원문: {ev.get('source_text', '') or ''}"
        )
    return "\n".join(lines) if lines else "(구조화된 경험 없음)"


def _parse_json(raw: str) -> dict:
    """LLM 응답에서 JSON 객체를 뽑아 파싱한다. 추출 방식만 여러 겹으로
    시도할 뿐, 내용을 고치거나 채우지 않는다(모델이 실제로 준 답 그대로).
    2026-09-02(사용자 확정 "추출 정보는 그대로, 속도만") - 여기서 못 뽑아내면
    run_quick_analysis()가 LLM을 한 번 더 부르는데(전체 응답 하나를 통째로
    다시 기다리는 것과 같은 시간이 걸림), 그 재호출 빈도를 줄이는 게 목적."""
    text = raw.strip()

    # ① 그대로 파싱(정상 케이스 - 대부분 여기서 끝남)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ② ```json ... ``` 펜스 - 앞뒤에 설명 문장이 붙어 있어도 펜스 안의
    #    { ... } 만 찾는다(기존엔 응답이 ``` 로 "시작"할 때만 처리했음).
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # ③ 펜스가 아예 없을 때 - 첫 '{' 부터 마지막 '}' 까지(설명 문장이
    #    앞/뒤에 섞여 있는 경우, 예: "다음은 분석 결과입니다:\n{...}").
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("JSON 객체를 찾지 못함", text, 0)


def _norm(s: str) -> str:
    return re.sub(r"[\s·/]+", "", (s or "").lower())


def _validate_resume_order(qa: dict, resume_raw: str) -> None:
    """LLM 이 돌려준 project_priority/skill_priority 를 이력서에 실제로
    존재하는 항목으로만 제한한다(없는 프로젝트/기술 생성 방지)."""
    _, blocks, _ = customizer._split_projects(resume_raw)
    real_projects = [t.strip() for t, _b in blocks if t.strip()]
    _, skill_lines, _ = customizer._split_skill_lines(resume_raw)
    real_skills = [ln.split(":", 1)[0].strip() for ln in skill_lines if ln.strip() and ":" in ln]

    def _match_one(name: str, pool: list[str]) -> str | None:
        n = _norm(name)
        if not n:
            return None
        for p in pool:
            if n == _norm(p) or n in _norm(p) or _norm(p) in n:
                return p
        return None

    ro = qa.setdefault("resume_order", {})
    proj_out, seen = [], set()
    for name in ro.get("project_priority") or []:
        m = _match_one(name, real_projects)
        if m and m not in seen:
            proj_out.append(m); seen.add(m)
    for p in real_projects:  # 지정 안 된 나머지는 원래 순서로 뒤에
        if p not in seen:
            proj_out.append(p)
    ro["project_priority"] = proj_out

    skill_out, seen = [], set()
    for name in ro.get("skill_priority") or []:
        m = _match_one(name, real_skills)
        if m and m not in seen:
            skill_out.append(m); seen.add(m)
    for s in real_skills:
        if s not in seen:
            skill_out.append(s)
    ro["skill_priority"] = skill_out


def run_quick_analysis(
    resume_objects: list[dict],
    resume_raw: str,
    job: dict,
    qualification_validation: dict,
    provider: str = DEFAULT_PROVIDER,
) -> dict:
    """LLM 1회. 반환: {job_core, role_context, requirements[], resume_order,
    schema_version}. 실패 시 예외(호출부가 처리)."""
    _, blocks, _ = customizer._split_projects(resume_raw)
    project_titles = "\n".join(f"- {t.strip()}" for t, _b in blocks if t.strip()) or "(없음)"
    _, skill_lines, _ = customizer._split_skill_lines(resume_raw)
    skill_names = "\n".join(
        f"- {ln.split(':', 1)[0].strip()}" for ln in skill_lines if ln.strip() and ":" in ln
    ) or "(없음)"

    prompt = _PROMPT.format(
        qualification_validation_json=json.dumps(qualification_validation, ensure_ascii=False),
        company=job.get("company", "") or "",
        title=job.get("title", "") or "",
        posting_text=(job.get("posting_text") or "")[:8000],
        resume_objects=_fmt_resume_objects(resume_objects),
        project_titles=project_titles,
        skill_names=skill_names,
    )
    # max_tokens 여유(4096→6000) - 21개까지 나온 실측 requirements 배열이
    # 한도에 걸려 중간에 잘리면(파싱 실패→아래 재호출로 응답시간이 그대로
    # 2배가 됨) 그것 자체가 불필요한 지연이라 안전 마진만 늘린다. 모델이
    # 실제로 만드는 답은 그대로이고 한도만 넉넉해진다(2026-09-02).
    _t0 = time.monotonic()
    raw = call_llm(prompt, provider=provider, max_tokens=6000)
    try:
        qa = _parse_json(raw)
    except json.JSONDecodeError as e:
        print(f"[quick_analysis] JSON 파싱 실패({e}) - 재호출 1회 "
              f"(1차 응답 {time.monotonic() - _t0:.1f}초 소요)")
        raw = call_llm(prompt + "\n\n(JSON 형식으로만, 잘리지 않게 다시 답하세요.)",
                       provider=provider, max_tokens=6000)
        qa = _parse_json(raw)
    print(f"[quick_analysis] LLM 응답 {time.monotonic() - _t0:.1f}초")

    qa.setdefault("job_core", "")
    qa.setdefault("role_context", {})
    rc = qa["role_context"]
    rc.setdefault("purpose", ""); rc.setdefault("domain", None)
    rc.setdefault("main_tasks", []); rc.setdefault("important_capabilities", [])
    reqs = []
    for r in qa.get("requirements") or []:
        rel = r.get("relation")
        if rel not in _RELATION_MAP:
            rel = "no_match"
        gap = r.get("gap_type")
        gap = gap if gap in _VALID_GAP else None
        label = re.sub(r"\s+", " ", (r.get("short_label") or "").strip()) or None
        reqs.append({
            "requirement": (r.get("requirement") or "").strip(),
            "type": r.get("type") or "task",
            "relation": rel,
            "gap_type": gap,
            "short_label": label,
            "resume_evidence": (r.get("resume_evidence") or None),
        })
    qa["requirements"] = [r for r in reqs if r["requirement"]]
    _validate_resume_order(qa, resume_raw)
    qa["schema_version"] = SCHEMA_VERSION
    return qa


# ── 기존 엔진 호환 어댑터 ──────────────────────────────────────────────
def to_link_result(qa: dict) -> dict:
    """link_engine.run_semantic_linking() 결과와 같은 shape 으로 변환.
    judge_engine.judge() / analysis_engine.build_analysis() /
    cover_letter_engine 이 그대로 소비한다."""
    links = []
    for i, r in enumerate(qa.get("requirements") or []):
        rel = _RELATION_MAP.get(r["relation"], "no_match")
        matched = []
        if r.get("resume_evidence"):
            matched = [{
                "resume_object_id": f"ev{i}", "evidence_text": r["resume_evidence"],
                "source_project": "", "source_section": "",
            }]
        missing = []
        if rel == "no_match":
            missing = [{"type": r.get("type") or "experience", "text": r["requirement"]}]
        reason_type = r.get("gap_type") if rel != "match" else None
        links.append({
            "layer": _TYPE_TO_LAYER.get(r.get("type"), "task"),
            "jd_object_id": f"req{i}",
            "jd_requirement": r["requirement"],
            "relation": rel,
            "match_strength": {"match": "moderate", "partial_match": "weak", "no_match": "none"}[rel],
            "relation_reason_type": reason_type,
            "matched_dimensions": [],
            "matched_resume_objects": matched,
            "missing": missing,
            "improvement": None,
        })
    return {"link_version": SCHEMA_VERSION, "links": links, "unmatched_resume_objects": []}


def to_jd_objects(qa: dict) -> list[dict]:
    """judge_engine 의 Hard Eligibility 정규식이 훑을 최소 JD object.
    requirement 문구를 evidence.source_text 로 그대로 쓴다."""
    return [
        {
            "id": f"req{i}",
            "normalized_text": r["requirement"],
            "layer": _TYPE_TO_LAYER.get(r.get("type"), "task"),
            "evidence": {"section": "", "source_text": r["requirement"]},
        }
        for i, r in enumerate(qa.get("requirements") or [])
    ]
