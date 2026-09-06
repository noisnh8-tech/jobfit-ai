# -*- coding: utf-8 -*-
"""
resume_input/rewrite_engine.py

Customization - Rewrite(2026-08-14, 신규). resume_customizing.py(v4,
2026-08-03)가 의도적으로 빼둔 문장 재작성(rephrase) 하나만 담당한다 -
reposition/add_emphasis/reorder/de-emphasize는 이미 resume_customizing.py
(Command 생성) + resume_apply_engine.py(실행) + selection_engine.py/
composition_engine.py(표시)가 LLM 없이 전부 처리한다. 여기서 LLM을 쓰는
이유는 "문장을 자연스럽게 다시 쓰는 것"이 결정론적 규칙으로는 안 된다는
게 이미 실측 확인됐기 때문이다(resume_customizing.py 상단 docstring,
2026-08-03 - Rule Template로 만든 문장 3건 전부 비문).

입력 제약(허용 목록/금지 목록, 사용자 확정):
- 허용: rephrase(표현만 바꿈) / reposition·add_emphasis·reorder·
  de-emphasize(이미 다른 엔진이 처리 - 여기서는 review 표시용으로만 통과)
- 금지: 새 경험/새 스킬/새 숫자/과장/공백 은폐(Gap을 숨기는 문장)/
  Unverified 항목을 사실인 것처럼 서술.
- 프롬프트에 Resume+JD 원문을 통째로 넣지 않는다. link_engine.py가 이미
  근거를 확정한 딱 1개 Resume Object의 evidence.source_text + 그 JD
  요구사항의 normalized_text/concepts만 입력으로 준다(그 Resume Object가
  이미 "관련 있다"고 확정됐으므로, 이 범위를 벗어난 재작성은 애초에
  근거를 벗어난 것이다).

LLM 호출 후 반드시 자동 사실 검증(verify_rewrite)을 거친다 - 검증
실패 항목은 자동 적용되지 않고 "원문 유지"가 기본값으로 남는다(review
UI가 그대로 승인해도 실패 항목은 별도로 표시). 검증은 새 유사도 점수를
만들지 않는다 - 숫자/개체명(Understanding이 이미 뽑아둔 concepts)이
원문에 그대로 있는지 존재 여부만 확인하는 사실 검사다.

원본 이력서(resume_raw)는 이 파일의 어떤 함수도 직접 수정하지 않는다 -
apply_rewrite_decisions()는 새 텍스트를 반환할 뿐이고, 실제 저장은
rewrite_versions.save_rewrite_version()이 별도 버전으로만 한다.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from resume_input import customizer
from resume_input import generation_critic
from resume_input.llm_client import DEFAULT_PROVIDER, call_llm

_NUMBER_RE = re.compile(r"[₩$€¥]?\d[\d,\.]*\s*%?")


def _obj_id(obj: dict) -> str:
    return obj.get("id") or obj.get("normalized_text") or ""


def _norm_ws(text: str) -> str:
    return re.sub(r"[\s•]+", "", text or "")


def _bullet_matches(bullet: str, source_text: str) -> bool:
    """resume_apply_engine._bullet_matches()와 동일 로직(매칭 기준을 새로
    만들지 않고 그대로 복제) - 이 불릿이 이 Resume Object의 evidence.
    source_text에 대응하는가."""
    b, s = _norm_ws(bullet), _norm_ws(source_text)
    if not b or not s:
        return False
    if b in s or s in b:
        return True
    return any(_norm_ws(line) and _norm_ws(line) in b for line in (source_text or "").split("\n"))


# ── 1) Rewrite 대상 수집(rephrase만, reposition/add_emphasis는 이미 처리됨) ──

def collect_rewrite_targets(link_result: dict, resume_semantic_objects: list[dict]) -> list[dict]:
    """Semantic Link 결과에서 improvement.type=rewrite, action=rephrase인
    항목만 골라 Rewrite 대상을 만든다. target_resume_object_id는 link_
    engine._validate_and_clean_links()가 이미 matched_resume_objects 안의
    id로만 존재하도록 강제해뒀다(스키마 보장) - 여기서 다시 그 존재를
    신뢰하고 조회만 한다."""
    obj_by_id = {_obj_id(o): o for o in resume_semantic_objects}
    targets = []
    for link in link_result.get("links") or []:
        improvement = link.get("improvement")
        if not improvement or improvement.get("type") != "rewrite":
            continue
        if improvement.get("action") != "rephrase":
            continue  # reposition/add_emphasis는 이미 다른 엔진이 처리
        target_id = improvement.get("target_resume_object_id")
        robj = obj_by_id.get(target_id)
        if not robj:
            continue
        ev = robj.get("evidence") or {}
        targets.append({
            "jd_object_id": link.get("jd_object_id"),
            "jd_requirement": link.get("jd_requirement"),
            "layer": link.get("layer"),
            "jd_concepts": link.get("concepts") or [],
            "target_resume_object_id": target_id,
            "project": ev.get("project", ""),
            "section": ev.get("section", ""),
            "original_text": ev.get("source_text", ""),
            "resume_object_concepts": robj.get("concepts") or [],
        })
    return targets


# ── 2) 제한된 입력만 담은 프롬프트 ────────────────────────────────────

_REWRITE_PROMPT = """당신은 이력서의 문장 1개를 더 명확하게 다듬는 역할만
합니다. 아래 [원문 문장]에 이미 있는 사실만 사용해서 표현을 다듬으세요.

[이 문장이 대응하는 채용공고 요구사항]
{jd_requirement}
관련 키워드(참고용, 원문에 이미 그 의미가 있을 때만 그 단어를 그대로
써도 됩니다 - 원문에 없는 의미를 새로 넣는 게 아닙니다): {jd_concepts}

[원문 문장]
{original_text}

[절대 금지]
- 원문에 없는 숫자를 새로 쓰지 마세요. 원문에 있는 숫자는 전부 그대로
  유지하세요(단위/기호 포함).
- 원문에 없는 새로운 경험, 새로운 기술/도구, 새로운 성과를 지어내지
  마세요.
- 과장하지 마세요("최초로", "압도적으로" 같은 원문에 없는 강조 표현 금지).
- 원문에 있는 한계나 부족한 점을 숨기거나 삭제하지 마세요.
- 문장을 통째로 새로 짓지 마세요 - 어순 조정, 표현 교체, 강조 위치 이동
  정도만 허용됩니다.

[출력 형식 - JSON만, 다른 설명 없이]
{{"rewritten_text": "...", "changed_because": "무엇을 어떻게 바꿨는지 1문장"}}
"""


def build_rewrite_prompt(target: dict) -> str:
    return _REWRITE_PROMPT.format(
        jd_requirement=target["jd_requirement"] or "",
        jd_concepts=", ".join(target["jd_concepts"]) or "(없음)",
        original_text=target["original_text"],
    )


def _parse_json_obj(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ── 3) 자동 사실 검증(새 유사도 점수 아님 - 존재 여부만 확인) ─────────

def verify_rewrite(original_text: str, rewritten_text: str, resume_object_concepts: list[str]) -> dict:
    """카테고리형 사실 검사만 한다(가중합/유사도 점수 없음):
    1) numbers_preserved: 원문의 모든 숫자가 재작성문에도 그대로 있는가
       (새 숫자가 추가됐거나, 있던 숫자가 사라졌으면 실패).
    2) concepts_preserved: 이 Resume Object에 이미 태깅된 concepts(Understanding
       단계에서 추출, 새로 만드는 값 아님)가 재작성문에서도 확인되는가.
    3) not_empty: 빈 문자열이 아닌가.
    셋 다 통과해야 passed=True. 실패 항목은 failed_checks에 나열한다."""
    failed = []

    orig_numbers = sorted(set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(original_text)))
    new_numbers = sorted(set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(rewritten_text)))
    if orig_numbers != new_numbers:
        failed.append({
            "check": "numbers_preserved",
            "detail": f"원문 숫자={orig_numbers} vs 재작성 숫자={new_numbers}",
        })

    missing_concepts = []
    rewritten_norm = _norm_ws(rewritten_text)
    for c in resume_object_concepts:
        if _norm_ws(c) and _norm_ws(c) not in rewritten_norm and _norm_ws(c) in _norm_ws(original_text):
            missing_concepts.append(c)
    if missing_concepts:
        failed.append({"check": "concepts_preserved", "detail": f"원문에 있던 핵심 개념이 사라짐: {missing_concepts}"})

    if not rewritten_text or not rewritten_text.strip():
        failed.append({"check": "not_empty", "detail": "재작성 결과가 비어 있음"})

    return {"passed": not failed, "failed_checks": failed}


# ── 4) 오케스트레이션(제안 생성 + 검증, resume_raw 변형 없음) ─────────

def generate_rewrite_proposals(
    link_result: dict, resume_semantic_objects: list[dict], provider: str = DEFAULT_PROVIDER,
) -> list[dict]:
    """rephrase 대상 전부에 대해 LLM 제안 1건씩 생성하고 즉시 검증한다.
    검증 실패 시 그 항목의 default_choice는 "current"(원문 유지)로
    강제된다 - 실패한 제안이 조용히 채택되는 경로는 없다."""
    targets = collect_rewrite_targets(link_result, resume_semantic_objects)
    proposals = []
    for t in targets:
        prompt = build_rewrite_prompt(t)
        raw = call_llm(prompt, provider=provider, max_tokens=800)
        try:
            parsed = _parse_json_obj(raw)
        except json.JSONDecodeError:
            raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=800)
            parsed = _parse_json_obj(raw)

        rewritten_text = parsed.get("rewritten_text", "") or ""
        changed_because = parsed.get("changed_because", "")
        verification = verify_rewrite(t["original_text"], rewritten_text, t["resume_object_concepts"])

        proposals.append({
            **t,
            "recommended_text": rewritten_text,
            "reason": changed_because,
            "verification": verification,
            # 검증 실패면 무조건 원문 유지가 기본값이다 - review UI가 이
            # 기본값을 뒤집을 수 없다(실패 항목은 "추천안 적용" 버튼 자체를
            # 비활성화해야 한다는 뜻 - UI 쪽 책임).
            "default_choice": "recommended" if verification["passed"] else "current",
        })
    return proposals


def build_rewrite_review(proposals: list[dict]) -> list[dict]:
    """화면(현재/추천/이유 + [원문유지]/[추천안적용] + 전체적용) 표시용
    최소 데이터만 뽑는다."""
    return [
        {
            "target_resume_object_id": p["target_resume_object_id"],
            "project": p["project"],
            "section": p["section"],
            "current_text": p["original_text"],
            "recommended_text": p["recommended_text"],
            "reason": p["reason"],
            "can_apply": p["verification"]["passed"],
            "verification_failed_checks": p["verification"]["failed_checks"],
            "default_choice": p["default_choice"],
        }
        for p in proposals
    ]


# ── 5) 사용자가 선택한 결정만 실제 텍스트에 반영(resume_raw는 불변) ───

def apply_rewrite_decisions(resume_raw: str, decisions: list[dict]) -> tuple[str, list[dict]]:
    """decisions: [{"target_resume_object_id", "project", "original_text",
    "chosen_text"}]. chosen_text가 original_text와 다른 항목만 실제로
    치환한다("추천안 적용"을 선택한 항목만). 검증 실패 항목이 여기 섞여
    들어와도(호출자 실수) chosen_text==original_text면 아무 일도 안
    일어난다 - 이 함수 자체는 "무엇을 바꿀지 이미 정해진" 것만 실행한다
    (resume_apply_engine.py와 동일한 책임 분리 원칙)."""
    text = resume_raw
    applied = []
    by_project: dict[str, list[dict]] = {}
    for d in decisions:
        if d["chosen_text"] == d["original_text"]:
            continue
        by_project.setdefault(d["project"], []).append(d)

    for project_title, ds in by_project.items():
        _, blocks, _ = customizer._split_projects(text)
        block_idx = next((i for i, (t, _b) in enumerate(blocks) if t == project_title), None)
        if block_idx is None:
            continue
        title, block = blocks[block_idx]

        for section_title, section_body in customizer.split_project_subsections(block):
            bullets = customizer.split_bullets(section_body)
            if not bullets:
                continue
            changed = False
            new_bullets = list(bullets)
            for d in ds:
                for i, b in enumerate(new_bullets):
                    if _bullet_matches(b, d["original_text"]):
                        new_bullets[i] = d["chosen_text"]
                        changed = True
                        applied.append({"project": project_title, "section": section_title,
                                         "target_resume_object_id": d["target_resume_object_id"]})
                        break
            if changed:
                new_block = customizer._replace_subsection_bullets(block, section_title, new_bullets)
                if new_block != block:
                    _, all_blocks, _ = customizer._split_projects(text)
                    preamble, _, postamble = customizer._split_projects(text)
                    all_blocks[block_idx] = (title, new_block)
                    text = preamble + "".join(b for _, b in all_blocks) + postamble
                    block = new_block

    return text, applied


# ── 6) ① 한줄 소개 - Planner(판단) → Writer(작성) → Critic(비판) →
# Code Verifier(사실검증)(2026-08-16 3차 수정, 사용자 확정, Context
# Engineering 재설계) ───────────────────────────────────────────────────
# 2차 수정(Microsoft 사례)까지는 "분해+근거확인+재작성"을 LLM 1회에
# 몰아넣었다. 실측 결과("자율적으로 지표를 재정의해...") 이 구조는
# 근거가 조금이라도 있으면 거의 항상 "자율적으로/주도적으로" 같은
# 인재상 수식어를 억지로 끼워 넣는 문제를 만들었다 - 판단과 작성이
# 분리되지 않아서, "이 특성이 근거가 있는가"라는 질문이 곧바로
# "그러면 문장에 넣어라"로 이어졌기 때문이다(docs/verification/
# 2026-08-16_s4_s6_e2e_freeze/summary.md에서 실사용 확인).
#
# 그래서 세 역할로 분리한다:
# 1) Planner(decide_headline_change) - "바꿀 필요가 있는가"만 구조화된
#    KEEP/REWRITE로 판단한다. 인재상 수식어 단순 삽입은 REWRITE 사유가
#    될 수 없다고 프롬프트에 명시한다 - 기본값은 KEEP.
# 2) Writer(write_headline_candidate) - REWRITE로 판단된 경우에만
#    호출된다. JD 인재상 원문 전체나 이력서 근거 목록 전체를 다시 주지
#    않고, Planner가 고른 missing_meaning + evidence_source_text만 준다
#    (Context Engineering - Writer에게 필요한 것만).
# 3) Critic(generation_critic.evaluate_headline_candidate) - "억지스러운가/
#    자연스러운가"는 코드로 확인할 수 없으니 별도 LLM 시선으로 한 번 더
#    본다. 핵심 조건 중 하나라도 실패하면 원문 유지(USE_ORIGINAL).
#
# 그 뒤 verify_headline_rewrite(Code Verifier, 그대로 유지)가 숫자 보존/
# evidence 실재/최소편집 여부를 사실로만 검사한다. 확률적 판단(자연스러운가)
# 과 결정론적 검증(숫자가 보존됐는가)을 분리하는 게 이번 수정의 핵심이다 -
# 새 매칭 알고리즘이나 프레임워크를 추가하는 게 아니라, 이미 있던 LLM
# 호출 1개를 역할별로 쪼갠 것뿐이다.

_HEADLINE_DECIDE_PROMPT = """당신은 이력서 상단의 한줄 소개(헤드라인)
문장을 이 공고에 맞춰 바꿀 필요가 있는지만 판단하는 역할입니다. 문장을
쓰지 마세요 - 바꿀지 말지와 그 이유만 답하세요.

[현재 한줄 소개]
{current_headline}

[이 공고가 가장 강조하는 인재상]
{jd_culture}

이 인재상 문장 안에는 여러 특성이 섞여 있을 수 있습니다(예: "성장
마인드셋 및 포용적 리더십" = 성장 마인드셋 + 포용적 리더십, 서로 다른
특성). 아래 [내 이력서 근거 목록]을 보고, 이 인재상을 이루는 특성 중
실제로 뒷받침하는 근거가 있는지 확인하세요.

근거는 그 특성을 그대로 말한 문장일 필요가 없습니다. 실제 행동이나
사고방식에서 그 특성이 간접적으로 드러나면 인정하세요. 예를 들어
"기존 접근 방식이 안 맞아서 다른 방식으로 전환/재정의했다"는 유연성/
적응력의 근거가 될 수 있습니다. 단, 근거 문장에 실제로 있는 행동에서
자연스럽게 이어지는 특성만 인정하세요 - 무관한 특성을 억지로 끌어다
붙이지 마세요.

[내 이력서 근거 목록 - 각 줄이 실제 프로젝트에서 나온 서로 다른 근거입니다]
{resume_objects_list}

[중요 - REWRITE 기준]
근거가 있다고 해서 자동으로 REWRITE가 아닙니다. 다음 경우에만 REWRITE:
- 현재 한줄 소개에 이 지원자의 "직무 정체성이나 실제 수행 영역"과 관련해
  분명히 빠진 의미가 있고, 그 의미를 더하면 이 사람이 무슨 일을 하는
  사람인지가 더 정확해지는 경우.

다음은 REWRITE 사유가 될 수 없습니다(그 자체만으로는 KEEP):
- "자율적으로", "주도적으로", "빠르게", "적극적으로", "도전적으로",
  "성장하는", "협업하는" 같은 수식어/부사 하나를 추가할 수 있다는 것.
  이런 수식어는 거의 모든 경험에 갖다 붙일 수 있어서 판단 기준으로
  쓰면 안 됩니다.
- 이미 현재 한줄 소개에 있는 의미를 다른 단어로 바꿔 말할 수 있다는 것.
- 인재상과 "관련은 있지만" 직무 정체성 자체를 바꾸지는 않는 경우.

근거가 하나도 없거나, 있어도 위 기준의 REWRITE에 해당하지 않으면 KEEP.

[출력 형식 - JSON만, 다른 설명 없이]
{{"decision": "KEEP 또는 REWRITE",
  "missing_meaning": "REWRITE일 때만: 현재 문장에 빠진 실질적 의미 1구절(KEEP이면 null)",
  "evidence_source_text": "REWRITE일 때만: 근거로 쓴 이력서 문장 원문 그대로(KEEP이면 null)",
  "reason": "판단 이유 1문장"}}
"""

_HEADLINE_WRITE_PROMPT = """당신은 이력서 한줄 소개 문장에 빠진 의미
하나를 최소한으로만 추가하는 역할입니다.

[현재 한줄 소개]
{current_headline}

[추가해야 할 의미]
{missing_meaning}

[근거]
{evidence_source_text}

[규칙]
- 기존 문장의 핵심 정체성과 톤은 그대로 유지하고, 위 의미만 자연스럽게
  녹이세요(완전히 새 문장으로 다시 쓰지 마세요 - 어순 조정/표현 추가
  정도만).
- "자율적으로", "주도적으로", "빠르게", "적극적으로", "도전적으로" 같은
  수식어/부사를 단순히 끼워 넣는 방식으로 의미를 추가하지 마세요 - 그건
  의미를 추가한 게 아니라 장식을 추가한 것입니다. [추가해야 할 의미]가
  실제 직무 정체성/수행 영역에 반영되게 쓰세요.
- 근거 문장을 그대로 복사하지 말고 취지만 반영하세요.
- 원문에 없는 새로운 경험, 기술, 성과, 숫자를 지어내지 마세요.
- 과장하지 마세요("최초로", "압도적으로" 같은 표현 금지).

[출력 형식 - JSON만, 다른 설명 없이]
{{"rewritten_headline": "..."}}
"""


def _resume_objects_for_headline(resume_semantic_objects: list[dict]) -> list[dict]:
    """헤드라인 판단용으로 culture를 제외한 전 레이어(problem/thinking/
    task/skill/qualification)의 실제 근거 문장만 뽑는다 - culture는
    JD 쪽 개념이라 Resume Object에는 없다."""
    out = []
    for o in resume_semantic_objects:
        if o.get("layer") == "culture":
            continue
        ev = (o.get("evidence") or {}).get("source_text", "")
        if ev:
            out.append({"id": _obj_id(o), "source_text": ev})
    return out


def decide_headline_change(
    headline_note: dict, resume_semantic_objects: list[dict], provider: str = DEFAULT_PROVIDER,
) -> dict:
    """Planner(1단계) - 바꿀지 말지만 구조화된 출력으로 판단한다. 문장을
    쓰지 않는다(Writer의 책임과 분리)."""
    objs = _resume_objects_for_headline(resume_semantic_objects)
    listing = "\n".join(f"- {o['source_text']}" for o in objs) or "(근거 없음)"
    prompt = _HEADLINE_DECIDE_PROMPT.format(
        current_headline=headline_note.get("current_headline") or "",
        jd_culture=headline_note.get("jd_culture") or "",
        resume_objects_list=listing,
    )
    raw = call_llm(prompt, provider=provider, max_tokens=500)
    try:
        parsed = _parse_json_obj(raw)
    except json.JSONDecodeError:
        parsed = {}
    if not parsed:
        try:
            raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=500)
            parsed = _parse_json_obj(raw)
        except json.JSONDecodeError:
            # 2026-08-16 - JSON을 두 번 다 못 뱉어도 예외를 던지지 않는다
            # (generation_critic.py와 동일한 이유, 특정 해외 테크기업 케이스로 실측
            # 확인) - decision 기본값 KEEP으로 안전하게 떨어진다.
            parsed = {}
    decision = parsed.get("decision") if parsed.get("decision") in ("KEEP", "REWRITE") else "KEEP"
    return {
        "decision": decision,
        "missing_meaning": parsed.get("missing_meaning") if decision == "REWRITE" else None,
        "evidence_source_text": parsed.get("evidence_source_text") if decision == "REWRITE" else None,
        "reason": parsed.get("reason", ""),
    }


def write_headline_candidate(
    current_headline: str, missing_meaning: str, evidence_source_text: str, provider: str = DEFAULT_PROVIDER,
) -> str:
    """Writer(2단계) - Planner가 고른 missing_meaning/evidence_source_text
    만 받는다. JD 인재상 원문이나 이력서 근거 목록 전체를 다시 주지
    않는다(Context Engineering - 필요한 것만)."""
    prompt = _HEADLINE_WRITE_PROMPT.format(
        current_headline=current_headline,
        missing_meaning=missing_meaning or "",
        evidence_source_text=evidence_source_text or "",
    )
    raw = call_llm(prompt, provider=provider, max_tokens=400)
    try:
        parsed = _parse_json_obj(raw)
    except json.JSONDecodeError:
        parsed = {}
    if not parsed:
        try:
            raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=400)
            parsed = _parse_json_obj(raw)
        except json.JSONDecodeError:
            parsed = {}  # 2026-08-16 - 두 번 다 실패해도 예외 없이 빈 후보로 처리
    return parsed.get("rewritten_headline") or ""


def verify_headline_rewrite(original: str, rewritten: str, evidence_source_text: str, resume_semantic_objects: list[dict]) -> dict:
    """Code Verifier(4단계, 그대로 유지) - 새 유사도 점수를 만들지 않는다 -
    세 가지 사실 검사만:
    1) not_empty
    2) numbers_preserved: 원문 숫자가 그대로 있는가(bullet 검증과 동일 원칙).
    3) evidence_grounded: LLM이 인용한 evidence_source_text가 실제로
       주어진 Resume Object 목록 중 하나의 문장 안에 있는가(부분 문자열
       존재 확인만 - 없는 근거를 지어내 인용하는 걸 막는 안전장치).
    4) minimal_edit: 원문과의 문자열 유사도(SequenceMatcher)가 0.3
       미만이면 "정체성 유지" 규칙 위반으로 실패 처리(임계값은 "완전히
       다른 문장이 됐는가"를 거르는 최소 안전장치일 뿐, 점수 자체를
       노출하거나 재사용하지 않는다)."""
    failed = []
    if not rewritten or not rewritten.strip():
        failed.append({"check": "not_empty", "detail": "재작성 결과가 비어 있음"})
        return {"passed": False, "failed_checks": failed}

    orig_numbers = sorted(set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(original)))
    new_numbers = sorted(set(m.group().replace(" ", "") for m in _NUMBER_RE.finditer(rewritten)))
    if orig_numbers != new_numbers:
        failed.append({"check": "numbers_preserved", "detail": f"원문 숫자={orig_numbers} vs 재작성 숫자={new_numbers}"})

    objs = _resume_objects_for_headline(resume_semantic_objects)
    ev_norm = _norm_ws(evidence_source_text or "")
    grounded = bool(ev_norm) and any(ev_norm in _norm_ws(o["source_text"]) or _norm_ws(o["source_text"]) in ev_norm for o in objs)
    if not grounded:
        failed.append({"check": "evidence_grounded", "detail": f"인용한 근거 '{evidence_source_text}'가 이력서 근거 목록에 없음"})

    similarity = SequenceMatcher(None, original, rewritten).ratio()
    if similarity < 0.3:
        failed.append({"check": "minimal_edit", "detail": f"원문과 유사도 {similarity:.2f} - 정체성이 크게 바뀜"})

    return {"passed": not failed, "failed_checks": failed}


def generate_headline_proposal(
    headline_note: dict, resume_semantic_objects: list[dict], provider: str = DEFAULT_PROVIDER,
) -> dict:
    """headline_note(customization_planner.decide_headline()의 decision==
    "평가 필요" 결과)를 받아 Planner→Writer→Critic→Code Verifier 순으로
    실행한다. decision이 "평가 필요"가 아니면 호출하지 않는다(호출부
    책임). KEEP이면 LLM 1회만 쓰고 끝난다(Writer/Critic 호출 없음) -
    REWRITE로 판단된 경우에만 총 3회(Planner+Writer+Critic)를 쓴다."""
    original = headline_note.get("current_headline") or ""

    change = decide_headline_change(headline_note, resume_semantic_objects, provider)
    if change["decision"] == "KEEP":
        return {
            "current_headline": original, "recommended_headline": original,
            "matched_trait": None, "reason": change["reason"],
            "verification": {"passed": True, "failed_checks": []},
            "critic": None,
            "default_choice": "current", "status": "no_evidence",
        }

    missing_meaning = change["missing_meaning"] or ""
    evidence_source_text = change["evidence_source_text"] or ""
    rewritten = write_headline_candidate(original, missing_meaning, evidence_source_text, provider)

    if not rewritten:
        return {
            "current_headline": original, "recommended_headline": original,
            "matched_trait": missing_meaning, "reason": "Writer가 후보를 만들지 못함",
            "verification": {"passed": True, "failed_checks": []},
            "critic": None,
            "default_choice": "current", "status": "no_evidence",
        }

    verification = verify_headline_rewrite(original, rewritten, evidence_source_text, resume_semantic_objects)
    if not verification["passed"]:
        return {
            "current_headline": original, "recommended_headline": original,
            "matched_trait": missing_meaning, "evidence_source_text": evidence_source_text,
            "reason": change["reason"], "verification": verification, "critic": None,
            "default_choice": "current", "status": "verification_failed",
        }

    critic = generation_critic.evaluate_headline_candidate(original, rewritten, missing_meaning, provider)
    if critic["verdict"] != "USE_CANDIDATE":
        return {
            "current_headline": original, "recommended_headline": original,
            "matched_trait": missing_meaning, "evidence_source_text": evidence_source_text,
            "reason": change["reason"], "verification": verification, "critic": critic,
            "default_choice": "current", "status": "critic_rejected",
        }

    return {
        "current_headline": original,
        "recommended_headline": rewritten,
        "matched_trait": missing_meaning,
        "evidence_source_text": evidence_source_text,
        "reason": change["reason"],
        "verification": verification,
        "critic": critic,
        "default_choice": "recommended",
        "status": "proposed",
    }
