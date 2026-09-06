# -*- coding: utf-8 -*-
"""
resume_input/generation_critic.py

Context Engineering 재설계(2026-08-16, 사용자 확정) - S4/S6의 마지막 LLM
생성 단계에 "두 번째 시선"을 추가한다. Planner/Writer가 만든 결과를 그대로
쓰지 않고, 확률적으로만 판단 가능한 것(억지 맞춤/부자연스러움/근거 없는
인과/경험 과적재)만 여기서 검사한다. 코드로 확인 가능한 것(숫자 보존/
evidence 실재/길이/PDF overflow)은 여기서 다루지 않는다 - 각 엔진
(rewrite_engine.py의 verify_headline_rewrite, cover_letter_engine.py의
_verify_composition)이 이미 담당한다. 확률적 판단(LLM)과 결정론적 검증
(코드)을 분리하는 것 자체가 이 모듈의 존재 이유다.

점수(숫자)를 만들지 않는다 - 관찰 가능한 이진 상태(bool)와 최종 verdict만
반환한다. 핵심 조건 중 하나라도 어기면 LLM이 verdict 필드를 잘못 채워도
안전하게 KEEP/USE_ORIGINAL 쪽으로 강제한다(이중 안전장치). verdict가
실패면 호출부가 원문 유지하거나 최대 1회 재작성한다 - 이 파일 자체는
재시도 루프를 갖지 않는다(오케스트레이션은 각 엔진의 책임).
"""
from __future__ import annotations

import json

from resume_input.llm_client import DEFAULT_PROVIDER, call_llm


def _parse_json_obj(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _call_critic(prompt: str, provider: str) -> dict:
    """실패 시 예외를 던지지 않는다 - Critic은 "두 번째 시선"일 뿐이라
    LLM이 JSON을 두 번 다 잘못 뱉어도 전체 생성 파이프라인이 죽으면 안
    된다(2026-08-16 실측 확인 - 특정 해외 테크기업 케이스에서 실제로 크래시 재현).
    빈 dict를 반환하고, 호출부의 core_ok 계산은 .get()이 None을 돌려주므로
    자동으로 안전한 쪽(USE_ORIGINAL/REVISE)으로 떨어진다."""
    raw = call_llm(prompt, provider=provider, max_tokens=400)
    try:
        return _parse_json_obj(raw)
    except json.JSONDecodeError:
        pass
    try:
        raw = call_llm(prompt + "\n\n(JSON 형식으로만 다시 답하세요.)", provider=provider, max_tokens=400)
        return _parse_json_obj(raw)
    except json.JSONDecodeError:
        return {}


# ── ① 헤드라인 Critic ──────────────────────────────────────────────────

_HEADLINE_CRITIC_PROMPT = """당신은 이력서 한줄 소개 수정안을 검토하는
역할입니다. 새로 만들지 마세요 - 아래 후보를 그대로 쓸지(USE_CANDIDATE)
원문을 유지할지(USE_ORIGINAL)만 판단하세요.

[원문]
{original}

[후보]
{candidate}

[후보가 추가하려던 의미]
{missing_meaning}

다음을 확인하세요:
- adds_useful_meaning: 후보가 원문에 없던 실질적인 직무 정체성/수행
  영역을 추가하는가? 단순히 "자율적으로", "주도적으로", "빠르게",
  "적극적으로", "도전적으로" 같은 수식어/부사 하나만 끼워 넣은 거라면
  false.
- natural: 문장이 자연스럽게 읽히는가? 수식어를 억지로 끼워 넣어
  어색해졌다면 false.
- forced_customization: 공고에 맞추려고 억지로 끼워 맞춘 느낌이 나는가?
  (이 항목은 위 두 개와 반대 방향입니다 - 억지스러우면 true)
- evidence_supported: 추가된 내용이 실제로 이력서 근거에서 자연스럽게
  나온 것인가, 아니면 근거를 넘어서는 과장인가?

[출력 형식 - JSON만, 다른 설명 없이]
{{"adds_useful_meaning": true/false, "natural": true/false,
  "forced_customization": true/false, "evidence_supported": true/false,
  "verdict": "USE_CANDIDATE 또는 USE_ORIGINAL"}}
"""


def evaluate_headline_candidate(
    original: str, candidate: str, missing_meaning: str, provider: str = DEFAULT_PROVIDER,
) -> dict:
    """핵심 조건(adds_useful_meaning=true, natural=true,
    forced_customization=false, evidence_supported=true)을 하나라도
    어기면 LLM이 verdict를 뭐라고 채웠든 USE_ORIGINAL로 강제한다."""
    prompt = _HEADLINE_CRITIC_PROMPT.format(
        original=original, candidate=candidate, missing_meaning=missing_meaning or "",
    )
    parsed = _call_critic(prompt, provider)
    core_ok = (
        parsed.get("adds_useful_meaning") is True
        and parsed.get("natural") is True
        and parsed.get("forced_customization") is False
        and parsed.get("evidence_supported") is True
    )
    return {
        "adds_useful_meaning": parsed.get("adds_useful_meaning"),
        "natural": parsed.get("natural"),
        "forced_customization": parsed.get("forced_customization"),
        "evidence_supported": parsed.get("evidence_supported"),
        "verdict": "USE_CANDIDATE" if core_ok else "USE_ORIGINAL",
    }


# ── S6 지원동기 Critic ─────────────────────────────────────────────────

_MOTIVATION_CRITIC_PROMPT = """당신은 자소서 지원동기 초안을 검토하는
역할입니다. 아래 초안이 [허용된 사실] 안에서만 쓰였는지, 사실 사이에
근거 없는 인과관계를 만들지 않았는지 확인하세요.

[허용된 회사 사실]
{company_facts}

[허용된 JD 사실]
{jd_facts}

[초안]
{draft}

다음을 확인하세요:
- fact_grounded: 초안의 모든 주장이 위 [허용된 사실] 안에서 나온 것인가?
  (허용 목록에 없는 회사 정보나 수치를 새로 지어냈으면 false)
- unsupported_causality: 사실 사이에 근거 없는 인과관계를 만들었는가?
  예: "흑자를 달성했다" + "데이터 전략을 강조한다" 라는 두 사실이 각각
  주어졌을 뿐인데 "치밀한 데이터 전략이 있었기 때문에 흑자가 가능했다"
  처럼 원인-결과로 단정하면 true.
- company_specific: 이 초안을 다른 회사 이름으로 바꿔도 그대로 쓸 수
  있을 만큼 일반적인가? 그렇다면 company_specific=false.
- generic_flattery: "훌륭한 기업", "업계를 선도하는" 같은 추상적인 회사
  칭찬이 있는가?

[출력 형식 - JSON만, 다른 설명 없이]
{{"fact_grounded": true/false, "unsupported_causality": true/false,
  "company_specific": true/false, "generic_flattery": true/false,
  "verdict": "PASS 또는 REVISE", "fail_reason": "실패 이유 1문장(통과면 \\"\\")"}}
"""


def evaluate_motivation_draft(
    draft: str, company_facts: list[str], jd_facts: list[str], provider: str = DEFAULT_PROVIDER,
) -> dict:
    """fact_grounded=true, unsupported_causality=false, company_specific=true,
    generic_flattery=false 를 전부 만족해야 PASS. 하나라도 어기면 REVISE
    (호출부가 fail_reason을 Writer에 돌려 최대 1회 재작성)."""
    prompt = _MOTIVATION_CRITIC_PROMPT.format(
        company_facts="\n".join(f"- {f}" for f in company_facts) or "(없음)",
        jd_facts="\n".join(f"- {f}" for f in jd_facts) or "(없음)",
        draft=draft,
    )
    parsed = _call_critic(prompt, provider)
    core_ok = (
        parsed.get("fact_grounded") is True
        and parsed.get("unsupported_causality") is False
        and parsed.get("company_specific") is True
        and parsed.get("generic_flattery") is False
    )
    return {
        "fact_grounded": parsed.get("fact_grounded"),
        "unsupported_causality": parsed.get("unsupported_causality"),
        "company_specific": parsed.get("company_specific"),
        "generic_flattery": parsed.get("generic_flattery"),
        "fail_reason": parsed.get("fail_reason", ""),
        "verdict": "PASS" if core_ok else "REVISE",
    }


# ── S6 관련 직무 경험 Critic ───────────────────────────────────────────

_EXPERIENCE_CRITIC_PROMPT = """당신은 자소서 직무 경험 초안을 검토하는
역할입니다. 초안이 프로젝트 전체 요약이 아니라 "하나의 이야기"에
집중하고 있는지, 근거 없는 내용을 담지 않았는지 확인하세요.

[이 초안이 다뤄야 할 이야기]
{story_angle}

[허용된 근거]
{evidence_list}

[초안]
{draft}

다음을 확인하세요:
- one_story: 초안이 [이 초안이 다뤄야 할 이야기] 하나에 집중하는가,
  아니면 여러 사건/수치를 나열해 프로젝트 전체 요약처럼 됐는가?
- evidence_grounded: 초안의 모든 구체적 사실(수치 포함)이 [허용된 근거]
  안에서 나온 것인가?
- evidence_overload: 필요 이상으로 많은 수치/세부사항을 한 문단에
  욱여넣었는가?
- project_summary_like: "이런 것도 했고 저런 것도 했다" 식으로 프로젝트
  전체를 요약하는 느낌인가?

[출력 형식 - JSON만, 다른 설명 없이]
{{"one_story": true/false, "evidence_grounded": true/false,
  "evidence_overload": true/false, "project_summary_like": true/false,
  "verdict": "PASS 또는 REVISE", "fail_reason": "실패 이유 1문장(통과면 \\"\\")"}}
"""


def evaluate_experience_draft(
    draft: str, story_angle: str, evidence_list: list[str], provider: str = DEFAULT_PROVIDER,
) -> dict:
    """one_story=true, evidence_grounded=true, evidence_overload=false,
    project_summary_like=false 를 전부 만족해야 PASS."""
    prompt = _EXPERIENCE_CRITIC_PROMPT.format(
        story_angle=story_angle,
        evidence_list="\n".join(f"- {e}" for e in evidence_list) or "(없음)",
        draft=draft,
    )
    parsed = _call_critic(prompt, provider)
    core_ok = (
        parsed.get("one_story") is True
        and parsed.get("evidence_grounded") is True
        and parsed.get("evidence_overload") is False
        and parsed.get("project_summary_like") is False
    )
    return {
        "one_story": parsed.get("one_story"),
        "evidence_grounded": parsed.get("evidence_grounded"),
        "evidence_overload": parsed.get("evidence_overload"),
        "project_summary_like": parsed.get("project_summary_like"),
        "fail_reason": parsed.get("fail_reason", ""),
        "verdict": "PASS" if core_ok else "REVISE",
    }
