"""
resume_input/llm_client.py

Claude/Gemini를 쉽게 교체할 수 있게 LLM 호출을 추상화한다. 목적은
비용 절감(Gemini 무료 티어 활용)이며, 추천 로직(judge/customizer/
checklist/cover_letter의 프롬프트·판단 기준)은 이 파일과 무관하게
그대로 유지된다 - 여기서는 "프롬프트를 어느 모델에 보낼지"만 바꾼다.

기본값은 그대로 Claude다(기존 동작 변경 없음). Gemini는 provider="gemini"
로 명시했을 때만 쓰인다.

API 안정성: Timeout + Retry(지수 백오프)를 적용한다. 재시도를 다 써도
실패하면 LLMCallError로 감싸서 던진다 - 호출자가 "무엇이 실패했는지"를
명확히 구분해서 처리(예: 이 공고만 건너뛰고 계속 진행)할 수 있게 한다.
"""
from __future__ import annotations

import time
from pathlib import Path

# job_ai_v3 소유 .env (2026-08-29 migration: da_job_market_2026 의존 제거).
# 키: ANTHROPIC_API_KEY, GEMINI_API_KEY, GEMINI_API_KEY_PAID. git-ignored.
_ENV_PATH = str(Path(__file__).resolve().parent.parent / ".env")

_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
# "gemini-2.0-flash"는 무료 티어에서 quota=0으로 즉시 429가 나서(2026-07-13
# 실측 확인) 대체 모델로 교체함.
_GEMINI_MODEL = "gemini-flash-latest"

# 2026-07-14: Claude 크레딧 고갈로 gemini 로 임시 전환했었음.
# 2026-08-31: 이번엔 반대로 Gemini 가 막혔다 - 무료 키는 504 DEADLINE_EXCEEDED
# (서버측 32초 타임아웃), 유료 키는 429 RESOURCE_EXHAUSTED "prepayment credits
# are depleted"(실측 확인). 신규 공고 분석이 매번 ~34초 낭비 후에야 폴백되는
# 걸 막기 위해 기본값을 claude 로 되돌린다. 아래 call_llm 의 gemini->claude
# 폴백은 그대로 두므로, Gemini 크레딧을 충전하면 이 한 줄만 "gemini" 로
# 되돌리면 원복된다(프롬프트/모델/키 구조는 그대로).
DEFAULT_PROVIDER = "gemini"  # 2026-09-02: 사용자가 Gemini 크레딧 충전 확인 → 원복
DEFAULT_TIMEOUT_SEC = 60
DEFAULT_MAX_RETRIES = 2


class LLMCallError(RuntimeError):
    """재시도까지 전부 실패했을 때만 던진다. 원인 예외는 __cause__에 보존된다."""


def _load_key(key_name: str, env_path: str = _ENV_PATH) -> str | None:
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key_name}="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        pass
    return None


def _call_claude(prompt: str, max_tokens: int, timeout: int) -> str:
    import anthropic

    api_key = _load_key("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(f"ANTHROPIC_API_KEY not found in {_ENV_PATH}")
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    resp = client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def _gemini_generate(api_key: str, prompt: str, max_tokens: int, timeout: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000))
    try:
        resp = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                # thinking_budget을 안 끄면 내부 추론에 토큰을 다 쓰고 실제
                # 답변이 빈 문자열로 오는 경우가 있었다(2026-07-13 실측 확인).
                thinking_config=types.ThinkingConfig(thinking_budget=200),
            ),
        )
    except Exception as e:
        # 2026-07-23(실측 확인) - "gemini-flash-latest" 별칭이 가리키는
        # 실제 모델이 바뀌면서(현재 gemini-3.6-flash) thinking_budget=0을
        # 거부하고 400 INVALID_ARGUMENT를 던지기 시작했다. 그래서 한동안
        # 이 except 블록이 thinking_config 자체를 빼고 재시도했는데, 그러면
        # 모델이 기본 사고 예산을 마음대로 써버려서 원래 막으려던 문제
        # (내부 사고가 토큰을 다 먹어 실제 답변이 잘림)가 그대로 재발했다
        # (2026-07-25, 홀드아웃 10건 중 7건이 JSON 중간에 끊겨서 파싱 실패
        # - 실측: thoughts_token_count=5641/6000, 답변은 355토큰뿐이었음).
        # thinking_budget=0은 거부돼도 200(작은 양수)은 승인되고 사고
        # 토큰이 90 수준으로 줄어드는 것을 확인(2026-07-25 실측) - 위
        # 기본 시도를 0->200으로 바꿔서 이 except 경로 자체가 거의 안 타게
        # 됐다. 그래도 미래 모델이 200마저 거부할 경우를 대비해 fallback은
        # 남겨두되, thinking_config 없이 호출하면 다시 이 버그가 재발할 수
        # 있다는 걸 알고 있어야 한다.
        if "INVALID_ARGUMENT" in str(e) or "400" in str(e):
            resp = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
        else:
            raise
    if not resp.text or not resp.text.strip():
        raise RuntimeError("Gemini가 빈 응답을 반환했다(재시도 대상)")
    return resp.text


def _call_gemini(prompt: str, max_tokens: int, timeout: int) -> str:
    api_key = _load_key("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY not found in {_ENV_PATH} - Gemini를 쓰려면 "
            f".env에 GEMINI_API_KEY=... 줄을 추가해야 한다."
        )
    try:
        return _gemini_generate(api_key, prompt, max_tokens, timeout)
    except Exception as e:
        # 2026-07-17(사용자 확정) - 무료 키가 쿼터 소진(429 RESOURCE_EXHAUSTED)
        # 이면 유료 키(GEMINI_API_KEY_PAID)로 즉시 전환한다. 무료/유료 키를
        # 둘 다 .env에 등록해두고, 무료가 막힐 때만 유료를 쓴다 - 유료 키가
        # 없으면(.env에 없으면) 원래 에러를 그대로 던진다(동작 변경 없음).
        #
        # 2026-08-30 - 쿼터(429)뿐 아니라 Gemini 의 일시적 서버/가용성 오류
        # (503 UNAVAILABLE, 504 DEADLINE_EXCEEDED)도 무료 티어가 못 쓰는 상태
        # 이므로 유료 키 fallback 대상에 포함한다. 인증(401/403)·잘못된 요청
        # (400) 은 유료 키로 다시 불러도 같은 에러라 fallback 안 함(그대로 raise).
        _FALLBACK_MARKERS = ("RESOURCE_EXHAUSTED", "429", "DEADLINE_EXCEEDED", "504", "503", "UNAVAILABLE")
        if any(m in str(e) for m in _FALLBACK_MARKERS):
            paid_key = _load_key("GEMINI_API_KEY_PAID")
            if paid_key:
                return _gemini_generate(paid_key, prompt, max_tokens, timeout)
        raise


_PROVIDERS = {
    "claude": _call_claude,
    "gemini": _call_gemini,
}


def call_llm(
    prompt: str,
    provider: str = DEFAULT_PROVIDER,
    max_tokens: int = 4096,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    from resume_input.runtime_mode import guard_external_call
    guard_external_call("실시간 LLM 호출")

    if provider not in _PROVIDERS:
        raise ValueError(f"지원하지 않는 provider: {provider} (가능: {list(_PROVIDERS)})")

    fn = _PROVIDERS[provider]
    last_exc: Exception | None = None

    # 2026-08-31 - Gemini 가 지금 크레딧 소진으로 못 쓰는 상태고(무료·유료
    # 둘 다), 그 증상이 타임아웃/빈응답이라 재시도마다 ~24초씩 낭비된다.
    # Claude 폴백 키가 있으면 Gemini 재시도를 생략하고(1회만 시도) 바로
    # 아래 폴백으로 넘긴다 - Gemini 가 정상일 땐 첫 시도가 성공하므로
    # happy path 에는 영향이 없다.
    if provider == "gemini" and _load_key("ANTHROPIC_API_KEY"):
        max_retries = 0

    for attempt in range(max_retries + 1):
        try:
            return fn(prompt, max_tokens, timeout)
        except Exception as e:  # 네트워크/타임아웃/429 등 - 재시도 대상
            last_exc = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 1초, 2초, ... 지수 백오프

    # 2026-08-31 - Gemini(무료 키 -> 유료 키 fallback 포함, _call_gemini)가
    # 재시도를 다 쓰고도 실패하면 ANTHROPIC_API_KEY 로 마지막 한 번 폴백한다.
    # 실측(2026-08-31): 무료·유료 키가 모두 크레딧 소진 상태이고, 그 증상이
    # RESOURCE_EXHAUSTED/429 가 아니라 "빈 응답"으로도 나타난다 - 그래서
    # 특정 에러 문구가 아니라 "Gemini 가 지금 아예 못 쓰는 상태"면 폴백한다.
    # provider 재설계가 아니라 "완전히 막혔을 때 신규 공고 분석이 아예 안
    # 되는" blocker 를 여는 최소 처리(정상화되면 Gemini 가 다시 1순위).
    if provider == "gemini" and _load_key("ANTHROPIC_API_KEY"):
        try:
            return _call_claude(prompt, max_tokens, timeout)
        except Exception as claude_exc:  # noqa: BLE001
            last_exc = claude_exc

    raise LLMCallError(
        f"{provider} 호출이 {max_retries + 1}번 시도 후에도 실패했다: {last_exc}"
    ) from last_exc
