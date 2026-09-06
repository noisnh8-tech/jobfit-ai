"""
resume_input/checklist.py

체크리스트 = "공고와 무관하게 이력서 자체가 좋은가"만 본다(공고
때문에 바뀌는 것은 customizer.py의 customization 몫).

13개 항목 중 의미 판단이 필요한 7개(근거유무/Why-How-Result 구조/
이해가능성/자연스러움/과장없음/직무연결성/지원논리)는 judge_service가
Resume+JD를 이미 읽은 김에 같은 LLM 호출에서 함께 판단해서 반환한다
(체크리스트만을 위한 추가 LLM 호출은 하지 않는다) - 여기서는 그 결과를
그대로 꺼내 쓰기만 한다.

나머지 6개(부정표현/중복/숫자성과/이모지/링크/PDF)는 순수 코드로
검증한다. AI/LLM 호출 없음.
"""
from __future__ import annotations

import re
import urllib.request

_NUMBER_RE = re.compile(r"\d[\d,.]*\s*(%|건|원|배|위|만|천|억|점)")
_URL_RE = re.compile(r"https?://[^\s|]+")
_NEGATIVE_WORDS = ["잘못된", "실패한", "부족했던", "미흡한", "아쉬운"]
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)


def _item(name: str, passed: bool | None, reason: str = "") -> dict:
    return {"item": name, "passed": passed, "reason": reason}


# ── 의미 판단 7개: judge_service 결과를 그대로 반환 ──────────────────────

def extract_semantic_checklist(judge_result: dict) -> list[dict]:
    return judge_result.get("checklist_semantic") or []


# ── 기계 검증 4개(문장/표현) ─────────────────────────────────────────────

def run_mechanical_checklist(resume_customized: str) -> list[dict]:
    found_negative = [w for w in _NEGATIVE_WORDS if w in resume_customized]

    lines = [ln.strip() for ln in resume_customized.split("\n") if len(ln.strip()) >= 15]
    has_duplicates = len(lines) != len(set(lines))

    has_numeric_outcome = bool(_NUMBER_RE.search(resume_customized))
    emoji_count = len(_EMOJI_RE.findall(resume_customized))

    return [
        _item(
            "불필요한 부정 표현이 없는가",
            not found_negative,
            f"발견: {found_negative}" if found_negative else "",
        ),
        _item("중복된 문장이 없는가", not has_duplicates),
        _item("성과가 숫자로 뒷받침되는가", has_numeric_outcome),
        _item(
            "이모지가 과도하지 않은가(3개 이하)",
            emoji_count <= 3,
            f"이모지 {emoji_count}개 발견",
        ),
    ]


# ── 제출 전 2개(링크/PDF) ────────────────────────────────────────────────

def _check_links(resume_text: str, timeout: float = 3.0) -> bool | None:
    urls = _URL_RE.findall(resume_text)
    if not urls:
        return None
    for url in urls:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=timeout)
        except Exception:
            return False
    return True


def run_submission_checklist(resume_customized: str, pdf_generated: bool) -> list[dict]:
    return [
        _item("PDF 생성이 완료됐는가", pdf_generated),
        _item("이력서 내 링크(GitHub/포트폴리오)가 정상 동작하는가", _check_links(resume_customized)),
    ]
