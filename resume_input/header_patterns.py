"""
resume_input/header_patterns.py

Header Detector - 공고 원문에서 섹션 헤더 위치를 찾는 로직의 단일
출처. header_registry.py("무엇이 헤더 후보 문구인가")를 가져다가
"어떻게 찾을지"만 여기서 정의한다.

배경(2026-07-25, 실측): 기존 job_detail._classify_display_sections()는
"20자 이하 독립 줄"만 헤더로 인정했다(오탐 방지 목적, 2026-07-16 설계).
그런데 실측(기업 홈페이지 소스 426건 중 183건, 43.0%가 task+qualification
둘 다 공백) 결과 95%가 "헤더 문구는 원문에 있지만 그 줄이 20자를
넘는다"는 이유로 실패했다 - 독립 줄 조건 자체가 실패 원인이었다.

그래서 헤더를 "독립 줄"이 아니라 "문장 어디서든 패턴"으로 찾도록
바꾸되(기업 홈페이지 43.0% -> 0.7%), 짧고 일반적인 단어("우대", "FAQ",
"복지", "혜택", "기술")가 문장 중간에서 오탐을 냈다(예: "FAQ 자산화"라는
업무 설명을 헤더로 오인, "주주우대"라는 상품명의 일부를 헤더로 오인).

그래서 2단계로 나눈다:
- STRONG: 길고 구체적인 문구 - 문장 어디서든 매치되면 바로 헤더로 인정.
- WEAK: 위 오탐이 실측된 짧고 일반적인 단어 - 아래 추가 조건을 만족해야만
  헤더로 인정한다(콜론 뒤따름 / 괄호로 온전히 감싸임 / 그 자체로 독립된
  짧은 줄).

주의(2026-07-25 실측 교훈) - "위험해 보인다"는 감으로 WEAK을 늘리지
않는다: "role"을 처음에 감으로 WEAK에 넣었다가 해외 기업 공고 9건에서
실제 회귀가 발생했다("How Do I Know if the Role is Right For Me?"처럼
캐주얼한 JD에서 "role"이 진짜 섹션 전환 표시로 반복 사용됨) - 오탐이
실측으로 확인된 것만 WEAK으로 분류한다.

Known Limitation(2026-07-25, Section Boundary Audit 20건 중 1건) - 일부
기업(예: 특정 대기업 계열사)은 "담당 업무"를 섹션 헤더가 아니라 일반 문장
("모집 분야 및 담당 업무에 따라 영어 구술평가가 실시될 수 있어요"처럼
전형 안내 문장의 일부)으로 쓴다. 이 경우 문서 말미의 전형 안내 일부가
resp(task) 섹션에 섞여 들어갈 수 있다. "담당 업무"를 WEAK으로 옮기면
이 1건은 고쳐지지만, 대부분의 정상 공고(주요업무/담당업무가 실제
헤더로 쓰이는 수백 건)에서 진짜 헤더를 놓칠 위험이 훨씬 크다 - 영향
범위가 제한적이라(20건 중 1건, 문서 말미 일부만 영향) 현재 버전에서는
의도적으로 수정하지 않는다. 새 사례가 반복적으로 발견되면 그때
재검토한다.
"""
from __future__ import annotations

import re

from resume_input.header_registry import DUTIES, NOISE, PREFERRED, REQUIRED

HEADER_DETECTOR_VERSION = "v3-strong-weak-2026-07-25"

# 실측(원티드 22건 과다분할 의심 샘플, 기업 홈페이지 오탐 사례)으로 확인된
# 오탐 유발 패턴만 WEAK으로 분류한다. 새 오탐이 실측되면 여기에만 추가한다.
WEAK_PATTERNS: set[str] = {"우대", "faq", "복지", "혜택", "기술", "혜택과 복지"}

_ALL_PATTERNS = DUTIES + REQUIRED + PREFERRED + NOISE
STRONG_PATTERNS: list[str] = sorted({p for p in _ALL_PATTERNS if p.lower() not in WEAK_PATTERNS}, key=len, reverse=True)
_WEAK_PATTERNS_LIST: list[str] = sorted({p for p in _ALL_PATTERNS if p.lower() in WEAK_PATTERNS}, key=len, reverse=True)

_STRONG_RE = re.compile("(" + "|".join(re.escape(p) for p in STRONG_PATTERNS) + ")", re.I)
_WEAK_RE = re.compile("(" + "|".join(re.escape(p) for p in _WEAK_PATTERNS_LIST) + ")", re.I)
_MAX_HEADER_LINE_LEN = 20


def _weak_context_ok(text: str, start: int, end: int) -> bool:
    """WEAK 패턴이 실제 헤더인지 판단 - 콜론 뒤따름 / 괄호로 온전히
    감싸임 / 그 자체로 독립된 짧은 줄, 셋 중 하나만 만족하면 인정한다."""
    after = text[end : end + 2]
    if re.match(r"\s*[:：]", after):
        return True
    before_ch = text[start - 1] if start > 0 else ""
    after_ch = text[end] if end < len(text) else ""
    if before_ch in "([【" and after_ch in ")]】":
        return True
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end].strip()
    stripped = line.rstrip(":：ㆍ・ ")
    return len(stripped) <= _MAX_HEADER_LINE_LEN


def find_header_matches(text: str) -> list[re.Match]:
    """텍스트 전체에서 헤더 위치를 전부 찾는다(줄 위치·길이 제약 없음 -
    STRONG은 어디서든, WEAK은 추가 조건을 만족할 때만). 매치 순서대로
    정렬해서 반환한다 - 호출부가 이 경계를 기준으로 섹션을 나눈다."""
    matches = list(_STRONG_RE.finditer(text))
    for m in _WEAK_RE.finditer(text):
        if _weak_context_ok(text, m.start(), m.end()):
            matches.append(m)
    matches.sort(key=lambda m: m.start())
    return matches
