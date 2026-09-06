"""
resume_input/text_normalize.py

da_job_market_2026/agent/company_crawler/detail_parser.py의
normalize_invisible_chars()와 같은 목적, 같은 문자 집합 - job_ai_v3는
그 저장소 코드를 직접 import하지 않는 관례(두 프로젝트가 독립적으로
배포될 수 있어야 함, DB/env 파일 경로만 참조하는 기존 방식과 동일)를
따라 이 작은 순수 함수만 별도로 복제해둔다.

실측 버그(2026-07-16): job_store.py가 재크롤링된 원본(da_job_market_2026
쪽 jobs.db/market_analysis.db)에서 posting_text를 그대로 끌어와
candidate_jobs에 저장하는데, 그 원본이 아직 정규화 전(오래된 수집
시점)이면 한 번 정리했던 candidate_jobs 값을 다시 오염시켰다(특정 제조업
등 GreetingHR 공고 재확인됨). candidate_jobs에 쓰는 모든 지점이
원본을 신뢰하지 않고 저장 직전에 항상 이 함수를 거치도록 한다.
"""
from __future__ import annotations

import re

_INVISIBLE_CHARS_RE = re.compile("[​‌‍﻿⁠]")  # ZWSP, ZWNJ, ZWJ, BOM, word joiner


def normalize_invisible_chars(text: str) -> str:
    if not text:
        return text
    text = _INVISIBLE_CHARS_RE.sub("", text)
    text = text.replace("\xa0", " ")
    return text
