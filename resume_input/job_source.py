"""
resume_input/job_source.py

공고 URL 도메인으로 출처를 판별한다. LLM 미사용, 단순 문자열 매칭.
"""
from __future__ import annotations

_DOMAIN_MAP = {
    "wanted.co.kr": "원티드",
    "saramin.co.kr": "사람인",
    "jobkorea.co.kr": "잡코리아",
}


def detect_job_source(url: str) -> str:
    url = (url or "").lower()
    for domain, name in _DOMAIN_MAP.items():
        if domain in url:
            return name
    return "기업 홈페이지"
