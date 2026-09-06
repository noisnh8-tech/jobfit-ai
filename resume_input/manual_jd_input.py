"""
resume_input/manual_jd_input.py

Mode 2(개별 공고 분석): 사용자가 JD를 직접 붙여넣는 입력을 기존
judge_service/customizer/checklist 파이프라인이 쓰는 job dict 형태로
감싼다. DB 조회(jd_repository)를 거치지 않는다 — job_id가 없으므로
붙여넣은 텍스트의 해시로 대체한다(judge_results 캐시 키로 사용).
"""
from __future__ import annotations

import hashlib


def wrap_pasted_jd(pasted_text: str, company: str = "", title: str = "", url: str = "") -> dict:
    """url은 선택 입력이다 - 사용자가 URL 없이 본문만 붙여넣으면 이
    공고는 원문 링크가 없는 게 맞다(직접 타이핑/캡처한 공고일 수 있어서
    - 지어낼 수 없다). url을 같이 주면 그대로 보존해서 나중에(지원
    완료 시 지원기록 등) "공고 다시보기"가 계속 동작하게 한다."""
    text_hash = hashlib.sha256(pasted_text.encode("utf-8")).hexdigest()[:16]
    return {
        "job_id": f"manual-{text_hash}",
        "company": company,
        "title": title,
        "posting_text": pasted_text,
        "url": url,
    }
