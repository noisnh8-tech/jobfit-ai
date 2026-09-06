"""
resume_input/jd_repository.py

job_ai_v3/data/jobs.db (단일 Source of Truth) 를 읽기 전용으로 조회한다.

2026-08-29 Phase 2 DB 통합:
- 수집기가 job_ai_v3/data/jobs.db 의 `jobs` 테이블에 직접 쓴다
  (raw_postings 뷰 = SELECT * FROM jobs).
- market_analysis.db 중간 hop 제거 — 품질 필터를 여기서 raw_postings 에
  직접 적용한다(로직은 옛 refresh_market_data.merge_market_db 와 동일).
- da_job_market_2026 / _vendor/data/*.db 런타임 의존 없음.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db

_COLUMNS = [
    "job_id", "company", "title", "posting_text", "url",
    "job_family", "job_category", "job_subcategory",
    "location", "experience_level", "education", "salary",
    "employment_type", "required_skills", "preferred_skills",
    "deadline", "posted_at", "is_active",
]

# ── 품질 필터 (옛 scripts/refresh_market_data.py 에서 그대로 이관) ──────────────
# merge_market_db() 가 jobs -> market_analysis.db.job_postings 로 넘길 때 쓰던
# 기준. Phase 2 에서 그 중간 테이블을 없애고 raw_postings 를 직접 거를 때
# 동일 기준을 적용한다(로직 변경 아님, 위치만 이동).
_MIN_TEXT_LEN = 1000
_MIN_SECTIONS = 2

_SECTION_KW = {
    "담당업무": ["담당업무", "주요 업무", "주요업무", "업무 소개", "역할 및 책임",
               "what you'll do", "responsibilities", "key responsibilities",
               "[key responsibilities]", "what you will do"],
    "자격요건": ["자격 요건", "자격요건", "필수 자격", "필수요건", "requirements",
               "qualifications", "required", "[requirements]", "[qualifications]",
               "minimum qualifications", "what you bring"],
    "우대사항": ["우대사항", "우대 사항", "우대조건", "preferred", "nice to have",
               "[preferred qualifications]", "bonus points"],
    "기술스택": ["기술 스택", "기술스택", "tech stack", "technologies", "tools"],
    "복지혜택": ["복지", "혜택", "benefits", "perks", "[benefits]"],
    "조직소개": ["조직 소개", "팀 소개", "about the team", "about us", "about the role"],
    "전형절차": ["전형 절차", "채용 절차", "hiring process", "interview process"],
}


def _count_sections(text: str) -> int:
    low = text.lower()
    return sum(1 for kws in _SECTION_KW.values() if any(kw.lower() in low for kw in kws))


def _passes_quality(text: str) -> bool:
    if not text or len(text) < _MIN_TEXT_LEN:
        return False
    return _count_sections(text) >= _MIN_SECTIONS


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_job(job_id: str, db_path: Path = DB_PATH) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM raw_postings WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_active_jobs(db_path: Path = DB_PATH) -> list[dict]:
    """(Deprecated, 2026-08-29 Phase 2) 예전엔 job_ai_v2 유래 `jobs` 테이블
    138건을 별도 소스로 합쳤으나, 그 138건은 전부 수집 코퍼스(raw_postings)의
    부분집합이고 이미 candidate_jobs 에 반영돼 있다. 이제 후보 풀은
    list_reference_jobs() 하나로 충분하다. 시그니처 호환을 위해 빈 리스트를
    반환한다(호출부 sync_jobs / refresh_stale_postings 무수정).

    원본 138건은 jobs_legacy_v2 테이블에 보존돼 있다.
    """
    return []


def _dedup_key(job: dict) -> str:
    """동일 공고 판별 우선순위: URL > job_id > (회사+제목+본문) 해시 > 본문
    단독. job_id는 재수집 때마다 새로 발급될 수 있어 단독으로는 신뢰할 수 없다."""
    url = (job.get("url") or "").strip()
    if url:
        return f"url::{url}"

    job_id = (job.get("job_id") or "").strip()

    company = (job.get("company") or "").strip()
    title = (job.get("title") or "").strip()
    text = (job.get("posting_text") or "").strip()
    if company and title and text:
        import hashlib
        h = hashlib.sha256(f"{company}|{title}|{text}".encode("utf-8")).hexdigest()[:16]
        return f"hash::{h}"

    if text:
        return f"text::{text}"

    return f"id::{job_id}"


def _dedup_jobs(jobs: list[dict]) -> tuple[list[dict], int]:
    """같은 공고가 여러 번 수집됐으면 1건만 남긴다. job_ai_v3 자체 데이터가
    있으면 그걸 우선한다(더 정제된 원본 소스) - 참조 코퍼스끼리만 겹치면
    가장 최신 수집본을 남긴다. 반환: (중복 제거된 리스트, 제거된 건수)."""
    best_by_key: dict[str, dict] = {}
    for j in jobs:
        key = _dedup_key(j)
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = j
            continue
        if prev.get("source") == "job_ai_v3":
            continue
        if j.get("source") == "job_ai_v3":
            best_by_key[key] = j
            continue
        if (j.get("created_at") or "") > (prev.get("created_at") or ""):
            best_by_key[key] = j
    deduped = list(best_by_key.values())
    return deduped, len(jobs) - len(deduped)


# experience_level(2026-08-30): 원천 구조화 경력(Wanted annual_from / GreetingHR
# careerFrom / Saramin 리스트)이 여기 저장돼 있다. career_filter.extract_jd_career_level
# 이 이 값을 authoritative 로 우선 사용한다 - jd_repository 가 후보에 실어 보내야 함.
_REF_COLUMNS = ["job_id", "company", "title", "posting_text", "url", "location",
                "experience_level", "collected_at"]


def list_reference_jobs(db_path: Path = DB_PATH) -> list[dict]:
    """후보 풀의 원천. raw_postings(= 수집기가 채우는 jobs 테이블)에서
    품질 필터(본문 1,000자 이상 + 섹션 2개 이상, 옛 merge_market_db 와
    동일)를 통과한 공고만 돌려준다.

    Phase 2 이전: da_job_market_2026/…/market_analysis.db.job_postings 를
    읽었다. 지금은 그 중간 테이블을 없애고 raw_postings 를 직접 거른다.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_REF_COLUMNS)} FROM raw_postings "
            f"WHERE is_active = 1 AND posting_text IS NOT NULL "
            f"AND LENGTH(posting_text) >= ?",
            (_MIN_TEXT_LEN,),
        ).fetchall()
    finally:
        conn.close()

    jobs = []
    for r in rows:
        d = dict(r)
        if not _passes_quality(d.get("posting_text") or ""):
            continue
        # 옛 _REF_COLUMNS 는 created_at 을 노출했다 — _dedup_jobs 가
        # "최신 수집본 우선" 판단에 created_at 을 본다. raw_postings 에는
        # collected_at 이 대응 필드라 그 값을 created_at 키로 넘긴다.
        d["created_at"] = d.get("collected_at") or ""
        d["source"] = "reference"
        jobs.append(d)

    deduped, removed = _dedup_jobs(jobs)
    if removed:
        print(f"[jd_repository] 참조 코퍼스 중복 제거: {len(jobs)}건 -> {len(deduped)}건 ({removed}건 제거)")
    return deduped


def list_all_candidate_jobs() -> list[dict]:
    """job_ai_v3 후보 풀 전체(중복 제거). Phase 2 이후로는
    list_reference_jobs() 하나가 원천이다."""
    combined = list_active_jobs() + list_reference_jobs()
    deduped, removed = _dedup_jobs(combined)
    if removed:
        print(f"[jd_repository] 소스 간 중복 제거: {len(combined)}건 -> {len(deduped)}건 ({removed}건 제거)")
    return deduped
