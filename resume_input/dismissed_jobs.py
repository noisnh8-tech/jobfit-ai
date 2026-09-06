"""
resume_input/dismissed_jobs.py

사용자가 추천 목록에서 "이 공고는 나와 관련 없다"고 직접 제외한 공고 목록.
AI 사용 안 함, 순수 SQLite CRUD - saved_jobs.py와 같은 패턴.

중요: `dismissed`는 "공고가 마감/종료됐다"는 의미가 **아니다**. source
lifecycle(is_active / link_dead / deadline_expired)과 완전히 무관한
"사용자 선택"이다. 그래서 jobs / candidate_jobs 원본은 절대 건드리지
않고, 이 별도 테이블에만 job_id를 기록한다. 되돌리기(remove)로 즉시
목록에 다시 포함될 수 있다.

앱은 single-tenant이므로 user_id 개념을 두지 않는다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dismissed_jobs (
            job_id TEXT PRIMARY KEY,
            company TEXT,
            title TEXT,
            url TEXT,
            source TEXT,
            dismissed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def add_dismissed_job(job: dict, db_path: Path = DB_PATH) -> None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO dismissed_jobs (job_id, company, title, url, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job.get("job_id", ""), job.get("company", ""), job.get("title", ""),
         job.get("url", ""), job.get("source", "")),
    )
    conn.commit()
    conn.close()


def remove_dismissed_job(job_id: str, db_path: Path = DB_PATH) -> None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM dismissed_jobs WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()


def list_dismissed_job_ids(db_path: Path = DB_PATH) -> set[str]:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT job_id FROM dismissed_jobs").fetchall()
    conn.close()
    return {r[0] for r in rows}


def list_dismissed_posting_keys(db_path: Path = DB_PATH) -> set[tuple[str, str]]:
    """제외한 공고를 (회사, 정규화 직무명) 키 집합으로. 같은 자리를
    가리키는 다른 URL 쌍까지 목록에서 빼기 위한 것(posting_identity 참고)."""
    from resume_input.posting_identity import posting_key
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT company, title FROM dismissed_jobs").fetchall()
    conn.close()
    # 회사·제목이 다 있어야 의미 있는 키 - 빈 값이 섞이면 ("","")·("회사","")
    # 같은 키가 정상 공고까지 잘못 제외한다(2026-09-02).
    return {
        posting_key(co, ti) for co, ti in rows
        if (co or "").strip() and (ti or "").strip()
    }


def list_dismissed_jobs(db_path: Path = DB_PATH) -> list[dict]:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM dismissed_jobs ORDER BY dismissed_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_dismissed(job_id: str, db_path: Path = DB_PATH) -> bool:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT 1 FROM dismissed_jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return row is not None
