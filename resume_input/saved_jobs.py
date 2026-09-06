"""
resume_input/saved_jobs.py

관심 공고(즐겨찾기) 저장. AI 사용 안 함, 순수 SQLite CRUD.
지원 여부와 무관하게 별도로 유지되는 목록 - application_manager와
독립적이다. status 컬럼을 "즐겨찾기" 외 다른 값("읽음"/"관심"/
"나중에보기" 등)으로 확장할 수 있게 남겨뒀지만, 이번 구현에서는
"즐겨찾기" 상태만 쓴다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_jobs (
            job_id TEXT PRIMARY KEY,
            company TEXT,
            title TEXT,
            url TEXT,
            source TEXT,
            status TEXT DEFAULT '즐겨찾기',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def add_saved_job(job: dict, status: str = "즐겨찾기", db_path: Path = DB_PATH) -> None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO saved_jobs (job_id, company, title, url, source, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job.get("job_id", ""), job.get("company", ""), job.get("title", ""),
         job.get("url", ""), job.get("source", ""), status),
    )
    conn.commit()
    conn.close()


def remove_saved_job(job_id: str, db_path: Path = DB_PATH) -> None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM saved_jobs WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()


def list_saved_jobs(status: str | None = None, db_path: Path = DB_PATH) -> list[dict]:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if status is None:
        rows = conn.execute("SELECT * FROM saved_jobs ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM saved_jobs WHERE status = ? ORDER BY created_at DESC", (status,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_saved(job_id: str, db_path: Path = DB_PATH) -> bool:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT 1 FROM saved_jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return row is not None
