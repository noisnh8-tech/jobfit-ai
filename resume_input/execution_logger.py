"""
resume_input/execution_logger.py

Pipeline 각 단계(retrieval/judge/customizer/checklist/
application)의 결과를 기록한다. 디버깅, 추천 품질 분석, 포트폴리오
설명(왜 이 공고를 추천했는지 재현 가능)에 쓴다. AI 판단에 관여하지
않는 순수 기록 기능이다.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db


def resume_hash(resume_raw: str) -> str:
    return hashlib.sha256(resume_raw.encode("utf-8")).hexdigest()[:16]


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_hash TEXT NOT NULL,
            job_id TEXT,
            stage TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def log_stage(stage: str, resume_hash: str, job_id: str | None, data: dict, db_path: Path = DB_PATH) -> None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO execution_log (resume_hash, job_id, stage, data_json) VALUES (?, ?, ?, ?)",
        (resume_hash, job_id, stage, json.dumps(data, ensure_ascii=False, default=str)),
    )
    conn.commit()
    conn.close()


def list_stages(resume_hash: str, job_id: str | None = None, db_path: Path = DB_PATH) -> list[dict]:
    """디버깅/재현용 - 특정 이력서(+선택적으로 특정 공고)의 실행 로그를 시간순으로."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if job_id is not None:
            rows = conn.execute(
                "SELECT * FROM execution_log WHERE resume_hash = ? AND (job_id = ? OR job_id IS NULL) ORDER BY log_id",
                (resume_hash, job_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM execution_log WHERE resume_hash = ? ORDER BY log_id", (resume_hash,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d.pop("data_json"))
            result.append(d)
        return result
    finally:
        conn.close()
