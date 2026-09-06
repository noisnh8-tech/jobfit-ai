# -*- coding: utf-8 -*-
"""
resume_input/rewrite_versions.py

Rewrite 결과를 "이 공고 전용 이력서 버전"으로 저장한다(2026-08-14, 신규).
resume_versions.py(좌표 등록된 표준 이력서 1건에 대한 PDF 버전 저장,
resume_hash 단위)와는 목적이 다르다 - Rewrite는 이력서 템플릿 좌표 등록
여부와 무관하게(text만 있으면) 항상 저장 가능해야 하고, 반드시 job_id
단위로 구분돼야 한다(같은 이력서라도 공고마다 다른 rephrase 결과가
나오므로). 원본 이력서 텍스트는 이 파일의 어떤 함수도 수정하지 않는다 -
save_rewrite_version()은 항상 새 행을 추가만 한다(원본을 덮어쓰는 UPDATE
없음)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_rewrite_versions (
            id TEXT PRIMARY KEY,
            resume_hash TEXT NOT NULL,
            job_id TEXT NOT NULL,
            original_text TEXT NOT NULL,
            rewritten_text TEXT NOT NULL,
            decisions TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_rewrite_version(
    resume_hash: str, job_id: str, original_text: str, rewritten_text: str,
    decisions: list[dict], db_path: Path = DB_PATH,
) -> dict:
    """원본(original_text)은 그대로 기록만 하고 수정하지 않는다 -
    rewritten_text는 별도 값으로 새 행에 저장된다. 항상 새 버전(호출자가
    직접 append-only로 다룬다 - 재사용 판단은 이 파일의 책임이 아니다,
    필요해지면 resume_versions.find_reusable()과 같은 방식을 추가하되
    지금은 "저장"만 요구됐다)."""
    _init_table(db_path)
    version_id = uuid.uuid4().hex[:12]
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO resume_rewrite_versions "
        "(id, resume_hash, job_id, original_text, rewritten_text, decisions) VALUES (?, ?, ?, ?, ?, ?)",
        (version_id, resume_hash, job_id, original_text, rewritten_text,
         json.dumps(decisions, ensure_ascii=False, default=str)),
    )
    conn.commit()
    conn.close()
    return {"id": version_id, "resume_hash": resume_hash, "job_id": job_id, "rewritten_text": rewritten_text}


def get_latest_rewrite_version(resume_hash: str, job_id: str, db_path: Path = DB_PATH) -> dict | None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id, original_text, rewritten_text, decisions, created_at FROM resume_rewrite_versions "
        "WHERE resume_hash = ? AND job_id = ? ORDER BY created_at DESC LIMIT 1",
        (resume_hash, job_id),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0], "original_text": row[1], "rewritten_text": row[2],
        "decisions": json.loads(row[3]), "created_at": row[4],
    }


def list_rewrite_versions(resume_hash: str, job_id: str, db_path: Path = DB_PATH) -> list[dict]:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, original_text, rewritten_text, decisions, created_at FROM resume_rewrite_versions "
        "WHERE resume_hash = ? AND job_id = ? ORDER BY created_at ASC",
        (resume_hash, job_id),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "original_text": r[1], "rewritten_text": r[2], "decisions": json.loads(r[3]), "created_at": r[4]}
        for r in rows
    ]
