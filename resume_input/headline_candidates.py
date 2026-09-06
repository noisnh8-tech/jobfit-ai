"""
resume_input/headline_candidates.py

자기소개(첫 줄) 승인 후보 저장소 - LLM 호출 없음. 이력서 하나당 여러
후보를 쌓을 수 있는 구조로 처음부터 설계한다(2026-07-20, 사용자 확정 -
"원본 첫 문장 하나로 제한하지 않는다"). 최초엔 이력서의 현재 첫 줄을
후보 1개로 시드하고, 이후 사용자가 승인하는 다른 버전이 그대로 누적
되게 한다(나중에 "데이터분석형"/"CRM형"/"운영형" 식으로 자연스럽게
여러 벌 쌓이는 게 목표 - 지금은 그 저장/조회 뼈대만 만든다, UI에서
새 후보를 승인하는 화면은 이번 범위 밖).

customization_rules.evaluate_headline()이 기대하는 shape 그대로 반환한다:
{"id": ..., "text": ..., "resume_object_ids": [...]}.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from resume_input import customizer

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_headline_candidates (
            id TEXT NOT NULL,
            resume_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            resume_object_ids TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL DEFAULT 'approved',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id, resume_hash)
        )
        """
    )
    conn.commit()
    conn.close()


def list_headline_candidates(resume_hash: str, db_path: Path = DB_PATH) -> list[dict]:
    """이 이력서의 저장된 모든 승인 후보를 반환한다(N개, 시간순).
    없으면 빈 리스트 - 호출부가 필요시 seed_original_headline()으로
    최소 1개(원본)를 채운다."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, text, resume_object_ids FROM resume_headline_candidates "
        "WHERE resume_hash = ? ORDER BY created_at ASC",
        (resume_hash,),
    ).fetchall()
    conn.close()
    return [
        {"id": row[0], "text": row[1], "resume_object_ids": json.loads(row[2] or "[]")}
        for row in rows
    ]


def add_headline_candidate(
    resume_hash: str, candidate_id: str, text: str, resume_object_ids: list[str],
    source: str = "approved", db_path: Path = DB_PATH,
) -> None:
    """사용자가 승인한 새 자기소개 후보를 저장소에 추가한다(UI 연결은
    이번 범위 밖 - 저장/조회 뼈대만). resume_object_ids: 이 문장을
    증명하는 Resume Semantic Object id 목록(비어 있으면 customization_
    rules.evaluate_headline()이 이 후보를 애초에 선택 대상에서 제외
    한다 - 근거 없는 후보는 안전을 위해 절대 선택 안 됨)."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO resume_headline_candidates "
        "(id, resume_hash, text, resume_object_ids, source) VALUES (?, ?, ?, ?, ?)",
        (candidate_id, resume_hash, text, json.dumps(resume_object_ids, ensure_ascii=False), source),
    )
    conn.commit()
    conn.close()


def seed_original_headline(resume_hash: str, resume_raw: str, resume_semantic_objects: list[dict],
                            db_path: Path = DB_PATH) -> None:
    """이 이력서로 저장된 후보가 하나도 없으면, 현재 첫 줄을 "original"
    후보로 시드한다(멱등 - 이미 있으면 아무것도 안 함, 매 호출마다
    중복 저장하지 않는다)."""
    existing = list_headline_candidates(resume_hash, db_path)
    if existing:
        return
    current = customizer.extract_current_headline(resume_raw)
    if not current:
        return
    thinking_ids = [o["id"] for o in resume_semantic_objects if o.get("layer") == "thinking" and o.get("id")]
    add_headline_candidate(resume_hash, "original", current, thinking_ids, source="original", db_path=db_path)


def ensure_and_list(resume_hash: str, resume_raw: str, resume_semantic_objects: list[dict],
                     db_path: Path = DB_PATH) -> list[dict]:
    """호출부(pipeline.py)가 쓰는 통합 진입점 - 시드 후 목록을 그대로
    반환한다."""
    seed_original_headline(resume_hash, resume_raw, resume_semantic_objects, db_path)
    return list_headline_candidates(resume_hash, db_path)
