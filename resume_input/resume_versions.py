"""
resume_input/resume_versions.py

같은 이력서에 대해 이미 만든 PDF와 똑같은 결과가 다시 나올 상황이면
새로 만들지 않고 재사용한다(계획 확정 - 버전 생성 기준: 기술순서/
프로젝트순서/불릿순서/승인용어치환/자기소개, 5개 값이 그 이력서의
가장 최근 저장 버전과 전부 동일하면 재사용, 하나라도 다르면 새 버전).
LLM 호출 없음, PyMuPDF/customization_rules 어느 쪽도 참조하지 않는다 -
customizer.apply()의 "applied" dict만 입력으로 받는다.

사용 순서(app.py/pipeline.py가 호출):
    reusable = find_reusable(resume_hash, applied)
    if reusable is None:
        result = pdf_generator.generate(...)
        if result["status"] == "ok":
            reusable = save_version(resume_hash, applied, result["pdf_bytes"])
    # reusable["pdf_path"]를 그대로 사용
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db
VERSIONS_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "resume_versions"

_SIGNATURE_FIELDS = ("skill_order", "project_order", "bullet_reorders", "term_replacements", "headline")


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_versions (
            id TEXT PRIMARY KEY,
            resume_hash TEXT NOT NULL,
            signature TEXT NOT NULL,
            applied TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def compute_signature(applied: dict) -> str:
    """기술순서/프로젝트순서/불릿순서/승인용어치환/자기소개 5개 값만
    뽑아 순서까지 그대로 유지한 채 결정적으로 직렬화 -> 해시한다(정렬
    안 함 - 순서 자체가 결과에 영향을 주는 값들이므로 정렬하면 다른
    순서가 같은 signature로 뭉개짐)."""
    payload = {k: applied.get(k) for k in _SIGNATURE_FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _get_latest(resume_hash: str, db_path: Path = DB_PATH) -> dict | None:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id, signature, applied, pdf_path, created_at FROM resume_versions "
        "WHERE resume_hash = ? ORDER BY created_at DESC LIMIT 1",
        (resume_hash,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0], "signature": row[1], "applied": json.loads(row[2]),
        "pdf_path": row[3], "created_at": row[4],
    }


def find_reusable(resume_hash: str, applied: dict) -> dict | None:
    """applied의 signature가 이 이력서의 가장 최근 저장 버전과 같으면
    그 버전을 반환한다(새 PDF 생성 불필요). 다르거나 저장된 버전이
    아예 없으면 None - 호출부가 pdf_generator.generate()를 실행해야
    한다는 뜻."""
    latest = _get_latest(resume_hash)
    if latest is None:
        return None
    if latest["signature"] == compute_signature(applied):
        return latest
    return None


def save_version(resume_hash: str, applied: dict, pdf_bytes: bytes, db_path: Path = DB_PATH) -> dict:
    """새로 생성된 PDF를 새 버전으로 저장한다(재사용 가능 여부는 호출부가
    find_reusable()로 이미 먼저 확인했다고 가정 - 여기서는 무조건 새로
    만든다)."""
    _init_table(db_path)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

    version_id = uuid.uuid4().hex[:12]
    signature = compute_signature(applied)
    pdf_path = VERSIONS_DIR / f"{resume_hash}_{version_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO resume_versions (id, resume_hash, signature, applied, pdf_path) VALUES (?, ?, ?, ?, ?)",
        (version_id, resume_hash, signature, json.dumps(applied, ensure_ascii=False, default=str), str(pdf_path)),
    )
    conn.commit()
    conn.close()

    return {
        "id": version_id, "signature": signature, "applied": applied,
        "pdf_path": str(pdf_path), "created_at": None,
    }


def get_version(version_id: str, db_path: Path = DB_PATH) -> dict | None:
    """version_id 하나로 직접 조회(2026-08-16, 지원기록 상세 화면용 -
    "그 지원 당시 사용한 맞춤 이력서" 링크). 없으면 None."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id, resume_hash, signature, applied, pdf_path, created_at FROM resume_versions WHERE id = ?",
        (version_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0], "resume_hash": row[1], "signature": row[2],
        "applied": json.loads(row[3]), "pdf_path": row[4], "created_at": row[5],
    }


def list_versions(resume_hash: str, db_path: Path = DB_PATH) -> list[dict]:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, signature, applied, pdf_path, created_at FROM resume_versions "
        "WHERE resume_hash = ? ORDER BY created_at ASC",
        (resume_hash,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "signature": r[1], "applied": json.loads(r[2]), "pdf_path": r[3], "created_at": r[4]}
        for r in rows
    ]
