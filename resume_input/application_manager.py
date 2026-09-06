"""
resume_input/application_manager.py

지원 기록 관리. AI 사용 안 함, LLM 호출 없음. 순수 SQLite CRUD.

status가 바뀔 때마다 application_status_history에 이력을 남긴다(지원
-> 서류 -> 코딩테스트 -> 면접 -> 최종합격 같은 퍼널 전체를 나중에 분석할
수 있도록). applications 테이블은 "현재 상태"만, history 테이블은
"상태가 바뀐 시점들"을 전부 보존한다. 지금은 저장만 하고 가중치 자동
학습 등은 하지 않는다.

2026-08-16(지원기록 재설계, 사용자 확정) - 지원기록은 "지원 후 관리 +
자동 기록 + 나중에 성과 검증"만 한다(새 분석/새 판단 없음). 상태값을
고정 7단계로 좁히고(STATUS_OPTIONS), next_action/next_action_at(다음
일정)/resume_version(그 지원 당시 쓴 맞춤 이력서 버전)/updated_at
컬럼을 추가했다. 기존 컬럼(interview_at/result)은 지우지 않는다(과거
데이터 호환) - 화면은 이제 next_action/next_action_at만 쓴다.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db

# 고정 퍼널 - 사용자가 매번 문장을 새로 적지 않도록 정해진 값만 고른다.
STATUS_OPTIONS = ["지원 완료", "서류 진행", "과제/코테", "면접", "최종 합격", "불합격", "지원 철회"]


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            application_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            company TEXT,
            title TEXT,
            url TEXT,
            source TEXT,
            status TEXT NOT NULL DEFAULT '지원예정',
            applied_at TEXT,
            interview_at TEXT,
            result TEXT,
            memo TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # 기존에 이미 applications 테이블이 있던 경우(더 적은 컬럼으로 생성된
    # 이전 버전) ALTER TABLE로 컬럼을 추가한다 - SQLite는 "ADD COLUMN IF
    # NOT EXISTS"가 없어서 이미 있으면 나는 에러를 무시한다.
    for col_def in (
        "url TEXT", "source TEXT",
        "next_action TEXT", "next_action_at TEXT",
        "resume_version TEXT", "updated_at TEXT",
    ):
        try:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS application_status_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            result TEXT,
            memo TEXT,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _record_history(
    conn: sqlite3.Connection, application_id: int, status: str,
    result: str | None = None, memo: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO application_status_history (application_id, status, result, memo) VALUES (?, ?, ?, ?)",
        (application_id, status, result, memo),
    )


def add_application(
    job_id: str, company: str, title: str,
    url: str = "", source: str = "",
    status: str = "지원 완료", memo: str = "",
    resume_version: str | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """지원기록 생성 - S8 [지원 완료] 버튼에서만 호출된다(pipeline.
    finalize_application). 회사명/직무명/URL/지원일/상태/이력서 버전이
    전부 자동으로 채워진다 - 사용자가 여기서 다시 입력할 항목은 없다
    (2026-08-16 사용자 확정 - "지원했어요" 한 번만 누르면 끝)."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    now_iso = datetime.now().isoformat()
    cur = conn.execute(
        """
        INSERT INTO applications
            (job_id, company, title, url, source, status, applied_at, memo, resume_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (job_id, company, title, url, source, status, now_iso, memo, resume_version, now_iso),
    )
    app_id = cur.lastrowid
    _record_history(conn, app_id, status, memo=memo)
    conn.commit()
    conn.close()
    return app_id


def list_applied_job_ids(db_path: Path = DB_PATH) -> set[str]:
    """applications 테이블에 한 번이라도 등록된 job_id 전체(상태
    무관 - 지원예정도 포함). 추천 후보에서 자동 제외하는 데 쓴다."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT job_id FROM applications").fetchall()
    conn.close()
    return {r[0] for r in rows}


def list_applied_posting_keys(db_path: Path = DB_PATH) -> set[tuple[str, str]]:
    """지원한 공고를 (회사, 정규화 직무명) 키 집합으로. 같은 자리를
    가리키는 다른 URL 쌍까지 추천 목록에서 빼기 위한 것
    (posting_identity 참고 - dismissed 와 동일 방식)."""
    from resume_input.posting_identity import posting_key
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT company, title FROM applications").fetchall()
    conn.close()
    # 회사·제목이 다 있어야 의미 있는 키 - 빈 값(붙여넣기 지원 등)이 섞이면
    # ("","") 키가 회사/제목 없는 정상 공고까지 잘못 제외한다(2026-09-02).
    return {
        posting_key(co, ti) for co, ti in rows
        if (co or "").strip() and (ti or "").strip()
    }


def update_status(
    application_id: int, status: str,
    next_action: str | None = None, next_action_at: str | None = None,
    memo: str | None = None, db_path: Path = DB_PATH,
) -> None:
    """상태/다음 일정/메모 변경 - 사용자가 지원기록 상세에서 직접 관리
    하는 유일한 3항목(2026-08-16 사용자 확정 - 나머지는 전부 자동 채움).
    매 호출마다 application_status_history에 이력이 쌓여서, 나중에
    "지원 -> 서류 통과율 -> 면접 통과율 -> 최종 합격률" 같은 분석에
    쓸 수 있다."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    fields, values = ["status = ?", "updated_at = ?"], [status, datetime.now().isoformat()]
    if next_action is not None:
        fields.append("next_action = ?")
        values.append(next_action)
    if next_action_at is not None:
        fields.append("next_action_at = ?")
        values.append(next_action_at)
    if memo is not None:
        fields.append("memo = ?")
        values.append(memo)
    values.append(application_id)
    conn.execute(f"UPDATE applications SET {', '.join(fields)} WHERE application_id = ?", values)
    _record_history(conn, application_id, status, memo=memo)
    conn.commit()
    conn.close()
    # 상태를 직접 확인/변경했으니 "결과 메시지 확인 필요" 표시가 있었다면 해제.
    dismiss_pending_messages(application_id, db_path)


_ORDER_BY_SQL = {
    "recent": "created_at DESC",
    "status": "status ASC, created_at DESC",
    "company": "company ASC, created_at DESC",
}


def list_applications(order_by: str = "recent", db_path: Path = DB_PATH) -> list[dict]:
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql_order = _ORDER_BY_SQL.get(order_by, _ORDER_BY_SQL["recent"])
    rows = conn.execute(f"SELECT * FROM applications ORDER BY {sql_order}").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_application(application_id: int, db_path: Path = DB_PATH) -> None:
    """지원 기록을 사용자가 직접 삭제할 때 쓴다(예: 테스트/잘못 등록된
    기록 정리). status_history도 같이 지운다 - 부모 기록이 없는 이력만
    남으면 list_status_history()가 참조할 곳 없는 고아 데이터가 된다."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM application_status_history WHERE application_id = ?", (application_id,))
    conn.execute("DELETE FROM applications WHERE application_id = ?", (application_id,))
    conn.commit()
    conn.close()


def list_status_history(application_id: int, db_path: Path = DB_PATH) -> list[dict]:
    """Recommendation Feedback 분석용 - 특정 지원 건의 상태 변화 전체 이력."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM application_status_history WHERE application_id = ? ORDER BY history_id",
        (application_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── 결과 메시지 도착 (확인 필요) ────────────────────────────────────────────
# 2026-09-04(사용자 확정) - Workflow C 가 본문에 합격/불합격 텍스트 없이
# "새 메시지가 도착했습니다"(나인하이어/잡코리아 등 ATS 알림)만 있는 메일을
# 지원기록 1건에 매칭했을 때, 상태를 추측해서 자동으로 바꾸지 않는다(근거
# 없는 판정 금지). 대신 이 표를 통해 "확인 필요"만 표시하고, 사용자가
# 실제로 확인해 update_status()를 부르면(또는 명시적으로 확인 처리하면)
# 그때 지워진다.
def _init_pending_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_result_messages (
            message_id     TEXT PRIMARY KEY,
            application_id INTEGER NOT NULL,
            subject        TEXT,
            received_at    TEXT,
            detected_at    TEXT,
            dismissed      INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    conn.close()


def add_pending_message(
    message_id: str, application_id: int, subject: str = "", received_at: str = "",
    db_path: Path = DB_PATH,
) -> None:
    """ops_api 가 "새 메시지 도착" 메일을 지원기록 1건에 확실히 매칭했을 때만
    호출한다. message_id 로 dedupe(같은 메일 재수신해도 중복 안 쌓임)."""
    _init_pending_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO pending_result_messages
            (message_id, application_id, subject, received_at, detected_at, dismissed)
        VALUES (?, ?, ?, ?, ?, 0)
        ON CONFLICT(message_id) DO UPDATE SET
            application_id = excluded.application_id, subject = excluded.subject,
            received_at = excluded.received_at
        """,
        (message_id, application_id, subject, received_at, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def list_pending_messages(db_path: Path = DB_PATH) -> list[dict]:
    """아직 확인 안 한(dismissed=0) 결과 메시지 알림 - 지원기록 정보와 조인."""
    _init_pending_table(db_path)
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT p.message_id, p.application_id, p.subject, p.received_at, p.detected_at,
               a.company, a.title, a.status
        FROM pending_result_messages p
        JOIN applications a ON a.application_id = p.application_id
        WHERE p.dismissed = 0
        ORDER BY p.detected_at DESC
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def dismiss_pending_messages(application_id: int, db_path: Path = DB_PATH) -> None:
    """이 지원건의 "확인 필요" 표시를 전부 해제(update_status 가 자동 호출,
    또는 "확인함" 버튼)."""
    _init_pending_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE pending_result_messages SET dismissed = 1 WHERE application_id = ? AND dismissed = 0",
        (application_id,),
    )
    conn.commit()
    conn.close()


def count_pending_messages(db_path: Path = DB_PATH) -> int:
    _init_pending_table(db_path)
    conn = sqlite3.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM pending_result_messages WHERE dismissed = 0").fetchone()[0]
    conn.close()
    return int(n)
