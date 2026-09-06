"""
resume_input/preparation_tracker.py

리소스 측정 데이터(2026-08-16, 사용자 확정) - "지원 리소스를 줄이기
위해 만들었다"는 프로젝트 목적을 나중에 실제 사용 로그로 검증하기 위한
계측. 사용자가 보는 지원기록(application_manager.py)과 완전히 분리된
테이블에 쌓는다 - 화면에는 절대 노출하지 않는다(개발자가 나중에 직접
쿼리해서 분석하는 용도).

preparation_sessions: 공고 1건을 준비하는 과정 하나(공고 분석 시작
~ 지원기록 생성 / 보류 / 미지원)의 요약 = "현재 결론".
preparation_events: 그 과정에서 실제로 벌어진 사건의 원시 로그
(시각+이름+manual/system 구분) = "언제 무엇을".

이 파일 자체는 판단하지 않는다 - 이벤트 이름이 오면 그게 manual인지
system인지는 고정 매핑(_EVENT_ACTOR)으로만 정해지고, 세션 연결/카운트
집계는 전부 결정적 로직이다. LLM 호출 없음.

2026-08-30 (포트폴리오 KPI 원천 기록 보강, 사용자 확정) - 기존 구조를
그대로 재사용해서 측정 3점을 추가한다:
  T0 = analysis_opened        (실제 새 공고 분석 시작; 추천/붙여넣기 공통)
  T1 = analysis_started(지원) / judgment_hold(보류) / judgment_skip(미지원)
  T2 = customization_confirmed (제출 가능한 맞춤 이력서 준비를 사용자가 명시적으로 확정)
preparation_sessions 에 judgment / judgment_reason nullable 2컬럼만 추가.
새 telemetry 테이블은 만들지 않는다. historical backfill 안 한다.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db

# manual = 사용자가 실제로 버튼을 눌러 명시적 결정을 내린 지점(② 사용자
# 직접 개입 카운트 대상). system = 시스템이 자동 처리했거나 화면을 그냥
# 봤을 뿐인 지점.
_EVENT_ACTOR = {
    "analysis_opened": "manual",         # T0 - 공고 분석 화면 진입 / "분석" 실행
    "analysis_started": "manual",        # T1(지원) - 지원 준비 진행 확정
    "analysis_completed": "system",
    "customization_viewed": "system",
    "customization_confirmed": "manual",  # T2 - "이 이력서로 지원 준비 완료" 확정
    "cover_letter_source_viewed": "system",
    "judgment_hold": "manual",           # T1(보류)
    "judgment_skip": "manual",           # T1(미지원)
    "application_created": "manual",
}
_ANALYSIS_EVENTS = {"analysis_opened", "analysis_started", "analysis_completed"}
_CUSTOMIZATION_EVENTS = {"customization_viewed", "customization_confirmed"}
_COVER_LETTER_EVENTS = {"cover_letter_source_viewed"}

# 이벤트 -> 세션에 남길 최종 사용자 판단(지원/보류/미지원). 이벤트
# (preparation_events)는 "언제" 를, 이 컬럼은 "현재 결론" 을 담당한다.
_EVENT_JUDGMENT = {
    "analysis_started": "지원",
    "judgment_hold": "보류",
    "judgment_skip": "미지원",
    "application_created": "지원",
}
# 세션을 종료(completed_at / elapsed_seconds 확정)시키는 이벤트.
# 기존엔 application_created 만 종료였다 - 보류/미지원도 "결론 난" 상태라
# 종료하고, 같은 공고를 다시 열면 새 세션을 연다.
_TERMINAL_EVENTS = {"application_created", "judgment_hold", "judgment_skip"}


def _init_tables(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preparation_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            application_id INTEGER,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            elapsed_seconds REAL,
            analysis_used INTEGER NOT NULL DEFAULT 0,
            customization_used INTEGER NOT NULL DEFAULT 0,
            cover_letter_source_used INTEGER NOT NULL DEFAULT 0,
            manual_action_count INTEGER NOT NULL DEFAULT 0,
            system_action_count INTEGER NOT NULL DEFAULT 0,
            judgment TEXT,
            judgment_reason TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preparation_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            job_id TEXT NOT NULL,
            application_id INTEGER,
            event_name TEXT NOT NULL,
            actor TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """
    )
    # 이미 preparation_sessions 가 있던 DB(컬럼 2개 없이 생성됨)용 - SQLite 는
    # "ADD COLUMN IF NOT EXISTS" 가 없어서 이미 있으면 나는 에러를 무시한다
    # (application_manager._init_table 과 동일 패턴). 기존 행은 NULL 로 둔다(backfill 안 함).
    for col_def in ("judgment TEXT", "judgment_reason TEXT"):
        try:
            conn.execute(f"ALTER TABLE preparation_sessions ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


# 미완료 세션 재사용 허용 최대 유휴 시간(마지막 이벤트로부터). 실측
# preparation_events 분포 근거: 한 번의 실제 준비 안에서 반복되는 활동
# (분석 재실행/맞춤화 조회/자소서 소재 조회)은 길어야 ~20분 간격으로
# 몰려 있고(예: session 22 는 16분, session 24 는 20분 간격), "나중에
# 다시 방문"한 흔적은 80~120분+ 간격이다(session 1: 20:33→21:56→23:57,
# session 6: 21:53→23:45). 30분이면 반복 작업은 한 세션에 유지하면서
# 방치 후 재방문은 새 세션으로 분리한다. started_at 이 아니라 마지막
# 이벤트 시각을 기준으로 삼으므로, 실제로 계속 작업 중인 긴 세션은
# 계속 재사용되고 진짜 유휴 상태만 새 세션을 만든다.
_SESSION_REUSE_MAX_IDLE_SECONDS = 30 * 60


def _open_session_id(conn: sqlite3.Connection, job_id: str) -> int | None:
    row = conn.execute(
        """
        SELECT s.session_id,
               COALESCE(
                   (SELECT MAX(e.recorded_at) FROM preparation_events e
                    WHERE e.session_id = s.session_id),
                   s.started_at
               ) AS last_activity
        FROM preparation_sessions s
        WHERE s.job_id = ? AND s.completed_at IS NULL
        ORDER BY s.session_id DESC LIMIT 1
        """,
        (job_id,),
    ).fetchone()
    if not row:
        return None
    session_id, last_activity = row
    try:
        idle = (datetime.now() - datetime.fromisoformat(last_activity)).total_seconds()
    except (ValueError, TypeError):
        return session_id  # 시각 파싱 실패 시 기존 동작(재사용) 유지
    if idle > _SESSION_REUSE_MAX_IDLE_SECONDS:
        # 오래 방치된 미완료 세션 - 건드리지 않고 그대로 보존하되
        # 이번 이벤트는 새 세션에서 시작한다(elapsed_seconds 과다 계상 방지).
        return None
    return session_id


def log_event(
    job_id: str,
    event_name: str,
    application_id: int | None = None,
    reason: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """이벤트 하나를 기록한다. 이 job_id에 아직 안 끝난 세션이 있으면
    거기 이어붙이고, 없으면(이 job의 첫 이벤트) 새 세션을 연다.

    - _EVENT_JUDGMENT 에 있는 이벤트면 세션의 judgment 컬럼을 그 값으로 갱신한다
      (지원/보류/미지원). reason 이 오면 judgment_reason 도 함께 갱신한다.
    - _TERMINAL_EVENTS(application_created / judgment_hold / judgment_skip)면
      세션을 종료한다(completed_at / elapsed_seconds 확정). application_created
      는 application_id 도 채운다.

    화면 흐름을 막으면 안 되므로 호출부는 항상 try/except로 감싸서
    쓴다(계측 실패가 실제 기능 실패로 번지지 않게)."""
    _init_tables(db_path)
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now()
        now_iso = now.isoformat()

        session_id = _open_session_id(conn, job_id)
        if session_id is None:
            cur = conn.execute(
                "INSERT INTO preparation_sessions "
                "(job_id, started_at, analysis_used, customization_used, cover_letter_source_used, "
                " manual_action_count, system_action_count) VALUES (?, ?, 0, 0, 0, 0, 0)",
                (job_id, now_iso),
            )
            session_id = cur.lastrowid

        actor = _EVENT_ACTOR.get(event_name, "system")
        conn.execute(
            "INSERT INTO preparation_events (session_id, job_id, application_id, event_name, actor, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, job_id, application_id, event_name, actor, now_iso),
        )

        set_sql: list[str] = []
        params: list = []
        if event_name in _ANALYSIS_EVENTS:
            set_sql.append("analysis_used = 1")
        if event_name in _CUSTOMIZATION_EVENTS:
            set_sql.append("customization_used = 1")
        if event_name in _COVER_LETTER_EVENTS:
            set_sql.append("cover_letter_source_used = 1")
        count_col = "manual_action_count" if actor == "manual" else "system_action_count"
        set_sql.append(f"{count_col} = {count_col} + 1")

        if event_name in _EVENT_JUDGMENT:
            set_sql.append("judgment = ?")
            params.append(_EVENT_JUDGMENT[event_name])
            if reason is not None and str(reason).strip():
                set_sql.append("judgment_reason = ?")
                params.append(str(reason).strip())

        if event_name in _TERMINAL_EVENTS:
            started_row = conn.execute(
                "SELECT started_at FROM preparation_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            elapsed = None
            if started_row and started_row[0]:
                try:
                    elapsed = (now - datetime.fromisoformat(started_row[0])).total_seconds()
                except ValueError:
                    elapsed = None
            set_sql += ["completed_at = ?", "elapsed_seconds = ?"]
            params += [now_iso, elapsed]
            if event_name == "application_created":
                set_sql.append("application_id = ?")
                params.append(application_id)

        params.append(session_id)
        conn.execute(
            f"UPDATE preparation_sessions SET {', '.join(set_sql)} WHERE session_id = ?",
            params,
        )
        conn.commit()
    finally:
        conn.close()
