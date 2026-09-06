"""
resume_input/recommendation_cache.py

"공고 찾기" 추천 목록을 DB에 보존한다. AI 사용 안 함 - 순수 SQLite CRUD.

배경(2026-09-01, 사용자 확정): run_recommendation_mode()는 Career Filter +
M3/M4 랭킹(4천여 건) + URL별 죽은링크 확인이라 1~2분 걸린다. 지금은 그
결과가 st.session_state.top_cache(세션 메모리)에만 있어서
 - 지원 완료 후 top_cache 를 통째로 비우거나(app.py, 수정 전)
 - 게스트 재로그인 / 브라우저 새로고침 / streamlit 재시작
할 때마다 전체 재계산이 돌았다.

수집은 하루 1회(n8n Workflow B) 갱신이므로, 그날의 추천 결과를 그대로
재사용하는 게 맞다. collection_logs 의 마지막 SUCCESS/PARTIAL 시각을
`collection_stamp` 으로 함께 저장해서, 실제로 새 수집이 완료됐을 때만
캐시가 무효화되도록 한다. 사용자가 직접 [새로고침]을 누르면 clear_all().

single-tenant 이므로 user_id 없음. cache_key = "{resume_hash}|{career}|{challenge}".
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db
_KST = timezone(timedelta(hours=9))


def _conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_cache (
            cache_key TEXT PRIMARY KEY,
            collection_stamp TEXT,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def last_collection_stamp(db_path: Path = DB_PATH) -> str | None:
    """collection_logs 의 마지막 refresh 완료(SUCCESS/PARTIAL) 시각(ISO).
    수집 기록이 없으면 None - 그 경우 캐시는 stamp=None 으로 저장되고,
    새 수집이 한 번이라도 기록되면 stamp 불일치로 자연히 무효화된다."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            row = conn.execute(
                "SELECT collected_at FROM collection_logs "
                "WHERE source = 'refresh' AND status IN ('SUCCESS', 'PARTIAL') "
                "ORDER BY collected_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def last_collection_date_kst(db_path: Path = DB_PATH) -> str | None:
    """last_collection_stamp() 를 캐시 "무효화 키"로 쓰기 위한 날짜(KST, YYYY-MM-DD)
    버전. 2026-09-02(사용자 확정) - 실측해보니 하루 안에 정기 수집(10:00 KST)과
    missed-run catch-up([[collection.catchup]])이 겹쳐 같은 KST 날짜에 refresh
    완료 로그가 2~3건 남는 경우가 있었다(예: 08-31 하루에 14:59/23:24 KST 2건).
    last_collection_stamp() 는 그중 "가장 최근" 정확 시각을 그대로 돌려주므로,
    그 시각이 하루 안에서 또 바뀔 때마다 이 함수를 쓰는 recommendation_cache 전체가
    무효화돼 "하루 1번만 재계산"이라는 설계 의도와 달리 하루에 여러 번 1~2분
    재계산이 도는 원인이었다(실측: recommendation_cache 3건이 전부 옛 stamp
    2026-09-01T14:27:04 로 저장돼 있는데 최신 stamp 는 그 뒤 완료된
    2026-09-01T16:04:26 이라 셋 다 항상 미스). 표시용 last_collection_stamp()
    (시:분까지)는 그대로 두고, 캐시 무효화만 "그 시각이 속한 KST 날짜"로
    뭉뚱그려서 같은 날 여러 번 갱신돼도 캐시가 유지되게 한다."""
    stamp = last_collection_stamp(db_path)
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_KST).date().isoformat()


def load(cache_key: str, collection_stamp: str | None, db_path: Path = DB_PATH) -> dict | None:
    """저장된 결과. 단 collection_stamp 가 일치할 때만(그 사이 새 수집이
    완료됐으면 None 을 돌려 재계산을 유도한다)."""
    conn = _conn(db_path)
    try:
        row = conn.execute(
            "SELECT collection_stamp, result_json FROM recommendation_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    stored_stamp, result_json = row
    if (stored_stamp or None) != (collection_stamp or None):
        return None
    try:
        return json.loads(result_json)
    except (TypeError, ValueError):
        return None


def save(cache_key: str, collection_stamp: str | None, result: dict, db_path: Path = DB_PATH) -> None:
    """result 를 저장한다. resume_raw(이력서 원문)는 목록 조회에 안 쓰이고
    용량만 차지하므로 빼고 저장한다."""
    slim = {k: v for k, v in (result or {}).items() if k != "resume_raw"}
    conn = _conn(db_path)
    try:
        conn.execute(
            """
            INSERT INTO recommendation_cache (cache_key, collection_stamp, result_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                collection_stamp = excluded.collection_stamp,
                result_json = excluded.result_json,
                created_at = excluded.created_at
            """,
            (cache_key, collection_stamp, json.dumps(slim, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def clear_all(db_path: Path = DB_PATH) -> None:
    """[새로고침] 버튼 - 다음 진입 때 전부 재계산."""
    conn = _conn(db_path)
    try:
        conn.execute("DELETE FROM recommendation_cache")
        conn.commit()
    finally:
        conn.close()
