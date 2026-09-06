"""
ops_api.py — job_ai_v3 운영 자동화용 thin FastAPI 어댑터
========================================================

Streamlit 앱(app.py)과 별개로 도는 얇은 HTTP 어댑터. 비즈니스 로직을
새로 구현하지 않고, n8n 이 전달한 "이미 분류된" 채용 결과 메일을
받아서 (1) 기존 지원기록과 매칭하고 (2) 확실할 때만 job_ai_v3 의 공식
상태 변경 경로(resume_input.application_manager.update_status)를 호출한다.
상태 이력(application_status_history)은 그 공식 경로가 남긴다.

  POST /application-result       채용 결과 메일 1건 처리
  GET  /application-result/log   최근 처리 내역
  GET  /health

실행:
  cd job_ai_v3
  python -m uvicorn ops_api:app --host 127.0.0.1 --port 8000

포트 8000 (Streamlit 8501 / n8n 5678 과 충돌 없음).

메일 → {stage, result} 분류는 n8n 의 rule 기반 Code 노드가 담당한다
(LLM 미사용). 이 API 는 분류 결과를 신뢰하고 매칭·상태변경만 한다.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from resume_input.application_manager import (
    STATUS_OPTIONS,
    list_applications,
    update_status,
    add_pending_message,
)

KST = timezone(timedelta(hours=9))
from resume_input.runtime_mode import DB_PATH, IS_DEMO  # 운영/데모 모드에 따라 jobs.db 또는 demo.db

app = FastAPI(title="job_ai_v3 ops api", version="1.0")

# Figma 플러그인(www.figma.com origin)이 /resume-order 를 fetch 하려면 CORS 허용 필요.
# GET 읽기 전용 엔드포인트만 노출하고, 로컬 전용(127.0.0.1:8000)이라 origin 은 넓게 둔다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── (stage, result) → 기존 STATUS_OPTIONS 매핑 ────────────────────────────────
# stage:  서류 / 코테 / 면접 / 최종 / unknown   (n8n classify 노드가 판정)
# result: 합격 / 불합격 / unknown
# 매핑 안 되는 조합은 None → HUMAN_REVIEW(부분 신호) 또는 IGNORED(신호 없음).
# 값은 반드시 application_manager.STATUS_OPTIONS 안의 문자열 그대로 쓴다.
_PASS_BY_STAGE = {
    "서류": "서류 진행",
    "코테": "과제/코테",
    "면접": "면접",
    "최종": "최종 합격",
}


def _map_status(stage: str | None, result: str | None) -> str | None:
    """확실히 매핑 가능한 경우에만 STATUS_OPTIONS 값을 반환. 아니면 None."""
    stage = (stage or "").strip()
    result = (result or "").strip()
    if result == "불합격":
        # 어느 전형이든 불합격은 불합격 (stage 몰라도 확실)
        return "불합격"
    if result == "합격":
        return _PASS_BY_STAGE.get(stage)  # stage 모르면 None → HUMAN_REVIEW
    return None


# 안전장치: 위 매핑이 STATUS_OPTIONS 를 벗어나면 import 시점에 터지게 한다.
assert set(_PASS_BY_STAGE.values()) | {"불합격"} <= set(STATUS_OPTIONS), (
    "ops_api 매핑이 application_manager.STATUS_OPTIONS 와 어긋남"
)


# ── 회사/직무 정규화 + 매칭 ──────────────────────────────────────────────────
# 메일 회사명(한글) ↔ 지원기록 회사명(영문 등) 별칭. 정규화 결과를 canonical 로 통일.
# 운영 환경에서는 본인이 실제 지원한 회사명 별칭을 여기에 추가해서 쓴다
# (예시: {"메일에 오는 한글 회사명": "지원기록에 저장된 영문 표기"}).
_COMPANY_ALIASES: dict[str, str] = {}


def _norm(s: str | None) -> str:
    s = re.sub(r"\s+", "", (s or "").lower())
    s = re.sub(r"(주식회사|㈜|\(주\)|\(주식회사\)|inc\.?|corp\.?|ltd\.?|co\.?)", "", s)
    s = s.strip()
    return _COMPANY_ALIASES.get(s, s)


def _match_applications(apps: list[dict], company: str, position: str | None) -> list[dict]:
    """정규화 회사명 exact 우선 → 없으면 substring 양방향(단, 그 결과가
    정확히 1건일 때만 신뢰). 후보 2건 이상이면 직무명으로 좁힌다.
    추측으로 '가장 비슷한' 걸 고르지 않는다 - 좁혀도 2건 이상이면 그대로 둔다."""
    cn = _norm(company)
    if not cn:
        return []

    exact = [a for a in apps if _norm(a.get("company")) == cn]
    cands = exact
    if not cands:
        sub = [
            a for a in apps
            if _norm(a.get("company")) and (cn in _norm(a["company"]) or _norm(a["company"]) in cn)
        ]
        cands = sub

    if position and len(cands) > 1:
        pn = _norm(position)
        narrowed = [
            a for a in cands
            if pn and (pn in _norm(a.get("title")) or _norm(a.get("title")) in pn)
        ]
        if narrowed:
            cands = narrowed

    return cands


# ── dedupe 테이블 (jobs.db, 최소 구조) ───────────────────────────────────────
def _ensure_mail_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS mail_processed (
            message_id     TEXT PRIMARY KEY,
            application_id INTEGER,
            outcome        TEXT,
            processed_at   TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# ── 요청/응답 ───────────────────────────────────────────────────────────────
class ApplicationResultRequest(BaseModel):
    company: str = Field(..., min_length=1)
    position: str | None = None
    stage: str | None = None       # 서류 / 코테 / 면접 / 최종 / unknown
    result: str | None = None      # 합격 / 불합격 / unknown
    message_id: str = Field(..., min_length=1)
    received_at: str | None = None
    subject: str | None = None
    # 본문에 합격/불합격 텍스트가 없는 "새 메시지 도착" 알림(나인하이어 등 ATS).
    # true 면 stage/result 를 신뢰하지 않고 상태를 절대 자동으로 안 바꾼다 -
    # 매칭 1건이면 "확인 필요" 표시만 남긴다(2026-09-04 사용자 확정).
    pending: bool = False


# outcome: UPDATED | MESSAGE_PENDING | HUMAN_REVIEW | NOT_FOUND | DUPLICATE | IGNORED | ERROR
def process_application_result(req: ApplicationResultRequest, db_path: Path = DB_PATH) -> dict:
    """엔드포인트 본체(테스트에서 db_path 를 복사본으로 바꿔 부를 수 있게 분리)."""
    if IS_DEMO:
        return {
            "outcome": "IGNORED",
            "message": "[DEMO MODE] 결과메일 자동 반영은 데모 모드에서 비활성화되어 있습니다.",
            "processed_at": datetime.now(KST).isoformat(timespec="seconds"),
        }
    now = datetime.now(KST).isoformat(timespec="seconds")
    _ensure_mail_table(db_path)
    conn = sqlite3.connect(db_path)
    try:
        prev = conn.execute(
            "SELECT application_id, outcome FROM mail_processed WHERE message_id = ?",
            (req.message_id,),
        ).fetchone()
        # 이미 상태를 바꾼(UPDATED) 메일만 재처리 차단. NOT_FOUND/HUMAN_REVIEW/
        # IGNORED/ERROR 는 분류 규칙이 개선되면 다음 폴링에서 다시 시도되도록 통과.
        if prev and prev[1] == "UPDATED":
            return {
                "outcome": "DUPLICATE",
                "application_id": prev[0],
                "prev_outcome": prev[1],
                "message": "이미 반영된 메일",
                "processed_at": now,
            }

        stage_tok = (req.stage or "").strip() or "unknown"
        result_tok = (req.result or "").strip() or "unknown"

        app_id: int | None = None
        candidates: list[dict] = []

        if req.pending:
            # 결과 텍스트 없는 알림 메일 - 상태는 절대 추측/자동변경 안 함.
            apps = list_applications(db_path=db_path)
            candidates = _match_applications(apps, req.company, req.position)
            if len(candidates) == 0:
                outcome = "NOT_FOUND"
                message = f"'{req.company}' 매칭되는 지원기록 없음(결과 메시지 알림)"
            elif len(candidates) > 1:
                outcome = "HUMAN_REVIEW"
                message = f"후보 {len(candidates)}건 - 어느 지원인지 불명확(결과 메시지 알림)"
            else:
                a = candidates[0]
                app_id = int(a["application_id"])
                add_pending_message(
                    req.message_id, app_id, subject=req.subject or "",
                    received_at=req.received_at or "", db_path=db_path,
                )
                outcome = "MESSAGE_PENDING"
                message = f"app {app_id} ({a.get('company')} · {a.get('title')}) - 결과 메시지 도착, 확인 필요"
            mapped = None
        else:
            mapped = _map_status(req.stage, req.result)
            if mapped is None:
                if stage_tok == "unknown" and result_tok == "unknown":
                    outcome = "IGNORED"
                    message = "채용 결과 신호 없음"
                else:
                    outcome = "HUMAN_REVIEW"
                    message = f"전형/결과 조합 매핑 불가 (stage={stage_tok}, result={result_tok})"
            else:
                apps = list_applications(db_path=db_path)
                candidates = _match_applications(apps, req.company, req.position)
                if len(candidates) == 0:
                    outcome = "NOT_FOUND"
                    message = f"'{req.company}' 매칭되는 지원기록 없음"
                elif len(candidates) > 1:
                    outcome = "HUMAN_REVIEW"
                    message = f"후보 {len(candidates)}건 - 어느 지원인지 불명확"
                else:
                    a = candidates[0]
                    app_id = int(a["application_id"])
                    prev_memo = (a.get("memo") or "").strip()
                    marker = f"[메일자동 {now[:10]}] {stage_tok}·{result_tok} → {mapped}"
                    if not prev_memo:
                        new_memo = marker
                    elif marker in prev_memo:
                        new_memo = prev_memo
                    else:
                        new_memo = f"{prev_memo}\n{marker}"
                    # 공식 경로. application_status_history 는 여기서 남는다.
                    update_status(app_id, mapped, memo=new_memo, db_path=db_path)
                    outcome = "UPDATED"
                    message = f"app {app_id} ({a.get('company')} · {a.get('title')}) → {mapped}"

        conn.execute(
            "INSERT OR REPLACE INTO mail_processed (message_id, application_id, outcome, processed_at) "
            "VALUES (?, ?, ?, ?)",
            (req.message_id, app_id, outcome, now),
        )
        conn.commit()

        return {
            "outcome": outcome,
            "application_id": app_id,
            "mapped_status": mapped,
            "stage": stage_tok,
            "result": result_tok,
            "company": req.company,
            "position": req.position or "",
            "candidates": [
                {"application_id": c["application_id"], "company": c.get("company"),
                 "title": c.get("title"), "status": c.get("status")}
                for c in candidates
            ],
            "message": message,
            "processed_at": now,
        }
    except Exception as exc:  # 어떤 실패든 n8n 이 ERROR 로 구분할 수 있게
        return {"outcome": "ERROR", "application_id": None, "message": f"{type(exc).__name__}: {exc}",
                "processed_at": now}
    finally:
        conn.close()


@app.post("/application-result")
def application_result(req: ApplicationResultRequest) -> dict:
    return process_application_result(req, db_path=DB_PATH)


@app.get("/application-result/log")
def application_result_log(limit: int = 20) -> dict:
    _ensure_mail_table(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT message_id, application_id, outcome, processed_at "
            "FROM mail_processed ORDER BY processed_at DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
    finally:
        conn.close()
    keys = ["message_id", "application_id", "outcome", "processed_at"]
    return {"rows": [dict(zip(keys, r)) for r in rows]}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "db": str(DB_PATH), "status_options": STATUS_OPTIONS}


# ── /resume-order : Figma 이력서 플러그인이 순서 반영에 쓰는 읽기 전용 엔드포인트 ──
# 이미 계산돼 semantic_link_cache(quick-analysis-v1) 에 저장된 resume_order 를
# 그대로 돌려준다. LLM/판단 로직 없음. job_id 지정 시 그 공고, 없으면 가장 최근 분석.
@app.get("/resume-order")
def resume_order(job_id: str | None = None, resume_hash: str | None = None) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        q = (
            "SELECT job_id, resume_hash, result_json, created_at "
            "FROM semantic_link_cache WHERE linking_version LIKE 'quick-analysis-%'"
        )
        params: list = []
        if job_id:
            q += " AND job_id = ?"
            params.append(job_id)
        if resume_hash:
            q += " AND resume_hash = ?"
            params.append(resume_hash)
        q += " ORDER BY created_at DESC LIMIT 1"
        row = conn.execute(q, params).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"found": False, "skill_priority": [], "project_priority": [],
                "message": "해당 조건의 quick-analysis 결과가 없습니다. JobFit에서 먼저 분석하세요."}

    import json as _json
    try:
        data = _json.loads(row["result_json"]) or {}
    except (TypeError, ValueError):
        data = {}
    ro = data.get("resume_order") or {}
    return {
        "found": True,
        "job_id": row["job_id"],
        "resume_hash": row["resume_hash"],
        "analyzed_at": row["created_at"],
        "skill_priority": ro.get("skill_priority") or [],
        "project_priority": ro.get("project_priority") or [],
    }
