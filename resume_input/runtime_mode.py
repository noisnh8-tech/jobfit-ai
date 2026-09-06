# -*- coding: utf-8 -*-
"""
resume_input/runtime_mode.py — 운영/데모 모드 단일 스위치.

JOBFIT_MODE 환경변수로 결정한다.
  - "local" : 실사용자가 로컬에서 본인 DB(jobs.db)로 돌릴 때 (.env에만 설정, git 추적 안 됨)
  - "demo"  : 그 외 전부 (기본값). 공개 저장소를 그냥 clone 해서 실행하면
              .env 자체가 없어서 항상 이 값이 된다 — 안전장치.

이 모듈 하나만 보고 모든 DB_PATH / LLM 호출 / 외부 연동 가드가 결정된다.
"""
from __future__ import annotations

import os
from pathlib import Path

MODE: str = os.environ.get("JOBFIT_MODE", "demo").strip().lower()
IS_DEMO: bool = MODE != "local"

_ROOT = Path(__file__).resolve().parent.parent
DB_FILENAME: str = "demo.db" if IS_DEMO else "jobs.db"
DB_PATH: Path = _ROOT / "data" / DB_FILENAME

DEMO_BANNER_TEXT = "DEMO MODE · 비식별 예시 데이터 · 외부 연동 비활성화"


def guard_external_call(feature_name: str) -> None:
    """데모 모드에서 외부 연동(LLM/수집/링크확인 등) 진입을 막는 공통 가드.

    데모 모드에선 항상 사전 계산된 캐시가 있어 정상 동작 중엔 여기 도달하지
    않는다 — 방어선(캐시 미스 등 예외 상황에서도 실제 키/네트워크를 쓰지 않게).
    llm_client.LLMCallError 로 던져서, app.py 곳곳에 이미 있는
    "except LLMCallError: show_llm_error(...)" 처리로 자연스럽게 잡히게 한다
    (새 예외 타입을 앱 전역에 추가로 넣지 않아도 됨).
    """
    if IS_DEMO:
        from resume_input.llm_client import LLMCallError
        raise LLMCallError(
            f"[DEMO MODE] '{feature_name}' 은 데모 모드에서 비활성화되어 있습니다. "
            f"실제 운영 모드는 .env 에 JOBFIT_MODE=local 을 설정하고 본인 API 키를 넣어야 합니다."
        )
