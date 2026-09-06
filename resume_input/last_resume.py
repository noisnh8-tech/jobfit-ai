"""
resume_input/last_resume.py

마지막으로 업로드한 이력서 PDF 를 디스크에 1부 보관한다. 게스트 세션은
이력서를 세션에 안 들고 시작하므로, 새 세션마다 홈 대시보드("분석한 공고"/
"최근 분석한 공고")가 비어 보였다 - 세션 시작 시 세션에 이력서가 없으면
이 파일로 복원한다(app.py 라우팅 상단).

- 저장 위치: data/last_resume/ (resume.pdf + meta.json)
- 사용자당 1부만. "이력서 교체" 로 새 파일을 올리면 그때 덮어쓴다.
- 판정/랭킹 로직과 무관 - 순수 편의 기능. 실패해도 조용히 무시(앱 로딩 안 막음).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_DIR = Path(__file__).resolve().parent.parent / "data" / "last_resume"
_PDF = _DIR / "resume.pdf"
_META = _DIR / "meta.json"


def save(pdf_bytes: bytes, file_name: str, file_size: int) -> None:
    """업로드 성공 직후 호출. 원본 PDF 바이트와 파일명/크기를 보관한다."""
    if not pdf_bytes:
        return
    _DIR.mkdir(parents=True, exist_ok=True)
    _PDF.write_bytes(pdf_bytes)
    _META.write_text(
        json.dumps(
            {
                "file_name": file_name or "이력서.pdf",
                "file_size": int(file_size or len(pdf_bytes)),
                "saved_at": datetime.now().strftime("%Y.%m.%d %H:%M"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load() -> dict | None:
    """{pdf_bytes, file_name, file_size, saved_at} 또는 None(없음/손상)."""
    if not (_PDF.exists() and _META.exists()):
        return None
    try:
        meta = json.loads(_META.read_text(encoding="utf-8"))
        data = _PDF.read_bytes()
        if not data:
            return None
        return {
            "pdf_bytes": data,
            "file_name": meta.get("file_name") or "이력서.pdf",
            "file_size": int(meta.get("file_size") or len(data)),
            "saved_at": meta.get("saved_at") or "",
        }
    except Exception:
        return None
