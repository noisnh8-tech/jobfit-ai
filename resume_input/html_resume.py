# -*- coding: utf-8 -*-
"""
resume_input/html_resume.py — 개인 이력서 전용 HTML→PDF 맞춤 경로
=====================================================================
[공개 저장소 방침(2026-09-06 확정): 이 파일은 순서 재배치 로직과
 렌더링 메커니즘(범용 코드)만 담는다. 실제 이력서 내용(이름/학력/
 프로젝트/기술/링크 등)은 resume_input/personal_resume_data.py에서
 가져오는데, 그 파일은 .gitignore 대상이라 공개 저장소에는 없다 -
 "완성된 개인 문서 파일을 저장소에서 직접 다운로드할 수 있게 하지는
 않는다"는 방침에 따라 맞춤 이력서 PDF 생성 자체가 공개 데모에서는
 비활성화된다(_PERSONAL_DATA_AVAILABLE=False → generate()가
 status="unavailable"을 돌려주고 호출부는 안내 문구만 보여준다).
 로컬에서 본인 이력서로 이 기능을 쓰려면
 resume_input/personal_resume_data.py.example 을 참고해 같은 폴더에
 personal_resume_data.py 를 만들면 된다.]

S4(지원 준비) 다운로드 버튼에서, **지금 업로드된 이력서가 개인 이력서
데이터의 TARGET_SIGNATURE_STRINGS와 일치할 때만** 이 경로를 탄다.
그 외 이력서는 기존 범용 경로(pipeline.generate_resume_pdf →
pdf_layout.py)를 그대로 쓴다.

기존 엔진(pipeline.py / pdf_layout.py / pdf_generator.py / resume_template.py)은
전혀 건드리지 않는다. 이 파일은 완전히 독립적인 추가 경로다.

자동 반영 범위(사용자 확정 2026-09-01):
  ① 기술(Skills) 순서   — quick_analysis.resume_order.skill_priority
  ② 프로젝트 순서       — quick_analysis.resume_order.project_priority
  성과(bullet) 순서는 반영하지 않는다(신호 부족 — 사용자 결정으로 제외).

이력서 문구는 절대 생성·수정·요약하지 않는다. 순서만 바꾼다. 새 LLM 호출 없음.

렌더: Chrome 헤드리스(--headless=new --print-to-pdf). Chrome 이 없으면
generate()가 status="error"를 돌려주고 호출부는 기존 경로로 폴백한다.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _ROOT / "data" / "html_resume_cache"
_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
]

UNAVAILABLE_MESSAGE = "개인 문서 템플릿은 공개 저장소에서 제외되었습니다."

try:
    from resume_input import personal_resume_data as _pd
    NAME = _pd.NAME
    CONTACT = _pd.CONTACT
    TAGLINE = _pd.TAGLINE
    GITHUB_URL = _pd.GITHUB_URL
    PORTFOLIO_URL = _pd.PORTFOLIO_URL
    DASHBOARD_URL = _pd.DASHBOARD_URL
    SKILLS = _pd.SKILLS
    PROJECTS = _pd.PROJECTS
    EXTRAS_HTML = _pd.EXTRAS_HTML
    _TARGET_SIGNATURE_STRINGS = _pd.TARGET_SIGNATURE_STRINGS
    _PERSONAL_DATA_AVAILABLE = bool(_TARGET_SIGNATURE_STRINGS)
except ImportError:
    NAME = CONTACT = TAGLINE = GITHUB_URL = PORTFOLIO_URL = DASHBOARD_URL = EXTRAS_HTML = ""
    SKILLS = []
    PROJECTS = []
    _TARGET_SIGNATURE_STRINGS = ()
    _PERSONAL_DATA_AVAILABLE = False

# 공개 데모에는 사진 원본이 없음 - 없으면 _photo_data_uri()가 빈 값으로
# 폴백해서 사진 없이 렌더링된다(운영 모드에서만 실제 경로를 .env 등으로 지정).
_PHOTO_PDF_CANDIDATES: list[Path] = []

# 내용 버전 — 아래 상수/CSS를 바꾸면 올린다(디스크 캐시 무효화).
_CONTENT_VERSION = "html-resume-v5-public-20260906"

# jobfit-html-resume/_resume.css 와 동일하게 유지(그쪽이 시각 기준 원본).
_CSS = r"""
  :root {
    --surround: #e9edf3;
    --ink: #1f2430;
    --ink-strong: #10131a;
    --muted: #55606f;
    --rule: #10131a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--surround); }
  body {
    font-family: "Pretendard", "Pretendard Variable", -apple-system,
      "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    color: var(--ink);
    font-size: 10pt;
    line-height: 1.5;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  b, strong { font-weight: 700; }

  .page {
    width: 595.5pt;
    min-height: 842.25pt;
    margin: 14px auto;
    padding: 44pt 26pt 40pt 47pt;
    background: #fff;
    box-shadow: 0 2px 16px rgba(20, 28, 46, 0.14);
  }

  .head { display: flex; align-items: flex-start; gap: 16pt; }
  .head-main { flex: 1; padding-top: 2pt; }
  .name-row { display: flex; align-items: baseline; gap: 7pt; flex-wrap: wrap; }
  .name { font-size: 20pt; font-weight: 700; color: var(--ink-strong); letter-spacing: -0.3pt; }
  .name-sep { color: var(--ink-strong); font-weight: 700; font-size: 12pt; }
  .name-links a { font-size: 11pt; color: var(--ink); text-decoration: underline; }
  .contact { font-size: 11pt; color: var(--ink); margin-top: 8pt; }
  .photo {
    width: 94pt; height: 100pt; object-fit: cover; object-position: center 16%;
    border-radius: 50%; flex-shrink: 0; margin-top: 2pt; margin-right: 22pt;
  }

  .tagline { font-size: 13pt; font-weight: 700; color: var(--ink-strong); margin-top: 17pt; }

  .skills-label { font-size: 11pt; font-weight: 700; color: var(--ink-strong); margin-top: 20pt; }
  .skills { margin-top: 6pt; }
  .skill { line-height: 1.52; }
  .skill b { font-weight: 700; color: var(--ink-strong); }

  .section-title {
    font-weight: 700; color: var(--ink-strong);
    border-bottom: 1.4pt solid var(--rule);
    padding-bottom: 5pt;
    margin: 22pt 0 10pt;
  }
  .section-title.s-proj { font-size: 17pt; }
  .section-title.s-mid  { font-size: 15pt; }
  .section-title.s-sm   { font-size: 14pt; }

  .proj-sep { border-top: 1.4pt solid var(--rule); margin-bottom: 12pt; }

  .proj-title { font-size: 13pt; font-weight: 700; color: var(--ink-strong); }
  .proj-title .reg { font-weight: 400; }
  .proj-title a.dash { font-weight: 700; text-decoration: underline; color: var(--ink-strong); }
  .tech { font-size: 10pt; color: var(--ink); margin-top: 7pt; }
  .tech b { font-weight: 700; color: var(--ink-strong); }
  .sub { font-size: 11pt; font-weight: 700; color: var(--ink-strong); margin: 12pt 0 2pt; }
  p.line { line-height: 1.5; }
  p.line + p.line { margin-top: 0; }
  .bullet { margin-top: 4.5pt; padding-left: 11pt; text-indent: -11pt; line-height: 1.5; }
  .bullet b { font-weight: 700; color: var(--ink-strong); }
  p.line.note { font-size: 9pt; color: var(--muted); margin-top: 5pt; padding-left: 11pt; }

  .row3 { display: flex; align-items: baseline; margin-top: 4pt; font-size: 10pt; }
  .row3 .c1 { font-weight: 700; color: var(--ink-strong); }
  .row3 .sep { color: var(--ink-strong); font-size: 12pt; margin: 0 10pt; }
  .row3 .col-a { flex: 0 0 168pt; }
  .row3 .col-b { flex: 1; }

  @media print {
    html, body { background: #fff; }
    .page {
      width: auto; min-height: 0; height: auto;
      margin: 0; padding: 0;
      box-shadow: none;
      break-after: page; page-break-after: always;
    }
    .page:last-child { break-after: auto; page-break-after: auto; }
  }
  @page { size: A4; margin: 44pt 26pt 40pt 47pt; }
"""

_PRETENDARD_CDN = ("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/"
                   "dist/web/static/pretendard.min.css")


# ─────────────────────────────────────────────────────────────
# 이력서 식별
# ─────────────────────────────────────────────────────────────
def is_target_resume(resume_raw: str, resume_hash: str | None = None) -> bool:
    """지금 이력서가 personal_resume_data.py에 등록된 개인 이력서인지.
    그 파일이 없는 공개 데모에서는 항상 False(이 특수 경로 자체가 진입
    안 됨 - 호출부는 기존 범용 경로로 자연스럽게 폴백한다)."""
    if not _PERSONAL_DATA_AVAILABLE:
        return False
    t = resume_raw or ""
    return all(sig in t for sig in _TARGET_SIGNATURE_STRINGS)


# ─────────────────────────────────────────────────────────────
# 순서 재배치 (문구 불변, 순서만)
# ─────────────────────────────────────────────────────────────
def _nrm(s: str) -> str:
    """공백·중점·슬래시 제거 + 소문자. quick_analysis._norm 과 같은 규칙 -
    PDF 추출 과정에서 생긴 잉여 공백("AI 기 반")도 뭉개서 매칭한다."""
    return re.sub(r"[\s·/]+", "", (s or "")).lower()


def apply_skill_order(skills, priority):
    if not priority:
        return list(skills)
    idx = {lab: i for i, (lab, _) in enumerate(skills)}
    order, used = [], set()
    for name in priority:
        n = _nrm(name)
        for lab, _ in skills:
            if lab in used:
                continue
            if lab == name or _nrm(lab) == n or n in _nrm(lab) or _nrm(lab) in n:
                order.append(idx[lab]); used.add(lab)
                break
    for i, (lab, _) in enumerate(skills):
        if lab not in used:
            order.append(i); used.add(lab)
    return [skills[i] for i in order]


def apply_project_order(projects, priority):
    if not priority:
        return list(projects)
    order, used = [], set()
    for name in priority:
        n = _nrm(name)
        for i, p in enumerate(projects):
            if i in used:
                continue
            if p["match"] == name or n in _nrm(p["match"]) or _nrm(p["match"]) in n:
                order.append(i); used.add(i); break
    for i in range(len(projects)):
        if i not in used:
            order.append(i); used.add(i)
    return [projects[i] for i in order]


# ─────────────────────────────────────────────────────────────
# 렌더링
# ─────────────────────────────────────────────────────────────
_photo_cache: str | None = None


def _photo_data_uri() -> str:
    global _photo_cache
    if _photo_cache is not None:
        return _photo_cache
    try:
        import fitz  # PyMuPDF (이미 프로젝트 의존성)
        for pdf in _PHOTO_PDF_CANDIDATES:
            if not pdf.exists():
                continue
            d = fitz.open(str(pdf))
            for img in d[0].get_images(full=True):
                b = d.extract_image(img[0])
                _photo_cache = "data:image/jpeg;base64," + base64.b64encode(b["image"]).decode()
                return _photo_cache
    except Exception:
        pass
    _photo_cache = ""  # 사진 없이도 렌더는 되게(빈 img)
    return _photo_cache


def _render_blocks(blocks) -> str:
    out = []
    for kind, text in blocks:
        if kind == "sub":
            out.append(f'<div class="sub">{text}</div>')
        elif kind == "p":
            out.append(f'<p class="line">{text}</p>')
        elif kind == "bullet":
            out.append(f'<div class="bullet">{text}</div>')
        elif kind == "note":
            out.append(f'<p class="line note">{text}</p>')
    return "\n  ".join(out)


def _render_project(p) -> str:
    return (f'<div class="proj-title">{p["title"]}</div>\n'
            f'  <div class="tech"><b>[기술]</b> {p["tech"]}</div>\n'
            f'  {_render_blocks(p["blocks"])}')


def build_html(skills, projects, photo: str) -> str:
    """1페이지 = 헤더 + 보유 기술 + 첫 프로젝트. 이후 프로젝트는 각자 새
    페이지(위에 구분선), EXTRAS(대외활동/학력/자격증)는 마지막 페이지에."""
    skill_rows = "\n    ".join(
        f'<div class="skill"><b>{lab}:</b>　{desc}</div>' for lab, desc in skills
    )
    photo_html = f'<img class="photo" src="{photo}" alt="프로필 사진"/>' if photo else ""
    head_page = f'''<div class="page">
  <div class="head">
    <div class="head-main">
      <div class="name-row">
        <span class="name">{NAME}</span>
        <span class="name-sep">|</span>
        <span class="name-links"><a href="{GITHUB_URL}">GitHub</a></span>
        <span class="name-sep">|</span>
        <span class="name-links"><a href="{PORTFOLIO_URL}">Portfolio</a></span>
      </div>
      <div class="contact">{CONTACT}</div>
      <div class="tagline">{TAGLINE}</div>
    </div>
    {photo_html}
  </div>
  <div class="skills-label">[보유 기술]</div>
  <div class="skills">
    {skill_rows}
  </div>
  <div class="section-title s-proj">• 프로젝트</div>
  {_render_project(projects[0])}
{"  " + EXTRAS_HTML if len(projects) == 1 else ""}
</div>'''

    rest_pages = []
    for i, p in enumerate(projects[1:], start=1):
        extras = ("\n  " + EXTRAS_HTML) if i == len(projects) - 1 else ""
        rest_pages.append(f'''<div class="page">
  <div class="proj-sep"></div>
  {_render_project(p)}{extras}
</div>''')

    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<title>{NAME} 이력서</title>
<link rel="stylesheet" as="style" crossorigin href="{_PRETENDARD_CDN}"/>
<style>{_CSS}</style></head><body>

{head_page}
{"".join(rest_pages)}

</body></html>'''


def _find_chrome() -> str | None:
    for c in _CHROME_CANDIDATES:
        if c and os.path.exists(c):
            return c
    return shutil.which("chrome") or shutil.which("chrome.exe") or shutil.which("msedge")


def _html_to_pdf_bytes(html: str) -> bytes:
    chrome = _find_chrome()
    if not chrome:
        raise RuntimeError("Chrome(헤드리스) 실행 파일을 찾지 못했습니다.")
    tmp_html = tmp_pdf = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html); tmp_html = f.name
        fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={tmp_pdf}", tmp_html],
            check=True, capture_output=True, timeout=60,
        )
        time.sleep(0.3)
        return Path(tmp_pdf).read_bytes()
    finally:
        for p in (tmp_html, tmp_pdf):
            try:
                if p:
                    os.unlink(p)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────
# 공개 엔트리
# ─────────────────────────────────────────────────────────────
def generate(skill_priority: list[str] | None, project_priority: list[str] | None) -> dict:
    """맞춤 PDF 를 만들어 디스크 캐시에 저장하고 경로를 돌려준다.

    반환: {"status": "ok", "pdf_path": ...} | {"status": "unavailable", "reason": ...}
    | {"status": "error", "reason": ...}. unavailable/error 면 호출부는
    기존 범용 경로로 폴백하거나 안내 문구만 보여줘야 한다."""
    if not _PERSONAL_DATA_AVAILABLE:
        return {"status": "unavailable", "reason": UNAVAILABLE_MESSAGE}
    skill_priority = list(skill_priority or [])
    project_priority = list(project_priority or [])

    key = hashlib.sha1(
        f"{_CONTENT_VERSION}|S={skill_priority}|P={project_priority}".encode("utf-8")
    ).hexdigest()[:16]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = _CACHE_DIR / f"맞춤_이력서_{key}.pdf"
    if out_pdf.exists() and out_pdf.stat().st_size > 0:
        return {"status": "ok", "pdf_path": str(out_pdf), "cached": True}

    try:
        skills = apply_skill_order(SKILLS, skill_priority)
        projects = apply_project_order(PROJECTS, project_priority)
        html = build_html(skills, projects, _photo_data_uri())
        pdf_bytes = _html_to_pdf_bytes(html)
        if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
            return {"status": "error", "reason": "PDF 생성 결과가 유효하지 않습니다."}
        out_pdf.write_bytes(pdf_bytes)
        return {"status": "ok", "pdf_path": str(out_pdf), "cached": False}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
