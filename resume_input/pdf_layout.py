"""
resume_input/pdf_layout.py

업로드된 어떤 이력서 PDF든 **사전 등록 없이** 맞춤화 결과를 반영한다.

동작(사전 등록/hash/특정 파일명/섹션명/bbox 하드코딩 전혀 없음):
  1. 맞춤화로 바뀐 텍스트 단위를 추출(_iter_change_units) - applied dict +
     원문 재파싱만 사용, 섹션명 하드코딩 없음.
  2. 각 단위의 '원래 자리'를 업로드된 PDF 에서 직접 찾는다(fitz.search_for).
  3. 그 자리의 폰트·크기·색·baseline 을 PDF 에서 읽는다(_load_style).
  4. 새 텍스트의 서식 구간(run) 을 결정(_style_runs):
     - reorder: 그 텍스트가 문서 다른 곳에 그대로 있으므로 그 span 구조(굵게/
       보통 섞임)를 그대로 가져와 각 run 을 자기 임베드 폰트로 그린다
       → 서식 보존 + 각 subset 이 자기 글리프(→·• 등)를 다 갖고 있어 안 깨짐.
     - 용어 치환: 원문 span 구조 + 해당 span 만 치환('SQL:' 굵기 유지).
     - 새 문구(자기소개): 디스크 완전 폰트 단일 run.
  5. 원래 글자의 잉크 영역만 redact(위/아래 안 넓힘 - 인접 줄 보존) 후
     run 들을 baseline 에 왼쪽부터 그린다.
  6. 사후 검증 - 결과 PDF 에서 새 텍스트를 search_for 로 다시 찾는다.
     못 찾으면(폰트 글리프 문제 등) manual_review 로 되돌린다
     (깨진 PDF 를 절대 내보내지 않는다).

한 줄이던 문장이 두 줄로 넘치면(아래 줄 침범) manual_review. 원본 칸을
망가뜨리지 않는 게 최우선 - 애매하면 반영하지 않고 판단 카드로 안내.

pipeline.generate_resume_pdf() 는 등록 템플릿(resume_template.py)이 있으면
pdf_generator 를 먼저 쓰고(고속 경로, 하위호환), 없으면 이 모듈로 온다.
두 경로 모두 반환 스키마가 같다(status/pdf_bytes/fields_changed/changed_fields).

**텍스트 계층은 건드리지 않는다.** customizer.py / resume_apply_engine.py
(무엇을 바꿀지 정하는 로직)가 만든 resume_customized 텍스트 + applied
dict 만 소비한다 - pdf_generator.py 와 동일한 입력 계약.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import fitz

from resume_input import customizer, resume_template

# 삽입 상자 아래쪽에만 주는 여유(내림 글자 여유). 세로로 이 이상 늘리면
# 아래 줄을 침범할 수 있으므로 작게 잡는다. redact(원본 글자 지우기) 상자는
# 절대 세로로 넓히지 않는다 - 잉크 상자 그대로 써야 옆/아래 줄이 안 지워진다.
_INSERT_PAD_BOTTOM = 3.0
# redact 상자를 아주 살짝 안쪽으로 좁혀(위/아래) 줄 간격이 좁은 이력서에서
# 인접 줄의 위/아래 획이 함께 지워지는 것을 막는다.
_REDACT_INSET_V = 0.6

# 폰트 파일을 찾을 위치(Windows). 시스템 + 사용자 설치 폰트.
_FONT_DIRS = [
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
]
# 시스템 폰트도 못 찾을 때 쓰는 PyMuPDF 내장 CJK 폰트. 글자폭이 원본과
# 달라 가짜 오버플로우가 날 수 있으므로 _fits() 가 그 경우를 잡아
# manual_review 로 돌린다(원본 PDF 를 깨진 채 남기지 않는다).
_BUILTIN_CJK = "china-ss"


# ── 폰트 해석 ───────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_EMB_DIR = Path(tempfile.gettempdir()) / "job_ai_resume_fonts"
_EMB_CACHE: dict[str, str] = {}  # 폰트 바이트 해시 -> 파일 경로


def _embedded_font_file(doc: "fitz.Document", family: str) -> str | None:
    """원본 PDF 에 임베드된 바로 그 폰트를 추출해 파일로 저장하고 경로를
    반환한다. 문서에 실제로 쓰인 글리프(→, ·, • 등)를 그대로 갖고 있어
    디스크 시스템 폰트보다 재현이 정확하다. subset prefix
    (예: 'AAAAAA+Pretendard-Bold')는 무시하고 이름으로 매칭. 폰트 바이트
    해시를 파일명으로 써서 같은 폰트를 여러 번 뽑아도 파일이 안 쌓인다."""
    import hashlib

    want = _norm(re.sub(r"^[A-Z]{6}\+", "", family))
    if not want:
        return None
    for pno in range(doc.page_count):
        for xref, ext, ftype, basefont, *_ in doc.get_page_fonts(pno, full=True):
            base = _norm(re.sub(r"^[A-Z]{6}\+", "", basefont or ""))
            if not base or (want != base and want not in base and base not in want):
                continue
            try:
                _name, fext, _ft2, buf = doc.extract_font(xref)
                if not buf or fext not in ("ttf", "otf", "cff"):
                    continue
                key = hashlib.sha1(buf).hexdigest()[:16]
                if key in _EMB_CACHE and os.path.exists(_EMB_CACHE[key]):
                    return _EMB_CACHE[key]
                _EMB_DIR.mkdir(parents=True, exist_ok=True)
                path = _EMB_DIR / f"{key}.{'otf' if fext == 'cff' else fext}"
                if not path.exists():
                    path.write_bytes(buf)
                _EMB_CACHE[key] = str(path)
                return str(path)
            except Exception:
                continue
    return None


def _resolve_font_file(family: str) -> str | None:
    """PDF span 이 준 폰트 패밀리명(예: "Pretendard-Bold", "MalgunGothic")
    을 실제 폰트 파일 경로로 해석한다. 못 찾으면 None(→ 내장 폰트)."""
    # 1) 이 프로젝트가 이미 아는 Pretendard 경로 우선
    direct = resume_template.FONT_FILES.get(family)
    if direct and os.path.exists(direct):
        return direct

    target = _norm(family)
    if not target:
        return None

    # 2) 시스템/사용자 폰트 디렉터리 스캔 - 파일명이 패밀리명을 포함하면 채택
    best: str | None = None
    for d in _FONT_DIRS:
        if not d.is_dir():
            continue
        for fp in d.iterdir():
            if fp.suffix.lower() not in (".otf", ".ttf", ".ttc"):
                continue
            stem = _norm(fp.stem)
            if not stem:
                continue
            if target == stem:
                return str(fp)  # 완전 일치 - 즉시 채택
            if best is None and (target in stem or stem in target):
                best = str(fp)
    if best:
        return best

    # 3) 굵기 표기만 다른 경우(SemiBold/Medium 등) - 같은 패밀리의 아는 파일로
    base = re.sub(r"(bold|semibold|extrabold|medium|light|regular|thin|black)$", "", target)
    for known_family, path in resume_template.FONT_FILES.items():
        if base and base in _norm(known_family) and os.path.exists(path):
            return path
    return None


def _load_style(page: "fitz.Page", rect: "fitz.Rect") -> tuple[str, float, tuple, float]:
    """rect 영역에서 (글자수 기준 최빈 폰트패밀리, 크기, 색 RGB0~1,
    첫 줄 baseline y) 를 읽는다. 한 줄 안에 굵게/보통이 섞여 있으면
    더 많이 쓰인 쪽을 택한다(예: 'SQL:'만 굵고 나머지는 보통 → 보통).
    못 읽으면 기본값."""
    data = page.get_text("dict", clip=rect)
    by_font: dict[tuple, int] = {}
    first_origin_y = None
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "")
                if not txt.strip():
                    continue
                key = (span.get("font", ""), round(float(span.get("size", 10.0)), 1), span.get("color", 0))
                by_font[key] = by_font.get(key, 0) + len(txt.strip())
                if first_origin_y is None:
                    first_origin_y = (span.get("origin") or (rect.x0, rect.y1))[1]
    if not by_font:
        return "", 10.0, (0.0, 0.0, 0.0), rect.y1
    (font, size, color_int), _ = max(by_font.items(), key=lambda kv: kv[1])
    return font, float(size), fitz.sRGB_to_pdf(color_int), float(first_origin_y or rect.y1)


# ── 원본 PDF 에서 텍스트 위치 찾기 ──────────────────────────────────────
def _clean(text: str) -> str:
    """검색용 정규화 - 불릿 기호와 앞뒤 공백 제거, 내부 연속 공백 축소."""
    t = re.sub(r"^\s*[•◦▪·\-\u2022]\s*", "", text or "")
    t = t.replace("　", " ").replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t　]+", " ", t).strip()


def _search_page(page: "fitz.Page", text: str) -> list["fitz.Rect"]:
    hits = page.search_for(text)
    if hits:
        return hits
    # 줄바꿈이 섞인(원문 한 줄이 PDF 에서 여러 줄) 경우: dehyphenate 옵션
    try:
        hits = page.search_for(text, flags=fitz.TEXT_DEHYPHENATE)
    except Exception:
        hits = []
    return hits


def _locate(doc: "fitz.Document", text: str) -> tuple[int, "fitz.Rect"] | None:
    """text 가 원본 PDF 에서 유일하게 나타나는 잉크 영역을 (page_no, Rect)
    로 반환한다(글자에 딱 붙은 상자). 못 찾거나 여러 곳이면 None
    (→ 호출부가 manual_review). 삽입 가능 폭은 호출부가 따로 계산한다."""
    cleaned = _clean(text)
    if len(cleaned) < 4:
        return None

    def _union(rects: list) -> "fitz.Rect":
        return fitz.Rect(
            min(r.x0 for r in rects), min(r.y0 for r in rects),
            max(r.x1 for r in rects), max(r.y1 for r in rects),
        )

    # 1) 전체 문자열 검색
    for pno in range(doc.page_count):
        hits = _search_page(doc[pno], cleaned)
        if len(hits) == 1:
            return pno, hits[0]
        if len(hits) >= 2:
            # 여러 줄로 접힌 한 덩어리(세로로 연속) => 합쳐서 하나로 본다.
            hits_sorted = sorted(hits, key=lambda r: r.y0)
            spans = [hits_sorted[0]]
            for r in hits_sorted[1:]:
                if r.y0 - spans[-1].y1 <= 6:
                    spans.append(r)
            if len(spans) == len(hits):
                return pno, _union(spans)
            return None  # 진짜로 문서 여러 곳에 등장 - 모호

    # 2) 앞/뒤 조각으로 범위 잡기(전체 검색이 인코딩 차이로 실패한 경우 포함)
    words = cleaned.split(" ")
    head = " ".join(words[:5]) if len(words) >= 4 else cleaned[: max(10, len(cleaned) // 2)]
    tail = " ".join(words[-5:]) if len(words) >= 4 else cleaned[-max(10, len(cleaned) // 2):]
    for pno in range(doc.page_count):
        h_hits = _search_page(doc[pno], head)
        t_hits = _search_page(doc[pno], tail)
        if len(h_hits) == 1 and len(t_hits) == 1:
            h, t = h_hits[0], t_hits[0]
            if -2 <= (t.y0 - h.y0) <= 90:  # 같은 블록(아래로 몇 줄 이내)
                return pno, _union([h, t])
    return None


def _insert_box(page: "fitz.Page", ink: "fitz.Rect") -> "fitz.Rect":
    """잉크 상자를 '이 줄이 실제로 쓸 수 있었던 폭'으로 넓힌다. 원본 텍스트는
    글자에 딱 맞는 상자로 잡히지만(여백 0), 그 줄은 원래 더 오른쪽까지
    늘어날 수 있었다 - 그 경계까지 허용해야 원본에서 한 줄이던 문장이
    몇 글자 늘어난다고 가짜 오버플로우가 나지 않는다. 세로(높이)는 절대
    넓히지 않는다(아래 줄 침범 방지 - 줄 넘김 필요하면 _line_count 가 잡음).

    폭 기준(넓은 순으로 시도):
      1) 같은 x0 에서 시작하는 '형제 줄'(불릿/스킬 목록 등)의 최대 우측끝
      2) 이 줄 세로 근처(±60pt)에 있는 모든 줄의 최대 우측끝
      3) 페이지 우측 여백(x1 - 40)
    """
    near_sib = ink.x1
    near_any = ink.x1
    try:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                lb = line.get("bbox")
                if not lb:
                    continue
                dy = abs(lb[1] - ink.y0)
                if dy < 120 and abs(lb[0] - ink.x0) < 14:      # 형제 줄
                    near_sib = max(near_sib, lb[2])
                if dy < 60:                                      # 세로로 아주 가까운 줄
                    near_any = max(near_any, lb[2])
    except Exception:
        pass
    page_right = page.rect.x1 - 40.0
    right = max(near_sib, near_any, min(page_right, ink.x1 + 6 * (ink.height or 12)))
    right = min(right, page.rect.x1 - 10.0)
    return fitz.Rect(ink.x0, ink.y0, max(ink.x1, right), ink.y1 + _INSERT_PAD_BOTTOM)


# ── 텍스트 폭 측정 ─────────────────────────────────────────────────────
_FONT_OBJ_CACHE: dict[str, "fitz.Font"] = {}


def _text_len(text: str, size: float, fontref: str, fontfile: str | None) -> float:
    """text 를 그 폰트/크기로 그렸을 때의 가로 폭(pt)."""
    if not text:
        return 0.0
    key = fontfile or "cjk"
    font = _FONT_OBJ_CACHE.get(key)
    if font is None:
        try:
            font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font("cjk")
        except Exception:
            font = fitz.Font("cjk")
        _FONT_OBJ_CACHE[key] = font
    try:
        return font.text_length(text, fontsize=size)
    except Exception:
        return sum(0.98 if ord(c) > 0x2E00 else 0.5 for c in text) * size


# ── 바뀐 텍스트 단위 추출(섹션명/좌표 하드코딩 없음) ────────────────────
def _iter_change_units(resume_raw: str, resume_customized: str, applied: dict):
    """(original_text, replacement_text, label) 을 yield 한다. original 은
    '원본 PDF 에서 찾을 문자열', replacement 는 '그 자리에 새로 쓸 문자열'.
    무엇이 바뀌었는지는 applied dict + 원문 재파싱으로만 판단한다."""

    # ① 자기소개(첫 줄) 교체
    if applied.get("headline"):
        o = customizer.extract_current_headline(resume_raw)
        n = customizer.extract_current_headline(resume_customized)
        if o and n and o.strip() != n.strip():
            yield (o.strip(), n.strip(), "자기소개", "", "")

    # ② 기술 목록 순서 변경(줄 단위 - 위치 i 의 원본 줄 자리에 새 줄을 쓴다)
    if applied.get("skill_order"):
        _, o_lines, _ = customizer._split_skill_lines(resume_raw)
        _, n_lines, _ = customizer._split_skill_lines(resume_customized)
        o_lines = [l.strip() for l in o_lines if l.strip()]
        n_lines = [l.strip() for l in n_lines if l.strip()]
        for i in range(min(len(o_lines), len(n_lines))):
            if o_lines[i] != n_lines[i]:
                yield (_clean(o_lines[i]), _clean(n_lines[i]), f"기술 목록 {i + 1}번째 줄", "", "")

    # ③ 프로젝트 내부 불릿 순서 변경
    for change in applied.get("bullet_reorders") or []:
        proj = change.get("project") or change.get("project_title") or ""
        sect = change.get("section") or change.get("section_title") or ""
        new_order = [b.strip() for b in (change.get("new_order") or [])]
        _, blocks, _ = customizer._split_projects(resume_raw)
        block = next((b for t, b in blocks if t.strip() == proj.strip() or proj.strip() in t or t in proj), None)
        if block is None:
            continue
        orig = None
        for stitle, sbody in customizer.split_project_subsections(block):
            if stitle.strip() == sect.strip() or sect.strip() in stitle or stitle in sect:
                orig = [b.strip() for b in customizer.split_bullets(sbody)]
                break
        if not orig:
            continue
        for i in range(min(len(orig), len(new_order))):
            if orig[i] != new_order[i]:
                # 불릿 기호(•)는 원본 PDF 에 그대로 두고, 그 뒤 텍스트만 교체한다
                # (안 그러면 '• •' 두 개가 찍힌다). _clean 이 앞 불릿을 떼어낸다.
                yield (_clean(orig[i]), _clean(new_order[i]), f"{proj} · {sect} {i + 1}번째 항목", "", "")

    # ④ JD 용어 치환(문장 단위)
    for r in applied.get("term_replacements") or []:
        before, after = r.get("before", ""), r.get("after", "")
        src = r.get("source_text", "")
        if src and before and before in src:
            yield (src.strip(), src.replace(before, after, 1).strip(), f"용어 치환({before}→{after})", before, after)


def _fontref(fontfile: str | None) -> str:
    return "F" + re.sub(r"[^A-Za-z0-9]", "", (fontfile or "cjk"))[-14:]


def _spans_for(doc: "fitz.Document", text: str) -> list[dict] | None:
    """text 가 문서에 유일하게 있으면 그 위치의 span 리스트를 반환.
    전체 검색이 공백/인코딩 차이로 실패하면 앞/중간/뒤 조각으로 재시도."""
    cleaned = _clean(text)
    if len(cleaned) < 4:
        return None

    def _spans_at(page, rect):
        data = page.get_text("dict", clip=rect + (-1, -1, 1, 1))
        return [
            sp for block in data.get("blocks", []) for line in block.get("lines", [])
            for sp in line.get("spans", []) if sp.get("text", "").strip()
        ]

    for pno in range(doc.page_count):
        page = doc[pno]
        hits = _search_page(page, cleaned)
        if len(hits) == 1:
            spans = _spans_at(page, hits[0])
            joined = _clean("".join(sp["text"] for sp in spans))
            if spans and (cleaned in joined or joined in cleaned):
                return spans
        # 조각 검색 - 한 줄 안이면 그 줄 전체 span 을 준다
        for frag in (cleaned[:14], cleaned[len(cleaned) // 3: len(cleaned) // 3 + 14], cleaned[-14:]):
            fh = _search_page(page, frag)
            if len(fh) == 1:
                spans = _spans_at(page, fitz.Rect(hits[0].x0 if hits else fh[0].x0 - 3, fh[0].y0 - 1,
                                                  page.rect.x1, fh[0].y1 + 1))
                joined = _clean("".join(sp["text"] for sp in spans))
                if spans and (cleaned[:20] in joined or joined[:20] in cleaned):
                    return spans
    return None


def _runs_from_spans(spans: list[dict], fallback_style: tuple) -> list[dict]:
    fam0, size0, _c = fallback_style
    runs = [{
        "text": sp["text"],
        "font": sp.get("font", fam0),
        "size": float(sp.get("size", size0)),
        "color": fitz.sRGB_to_pdf(sp.get("color", 0)),
        "from_source": True,
    } for sp in spans]
    runs[0]["text"] = re.sub(r"^\s*[•◦▪·]\s*", "", runs[0]["text"])
    return runs


def _style_runs(doc: "fitz.Document", original: str, replacement: str,
                before: str, after: str, fallback_style: tuple) -> list[dict]:
    """replacement 를 '폰트/크기/색이 같은 구간(run)' 리스트로 나눈다. 각 run 을
    자기 폰트로 그리면 굵게/보통 서식이 보존되고, 각 subset 이 자기 글리프를
    다 갖고 있어 깨진 글자(tofu)가 안 생긴다.
      · reorder: replacement 문자열이 문서 다른 곳에 그대로 있음 → 그 span 구조 사용
      · 용어 치환: 원문(original) span 구조를 가져와, before→after 가 들어있는
        span 의 텍스트만 교체 → 'SQL:' 굵기 같은 서식 유지
      · 그 외(자기소개 새 문구): fallback_style 단일 run(디스크 완전 폰트)"""
    fam0, size0, color0 = fallback_style

    # 1) replacement 가 이미 문서에 존재(reorder)
    spans = _spans_for(doc, replacement)
    if spans:
        runs = _runs_from_spans(spans, fallback_style)
        if abs(len(_clean("".join(r["text"] for r in runs))) - len(_clean(replacement))) <= 3:
            return runs

    # 2) 용어 치환 - 원문 span 을 가져와 해당 span 텍스트만 치환
    if before and after and before != after:
        spans = _spans_for(doc, original)
        if spans:
            runs = _runs_from_spans(spans, fallback_style)
            for rn in runs:
                if before in rn["text"]:
                    rn["text"] = rn["text"].replace(before, after, 1)
                    break
            if abs(len(_clean("".join(r["text"] for r in runs))) - len(_clean(replacement))) <= 4:
                return runs

    # 3) 새 문구
    return [{"text": _clean(replacement), "font": fam0, "size": size0, "color": color0, "from_source": False}]


# ── 진입점 ─────────────────────────────────────────────────────────────
def generate_from_pdf(pdf_bytes: bytes, resume_raw: str, custom_out: dict) -> dict:
    """반환 스키마는 pdf_generator.generate() 과 동일:
      {"status":"ok", "pdf_bytes":bytes, "fields_changed":int, "changed_fields":[...]}
    | {"status":"manual_review", "reason":str}
    | {"status":"no_change", "reason":str}
    | {"status":"error", "reason":str}
    """
    if not pdf_bytes:
        return {"status": "error", "reason": "원본 PDF 데이터가 없습니다. 이력서를 다시 업로드해 주세요."}

    resume_customized = custom_out.get("resume_customized", resume_raw)
    applied = custom_out.get("applied") or {}

    units = list(_iter_change_units(resume_raw, resume_customized, applied))
    if not units:
        return {"status": "no_change", "reason": "PDF에 반영할 변경 사항이 없습니다."}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": f"원본 PDF를 열 수 없습니다: {e}"}

    resolved: list[dict] = []
    try:
        for original, replacement, label, before, after in units:
            if original == replacement:
                continue
            loc = _locate(doc, original)
            if loc is None:
                doc.close()
                return {
                    "status": "manual_review",
                    "reason": f"'{label}'의 원래 위치를 원본 PDF에서 정확히 찾지 못했습니다. 위 판단 카드를 참고해 직접 반영해주세요.",
                }
            pno, ink = loc
            family, size, color, baseline_y = _load_style(doc[pno], ink)
            box = _insert_box(doc[pno], ink)
            runs = _style_runs(doc, original, replacement, before, after, (family, size, color))
            # 폰트 파일 해석:
            #  · reorder(기존 텍스트 이동) run → 원본 임베드 subset 우선
            #    (그 subset 이 그 텍스트의 글리프를 전부 갖고 있고 →·• 도 정확)
            #  · 새 텍스트(자기소개/용어) run → 디스크 완전 폰트 우선
            #    (임의의 새 한글 음절이 subset 에 없어 깨지는 것 방지)
            for rn in runs:
                if rn.get("from_source"):
                    ff = _embedded_font_file(doc, rn["font"]) or _resolve_font_file(rn["font"])
                else:
                    ff = _resolve_font_file(rn["font"]) or _embedded_font_file(doc, rn["font"])
                rn["fontfile"] = ff
                rn["fontref"] = _fontref(ff)
            total_w = sum(
                _text_len(rn["text"], rn["size"], rn["fontref"], rn["fontfile"]) for rn in runs
            )
            line_h = size * 1.36
            orig_lines = max(1, round(ink.height / line_h))
            avail_w = box.x1 - ink.x0
            new_lines = max(1, -(-int(total_w) // max(1, int(avail_w))))
            if new_lines > orig_lines:
                doc.close()
                return {
                    "status": "manual_review",
                    "reason": f"'{label}'의 수정된 내용이 원본 칸을 벗어납니다. 위 판단 카드를 참고해 직접 반영해주세요.",
                }
            resolved.append({
                "page": pno, "ink": ink, "box": box, "runs": runs, "label": label,
                "baseline_y": baseline_y, "verify": _clean(replacement),
            })

        if not resolved:
            doc.close()
            return {"status": "no_change", "reason": "PDF에 반영할 변경 사항이 없습니다."}

        # 원본 텍스트 제거 - 잉크 영역만(위/아래 살짝 안쪽), 사진/선은 유지.
        for r in resolved:
            ink = r["ink"]
            clear = fitz.Rect(ink.x0 - 0.5, ink.y0 + _REDACT_INSET_V, ink.x1 + 1.5, ink.y1 - _REDACT_INSET_V)
            doc[r["page"]].add_redact_annot(clear, fill=(1, 1, 1))
        for page in doc:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)

        # run 을 왼쪽부터 순서대로, 각자 자기 폰트로 baseline 에 그린다.
        for r in resolved:
            page = doc[r["page"]]
            x = r["ink"].x0
            for rn in r["runs"]:
                kw = {"fontsize": rn["size"], "color": rn["color"], "fontname": rn["fontref"]}
                if rn["fontfile"]:
                    kw["fontfile"] = rn["fontfile"]
                try:
                    page.insert_text((x, r["baseline_y"]), rn["text"], **kw)
                except Exception:
                    page.insert_text((x, r["baseline_y"]), rn["text"], fontsize=rn["size"],
                                     color=rn["color"], fontname=_BUILTIN_CJK)
                x += _text_len(rn["text"], rn["size"], rn["fontref"], rn["fontfile"])

        pdf_out = doc.tobytes()
    finally:
        if not doc.is_closed:
            doc.close()

    # ── 사후 검증 - 새 텍스트가 결과 PDF 에 제대로 렌더됐는지 확인.
    # 폰트 글리프 누락/깨짐(→ 등)이 있으면 search_for 가 못 찾는다 → 원본 유지.
    # 라벨:본문 사이 공백 종류 차이로 앞 조각이 안 맞을 수 있어 여러 조각을 본다.
    try:
        vd = fitz.open(stream=pdf_out, filetype="pdf")
        for r in resolved:
            v = r["verify"]
            probes = {v[:16], v[max(0, len(v) // 2 - 8): len(v) // 2 + 8], v[-16:]}
            probes = {p for p in probes if len(p) >= 8}
            if probes and not any(
                _search_page(vd[p], pr) for p in range(vd.page_count) for pr in probes
            ):
                vd.close()
                return {
                    "status": "manual_review",
                    "reason": f"'{r['label']}'을(를) 원본 서식으로 정확히 다시 그리지 못했습니다"
                              f"(특수문자/폰트 문제일 수 있음). 위 판단 카드를 참고해 직접 반영해주세요.",
                }
        vd.close()
    except Exception:
        pass

    return {
        "status": "ok",
        "pdf_bytes": pdf_out,
        "fields_changed": len(resolved),
        "changed_fields": [
            {"page": r["page"], "bbox": tuple(r["ink"]), "label": r["label"]} for r in resolved
        ],
    }
