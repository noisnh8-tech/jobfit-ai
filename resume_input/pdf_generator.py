"""
resume_input/pdf_generator.py

customizer.apply()가 만든 "커스터마이징 완료 데이터"(resume_customized
텍스트)와 resume_template.py에 등록된 원본 PDF 좌표만 입력받아, 원본
PDF에서 실제로 바뀐 영역만 redact(투명 삭제)+insert_text로 교체한다.
customization_rules.py(판단 로직)를 전혀 참조하지 않는다 - "무엇이
바뀌었는지"는 원본 텍스트와 완료 텍스트를 직접 비교(diff)해서 이 파일
스스로 알아내고, 그 결과만 좌표에 반영한다. LLM 호출 없음.

오버플로우 규칙(계획 확정, 유일한 처리): 수정된 내용이 등록된 bbox
영역(또는 등록된 불릿 칸 개수)을 벗어나면 폰트를 줄이거나 문장을
요약하거나 다른 위치로 옮기지 않는다 - 즉시 PDF를 만들지 않고
manual_review를 반환한다. 실제 PDF는 모든 필드가 영역 안에 들어간다는
것을 사전 확인(dry-run)한 뒤에만 만든다(원본 PDF를 부분적으로 망가뜨린
채로 남기지 않기 위함).

폰트는 이 PC에 설치된 실제 Pretendard 폰트 파일(resume_template.
FONT_FILES)을 그대로 embed한다 - PyMuPDF 내장 CJK 대체 폰트("korea-s")
로 먼저 시도했으나, 원본에서 정확히 한 줄로 들어가던 텍스트도 글자폭
차이 때문에 자기 자리에 다시 안 들어가는(가짜 오버플로우) 문제가
6개 기술 항목 전수 실측으로 확인되어 폐기했다.

원본 bbox는 pdfplumber가 측정한 글자 잉크 영역 그대로라 여백이 거의
없다 - 다른 렌더링 엔진(PyMuPDF)이 같은 폰트/크기로 다시 그릴 때 경계값
반올림만으로도 줄바꿈이 갈릴 수 있어(실측: 원본 그대로도 -0.02pt 차이로
줄바꿈 발생), 판단/좌표 자체를 바꾸지 않고 여유 폭만 작게 준다
(_PAD_W, _PAD_H - 모든 줄 사이 실제 여백보다 작게 잡아 옆 줄을 침범하지
않음, 실측으로 확인).
"""
from __future__ import annotations

import fitz

from resume_input import customizer, resume_template

_PAD_W = 6.0
_PAD_H = 6.0


def _padded(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (bbox[0], bbox[1], bbox[2] + _PAD_W, bbox[3] + _PAD_H)


def _union_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in bboxes)
    top = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    bottom = max(b[3] for b in bboxes)
    return (x0, top, x1, bottom)


def _fits(text: str, bbox: tuple[float, float, float, float], fontsize: float, font_family: str) -> bool:
    if not text:
        return True
    fontfile = resume_template.FONT_FILES[font_family]
    scratch = fitz.open()
    page = scratch.new_page(width=700, height=1000)
    rc = page.insert_textbox(
        fitz.Rect(*_padded(bbox)), text, fontsize=fontsize, fontname=font_family, fontfile=fontfile,
    )
    scratch.close()
    return rc >= 0


def _project_header_line(block: str) -> str:
    return block.split("\n", 1)[0].strip()


def _project_tech_line(block: str) -> str:
    for ln in block.split("\n"):
        if ln.strip().startswith("[기술]"):
            return ln.strip()
    return ""


def _section_body(block: str, section_title: str) -> str:
    for title, body in customizer.split_project_subsections(block):
        if title == section_title:
            return body.strip("\n")
    return ""


def _slot_fields(slot: dict, orig_block: str, new_block: str, body_font_size: float) -> tuple[list[dict], str | None]:
    """슬롯 하나(프로젝트 전체)에서, 원본과 완료 텍스트가 실제로 다른
    필드만 골라 채워야 할 목록을 만든다. 불릿 수가 등록된 칸보다 많으면
    (빈 리스트, 실패 사유)를 반환한다 - 오버플로우."""
    fields: list[dict] = []

    orig_title_line = _project_header_line(orig_block)
    new_title_line = _project_header_line(new_block)
    if new_title_line != orig_title_line:
        fields.append({
            "bbox": slot["title_bbox"], "text": new_title_line, "size": 13.0,
            "font": resume_template.FONT_FAMILY_BOLD,
            "label": "프로젝트 제목", "page": slot["page"],
        })

    orig_tech = _project_tech_line(orig_block)
    new_tech = _project_tech_line(new_block)
    if new_tech != orig_tech:
        fields.append({
            "bbox": slot["tech_bbox"], "text": new_tech, "size": body_font_size,
            "font": resume_template.FONT_FAMILY_REGULAR,
            "label": "프로젝트 기술 스택", "page": slot["page"],
        })

    for key, section_title in (("purpose", "프로젝트 목적"), ("problem", "문제 정의")):
        orig_body = _section_body(orig_block, section_title)
        new_body = _section_body(new_block, section_title)
        if new_body != orig_body:
            bbox = _union_bbox(slot[key]["body_bboxes"])
            fields.append({
                "bbox": bbox, "text": new_body, "size": body_font_size,
                "font": resume_template.FONT_FAMILY_REGULAR,
                "label": f"프로젝트 {section_title}", "page": slot["page"],
            })

    for key, section_title in (("process", "해결 과정 및 역할"), ("results", "성과")):
        orig_bullets = customizer.split_bullets(_section_body(orig_block, section_title))
        new_bullets = customizer.split_bullets(_section_body(new_block, section_title))
        if new_bullets == orig_bullets:
            continue
        registered = slot[key]["bullet_bboxes"]
        if len(new_bullets) > len(registered):
            return [], (
                f"'{section_title}' 영역에 불릿 {len(new_bullets)}개가 필요하지만 "
                f"등록된 칸은 {len(registered)}개뿐입니다."
            )
        for i, bbox in enumerate(registered):
            orig_text = orig_bullets[i] if i < len(orig_bullets) else ""
            new_text = new_bullets[i] if i < len(new_bullets) else ""
            if new_text == orig_text:
                continue  # 안 바뀐 형제 불릿은 건드리지 않는다 - 원본 content stream 그대로 보존(부분 Bold/특수문자/공백 원본 유지)
            fields.append({
                "bbox": bbox, "text": new_text, "size": body_font_size,
                "font": resume_template.FONT_FAMILY_REGULAR,
                "label": f"프로젝트 {section_title} 불릿", "page": slot["page"],
            })

    return fields, None


def generate(template_id: str, resume_raw: str, custom_out: dict) -> dict:
    """반환: {"status": "ok", "pdf_bytes": bytes, "fields_changed": int}
    | {"status": "manual_review", "reason": str}
    | {"status": "no_change", "reason": str}"""
    template = resume_template.get_template(template_id)
    resume_customized = custom_out["resume_customized"]
    applied = custom_out.get("applied") or {}
    body_font_size = template.get("body_size", 10.0)

    fields: list[dict] = []

    if applied.get("headline"):
        orig_headline = customizer.extract_current_headline(resume_raw) or ""
        new_headline = customizer.extract_current_headline(resume_customized) or ""
        if new_headline != orig_headline:
            h = template["headline"]
            fields.append({
                "bbox": h["bbox"], "text": new_headline, "size": h["size"],
                "font": h["font"], "label": "자기소개", "page": h["page"],
            })

    _, orig_skill_lines, _ = customizer._split_skill_lines(resume_raw)
    _, new_skill_lines, _ = customizer._split_skill_lines(resume_customized)
    orig_skill_lines = [ln for ln in orig_skill_lines if ln.strip()]
    new_skill_lines = [ln for ln in new_skill_lines if ln.strip()]
    if new_skill_lines != orig_skill_lines:
        s = template["skills"]
        registered = s["line_bboxes"]
        if len(new_skill_lines) > len(registered):
            return {
                "status": "manual_review",
                "reason": f"기술 목록 {len(new_skill_lines)}개가 등록된 칸({len(registered)}개)보다 많습니다.",
            }
        for i, bbox in enumerate(registered):
            text = new_skill_lines[i] if i < len(new_skill_lines) else ""
            fields.append({
                "bbox": bbox, "text": text, "size": s["size"],
                "font": s["font"], "label": "기술 목록", "page": s["page"],
            })

    orig_blocks = customizer._split_projects(resume_raw)[1]
    new_blocks = customizer._split_projects(resume_customized)[1]
    for slot_idx, slot in enumerate(template["projects"]):
        if slot_idx >= len(orig_blocks) or slot_idx >= len(new_blocks):
            continue
        _, orig_block = orig_blocks[slot_idx]
        _, new_block = new_blocks[slot_idx]
        slot_fields, error = _slot_fields(slot, orig_block, new_block, body_font_size)
        if error:
            return {"status": "manual_review", "reason": error}
        fields.extend(slot_fields)

    if not fields:
        return {"status": "no_change", "reason": "PDF에 반영할 변경 사항이 없습니다."}

    for f in fields:
        if not _fits(f["text"], f["bbox"], f["size"], f["font"]):
            return {
                "status": "manual_review",
                "reason": f"'{f['label']}' 영역에 수정된 내용이 원본 칸을 벗어납니다.",
            }

    doc = fitz.open(template["pdf_path"])
    for f in fields:
        page = doc[f["page"]]
        page.add_redact_annot(fitz.Rect(*_padded(f["bbox"])), fill=(1, 1, 1))
    for page in doc:
        page.apply_redactions()
    for f in fields:
        page = doc[f["page"]]
        if f["text"]:
            fontfile = resume_template.FONT_FILES[f["font"]]
            page.insert_textbox(
                fitz.Rect(*_padded(f["bbox"])), f["text"],
                fontsize=f["size"], fontname=f["font"], fontfile=fontfile,
            )

    pdf_bytes = doc.tobytes()
    doc.close()
    changed_fields = [{"page": f["page"], "bbox": f["bbox"], "label": f["label"]} for f in fields]
    return {
        "status": "ok", "pdf_bytes": pdf_bytes, "fields_changed": len(fields),
        "changed_fields": changed_fields,
    }
