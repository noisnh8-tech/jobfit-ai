"""
resume_input/resume_template.py

원본 이력서 PDF 위에서 어느 필드가 어느 좌표(bbox)에 있는지 등록·조회
하는 모듈. pdf_generator.py(Task #67)가 이 레지스트리만 보고 원본 PDF의
해당 영역을 redact+insert_text로 교체한다 - 판단 로직(customization_
rules.py)은 이 모듈을 전혀 모르고, 이 모듈도 판단 로직을 전혀 모른다
(계획 문서 "판단 로직을 전혀 참조하지 않는다" 원칙).

좌표는 pdfplumber로 반자동 추출했다(docs/verification/2026-07-21_
customization_rule_engine/scripts/extract_resume_coordinates.py 결과
참고, logs/02_coordinate_extraction.txt). bbox는 전부 (x0, top, x1,
bottom) - pdfplumber 좌표계(원점 좌상단)이며 PyMuPDF Rect 생성 시 그대로
사용 가능하다.

프로젝트는 "슬롯" 단위로 등록한다(slot 1 = 페이지0의 첫 프로젝트 자리,
slot 2 = 페이지1의 둘째 프로젝트 자리). "• 프로젝트" 구역 제목 자체는
두 프로젝트가 공유하는 고정 텍스트라 슬롯에 포함하지 않는다 - 순서
변경은 슬롯 안의 본문(제목~성과 불릿)만 서로 바꿔 끼우는 것이다.
불릿 bbox는 원본에서 여러 줄로 접혀 있으면 첫 줄 top ~ 마지막 줄 bottom,
x1은 그 불릿에 속한 모든 줄 중 가장 큰 x1을 그대로 감싸는 사각형이다.

[공개 데모용: 실제 등록 템플릿(개인 이력서 좌표·경로·해시)은 제외했음.
 _TEMPLATES 가 비어 있으면 get_template()이 못 찾고, pipeline의 다음
 폴백 경로(html_resume → pdf_layout)로 자연스럽게 넘어간다 - 좌표 치환
 "메커니즘" 자체는 아래에 그대로 남겨 코드 리뷰가 가능하게 했다.]
"""
from __future__ import annotations

import os
from pathlib import Path

FONT_FAMILY_BOLD = "Pretendard-Bold"
FONT_FAMILY_REGULAR = "Pretendard-Regular"

# 실제 폰트 파일 경로(로컬에 설치된 Pretendard - pdf_generator.py가
# insert_textbox(fontfile=...)로 그대로 embed한다. PyMuPDF 내장 CJK
# 대체 폰트("korea-s")로는 글자폭이 원본과 달라 원본에서 한 줄로
# 들어가던 텍스트도 자기 자리에 다시 안 들어가는(가짜 오버플로우)
# 문제가 실측으로 확인되어, 실제 원본 폰트 파일을 직접 쓰는 쪽으로
# 확정했다. 운영 모드에서 이 고속 경로를 쓰려면 본인 PC의 Pretendard
# 폰트 파일 경로를 아래에 채워 넣으면 된다.
FONT_FILES = {
    FONT_FAMILY_BOLD: str(
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts\Pretendard-Bold.otf"))
    ),
    FONT_FAMILY_REGULAR: str(
        Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts\Pretendard-Regular.otf"))
    ),
}

# 공개 데모에는 등록된 템플릿이 없음(개인 이력서 좌표 정보라 제외).
_TEMPLATES: dict[str, dict] = {}
_TEMPLATES_DISABLED_EXAMPLE: dict = {
    "sample_default": {
        "pdf_path": "",  # 운영 모드: 본인 이력서 PDF 경로
        "resume_hash": "",  # 운영 모드: execution_logger.resume_hash(resume_raw) 값
        "page_size": {"width": 595.5, "height": 842.2},
        "headline": {
            "page": 0,
            "bbox": (45.3, 95.6, 259.5, 108.6),
            "font": FONT_FAMILY_BOLD,
            "size": 13.0,
        },
        "skills": {
            "page": 0,
            "block_bbox": (46.2, 172.4, 395.9, 264.9),
            "font": FONT_FAMILY_BOLD,
            "size": 10.0,
            "line_bboxes": [
                (46.2, 172.4, 292.0, 182.4),
                (46.2, 188.9, 307.7, 198.9),
                (46.2, 205.4, 320.5, 215.4),
                (46.2, 221.5, 395.9, 231.9),
                (46.2, 238.4, 342.5, 248.4),
                (46.2, 254.9, 318.5, 264.9),
            ],
        },
        "projects": [
            {
                "slot": 1,
                "page": 0,
                "region_bbox": (45.3, 340.0, 595.5, 780.0),
                "title_bbox": (62.9, 351.1, 437.1, 364.1),
                "tech_bbox": (62.0, 376.6, 508.3, 386.6),
                "purpose": {
                    "header_bbox": (59.6, 406.9, 131.0, 417.9),
                    "body_bboxes": [(62.0, 424.9, 532.0, 434.9)],
                },
                "problem": {
                    "header_bbox": (59.6, 453.5, 113.6, 464.5),
                    "body_bboxes": [
                        (62.0, 474.7, 342.9, 484.7),
                        (59.6, 491.2, 547.1, 501.2),
                    ],
                },
                "process": {
                    "header_bbox": (59.6, 520.6, 147.6, 531.6),
                    "bullet_bboxes": [
                        (59.0, 544.0, 438.1, 570.5),
                        (60.8, 583.3, 362.5, 609.8),
                        (59.0, 621.0, 474.7, 631.0),
                    ],
                },
                "results": {
                    "header_bbox": (59.6, 650.5, 92.6, 661.5),
                    "bullet_bboxes": [
                        (59.0, 674.6, 406.1, 701.1),
                        (59.0, 711.4, 419.1, 737.1),
                        (59.0, 748.1, 452.7, 771.7),
                    ],
                },
            },
            {
                "slot": 2,
                "page": 1,
                "region_bbox": (45.3, 20.0, 595.5, 510.0),
                "title_bbox": (68.5, 30.1, 381.9, 43.1),
                "tech_bbox": (69.4, 55.9, 317.8, 65.9),
                "purpose": {
                    "header_bbox": (65.2, 78.6, 136.8, 89.6),
                    "body_bboxes": [(65.2, 94.8, 463.4, 104.8)],
                },
                "problem": {
                    "header_bbox": (65.2, 124.9, 119.3, 135.9),
                    "body_bboxes": [
                        (65.2, 144.0, 208.7, 154.0),
                        (65.2, 157.5, 495.2, 167.5),
                    ],
                },
                "process": {
                    "header_bbox": (65.2, 185.4, 153.2, 196.4),
                    "bullet_bboxes": [
                        (65.2, 204.7, 481.8, 230.5),
                        (65.2, 246.3, 543.4, 272.0),
                        (65.2, 284.8, 518.9, 326.3),
                        (65.2, 335.8, 460.0, 345.8),
                    ],
                },
                "results": {
                    "header_bbox": (65.2, 367.1, 98.3, 378.1),
                    "bullet_bboxes": [
                        (65.2, 386.5, 443.1, 412.2),
                        (65.2, 422.0, 350.3, 447.8),
                        (65.2, 457.5, 354.8, 483.2),
                    ],
                },
            },
        ],
        "body_font": FONT_FAMILY_REGULAR,
        "body_size": 10.0,
    },
}


def get_template(template_id: str = "sample_default") -> dict:
    """등록된 이력서 템플릿(경로/필드 bbox/폰트)을 반환한다. pdf_generator.py
    가 판단 결과를 좌표에 매핑할 때 이 함수만 호출한다."""
    template = _TEMPLATES.get(template_id)
    if template is None:
        raise KeyError(f"등록되지 않은 이력서 템플릿: {template_id}")
    return template


def list_template_ids() -> list[str]:
    return list(_TEMPLATES.keys())


def find_template_id_by_resume_hash(resume_hash: str) -> str | None:
    """지금 업로드된 이력서(resume_hash)가 좌표가 등록된 템플릿과 같은
    파일인지 찾는다. 없으면 None - 호출부(pipeline.generate_resume_pdf)
    는 이 경우 PDF 자동 생성을 시도하지 않고 안내만 보여줘야 한다(등록
    안 된 이력서에 잘못된 좌표를 적용하면 안 되므로)."""
    for template_id, template in _TEMPLATES.items():
        if template.get("resume_hash") == resume_hash:
            return template_id
    return None
