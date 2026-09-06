"""
ui_components.py

화면 여러 곳(공고 찾기 / 공고 분석 / 매칭 결과 등, 레퍼런스 스크린 03~05)
에서 반복되는 시각 요소를 한 곳에 모아둔 재사용 컴포넌트 모음. 순수
프레젠테이션 레이어 - 판단 로직 없음(resume_input 패키지를 모른다).
화면마다 같은 HTML을 다시 만들지 않도록, 그리고 여러 화면의 디자인이
어긋나지 않도록 하기 위함(사용자 요청, 2026-07-14: "공통 컴포넌트를
먼저 만들고 화면에서는 조립만 하자").
"""
from __future__ import annotations

import html

from ui_theme import badge_html
from ui_icons import icon_png

# 실제 회사가 자기 도메인에서 쓰는 파비콘을 가져온다(지어낸 아이콘이
# 아니라 진짜 사이트 아이콘) - Google 파비콘 캐시를 쓰면 별도 API
# 키/스크래핑 없이도 실제 로고에 가장 가깝게 접근할 수 있다. 도메인이
# 확인되지 않는(매핑에 없는) 회사는 아무 아이콘도 표시하지 않는다
# (이니셜 아바타로도 대체하지 않는다 - 사용자 명시적 요청, 2026-07-15:
# "없는 기업은 그냥 아무것도 넣지 마" - 실제 로고를 지어내지 않는다는
# 원칙을 더 엄격하게 적용한 것).
#
# [공개 저장소 방침] 운영 버전의 이 표는 실제 후보 풀(candidate_jobs)에서
# 자주 보이는 회사 위주로 채워왔다 - 즉 실제 수집·관심 기업군을 그대로
# 드러낸다. 공개 데모는 가상 회사명(A커머스 등)만 쓰고 이 표와 매칭될
# 일이 없으므로(위 주석대로 매칭 실패 시 이니셜 아바타로 자연 대체),
# 표를 비워도 데모 화면은 동일하게 동작한다 - 기능 유지를 위해 비워둔다.
_COMPANY_DOMAINS: dict[str, str] = {}


def _match_domain(company: str) -> str | None:
    key = (company or "").strip().lower()
    if not key:
        return None
    for name, domain in _COMPANY_DOMAINS.items():
        if name in key:
            return domain
    return None


def resolve_company_logo(company: str) -> dict:
    """company_logo_html()과 같은 판단(도메인 매핑 있으면 실제 파비콘,
    없으면 이니셜)을 CCv2 컴포넌트(job_card_component.py)에 JSON으로
    넘기기 위한 버전 - HTML 문자열이 아니라 데이터만 반환한다(CCv2는
    JS에서 innerHTML로 조립하므로, app.py가 만든 완성 HTML을 그대로
    꽂으면 XSS 위험이 있다 - 원본 판단 로직은 그대로 재사용하고
    표현만 분리)."""
    domain = _match_domain(company)
    if domain:
        return {"type": "favicon", "url": f"https://www.google.com/s2/favicons?domain={domain}&sz=128"}
    initial = (company or "?").strip()[:1].upper() or "?"
    return {"type": "initial", "letter": initial}


def company_logo_html(company: str, size: int = 44) -> str:
    """실제 회사 로고(파비콘) - 매핑에 있으면 그 회사 도메인의 실제
    파비콘을 쓴다. 매핑에 없으면 회사명 첫 글자로 만든 이니셜
    placeholder를 표시한다(2026-08-14, 사용자 확정 - 카드마다 로고
    자리가 있다 없다로 들쭉날쭉해 보이는 문제를 통일하려고 2026-07-15
    "빈 자리 유지" 결정을 대체함). 지어낸 브랜드 로고가 아니라 순수
    타이포 placeholder이므로 실제 로고를 사칭하지 않는다 - 디자인
    시스템의 White+Navy 톤(2026-07-24)을 그대로 써서 카드마다 색이
    제각각으로 튀지 않게 한다."""
    domain = _match_domain(company)
    if not domain:
        initial = html.escape((company or "?").strip()[:1].upper() or "?")
        font_size = max(14, size // 2)
        return (
            f'<div style="width:{size}px;height:{size}px;flex-shrink:0;'
            f'border-radius:10px;background:var(--navy,#0F1F4A);'
            f'display:flex;align-items:center;justify-content:center;'
            f'color:#FFFFFF;font-weight:800;font-size:{font_size}px;">'
            f'{initial}</div>'
        )
    return (
        f'<img src="https://www.google.com/s2/favicons?domain={domain}&sz=128" '
        f'width="{size}" height="{size}" alt="{company}" '
        f'style="border-radius:10px;object-fit:contain;background:white;'
        f'border:1px solid var(--card-border);padding:5px;flex-shrink:0;" />'
    )


# ── 공고 직무 카테고리(전체/데이터 분석/BI/마케팅/기타) ─────────────
# LLM 호출 없음 - 제목 키워드 기반 정규식/부분일치 분류(career_filter.py,
# job_prep.py와 같은 방식). 애매하면 "기타"로 분류한다(추측하지 않음).
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "데이터 분석": ["데이터 분석", "데이터분석", "data analyst", "data analytics", "data scientist", "데이터 사이언"],
    "BI": ["bi analyst", "business intelligence", " bi ", "bi팀", "bi 담당"],
    "마케팅": ["마케팅", "marketing", "그로스", "growth"],
}
CATEGORY_TABS = ["전체", "데이터 분석", "BI", "마케팅", "기타"]


def classify_job_category(title: str) -> str:
    t = f" {(title or '').lower()} "
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return cat
    return "기타"


def match_tag_html(text: str) -> str:
    """강점/기술 태그 - 매칭 결과·요구스킬 화면에서 공통으로 쓰는 둥근 배지."""
    return badge_html(text, "pink")


# ── STEP1(app.py render_s0) 공통 카드 프리미티브(2026-07-24) ────────
# 카드마다 아이콘/제목/본문 HTML을 따로 조립하지 않고, 같은 역할의
# 요소는 전부 이 세 함수를 거치게 한다 - CSS class 하나(icon-title-row,
# meta-block, action-layout)가 정렬을 책임지므로 카드별 px 보정이
# 필요 없어진다(사용자 요청 - "카드 크기에 맞게 자동으로 균형있게
# 배치"). 판단 로직 없음 - 순수 HTML 조립.

def icon_title_row(
    icon_name: str, title: str, *,
    icon_size: int = 40, css_class: str = "icon-title-row", variant: str | None = None,
) -> str:
    """아이콘 + 제목이 같은 줄에 오는 공통 행(요약 카드/상태 카드
    헤더에서 재사용). frame 크기는 CSS(.icon-frame 등)가 고정하므로
    icon_size는 PNG 자체 렌더 크기만 조정한다."""
    return (
        f'<div class="{css_class}">'
        f'<div class="icon-frame">{icon_png(icon_name, icon_size, variant=variant)}</div>'
        f'<div class="icon-title-text">{html.escape(title)}</div>'
        f'</div>'
    )


def meta_block(label: str, value: str, *, divider: bool = False) -> str:
    """라벨+값 한 쌍(인식된 연차/분석 완료일 등) - divider=True면 왼쪽에
    구분선을 붙인다. 두 meta_block을 나란히 두면 구조가 같아서 높이가
    자동으로 맞는다."""
    cls = "meta-block meta-block--divider" if divider else "meta-block"
    return (
        f'<div class="{cls}">'
        f'<div class="meta-label">{html.escape(label)}</div>'
        f'<div class="meta-value">{html.escape(value)}</div>'
        f'</div>'
    )


def action_card_content(
    icon_name: str, title: str, description: str, *,
    icon_size: int = 40, variant: str | None = None,
) -> str:
    """액션 카드(채용공고 탐색하기/직접 공고 분석하기) 본문 - 아이콘+제목
    한 줄, 그 아래 설명. 화살표는 여기 포함하지 않는다(별도 .arrow-circle
    div로 action-layout의 두 번째 grid column에 놓는다)."""
    return (
        '<div class="action-main">'
        f'<div class="action-heading-row">'
        f'<div class="icon-frame">{icon_png(icon_name, icon_size, variant=variant)}</div>'
        f'<div class="action-title">{html.escape(title)}</div>'
        f'</div>'
        f'<div class="action-desc">{html.escape(description)}</div>'
        '</div>'
    )


def empty_state_html(message: str, sub: str = "") -> str:
    """빈 상태(지원 기록 없음/관심 공고 없음 등) 공통 컴포넌트
    (2026-07-16 추가) - 텍스트 한 줄(st.info)만 있던 걸 3D 일러스트
    (Fluent Emoji "빈 우편함", MIT 라이선스)로 보완한다. 새 판단/로직
    없음 - 순수 표시용."""
    from ui_icons import svg_icon
    return (
        '<div style="text-align:center;padding:2.4rem 1rem 1.6rem;">'
        f'<div style="line-height:0;opacity:0.55;">{svg_icon("folder", 64, "var(--navy)")}</div>'
        f'<div style="margin-top:0.9rem;font-weight:700;color:var(--text-primary);font-size:1.02rem;">{message}</div>'
        + (f'<div style="margin-top:4px;color:var(--text-secondary);font-size:0.88rem;">{sub}</div>' if sub else "")
        + '</div>'
    )
