"""
app.py

Streamlit UI. 판단 로직은 전혀 없다 - resume_input 패키지의 함수를
호출해서 결과를 보여주기만 한다. 화면 구조는 docs/ui_design.md
(v8) 참고.

이 라운드(SaaS 스타일 리디자인)는 시각적 표현만 바꾼다 - ui_theme.py/
ui_icons.py가 전부 담당하고, resume_input/*.py의 판단 로직은 단
한 줄도 건드리지 않았다. pipeline.py 호출 시그니처와 반환값 사용
방식은 이전 라운드 그대로다.

추천 모드에는 지원추천/보류/비추천(decision) 개념이 없다 - 분석
엔진(judge)은 [지원하기]를 눌렀을 때 내부적으로만 호출되고, 그
결과의 decision/reason은 이 파일 어디에서도 읽거나 표시하지 않는다.
JD 붙여넣기 모드만 S3에서 decision을 표시한다.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import sys
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 2026-07-19(사용자 요청) - "공고 검색 중입니다" 화면이 수 분씩 걸리던
# 원인(candidate_search.py의 임베딩 모델 로딩이 매번 huggingface.co에
# 네트워크 조회를 시도함) 대응. 이미 로컬에 캐시된 모델만 쓰도록 이
# 환경변수를 다른 Hugging Face 관련 모듈(sentence_transformers/
# transformers/huggingface_hub)이 import되기 전에, 앱 진입점 맨 위에서
# 설정한다 - huggingface_hub는 이 값을 import 시점에 읽으므로 함수
# 안에서 나중에 설정하면 이미 늦을 수 있다.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import fitz
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resume_input import pipeline
from resume_input import customizer as customizer_mod
from resume_input import presentation_layer
from resume_input import last_resume
from resume_input.understanding import get_or_generate_resume_understanding, get_cached_resume_understanding
from resume_input.pdf_parser import extract_text_from_pdf
from resume_input.job_detail import extract_display_sections, format_company_intro, build_jd_detail_view, derive_strengths_and_gaps, derive_surface_comparisons
from resume_input.job_store import get_candidate_job, list_recent_analyses, count_analyses
from resume_input import quick_analysis
from resume_input.saved_jobs import add_saved_job, remove_saved_job, list_saved_jobs, is_saved
from resume_input.dismissed_jobs import (
    add_dismissed_job, remove_dismissed_job, list_dismissed_job_ids, list_dismissed_jobs,
    list_dismissed_posting_keys,
)
from resume_input.application_manager import (
    list_applications, update_status, delete_application, list_status_history,
    list_applied_job_ids, list_applied_posting_keys,
    list_pending_messages, dismiss_pending_messages,
    STATUS_OPTIONS as APPLICATION_STATUS_OPTIONS,
)
from resume_input.posting_identity import posting_key as _posting_key
from resume_input import recommendation_cache
from resume_input.llm_client import LLMCallError
# 2026-08-14(신규) - Analysis(Evidence Contract 5-state)/Customization
# Selection·Composition·Rewrite/Cover Letter Source 화면 연결. 전부 이미
# 검증된 엔진(docs/verification/2026-08-14_customization_features)을
# 그대로 소비만 한다 - 여기서 새 판단 로직을 추가하지 않는다.
from resume_input import judge_engine
from resume_input.resume_facts import extract_resume_facts
from resume_input.resume_career import extract_user_career_level
from resume_input.job_family import DATA_ANALYTICS
from resume_input import analysis_engine
from resume_input.analysis_component import analysis_view
from resume_input.job_card_component import job_list_view
from resume_input import rewrite_engine
from resume_input import rewrite_versions
from resume_input import cover_letter_engine
from resume_input import execution_logger
from resume_input import preparation_tracker
from resume_input import resume_versions

from ui_theme import inject_theme, page_header, badge_html, step_progress_html
from ui_icons import icon_badge, svg_icon, icon_png, icon_png_data_uri
from ui_components import company_logo_html, empty_state_html, icon_title_row, meta_block, action_card_content, _match_domain, resolve_company_logo

st.set_page_config(page_title="AI 취업 지원 도우미", layout="wide", page_icon="✨")
inject_theme()

from resume_input.runtime_mode import IS_DEMO, DEMO_BANNER_TEXT  # noqa: E402

CAREER_LEVEL_OPTIONS = ["신입", "1~3년", "3~5년", "5년+"]

STATUS_TONE = {
    "적합": "green",
    "약간상향": "orange",
    "약간하향": "orange",
    "확인필요": "gray",
    "상향": "orange",
    "크게상향": "red",
}

DECISION_TONE = {"지원": "green", "보류": "orange", "비추천": "red"}
# APPLICATION_STATUS_OPTIONS는 application_manager.STATUS_OPTIONS를
# import alias로 그대로 쓴다(위 import 문) - 상태값을 두 곳에서 따로
# 관리하지 않는다.


def _resume_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _init_state() -> None:
    defaults = {
        "screen": "LOGIN",
        "user_display_name": "",
        "show_google_demo_msg": False,
        "nav_stack": [],
        "resume_raw": None,
        "resume_pdf_bytes": None,
        "uploaded_file_id": None,
        "show_uploader": True,
        "uploader_version": 0,
        "resume_file_name": None,
        "resume_file_size": None,
        "resume_uploaded_at": None,
        "resume_semantic_objects": [],
        "user_career_level": None,
        "career_level_confidence": None,
        "career_level_reason": None,
        "career_level_source": "auto",  # "auto" | "manual"
        "challenge_option": False,
        "top_cache": {},
        "s1_browsing": False,  # 공고 찾기: True면 공고 목록, False면 "어떤 방식으로 진행할까요?" 선택
        "s1_filter_sig": None,
        "selected_job": None,
        "judge_result": None,
        "apply_result": None,
        "resume_pdf_result": None,
        "html_resume_result": None,   # 개인 표준 이력서 HTML 경로 결과(공고 바뀌면 리셋)
        "html_resume_failed": False,  # HTML 경로 실패 시 이 세션에선 범용 경로로 폴백
        "portfolio_pptx_result": None,  # 개인 맞춤 포트폴리오(.pptx) 결과(공고 바뀌면 리셋)
        "analysis_by_job": {},
        "rewrite_proposals_by_job": {},
        "rewrite_decisions_by_job": {},
        "rewrite_result_by_job": {},
        "headline_proposal_by_job": {},
        "headline_choice_by_job": {},
        "composition_by_job": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go(screen: str) -> None:
    st.session_state.nav_stack.append(st.session_state.screen)
    st.session_state.screen = screen
    st.session_state.pop("_last_dismissed", None)  # 되돌리기 안내는 그 화면에서만
    st.rerun()


def go_back() -> None:
    if st.session_state.nav_stack:
        st.session_state.screen = st.session_state.nav_stack.pop()
    else:
        st.session_state.screen = "S0"
    st.rerun()


def back_button() -> None:
    if st.button("← 뒤로", key=f"back_{st.session_state.screen}"):
        go_back()


def _require_resume() -> bool:
    """이력서 없이 진입하면(사이드바에서 바로 이동한 경우 등) 안내만 하고
    False. 공고 찾기/직접 공고 분석은 이력서가 있어야 동작한다."""
    if st.session_state.get("resume_raw"):
        return True
    st.markdown(
        empty_state_html("먼저 이력서를 업로드해주세요.",
                         "‘공고 찾기’에서 이력서를 올리면 공고 탐색·분석을 시작할 수 있어요."),
        unsafe_allow_html=True,
    )
    if st.button("공고 찾기로 가기", key=f"goto_search_{st.session_state.screen}", type="primary"):
        go("S1")
    return False


def reset_all() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def show_llm_error(prefix: str, e: Exception) -> None:
    print(f"[LLM 오류] {prefix}: {e}")
    if IS_DEMO and "[DEMO MODE]" in str(e):
        st.info(f"🔒 {prefix} - 이 기능은 DEMO MODE에서 비활성화되어 있습니다 "
                f"(준비된 예시 4개 공고만 사전 계산된 결과로 볼 수 있습니다).")
    else:
        st.error(f"{prefix} - LLM 호출에 실패했습니다. 잠시 후 다시 시도해주세요.")


def status_badge(status: str) -> str:
    return badge_html(status, STATUS_TONE.get(status, "gray"))


# ── LOGIN. 시작 화면(게스트 모드 중심) ─────────────────────────────────
# 실제 계정 백엔드/OAuth는 구현하지 않는다(사용자 명시적 결정, 2026-07-14:
# "OAuth 시간에 JD Engine/Resume Engine/Recommendation 만든다") - 자리만
# 잡아두고(UI), 나중에 실제 서비스로 확장할 때 Google OAuth를 이 버튼
# 자리에 그대로 꽂을 수 있는 구조만 유지한다. 지금은 두 버튼 모두
# 결국 게스트로 진입시킨다 - Google 버튼은 "데모에서는 미지원" 안내 후
# 게스트 진입을 유도한다.

def render_login() -> None:
    _, mid, _ = st.columns([1, 2.6, 1])
    with mid:
        with st.container(key="auth-card-main"):
            _demo_badge = (
                "<span title='" + DEMO_BANNER_TEXT + "' "
                "style='margin-left:8px;font-size:0.62rem;font-weight:700;color:#1E3A8A;"
                "background:#E7ECFB;border-radius:999px;padding:2px 8px;vertical-align:middle;'>DEMO</span>"
            ) if IS_DEMO else ""
            st.markdown(f"<div class='sc-auth-logo'>JobFit AI<span>+</span></div>{_demo_badge}", unsafe_allow_html=True)
            st.markdown("<div class='sc-auth-sub'>AI가 당신에게 딱 맞는 공고를 찾아드려요.</div>", unsafe_allow_html=True)

            if st.button("게스트로 시작하기", type="primary", use_container_width=True, key="guest-start-btn"):
                st.session_state.user_display_name = "게스트"
                go("S0")

            if st.button("Google로 시작하기", key="google-btn-login"):
                st.session_state.show_google_demo_msg = True

            if st.session_state.get("show_google_demo_msg"):
                st.info("Google 로그인은 데모 버전에서는 지원하지 않습니다.")
                if st.button("게스트로 계속하기", type="primary", use_container_width=True, key="google-fallback-guest-btn"):
                    st.session_state.user_display_name = "게스트"
                    go("S0")

            st.markdown("<div class='sc-auth-footer'>로그인 없이 체험 가능합니다.</div>", unsafe_allow_html=True)


# ── S0. 메인 ──────────────────────────────────────────────────────────

# JobFit AI Design System v1(2026-07-24, White+Navy only - 사용자 확정)의
# :root 토큰. STEP1/STEP2가 같은 값을 쓰도록 한 곳에만 정의한다 - 화면마다
# 복붙하면 하나만 고쳤을 때 두 화면 색이 어긋난다(사용자 요청, 2026-07-24:
# "다음단계 공고목록도 첫번째에서 한 디자인을 통일한다 생각해").
_DESIGN_TOKENS_CSS = """
:root {
    /* White + Navy only(2026-07-24, 사용자 확정 - "포인트 컬러는 나중에,
       지금은 White+Navy로 완성한다"). 핑크/초록/보라/블루 파스텔 톤과
       그라데이션은 전부 제거했다. */
    --navy: #0F1F4A;
    --navy-hover: #1A2C63;
    --text: #0F1F4A;
    --muted: #6B7280;
    --muted-strong: #4B5563;
    --disabled: #C8D0DD;
    --white: #FFFFFF;
    --line: #E8EDF5;
    --divider: #EEF2F7;

    /* 4px spacing scale - 모든 margin/padding/gap은 이 값만 쓴다. */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 20px;
    --space-6: 24px;
    --space-7: 32px;
    --space-8: 40px;

    /* Icon frame - PNG마다 실제 그림 비율이 달라도(icon_png()가
       Pillow로 측정해 렌더 크기를 보정) frame box 자체는 역할별로
       고정이라 아이콘끼리 자리가 항상 맞는다. */
    --icon-frame-xs: 28px;
    --icon-frame-sm: 32px;
    --icon-frame-md: 40px;
    --icon-frame-lg: 44px;

    /* Button */
    --btn-height: 38px;
    --btn-radius: 10px;
    --btn-padding-x: 18px;

    /* Card - 2026-08-31 디자인 시스템 통일: radius 12px, 흰 배경, soft shadow, 테두리 없음 */
    --card-radius: 12px;
    --card-padding-x: 20px;
    --card-padding-y: 16px;
    --section-gap: 24px;

    /* Typography scale */
    --font-hero: 28px;
    --font-section: 20px;
    --font-card-title: 18px;
    --font-body: 14px;
    --font-meta: 13px;

    --shadow-soft: 0 4px 20px rgba(30, 58, 138, 0.06);
    --shadow-hover: 0 8px 24px rgba(30, 58, 138, 0.12);
    --card-bg: #FFFFFF;
}
"""

# 공통 프리미티브(2026-07-24) - 아이콘 프레임/아이콘+제목 행/meta 블록/
# 스킬 칩/불릿 목록. STEP1(요약 카드)과 STEP2(공고 카드)가 둘 다 쓴다 -
# 화면마다 각자 정의하면 하나만 고쳤을 때 두 화면 스타일이 어긋난다.
_COMPONENT_CSS = """
.icon-frame {
    width: var(--icon-frame-md);
    height: var(--icon-frame-md);
    flex: 0 0 var(--icon-frame-md);
    display: flex;
    align-items: center;
    justify-content: center;
}
.icon-frame img {
    display: block;
}
.icon-title-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}
.icon-title-text {
    color: var(--navy);
    font-size: var(--font-card-title);
    font-weight: 800;
    line-height: 1.3;
}

.meta-block {
    min-height: 28px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: var(--space-1);
    padding: 0 20px;
}
.meta-block--divider {
    border-left: 1px solid var(--line);
    padding-left: var(--space-6);
}
.meta-label {
    color: var(--muted);
    font-size: var(--font-meta);
    font-weight: 400;
    line-height: 1.3;
    margin: 0;
}
.meta-value {
    color: var(--navy);
    font-size: 15px;
    font-weight: 800;
    line-height: 1.3;
    margin: 0;
}

.summary-list {
    margin: 0;
    padding-left: var(--space-5);
    color: #425684;
    line-height: 1.85;
    font-size: var(--font-body);
}
.summary-list li {
    margin-bottom: var(--space-2);
}

.skill-wrap {
    display: flex;
    flex-wrap: wrap;
    align-content: flex-start;
    column-gap: var(--space-4);
    row-gap: var(--space-3);
}
.skill-chip {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 30px;
    background: #F5F7FA;
    color: var(--navy);
    border: none;
    border-radius: 999px;
    padding: var(--space-2) var(--space-4);
    font-size: 12px;
    font-weight: 700;
    line-height: 1.2;
    white-space: nowrap;
}
"""

_STEP1_CSS = """
<style>
/* ══════════════════════════════════════════════════════════════
   JobFit AI Design System v1(2026-07-24) - STEP1 스코프, White+Navy
   only(포인트 컬러/그라데이션 없음). 개별 카드에 px를 따로 넣지 않는다.
   모든 카드/버튼/아이콘/타이포는 아래 토큰 + 공통 클래스(icon-frame,
   meta-block 등)만 쓴다 - 정렬은 Flex/Grid가 자동으로 맞추고,
   margin-top/transform/top 같은 개별 보정은 쓰지 않는다.
   render_s0()에서만 주입되므로 다른 화면엔 영향 없음.
   ══════════════════════════════════════════════════════════════ */
""" + _DESIGN_TOKENS_CSS + _COMPONENT_CSS + """

/* .stApp 배경은 ui_theme.py 전역(var(--bg), 로그인과 동일)만 쓴다 -
   여기서 흰색으로 덮지 않는다(2026-08-31 배경 통일). */
.block-container,
div[data-testid="stMainBlockContainer"] {
    max-width: 1000px !important;
    padding-top: 30px !important;
    padding-bottom: 48px !important;
}
@media (max-width: 768px) {
    .block-container,
    div[data-testid="stMainBlockContainer"] {
        padding-left: var(--space-4) !important;
        padding-right: var(--space-4) !important;
    }
    :root {
        --card-padding-x: var(--space-4);
        --card-padding-y: var(--space-4);
    }
    .hero-title {
        font-size: 24px;
    }
    .action-desc {
        padding-left: 0;
    }
    .meta-block--divider {
        padding-left: var(--space-4);
    }
}

div[class*="st-key-main-shell"] {
    width: 100% !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

div[class*="st-key-status-card"] div:has(> [data-testid="stMarkdownContainer"] .meta-block) {
    height: auto !important;
}

/* ── Hero ─────────────────────────────────────────────────────── */
/* hero -> section-label, section-label -> card 간격은 여기 CSS margin/
   padding으로 주지 않는다(2026-07-25, Chrome DevTools getBoundingClientRect
   로 실측 후 확정) - .hero-subtitle/.section-label을 담은 st.markdown()
   호출의 element-container는 Streamlit이 처음 측정한 높이로 굳어서,
   그 뒤에 이 <style>이 margin/padding으로 콘텐츠를 늘려도(심지어 같은
   markdown 호출 안에 고정 높이 spacer <div>를 추가해도) element-container
   자체 높이가 그만큼 정확히 늘지 않는다 - 실측 결과 항상 16px가
   빠짐(의도한 간격 - 16px만 육안에 보임). render_s0_hero()/render_s0()의
   spacer <div style="height:...">가 의도한 값보다 16px 더 큰 값
   (24->40, 16->32)으로 박혀 있는 건 오타가 아니라 이 16px 결손을
   메우기 위한 보정값이다 - DevTools로 재측정해서 실제 렌더 간격이
   24px/16px로 나오는 것까지 확인했다(main-shell 전체 gap="small" 한
   값으로 통일했을 때는 이 문제가 없었다 - flex 아이템 사이 gap은
   정상 작동, 문제는 "하나의 st.markdown 호출 안에서" 콘텐츠로 간격을
   늘리려 할 때만 발생). 이 스펙서 값을 손대야 한다면 반드시 브라우저에서
   getBoundingClientRect()로 재실측할 것 - 코드만 보고 픽셀 값을
   판단할 수 없다. */
.hero-icon-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    margin: 0 0 8px 0;
}
.hero-icon-row .icon-frame {
    width: 40px;
    height: 40px;
    flex: 0 0 40px;
}
.hero-title {
    color: var(--navy);
    font-size: var(--font-hero);
    font-weight: 800;
    line-height: 1.25;
    letter-spacing: -0.6px;
    margin: 0;
}
.hero-subtitle {
    color: var(--muted-strong);
    font-size: 15px;
    font-weight: 500;
    line-height: 1.5;
    margin: 0;
}

.section-label {
    color: var(--navy);
    font-size: 17px;
    font-weight: 800;
    line-height: 1.3;
    margin: 0;
}

/* ── 업로드 / 처리완료 카드 ────────────────────────────────────── */
div[class*="st-key-upload-card"],
div[class*="st-key-status-card"] {
    width: 100% !important;
    box-sizing: border-box !important;
    border-radius: var(--card-radius) !important;
    padding: var(--card-padding-y) var(--card-padding-x) !important;
    margin: 0 !important;
    box-shadow: var(--shadow-soft) !important;
    overflow: visible !important;
    transition: transform .18s ease, box-shadow .18s ease;
}
div[class*="st-key-upload-card"]:hover,
div[class*="st-key-status-card"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover) !important;
}
div[class*="st-key-upload-card"] {
    border: none !important;
    background: var(--card-bg) !important;
    margin-bottom: 0 !important;
    padding: 16px 30px !important;
}
div[class*="st-key-status-card"] {
    min-height: 90px !important;
    border: none !important;
    background: var(--card-bg) !important;
    margin-bottom: var(--section-gap) !important;
}

.file-copy {
    min-width: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: var(--space-1);
    padding-left: 8px;
}
.file-name {
    color: var(--navy);
    font-size: 16px;
    font-weight: 700;
    line-height: 1.35;
    margin: 0;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
.file-meta {
    color: var(--muted);
    font-size: var(--font-meta);
    font-weight: 400;
    line-height: 1.4;
    margin: 0;
}

.status-icon-row {
    margin-bottom: 8px;
}
.status-icon-row .icon-frame {
    width: 34px;
    height: 34px;
    flex: 0 0 34px;
}
.status-icon-row .icon-title-text {
    font-size: 17px;
}

div[class*="st-key-upload-card"] .stButton > button,
div[class*="st-key-status-card"] .stButton > button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: auto !important;
    min-width: 84px !important;
    height: var(--btn-height) !important;
    min-height: var(--btn-height) !important;
    padding: 0 var(--btn-padding-x) !important;
    border: 1px solid var(--navy) !important;
    border-radius: var(--btn-radius) !important;
    background: var(--white) !important;
    color: var(--navy) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    box-shadow: none !important;
}
div[class*="st-key-upload-card"] .stButton > button:hover,
div[class*="st-key-status-card"] .stButton > button:hover {
    border-color: var(--navy-hover) !important;
    color: var(--navy-hover) !important;
    background: #F4F6F9 !important;
}

.status-footer-divider {
    border-top: 1px solid var(--line);
    margin: 8px 0 6px 0;
}
.challenge-copy {
    min-height: 24px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-2);
    margin: 0;
    padding: 0 20px;
}
.challenge-title {
    color: var(--navy);
    font-size: 14px;
    font-weight: 700;
    line-height: 1.4;
}
.challenge-desc {
    color: var(--muted);
    font-size: var(--font-meta);
    font-weight: 400;
    line-height: 1.4;
}
div[class*="st-key-status-card"] div[data-testid="stToggle"] {
    margin: 0 !important;
    padding: 0 !important;
}
div[class*="st-key-status-card"] label {
    margin: 0 !important;
}

/* ── 내 이력서 한눈에 보기 ────────────────────────────────────── */
.summary-heading {
    color: var(--navy);
    font-size: var(--font-section);
    font-weight: 800;
    line-height: 1.3;
    margin: 0 0 8px 0;
}
.summary-meta {
    color: var(--muted);
    font-size: var(--font-meta);
    line-height: 1.4;
    margin: 0 0 var(--space-4) 0;
}

div[class*="st-key-summary-card-"] {
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    border: none !important;
    border-radius: var(--card-radius) !important;
    background: var(--card-bg) !important;
    padding: 22px 24px !important;
    box-shadow: var(--shadow-soft) !important;
    overflow: visible !important;
    transition: transform .18s ease, box-shadow .18s ease;
}
div[class*="st-key-summary-card-"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover) !important;
}

.summary-icon-row .icon-frame {
    width: 44px;
    height: 44px;
    flex: 0 0 44px;
    background: #F4F7FD;
    border-radius: 10px;
}
.summary-body {
    flex: 1 1 auto;
    min-width: 0;
}

div[class*="st-key-linkbtn-skill"] {
    text-align: center;
}
div[class*="st-key-linkbtn-skill"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--navy) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: var(--space-2) 0 0 0 !important;
    width: auto !important;
}
div[class*="st-key-linkbtn-skill"] button:hover {
    text-decoration: underline;
}

/* ── 어떤 방식으로 진행할까요? ────────────────────────────────── */
.action-heading {
    color: var(--navy);
    font-size: var(--font-section);
    font-weight: 800;
    line-height: 1.3;
    margin: var(--section-gap) 0 var(--space-4) 0;
}

div[class*="st-key-action-card-"] {
    min-height: 116px !important;
    box-sizing: border-box !important;
    border: none !important;
    background: var(--card-bg) !important;
    border-radius: var(--card-radius) !important;
    padding: var(--space-4) !important;
    margin-bottom: var(--space-4) !important;
    box-shadow: var(--shadow-soft) !important;
    overflow: visible !important;
    transition: transform .18s ease, box-shadow .18s ease;
}
div[class*="st-key-action-card-"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover) !important;
}
/* 화살표 = 진짜 st.button(Native Layout 전환) - 위치는 이 카드를 감싸는
   st.container(horizontal=True, horizontal_alignment="distribute")가
   담당하므로, 여기선 버튼의 "생김새"(원형)만 정한다. */
div[class*="st-key-action-card-"] .stButton > button {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    border: 1px solid #DCE5F4 !important;
    background: var(--white) !important;
    color: var(--navy) !important;
    font-size: 18px !important;
    font-weight: 400 !important;
    padding: 0 !important;
    box-shadow: none !important;
    transition: border-color .18s ease, transform .18s ease;
}
div[class*="st-key-action-card-"] .stButton > button:hover {
    border-color: var(--navy) !important;
    transform: translateX(2px);
}
/* 내용 블록 + 화살표는 항상 한 줄로(줄바꿈되면 화살표가 아래로 내려가
   카드 높이가 옆 카드와 달라진다). 내용 블록은 대신 줄어든다. */
div[class*="st-key-action-card-"] div[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
}
div[class*="st-key-action-card-"] div[data-testid="stHorizontalBlock"] > div[data-testid="stElementContainer"]:first-child {
    min-width: 0 !important;
}

.action-main {
    min-width: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: var(--space-3);
}
.action-heading-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}
.action-heading-row .icon-frame {
    width: 44px;
    height: 44px;
    flex: 0 0 44px;
}
.action-title {
    color: var(--navy);
    font-size: var(--font-card-title);
    font-weight: 800;
    line-height: 1.3;
    margin: 0;
}
.action-desc {
    color: var(--muted);
    font-size: var(--font-meta);
    font-weight: 400;
    line-height: 1.5;
    margin: 0;
    padding-left: calc(44px + var(--space-3));
    overflow-wrap: break-word;
}

div[class*="st-key-upload-card"] div[data-testid="stFileUploader"] {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
div[class*="st-key-upload-card"] div[data-testid="stFileUploaderDropzone"] {
    min-height: auto !important;
    padding: 0 !important;
}
/* 공고 목록 페이지네이션 번호를 가운데로(감싼 container 는 center 인데
   위젯 내부 div 가 flex-start 라 왼쪽으로 쏠림 - S2/S7 와 동일 처리). */
.stPagination { justify-content: center !important; }
</style>
"""


def render_s0_hero(name: str) -> None:
    st.markdown(
        f"""
        <div class="hero-inline">
          <div class="hero-icon-row">
            <div class="hero-title">안녕하세요, {html.escape(name)}님</div>
            <div class="icon-frame">{icon_png("sparkles_neon", 40)}</div>
          </div>
          <div style="height:32px"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resume_upload_card(show_uploader: bool) -> None:
    """이력서 업로드 카드. show_uploader=True면 업로더 위젯을 그리고
    업로드 후 처리(텍스트 추출/연차 감지/Understanding)까지 이 함수
    안에서 끝낸다. False면 완료 상태(파일명 + 이력서 교체 버튼)만
    그린다 - 두 상태 모두 같은 2열 레이아웃([텍스트 | 버튼])을 쓴다
    (아이콘 제거 - White+Navy 원칙)."""
    if show_uploader:
        with st.container(border=True, key="upload-card", height=84, vertical_alignment="center"):
            ucol2, ucol3 = st.columns([5.2, 1.8], vertical_alignment="center")
            with ucol2:
                st.markdown(
                    "<div class='file-copy'>"
                    "<div class='file-name'>PDF를 업로드해주세요</div>"
                    "<div class='file-meta'>최대 10MB</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with ucol3:
                uploaded = st.file_uploader(
                    "이력서 업로드", type=["pdf"],
                    key=f"resume_uploader_{st.session_state.uploader_version}",
                    label_visibility="collapsed",
                )

        if uploaded is not None:
            file_id = f"{uploaded.name}:{uploaded.size}"
            if file_id != st.session_state.uploaded_file_id:
                try:
                    # 원본 PDF 바이트를 보관한다 - 맞춤 PDF 생성(pdf_layout)이
                    # 업로드된 PDF 자체에서 좌표/폰트를 런타임에 읽는다.
                    _pdf_bytes = uploaded.getvalue()
                    with st.spinner("이력서 텍스트를 추출하는 중입니다..."):
                        text = extract_text_from_pdf(io.BytesIO(_pdf_bytes))
                    st.session_state.resume_raw = text
                    st.session_state.resume_pdf_bytes = _pdf_bytes
                    st.session_state.uploaded_file_id = file_id
                    st.session_state.resume_file_name = uploaded.name
                    st.session_state.resume_file_size = uploaded.size
                    st.session_state.resume_uploaded_at = datetime.now().strftime("%Y.%m.%d %H:%M")

                    detected = pipeline.detect_user_career_level(text)
                    st.session_state.user_career_level = detected["career_level"]
                    st.session_state.career_level_confidence = detected["confidence"]
                    st.session_state.career_level_reason = detected["reason"]
                    st.session_state.career_level_source = "auto"
                    st.session_state.top_cache = {}

                    # Resume Understanding(사용자 확정) - 업로드 직후
                    # 바로 실행. resume_hash 기준 캐시라 최초 1회만
                    # 이 스피너를 거친다.
                    with st.spinner("이력서를 분석하여 핵심 역량을 정리하고 있습니다. 처음 한 번만 시간이 소요됩니다(최대 1~2분)..."):
                        understanding = get_or_generate_resume_understanding(text)
                    st.session_state.resume_semantic_objects = understanding.get("semantic_objects") or []
                    st.session_state.show_uploader = False
                    # 세션이 끊겨도 다음 진입 때 홈 대시보드가 채워지도록
                    # 마지막 이력서 PDF 1부를 디스크에 보관한다(편의 기능,
                    # 판정/랭킹과 무관).
                    try:
                        last_resume.save(_pdf_bytes, uploaded.name, uploaded.size)
                    except Exception as _e:  # noqa: BLE001
                        print(f"[last_resume] save skip: {_e}", flush=True)
                    # 새 이력서 업로드 직후에는 공고 목록으로 자동 진입하지
                    # 않는다 - "어떤 방식으로 진행할까요?" 선택 화면을 다시 보인다.
                    st.session_state.s1_browsing = False
                    st.rerun()
                except LLMCallError as e:
                    show_llm_error("이력서 분석", e)
                except Exception as e:
                    print(f"[PDF 파싱/이해 오류] {e}")
                    st.error("이력서를 분석하지 못했습니다. 다른 파일로 시도해주세요.")
        return

    size = st.session_state.resume_file_size or 0
    size_kb = max(1, round(size / 1024))
    upload_date = (st.session_state.resume_uploaded_at or "").split(" ")[0]

    with st.container(border=True, key="upload-card", height=84, vertical_alignment="center"):
        ucol2, ucol3 = st.columns([5.2, 1.8], vertical_alignment="center")
        with ucol2:
            st.markdown(
                "<div class='file-copy'>"
                f"<div class='file-name'>{html.escape(st.session_state.resume_file_name or '이력서.pdf')}</div>"
                f"<div class='file-meta'>업로드 완료 &nbsp;&nbsp; {upload_date} · {size_kb}KB</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with ucol3:
            if st.button("이력서 교체", key="replace-resume-btn"):
                st.session_state.uploader_version += 1
                st.session_state.uploaded_file_id = None
                st.session_state.show_uploader = True
                st.rerun()


def render_resume_status_card(level: str, upload_date: str) -> None:
    with st.container(border=True, key="status-card"):
        st.markdown(
            icon_title_row("check_neon_blue", "이력서 분석 완료", css_class="status-icon-row icon-title-row", variant="colors", icon_size=34),
            unsafe_allow_html=True,
        )
        scol1, scol2, scol3 = st.columns([2.2, 2.2, 0.95], vertical_alignment="center")
        scol1.markdown(meta_block("인식된 연차", level), unsafe_allow_html=True)
        scol2.markdown(meta_block("분석 완료일", upload_date, divider=True), unsafe_allow_html=True)
        with scol3:
            if st.button("연차 수정", key="career-edit-btn"):
                st.session_state.career_level_source = "manual"
                st.rerun()

        if st.session_state.career_level_source == "manual":
            chosen = st.selectbox(
                "연차 직접 선택", CAREER_LEVEL_OPTIONS,
                index=CAREER_LEVEL_OPTIONS.index(level) if level in CAREER_LEVEL_OPTIONS else 0,
                key="career_level_select",
            )
            if chosen != st.session_state.user_career_level:
                st.session_state.user_career_level = chosen
                st.session_state.top_cache = {}

        st.markdown('<div class="status-footer-divider"></div>', unsafe_allow_html=True)
        fcol1, fcol2 = st.columns([5.4, 0.8], vertical_alignment="center")
        fcol1.markdown(
            """
            <div class="challenge-copy">
                <span class="challenge-title">도전 공고 포함</span>
                <span class="challenge-desc">(내 연차보다 조금 더 높은 공고도 함께 보기)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with fcol2:
            challenge = st.toggle("도전 공고 포함", value=st.session_state.challenge_option,
                                   label_visibility="collapsed", key="challenge-toggle")
    if challenge != st.session_state.challenge_option:
        st.session_state.challenge_option = challenge
        st.session_state.top_cache = {}


def render_resume_summary_cards(summary: dict) -> None:
    for k in ("s0_expand_comp", "s0_expand_skill", "s0_expand_exp"):
        if k not in st.session_state:
            st.session_state[k] = False
    skill_expanded = st.session_state.s0_expand_skill
    shown_skills = summary["skills"] if skill_expanded else summary["skills"][:8]
    has_more_skills = len(summary["skills"]) > 8

    st.markdown(
        f"""
        <div class="summary-heading">내 이력서 한눈에 보기</div>
        <div class="summary-meta">✓ 분석 완료 · 핵심 역량 {summary['core_competencies_total']}개 · 기술 {summary['skills_total']}개 · 경험 {summary['representative_experiences_total']}개 추출되었습니다</div>
        """,
        unsafe_allow_html=True,
    )

    def _bullet_list(items: list[str]) -> str:
        lis = "".join(f"<li>{html.escape(x)}</li>" for x in items)
        return f"<ul class='summary-list'>{lis}</ul>"

    sum_cols = st.columns(3, gap="medium")
    with sum_cols[0]:
        with st.container(border=True, key="summary-card-comp", height="stretch", gap="medium"):
            st.markdown(icon_title_row("user", "핵심 역량", css_class="summary-icon-row icon-title-row", variant="flat_navy", icon_size=44), unsafe_allow_html=True)
            if summary["core_competencies"]:
                st.markdown(f"<div class='summary-body'>{_bullet_list(summary['core_competencies'])}</div>", unsafe_allow_html=True)
            else:
                st.caption("확인 중입니다")
    with sum_cols[1]:
        with st.container(border=True, key="summary-card-skill", height="stretch", gap="medium"):
            st.markdown(icon_title_row("settings", "주요 기술", css_class="summary-icon-row icon-title-row", variant="flat_navy", icon_size=44), unsafe_allow_html=True)
            if summary["skills"]:
                chips = "".join(f"<span class='skill-chip'>{html.escape(s)}</span>" for s in shown_skills)
                st.markdown(f"<div class='summary-body'><div class='skill-wrap'>{chips}</div></div>", unsafe_allow_html=True)
                if has_more_skills:
                    with st.container(key="linkbtn-skill"):
                        if st.button("접기" if skill_expanded else "+ 더 보기", key="btn-skill"):
                            st.session_state.s0_expand_skill = not skill_expanded
                            st.rerun()
            else:
                st.caption("확인 중입니다")
    with sum_cols[2]:
        with st.container(border=True, key="summary-card-exp", height="stretch", gap="medium"):
            st.markdown(icon_title_row("briefcase", "대표 경험", css_class="summary-icon-row icon-title-row", variant="flat_navy", icon_size=44), unsafe_allow_html=True)
            if summary["representative_experiences"]:
                st.markdown(f"<div class='summary-body'>{_bullet_list(summary['representative_experiences'])}</div>", unsafe_allow_html=True)
            else:
                st.caption("확인 중입니다")


def render_s0_action_cards(has_resume: bool) -> None:
    """"어떤 방식으로 진행할까요?" 선택 카드(2026-08-31 - 구 홈에서 공고
    찾기(S1)로 이동, 홈은 대시보드로 분리). 이력서 준비가 끝나도 공고
    목록으로 자동 진입하지 않고 여기서 사용자가 선택한다:
    - 채용공고 탐색하기 → s1_browsing=True (같은 화면에서 목록으로 전환)
    - 직접 공고 분석하기 → S2(붙여넣기 분석)

    화살표 버튼은 진짜 st.button 하나다(Native Layout 전환, 2026-07-25) -
    st.container(horizontal=True, horizontal_alignment="distribute")가
    내용 블록과 버튼을 양 끝으로 민다."""
    st.markdown('<div class="action-heading">어떤 방식으로 진행할까요?</div>', unsafe_allow_html=True)
    acol1, acol2 = st.columns(2, gap="medium")
    with acol1:
        with st.container(border=True, key="action-card-search"):
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.markdown(
                    action_card_content("search", "채용공고 탐색하기", "수집된 채용공고를 검색하고 비교해보세요.", variant="flat_navy", icon_size=44),
                    unsafe_allow_html=True,
                )
                if st.button("→", key="mode-recommend-btn"):
                    if has_resume:
                        st.session_state.s1_browsing = True
                        st.rerun()
                    else:
                        st.warning("먼저 이력서를 업로드해주세요.")
    with acol2:
        with st.container(border=True, key="action-card-analyze"):
            with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
                st.markdown(
                    action_card_content("clipboard", "직접 공고 분석하기", "채용공고를 붙여넣어 바로 분석해보세요.", variant="flat_navy", icon_size=44),
                    unsafe_allow_html=True,
                )
                if st.button("→", key="mode-analyze-btn"):
                    if has_resume:
                        go("S2")
                    else:
                        st.warning("먼저 이력서를 업로드해주세요.")


_DECISION_LABEL = {"지원": "지원 추천", "보류": "보류", "비추천": "비추천"}


def _recent_analysis_decision(row: dict, resume_facts: dict) -> str | None:
    """semantic_link_cache 한 행(quick-analysis-v1 결과 또는 구 link_engine
    결과)에서 Rule 판단(지원/보류/비추천)만 뽑는다. LLM 호출 없음 - 이미
    저장된 분석 결과를 judge_engine(Rule)로 재판정만 한다. 실패하면 None."""
    try:
        data = json.loads(row.get("result_json") or "{}")
    except (TypeError, ValueError):
        return None
    is_quick = (
        str(row.get("linking_version") or "").startswith("quick-analysis")
        or str(data.get("schema_version") or "").startswith("quick-analysis")
    )
    if is_quick:
        link_result = quick_analysis.to_link_result(data)
        jd_objs = quick_analysis.to_jd_objects(data)
    else:
        link_result, jd_objs = data, []
    if not link_result.get("links"):
        return None
    try:
        return judge_engine.judge(link_result, jd_objs, resume_facts).get("decision")
    except Exception:  # noqa: BLE001 - 표시용이라 실패 시 배지만 생략
        return None


def _dash_stat(label: str, value, delta_today: int = 0, align: str = "left") -> None:
    """요약 카드 한 칸. align: 첫 칸 left / 가운데 칸 center / 마지막 칸 right
    → 첫 값은 카드 왼쪽 여백에, 마지막 값은 오른쪽 여백에 붙어 좌우 대칭."""
    delta = (
        f"<span style='font-size:.8rem;font-weight:600;color:var(--accent);margin-left:6px;'>+{delta_today} 오늘</span>"
        if delta_today else ""
    )
    st.markdown(
        f"<div style='text-align:{align};font-size:.9rem;color:var(--muted);'>{label}</div>"
        f"<div style='text-align:{align};font-size:2rem;font-weight:800;line-height:1.4;margin-top:4px;'>{value}{delta}</div>",
        unsafe_allow_html=True,
    )


_STAT_ALIGN_4 = ["center", "center", "center", "center"]


# 화면 맨 아래 "다음 단계로" CTA 는 네이비 채움 + 흰 글자 + 가운데.
# (전역 버튼은 흰 배경/네이비 테두리 - 이 CTA 들만 스코프 override)
_NAVY_CTA_CSS = """
<style>
div[class*="st-key-navycta-"] { display:flex; justify-content:center; }
div[class*="st-key-navycta-"] button {
    background: #0F1F4A !important;
    border-color: #0F1F4A !important;
    padding: 0 40px !important;
}
div[class*="st-key-navycta-"] button p,
div[class*="st-key-navycta-"] button span { color: #FFFFFF !important; }
div[class*="st-key-navycta-"] button:hover {
    background: #1A2C63 !important;
    border-color: #1A2C63 !important;
}
div[class*="st-key-navycta-"] button:hover p,
div[class*="st-key-navycta-"] button:hover span { color: #FFFFFF !important; }
</style>
"""


def _navy_cta(label: str, key: str, **kwargs) -> bool:
    """가운데 정렬된 네이비 CTA 버튼. key 는 'navycta-' 접두어가 붙는다."""
    st.markdown(_NAVY_CTA_CSS, unsafe_allow_html=True)
    _, _mid, _ = st.columns([1, 1.6, 1])
    with _mid:
        return st.button(label, key=f"navycta-{key}", use_container_width=True, **kwargs)


def render_s0() -> None:
    """홈 = 지원 현황 대시보드(2026-08-31, 사용자 확정). "지금 내 취업 진행
    상황"만 보여준다 - 이력서 업로드/공고 탐색은 여기 없다(그건 "공고 찾기"
    화면의 역할). 새 집계 시스템/새 LLM 없음 - 이미 있는 DB(applications/
    saved_jobs/semantic_link_cache)만 읽는다. 없는 값은 "-"로 둔다."""
    st.markdown(_STEP1_CSS, unsafe_allow_html=True)
    name = st.session_state.user_display_name or "게스트"

    st.markdown(
        f"<div class='hero-inline'><div class='hero-icon-row'>"
        f"<div class='hero-title'>안녕하세요, {html.escape(name)}님</div>"
        f"<div class='icon-frame'>{icon_png('sparkles_neon', 40)}</div></div>"
        f"<div style='height:28px'></div></div>",
        unsafe_allow_html=True,
    )

    apps = list_applications()
    saved = list_saved_jobs()
    in_progress_n = sum(1 for a in apps if a.get("status") in {"서류 진행", "과제/코테", "면접"})

    resume_raw = st.session_state.get("resume_raw")
    r_hash = _resume_hash(resume_raw) if resume_raw else None
    analyzed_total, analyzed_today = count_analyses(r_hash) if r_hash else (0, 0)

    st.markdown("#### 지원 현황 요약")
    # 흰 카드 안 4개 지표 + 항목 사이 세로 칸막이선. key 스코프로 이 카드에만 적용.
    st.markdown(
        """
        <style>
        div[class*="st-key-dash-summary"] [data-testid="stColumn"] { padding: 0 18px; }
        div[class*="st-key-dash-summary"] [data-testid="stColumn"]:not(:first-child) {
            border-left: 1px solid var(--line, #E2E8F0);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="dash-summary"):
        st.space(14)
        cols = st.columns(4)
        with cols[0]:
            _dash_stat("분석한 공고", analyzed_total if r_hash else "-", analyzed_today, align=_STAT_ALIGN_4[0])
        with cols[1]:
            _dash_stat("관심 공고", len(saved), align=_STAT_ALIGN_4[1])
        with cols[2]:
            _dash_stat("지원한 공고", len(apps), align=_STAT_ALIGN_4[2])
        with cols[3]:
            _dash_stat("진행 중", in_progress_n, align=_STAT_ALIGN_4[3])
        st.space(14)

    st.space(24)
    st.markdown("#### 최근 분석한 공고")
    # 최신 10건까지, 이 영역 안에서만 스크롤(전체를 다 펼치지 않는다).
    recent = list_recent_analyses(r_hash, limit=10) if r_hash else []
    if not recent:
        st.caption("아직 분석한 공고가 없습니다. ‘공고 찾기’에서 공고를 분석해보세요.")
    else:
        resume_facts = _get_resume_facts()
        _recent_box = st.container(height=336, border=False) if len(recent) > 4 else nullcontext()
        with _recent_box:
            for row in recent:
                job_id = row["job_id"]
                with st.container(border=True):
                    rc = st.columns([3, 3, 2, 1.8, 1.4])
                    rc[0].markdown(f"**{html.escape(row.get('company') or '(공고 정보 없음)')}**")
                    rc[1].write(row.get("title") or "")
                    decision = _recent_analysis_decision(row, resume_facts)
                    if decision:
                        rc[2].markdown(
                            badge_html(_DECISION_LABEL.get(decision, decision), DECISION_TONE.get(decision, "gray")),
                            unsafe_allow_html=True,
                        )
                    else:
                        rc[2].write("-")
                    rc[3].write((row.get("created_at") or "")[:10].replace("-", "."))
                    if rc[4].button("다시 보기", key=f"dash_reopen_{job_id}", use_container_width=True):
                        job = get_candidate_job(job_id)
                        if job is None:
                            st.toast("공고 정보를 찾을 수 없습니다(삭제되었을 수 있어요).")
                        else:
                            st.session_state.selected_job = job
                            go("S1D")

    st.space(24)
    # CTA - 카드/문구 없이 가운데 네이비 "공고 찾기" 버튼만.
    if _navy_cta("공고 찾기 →", "dash_search"):
        go("S1")


# ── S1. 채용공고 탐색 ────────────────────────────────────────────────
# 2026-07-23(서비스 구조 변경, 사용자 확정 - docs/verification/
# 2026-07-23_decision_thin_link_experiment/summary.md 근거) - 이 화면은
# "AI 추천 순위" 화면이 아니다. 검증 결과 Rule Ranking(옛 ranking_score/
# fit_grade)이 실제로 좋은 공고를 하위로 떨어뜨리는 사례가 확인돼서,
# 카드 자체에는 순위/적합도 숫자를 노출하지 않는다(지금도 그대로).
# M3+M4(Recall)는 pipeline.run_recommendation_mode() 안에서 "완전히
# 무관한 공고를 거르는" 용도로 쓰인다.
#
# 2026-08-01(사용자 요청, 재확정) - "정렬" 드롭다운에 "추천순" 옵션을
# 추가한다. Recall 단계에서 이미 계산된 combined_score(위 M3+M4/
# Rule→old_m3→BM25 fallback 결합 점수, candidate_search.
# search_candidates_meaning)를 정렬 키로만 재사용한다 - 카드에 점수를
# 노출하거나 새 채점식을 만드는 게 아니라 "같은 화면 안에서 순서만
# 고를 수 있게" 하는 것뿐이다. 기본값은 "추천순"이고, "최신순"/
# "회사명순"으로 언제든 바꿀 수 있다. 2026-08-01 Candidate Generation
# Bottleneck Diagnosis(FROZEN v1)에서 이 Top100 자체가 FP 54%로 노이즈가
# 있다고 이미 확인된 상태라 "추천순"이 정확한 적합도 순서를 보장하지는
# 않는다 - 그 한계를 알고도 사용자가 다시 노출하기로 결정했다.
# Link Engine은 여기서 실행하지 않는다 - 사용자가 [공고분석]을 클릭한
# 공고 1건에 대해서만 실행된다(_render_job_analysis_panel 참고).

# 2026-07-24 전면 리디자인(사용자 요청 - "공고를 5~10초 안에 판단할 수
# 있는 요약 카드", STEP1과 같은 White+Navy 디자인 시스템으로 통일) -
# 카드 HTML을 render_s1() 안에서 직접 조립하지 않는다. 데이터는
# presentation_layer.build_job_card_view(job)이 ViewModel로 만들고,
# 화면은 render_job_card(view)가 그 ViewModel만 출력한다 - candidate
# dict(job)를 이 두 함수 밖에서 직접 들여다보지 않는다. 카드 정보가
# 늘어나도 build_job_card_view()/render_job_card()만 고치면 된다.
# 2026-08-14 두 번째 재작성 - 첫 시도에서는 카드 테두리가 흰 배경 위에서
# 안 보이는 문제를 "흰 배경 강제를 빼고 구형 전역 테마(파스텔 블루)로
# 되돌리는" 방식으로 풀었는데, 사용자가 실제 브라우저 스크린샷을 STEP1
# (이력서 업로드)과 나란히 비교해서 지적한 결과 그게 틀렸음이 확인됐다:
# STEP1은 이미 그 구형 테마가 아니라 White+Navy 전용 디자인 시스템
# (_DESIGN_TOKENS_CSS, 2026-07-24 확정)을 쓰고 있어서, 이 화면만 파스텔
# 블루로 남아있는 게 오히려 실제 불일치였다. 카드 테두리는 배경을
# 파스텔 블루로 되돌리는 대신 STEP1의 카드(upload-card/status-card)와
# 똑같은 레시피 - border: 1px solid var(--line) + box-shadow: var(
# --shadow-soft) - 로 만들어서 흰 배경 위에서도 카드 구분이 보이게
# 했다(그 레시피 자체가 이미 STEP1에서 검증된 값).
_STEP2_CSS = """
<style>
""" + _DESIGN_TOKENS_CSS + """
:root {
    /* 이 화면(S2)에서만 쓰는 blue-indigo accent(2026-08-14, 사용자가
       준 레퍼런스 색 시스템 - "공고 직접 붙여넣기"의 outline/text
       색으로만 쓴다). 다른 화면과 공유하는 --navy/--white 같은 전역
       토큰은 건드리지 않고 이 화면 전용 변수만 추가한다. */
    --indigo: #3155D9;
    --indigo-soft: #EEF1FD;
}
/* .stApp 배경은 ui_theme.py 전역(var(--bg), 로그인과 동일)만 쓴다
   (2026-08-31 배경 통일 - 이 화면만 #F8FAFC 로 다르게 두지 않는다). */
.block-container,
div[data-testid="stMainBlockContainer"] {
    max-width: 1248px !important; /* 좌우 여백 64px*2 + 콘텐츠 폭 1120px */
    padding-top: 30px !important;
    padding-bottom: 48px !important;
    padding-left: 64px !important;
    padding-right: 64px !important;
}
/* 2026-08-16(실측 확인) - h1이 --navy(#0F1F4A) 토큰을 안 받고
   Streamlit 기본 텍스트색(#14213D)으로 렌더링되고 있었다 - 스펙의
   Text Primary #0F1F4A와 달랐다. 이 화면에서만 강제 지정. */
h1 {
    color: #0F1F4A !important;
}
/* 2026-08-31 - .sc-step-line 의 first/last 처리는 이제 ui_theme.py 전역에서
   올바르게 한다(:first-child 숨김 제거, :last-child 만 숨김). 여기 있던
   화면별 되돌림 규칙은 삭제. */
/* 2026-08-16(사용자 확정) - 공고 카드는 job_card_component.py(CCv2)로
   옮겼다. Streamlit이 만드는 wrapper DOM(div[class*="st-key-job-..."])에
   맞춰 픽셀을 추측하며 CSS를 쌓던 규칙들은 전부 제거 - 카드 내부
   레이아웃/색/간격은 이제 그 컴포넌트 자신의 CSS(Shadow DOM 안)가
   전담한다. */
/* "공고 직접 붙여넣기" - 레퍼런스는 진한 네이비 아웃라인이 아니라
   blue-indigo 아웃라인/텍스트를 쓴다(2026-08-14, 사용자 지적 - "현재의
   dark navy outline 스타일을 그대로 쓰지 않는다"). */
div[class*="st-key-goto_paste_mode"] button {
    background: #FFFFFF !important;
    border: 1px solid var(--indigo) !important;
    color: var(--indigo) !important;
}
div[class*="st-key-goto_paste_mode"] button p,
div[class*="st-key-goto_paste_mode"] button span {
    color: var(--indigo) !important;
}
div[class*="st-key-goto_paste_mode"] button:hover {
    background: var(--indigo-soft) !important;
    border-color: var(--indigo) !important;
}
/* pagination 가운데 정렬 - 감싸는 st.container(horizontal_alignment=
   "center")의 align-items:center는 실제로 적용돼 있었지만(실측 확인),
   st.pagination 위젯 자신의 내부 div(.stPagination)가 width:100% +
   justify-content:flex-start라서 왼쪽으로 쏠려 보였다(2026-08-14,
   사용자 지적 - "숫자 부분도 가운데에"). 위젯 안쪽 정렬을 직접
   가운데로 바꾼다 - 이 화면에서 st.pagination을 쓰는 곳이 여기
   하나뿐이라 전역이 아니라 이 화면 CSS에 둬도 다른 화면에 영향 없다.
   */
.stPagination {
    justify-content: center !important;
}
</style>
"""

# 사이드바 최상위 리스트 화면(지원 기록 S7 / 관심 공고 S9 / 제외한 공고 S11) 공용
# (2026-08-30, 사용자 지적 - "배경 색깔도 다른데... 이력서 업로드랑 목록부분이랑
# 똑같이해"). 이 화면들은 CSS 주입이 전혀 없어서 전역 테마의 파란빛 배경
# (--bg #F6F8FF)으로 샜었다 - "공고 찾기"(STEP2)와 같은 중립 배경 #F8FAFC +
# 같은 콘텐츠 폭/여백/제목색으로 통일한다. 카드 구조는 그대로 둔다.
_SUBLIST_CSS = """
<style>
""" + _DESIGN_TOKENS_CSS + """
/* .stApp 배경은 ui_theme.py 전역(var(--bg), 로그인과 동일)만 쓴다 (2026-08-31 배경 통일). */
.block-container,
div[data-testid="stMainBlockContainer"] {
    max-width: 1248px !important;
    padding-top: 30px !important;
    padding-bottom: 48px !important;
    padding-left: 64px !important;
    padding-right: 64px !important;
}
h1 { color: #0F1F4A !important; }
/* 지원 기록(S7) 페이지네이션 - 공고 목록과 동일하게 번호를 가운데로. */
.stPagination { justify-content: center !important; }
</style>
"""

# S1D(공고 분석)/S2(붙여넣기)/S3(판단 결과) 공용 - 이 화면들은 지금까지
# STEP1/STEP2용 디자인 토큰이 전혀 주입되지 않아서 전역 테마
# (ui_theme.py, 파스텔 블루 배경 + 다른 --navy 값)로 샜었다(사용자
# 실측 보고, 2026-07-25). 카드 구조 전체를 새로 짜지는 않고, 배경/폭/
# 색 토큰만 STEP1/STEP2와 맞춘다 - :root는 나중에 선언된 쪽이 이기므로
# 이걸 각 render_*() 맨 앞에서 주입하면 ui_theme.py의 --navy(#163A70)를
# 이 화면에서만 STEP1과 같은 값(#0F1F4A)으로 덮어쓴다.
_ANALYSIS_CSS = """
<style>
""" + _DESIGN_TOKENS_CSS + """
/* .stApp 배경은 ui_theme.py 전역(var(--bg), 로그인과 동일)만 쓴다 (2026-08-31 배경 통일). */
.block-container,
div[data-testid="stMainBlockContainer"] {
    max-width: 1240px !important;
    padding-top: 30px !important;
    padding-bottom: 48px !important;
}
/* st.tabs 활성 탭 글자색 - 실측(getComputedStyle)으로 확인한 버그:
   활성 탭 배경은 navy로 채워지는데, 탭 안 <p> 텍스트는 전역 --text-
   primary(#14213D, 짙은 네이비)로 덮여서 navy 배경 위에 거의 안 보이는
   짙은 네이비 글자가 된다(2026-08-14). 활성 탭일 때만 흰색으로
   강제한다. */
[role="tab"][aria-selected="true"] p {
    color: #FFFFFF !important;
}
</style>
"""


def _chip_row(items: list[str]) -> str:
    chips = "".join(f"<span class='skill-chip'>{html.escape(i)}</span>" for i in items)
    return f"<div class='skill-wrap'>{chips}</div>"


def _bullet_list(items: list[str]) -> str:
    bullets = "".join(f"<li>{html.escape(i)}</li>" for i in items)
    return f"<ul class='summary-list'>{bullets}</ul>"


@st.dialog("회사 알아보기")
def _render_company_modal(info: dict, company: str) -> None:
    """"회사 알아보기" 모달(2026-07-25 재설계) - presentation_layer.
    build_company_info_view()가 company_research 테이블(반자동 리서치,
    LLM 호출 없음)에서 만든 값을 그대로 보여준다. 새 판단/새 LLM 호출
    없음. 닫기(X)는 st.dialog 기본 제공.

    섹션 간 간격은 CSS margin(.job-divider) 대신 st.space("small")를
    쓴다(2026-07-25, 사용자 확인 - Streamlit 공식 skill의 layouts.md에
    이미 있던 네이티브 명령. 컨테이너 전체에 거는 gap= 파라미터는 값
    하나만 지정 가능해서 형제 요소 경계마다 다른 간격을 줄 수 없었던
    게 지금까지 간격이 어긋난 원인 - st.space는 원하는 두 요소 사이에만
    정확한 간격을 넣을 수 있어 CSS가 필요 없다)."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'{company_logo_html(company, size=48)}'
        f'<div><div style="font-size:1.3rem;font-weight:700;">{html.escape(company)}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(info["intro"])
    st.space("small")

    if not info["has_research"]:
        st.info("아직 상세 리서치가 안 된 회사입니다. 아래는 공고 원문에서 추출한 기본 정보입니다.")

    st.markdown(icon_title_row("building", "기본 정보", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
    basic_cols = st.columns(4)
    for col, (label, value) in zip(basic_cols, [
        ("산업", info["industry"]), ("회사 규모", info["company_size"]),
        ("설립연도", info["founded_year"]), ("본사 위치", info["location"]),
    ]):
        with col:
            st.caption(label)
            st.markdown(f"**{html.escape(value)}**")
    st.space("small")

    if info["products_services"]:
        st.markdown(icon_title_row("job", "주요 서비스", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.markdown(_chip_row(info["products_services"]), unsafe_allow_html=True)
        st.space("small")

    if info["business_model"] and info["business_model"] != "정보 없음":
        st.markdown(icon_title_row("bar_chart", "비즈니스 모델", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.write(info["business_model"])
        st.space("small")

    if info["culture"]:
        st.markdown(icon_title_row("user", "조직문화", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.markdown(_chip_row(info["culture"]), unsafe_allow_html=True)
        st.space("small")

    if info["work_style"]:
        st.markdown(icon_title_row("settings", "일하는 방식", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.markdown(_bullet_list(info["work_style"]), unsafe_allow_html=True)
        st.space("small")

    if info["benefits"]:
        st.markdown(icon_title_row("success", "주요 복지", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.markdown(_chip_row(info["benefits"]), unsafe_allow_html=True)
        st.space("small")

    if info["hiring_values"]:
        st.markdown(icon_title_row("target", "강조 가치관", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.markdown(_chip_row(info["hiring_values"]), unsafe_allow_html=True)
        st.space("small")

    if info["highlights"]:
        st.markdown(icon_title_row("chart", "주요 이슈", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        st.markdown(_bullet_list(info["highlights"]), unsafe_allow_html=True)
        st.space("small")

    if info["quick_summary"]:
        st.markdown(icon_title_row("clipboard", "3줄 요약", icon_size=18, variant="navy", css_class="job-section-label icon-title-row"), unsafe_allow_html=True)
        with st.container(border=True):
            for i, line in enumerate(info["quick_summary"], 1):
                st.markdown(f"{i}. {html.escape(line)}")
        st.space("small")

    domain = _match_domain(company)
    foot = st.columns(2)
    with foot[0]:
        if domain:
            st.link_button("홈페이지 방문", f"https://{domain}", width="stretch")
        else:
            st.button("홈페이지 방문", disabled=True, width="stretch")
    with foot[1]:
        if info["job_url"]:
            st.link_button("공고 원문 보기", info["job_url"], width="stretch")
        else:
            st.button("공고 원문 보기", disabled=True, width="stretch")

    if info["sources"]:
        st.caption("출처: " + ", ".join(info["sources"]))




# "공고 찾기" 목록에 유지할 최대 표시 개수. pipeline은 이 개수 +
# _DISMISS_RESERVE(25)만큼을 넘겨주고, 여기서 사용자가 세션 중 제외(×)한
# 공고를 뺀 뒤 관련도 상위 이만큼만 보여준다(pipeline._RECALL_POOL_SIZE와
# 같은 값 - 랭킹 깊이가 아니라 화면 표시 개수).
_S1_DISPLAY_LIMIT = 100


def _fmt_collection_stamp(stamp: str | None) -> str:
    """collection_logs.collected_at(UTC ISO) -> "MM.DD HH:MM"(KST). 실패/없음 -> ''."""
    if not stamp:
        return ""
    try:
        dt = datetime.fromisoformat(str(stamp))
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=9))).strftime("%m.%d %H:%M")


@st.fragment(parallel=True)
def _render_s1_results() -> None:
    """검색창부터 pagination까지(2026-08-14, st.fragment로 분리) - 이전엔
    render_s1() 안에서 pipeline.run_recommendation_mode()를 st.spinner로
    직접 감쌌는데, 그 호출이 1~2분 스크립트를 막고 있는 동안 이전 화면
    (S0, 이력서 업로드)의 카드/버튼 잔상이 안 지워진 채 이 화면 헤더
    밑에 겹쳐 보이는 버그가 실측으로 확인됐다(Streamlit은 스크립트가
    "이번 실행"을 끝까지 마쳐야 이전 실행의 잔여 엘리먼트를 정리한다).
    `st.rerun()`을 반복하는 방식으로는 못 고쳤다 - 근본 원인은
    "메인 스크립트가 안 끝난다"는 것 자체였기 때문이다.

    developing-with-streamlit skill의 성능 문서가 안내하는 실제
    해법을 썼다: `@st.fragment(parallel=True)`로 이 무거운 부분만
    분리하면, 첫 렌더에서도 이 함수는 백그라운드 스레드로 실행되고
    render_s1()의 나머지(헤더/스텝바/뒤로가기)는 이 함수를 기다리지
    않고 즉시 완료된다 - 그래서 메인 스크립트가 빨리 끝나 이전 화면
    잔상이 바로 정리되고, 이 프래그먼트 자리에만 로딩 상태가 표시된다.
    새 판단/새 API 호출 없음 - 순수 렌더링 구조 변경이다."""
    resume_raw = st.session_state.resume_raw
    r_hash = _resume_hash(resume_raw)
    career = st.session_state.user_career_level
    challenge = st.session_state.challenge_option
    cache_key = (r_hash, career, challenge)
    db_key = f"{r_hash}|{career}|{challenge}"
    # 캐시 무효화 키는 "날짜"(KST) 단위 - 같은 날 안에서 정기수집(10:00)과
    # missed-run catch-up이 겹쳐 refresh 완료가 여러 번 찍혀도(실측 확인)
    # 재계산은 하루 1번만 돈다. 화면에 보여줄 정확 시각은 last_collection_stamp()
    # 그대로 따로 쓴다(2026-09-02, 재계산이 하루에 여러 번 도는 버그 수정).
    cache_stamp = recommendation_cache.last_collection_date_kst()

    # 목록 조회 = 일일 수집 갱신과 분리(2026-09-01, 사용자 확정). 무거운
    # run_recommendation_mode(Career Filter + M3/M4 랭킹 4천여 건 + URL별
    # 죽은링크 확인, 1~2분)는 "그날 처음" 한 번만 돈다:
    #   ① 세션 메모리(top_cache) 히트 -> 즉시
    #   ② DB(recommendation_cache) 히트(cache_stamp=오늘 날짜 동일) -> 즉시
    #   ③ 둘 다 미스 -> 재계산 후 DB+세션에 저장
    # 새 수집이 다음 날로 넘어가면 cache_stamp(날짜)가 바뀌어 ②가 자동
    # 무효화되고, 사용자가 [새로고침]을 누르면 clear_all() 로 강제 재계산.
    if cache_key not in st.session_state.top_cache:
        cached = recommendation_cache.load(db_key, cache_stamp)
        if cached is not None:
            st.session_state.top_cache[cache_key] = cached
        else:
            with st.spinner("채용공고를 찾는 중입니다(최초 1회, 1~2분 정도 걸릴 수 있습니다)..."):
                try:
                    result = pipeline.run_recommendation_mode(
                        resume_raw,
                        user_career_level=career,
                        challenge_option=challenge,
                    )
                except Exception as e:
                    print(f"[공고 검색 오류] {e}")
                    st.error("채용공고를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
                    return
                recommendation_cache.save(db_key, cache_stamp, result)
                st.session_state.top_cache[cache_key] = result

    result = st.session_state.top_cache[cache_key]

    # (2026-09-02) 상단 "이력서 한 줄 / 마지막 업데이트 / 안내 caption" 블록은
    # render_s1() 의 아이콘 제목(page_header)+뒤로가기+새로고침으로 이동. 여기선
    # 필터링부터 시작한다.

    # 지원했거나 제외한 공고는 캐시·수집 갱신과 무관하게 화면 직전에
    # 현재 상태로 다시 걸러낸다(bug fix 2026-09-01):
    #  - job_id(url::...) 뿐 아니라 (회사, 정규화 직무명) 키로도 매칭한다.
    #    같은 자리를 가리키는 공고가 다른 URL 로 여러 건 있어서(실측 165개
    #    그룹), job_id 하나만 빼면 _dedupe_same_posting 이 다른 쌍을 남기며
    #    되살아났다. posting_identity 참고.
    #  - pipeline 이 _DISMISS_RESERVE(25) 여유분을 함께 넘겨주므로 빠진
    #    자리는 다음 순위 공고로 즉시 채워 최대 _S1_DISPLAY_LIMIT 개 유지.
    _applied_ids = list_applied_job_ids()
    _dismissed_ids = list_dismissed_job_ids()
    _suppressed_keys = list_applied_posting_keys() | list_dismissed_posting_keys()
    candidates = [
        c for c in result["candidates"]
        if c["job_id"] not in _applied_ids
        and c["job_id"] not in _dismissed_ids
        and _posting_key(c.get("company") or "", c.get("title") or "") not in _suppressed_keys
    ]
    candidates.sort(
        key=lambda c: c.get("combined_score") if c.get("combined_score") is not None else float("-inf"),
        reverse=True,
    )
    candidates = candidates[:_S1_DISPLAY_LIMIT]

    # (2026-09-02) "이미 지원했거나 제외한 공고는 자동으로 빠집니다" 안내는
    # render_s1() page_header 부제로 이동 - 여기선 중복이라 제거.

    # 방금 제외한 공고 되돌리기(2026-08-30) - 확인 모달 없이 즉시 제외하되,
    # 바로 아래에 되돌리기 한 줄을 둔다. 다른 화면으로 이동하면(go) 사라진다.
    _ld = st.session_state.get("_last_dismissed")
    if _ld:
        _uc = st.columns([5, 1])
        _uc[0].caption(f"‘{_ld.get('company','')} · {_ld.get('title','')}’ 공고를 목록에서 제외했습니다.")
        if _uc[1].button("되돌리기", key="undo_dismiss", width="stretch"):
            # 방금 제외한 공고는 아직 세션 캐시(top_cache)의 후보군 안에
            # 그대로 있으므로, dismissed 기록만 지우면 다음 렌더에서 원래
            # 순위로 다시 나타난다 - 목록 재계산 불필요.
            remove_dismissed_job(_ld["job_id"])
            st.session_state.pop("_last_dismissed", None)
            st.rerun()

    # 검색창 + 붙여넣기 버튼 한 줄(2026-08-14, 사용자 확정 - 지역/경력/
    # 추천순 드롭다운과 필터 초기화 버튼을 전부 제거하고 검색창 하나만
    # 남긴다). 공고 후보 자체(candidates)는 pipeline.run_recommendation_mode()가
    # 이미 만든 순서(등록일 기준 최신순) 그대로 쓴다 - 여기서 다시
    # 정렬하지 않는다.
    #
    # 검색창 왼쪽 아이콘 - 원래 placeholder에 "🔍" 이모지를 직접 넣었었는데
    # 이 프로젝트는 아이콘을 항상 assets/icons/*.png(icon_png)로만
    # 쓴다는 규칙이 있다(CLAUDE.md, 2026-07-24) - 이모지는 그 규칙
    # 위반이라는 사용자 지적(2026-08-14)으로 뺐다. st.text_input은
    # 아이콘을 넣을 네이티브 자리가 없어서(icon= 파라미터 자체가 없음),
    # CSS background-image로 입력창 안쪽에 커스텀 PNG를 얹는다 - 이
    # 카드 밖 다른 입력창까지 영향 주지 않도록 key="job-search-row"
    # 스코프로만 좁혔다(사용자가 명시적으로 요청한 경우에만 최소한의
    # CSS를 쓴다는 CLAUDE.md 예외 조건에 해당).
    _search_icon_uri = icon_png_data_uri("search", variant="navy")
    if _search_icon_uri:
        st.markdown(
            f"""
            <style>
            div[class*="st-key-job-search-row"] div[data-testid="stTextInput"] input {{
                padding-left: 40px !important;
                background-image: url("{_search_icon_uri}") !important;
                background-repeat: no-repeat !important;
                background-position: 14px center !important;
                background-size: 18px 18px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    with st.container(key="job-search-row"):
        search_cols = st.columns([4.2, 1.4])
        with search_cols[0]:
            search_query = st.text_input(
                "검색", key="s1_search", placeholder="직무, 키워드, 회사명 검색", label_visibility="collapsed",
            )
        with search_cols[1]:
            if st.button("+ 공고 직접 붙여넣기", key="goto_paste_mode", width="stretch"):
                go("S2")

    shown = candidates
    if search_query.strip():
        q = search_query.strip().lower()
        shown = [
            c for c in shown
            if q in (c.get("title") or "").lower() or q in (c.get("company") or "").lower()
        ]

    if not shown:
        st.markdown(
            empty_state_html("선택한 조건에 맞는 채용공고가 없습니다.", "검색어를 조정해 보세요."),
            unsafe_allow_html=True,
        )
        return

    # 2026-08-14 재확정 - 처음엔 "제일 추천순으로 나와야 되고"는 요구를
    # 드롭다운 없이 고정 정렬로만 반영했는데, 사용자가 준 레퍼런스
    # 이미지에 정렬 드롭다운이 있어서 다시 넣는다. 대신 장식만 있는
    # 가짜 컨트롤은 만들지 않는다 - 두 옵션 다 실제로 다르게 정렬한다.
    # combined_score/posted 둘 다 Candidate Generation이 이미 만들어
    # 넘기는 값 그대로 쓴다 - 새 점수식 없음.
    #
    # 2026-08-30(사용자 확정) - 기본 정렬 라벨을 "추천순" -> "관련도순"으로
    # 바꾼다. 이 정렬 키는 combined_score(M3+M4 의미 유사도 z-score 0.5:0.5
    # + Career penalty)라, 학력/필수자격 충족까지 반영한 "지원 추천도"가
    # 아니라 "이력서-공고 업무/역량 관련도"다. 라벨을 실제 의미에 맞춘다.
    # 정렬 로직/점수식은 그대로.
    count_col, sort_col = st.columns([3, 1.4])
    with count_col:
        st.caption(f"검색 결과 {len(shown)}건")
    with sort_col:
        sort_option = st.selectbox(
            "정렬", ["관련도순", "최신순"], key="s1_sort", label_visibility="collapsed",
        )
    if sort_option == "최신순":
        shown = sorted(shown, key=lambda c: c.get("created_at") or "", reverse=True)
    else:  # 관련도순 (기본)
        shown = sorted(
            shown,
            key=lambda c: c.get("combined_score") if c.get("combined_score") is not None else float("-inf"),
            reverse=True,
        )

    shown_ids = {c["job_id"] for c in shown}
    current = st.session_state.selected_job
    if not current or current.get("job_id") not in shown_ids:
        st.session_state.selected_job = shown[0] if shown else None

    # 페이지네이션(2026-08-14 재작성, 사용자 확정 - developing-with-streamlit
    # 공식 skill 기준으로 native만 사용. 직접 만든 페이지 버튼 대신
    # Streamlit 1.59.1 내장 st.pagination을 그대로 쓴다). 검색 조건이
    # 바뀌면 이전 페이지에 남아있을 경우 조건에 안 맞는 빈 화면처럼 보일
    # 수 있어 자동으로 1페이지로 되돌린다. 공고 후보 자체(candidates)의
    # 생성·정렬·검색 로직(위 shown 계산)은 건드리지 않는다 - 여기서는
    # 이미 계산된 shown 리스트를 몇 건씩 잘라서 보여줄지만 결정한다.
    filter_sig = (search_query.strip(),)
    filter_changed = st.session_state.s1_filter_sig != filter_sig
    st.session_state.s1_filter_sig = filter_sig

    _PAGE_SIZE = 8
    total_pages = max(1, -(-len(shown) // _PAGE_SIZE))

    # st.pagination(key="s1_page")는 자기 default(=1)로 첫 실행 때
    # session_state["s1_page"]를 스스로 채운다 - 위젯이 아직 한 번도
    # 그려지기 전에 우리가 먼저 그 키를 만들어두면(과거엔 _init_state가
    # 미리 1로 시딩했었다) "default와 Session State API가 동시에 값을
    # 정한다"는 Streamlit 경고가 뜬다. 그래서 키가 이미 존재할 때만
    # (=위젯이 최소 한 번 그려진 뒤) 리셋/clamp하고, 실제로 값이 바뀔
    # 때만 쓴다 - 매 rerun마다 같은 값을 다시 쓰기만 해도 같은 경고가
    # 계속 뜬다는 걸 실측으로 확인했다(save/analyze 클릭 후 재실행에서
    # 매번 재현됨).
    if "s1_page" in st.session_state:
        if filter_changed and st.session_state.s1_page != 1:
            st.session_state.s1_page = 1
        elif st.session_state.s1_page > total_pages:
            st.session_state.s1_page = total_pages
        elif st.session_state.s1_page < 1:
            st.session_state.s1_page = 1

    # st.pagination의 값(선택된 페이지)은 카드보다 먼저 알아야 하지만,
    # 화면 순서는 "카드 목록 -> 페이지네이션"이어야 한다(사용자 명세).
    # st.empty()로 카드 자리만 먼저 잡아두고, 아래에서 pagination 위젯을
    # 그린 뒤 그 값으로 카드를 채운다(레이아웃 문서의 "컨트롤 값이
    # 먼저 필요하지만 화면상 아래에 나와야 하는 경우" 패턴 그대로).
    cards_slot = st.empty()

    with st.container(horizontal_alignment="center"):
        current_page = st.pagination(total_pages, key="s1_page")

    page_start = (current_page - 1) * _PAGE_SIZE
    visible = shown[page_start:page_start + _PAGE_SIZE]

    # 2026-08-16(사용자 확정) - 카드 리스트는 CCv2 컴포넌트(job_card_
    # component.py)로 그린다. Streamlit이 만드는 wrapper DOM에 CSS
    # 셀렉터로 덮어쓰는 방식(예: "원문보기+별 버튼 폭 = 분석하기 버튼
    # 폭"을 맞추려고 fit-content 래퍼를 역으로 추측하던 것)을 더 쌓지
    # 않는다 - 컴포넌트 내부 CSS Grid/Flex가 그 레이아웃을 직접 정의
    # 하므로 추측할 DOM 자체가 없다. 데이터는 이미 있는 build_job_card_
    # view()/resolve_company_logo()/is_saved()만 모아서 넘긴다 - 새
    # 판단 없음. 검색/정렬/페이지네이션/저장/분석 전환 로직은 그대로
    # Python(위 코드)에 남아있다.
    with cards_slot.container():
        cards_payload = []
        for c in visible:
            view = presentation_layer.build_job_card_view(c)
            meta = view["meta"]
            deadline = meta["deadline"]
            cards_payload.append({
                "jobId": view["job_id"],
                "company": view["company"],
                "title": view["title"],
                "deadlineBadge": deadline["badge"],
                "deadlineTone": deadline["tone"],
                "meta": (
                    f"{meta['career']} · {meta['location']} · "
                    f"등록 {(meta['posted'] or '').replace('-', '.')} · "
                    f"마감 {deadline['date'] or deadline['text']}"
                ),
                "skills": view["skill_summary"],
                "skillOverflow": view["skill_overflow"],
                "url": view["url"],
                "logo": resolve_company_logo(view["company"]),
                "saved": is_saved(view["job_id"]),
            })
        result = job_list_view(cards_payload, key="job-card-list")
        if result is not None and getattr(result, "action", None):
            act = result.action
            job_id = act.get("jobId")
            c = next((j for j in visible if j["job_id"] == job_id), None)
            if c:
                if act.get("type") == "save":
                    if is_saved(job_id):
                        remove_saved_job(job_id)
                    else:
                        add_saved_job(c)
                    st.rerun()
                elif act.get("type") == "dismiss":
                    # 확인 모달 없이 즉시 제외. DB의 공고 자체(jobs/
                    # candidate_jobs)는 건드리지 않고 dismissed_jobs 테이블에만
                    # 기록한다. 목록 재계산 없이(캐시 유지) 다음 렌더에서
                    # 빠지고, pipeline이 넘긴 여유분에서 다음 순위가 채워진다.
                    add_dismissed_job(c)
                    st.session_state["_last_dismissed"] = {
                        "job_id": job_id, "company": c.get("company"), "title": c.get("title"),
                    }
                    st.rerun()
                elif act.get("type") == "analyze":
                    # 추천 목록은 DB 캐시(recommendation_cache)에서 올 수
                    # 있어 최대 하루 지난 dict 일 수 있다 - 분석 진입 시엔
                    # job_id 로 최신 행을 다시 읽어 넘긴다(대시보드 "다시
                    # 보기"와 동일 방식). 못 찾으면 캐시 dict 그대로.
                    st.session_state.selected_job = get_candidate_job(c["job_id"]) or c
                    go("S1D")


def render_s1() -> None:
    """공고 찾기 = 구 홈 화면을 그대로 옮겨온 화면(2026-08-31, 사용자 확정 -
    홈은 지원현황 대시보드로 분리). 두 단계:
    ① s1_browsing=False: 구 render_s0 본문 그대로 - render_s0_hero /
       render_resume_upload_card / render_resume_status_card /
       render_resume_summary_cards + "어떤 방식으로 진행할까요?" 선택 카드
       (render_s0_action_cards). 이력서 업로드만으로 목록을 자동 로딩하지
       않는다 - 사용자가 "채용공고 탐색하기"를 눌러야 한다.
    ② s1_browsing=True: 공고 목록(_render_s1_results, st.fragment)만.
       위에 이력서 한 줄 요약 + "이력서 화면" 되돌아가기.
    새 이력서 업로드 시 s1_browsing 이 False 로 리셋된다(업로드 핸들러)."""
    st.markdown(_STEP1_CSS, unsafe_allow_html=True)

    name = st.session_state.user_display_name or "게스트"
    has_resume = bool(st.session_state.resume_raw)
    show_uploader = st.session_state.show_uploader or not has_resume
    resume_ready = has_resume and not show_uploader

    # 진행 단계 바(맨 위) - 이력서 없으면 1단계, 목록 탐색 중이면 2단계.
    st.markdown(
        step_progress_html(1 if (resume_ready and st.session_state.get("s1_browsing")) else 0),
        unsafe_allow_html=True,
    )

    # ── ② 목록 화면(사용자가 "채용공고 탐색하기"를 이미 선택) ──
    if resume_ready and st.session_state.get("s1_browsing"):
        # 왼쪽 뒤로가기 + 다른 화면(S7/S9/S11)과 동일한 아이콘 제목 형식.
        if st.button("← 뒤로", key="s1_back_to_resume"):
            st.session_state.s1_browsing = False
            st.rerun()
        _h1, _h2 = st.columns([4, 1], vertical_alignment="center")
        with _h1:
            page_header(
                svg_icon("search", 40, "var(--navy)"), "공고 찾기",
                "이력서를 바탕으로 추천된 공고입니다. 지원했거나 제외한 공고는 목록에서 자동으로 빠집니다.",
            )
        with _h2:
            if st.button("새로고침", key="s1_refresh", use_container_width=True):
                recommendation_cache.clear_all()
                st.session_state.top_cache = {}
                st.rerun()
        _render_s1_results()
        return

    # ── ① 이력서 확인 + 방식 선택 (구 홈 본문 그대로) ──
    # 2026-09-03(사용자 확정) - "안녕하세요, 게스트님" 인사말 제거(단계 바 +
    # "이력서 업로드" 라벨이면 충분, 중복).
    st.space(20)
    with st.container(key="main-shell", gap=None):
        st.markdown('<div class="section-label">이력서 업로드</div><div style="height:20px"></div>', unsafe_allow_html=True)
        render_resume_upload_card(show_uploader)
        st.space(20)
        if resume_ready:
            level = st.session_state.user_career_level
            upload_date = (st.session_state.resume_uploaded_at or "").split(" ")[0]
            render_resume_status_card(level, upload_date)

    if not resume_ready:
        return

    # "내 이력서 한눈에 보기"(핵심 역량/주요 기술/대표 경험 카드) 제거 - 2026-09-04 사용자 요청.
    render_s0_action_cards(True)


# ── S1-D. 공고 분석 + 매칭 결과(별도 화면 - 사용자 명시적 요청, ─────────
# 2026-07-15: "화살표 누르면 같은 페이지 아래가 아니라 다른 화면으로
# 넘어가서 분석결과가 나오게 해야 되는거야, 뒤로가기 버튼도 넣고" -
# 앞서 한 페이지로 합쳤던 걸 되돌린다. 목록 자체는 여전히 S1 하나에서
# 검색+필터를 담당한다).

def render_s1d() -> None:
    st.markdown(_ANALYSIS_CSS, unsafe_allow_html=True)
    job = st.session_state.selected_job
    if job is None:
        st.warning("선택된 공고가 없습니다.")
        back_button()
        return
    st.markdown(step_progress_html(2), unsafe_allow_html=True)  # 3단계: 공고 분석
    # 2026-08-14 - 아이콘/부제를 LIST(공고 찾기) 화면과 같은 스타일로
    # 맞췄다(icon_png navy + subtitle) - 사용자 확정 "LIST와 동일한
    # visual system".
    page_header(
        svg_icon("clipboard-list", 40, "var(--navy)"), "공고 분석",
        subtitle="선택한 공고의 요구사항과 내 이력서를 연결해 분석한 결과입니다.",
    )
    back_button()
    # 2026-07-23(서비스 구조 변경) - 목록(S1)에서는 더 이상 _semantic_
    # match를 미리 계산해두지 않는다(Ranking 제거). ②③ 섹션이 여전히
    # 이 값을 쓰므로, 사용자가 실제로 이 공고를 선택한 지금 1건에
    # 대해서만 계산한다(이미 있으면 다시 계산 안 함 - pipeline.
    # ensure_job_semantic_match 참고).
    pipeline.ensure_job_semantic_match(st.session_state.resume_raw, job)
    # T0(추천모드) - 사용자가 공고를 선택해 이 분석 화면에 진입하고
    # 매칭/판단이 실제 계산되는 지점. _log_prep_event_once 가드로 rerun
    # 중복은 막힌다(browser session 당 1회).
    _log_prep_event_once(str(job.get("job_id")), "analysis_opened")
    # 2026-08-15(사용자 확정 - "Component v2 구현 → Code to Canvas →
    # 다시 코드" 흐름) - jd_semantic_objects가 있는 job은 새 화면
    # (_render_job_analysis_v4, CCv2 컴포넌트 기반)을 쓴다. 없는 job(옛
    # 데이터/JD 붙여넣기 폴백)은 기존 화면(_render_job_analysis_panel)
    # 그대로 - 이번 작업 대상이 아니라 손대지 않는다.
    # 2026-08-31(경량 분석, 사용자 확정) - quick_analysis(LLM 1회) 결과가
    # 붙은 job 은 짧은 판단 화면(_render_quick_analysis)을 쓴다. 기존 상세
    # 리포트 UI(_render_job_analysis_v4 / _render_job_analysis_panel)는
    # 메인 경로에서 빠지고, 구버전 데이터 폴백으로만 남는다.
    if job.get("_quick_analysis") is not None:
        _render_quick_analysis(job)
    elif presentation_layer.has_semantic_data(job):
        _render_job_analysis_v4(job)
    else:
        _render_job_analysis_panel(job)
    _render_judgment_controls(job)  # T1(보류/미지원) 명시적 액션


# ── Analysis Evidence Contract(2026-08-14, 신규) ───────────────────────
# analysis_engine.py의 5-state(A~E) 결과를 S1D/S3 공통으로 보여준다.
# 새 판단/점수를 만들지 않는다 - link_engine.py/judge_engine.py가 이미
# 계산한 값을 사람이 읽을 수 있는 화면 단위로만 옮긴다. A/B/C/D/E,
# relation_reason_type, matched_dimensions, object_id 같은 개발용
# 필드명은 메인 화면에 노출하지 않는다("근거 원문 보기"에서만 원문 인용).

_EVIDENCE_STATE_TONE = {"A": "green", "B": "blue", "C": "orange", "D": "gray", "E": "red"}
_EVIDENCE_STATE_SHORT = {
    "A": "높은 적합", "B": "전이 가능한 경험", "C": "근거 보강 가능",
    "D": "확인 필요", "E": "실제 공백",
}


def _get_resume_facts() -> dict:
    """judge_engine.judge()의 Hard Eligibility 판정에 넘길 학력/자격증/
    경력 factual value(resume_facts.py, LLM 없음). 순수 정규식/휴리스틱
    이라 매번 다시 계산해도 비용이 거의 없다 - session_state에 캐시하지
    않는다(이력서가 바뀌면 자동으로 최신 값을 쓰게 하기 위함)."""
    resume_raw = st.session_state.resume_raw
    career_result = extract_user_career_level(resume_raw, DATA_ANALYTICS)
    return extract_resume_facts(resume_raw, career_result)


def _log_prep_event_once(job_id: str | None, event_name: str, reason: str | None = None) -> None:
    """preparation_tracker 이벤트 - render_s4/render_s6처럼 화면이
    재실행(rerun)될 때마다 다시 그려지는 함수에서 부르면, 이 job_id에
    이 이벤트를 이번 세션에서 이미 기록했는지 session_state로 확인해서
    중복 기록을 막는다(Streamlit은 위젯 조작마다 스크립트 전체를 다시
    실행하므로 가드가 없으면 화면을 볼 때마다 새 행이 쌓인다).

    reason: judgment_hold/judgment_skip 의 최소 사유(선택). 그 외 이벤트는 무시된다."""
    if not job_id:
        return
    seen = st.session_state.setdefault("_prep_events_logged", set())
    key = (job_id, event_name)
    if key in seen:
        return
    seen.add(key)
    try:
        preparation_tracker.log_event(job_id, event_name, reason=reason)
    except Exception:
        pass  # 계측 실패가 실제 화면 기능을 막으면 안 됨


# 보류/미지원 제외 사유 - 고정 선택지(자유 장문 입력 안 받음). 선택은 optional.
_JUDGMENT_REASONS = ["자격조건", "직무불일치", "성장방향", "지원비용", "조건", "기타"]


def _render_judgment_controls(job: dict | None) -> None:
    """공고 분석 화면(S1D / S3) 하단 - 사용자가 이 공고를 "보류" 또는
    "지원 안 함"으로 명시적으로 확정하는 액션(T1). "지원(진행)"은 기존
    "지원 준비하기"/"다음 단계 진행" 버튼이 담당하므로 여기 없다.

    사유는 빠른 단일 선택(강제 아님). 클릭 시 preparation_tracker 에
    judgment_hold/judgment_skip 이벤트 1회 기록 + 세션 judgment 확정.
    측정 telemetry는 이 화면에 노출하지 않는다(개발자 전용)."""
    if not job or not job.get("job_id"):
        return
    job_id = str(job.get("job_id"))
    done = st.session_state.setdefault("_judgment_done", set())
    if job_id in done:
        st.caption("이 공고는 '보류 / 지원 안 함'으로 기록되었습니다.")
        return
    st.divider()
    reason = st.pills(
        "이 공고를 제외한다면 사유 (선택)", _JUDGMENT_REASONS,
        selection_mode="single", key=f"judg_reason_{job_id}",
    )
    hc, sc = st.columns(2)
    if hc.button("나중에 (보류)", key=f"judg_hold_{job_id}", use_container_width=True):
        _log_prep_event_once(job_id, "judgment_hold", reason=reason)
        done.add(job_id)
        st.rerun()
    if sc.button("지원 안 함", key=f"judg_skip_{job_id}", use_container_width=True):
        _log_prep_event_once(job_id, "judgment_skip", reason=reason)
        done.add(job_id)
        st.rerun()


def _get_analysis(job: dict) -> dict | None:
    """job["_semantic_match"](Semantic Linking 결과, ensure_job_semantic_
    match()가 이미 계산해 job dict에 붙여둠)가 있을 때만 analysis_engine.
    build_analysis()로 Evidence Contract 뷰를 만든다. judge_engine.judge()는
    Rule(LLM 호출 없음) - 여기서 새 LLM을 부르지 않는다. job_id 단위로
    session_state에 캐시해서 confirm_genuine_gap()으로 사용자가 "경험
    없음"을 확인한 상태가 재실행(rerun) 사이에도 유지되게 한다."""
    link_result = job.get("_semantic_match")
    if not link_result or not link_result.get("links"):
        return None
    job_id = str(job.get("job_id"))
    cache = st.session_state.analysis_by_job
    if job_id not in cache:
        # 2026-08-31(경량 분석) - quick_analysis 경로는 jd_semantic_objects
        # 컬럼을 안 만든다. Hard Eligibility 정규식이 훑을 최소 JD object를
        # job["_qa_jd_objects"]로 이미 갖고 있으므로 그걸 우선 쓴다(구경로도 지원).
        jd_semantic_objects = job.get("_qa_jd_objects") or json.loads(job.get("jd_semantic_objects") or "[]")
        jr = judge_engine.judge(link_result, jd_semantic_objects, _get_resume_facts())
        cache[job_id] = analysis_engine.build_analysis(link_result, jr)
    return cache[job_id]


def _confirm_no_experience(job: dict, jd_object_id: str) -> None:
    """"경험 없음" 확인 버튼 - 이 함수를 거쳐야만 D(Unverified)가
    E(Genuine Gap)로 바뀐다. 자동 승격 경로는 없다."""
    job_id = str(job.get("job_id"))
    current = st.session_state.analysis_by_job.get(job_id)
    if current is None:
        return
    st.session_state.analysis_by_job[job_id] = analysis_engine.confirm_genuine_gap(current, jd_object_id)


def _render_analysis_evidence_section(job: dict) -> None:
    analysis = _get_analysis(job)
    if analysis is None:
        return  # Semantic Link 결과가 없는 공고(구버전/붙여넣기 폴백) - 기존 화면만 유지

    st.markdown("### ⑤ 최종 판단")
    decision = analysis["decision"]
    with st.container(border=True):
        st.markdown(f"### {badge_html(decision, DECISION_TONE.get(decision, 'gray'))}", unsafe_allow_html=True)
        reasons = analysis["decision_reason"] or []
        if reasons:
            st.write(reasons[0])
            if len(reasons) > 1:
                with st.expander(f"판단 이유 더 보기 ({len(reasons)}건)"):
                    for r in reasons[1:]:
                        st.write(f"- {r}")

        counts = analysis["counts"]
        strong_items = analysis["by_state"]["A"]
        risk_items = [it for it in analysis["by_state"]["D"] if it["importance"] in ("critical", "core")]

        st.space(10)
        col_strength, col_risk = st.columns(2)
        with col_strength:
            st.markdown("**핵심 강점**")
            if strong_items:
                top = strong_items[0]
                st.write(f"- {top['jd_requirement']}")
                ev_list = top["matched_resume_objects"] or []
                if ev_list and ev_list[0].get("evidence_text"):
                    st.caption("대표 근거: " + ev_list[0]["evidence_text"][:80])
            else:
                st.caption("직접 대응하는 근거가 아직 없습니다.")
        with col_risk:
            st.markdown("**가장 큰 리스크**")
            if risk_items:
                st.write(f"- {risk_items[0]['jd_requirement']} (확인 필요)")
            elif analysis["by_state"]["C"]:
                st.write(f"- {analysis['by_state']['C'][0]['jd_requirement']} (근거 보강 필요)")
            else:
                st.caption("두드러진 리스크가 확인되지 않았습니다.")

        st.space(10)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("강하게 맞는 부분", counts["A"])
        m2.metric("전이 가능한 부분", counts["B"])
        m3.metric("근거 보강 가능", counts["C"])
        m4.metric("확인 필요", counts["D"])

    with st.expander("요구사항별 상세 근거 보기"):
        for it in analysis["items"]:
            tone = _EVIDENCE_STATE_TONE.get(it["state"], "gray")
            with st.container(border=True):
                st.markdown(
                    f"{badge_html(_EVIDENCE_STATE_SHORT.get(it['state'], it['state']), tone)} &nbsp; **{it['jd_requirement']}**",
                    unsafe_allow_html=True,
                )
                st.caption(it["why"])
                matched = it["matched_resume_objects"]
                if matched:
                    projects = sorted({m.get("source_project", "") for m in matched if m.get("source_project")})
                    if projects:
                        st.write("내 경험: " + " / ".join(projects))
                    with st.expander("근거 원문 보기", key=f"ev_{job.get('job_id')}_{it['jd_object_id']}"):
                        for m in matched:
                            st.caption(f"[{m.get('source_project','')}] {m.get('evidence_text','')}")
                elif it["state"] == "D":
                    st.write("이력서에서 관련 근거를 찾지 못했습니다 - 실제로 경험이 없다는 뜻은 아닙니다.")
                    if st.button("이 항목은 실제로 경험이 없습니다", key=f"confirm_gap_{job.get('job_id')}_{it['jd_object_id']}"):
                        _confirm_no_experience(job, it["jd_object_id"])
                        st.rerun()
                elif it["state"] == "E":
                    st.write("사용자가 실제 경험 없음으로 확인한 항목입니다.")

    st.divider()


# ── STEP3 "공고 분석" v4(2026-08-15, 사용자 확정 - "Component v2
# 구현 → Code to Canvas → 다시 코드" 흐름) - native st.columns/CSS
# 조합(v2/v3) 대신 CCv2 컴포넌트(resume_input/analysis_component.py)
# 하나로 판단배너~CTA 전체를 그린다. jd_semantic_objects가 없는 job
# (옛 데이터/JD 붙여넣기 폴백)은 이 함수를 쓰지 않고 기존
# _render_job_analysis_panel로 폴백한다(render_s1d 참고) - 그 경로는
# 이번 작업 대상이 아니라 손대지 않았다.

# CCv2 컴포넌트는 Shadow DOM 안에서 독립 렌더링되므로 Streamlit
# _DESIGN_TOKENS_CSS의 var(--mint-soft) 같은 전역 CSS 변수를 못 읽는다
# - analysis_component.py의 CSS가 쓰는 배지 색과 동일한 hex 값을
# 그대로 맞춰서 넘긴다(새 색 아님, 기존 badge_html 팔레트와 동일).
_DECISION_BANNER_BG = {"지원": "#F2FFF6", "보류": "#FFFBEA", "비추천": "#FBE3E5"}
_DECISION_BANNER_FG = {"지원": "#1F8F5E", "보류": "#B9702E", "비추천": "#E0555F"}


# ── 경량 분석 화면(2026-08-31, 사용자 확정) ───────────────────────────
# "분석하기"는 더 이상 긴 리포트가 아니다. [회사·직무] / 지원 판단 /
# 직무 핵심(1~2문장) / 연결 근거(최대 3) / 걸리는 점(최대 3) / 최종 1줄.
# 같은 내용을 다른 섹션에서 반복하지 않는다. 판단(지원/보류/비추천)은
# 여전히 Rule(judge_engine, LLM 0회)이 내린다 - 화면은 그 결과를 짧게
# 설명만 한다.

_QUICK_GAP_LABEL = {
    "domain_gap": "업무 도메인·맥락이 다름",
    "skill_gap": "요구 기술 경험이 확인되지 않음",
    "experience_gap": "관련 직접 경험이 부족",
    "responsibility_gap": "책임/권한 범위가 다름",
    "qualification_gap": "자격요건 보완 필요",
}
# 걸리는 점 정렬 순위(작을수록 먼저). 우대사항은 항상 뒤로.
_GAP_TYPE_RANK = {"domain_gap": 0, "responsibility_gap": 0, "experience_gap": 1,
                  "skill_gap": 2, "qualification_gap": 3, None: 4}


def _quick_final_line(decision: str, analysis: dict | None) -> str:
    """상단 카드의 '가장 중요한 이유 한 줄'. 걸리는 점 목록과 같은 문장을
    그대로 반복하지 않는다 - 여기선 요약, 걸리는 점에선 항목별 근거."""
    a = analysis or {}
    blocking = a.get("blocking_requirements") or []
    struct = [
        it for it in a.get("items") or []
        if it.get("relation") == "no_match"
        and it.get("relation_reason_type") in ("domain_gap", "responsibility_gap", "experience_gap")
        and not it.get("hard_eligibility_status")
        and "우대" not in (it.get("jd_requirement") or "")
    ]
    if decision == "비추천":
        if blocking:
            base = f"{blocking[0].get('jd_requirement', '필수 자격요건')} 필수요건을 충족하지 못합니다"
            return base + (". 직무 핵심 경험도 이력서와 방향 차이가 있습니다." if struct else ".")
        return "핵심 업무에 필요한 직접 경험이 부족해 지원을 권하지 않습니다."
    if decision == "보류":
        return "관련 경험은 있으나 확인·보완이 필요한 부분이 있습니다."
    return "핵심 요구사항에 대응하는 경험이 이력서에서 확인됩니다."


# eligibility_compare.compare() 가 만드는 내부 reason 문구 → 사용자 문구.
# 이 값들은 개발/디버깅용이라 그대로 화면에 노출하면 안 된다(2026-09-02).
# 시스템이 실제로 아는 범위(= "확인이 필요하다")까지만 표현한다.
_ELIG_REASON_UI = [
    ("이력서에서 학력 정보를 찾지 못했습니다", "이력서에서 관련 학력 확인 필요"),
    ("이력서에서 경력 정보를 찾지 못했습니다", "이력서에서 관련 경력 확인 필요"),
    ("이력서에서 자격증 정보를 찾지 못했습니다", "이력서에서 자격증 정보 확인 필요"),
    ("이 유형은 아직 factual 추출기가 없습니다", "조건 충족 여부 확인 필요"),
    ("복합(또는) 조건이라 안전하게 판정할 수 없습니다", "복합 조건으로 추가 확인 필요"),
]


def _elig_reason_to_pair(s: str) -> tuple[str, str]:
    """judge decision_reason 한 줄('지원 자격 조건 미충족: X (Y).' / '필수 조건
    확인 필요: X (Y).')을 (문제, 이유)로 분해한다. 요구사항 본문에도 괄호가
    있을 수 있어(예: '고객경험(CX) 유관 경력') 문자열을 함부로 쪼개지 않고,
    맨 뒤의 ' (...)' 만 이유로 떼어낸 뒤 알려진 내부 문구를 사용자 문구로
    치환한다."""
    body = (s or "").strip().rstrip(".")
    for pfx in ("지원 자격 조건 미충족: ", "필수 조건 확인 필요: "):
        if body.startswith(pfx):
            body = body[len(pfx):]
            break
    body = body.strip()
    prob, why = body, ""
    m = re.match(r"^(.*\S)\s+\((.*)\)$", body)  # 공백 뒤 여는 괄호 = 이유 시작
    if m:
        prob, why = m.group(1).strip(), m.group(2).strip()
    for dev, ui in _ELIG_REASON_UI:
        if dev in why:
            return prob, ui
    if why.startswith("요구=") or "보유=" in why:  # 실제 사실 비교값은 유지
        return prob, f"요구 조건과 이력서 정보 불일치 ({why})"
    return prob, (why or "지원 전 확인 필요")


def _quick_gap_lines(qa: dict, analysis: dict | None) -> list[tuple[str, str]]:
    """걸리는 점 (문제, 이유) 목록. 중요도 순:
    ① Hard Eligibility(미충족/확인불가) ② 핵심 업무 no_match ③ 핵심 업무 partial
    ④ skill_gap ⑤ 우대사항. Hard Eligibility 로 이미 잡힌 requirement 는
    quick_analysis requirements 에서 다시 넣지 않는다(중복 제거)."""
    a = analysis or {}
    pairs: list[tuple[str, str]] = []

    # ① Hard Eligibility - judge decision_reason 이 이미 정확한 목록
    #    (비추천이면 blocking 만, 보류면 확인불가만 담겨 있다)
    for r in a.get("decision_reason") or []:
        if r.startswith("지원 자격 조건 미충족: ") or r.startswith("필수 조건 확인 필요: "):
            pairs.append(_elig_reason_to_pair(r))

    # Hard Eligibility 로 처리된 jd_object_id (충족/미충족/확인불가 전부) -
    #   아래 루프에서 건너뛴다.
    hard_ids = {
        it.get("jd_object_id") for it in a.get("items") or []
        if it.get("hard_eligibility_status")
    }

    # ②~⑤ 구조적 gap
    cands = []
    for i, r in enumerate(qa.get("requirements") or []):
        if f"req{i}" in hard_ids:
            continue
        rel, gap = r.get("relation"), r.get("gap_type")
        req = (r.get("requirement") or "").strip()
        if not req:
            continue
        if rel == "no_match":
            why = _QUICK_GAP_LABEL.get(gap, "관련 근거가 이력서에서 확인되지 않음")
        elif rel == "partial" and gap:
            why = _QUICK_GAP_LABEL.get(gap, "일부만 대응")
        else:
            continue
        # tier: ② 핵심 no_match  ③ 핵심 partial  ④ skill_gap  ⑤ 우대
        if "우대" in req:
            tier = 5
        elif gap == "skill_gap":
            tier = 4
        elif rel == "no_match":
            tier = 2
        else:
            tier = 3
        cands.append((tier, _GAP_TYPE_RANK.get(gap, 4), i, req, why))
    cands.sort()
    pairs.extend((req, why) for *_ , req, why in cands)

    seen, out = set(), []
    for prob, why in pairs:
        key = re.sub(r"\s+", "", prob)
        if key and key not in seen:
            seen.add(key)
            out.append((prob, why))
    return out


def _short_evidence(ev: str) -> str:
    """연결 근거의 한 줄 요약 - 여러 프로젝트/문장이 이어졌으면 앞부분만."""
    s = str(ev or "").strip()
    for sep in (" / ", " · "):
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    if len(s) > 110 and ". " in s:
        s = s.split(". ")[0].strip()
    return s[:110] + ("…" if len(s) > 110 else "")


def _short_evidence_full(ev: str) -> str:
    """_short_evidence 와 같되 길이 컷("…") 없음 - 여러 근거가 " / "·" · "로
    이어졌으면 첫 조각만 쓰고, 나머지는 그대로 둔다(화면에서 줄바꿈됨)."""
    s = str(ev or "").strip()
    for sep in (" / ", " · "):
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    return s


def _one_line(s: str) -> str:
    """공백 정리 + 뒤 "[우대]" 꼬리표만 제거. **자르지 않는다** - 요약 목록에서
    "…" 중간 잘림을 전면 금지(2026-09-03 사용자 확정). 긴 문장은 화면에서
    자연스럽게 줄바꿈되고, 더 자세한 원문은 "상세 근거 보기"에 그대로 있다."""
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return re.sub(r"\s*\[(우대|우대사항)\]\s*$", "", s)


_QA_CSS = """
<style>
/* 직무 핵심 - 카드 상단 흰 영역, 아래에만 회색 경계선 (full-bleed) */
.qa-jobcore {
    margin:-1.15rem -1.5rem 16px;
    padding:14px 24px;
    border-bottom:1px solid #E2E8F0;
}
.qa-jobcore-t { font-size:15px; font-weight:700; color:#0F172A; margin-bottom:4px; }
.qa-jobcore-b { color:#334155; font-size:14px; line-height:1.6; }

/* 나와의 적합성 2-col: 각 컬럼을 흰 카드로(가운데 세로선 대신) */
div[class*="st-key-qa-fit"] [data-testid="stHorizontalBlock"] { gap: 24px; }
div[class*="st-key-qa-fit"] [data-testid="stColumn"] {
    border: 1px solid #E5E9F0;
    border-radius: 12px;
    padding: 20px 22px;
    background: #FFFFFF;
}

/* 컬럼 헤더: 아이콘 배지 + 라벨 + 컬러 밑줄(카드 폭 전체) */
.qa-sub { display:flex; align-items:center; gap:8px; font-size:14px; font-weight:700; padding-bottom:10px; }
.qa-sub-ok { color:#1B8A5A; border-bottom:2px solid #1B8A5A; }
.qa-sub-warn { color:#DD8A2C; border-bottom:2px solid #ECA24E; }
.qa-ic { width:20px; height:20px; display:inline-flex; align-items:center; justify-content:center;
         color:#fff; font-size:12px; font-weight:800; flex:none; line-height:1;
         background-size:contain; background-repeat:no-repeat; background-position:center; }
.qa-ic-ok { background:#1B8A5A; border-radius:50%; }
.qa-ic-warn {
    color:transparent;
    background-image:url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%3E%3Cpath%20d='M12%202.5L22.5%2021.5L1.5%2021.5Z'%20fill='%23DD8A2C'/%3E%3Crect%20x='11'%20y='9'%20width='2'%20height='6'%20rx='1'%20fill='white'/%3E%3Crect%20x='11'%20y='16.6'%20width='2'%20height='2'%20rx='1'%20fill='white'/%3E%3C/svg%3E");
}

/* 항목: 굵은 제목 + 회색 설명, 사이는 점선 구분 */
.qa-col-body { padding-top:14px; }
.qa-ititle { font-size:14.5px; font-weight:700; color:#1E293B; line-height:1.5; }
.qa-idesc { font-size:13px; color:#64748B; line-height:1.55; margin-top:4px; }
.qa-dash { border-top:1px dashed #DBE1EA; margin:14px 0; }
.qa-empty { font-size:13px; color:#94A3B8; padding-top:14px; }
</style>
"""


def _render_quick_analysis(job: dict) -> None:
    st.markdown(_QA_CSS, unsafe_allow_html=True)
    qa = job["_quick_analysis"]
    analysis = _get_analysis(job)  # Rule 판단(judge_engine, LLM 0회)
    decision = (analysis or {}).get("decision") or "보류"

    company = job.get("company") or ""
    title = job.get("title") or ""
    career = (job.get("career_level") or "").strip()

    # ── 1) 공고 + 최종 판단 카드 (판단 설명을 배지·연차 옆 한 줄로) ──
    with st.container(border=True):
        st.markdown(
            "<div style='font-size:17px;font-weight:700;color:#0F172A;line-height:1.45;'>"
            f"{html.escape(company)}{(' · ' + html.escape(title)) if title else ''}</div>",
            unsafe_allow_html=True,
        )
        st.space(6)
        badge = badge_html(_DECISION_LABEL.get(decision, decision), DECISION_TONE.get(decision, "gray"))
        _career_span = (
            f"&nbsp;&nbsp;<span style='color:#64748B;font-size:13px;'>{html.escape(career)}</span>" if career else ""
        )
        _final = html.escape(_quick_final_line(decision, analysis))
        st.markdown(
            f"<div style='line-height:1.9;'>{badge}{_career_span}"
            f"&nbsp;&nbsp;<span style='color:#CBD5E1;'>|</span>&nbsp;&nbsp;"
            f"<span style='color:#334155;font-size:13.5px;line-height:1.6;'>{_final}</span></div>",
            unsafe_allow_html=True,
        )

    st.space(16)

    # ── 데이터 준비 (연결 근거 / 걸리는 점) ──
    _hard_ids = {
        it.get("jd_object_id") for it in (analysis or {}).get("items") or []
        if it.get("hard_eligibility_status")
    }
    gaps = _quick_gap_lines(qa, analysis)
    _shown_gap_reqs = {re.sub(r"\s+", "", p) for p, _ in gaps[:3]}

    # 연결 근거 = 직접 대응(match) 우선. match 가 2개 미만이면, 화면에 gap 으로
    # 이미 보이지 않는 전이 가능(partial) requirement 로 채운다 - 같은
    # requirement 를 양쪽 컬럼에 중복 표시하지 않는다. Hard Eligibility 조건은 제외.
    _match_ev, _partial_ev = [], []
    for i, r in enumerate(qa.get("requirements") or []):
        if f"req{i}" in _hard_ids or not r.get("resume_evidence"):
            continue
        if re.sub(r"\s+", "", r.get("requirement") or "") in _shown_gap_reqs:
            continue
        if r.get("relation") == "match":
            _match_ev.append(r)
        elif r.get("relation") == "partial":
            _partial_ev.append(r)
    _match_ev.sort(key=lambda r: 1 if "우대" in (r.get("requirement") or "") else 0)
    _partial_ev.sort(key=lambda r: 1 if "우대" in (r.get("requirement") or "") else 0)
    evidence = (_match_ev + _partial_ev)[:3]

    def _qa_col_body(items: list[tuple[str, str]], empty_msg: str) -> str:
        """항목 (제목, 설명) 리스트를 점선 구분 HTML 로. 중간 잘림 없음."""
        if not items:
            return f"<div class='qa-empty'>{html.escape(empty_msg)}</div>"
        rows = []
        for i, (t, d) in enumerate(items):
            sep = "<div class='qa-dash'></div>" if i else ""
            desc = f"<div class='qa-idesc'>{html.escape(d)}</div>" if d else ""
            rows.append(f"{sep}<div class='qa-ititle'>{html.escape(t)}</div>{desc}")
        return f"<div class='qa-col-body'>{''.join(rows)}</div>"

    with st.container(border=True):
        # ── 직무 핵심 (job_core 그대로. 과도하게 길면 3줄까지만 보이고 펼침) ──
        jc = (qa.get("job_core") or "").strip()
        if jc:
            _jc_key = f"_jc_open_{job.get('job_id')}"
            _long = len(jc) > 170
            _open = st.session_state.get(_jc_key, False)
            _clamp = "" if (_open or not _long) else (
                "display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;")
            st.markdown(
                "<div class='qa-jobcore'><div class='qa-jobcore-t'>직무 핵심</div>"
                f"<div class='qa-jobcore-b' style='{_clamp}'>{html.escape(jc)}</div></div>",
                unsafe_allow_html=True,
            )
            if _long and st.button("전체 보기" if not _open else "접기", key=f"jc_toggle_{job.get('job_id')}"):
                st.session_state[_jc_key] = not _open
                st.rerun()
            st.space(10)

        # ── 나와의 적합성 (요약: 좌 연결 / 우 확인필요, 각 최대 3) ──
        # "…" 중간 잘림 금지: 짧은 라벨을 굵게, 원문은 그대로 줄바꿈. 상세는 아래 expander.
        st.markdown("<div style='font-size:15px;font-weight:700;color:#0F172A;margin-bottom:6px;'>나와의 적합성</div>", unsafe_allow_html=True)

        _ev_items: list[tuple[str, str]] = []
        for r in evidence[:3]:
            lbl = (r.get("short_label") or "").strip()
            req = _one_line(r.get("requirement") or "")
            if lbl:  # v2 - 표시용 명사구 라벨 + 어떤 요구사항과 연결됐는지
                _ev_items.append((_one_line(lbl), req))
            else:  # 구 캐시(v1) - short_label 없음
                _ev_items.append((req, _one_line(_short_evidence_full(r.get("resume_evidence") or ""))))
        _gap_items = [(_one_line(why), _one_line(prob)) for prob, why in gaps[:3]]

        with st.container(key="qa-fit"):
            ev_col, gap_col = st.columns(2, gap="medium")
            with ev_col:
                st.markdown(
                    "<div class='qa-sub qa-sub-ok'><span class='qa-ic qa-ic-ok'>&#10003;</span>연결되는 경험</div>"
                    + _qa_col_body(_ev_items, "직접 연결되는 근거를 찾지 못했습니다."),
                    unsafe_allow_html=True,
                )
            with gap_col:
                st.markdown(
                    "<div class='qa-sub qa-sub-warn'><span class='qa-ic qa-ic-warn'>!</span>확인이 필요한 부분</div>"
                    + _qa_col_body(_gap_items, "크게 걸리는 점은 없습니다."),
                    unsafe_allow_html=True,
                )

        # ── 상세 근거 보기 (기본 닫힘. 현재 분석 원문 전체 - 삭제 아님) ──
        with st.expander("상세 근거 보기", expanded=False):
            st.markdown("<div style='font-size:13px;font-weight:700;color:#0F172A;'>연결 근거</div>", unsafe_allow_html=True)
            _det = 0
            for i, r in enumerate(qa.get("requirements") or []):
                rel = r.get("relation")
                ev = (r.get("resume_evidence") or "").strip()
                if rel not in ("match", "partial") or not ev:
                    continue
                _tone = "green" if rel == "match" else "blue"
                _lbl = "직접 대응" if rel == "match" else "전이 가능"
                st.markdown(
                    badge_html(_lbl, _tone) + f"&nbsp; <b>{html.escape(r.get('requirement') or '')}</b>",
                    unsafe_allow_html=True,
                )
                st.caption(ev)
                _det += 1
            if _det == 0:
                st.caption("직접 연결되는 근거가 없습니다.")
            st.divider()
            st.markdown("<div style='font-size:13px;font-weight:700;color:#0F172A;'>걸리는 점</div>", unsafe_allow_html=True)
            if gaps:
                for prob, why in gaps:
                    st.markdown(f"<b>{html.escape(prob)}</b>", unsafe_allow_html=True)
                    st.caption(f"→ {why}")
            else:
                st.caption("크게 걸리는 점은 없습니다.")

    st.space(16)

    # ── 3) 지원 준비 Action (비추천이어도 직접 진행 가능) ──
    _label = "그래도 이 공고 지원 준비하기  →" if decision == "비추천" else "이 공고 지원 준비하기  →"
    if _navy_cta(_label, "qa_apply"):
        _run_apply_flow(job)


def _render_job_analysis_v4(job: dict) -> None:
    """STEP3 "공고 분석" v4 - 역할 소개/판단/공고 요약/적합성/(숨김)
    상세 근거/CTA를 전부 analysis_component.analysis_view()에 위임한다.
    데이터는 presentation_layer.py의 v3 Composition 함수(build_role_intro/
    build_decision_banner/build_overview_v3/build_fit_summary_v3/
    build_detail_rows_v3, 전부 이미 있던 순수 함수)만 그대로 모아 JSON
    payload로 넘긴다 - 새 판단/새 LLM 호출 없음. 컴포넌트가
    setTriggerValue로 보내는 apply/save 두 트리거만 받아서 기존
    _run_apply_flow/add_saved_job/remove_saved_job을 그대로 재호출한다.

    2026-08-16(사용자 확정 - "정보 위계/중복 제거" 재개편) - "지원 전략"
    섹션(앞세울 경험/보완할 부분)을 완전히 삭제했다: 앞세울 경험은
    "잘 맞는 부분"+상세 근거와, 보완할 부분은 "지원 전 확인 필요"와
    내용이 그대로 겹쳐서 같은 판단을 세 번째로 설명하는 카드였다(사용자
    지적). 그 섹션 전용이던 resume_semantic_objects 조회(build_strategy_
    view_v3 호출용)도 함께 제거한다 - 다른 곳에서 안 쓰이므로 남겨두면
    쓰이지 않는 이력서 Understanding 조회만 남는다."""
    view = presentation_layer.build_job_card_view(job)
    job_id = str(job.get("job_id"))
    analysis = _get_analysis(job)
    if analysis is None:
        with st.container(border=True, key=f"s1d-top-{job_id}"):
            st.markdown(f"**{view['company']} · {view['title']}**")
        st.info("이 공고는 아직 매칭 근거가 계산되지 않았습니다.")
        return

    banner = presentation_layer.build_decision_banner(analysis)
    overview = presentation_layer.build_overview_v3(job)
    fit = presentation_layer.build_fit_summary_v3(analysis)
    detail_rows = presentation_layer.build_detail_rows_v3(job, analysis)

    def _fit_group(key: str) -> dict:
        return {
            "count": fit[f"{key}_count"],
            "items": [{"title": it["jd_requirement"]} for it in fit[key]],
        }

    payload = {
        "company": f"{view['company']} · {view['title']}",
        "url": view["url"],
        "domain": presentation_layer.build_domain_meta(job),
        # "직무영역" - Rule 2(build_domain_view/_dominant_domain)가 이미
        # 계산해두는 이 공고의 대표 frame.domain 하나를 그대로 재사용한다
        # (2026-08-16 사용자 스펙 - "직무영역" 태그). 새 분류 체계나 LLM
        # 호출을 추가하지 않는다 - "공고 한눈에 보기 > 주요 업무 영역"
        # 카드와 같은 원본 데이터(frame.domain)를 쓰므로, 그 카드의 1위
        # 항목과 이 태그 값이 겹칠 수 있다(정보 계층이 달라 의도적으로
        # 허용 - 상단은 한 줄 요약, 카드는 전체 목록).
        "roleArea": presentation_layer.build_domain_view(job) or "",
        "roleIntro": presentation_layer.build_role_intro(job),
        "decision": banner["decision"],
        "bannerBg": _DECISION_BANNER_BG.get(banner["decision"], "#EEF2F7"),
        "bannerFg": _DECISION_BANNER_FG.get(banner["decision"], "#0F1F4A"),
        "reason": banner["reason"],
        "action": banner["action"],
        "hardEligibility": banner["hardEligibility"],
        "overview": {
            "tasks": overview["tasks"],
            "keyCompetencies": overview["key_competencies"],
            "focusAreas": overview["focus_areas"],
            "keywords": overview["keywords"],
        },
        "fit": {
            "strong": _fit_group("strong"),
            "partial": _fit_group("partial"),
            "risk": _fit_group("risk"),
        },
        "detailTotal": fit["total_count"],
        "detailGroups": detail_rows,
        "saved": is_saved(job["job_id"]),
    }

    result = analysis_view(payload, key=f"av_{job_id}")
    if result is not None:
        if getattr(result, "apply", None):
            _run_apply_flow(job)
        if getattr(result, "save", None):
            if payload["saved"]:
                remove_saved_job(job["job_id"])
            else:
                add_saved_job(job)
            st.rerun()


def _render_job_analysis_panel(job: dict, show_search_reset: bool = True, show_apply_button: bool = True) -> None:
    """공고 분석 + 매칭 결과 - ①요약 ②매칭근거 ③AI가 이해한 공고 ④상세정보
    4단 구조. 원래 S1D(추천 모드) 전용이었으나, 2026-07-18(사용자 요청 -
    "붙여넣기 모드는 판단/이유/부족한점뿐이라 화면이 다르다") S3(JD
    붙여넣기 판단 모드)에서도 같은 구조를 쓰도록 재사용한다 - Judge
    Engine만 있고 jd_semantic_objects가 없는 pasted JD는 이 함수 안의
    use_semantic=False 분기(comparisons/view 기반, 이미 있던 로직)로
    자연스럽게 폴백되므로 새 판단 로직은 추가하지 않는다.

    show_search_reset/show_apply_button: S3에서는 "다시 검색하기"(추천
    목록 개념이 없음)와 "지원 준비하기"(S3 자체의 판단 게이트 버튼과
    중복됨) 둘 다 숨긴다."""
    head_cols = st.columns([1, 4, 2]) if show_search_reset else st.columns([1, 5])
    head_cols[0].markdown(company_logo_html(job.get("company", ""), size=40), unsafe_allow_html=True)
    with head_cols[1]:
        st.markdown(f"**{job.get('company','')}**&nbsp;&nbsp;{job.get('title','')}")
    if show_search_reset:
        with head_cols[2]:
            if st.button("다시 검색하기", key="reset_search", use_container_width=True):
                go_back()

    mj = job.get("meaning_judgment")
    summary_bullets = (mj.get("summary_bullets") or []) if mj else []

    comparisons = (mj.get("comparisons") or []) if mj else []
    comparisons_is_rejudged = bool(comparisons)
    if not comparisons:
        # 이 공고도 이미 M3+M4(Understanding Representation 기반 Meaning
        # Matching, search_candidates_meaning)로 전체 후보 풀 안에서 의미
        # 기준으로 매칭·정렬된 것이다 - "AI가 의미를 안 봤다"는 뜻이
        # 아니다. 상위 30건에만 추가로 붙는 건, 개별 공고를 다시 한 번
        # LLM에 통째로 넣어 항목별 비교표/이유를 새로 생성하는 "재판단"
        # 단계 하나뿐이다(judge_and_reorder_top30). 그 재판단을 못 받은
        # 공고(31~50위, 배치 실패 등)에서는 새 LLM 호출 없이 이미 계산된
        # 키워드 겹침(matched_keywords/tech_stack)으로 같은 모양의 표를
        # 참고용으로 채운다 - 프로즈 대체 문구가 아니라 ②/① 둘 다 같은
        # 구조로 보여준다는 사용자 명시적 요청(2026-07-15).
        comparisons = derive_surface_comparisons(job)

    saved = is_saved(job["job_id"])
    row = st.columns([1, 1, 4])
    if job.get("url"):
        row[0].link_button("공고원문 보기", job["url"])
    elif str(job.get("job_id", "")).startswith("manual-"):
        pass  # JD 붙여넣기 모드는 url이 원래 선택 입력 - 경고 아님
    else:
        # 이 화면은 추천 목록(candidate_jobs)에서만 진입한다 - 그 풀의
        # 공고는 전부 url이 있어야 정상이다(실측: 0/1993건 누락). 여기
        # 걸리면 수집 단계의 데이터 결함이니 조사 대상.
        row[0].caption("⚠ 원문 링크 없음")
    if row[1].button("★ 저장됨" if saved else "☆ 즐겨찾기", key="detail_save"):
        if saved:
            remove_saved_job(job["job_id"])
        else:
            add_saved_job(job)
        st.rerun()

    view = build_jd_detail_view(job)

    # 체크포인트 6(Presentation Layer, 2026-07-16) - 이 job이 새 Semantic
    # Object 스키마로 backfill된 경우(job["jd_semantic_objects"] 존재),
    # ①②③번 섹션은 presentation_layer.py를 통해서만 데이터를 얻는다(app.py가
    # Engine 결과를 직접 읽지 않는다). backfill 안 된 job(다수)은 옛 화면
    # (comparisons/view 기반)으로 폴백 - 이건 새 판단을 만드는 게 아니라
    # 이미 있던 코드를 그대로 재사용하는 것뿐이다.
    use_semantic = presentation_layer.has_semantic_data(job)
    if use_semantic:
        resume_understanding = get_or_generate_resume_understanding(st.session_state.resume_raw)
        resume_semantic_objects = resume_understanding.get("semantic_objects") or []
        resume_relations = resume_understanding.get("relations") or []
        s1 = presentation_layer.build_section1_view(job, _get_resume_facts())
        s2_cards = presentation_layer.build_section2_cards(job, resume_semantic_objects)
        s3 = presentation_layer.build_section3_view(job)

    # ── ① 추천 결과 ────────────────────────────────────────────────────
    # 2026-07-18(사용자 요청, 참고 이미지 기준 재설계) - 적합도 링차트 |
    # AI 공고 한줄요약(+도메인/핵심업무/인재상) | 왜 추천됐는지(레이어별
    # 기여도)를 3열로 배치한다. 전부 presentation_layer.py의 Composition
    # Rule 1~5(순수 함수, LLM 호출 없음)로 만든다. use_semantic=False
    # (backfill 안 된 옛 공고)는 기존 화면을 그대로 유지 - 이 재료들
    # 자체가 jd_semantic_objects 기반이라 없는 공고에는 적용할 수 없다.
    st.markdown("### ① 추천 결과")
    st.markdown(
        f"{status_badge(job.get('career_status', '확인필요'))}&nbsp;&nbsp;"
        f"{badge_html(job.get('source','기업 홈페이지'), 'blue')}",
        unsafe_allow_html=True,
    )
    st.caption(job.get("career_reason", ""))

    if use_semantic:
        col_fit, col_summary, col_why = st.columns([1, 2, 1.3])
        with col_fit:
            # 2026-07-23(서비스 구조 변경, 사용자 확정) - 적합도 링차트만
            # 삭제한다. headline(자연어 문장)/summary_bullets는 이번
            # 작업 범위가 아니라 그대로 유지(Analysis 화면 재설계는
            # 다음 단계). 다만 headline은 원래 링차트 아래 배지로
            # 있었는데, 링차트가 빠지면서 badge_html의 nowrap 스타일이
            # 문장 길이(예: "지원 경쟁력이 높은 편입니다")를 못 감싸
            # 옆 컬럼과 겹치는 게 실제로 확인됐다 - 배지(nowrap, 짧은
            # 라벨 전용) 대신 줄바꿈 가능한 일반 텍스트로만 바꾼다
            # (내용/색상은 그대로, 형태만 pill -> 텍스트).
            _HEADLINE_TONE_COLOR = {
                "green": "var(--mint-dark)", "orange": "var(--peach-dark)",
                "red": "var(--red)", "gray": "var(--text-secondary)",
            }
            if s1["headline"]:
                color = _HEADLINE_TONE_COLOR.get(s1.get("headline_tone") or "gray", "var(--text-secondary)")
                st.markdown(
                    f"<div style='text-align:center;margin-top:0.5rem;font-weight:700;color:{color}'>{s1['headline']}</div>",
                    unsafe_allow_html=True,
                )
            for b in s1["summary_bullets"]:
                st.caption(b)

        with col_summary:
            one_liner = presentation_layer.build_jd_one_liner(job)
            domain = presentation_layer.build_domain_view(job)
            core_tasks = presentation_layer.build_core_tasks_label(job)
            persona = presentation_layer.build_persona_view(job)

            st.markdown(f"{icon_png('sparkles', 16, variant='navy')}&nbsp;**AI 공고 한줄 요약**", unsafe_allow_html=True)
            if one_liner:
                st.markdown(f"<div style='font-size:1.05rem;font-weight:700;margin:0.2rem 0 0.8rem'>{one_liner}</div>", unsafe_allow_html=True)
            else:
                st.caption("공고 원문에서 요약할 재료(문제/업무)를 찾지 못했습니다.")

            def _info_row(icon_name: str, label: str, value: str | None) -> None:
                if not value:
                    return
                st.markdown(
                    f"<div style='display:flex;gap:0.6rem;margin-bottom:0.6rem'>"
                    f"<div>{icon_png(icon_name, 20, variant='navy')}</div>"
                    f"<div><div style='font-size:0.8rem;font-weight:700;color:var(--text-secondary)'>{label}</div>"
                    f"<div>{value}</div></div></div>",
                    unsafe_allow_html=True,
                )

            _info_row("building", "도메인", domain)
            _info_row("chart", "핵심 업무", core_tasks)
            _info_row("user", "찾는 인재상", persona)

        with col_why:
            st.markdown(f"{icon_png('target', 16, variant='navy')}&nbsp;**왜 추천됐는지**", unsafe_allow_html=True)
            breakdown = presentation_layer.build_match_reason_breakdown(job)
            pct_by_layer = {b["layer"]: b["pct"] for b in breakdown}
            for layer, icon_name, color, label in (
                ("skill", "settings", "rgba(15,31,74,1)", "기술 역량"),
                ("task", "clipboard", "rgba(15,31,74,0.8)", "업무 경험"),
                ("problem", "target", "rgba(15,31,74,0.6)", "문제 해결"),
                ("thinking", "user", "rgba(15,31,74,0.4)", "업무 방식"),
            ):
                pct = pct_by_layer.get(layer, 0)
                st.markdown(
                    "<div style='margin-bottom:0.55rem'>"
                    f"<div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:0.2rem'>"
                    f"<span>{icon_png(icon_name, 14, variant='navy')}&nbsp;{label}</span><span style='font-weight:700'>{pct}%</span></div>"
                    f"<div style='background:var(--border-soft,#eee);border-radius:6px;height:7px;overflow:hidden'>"
                    f"<div style='background:{color};width:{pct}%;height:100%'></div></div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<div style='margin-top:0.6rem;padding:0.7rem;border:1px solid var(--border-soft,#eee);border-radius:10px'>"
                f"<div style='font-size:0.8rem;color:var(--text-secondary)'>핵심 요구사항 충족률</div>"
                f"<div style='display:flex;align-items:baseline;justify-content:space-between'>"
                f"<span style='font-size:1.6rem;font-weight:800'>{round(s1['matched_count']/s1['total_count']*100) if s1['total_count'] else 0}%</span>"
                f"<span style='color:var(--text-secondary)'>{s1['matched_count']} / {s1['total_count']} 충족</span></div></div>",
                unsafe_allow_html=True,
            )

        if s1["action_items"]:
            st.markdown("**다음 행동**")
            for a in s1["action_items"]:
                st.markdown(f"▶ {a}")
    else:
        top_left, top_right = st.columns([2, 1])
        with top_left:
            if summary_bullets:
                st.markdown("**AI 판단 요약**")
                for b in summary_bullets:
                    st.markdown(f"- {b}")
            elif mj is not None:
                # comparisons/summary_bullets가 없는 예전 캐시(구 프롬프트 버전) -
                # JUDGMENT_VERSION이 올라갔으므로 실제로는 재계산되지만, 배치
                # 실패로 판단 자체가 비었을 때의 안전한 대체 표시.
                st.caption(mj.get("reasoning") or "")
            if comparisons:
                met_c = sum(1 for c in comparisons if c.get("verdict") == "충족")
                partial_c = sum(1 for c in comparisons if c.get("verdict") == "부분충족")
                total_c = len(comparisons)
                st.markdown(f"**한눈에 보는 매칭** — 공고 핵심 요구사항 {met_c}/{total_c} 충족" + (f" (부분충족 {partial_c}건)" if partial_c else ""))
                st.progress(met_c / total_c if total_c else 0)
                if not comparisons_is_rejudged:
                    st.caption("※ 이 공고는 상위 30건 개별 재판단 대상이 아니어서, 키워드 겹침 기준 참고용 수치입니다.")
        # 2026-07-23(서비스 구조 변경, 사용자 확정) - "직무 적합도 예측"
        # 링차트/등급 배지를 삭제했다(top_right는 더 이상 채우지 않음 -
        # 나머지 ②③ 섹션은 이번 작업 범위가 아니라 그대로 유지).

    st.divider()

    # ── ② AI 의미 기반 매칭 ────────────────────────────────────────────
    # 2026-07-19(사용자 상세 명세) - "JD 요구사항을 늘어놓는 게 아니라
    # 요구 하나와 연결된 내 경험 하나를 1:1로 보여준다." 카드 구성
    # (회사가 원하는 것/내 경험/연결된 근거/매칭 수준/설명 문장)과
    # 노출 우선순위·제외 규칙은 전부 presentation_layer.build_section2_
    # cards()가 계산한다(이미 있는 Match Object 필드 조합, 새 LLM 없음).
    st.markdown("### ② AI 의미 기반 매칭")
    if use_semantic:
        def _render_match_card(card: dict) -> None:
            with st.container(border=True):
                st.markdown(badge_html(card["level_label"], card["level_tone"]), unsafe_allow_html=True)
                st.markdown(f"**{card['jd_point']}**")
                st.caption(f"내 경험: {card['resume_project']} — {card['resume_point']}")
                st.write(card["description"])

        for card in s2_cards["default"]:
            _render_match_card(card)
        if s2_cards["more"]:
            with st.expander(f"나머지 매칭 보기 ({len(s2_cards['more'])}건)"):
                for card in s2_cards["more"]:
                    _render_match_card(card)
        if not s2_cards["default"] and not s2_cards["more"]:
            st.caption("연결된 근거를 찾지 못했습니다.")
    elif comparisons:
        if not comparisons_is_rejudged:
            st.caption("이 공고는 상위 30건에만 적용되는 개별 재판단 대상이 아니어서, 아래는 이력서·공고 키워드가 겹치는지로 만든 참고용 표입니다.")
        st.markdown(
            "| 공고가 원하는 것 | 이력서에서 찾은 근거 | AI 판단 | 설명 |\n"
            "|---|---|---|---|\n"
            + "\n".join(
                f"| {c.get('jd_point','')} | {c.get('resume_evidence','')} | {c.get('verdict','')} | {c.get('explanation','')} |"
                for c in comparisons
            )
        )
    else:
        # matched_keywords/tech_stack 둘 다 비어 있는 아주 드문 경우
        # (표면 신호 자체가 없음) - 지어낼 근거가 없으니 솔직하게 표시.
        st.caption("일치하는 근거를 찾지 못했습니다.")

    # ── ③ 보완하면 좋은 점 ────────────────────────────────────────────
    # 2026-07-19(사용자 상세 명세) - "커스터마이징 제안이 아니라, 공고가
    # 요구하지만 확인되지 않는 필수·우대 조건을 분리해 보여주는 영역".
    # 기술명만 나열하지 않고 카드(공고 요구/현재 이력서/중요도/판정)로
    # 보여준다 - "확인 필요"와 "미충족"을 구분해서 실제 보유 여부를
    # 모르는 것을 "없다"고 단정하지 않는다.
    st.divider()
    st.markdown("### ③ 보완하면 좋은 점")
    if use_semantic:
        gaps = presentation_layer.build_gaps_detail_view(job)

        def _render_gap_card(card: dict) -> None:
            verdict_tone = "red" if card["verdict"] == "미충족" else "gray"
            with st.expander(f"{card['title']}  ·  {card['verdict']}"):
                st.markdown(badge_html(card["verdict"], verdict_tone), unsafe_allow_html=True)
                st.caption(f"공고 요구: {card['ask']}")
                st.caption(f"현재 이력서: {card['current']}")
                st.caption(f"중요도: {card['importance']}")

        gap_cols = st.columns(2)
        with gap_cols[0]:
            st.markdown(f"{icon_png('target', 15, variant='navy')}&nbsp;**지원 전에 반드시 확인할 조건**", unsafe_allow_html=True)
            if gaps["required"]:
                for card in gaps["required"]:
                    _render_gap_card(card)
                if gaps["required_more"]:
                    with st.expander(f"더보기 ({len(gaps['required_more'])}건)"):
                        for card in gaps["required_more"]:
                            _render_gap_card(card)
            else:
                st.caption("확인이 필요한 필수 조건이 없습니다.")
        with gap_cols[1]:
            st.markdown(f"{icon_png('success', 15, variant='navy')}&nbsp;**있으면 유리한 조건**", unsafe_allow_html=True)
            if gaps["preferred"]:
                for card in gaps["preferred"]:
                    _render_gap_card(card)
                if gaps["preferred_more"]:
                    with st.expander(f"더보기 ({len(gaps['preferred_more'])}건)"):
                        for card in gaps["preferred_more"]:
                            _render_gap_card(card)
            else:
                st.caption("확인이 필요한 우대 조건이 없습니다.")
    else:
        strengths, gaps = derive_strengths_and_gaps(job.get("matched_keywords") or [], job.get("posting_text", ""))
        tag_cols = st.columns(2)
        with tag_cols[0]:
            st.caption("강점 (이력서와 겹치는 키워드)")
            if strengths:
                st.markdown(" ".join(badge_html(t, "green") for t in strengths), unsafe_allow_html=True)
            else:
                st.caption("강점으로 확인된 항목이 없습니다.")
        with tag_cols[1]:
            st.caption("보완점 (공고가 요구하는 기술 중 이력서에 없는 키워드)")
            if gaps:
                st.markdown(" ".join(badge_html(t, "orange") for t in gaps), unsafe_allow_html=True)
            else:
                st.caption("확인된 보완점이 없습니다.")

    # "AI가 이해한 공고" 상세(목적/문제/사고방식/환경) - ①의 한줄요약/
    # 도메인/핵심업무/인재상과 내용이 겹쳐서 별도 번호 섹션 대신 접어서
    # 보조 정보로만 남긴다(이미 계산된 값이라 지우지 않고 유지).
    if use_semantic:
        jd_summary = s3.get("summary") or {}
        if jd_summary or s3["sections"]:
            with st.expander("공고 원문 요약 더 보기"):
                for key, label in (
                    ("purpose", "직무 목적"), ("problem", "해결하려는 문제"),
                    ("thinking", "중요하게 보는 사고방식"), ("environment", "일하는 환경"),
                ):
                    if jd_summary.get(key):
                        st.markdown(f"**{label}**")
                        st.write(jd_summary[key])
                for sec in s3["sections"]:
                    st.markdown(f"**{sec['label']}**")
                    for item in sec["items"]:
                        st.markdown(f"- {item}")
    elif view is None:
        sections = extract_display_sections(job)
        with st.expander("공고 소개 더 보기"):
            st.write(format_company_intro(job) or "정보 없음")
            st.markdown("**주요업무**")
            st.write(sections.get("resp") or "정보 없음")
    else:
        summary = view["요약"]
        detail = view["상세분석"]
        with st.expander("공고 소개 더 보기"):
            st.markdown("**직무 목적**")
            st.write(summary["핵심역할"] or "정보 없음")
            if summary["핵심업무"]:
                st.markdown("**핵심 업무**")
                for item in summary["핵심업무"]:
                    st.markdown(f"- {item}")
            st.markdown("**일하는 방식 / 사고방식**")
            st.write(detail["중요하게_보는_역량"] or "정보 없음")
            if detail.get("신중검토"):
                st.markdown("**이런 경우는 신중하게 검토해보세요**")
                st.write(detail["신중검토"])

    st.divider()

    # ── ④ 공고 상세 정보 (핵심만 구조화한 카드 - 원문 재출력 아님) ──────
    # 사용자 요청(2026-07-16): 상세정보는 공고를 재출력하는 곳이 아니라
    # 지원 조건/전형절차/자격요건/우대사항/주요기술만 카드로 정리하는
    # 곳이어야 한다. "동료의 한마디"/FAQ/카카오톡 문의 같은 노이즈는
    # job_detail._classify_display_sections()가 discard 버킷으로
    # 걸러서 애초에 어떤 카드에도 들어오지 않는다(노이즈 헤더를
    # 만나면 이후 텍스트를 폐기 - job_detail.py 참고).
    st.markdown("### ④ 공고 상세 정보")
    # 2026-07-19(사용자 상세 명세) - "공고 원문을 사람이 보기 쉽게
    # 재구성한다. 새 해석/매칭 설명을 넣는 곳이 아니다." 지원조건 →
    # 우대사항 → 근무및지원정보 → 복지및혜택 → 원문보기 순(사용자
    # 명시적 요청으로 주요 업무 카드는 제외). 구조화된 필드(job_prep.
    # extract_application_prep, career_level 컬럼)를 우선 쓰고, 없는
    # 것만 evidence/원문에서 보조한다 - 데이터가 없는 항목(전공, 근무
    # 시간, 재택 여부 등 이 파이프라인이 아직 추출하지 않는 필드)은
    # 지어내지 않고 행 자체를 숨긴다.
    if view is not None:
        culture = view["조직문화"]
        meta_bits = [v for v in (culture["산업"], culture["기업규모"], culture["근무지역"]) if v]
        if culture["회사소개"]:
            st.caption(culture["회사소개"] + (" · " + " · ".join(meta_bits) if meta_bits else ""))
        elif meta_bits:
            st.caption(" · ".join(meta_bits))

        info = view["지원_정보"]
        skills = view["요구스킬"]
        job_info = info.get("기타") or {}

        # 1. 지원 조건
        with st.container(border=True):
            st.markdown(f"{icon_png('briefcase', 15, variant='navy')}&nbsp;**지원 조건**", unsafe_allow_html=True)
            cond_rows = []
            if job.get("career_level"):
                cond_rows.append(("경력", job["career_level"]))
            if job_info.get("학력 조건"):
                cond_rows.append(("학력", job_info["학력 조건"]))
            if skills["주요_기술"]:
                cond_rows.append(("필수 기술", " · ".join(skills["주요_기술"])))
            if cond_rows:
                for label, value in cond_rows:
                    st.caption(f"{label}: {value}")
            qual_bullets = presentation_layer.split_qualification_bullets(skills["자격요건"])
            if qual_bullets:
                for b in qual_bullets:
                    st.markdown(f"- {b}")
            if not cond_rows and not qual_bullets:
                st.caption("정보 없음")

        # 2. 우대사항
        with st.container(border=True):
            st.markdown(f"{icon_png('success', 15, variant='navy')}&nbsp;**우대사항**", unsafe_allow_html=True)
            preferred_classified = presentation_layer.build_preferred_classified_view(job)
            if preferred_classified:
                for cat, items in preferred_classified.items():
                    st.caption(f"{cat}: " + " · ".join(items))
            elif skills["우대사항"]:
                for b in presentation_layer.split_qualification_bullets(skills["우대사항"]):
                    st.markdown(f"- {b}")
            else:
                st.caption("정보 없음")

        # 2.5 인재상 (2026-07-21 - 자동 커스터마이징 대상 아님, 표시만.
        # JD Understanding이 이미 추출한 culture 레이어를 그대로 보여준다.)
        with st.container(border=True):
            st.markdown(f"{icon_png('user', 15, variant='navy')}&nbsp;**이 회사가 중요하게 보는 인재상**", unsafe_allow_html=True)
            culture_items = culture.get("인재상") or []
            if culture_items:
                for c in culture_items:
                    st.markdown(f"- {c}")
            else:
                st.caption("정보 없음")

        # 3. 근무 및 지원 정보 (값 없는 행은 숨김 - "미기재" 반복 금지)
        with st.container(border=True):
            st.markdown(f"{icon_png('building', 15, variant='navy')}&nbsp;**근무 및 지원 정보**", unsafe_allow_html=True)
            work_rows = []
            if culture["근무지역"]:
                work_rows.append(("근무지", culture["근무지역"]))
            if job_info.get("근무 형태"):
                work_rows.append(("고용형태", job_info["근무 형태"]))
            if info["마감"]:
                work_rows.append(("마감일", info["마감"]))
            if info["필요서류"]:
                work_rows.append(("제출 서류", ", ".join(info["필요서류"])))
            if work_rows:
                for label, value in work_rows:
                    st.caption(f"{label}: {value}")
            if info["지원절차"]:
                st.caption("전형 절차: " + " → ".join(info["지원절차"]))
            if not work_rows and not info["지원절차"]:
                st.caption("정보 없음")

        # 4. 복지 및 혜택
        with st.container(border=True):
            st.markdown(f"{icon_png('success', 15, variant='navy')}&nbsp;**복지 및 혜택**", unsafe_allow_html=True)
            welfare = presentation_layer.build_welfare_classified_view(job_info.get("복리후생", ""))
            if welfare:
                _WELFARE_MAX = 3
                for cat, items in welfare.items():
                    shown, rest = items[:_WELFARE_MAX], items[_WELFARE_MAX:]
                    st.caption(f"{cat}: " + " · ".join(shown))
                    if rest:
                        with st.expander(f"{cat} 더보기 ({len(rest)}건)"):
                            for r in rest:
                                st.markdown(f"- {r}")
            else:
                st.caption("정보 없음")

        # 5. 원문 보기
        with st.container(border=True):
            st.markdown(f"{icon_png('document', 15, variant='navy')}&nbsp;**원문 보기**", unsafe_allow_html=True)
            if job.get("url"):
                st.caption("채용 공고 원문을 확인하세요.")
                st.link_button("공고 바로가기 ↗", job["url"], width="stretch")
            else:
                st.caption("원문 링크가 없습니다.")
    else:
        sections = extract_display_sections(job)
        with st.container(border=True):
            st.markdown("**자격요건**")
            st.write(sections.get("req") or "정보 없음")
        with st.container(border=True):
            st.markdown("**우대사항**")
            st.write(sections.get("pref") or "정보 없음")

    st.caption("※ 위 분석 결과는 AI가 JD와 이력서를 기반으로 도출한 참고 정보이며, 최종 판단은 지원자 본인이 결정하시기 바랍니다.")

    _render_analysis_evidence_section(job)

    if show_apply_button and st.button("이 공고 지원 준비하기", type="primary", key="detail_apply"):
        _run_apply_flow(job)


def _run_apply_flow(job: dict) -> None:
    """추천 모드 [지원하기] - 분석 엔진(judge)을 내부적으로만 호출한다.
    반환값의 decision/reason은 여기서도 읽지 않는다."""
    with st.spinner("분석 중입니다... (공고 맞춤 분석 준비)"):
        try:
            result = pipeline.run_apply_flow(st.session_state.resume_raw, job)
            st.session_state.apply_result = result
            st.session_state.resume_pdf_result = None
            st.session_state.html_resume_result = None
            st.session_state.html_resume_failed = False
            st.session_state.portfolio_pptx_result = None
            go("S4")
        except LLMCallError as e:
            show_llm_error("분석 실패", e)
        except Exception as e:
            print(f"[지원 플로우 오류] {e}")
            st.error("분석 중 오류가 발생했습니다.")


# ── S2. JD 붙여넣기 모드 ─────────────────────────────────────────────

def render_s2() -> None:
    st.markdown(_ANALYSIS_CSS, unsafe_allow_html=True)
    st.markdown(step_progress_html(2), unsafe_allow_html=True)  # 3단계: 공고 분석(붙여넣기)
    page_header(svg_icon("clipboard-list", 40, "var(--navy)"), "채용공고 붙여넣기 분석")
    back_button()
    if not _require_resume():
        return

    with st.container(border=True):
        company = st.text_input("회사명")
        title = st.text_input("직무명")
        jd_url = st.text_input("공고 원문 링크 (선택)", placeholder="https://... - 있으면 나중에 다시 볼 수 있어요")
        jd_text = st.text_area("채용공고 본문을 붙여넣으세요", height=300)

        if st.button("분석", type="primary"):
            if not jd_text.strip():
                st.warning("채용공고 본문을 입력해주세요.")
                return
            with st.spinner("판단 중입니다..."):
                try:
                    result = pipeline.run_jd_analysis_mode(
                        st.session_state.resume_raw, jd_text, company=company, title=title, url=jd_url.strip()
                    )
                    st.session_state.selected_job = result["job"]
                    st.session_state.judge_result = result["judge_result"]
                    st.session_state.apply_result = None
                    st.session_state.resume_pdf_result = None
                    st.session_state.html_resume_result = None
                    st.session_state.html_resume_failed = False
                    st.session_state.portfolio_pptx_result = None
                    # T0(붙여넣기모드) - 사용자가 JD를 붙여넣고 "분석"을
                    # 명시적으로 실행한 시점. 추천모드 analysis_opened 와 동일 기준.
                    _log_prep_event_once(str(result["job"].get("job_id")), "analysis_opened")
                    # 2026-09-01(사용자 확정 - "두 모드는 공고 찾기 앞부분만 다르고
                    # 나머지는 전부 동일") - 붙여넣기도 분석 이후는 추천 모드와 같은
                    # 화면/흐름(render_s1d → _render_quick_analysis → 지원 준비 → 지원
                    # 기록)을 그대로 탄다. run_jd_analysis_mode 가 이미 _quick_analysis
                    # 를 job 에 붙여줬으므로 render_s1d 는 새 LLM 없이 그 결과를 쓴다.
                    go("S1D")
                except LLMCallError as e:
                    show_llm_error("판단 실패", e)
                except Exception as e:
                    print(f"[JD 분석 오류] {e}")
                    st.error("판단 중 오류가 발생했습니다.")


# ── S3. Judge 결과 + Human Approval (JD 붙여넣기 모드 전용) ───────────

def render_s3() -> None:
    """2026-09-01(사용자 확정 - "두 모드는 공고 찾기 앞부분만 다르고 나머지는
    전부 동일") - 붙여넣기 모드의 분석 결과 화면을 없애고 추천 모드와 같은
    render_s1d 를 그대로 쓴다. render_s2 가 이제 go("S1D") 로 보내므로 이
    함수는 구 경로(북마크/뒤로가기 alias)로 들어온 경우의 안전한 폴백일 뿐이다."""
    render_s1d()


# ── S4. 공고 커스터마이징 (2026-07-21 UI/UX 전면 리디자인 - 화면만
#    변경, Planner/PDF/데이터 로직은 전혀 손대지 않는다) ──────────────

_S4_CSS = """
<style>
.s4-wrap { max-width: 1440px; margin: 0 auto; }
.s4-step-nav { display:flex; align-items:flex-start; justify-content:center; gap:2px; padding: 4px 0 28px; flex-wrap: wrap; }
.s4-step-item { display:flex; flex-direction:column; align-items:center; width:84px; }
.s4-step-circle { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:14px; }
.s4-step-circle.current { background:#1E3A8A; color:#fff; }
.s4-step-circle.done { background:#fff; border:1px solid #86EFAC; color:#16A34A; }
.s4-step-circle.past { background:#fff; border:1px solid #CBD5E1; color:#334155; }
.s4-step-circle.future { background:#F8FAFC; border:1px solid #E2E8F0; color:#94A3B8; }
.s4-step-label { font-size:12px; margin-top:6px; text-align:center; color:#64748B; }
.s4-step-label.current { color:#0F172A; font-weight:700; }
.s4-step-connector { height:1px; background:#E2E8F0; flex:1; margin-top:17px; min-width:16px; }
.s4-title { font-size:24px; font-weight:700; color:#0F172A; margin:4px 0 0; }
.s4-subtitle { font-size:14px; color:#64748B; margin:6px 0 20px; }
.s4-section-title { font-size:18px; font-weight:600; color:#0F172A; margin:0 0 2px; }
.s4-section-sub { font-size:13px; color:#64748B; margin-bottom:14px; }
.s4-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:16px; min-height:150px;
  box-shadow:0 1px 3px rgba(15,23,42,.04); display:flex; flex-direction:column; gap:8px; }
.s4-card-icon { width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; }
.s4-card-icon.change { background:#DCFCE7; }
.s4-card-icon.keep { background:#F1F5F9; }
.s4-card-icon.manual_review { background:#FFF7ED; }
.s4-card-title { font-size:16px; font-weight:600; margin:0; color:#0F172A; }
.s4-card-desc { font-size:12px; color:#64748B; line-height:1.5; flex-grow:1; margin:0; }
.s4-badge { display:inline-flex; align-items:center; height:22px; padding:0 10px; border-radius:999px; font-size:12px; font-weight:700; width:fit-content; }
.s4-badge.change { background:#DCFCE7; color:#16A34A; }
.s4-badge.keep { background:#F1F5F9; color:#64748B; }
.s4-badge.manual_review { background:#FFEDD5; color:#C2410C; }
.s4-compare-header { padding:11px 18px; font-size:14px; font-weight:700; border-radius:12px 12px 0 0; display:flex; justify-content:space-between; align-items:center; border:1px solid #E2E8F0; border-bottom:none; }
.s4-compare-header.original { background:#F1F5F9; color:#334155; }
.s4-compare-header.customized { background:#ECFDF5; color:#15803D; }
.s4-compare-body { padding:18px 20px; height:520px; overflow-y:auto; font-size:13px; line-height:1.9; color:#0F172A;
  border:1px solid #E2E8F0; border-radius:0 0 12px 12px; background:#fff; }
.s4-changed { background:#F0FDF4; color:#166534; font-weight:600; border-left:3px solid #22C55E; padding:1px 6px 1px 8px; margin-left:-3px; border-radius:3px; }
.s4-arrow-circle { width:40px; height:40px; border-radius:50%; background:#fff; border:1px solid #CBD5E1; box-shadow:0 2px 5px rgba(15,23,42,.08);
  display:flex; align-items:center; justify-content:center; margin: 240px auto 0; }
.s4-principle-block { display:flex; gap:14px; padding:15px 0; border-bottom:1px solid #E2E8F0; }
.s4-principle-block:last-child { border-bottom:none; }
.s4-principle-icon { width:34px; height:34px; border-radius:9px; background:#EFF6FF; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.s4-principle-title { font-size:14px; font-weight:600; color:#0F172A; margin:0 0 2px; }
.s4-principle-desc { font-size:13px; color:#64748B; line-height:1.5; margin:0; }
.s4-empty { text-align:center; padding: 32px 16px; color:#64748B; }
.s4-changeitem { border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.s4-changeitem-title { font-size:14px; font-weight:600; color:#0F172A; display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.s4-changeitem-desc { font-size:13px; color:#475569; margin-bottom:8px; }
.s4-reorder-tag { display:inline-flex; align-items:center; gap:2px; height:20px; padding:0 8px; margin-left:8px;
  border-radius:999px; font-size:11px; font-weight:700; background:#DCFCE7; color:#16A34A; vertical-align:middle; }
/* 지원 준비(2026-08-31 재구성) - 기술/프로젝트 순서 chip flow */
.s4-chip { display:inline-flex; align-items:center; padding:6px 12px; border:1px solid #E5E7EB; border-radius:8px;
  background:#FFFFFF; font-size:13px; color:#0F172A; white-space:nowrap; }
.s4-chip-arrow { color:#94A3B8; }
.s4-chip-row { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:10px 0 6px; }
.s4-ok-badge { display:inline-flex; align-items:center; height:22px; padding:0 10px; border-radius:999px;
  font-size:12px; font-weight:700; background:#ECFDF5; color:#16A34A; }
.s4-hint-badge { display:inline-flex; align-items:center; height:22px; padding:0 10px; border-radius:999px;
  font-size:12px; font-weight:700; background:#EEF4FF; color:#1E3A8A; }
.s4-note { font-size:12.5px; color:#64748B; margin:2px 0 0; }
/* 공고 헤더 - 얇게(제목+회사 한 줄 / 메타 한 줄 / 원문 링크). 카드 padding 축소. */
.stApp div[class*="st-key-s4-jobhead"] { padding: 0.8rem 1.4rem !important; }
.s4-jobcard-title { font-size:15px; font-weight:700; color:#0F172A; margin:0; }
.s4-jobcard-company { font-size:13px; font-weight:400; color:#64748B; margin-left:8px; }
.s4-jobcard-meta { font-size:12.5px; color:#64748B; margin:5px 0 0; }
</style>
"""

_S4_ICON_SVGS = {
    "structure": '<path d="M3 6h18M3 12h12M3 18h18"/><path d="M16 14l3 3-3 3"/>',
    "star": '<polygon points="12 2 15 9 22 9.5 17 14.5 18.5 22 12 18 5.5 22 7 14.5 2 9.5 9 9"/>',
    "pencil": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "badge-check": '<circle cx="12" cy="12" r="9"/><path d="m9 12 2 2 4-4"/>',
    "shield-check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/>',
    "list-checks": '<path d="M4 6h1M4 12h1M4 18h1"/><path d="m5.5 5.5 1 1 2-2"/><path d="m5.5 17.5 1 1 2-2"/><path d="M11 6h9M11 12h9M11 18h9"/>',
    "arrows-updown": '<path d="m21 16-4 4-4-4"/><path d="M17 20V4"/><path d="m3 8 4-4 4 4"/><path d="M7 4v16"/>',
    "file-check": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="m9 15 2 2 4-4"/>',
    "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h6"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "download": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
    "file-search": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h5"/><path d="M13 2v6h6"/><circle cx="16.5" cy="17.5" r="2.5"/><path d="m19.5 20.5-1.5-1.5"/>',
    "list-filter": '<path d="M4 6h16M7 12h10M10 18h4"/>',
    "circle-check": '<circle cx="12" cy="12" r="9"/><path d="m9 12 2 2 4-4"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "person": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/>',
    "layers": '<polygon points="12 2 22 8 12 14 2 8 12 2"/><polyline points="2 14 12 20 22 14"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "users": '<circle cx="9" cy="8" r="3.2"/><path d="M2.5 20c0-3.5 3-5.5 6.5-5.5s6.5 2 6.5 5.5"/><path d="M16.5 8a3 3 0 0 1 0 5.8"/><path d="M20 20c0-2.6-1.6-4.4-3.7-5.2"/>',
    "message-circle": '<path d="M21 11.5a8.4 8.4 0 0 1-8.9 8.4 8.6 8.6 0 0 1-4-1L3 20l1.1-5a8.4 8.4 0 0 1-1-4A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5Z"/>',
}


def _s4_icon(name: str, color: str = "#334155", size: int = 20, stroke_width: float = 2) -> str:
    body = _S4_ICON_SVGS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


# 2026-08-15(사용자 확정) - "매칭 결과"는 공고 분석 화면(v4)에 이미
# 흡수되어 실제로 도달하는 화면이 없던 죽은 단계라 제거(ui_theme.py의
# _STEP_LABELS와 동일한 5단계로 맞춘다).
_S4_STEPS = [
    (1, "이력서 업로드"), (2, "공고 찾기"), (3, "공고 분석"),
    (4, "이력서 수정"), (5, "자소서 준비"), (6, "지원 기록"),
]


def _s4_step_nav(current: int = 4) -> None:
    parts = ['<div class="s4-step-nav">']
    for i, (n, label) in enumerate(_S4_STEPS):
        if n < current:
            circle_cls, label_cls, content = "past", "", str(n)
        elif n == current:
            circle_cls, label_cls, content = "current", "current", str(n)
        else:
            circle_cls, label_cls, content = "future", "", str(n)
        if i > 0:
            parts.append('<div class="s4-step-connector"></div>')
        parts.append(
            f'<div class="s4-step-item"><div class="s4-step-circle {circle_cls}">{content}</div>'
            f'<div class="s4-step-label {label_cls}">{label}</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


_S4_PRINCIPLES = [
    ("shield-check", "기존 사실만 사용", "원본 이력서에 존재하는 경험·성과·기술만 사용합니다. 새로운 경험이나 성과를 만들지 않습니다."),
    ("list-checks", "필요한 항목만 변경", "모든 항목을 일괄 수정하지 않고, 공고에 맞춰 변경 실익이 있는 항목만 수정합니다."),
    ("arrows-updown", "순서와 표현만 최적화", "프로젝트·기술·기존 문장의 순서를 바꾸거나 승인된 표현만 반영합니다."),
    ("file-check", "원본 형식 유지", "사용자가 등록한 이력서의 디자인과 레이아웃을 유지하고 변경 대상 영역만 수정합니다."),
]


@st.dialog("이력서 수정 원칙", width="large")
def _s4_principles_dialog() -> None:
    st.caption("이번 맞춤 이력서는 다음 원칙에 따라 생성되었습니다.")
    blocks = ['<div>']
    for icon, title, desc in _S4_PRINCIPLES:
        blocks.append(
            '<div class="s4-principle-block">'
            f'<div class="s4-principle-icon">{_s4_icon(icon, "#1E3A8A", 18)}</div>'
            f'<div><div class="s4-principle-title">{title}</div>'
            f'<div class="s4-principle-desc">{desc}</div></div></div>'
        )
    blocks.append('</div>')
    st.markdown("".join(blocks), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("확인", type="primary", key="s4_principles_ok"):
        st.rerun()


_S4_ITEM_META = {
    "자기소개 최적화": ("한줄 소개", "person"),
    "프로젝트 순서": ("프로젝트 순서", "layers"),
    "기술 순서": ("기술 순서", "code"),
    "핵심 경험·성과": ("핵심 경험·성과", "target"),
    "인재상·업무방식": ("인재상·업무방식", "users"),
    "표현 및 JD 용어 최적화": ("JD 표현 정렬", "message-circle"),
}

_S4_KEEP_TEXT = {
    "프로젝트 순서": "현재 프로젝트 순서가 공고의 핵심 직무 요구를 이미 잘 보여주고 있어 유지했습니다.",
    "기술 순서": "현재 기술 순서가 공고의 요구와 이미 일치하거나, 연결되는 보유 기술이 없어 유지했습니다.",
    "핵심 경험·성과": "현재 경험과 성과의 강조 순서가 공고의 핵심 요구와 이미 일치해 유지했습니다.",
    "인재상·업무방식": "이 공고의 인재상과 A/B로 연결된 실제 근거가 없어 새로 만들지 않고 유지했습니다.",
    "표현 및 JD 용어 최적화": "같은 의미로 안전하게 바꿀 수 있는 승인 표현이 없어 원문을 유지했습니다.",
    "자기소개 최적화": "현재 자기소개가 공고에서 요구하는 업무 방향을 이미 잘 설명하고 있어 유지했습니다.",
}
_S4_REVIEW_TEXT = "기존 정보만으로 안전하게 자동 변경할 근거가 부족해 직접 확인이 필요합니다."

_S4_INTRO_NOT_SUPPORTED = (
    "현재 엔진은 자기소개 문구를 새로 생성하지 않습니다 - 프로젝트/기술 "
    "강조와 승인된 용어 치환만 자동 적용합니다."
)


def _s4_build_decisions(
    resume_commands: list[dict], applied: dict, term_replacements: list[dict],
    planner_notes: dict | None = None,
) -> list[dict]:
    """Resume Commands + customization_planner.py의 판단 근거(2026-08-16
    추가된 planner_notes) -> S4 화면이 쓰는 4항목 decisions(item/status/
    reason/before/after) 뷰모델로 변환하는 Adapter(사용자 확정, 2026-08-06,
    2026-08-16 헤드라인/기술순서 갱신). resume_commands는 move_project/
    move_bullet/reorder_skills 같은 내부 구현 타입을 담고 있어 화면에
    그대로 노출하면 UI가 엔진 구현에 종속된다 - 여기서 "무엇이 바뀌었는가"
    라는 의미 단위로만 변환한다.

    2026-08-16(Customization Planner 연결, 6개 슬롯 그대로 노출 - 사용자
    확정, 4카드로 합치지 않는다) - ②③④⑤①은 이제 customization_planner.py
    가 만든 planner_notes를 슬롯별로 그대로 읽는다(applied dict는 실제
    실행된 Command 결과 - ②③④는 거기서, ⑤는 실행되는 Command가 따로
    없어(④와 통합 실행) planner_notes에서만 읽는다). "핵심 경험·성과"와
    "인재상·업무방식"은 같은 move_bullet 판단(decide_bullet_order)의
    결과를 driven_by("job"/"culture")로 나눠서 보여준다 - 실제 배치는
    ④+⑤ 통합 판단 하나뿐이라는 원칙(사용자 확정 - 서로 다른 bullet을
    각자 앞으로 보내려는 충돌 방지)은 그대로 유지하면서, 화면에서는
    "왜 바뀌었는가"를 두 줄로 나눠 보여준다."""
    planner_notes = planner_notes or {}
    term_applied = applied.get("term_replacements") or []

    proj_note = planner_notes.get("project_order") or {}
    if proj_note.get("decision") == "변경":
        project_decision = {
            "item": "프로젝트 순서", "status": "change", "reason": proj_note.get("reason", ""),
            "before": None, "after": {"project_order": proj_note.get("project_order")},
        }
    else:
        project_decision = {"item": "프로젝트 순서", "status": "keep",
                             "evidence": bool(proj_note), "reason": proj_note.get("reason", "")}

    skill_note = planner_notes.get("skill_order") or {}
    if skill_note.get("decision") == "변경":
        skill_decision = {
            "item": "기술 순서", "status": "change", "reason": skill_note.get("reason", ""),
            "before": None, "after": {"skill_order": skill_note.get("skill_order")},
        }
    else:
        skill_decision = {"item": "기술 순서", "status": "keep",
                           "evidence": bool(skill_note), "reason": skill_note.get("reason", "")}

    bullet_notes = planner_notes.get("bullet_order") or []
    job_driven = [b for b in bullet_notes if b.get("decision") == "변경" and b.get("driven_by") == "job"]
    culture_driven = [b for b in bullet_notes if b.get("decision") == "변경" and b.get("driven_by") == "culture"]

    if job_driven:
        exp_decision = {
            "item": "핵심 경험·성과", "status": "change", "count": len(job_driven),
            "reason": " ".join(f"[{b['project_id']}] {b['reason']}" for b in job_driven),
            "before": None, "after": {"bullet_reorders": [b["new_order"] for b in job_driven]},
        }
    else:
        exp_decision = {"item": "핵심 경험·성과", "status": "keep"}

    culture_evidence = planner_notes.get("culture_evidence") or {}
    if culture_driven:
        persona_decision = {
            "item": "인재상·업무방식", "status": "change", "count": len(culture_driven),
            "reason": " ".join(f"[{b['project_id']}] {b['reason']}" for b in culture_driven),
            "before": None, "after": {"bullet_reorders": [b["new_order"] for b in culture_driven]},
        }
    elif culture_evidence.get("has_evidence"):
        persona_decision = {
            "item": "인재상·업무방식", "status": "keep", "evidence": True,
            "reason": "실제 연결된 인재상 근거가 있지만(" + ", ".join(culture_evidence["requirements"]) +
                      "), 직무 근거만으로 이미 최적 배치라 별도 조정이 필요하지 않습니다.",
        }
    else:
        persona_decision = {"item": "인재상·업무방식", "status": "keep"}

    if term_applied:
        term_decision = {
            "item": "표현 및 JD 용어 최적화", "status": "change", "count": len(term_applied),
            "reason": f"공고 표현에 맞춰 승인된 용어로 치환했습니다 ({len(term_applied)}건).",
            "before": [t.get("before") for t in term_applied],
            "after": [t.get("after") for t in term_applied],
        }
    else:
        term_decision = {"item": "표현 및 JD 용어 최적화", "status": "keep"}

    # 2026-08-16(① 2차 재설계, 사용자 확정) - Linking의 A/B/C/D(=JD 수준
    # "충족" 여부)를 더 이상 재사용하지 않는다. "어느 인재상을 볼지"만
    # 규칙으로 고르고(critical>core>normal, 동점이면 유지), "그 인재상
    # 중 이력서 전체에서 뒷받침되는 부분이 있는가"는 LLM 1회 호출
    # (rewrite_engine.generate_headline_proposal)로 판단한다 - 아직
    # 페이지 로드 시 자동 실행하지 않는다(LLM On-Demand 원칙 유지, 별도
    # 버튼으로 연결 예정). decision=="평가 필요"일 때만 manual_review로
    # 노출한다 - 예전처럼 매번 뜨는 게 아니라 최고 우선순위 인재상이
    # 하나로 정해질 때만 뜬다.
    headline_note = planner_notes.get("headline")
    if headline_note and headline_note.get("decision") == "평가 필요":
        headline_decision = {
            "item": "자기소개 최적화", "status": "manual_review",
            "reason": headline_note.get("reason", _S4_INTRO_NOT_SUPPORTED),
        }
    elif headline_note:
        headline_decision = {
            "item": "자기소개 최적화", "status": "keep",
            "evidence": True, "reason": headline_note.get("reason", _S4_INTRO_NOT_SUPPORTED),
        }
    else:
        headline_decision = {
            "item": "자기소개 최적화", "status": "keep",
            "evidence": True, "reason": _S4_INTRO_NOT_SUPPORTED,
        }

    return [
        headline_decision, project_decision, skill_decision,
        exp_decision, persona_decision, term_decision,
    ]


def _s4_card_summary(d: dict) -> str:
    status = d.get("status", "keep")
    if status == "keep":
        # 2026-07-21 - keep이어도 evidence(예: 표현 항목의 expression_
        # hits, 자기소개 항목의 matched_concepts)가 있으면 그 근거를
        # 보여준다 - 실제로는 판단 재료가 있었는데 고정 문구가 덮어써서
        # 안 보이던 문제를 수정(실사용 검증 중 발견). evidence가 없는
        # 진짜 "판단할 재료 자체가 없던" 경우만 기존 고정 문구를 쓴다.
        if d.get("evidence"):
            reason = (d.get("reason") or "").strip()
            return reason[:60] + ("…" if len(reason) > 60 else "")
        return _S4_KEEP_TEXT.get(d.get("item", ""), "변경할 실익이 없어 유지했습니다.")
    if status == "manual_review":
        return _S4_REVIEW_TEXT
    reason = (d.get("reason") or "").strip()
    return reason[:60] + ("…" if len(reason) > 60 else "")


def _s4_status_tone(status: str) -> tuple[str, str]:
    """(아이콘/제목 색, 배지 라벨)"""
    color = {"change": "#16A34A", "keep": "#94A3B8", "manual_review": "#D97706"}.get(status, "#94A3B8")
    label = {"change": "변경", "keep": "유지", "manual_review": "직접 확인"}.get(status, status)
    return color, label


def _s4_badge_label(d: dict) -> str:
    """배지 문구 - status가 change이고 count(변경 위치 개수)가 있으면
    "변경 N곳"으로, 없으면(프로젝트/기술 순서처럼 통째로 재배치하는 단일
    판단) 그냥 "변경"으로 보여준다. 새 판단이 아니라 이미 계산된 개수를
    문구에만 반영한다."""
    _, label = _s4_status_tone(d.get("status", "keep"))
    count = d.get("count")
    if d.get("status") == "change" and count:
        return f"{label} {count}곳"
    return label


@st.dialog("상세보기", width="large")
def _s4_card_detail_dialog(d: dict) -> None:
    short_title, icon = _S4_ITEM_META.get(d.get("item", ""), (d.get("item", ""), "structure"))
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
        f'<div style="font-size:18px;font-weight:700;color:#0F172A;">{short_title}</div>'
        f'<span class="s4-badge {d.get("status")}">{_s4_badge_label(d)}</span></div>',
        unsafe_allow_html=True,
    )

    status = d.get("status", "keep")
    if status == "change":
        st.markdown("**변경 이유**")
        st.write(d.get("reason", ""))
        before, after = d.get("before"), d.get("after")
        if before is not None or after is not None:
            bcol, acol = st.columns(2)
            with bcol:
                st.markdown(
                    '<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px;">'
                    '<div style="font-size:12px;color:#64748B;font-weight:700;margin-bottom:6px;">변경 전</div>'
                    f'<div style="font-size:13px;color:#334155;white-space:pre-wrap;">{html.escape(_format_before_after(before))}</div></div>',
                    unsafe_allow_html=True,
                )
            with acol:
                st.markdown(
                    '<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:14px;">'
                    '<div style="font-size:12px;color:#16A34A;font-weight:700;margin-bottom:6px;">변경 후</div>'
                    f'<div style="font-size:13px;color:#166534;white-space:pre-wrap;">{html.escape(_format_before_after(after))}</div></div>',
                    unsafe_allow_html=True,
                )
    elif status == "manual_review":
        st.markdown("**자동 변경하지 않은 이유**")
        st.write(d.get("reason", ""))
    else:
        st.markdown("**유지 이유**")
        if d.get("evidence"):
            st.write(d.get("reason", ""))
        else:
            st.write(_S4_KEEP_TEXT.get(d.get("item", ""), d.get("reason", "")))

    if st.button("닫기", key="s4_detail_close"):
        st.rerun()


def _format_before_after(value) -> str:
    """decision["before"]/["after"]는 항목마다 모양이 다르다(문자열,
    리스트, {"project_order":...,"skill_order":...} 같은 dict 등) - 새
    판단/가공 없이 사람이 읽을 수 있는 텍스트로만 펼쳐 보여준다. 내부
    키 이름은 사용자용 한글 라벨로만 바꿔서 보여준다."""
    _KEY_LABEL = {"skill_order": "기술 순서", "project_order": "프로젝트 순서", "text": "자기소개", "replacements": "표현 치환"}
    if value is None:
        return "(없음)"
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return "(없음)"
        if all(isinstance(v, str) for v in value):
            return " → ".join(value)
        return "\n".join(f"- {v}" for v in value)
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if v is None or k in ("candidate_id", "bullet_reorders"):
                continue
            label = _KEY_LABEL.get(k, k)
            if isinstance(v, list):
                if not v:
                    continue
                v = " → ".join(v) if all(isinstance(x, str) for x in v) else f"{len(v)}건 변경"
            lines.append(f"{label}: {v}")
        return "\n".join(lines) if lines else "(없음)"
    return str(value)


# ── 원본/맞춤 이력서 비교 - 실제 PDF 렌더링 ────────────────────────────
# 2026-08-16(PDF Patch PoC를 S4에 연결, 사용자 확정) - 원문/맞춤화 텍스트를
# HTML로 재구성해서 보여주던 방식(구 _s4_render_compare)을 걷어내고,
# 실제 원본 PDF와 pdf_generator.py가 만든 patched PDF를 그대로 렌더링한다.
# 초록색 하이라이트는 이 렌더링(미리보기 전용 in-memory 사본)에만 그린다 -
# 다운로드/저장되는 실제 PDF 바이트는 절대 건드리지 않는다.

def _s4_render_pdf_pages(pdf_path_or_bytes, changed_fields: list[dict] | None = None, zoom: float = 1.6) -> list[bytes]:
    """PDF의 각 페이지를 PNG로 렌더링한다. changed_fields(pdf_generator.
    generate()가 반환하는 page+bbox 목록)가 있으면 그 위치에만 연한
    초록색 반투명 사각형을 얹는다 - 원본 파일/바이트는 건드리지 않고
    fitz로 새로 연 사본에만 그린 뒤 즉시 버린다(Preview 전용)."""
    if isinstance(pdf_path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)
    by_page: dict[int, list[tuple]] = {}
    for f in changed_fields or []:
        by_page.setdefault(f["page"], []).append(f["bbox"])

    images: list[bytes] = []
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        for bbox in by_page.get(i, []):
            rect = fitz.Rect(bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4)
            page.draw_rect(
                rect, color=(0.06, 0.5, 0.22), fill=(0.13, 0.77, 0.37),
                fill_opacity=0.22, stroke_opacity=0.55, width=1.1,
            )
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def _s4_find_original_template(resume_raw: str):
    """등록된 표준 이력서 템플릿(좌표+원본 PDF 경로)을 지금 이력서의
    resume_hash로 찾는다 - 없으면 None(좌표 미등록 이력서, PDF 미리보기
    자체를 지원하지 않음 - 표준 테스트 이력서 1건만 지원하는 기존 제약,
    이번 작업에서 새로 만들지 않음)."""
    from resume_input import resume_template as _rt
    r_hash = execution_logger.resume_hash(resume_raw)
    template_id = _rt.find_template_id_by_resume_hash(r_hash)
    return _rt.get_template(template_id) if template_id else None


def _s4_original_pdf_source(result: dict):
    """맞춤 전 원본 PDF의 소스 - 등록 템플릿 파일 경로가 있으면 그걸,
    없으면 업로드 시 보관한 원본 PDF 바이트를 반환한다. 둘 다 없으면 None.
    _s4_render_pdf_pages()가 경로/바이트 모두 받는다."""
    template = _s4_find_original_template(result["resume_raw"])
    if template and os.path.exists(template["pdf_path"]):
        return template["pdf_path"]
    return st.session_state.get("resume_pdf_bytes")


def _s4_render_pdf_compare(result: dict, pdf_result: dict | None) -> None:
    """2026-08-16 재설계(사용자 확정) - "실제로 바뀐 게 있을 때만" 원본→
    맞춤 두 패널을 나란히 보여준다. 바뀐 게 없거나(no_change) 자동 반영이
    막혔으면(manual_review) 원본과 똑같은 PDF를 "맞춤화된 이력서"라는
    이름으로 또 보여주는 게 의미가 없다 - 그 경우는 안내 문구 + PDF
    미리보기 1개만 보여준다. HTML로 재구성한 가짜 이력서 비교는 이제
    아예 없다 - 항상 실제 PDF(fitz 렌더링)만 보여준다."""
    orig_src = _s4_original_pdf_source(result)
    status = (pdf_result or {}).get("status")
    changed_fields = (pdf_result or {}).get("changed_fields") or []
    # "reused"(resume_versions 캐시 재사용)는 changed_fields를 새로 안 만들고
    # 넘어온다 - PDF 캐시 히트 여부와 무관하게 "실제로 뭔가 바뀌었는가"는
    # applied(Command 실행 결과, 유일한 진실 소스)로 판단한다.
    applied = result.get("applied") or {}
    any_applied = bool(
        applied.get("project_order") or applied.get("skill_order")
        or applied.get("bullet_reorders") or applied.get("term_replacements")
        or applied.get("headline")
    )
    has_real_change = status in ("ok", "reused") and any_applied and pdf_result.get("pdf_path") and os.path.exists(pdf_result["pdf_path"])

    if has_real_change:
        change_count_text = f"연한 초록색 영역 {len(changed_fields)}곳이 실제로 반영된 변경입니다." if changed_fields else "실제로 반영된 변경만 연한 초록색으로 표시됩니다."
        st.markdown('<div class="s4-section-title">맞춤화된 이력서</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="s4-section-sub">이 공고에 맞춰 필요한 부분만 수정했습니다 - {change_count_text}</div>',
            unsafe_allow_html=True,
        )
        col_orig, col_arrow, col_new = st.columns([1, 0.08, 1])
        with col_orig:
            st.markdown('<div class="s4-compare-header original">' + _s4_icon("file-text", "#334155", 16) + '&nbsp;&nbsp;원본</div>', unsafe_allow_html=True)
            if orig_src is not None:
                for png in _s4_render_pdf_pages(orig_src):
                    st.image(png, use_container_width=True)
            else:
                st.info("원본 PDF를 불러올 수 없습니다. 이력서를 다시 업로드해 주세요.")
        with col_arrow:
            st.markdown(f'<div class="s4-arrow-circle">{_s4_icon("arrow-right", "#1E3A8A", 18)}</div>', unsafe_allow_html=True)
        with col_new:
            badge_text = f"변경 {len(changed_fields)}곳 반영" if changed_fields else "변경 반영됨"
            st.markdown(
                '<div class="s4-compare-header customized">'
                f'{_s4_icon("file-check", "#15803D", 16)}&nbsp;&nbsp;맞춤화된 이력서'
                f'<span class="s4-badge change">{badge_text}</span></div>',
                unsafe_allow_html=True,
            )
            for png in _s4_render_pdf_pages(pdf_result["pdf_path"], changed_fields):
                st.image(png, use_container_width=True)
        st.caption("다운로드하는 최종 PDF에는 초록색 하이라이트가 들어가지 않습니다 - 화면 확인용입니다.")
        return

    # 변경이 없거나(no_change) 자동 반영이 막힌 경우(manual_review) - 단일 미리보기만.
    st.markdown('<div class="s4-section-title">이력서</div>', unsafe_allow_html=True)
    if status == "no_change":
        # 2026-08-16(버그 수정) - "no_change"는 PDF에 patch할 게 없다는
        # 뜻일 뿐, 판단 카드에서 프로젝트 순서/기술 순서가 "변경"으로
        # 나온 경우에도 나올 수 있다(_pdf_safe_commands가 move_project/
        # reorder_skills를 걸러내므로) - 그 경우 "수정할 필요가 있는
        # 항목을 찾지 못했다"는 문구는 사실과 다르다(6-슬롯 카드가 이미
        # "변경"을 보여주고 있음). any_applied가 있으면 "제안은 있지만
        # PDF 자동 반영 범위 밖"이라고 정확히 알린다.
        if any_applied:
            st.warning("이 공고를 위한 변경 제안이 있지만, 프로젝트 순서(블록 통째 이동)는 현재 PDF에 자동 반영되지 않습니다. 위 판단 카드의 내용을 참고해 직접 반영해주세요.")
        else:
            st.success("현재 이력서를 그대로 사용해도 좋습니다. 이 공고를 위해 별도로 수정할 필요가 있는 항목을 찾지 못했습니다.")
    elif status == "manual_review":
        st.warning(f"일부 변경이 자동 반영되지 않았습니다: {pdf_result.get('reason', '')} 위 판단 카드의 변경 제안을 참고해 직접 반영해주세요.")
    elif status == "error":
        st.error((pdf_result or {}).get("reason", "원본 PDF를 처리하지 못했습니다. 이력서를 다시 업로드해 주세요."))
    elif status not in ("ok", "reused"):
        st.info("PDF 미리보기를 준비하는 중입니다.")

    if orig_src is not None:
        for png in _s4_render_pdf_pages(orig_src):
            st.image(png, use_container_width=True)


@st.dialog("수정된 항목", width="large")
def _s4_changes_panel_dialog(decisions: list[dict]) -> None:
    changed = [d for d in decisions if d.get("status") == "change"]
    st.caption("이번 공고에 맞춰 실제로 변경된 내용만 모아봤습니다.")
    if not changed:
        st.markdown(
            f'<div class="s4-empty">{_s4_icon("circle-check", "#16A34A", 32)}'
            '<div style="margin-top:10px;font-weight:600;color:#0F172A;">변경된 항목이 없습니다.</div>'
            '<div style="margin-top:4px;font-size:13px;">현재 이력서가 이 공고에 맞는 구조를 이미 갖추고 있어 원본 버전을 그대로 사용할 수 있습니다.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**총 {len(changed)}개 항목 변경**")
        for d in changed:
            short_title, icon = _S4_ITEM_META.get(d.get("item", ""), (d.get("item", ""), "structure"))
            st.markdown(
                f'<div class="s4-changeitem"><div class="s4-changeitem-title">{_s4_icon(icon, "#16A34A", 16)}&nbsp;{short_title}</div>'
                f'<div class="s4-changeitem-desc">{html.escape(_s4_card_summary(d))}</div></div>',
                unsafe_allow_html=True,
            )
    if st.button("닫기", key="s4_changes_panel_close"):
        st.rerun()


# ── S4 확장: ① 자기소개 제안(2026-08-16, 사용자 확정) ─────────────────
# customization_planner.decide_headline()이 "평가 필요"로 넘긴 경우에만
# 보여준다. LLM은 버튼을 눌렀을 때만 1회 호출한다(LLM On-Demand 원칙
# 유지 - 페이지 진입만으로 자동 호출 안 함). 자동 적용 없음 - 사용자가
# [원문 유지]/[추천안 적용] 중 직접 고른다(사용자 확정: "추천안을 자동
# 적용하지 않는 것 자체가 최종 안전장치"). 자동 검증(숫자 보존/근거
# 실재/최소 변경 유사도)에 실패한 제안은 애초에 "추천안 적용" 선택지
# 자체를 막는다.

def _sync_headline_command(result: dict, headline_text: str | None) -> None:
    """result["resume_commands"](= st.session_state.apply_result와 같은
    객체, PDF 생성이 그대로 재사용)에 replace_headline Command를 반영하고,
    바뀌었으면 캐시된 PDF 미리보기를 무효화해 다음 렌더에서 다시 만들게
    한다. Command 목록이 유일한 진실 소스라는 원칙을 그대로 따른다 -
    텍스트를 따로 들고 다니지 않는다."""
    commands = result["resume_commands"]
    existing = next((c for c in commands if c["type"] == "replace_headline"), None)
    existing_text = existing.get("text") if existing else None
    if existing_text == headline_text:
        return  # 이미 같은 상태 - PDF 재생성 불필요

    commands[:] = [c for c in commands if c["type"] != "replace_headline"]
    if headline_text:
        commands.append({"type": "replace_headline", "text": headline_text, "priority": 1})
    st.session_state.resume_pdf_result = None  # 다음 렌더에서 새 Command로 재생성


def _render_headline_section(result: dict) -> None:
    """자기소개 제안 UI - Command 목록을 직접 갱신한다(반환값 없음).
    호출부는 이 함수 다음에 PDF를 생성/재사용해야 한다."""
    job = result.get("job") or {}
    job_id = str(job.get("job_id"))
    headline_note = (result.get("planner_notes") or {}).get("headline") or {}

    if headline_note.get("decision") != "평가 필요":
        return

    st.markdown('<div class="s4-section-title">자기소개 제안</div>', unsafe_allow_html=True)
    st.markdown('<div class="s4-section-sub">이 공고가 강조하는 인재상 중 실제 근거가 있는 부분만 반영한 제안입니다 - 자동 적용되지 않습니다.</div>', unsafe_allow_html=True)

    proposals_cache = st.session_state.headline_proposal_by_job
    if job_id not in proposals_cache:
        if st.button("자기소개 제안 받기", key=f"headline_gen_{job_id}"):
            with st.spinner("제안을 만드는 중입니다..."):
                try:
                    proposal = rewrite_engine.generate_headline_proposal(
                        headline_note, result["resume_semantic_objects"],
                    )
                    proposals_cache[job_id] = proposal
                    st.session_state.headline_choice_by_job[job_id] = proposal["default_choice"]
                    st.rerun()
                except LLMCallError as e:
                    show_llm_error("자기소개 제안 실패", e)
        st.space(20)
        return

    proposal = proposals_cache[job_id]
    if proposal.get("status") == "no_evidence":
        # 2026-08-16(Planner/Writer 분리) - "근거 자체가 없음"과 "근거는
        # 있지만 REWRITE할 만큼 실질적인 의미 추가가 아님"(예: 수식어
        # 하나만 붙일 수 있는 경우)이 둘 다 KEEP으로 이어질 수 있다 -
        # 고정 문구 대신 Planner가 실제로 낸 이유를 보여준다.
        st.caption(proposal.get("reason") or "이 공고가 강조하는 인재상을 뒷받침할 실제 근거를 이력서에서 찾지 못해 자기소개를 그대로 둡니다.")
        st.space(20)
        return
    if proposal.get("status") == "critic_rejected":
        # 2026-08-16(Planner/Writer/Critic 분리) - Code Verifier는 통과했지만
        # Critic이 "억지 맞춤/부자연스러움"으로 거부한 경우 - no_evidence와
        # 같은 톤으로 조용히 원문 유지만 알린다(옆에 똑같은 문장 두 개를
        # 나란히 보여주는 건 의미가 없다).
        st.caption("공고 인재상에 맞춰 문장을 만들어봤지만, 억지로 끼워 맞춘 느낌이 있어 원문을 그대로 둡니다.")
        st.space(20)
        return

    can_apply = proposal.get("status") == "proposed"
    with st.container(border=True):
        ccol, rcol = st.columns(2)
        with ccol:
            st.markdown("**현재**")
            st.write(proposal["current_headline"])
        with rcol:
            st.markdown("**추천**")
            st.write(proposal["recommended_headline"])
            if not can_apply:
                st.caption("⚠ 자동 검증 실패 - 적용할 수 없습니다(원문 유지).")
        if proposal.get("reason"):
            st.caption(f"근거: {proposal['reason']}")

        choice_state = st.session_state.headline_choice_by_job
        current_choice = choice_state.get(job_id, proposal["default_choice"])
        if can_apply:
            new_choice = st.radio(
                "적용 여부", ["원문 유지", "추천안 적용"],
                index=0 if current_choice == "current" else 1,
                key=f"headline_choice_{job_id}", horizontal=True, label_visibility="collapsed",
            )
            choice_state[job_id] = "current" if new_choice == "원문 유지" else "recommended"
        else:
            choice_state[job_id] = "current"

    st.space(20)

    chosen = choice_state.get(job_id, "current")
    if chosen == "recommended" and can_apply:
        _sync_headline_command(result, proposal["recommended_headline"])
    else:
        _sync_headline_command(result, None)


# ── S4 확장: Rewrite(2026-08-14, 신규 / 2026-08-16 화면에서 분리) ──────
# selection_engine.py -> composition_engine.py를 거쳐 "이번 공고에서
# 먼저 보여줄 경험 순서" 섹션을 그렸던 코드는 여기서 제거했다(2026-08-16,
# 사용자 확정) - Customization Planner(②)가 이미 같은 질문을 Hard
# Eligibility + 레이어 라우팅 + 최소변경 원칙으로 판단하는데, 이 섹션은
# raw link_result를 별도로 재정렬해서 같은 화면에서 다른 결론을 보여주는
# 모순이 실측 확인됐다. selection_engine.py 자체는 지우지 않는다 -
# cover_letter_engine.py(S6)가 별도 목적으로 재사용한다.

def _render_rewrite_section(result: dict) -> None:
    job = result.get("job") or {}
    job_id = str(job.get("job_id"))
    link_result = job.get("_semantic_match")
    if not link_result or not link_result.get("links"):
        return
    # Selection/Composition이 이미 반영한 순서(reposition) 위에 표현만
    # 다듬는다 - resume_raw가 아니라 resume_customized를 기준으로 삼는다.
    working_text = result["resume_customized"]
    resume_semantic_objects = result["resume_semantic_objects"]

    st.markdown('<div class="s4-section-title">문장 다듬기 제안</div>', unsafe_allow_html=True)
    st.markdown('<div class="s4-section-sub">기존 사실만 사용해 표현만 다듬은 제안입니다 - 새 경험/숫자는 만들지 않습니다.</div>', unsafe_allow_html=True)

    proposals_cache = st.session_state.rewrite_proposals_by_job
    if job_id not in proposals_cache:
        targets = rewrite_engine.collect_rewrite_targets(link_result, resume_semantic_objects)
        if not targets:
            st.caption("이번 공고에서 표현을 다듬을 대상이 없습니다.")
            st.space(20)
            return
        if st.button(f"문장 다듬기 제안 받기 ({len(targets)}건)", key=f"rewrite_gen_{job_id}"):
            with st.spinner("문장을 다듬는 중입니다..."):
                try:
                    proposals = rewrite_engine.generate_rewrite_proposals(link_result, resume_semantic_objects)
                    proposals_cache[job_id] = proposals
                    st.session_state.rewrite_decisions_by_job[job_id] = {
                        p["target_resume_object_id"]: p["default_choice"] for p in proposals
                    }
                    st.rerun()
                except LLMCallError as e:
                    show_llm_error("문장 다듬기 실패", e)
        st.space(20)
        return

    proposals = proposals_cache[job_id]
    review = rewrite_engine.build_rewrite_review(proposals)
    decisions_state = st.session_state.rewrite_decisions_by_job.setdefault(job_id, {
        p["target_resume_object_id"]: p["default_choice"] for p in proposals
    })

    for item in review:
        tid = item["target_resume_object_id"]
        with st.container(border=True):
            st.caption(f"{item['project']} · {item['section']}")
            ccol, rcol = st.columns(2)
            with ccol:
                st.markdown("**현재 문장**")
                st.write(item["current_text"])
            with rcol:
                st.markdown("**추천 문장**")
                st.write(item["recommended_text"])
                if not item["can_apply"]:
                    st.caption("⚠ 자동 검증 실패 - 적용할 수 없습니다(원문 유지).")
            if item["reason"]:
                st.caption(f"수정 이유: {item['reason']}")

            if item["can_apply"]:
                choice = decisions_state.get(tid, "current")
                new_choice = st.radio(
                    "적용 여부", ["원문 유지", "추천안 적용"],
                    index=0 if choice == "current" else 1,
                    key=f"rewrite_choice_{job_id}_{tid}", horizontal=True, label_visibility="collapsed",
                )
                decisions_state[tid] = "current" if new_choice == "원문 유지" else "recommended"
            else:
                decisions_state[tid] = "current"

    applicable = [r for r in review if r["can_apply"]]
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if applicable and st.button("검증 통과한 제안 전체 적용", key=f"rewrite_apply_all_{job_id}"):
            for r in applicable:
                decisions_state[r["target_resume_object_id"]] = "recommended"
            st.rerun()
    with bcol2:
        if st.button("이 상태로 저장", key=f"rewrite_save_{job_id}", type="primary"):
            decisions_payload = [
                {
                    "target_resume_object_id": p["target_resume_object_id"],
                    "project": p["project"],
                    "original_text": p["original_text"],
                    "chosen_text": p["recommended_text"] if decisions_state.get(p["target_resume_object_id"]) == "recommended" else p["original_text"],
                }
                for p in proposals
            ]
            final_text, applied = rewrite_engine.apply_rewrite_decisions(working_text, decisions_payload)
            r_hash = execution_logger.resume_hash(result["resume_raw"])
            saved = rewrite_versions.save_rewrite_version(r_hash, job_id, working_text, final_text, decisions_payload)
            st.session_state.rewrite_result_by_job[job_id] = {"text": final_text, "applied": applied, "version_id": saved["id"]}
            st.success(f"저장했습니다 (반영 {len(applied)}건). 원본 이력서는 그대로 보존됩니다.")

    saved_result = st.session_state.rewrite_result_by_job.get(job_id)
    if saved_result:
        with st.expander("저장된 최종 텍스트 보기 / 다운로드"):
            st.text_area("공고 맞춤 + 문장 다듬기 반영본", saved_result["text"], height=300, key=f"rewrite_final_text_{job_id}")
            st.download_button(
                "텍스트 다운로드", saved_result["text"], file_name="이력서_맞춤화_최종.txt",
                key=f"rewrite_download_{job_id}",
            )

    st.space(20)


def _s4_chip_flow(items: list[str]) -> str:
    """[SQL] → [Python] → ... 형태의 chip flow HTML(_S4_CSS 스코프)."""
    items = [str(x).strip() for x in (items or []) if str(x).strip()]
    if not items:
        return "<div class='s4-note'>표시할 항목이 없습니다.</div>"
    parts = []
    for i, it in enumerate(items):
        if i:
            parts.append("<span class='s4-chip-arrow'>→</span>")
        parts.append(f"<span class='s4-chip'>{html.escape(it)}</span>")
    return "<div class='s4-chip-row'>" + "".join(parts) + "</div>"


def _s4_job_card(job: dict, job_core: str) -> None:
    """선택된 공고 얇은 헤더 - 제목+회사 한 줄 / DB에 실제 있는 메타(지역/
    경력/산업) 한 줄 / 공고 원문 보기. job_core 설명은 넣지 않는다(분석
    화면에서 이미 봄). 없는 값은 추측 안 함."""
    company = job.get("company") or ""
    title = job.get("title") or ""
    url = job.get("url") or ""
    # 얇은 헤더라 지역은 시·구 수준까지만(상세 주소는 원문에서).
    _loc = (job.get("location") or "").strip()
    _loc_short = " ".join(_loc.split()[:2]) if len(_loc.split()) > 2 else _loc
    meta = [m for m in [
        _loc_short,
        (job.get("industry") or "").strip(),
        (job.get("career_level") or "").strip(),
    ] if m]
    with st.container(border=True, key="s4-jobhead"):
        head = st.columns([5, 1.4], vertical_alignment="center")
        with head[0]:
            _co = f"<span class='s4-jobcard-company'>{html.escape(company)}</span>" if (title and company) else ""
            st.markdown(f"<div class='s4-jobcard-title'>{html.escape(title or company)}{_co}</div>", unsafe_allow_html=True)
            if meta:
                st.markdown(f"<div class='s4-jobcard-meta'>{html.escape(' · '.join(meta))}</div>", unsafe_allow_html=True)
        with head[1]:
            if url:
                st.link_button("공고 원문 보기", url, use_container_width=True)


def _s4_pdf_download(result: dict, skill_priority: list | None = None,
                     project_priority: list | None = None) -> None:
    """맞춤 이력서 다운로드.

    ① 개인 표준 이력서 → HTML 경로(resume_input.html_resume): 기술 순서 +
       프로젝트 순서를 모두 자동 반영. 기존 엔진(pipeline/pdf_layout) 미사용.
       Chrome 없음 등으로 실패하면 아래 ②로 폴백한다.
    ② 그 외 이력서 → 기존 범용 PDF 경로(generate_resume_pdf) 그대로.
       기술 순서만 반영, 프로젝트 순서는 반영 안 함. pdf_layout.py 미수정."""
    from resume_input import html_resume

    if html_resume.is_target_resume(
        result.get("resume_raw") or "",
        execution_logger.resume_hash(result.get("resume_raw") or ""),
    ) and st.session_state.get("html_resume_failed") is not True:
        if st.session_state.get("html_resume_result") is None:
            with st.spinner("맞춤 이력서를 준비하는 중입니다..."):
                st.session_state.html_resume_result = html_resume.generate(
                    skill_priority or result.get("skill_priority") or [],
                    project_priority or result.get("project_priority") or [],
                )
        hr = st.session_state.html_resume_result or {}
        if hr.get("status") == "ok" and hr.get("pdf_path") and os.path.exists(hr["pdf_path"]):
            with open(hr["pdf_path"], "rb") as f:
                st.download_button(
                    "↓  맞춤 이력서 다운로드", f.read(), file_name="이력서_맞춤화.pdf",
                    mime="application/pdf", key="s4_download_final", type="primary",
                    use_container_width=True,
                )
            return
        # HTML 경로 실패 → 이 세션에선 재시도 안 하고 기존 범용 경로로 폴백.
        st.session_state.html_resume_failed = True
        print(f"[html_resume 폴백] {hr.get('reason')}")

    if st.session_state.get("resume_pdf_result") is None:
        with st.spinner("맞춤 이력서를 준비하는 중입니다..."):
            st.session_state.resume_pdf_result = pipeline.generate_resume_pdf(
                result["resume_raw"], result["resume_commands"],
                result["resume_semantic_objects"], result.get("term_replacements"),
                resume_pdf_bytes=st.session_state.get("resume_pdf_bytes"),
            )
    pdf_result = st.session_state.resume_pdf_result or {}
    status = pdf_result.get("status")
    if status in ("ok", "reused") and pdf_result.get("pdf_path") and os.path.exists(pdf_result["pdf_path"]):
        with open(pdf_result["pdf_path"], "rb") as f:
            st.download_button(
                "↓  맞춤 이력서 다운로드", f.read(), file_name="이력서_맞춤화.pdf",
                mime="application/pdf", key="s4_download_final", type="primary",
            )
    elif status == "no_change":
        # 기술 순서 변경이 없어(이미 그 순서) PDF 를 새로 만들 필요가 없는 경우 -
        # 원본을 그대로 내려받게 한다.
        src = _s4_original_pdf_source(result)
        data = None
        if isinstance(src, (bytes, bytearray)):
            data = bytes(src)
        elif src:
            try:
                data = open(src, "rb").read()
            except OSError:
                data = None
        if data:
            st.download_button(
                "↓  이력서 다운로드", data, file_name="이력서.pdf",
                mime="application/pdf", key="s4_download_orig", type="primary",
                use_container_width=True,
            )
            st.markdown("<div class='s4-note'>기술 순서가 이미 이 공고에 맞게 되어 있어 원본을 그대로 사용합니다.</div>", unsafe_allow_html=True)
        else:
            st.button("맞춤 이력서 다운로드", key="s4_dl_none", type="primary", disabled=True, use_container_width=True)
    elif status == "manual_review":
        st.button("맞춤 이력서 다운로드", key="s4_dl_mr", type="primary", disabled=True, use_container_width=True)
        st.markdown("<div class='s4-note'>일부 변경이 자동 반영되지 않았습니다. 이력서를 다시 업로드하거나 직접 순서를 조정해 주세요.</div>", unsafe_allow_html=True)
    else:
        st.button("맞춤 이력서 다운로드", key="s4_dl_err", type="primary", disabled=True, use_container_width=True)
        st.markdown(f"<div class='s4-note'>{html.escape(pdf_result.get('reason') or '맞춤 PDF를 준비하지 못했습니다. 이력서를 다시 업로드해 주세요.')}</div>", unsafe_allow_html=True)


def _s4_portfolio_download_button(project_priority: list | None) -> None:
    """맞춤 포트폴리오(PDF) 다운로드 버튼만 - 개인 포트폴리오 원본(.pptx)이
    로컬에 있을 때 전용. 이력서 다운로드 버튼 옆에 나란히 놓는다(별도
    섹션/카드/설명 없음, 사용자 확정 2026-09-02). 같은 project_priority 로
    About 카드·슬라이드 블록·PROJECT 번호를 결정적으로 재배치(LLM 0회,
    문구 불변)한 뒤 PDF 로 변환한다. 원본 .pptx가 없으면(공개 데모 기본
    상태) 안내 문구만 보여주고 버튼은 비활성화한다(이력서 다운로드엔
    영향 없음)."""
    from resume_input import portfolio_pptx

    if st.session_state.get("portfolio_pptx_result") is None:
        with st.spinner("맞춤 포트폴리오를 준비하는 중입니다..."):
            st.session_state.portfolio_pptx_result = portfolio_pptx.generate(project_priority or [])
    pr = st.session_state.portfolio_pptx_result or {}
    if pr.get("status") != "ok" or not (pr.get("pdf_path") and os.path.exists(pr["pdf_path"])):
        reason = pr.get("reason") or "맞춤 포트폴리오를 준비하지 못했습니다."
        print(f"[portfolio 생략] {reason}")
        st.button("맞춤 포트폴리오 다운로드", key="s4_dl_portfolio_err", type="secondary",
                   disabled=True, use_container_width=True)
        st.markdown(f"<div class='s4-note'>{html.escape(reason)}</div>", unsafe_allow_html=True)
        return

    with open(pr["pdf_path"], "rb") as f:
        st.download_button(
            "↓  맞춤 포트폴리오 다운로드", f.read(), file_name="포트폴리오.pdf",
            mime="application/pdf", key="s4_download_portfolio", type="secondary",
            use_container_width=True,
        )


def render_s4() -> None:
    """지원 준비(2026-08-31 재구성, 사용자 확정) - 구 "이력서 맞춤/커스터마이징"
    화면을 실제 남은 기능 규모에 맞게 축소했다. 새 기능/새 LLM/새 PDF 기능
    없음. 이력서 준비(기술 순서 자동 반영 + 프로젝트 순서 추천 + 맞춤 PDF) +
    자소서 준비(선택)만. Before/After 전체 비교, 적합도/역량 점수, 6개 변경
    카드, 자기소개 수정, 용어 치환, 불릿 재작성 등은 전부 제거(함수 정의는
    dead code 로 남김)."""
    result = st.session_state.apply_result
    if result is None:
        st.warning("먼저 공고 분석을 완료해주세요.")
        back_button()
        return

    job = result.get("job") or {}
    job_id = str(job.get("job_id") or "")
    _log_prep_event_once(job.get("job_id"), "customization_viewed")

    st.markdown(_S4_CSS, unsafe_allow_html=True)
    st.markdown('<div class="s4-wrap">', unsafe_allow_html=True)
    st.markdown(step_progress_html(3), unsafe_allow_html=True)  # 4단계: 지원 준비
    back_button()

    # 2026-09-03(사용자 확정) - "안녕하세요, 게스트님" 인사말 + 부제 삭제
    # (단계 바 "지원 준비" 라벨이면 충분).

    qa = job.get("_quick_analysis") or {}
    _s4_job_card(job, qa.get("job_core") or "")

    st.space(24)

    skill_priority = result.get("skill_priority") or (qa.get("resume_order") or {}).get("skill_priority") or []
    project_priority = result.get("project_priority") or (qa.get("resume_order") or {}).get("project_priority") or []

    # 개인 표준 이력서면 HTML 경로라 프로젝트 순서까지 PDF 자동 반영된다.
    from resume_input import html_resume as _hr
    _proj_auto = _hr.is_target_resume(result.get("resume_raw") or "")

    # ── 이력서 준비 (메인) ──
    with st.container(border=True):
        st.markdown('<div class="s4-section-title">이력서 준비</div>', unsafe_allow_html=True)
        st.space(8)

        h = st.columns([3, 1.4], vertical_alignment="center")
        h[0].markdown("**기술 순서**")
        h[1].markdown("<div style='text-align:right'><span class='s4-ok-badge'>자동 반영 ✓</span></div>", unsafe_allow_html=True)
        st.markdown(_s4_chip_flow(skill_priority), unsafe_allow_html=True)
        st.markdown("<div class='s4-note'>위 기술 순서는 맞춤 이력서 PDF에 자동으로 반영됩니다.</div>", unsafe_allow_html=True)

        st.divider()

        h2 = st.columns([3, 1.4], vertical_alignment="center")
        h2[0].markdown("**프로젝트 순서**")
        if _proj_auto:
            h2[1].markdown("<div style='text-align:right'><span class='s4-ok-badge'>자동 반영 ✓</span></div>", unsafe_allow_html=True)
            st.markdown(_s4_chip_flow(project_priority), unsafe_allow_html=True)
            st.markdown("<div class='s4-note'>위 프로젝트 순서는 맞춤 이력서 PDF와 맞춤 포트폴리오에 자동으로 반영됩니다.</div>", unsafe_allow_html=True)
        else:
            h2[1].markdown("<div style='text-align:right'><span class='s4-hint-badge'>변경 추천</span></div>", unsafe_allow_html=True)
            st.markdown(_s4_chip_flow(project_priority), unsafe_allow_html=True)
            st.markdown("<div class='s4-note'>※ 프로젝트 순서는 PDF에 자동 반영되지 않습니다. 직접 조정해 주세요.</div>", unsafe_allow_html=True)

        st.space(16)
        # 개인 표준 이력서면 이력서·포트폴리오 다운로드 버튼을 나란히
        # (별도 "포트폴리오 준비" 카드/설명 없음 - 2026-09-02 사용자 확정).
        if _proj_auto:
            _dl = st.columns(2)
            with _dl[0]:
                _s4_pdf_download(result, skill_priority, project_priority)
            with _dl[1]:
                _s4_portfolio_download_button(project_priority)
        else:
            _s4_pdf_download(result, skill_priority, project_priority)

    st.space(20)

    # ── 자소서 준비 (선택) ──
    with st.container(border=True):
        c = st.columns([4, 1], vertical_alignment="center")
        c[0].markdown('<div class="s4-section-title">자소서 준비</div>', unsafe_allow_html=True)
        c[1].markdown("<div style='text-align:right'><span class='s4-hint-badge'>선택</span></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='s4-note'>필요한 경우 공고와 이력서를 바탕으로 지원동기와 관련 경험 소재를 정리할 수 있습니다.</div>",
            unsafe_allow_html=True,
        )
        st.space(10)
        if st.button("자소서 준비하기  →", key="s4_goto_coverletter"):
            go("S6")

    st.space(20)

    # 실제 지원 기록으로 - T2(customization_confirmed) 는 여기서 확정한다.
    _s4_ready = st.session_state.setdefault("_s4_ready", set())
    if job_id and _navy_cta("이 이력서로 지원하기  →", "s4_apply"):
        if job_id not in _s4_ready:
            _log_prep_event_once(job_id, "customization_confirmed")
            _s4_ready.add(job_id)
        go("S8")

    st.markdown("</div>", unsafe_allow_html=True)



# ── S6. 자소서 준비(2026-08-16 2차 재설계, 사용자 확정) ────────────────
# "검색/정리 결과"(problem/purpose/situation/action을 그대로 나열)가
# 아니라 "실제로 자소서에 쓸 수 있는 소재+초안"을 보여주는 Writing UI로
# 바꾼다. A/B/C/D, 강/중/약, JD 요구사항 목록, 매칭 근거 목록 같은 분석
# UI는 여기서 노출하지 않는다(공고 분석 화면의 몫) - build_motivation_
# source()/build_related_experience_source()가 만든 사실 재료는 화면에
# 직접 안 보여주고, cover_letter_engine.generate_composition()(LLM 1회,
# 소재+초안 동시 생성)의 입력으로만 쓴다. LLM은 버튼을 눌렀을 때만
# 호출한다(LLM On-Demand 원칙 유지).

def render_s6() -> None:
    job = st.session_state.selected_job
    result = st.session_state.apply_result
    if job is None or result is None:
        st.warning("먼저 분석/맞춤화 단계를 완료해주세요.")
        back_button()
        return

    _log_prep_event_once(job.get("job_id"), "cover_letter_source_viewed")

    link_result = job.get("_semantic_match")
    if not link_result or not link_result.get("links"):
        st.warning("이 공고는 의미 기반 분석 결과가 없어 자소서 소재를 준비할 수 없습니다.")
        back_button()
        return

    st.markdown(_S4_CSS, unsafe_allow_html=True)
    st.markdown('<div class="s4-wrap">', unsafe_allow_html=True)
    back_button()
    st.markdown('<div class="s4-title">자소서 준비</div>', unsafe_allow_html=True)
    st.markdown('<div class="s4-subtitle">공고와 이력서에서 활용할 수 있는 소재를 먼저 확인합니다. 초안은 필요할 때만 생성합니다.</div>', unsafe_allow_html=True)
    st.caption(f"{job.get('company','')} · {job.get('title','')}")

    motivation_source = cover_letter_engine.build_motivation_source(job)
    resume_raw = result["resume_raw"]
    resume_semantic_objects = result["resume_semantic_objects"]
    jd_semantic_objects = job.get("_qa_jd_objects") or json.loads(job.get("jd_semantic_objects") or "[]")
    judge_result = judge_engine.judge(link_result, jd_semantic_objects, _get_resume_facts())
    experience_source = cover_letter_engine.build_related_experience_source(
        link_result, judge_result, resume_raw, resume_semantic_objects,
        quick_analysis=job.get("_quick_analysis"),
    )

    job_id = str(job.get("job_id"))
    composition_cache = st.session_state.composition_by_job

    if job_id not in composition_cache:
        if not motivation_source.get("has_content") and not experience_source:
            st.info("이 공고에서 지원동기·직무 경험 모두에 쓸 근거를 찾지 못했습니다.")
            st.space(20)
        elif st.button("자소서 소재·초안 준비하기", key=f"composition_gen_{job_id}", type="primary"):
            with st.spinner("소재와 초안을 준비하는 중입니다..."):
                try:
                    composition = cover_letter_engine.generate_composition(motivation_source, experience_source)
                    composition_cache[job_id] = composition
                    st.rerun()
                except LLMCallError as e:
                    show_llm_error("자소서 준비 실패", e)
            st.space(20)
        else:
            st.space(20)
        if job_id not in composition_cache:
            _, mid, _ = st.columns([1, 1, 1])
            with mid:
                if st.button("다음: 최종 지원 →", key="s6_next_apply_early", use_container_width=True):
                    go("S8")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    composition = composition_cache[job_id]
    motivation_out = composition.get("motivation")
    experience_out = composition.get("experience")

    if motivation_out and motivation_out.get("draft"):
        st.markdown('<div class="s4-section-title">지원동기</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**{motivation_out.get('title', '')}**")
            if motivation_out.get("material"):
                st.write(motivation_out["material"])
            if motivation_out.get("writing_point"):
                st.caption("활용 포인트: " + motivation_out["writing_point"])
            with st.expander("초안 보기"):
                st.write(motivation_out.get("draft", ""))
        st.space(16)

    if experience_out and experience_out.get("draft"):
        st.markdown('<div class="s4-section-title">관련 직무 경험</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**{experience_out.get('title', '')}**")
            if experience_out.get("situation"):
                st.write("**상황/문제** · " + experience_out["situation"])
            if experience_out.get("action"):
                st.write("**판단/행동** · " + experience_out["action"])
            if experience_out.get("result"):
                st.write("**결과** · " + experience_out["result"])
            if experience_out.get("writing_point"):
                st.caption("살릴 부분: " + experience_out["writing_point"])
            with st.expander("초안 보기"):
                st.write(experience_out.get("draft", ""))
        st.space(16)

    if not (motivation_out and motivation_out.get("draft")) and not (experience_out and experience_out.get("draft")):
        st.info("이 공고에서 실제로 쓸 수 있는 소재를 찾지 못했습니다 - 근거 없는 내용을 지어내지 않았습니다.")

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("다음: 최종 지원 →", key="s6_next_apply", use_container_width=True):
            go("S8")

    st.markdown("</div>", unsafe_allow_html=True)


# ── S8. 최종 지원 화면 ─────────────────────────────────────────────────

def render_s8() -> None:
    job = st.session_state.selected_job
    result = st.session_state.apply_result
    if job is None or result is None:
        st.warning("먼저 분석 단계를 완료해주세요.")
        back_button()
        return

    st.markdown(step_progress_html(4), unsafe_allow_html=True)  # 5단계: 지원 기록(제출)
    page_header(svg_icon("send", 40, "var(--navy)"), "최종 지원")
    back_button()

    with st.container(border=True):
        st.subheader(f"{job.get('company','')} - {job.get('title','')}")
        st.caption(f"출처: {job.get('source','기업 홈페이지')}")

        st.markdown(badge_html("커스터마이징 완료", "green"), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    if job.get("url"):
        col1.link_button("공고 열기", job["url"], use_container_width=True)
    else:
        # 추천 목록의 공고는 항상 url이 있다 - 여기서 없다는 건 "공고
        # 붙여넣기 모드"(S2)에서 링크 없이 본문만 입력한 경우뿐이다.
        col1.caption("⚠ 직접 붙여넣은 공고라 원문 링크가 없습니다.")
    if col2.button("지원 완료", type="primary", use_container_width=True):
        try:
            resume_version = (st.session_state.get("resume_pdf_result") or {}).get("version_id")
            if not resume_version and (st.session_state.get("html_resume_result") or {}).get("status") == "ok":
                # 개인 HTML 맞춤 경로는 resume_versions 에 버전을 만들지 않는다
                # (pdf_layout 미사용). "맞춤 이력서로 지원했다"는 사실만 남긴다 -
                # _s7_detail_dialog 의 resume_versions.get_version() 은 없는 id 면
                # 조용히 None(="맞춤 이력서 기록 없음")이라 안전하고, KPI 표의
                # '맞춤 이력서' 컬럼은 이 값의 존재 여부로 사용/미사용을 가른다.
                resume_version = "html-resume-v1"
            pipeline.finalize_application(job, st.session_state.resume_raw, resume_version=resume_version)
            # 추천 목록 캐시는 비우지 않는다(2026-09-01) - 방금 지원한 공고는
            # _render_s1_results 가 화면 직전에 list_applied_* 로 걸러내므로,
            # 목록을 통째로 재계산(1~2분)할 이유가 없다. "지원 → 다시 공고
            # 찾기" 반복에서 매번 대기가 걸리던 원인이 이 한 줄이었다.
            go("S10")
        except Exception as e:
            print(f"[지원 완료 처리 오류] {e}")
            st.error("지원 완료 처리 중 오류가 발생했습니다.")


# ── S10. 지원 완료 ─────────────────────────────────────────────────────

def render_s10() -> None:
    st.markdown(_ANALYSIS_CSS, unsafe_allow_html=True)
    job = st.session_state.selected_job
    page_header(svg_icon("check-circle", 40, "var(--navy)"), "지원 완료")

    company = (job or {}).get("company", "")
    title = (job or {}).get("title", "")

    # 2026-09-03(사용자 확정) - 완료 카드 세로 여유 확대(아이콘·문구 그대로, 카드만 큼).
    with st.container(border=True, key="apply-complete-card"):
        st.space(20)
        st.markdown(
            f'<div style="display:flex;justify-content:center;padding:8px 0 6px;line-height:0;">'
            f'{svg_icon("check-circle", 72, "var(--navy)")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="text-align:center;font-size:24px;font-weight:800;color:var(--navy);margin-bottom:10px;">지원이 완료되었습니다</div>',
            unsafe_allow_html=True,
        )
        if company or title:
            st.markdown(
                f'<div style="text-align:center;font-size:16px;font-weight:600;color:var(--muted-strong);margin-bottom:14px;">{html.escape(company)}'
                + (f" · {html.escape(title)}" if title else "")
                + '</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div style="text-align:center;font-size:14px;color:var(--muted);">이 공고는 추천 목록에서 자동으로 제외됩니다. 지원 기록에서 진행 상황을 확인할 수 있어요.</div>',
            unsafe_allow_html=True,
        )
        st.space(20)

    st.space(24)

    with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
        if st.button("지원 기록 보기", key="s10_goto_history", type="primary"):
            go("S7")
        if st.button("홈으로", key="s10_goto_home"):
            st.session_state.nav_stack = []
            go("S0")


# ── S7. 지원 기록(2026-08-16 재설계, 사용자 확정) ───────────────────────
# 지원기록은 새 분석/새 판단을 하는 화면이 아니다 - "지원 후 관리 + 자동
# 기록 + 나중에 성과 검증" 3가지 역할만 한다. 회사명/직무명/URL/지원일/
# 이력서 버전은 전부 pipeline.finalize_application()에서 이미 자동으로
# 채워져 있다 - 사용자가 여기서 다시 입력하는 건 상태/다음 일정/메모
# 3개뿐이다. 리소스 측정 데이터(preparation_tracker)는 이 화면에 절대
# 섞지 않는다(개발자 전용 - 별도로만 조회).

_S7_STATUS_TONE = {
    "지원 완료": "blue", "서류 진행": "blue", "과제/코테": "orange",
    "면접": "orange", "최종 합격": "green", "불합격": "red", "지원 철회": "gray",
}
_S7_IN_PROGRESS_STATUSES = {"서류 진행", "과제/코테"}


def _s7_build_excel(apps: list[dict]) -> bytes:
    """사용자가 보는 지원기록만 내보낸다(회사/직무/지원일/상태/다음일정/
    결과/메모) - preparation_events 같은 개발용 리소스 측정 로그는 절대
    섞지 않는다(사용자 확정)."""
    import pandas as pd
    from io import BytesIO

    rows = []
    for a in apps:
        next_txt = " ".join(x for x in [(a.get("next_action_at") or "")[:10], a.get("next_action")] if x)
        rows.append({
            "회사": a.get("company") or "", "직무": a.get("title") or "",
            "지원일": (a.get("applied_at") or "")[:10], "상태": a.get("status") or "",
            "다음 일정": next_txt or "-",
            "결과": a.get("status") if a.get("status") in ("최종 합격", "불합격") else "",
            "메모": a.get("memo") or "",
        })
    df = pd.DataFrame(rows, columns=["회사", "직무", "지원일", "상태", "다음 일정", "결과", "메모"])
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="지원기록")
    return buf.getvalue()


_S7_TIMELINE_COLOR = {
    "지원 완료": "#16A34A", "서류 진행": "#2563EB", "과제/코테": "#D97706",
    "면접": "#7C3AED", "최종 합격": "#CA8A04", "불합격": "#DC2626", "지원 철회": "#94A3B8",
}


@st.dialog("지원 상세", width="large")
def _s7_detail_dialog(app: dict) -> None:
    """지원 상세(2026-08-31 재구성) - 실제 지원 관리 목적에 맞게. 기본 정보 /
    공고 정보(DB에 실제 있는 값만) / 전형 진행 이력(application_status_history) /
    지원 자료(있을 때만) / 상태·일정·메모. 없는 값은 추측하지 않고 숨긴다.
    새 상태 저장 시스템 없음 - application_manager.update_status() 그대로."""
    app_id = app["application_id"]
    job = get_candidate_job(app.get("job_id") or "")

    st.markdown(f"### {app.get('company','')} · {app.get('title','')}")
    meta_bits = [
        f"지원일 {(app.get('applied_at') or '-')[:10].replace('-', '.')}",
        f"현재 상태 {app.get('status') or '-'}",
    ]
    if app.get("source"):
        meta_bits.append(f"지원 경로 {app['source']}")
    st.caption("  ·  ".join(meta_bits))
    _url = app.get("url") or (job or {}).get("url")
    if _url:
        st.link_button("공고 원문 보기 ↗", _url)

    # ── 공고 정보 (DB에 실제 값이 있는 것만) ──
    info_rows = []
    if job:
        for label, key in (("지역", "location"), ("경력", "career_level"), ("산업", "industry")):
            v = (job.get(key) or "").strip()
            if v:
                info_rows.append((label, v))
    if info_rows:
        st.space("small")
        st.markdown("**공고 정보**")
        for label, v in info_rows:
            st.markdown(f"<span style='color:#64748B;font-size:13px;'>{label}</span>&nbsp;&nbsp;{html.escape(v)}", unsafe_allow_html=True)

    # ── 전형 진행 이력 (application_status_history) ──
    try:
        history = list_status_history(app_id)
    except Exception:  # noqa: BLE001
        history = []
    if history:
        st.space("small")
        st.markdown("**전형 진행**")
        for h in history:
            d = (h.get("recorded_at") or "")[:10].replace("-", ".")
            color = _S7_TIMELINE_COLOR.get(h.get("status"), "#94A3B8")
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:baseline;padding:3px 0;'>"
                f"<span style='width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;flex-shrink:0;transform:translateY(-1px);'></span>"
                f"<span style='color:#64748B;font-size:13px;min-width:82px;'>{d}</span>"
                f"<span style='font-size:13px;color:#0F172A;'>{html.escape(h.get('status') or '')}</span></div>",
                unsafe_allow_html=True,
            )

    # ── 지원 자료 (있을 때만, 크게 강조 안 함) ──
    version = resume_versions.get_version(app["resume_version"]) if app.get("resume_version") else None
    has_pdf = bool(version and os.path.exists(version["pdf_path"]))
    st.space("small")
    st.markdown("**지원 자료**")
    if has_pdf:
        with open(version["pdf_path"], "rb") as f:
            st.download_button(
                "맞춤 이력서 다운로드", f.read(), file_name="맞춤_이력서.pdf",
                mime="application/pdf", key=f"s7_resume_{app_id}",
            )
    else:
        st.caption("맞춤 이력서 기록 없음")
    if st.button("자소서 소재 보기", key=f"s7_cl_{app_id}"):
        if job is None:
            st.caption("공고 정보를 찾을 수 없습니다.")
        else:
            st.session_state.selected_job = job
            go("S6")
            st.rerun()

    # ── 상태 / 일정 / 메모 ──
    st.divider()
    cur_status = app.get("status") or APPLICATION_STATUS_OPTIONS[0]
    status_idx = APPLICATION_STATUS_OPTIONS.index(cur_status) if cur_status in APPLICATION_STATUS_OPTIONS else 0
    new_status = st.selectbox("현재 상태", APPLICATION_STATUS_OPTIONS, index=status_idx, key=f"s7_status_{app_id}")

    ncol1, ncol2 = st.columns([1, 2])
    next_date_raw = (app.get("next_action_at") or "")[:10]
    try:
        default_date = datetime.strptime(next_date_raw, "%Y-%m-%d").date() if next_date_raw else None
    except ValueError:
        default_date = None
    next_date = ncol1.date_input("다음 일정 날짜", value=default_date, key=f"s7_next_date_{app_id}")
    next_label = ncol2.text_input(
        "다음 일정 내용", value=app.get("next_action") or "",
        placeholder="예) 결과 확인, 1차 면접", key=f"s7_next_label_{app_id}",
    )
    memo = st.text_area(
        "메모", value=app.get("memo") or "", max_chars=500, height=80, key=f"s7_memo_{app_id}",
    )

    dcol1, dcol2 = st.columns([3, 1])
    if dcol1.button("저장", type="primary", key=f"s7_save_{app_id}", use_container_width=True):
        update_status(
            app_id, status=new_status,
            next_action=next_label or None,
            next_action_at=next_date.isoformat() if next_date else None,
            memo=memo or None,
        )
        st.success("저장되었습니다.")
        st.rerun()
    if dcol2.button("삭제", key=f"s7_delete_{app_id}", use_container_width=True):
        delete_application(app_id)
        st.rerun()

    _upd = (app.get("updated_at") or app.get("created_at") or "")[:16].replace("T", " ")
    if _upd:
        st.caption(f"최종 업데이트 {_upd}")


def render_s7() -> None:
    st.markdown(_SUBLIST_CSS, unsafe_allow_html=True)
    st.markdown(step_progress_html(4), unsafe_allow_html=True)  # 5단계: 지원 기록
    apps = list_applications(order_by="recent")

    hcol1, hcol2 = st.columns([4, 1])
    with hcol1:
        page_header(svg_icon("clipboard-list", 40, "var(--navy)"), "지원 기록", "지원한 공고와 이후 진행 상황을 한곳에서 관리합니다.")
    with hcol2:
        st.download_button(
            "Excel로 내보내기", _s7_build_excel(apps), file_name="지원기록.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="s7_export_excel", use_container_width=True,
        )
    back_button()

    if not apps:
        st.markdown(empty_state_html("아직 지원 기록이 없습니다.", "공고에 지원하면 여기에 기록이 쌓여요."), unsafe_allow_html=True)
        return

    total_n = len(apps)
    in_progress_n = sum(1 for a in apps if a.get("status") in _S7_IN_PROGRESS_STATUSES)
    interview_n = sum(1 for a in apps if a.get("status") == "면접")
    offer_n = sum(1 for a in apps if a.get("status") == "최종 합격")

    # ── 결과 메시지 도착(확인 필요) — 2026-09-04 사용자 확정 ──
    # Workflow C 가 본문에 합격/불합격 텍스트 없이 "새 메시지 도착"만 있는
    # ATS 알림 메일(나인하이어 등)을 지원기록 1건에 매칭했을 때 여기 쌓인다.
    # 상태는 시스템이 추측해서 바꾸지 않는다 - 어떤 지원건을 확인해야 하는지만
    # 알려주고, 실제 상태 변경은 "확인하기"로 연 상세에서 사용자가 직접 한다.
    _pending = list_pending_messages()
    if _pending:
        st.markdown(
            f"<div style='font-size:14px;font-weight:700;color:#92400E;background:#FFFBEB;"
            f"border:1px solid #FDE68A;border-radius:10px;padding:10px 16px;margin-bottom:10px;'>"
            f"📩 결과 메시지 도착 (확인 필요) · {len(_pending)}건</div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="s7-pending"):
            for i, p in enumerate(_pending):
                if i:
                    st.divider()
                prow = st.columns([4, 1.4, 1, 1], vertical_alignment="center")
                prow[0].markdown(f"**{html.escape(p.get('company') or '')}**  ·  {html.escape(p.get('title') or '')}")
                prow[1].caption((p.get("received_at") or p.get("detected_at") or "")[:10].replace("-", "."))
                if prow[2].button("확인하기", key=f"pending_open_{p['message_id']}", use_container_width=True):
                    _app = next((a for a in apps if a["application_id"] == p["application_id"]), None)
                    if _app:
                        _s7_detail_dialog(_app)
                if prow[3].button("숨기기", key=f"pending_dismiss_{p['message_id']}", use_container_width=True):
                    dismiss_pending_messages(p["application_id"])
                    st.rerun()
        st.space(16)

    # 2026-09-02(사용자 확정, UI만): ① "지원 현황" 제목 추가 ② 요약 카드
    # 상하 여백 소폭 증가 ③ 상태 필터를 클릭 전에도 흰 배경+테두리+화살표로
    # 보이게 ④ 지원 내역을 개별 카드 반복 → 하나의 리스트(행)로, 컬럼 헤더
    # sticky ⑤ 페이지네이션 제거, 리스트 영역 안에서만 세로 스크롤.
    # 데이터/상태 업데이트/KPI 계산 로직·필드는 무변경.
    st.markdown(
        """
        <style>
        /* 요약 카드: 항목 사이 세로 칸막이선 + 좌우 대칭 여백(값은 칸 안 가운데 정렬) */
        div[class*="st-key-s7-summary"] [data-testid="stColumn"] { padding: 0 18px; }
        div[class*="st-key-s7-summary"] [data-testid="stColumn"]:not(:first-child) {
            border-left: 1px solid var(--line, #E2E8F0);
        }
        /* 상태 필터: 미선택 상태에서도 입력 영역이 보이도록(1.59 = react-aria ComboBox) */
        div[class*="st-key-s7-filter"] [data-testid="stSelectbox"] .react-aria-ComboBox > div {
            background: #FFFFFF !important;
            border-color: #94A3B8 !important;
        }
        /* 지원 내역 헤더 = 카드 상단 full-bleed 회색 바(바깥 카드 padding 1.15rem 1.5rem 상쇄).
           배경/카드와 확실히 구분되도록 톤 진하게 + 글자에 맞게 높이 확보 + 칸 사이 세로선(아래 표와 정렬). */
        div[class*="st-key-s7-hdr"] {
            background: #E7EDF6 !important;
            margin: -1.15rem -1.5rem 0 !important;
            width: calc(100% + 3rem) !important;
            max-width: none !important;
            padding: 0 1.5rem !important;
            border-radius: 12px 12px 0 0;
            border-bottom: 1px solid #C2CEDE;
        }
        div[class*="st-key-s7-hdr"] [data-testid="stHorizontalBlock"] { background: transparent !important; }
        /* 컬럼에 상하 여백을 줘서 회색 바 높이 확보 + 세로선이 바 전체 높이를 채우게(아래 표와 동일). */
        div[class*="st-key-s7-hdr"] [data-testid="stColumn"] {
            background: transparent !important;
            border: none !important;
            padding: 22px 10px !important;
        }
        div[class*="st-key-s7-hdr"] [data-testid="stColumn"]:not(:first-child) {
            border-left: 1px solid #E2E8F0 !important;
        }
        /* 지원 내역 행 = KPI 표처럼 셀마다 회색 격자선(세로·가로). 스크롤바 자리 항상 확보. */
        div[class*="st-key-s7-rows"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        div[class*="st-key-s7-rows"] [data-testid="stVerticalBlock"],
        div[class*="st-key-s7-rows"] > div { scrollbar-gutter: stable; }
        div[class*="st-key-s7-rows"] [data-testid="stColumn"] {
            border-bottom: 1px solid #E2E8F0 !important;
            padding: 8px 10px !important;
        }
        div[class*="st-key-s7-rows"] [data-testid="stColumn"]:not(:first-child) {
            border-left: 1px solid #E2E8F0 !important;
        }
        div[class*="st-key-s7-rows"] [data-testid="stHorizontalBlock"] { min-height: 0; }
        .s7-col-h { font-size: 12px; font-weight: 700; color: #64748B; }
        div[class*="st-key-s7-list"] button[kind="tertiary"] {
            border-color: transparent !important;
            background: transparent !important;
            color: #1E3A8A !important;
            padding: 2px 6px !important;
            min-height: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:15px;font-weight:700;color:var(--navy,#1E3A8A);margin-bottom:8px;'>지원 현황</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="s7-summary"):
        st.space(10)
        scols = st.columns(4)
        for col, (label, n), _al in zip(
            scols,
            [("전체 지원", total_n), ("진행 중", in_progress_n), ("면접", interview_n), ("최종 합격", offer_n)],
            _STAT_ALIGN_4,
        ):
            with col:
                st.markdown(
                    f"<div style='text-align:{_al};font-size:.9rem;color:var(--muted);'>{label}</div>"
                    f"<div style='text-align:{_al};font-size:1.7rem;font-weight:700;margin-top:2px;'>{n}</div>",
                    unsafe_allow_html=True,
                )
        st.space(10)

    st.space(16)

    _fcol, _ = st.columns([1, 2])
    with _fcol:
        with st.container(key="s7-filter"):
            status_filter = st.selectbox("상태", ["전체"] + APPLICATION_STATUS_OPTIONS, key="s7_status_filter")
    filtered = apps if status_filter == "전체" else [a for a in apps if a.get("status") == status_filter]

    st.space(8)

    if not filtered:
        st.caption("이 상태에 해당하는 지원 기록이 없습니다.")
        st.space(20)
        _render_s7_kpi(apps)
        return

    # 지원 내역 = 흰 카드 하나. 헤더(스크롤 밖 고정) + 그 아래 행만 세로 스크롤.
    # 회사·직무 / 지원일 / 지원 상태 / 다음 일정 / 상세. (4번째는 next_action[_at])
    _S7_COLS = [3.6, 1.3, 1.5, 1.7, 0.8]
    _S7_ROWS_H = 330

    with st.container(border=True, key="s7-list"):
        with st.container(key="s7-hdr"):
            hdr = st.columns(_S7_COLS, vertical_alignment="center")
            for c, t in zip(hdr, ["회사 · 직무", "지원일", "지원 상태", "다음 일정", ""]):
                c.markdown(f"<div class='s7-col-h'>{t}</div>", unsafe_allow_html=True)
        st.space(4)
        with st.container(height=_S7_ROWS_H, border=False, key="s7-rows"):
            for a in filtered:
                app_id = a["application_id"]
                _title = a.get("title") or ""
                if len(_title) > 40:
                    _title = _title[:40] + "…"
                rc = st.columns(_S7_COLS, vertical_alignment="center")
                rc[0].markdown(f"**{html.escape(a.get('company',''))}**  ·  {html.escape(_title)}")
                rc[1].caption((a.get("applied_at") or "-")[:10].replace("-", "."))
                rc[2].markdown(
                    badge_html(a.get("status") or "-", _S7_STATUS_TONE.get(a.get("status"), "gray")),
                    unsafe_allow_html=True,
                )
                next_txt = " ".join(
                    x for x in [(a.get("next_action_at") or "")[:10].replace("-", "."), a.get("next_action")] if x
                ) or "-"
                rc[3].caption(next_txt)
                if rc[4].button("상세", key=f"s7_detail_{app_id}", type="tertiary"):
                    _s7_detail_dialog(a)

    # 지원 내역 바로 아래 - 기본 접힘. 새 측정 없이 이미 수집 중인
    # preparation_sessions.elapsed_seconds / cover_letter_source_used /
    # applications.resume_version 만 계산·표시한다.
    st.space(20)
    _render_s7_kpi(apps)


# ── 지원 KPI(2026-09-01, 사용자 확정) - "지원 KPI 확인" expander ─────────
# 지원 완료 건마다 T0(analysis_opened) → application_created 실경과시간을
# 자동 계산해 지원 기록 화면에서 바로 본다. 새 이벤트/측정 로직/수동 입력
# 없음. 화면을 오래 열어둔 건에 평균이 크게 흔들려 메인 집계는 중앙값.
# preparation_events 의 세부 timestamp 는 DB 에 그대로 두되 여기 노출 안 함.
# KPI 측정 시작점(사용자 확정 2026-09-01): JobFit 기능·검증 완료 시점.
# 이 시각보다 먼저 완료된 지원(app 6·7·8 = 개발/수정/검증 중 발생, app 9 = 브라우저
# E2E 테스트)은 화면 오래 켜둔 시간 등이 섞여 있어 실사용 성과에서 제외한다.
# 지원 기록 목록 자체에는 그대로 남기고 "지원 KPI 확인" 표·집계에서만 뺀다.
_KPI_START_AT = "2026-09-01T23:25:00"
_KPI_EXCLUDED_APP_IDS = {6, 7, 8, 9}  # cutoff 로도 걸리지만 명시적으로도 제외(자기문서화).


def _fmt_kpi_duration(sec) -> str:
    if sec is None:
        return "-"
    try:
        total = int(round(float(sec)))
    except (TypeError, ValueError):
        return "-"
    m, s = divmod(max(total, 0), 60)
    return f"{m}분 {s}초" if m else f"{s}초"


def _load_application_kpi_rows(apps: list[dict]) -> list[dict]:
    """지원 완료 건 + preparation_sessions(elapsed_seconds / 자소서 사용여부)를
    합쳐 KPI 표용 행을 만든다. 읽기 전용 - 기존 측정 테이블을 건드리지 않는다."""
    import sqlite3

    # 2026-09-05(버그 수정) - status == "지원 완료" 조건이 있었는데, applications
    # 테이블의 모든 행은 이미 "지원 완료"(T2) 시점에 생성된다 - status 는 그
    # *이후* ATS 진행상황(서류 진행/면접/불합격 등)을 반영해 계속 바뀐다.
    # "지원까지 걸린 시간"은 그 이후 결과와 무관한데, 이 조건 때문에 상태가
    # 바뀐 건(예: 메일 자동 반영으로 불합격 처리)마다 측정 대상에서 빠져
    # 사용자가 실제 20건을 지원했는데도 화면엔 13건만 잡히는 오류가 있었다.
    done = [
        a for a in apps
        if a.get("application_id") not in _KPI_EXCLUDED_APP_IDS
        and (a.get("applied_at") or "") >= _KPI_START_AT  # 측정 시작점 이후만
    ]
    if not done:
        return []

    conn = sqlite3.connect(preparation_tracker.DB_PATH)
    try:
        sess = {
            row[0]: {"elapsed": row[1], "cl": bool(row[2])}
            for row in conn.execute(
                "SELECT application_id, elapsed_seconds, cover_letter_source_used "
                "FROM preparation_sessions WHERE application_id IS NOT NULL"
            )
        }
    except sqlite3.OperationalError:
        sess = {}
    finally:
        conn.close()

    rows = []
    for a in done:
        s = sess.get(a.get("application_id"), {})
        applied = a.get("applied_at") or ""
        rows.append({
            "_sort": applied,
            "_elapsed": s.get("elapsed"),
            "지원일": (applied[5:10].replace("-", "/") if len(applied) >= 10 else "-"),
            "회사 · 직무": " · ".join(x for x in [a.get("company") or "", a.get("title") or ""] if x) or "(정보 없음)",
            "총 소요시간": _fmt_kpi_duration(s.get("elapsed")),
            "맞춤 이력서": "사용" if (a.get("resume_version") or "").strip() else "미사용",
            "자소서": "포함" if s.get("cl") else "미포함",
        })
    rows.sort(key=lambda r: r["_sort"], reverse=True)  # 최신 지원순
    return rows


def _render_s7_kpi(apps: list[dict]) -> None:
    with st.expander("지원 KPI", expanded=False):
        rows = _load_application_kpi_rows(apps)
        if not rows:
            st.caption(
                f"KPI 측정 시작 시점({_KPI_START_AT[:10].replace('-', '.')}) 이후 완료된 지원부터 집계합니다. "
                "그 전 지원 기록은 개발·검증 중 발생분이라 제외되며, 목록에는 그대로 남아 있습니다."
            )
            return

        import statistics

        durs = [r["_elapsed"] for r in rows if r["_elapsed"] is not None]
        durs_cl = [r["_elapsed"] for r in rows if r["_elapsed"] is not None and r["자소서"] == "포함"]
        durs_no_cl = [r["_elapsed"] for r in rows if r["_elapsed"] is not None and r["자소서"] == "미포함"]

        def _med(xs):
            return _fmt_kpi_duration(statistics.median(xs)) if xs else "-"

        # 상단 요약 지표 - 항목 사이에만 얇은 세로 구분선(요약 카드와 같은 톤).
        st.markdown(
            "<style>div[class*='st-key-s7-kpi-metrics'] [data-testid='stColumn']{padding:0 16px;}"
            "div[class*='st-key-s7-kpi-metrics'] [data-testid='stColumn']:not(:first-child)"
            "{border-left:1px solid var(--line,#E2E8F0);}</style>",
            unsafe_allow_html=True,
        )
        with st.container(key="s7-kpi-metrics"):
            for col, (label, val), _al in zip(st.columns(4), [
                ("측정 대상", f"{len(rows)}건"),
                ("전체 중앙 소요시간", _med(durs)),
                ("자소서 포함 중앙", _med(durs_cl)),
                ("자소서 미포함 중앙", _med(durs_no_cl)),
            ], _STAT_ALIGN_4):
                with col:
                    st.markdown(
                        f"<div style='text-align:{_al};font-size:.9rem;color:var(--muted);'>{label}</div>"
                        f"<div style='text-align:{_al};font-size:1.35rem;font-weight:700;margin-top:2px;'>{val}</div>",
                        unsafe_allow_html=True,
                    )

        st.space(12)
        st.dataframe(
            [{k: r[k] for k in ("지원일", "회사 · 직무", "총 소요시간", "맞춤 이력서", "자소서")} for r in rows],
            hide_index=True, use_container_width=True,
        )
        st.caption(
            "소요시간 = 공고 분석 시작(T0) → 지원 완료까지의 실제 경과 시간. "
            "화면을 오래 열어두면 늘어날 수 있어 메인 집계는 중앙값 기준입니다."
        )


# ── SETTINGS. 설정(자리만 - 아직 실제 설정 항목 없음) ───────────────────

def render_settings() -> None:
    st.markdown("<div class='sc-home-greeting'>설정</div>", unsafe_allow_html=True)
    st.caption("설정 화면은 준비 중입니다.")


# ── S9. 관심공고 ──────────────────────────────────────────────────────

def render_s9() -> None:
    st.markdown(_SUBLIST_CSS, unsafe_allow_html=True)
    page_header(svg_icon("bookmark", 40, "var(--navy)"), "관심 공고",
                subtitle="☆로 저장한 공고입니다. 공고 찾기 목록에도 계속 표시됩니다.")
    back_button()

    saved = list_saved_jobs()
    if not saved:
        st.markdown(empty_state_html("저장한 관심 공고가 없습니다.", "마음에 드는 공고에서 ☆을 눌러 저장해보세요."), unsafe_allow_html=True)
        return

    for i, s in enumerate(saved):
        with st.container(border=True):
            st.markdown(f"**{s['company']}** - {s['title']}", unsafe_allow_html=True)
            st.caption(f"출처: {s.get('source','기업 홈페이지')}")

            cols = st.columns(3)
            if s.get("url"):
                cols[0].link_button("공고 다시보기", s["url"])
            else:
                # 관심 공고는 추천 목록(candidate_jobs)에서만 저장 가능하다
                # - 그 풀은 url이 항상 있어야 정상이라, 여기 걸리면
                # 데이터 결함이니 조사 대상.
                cols[0].caption("⚠ 원문 링크 없음(정상적으로는 없어야 함 - 데이터 확인 필요)")
            if cols[1].button("지원하기", key=f"saved_apply_{i}"):
                full_job = get_candidate_job(s["job_id"])
                if full_job is None:
                    st.error("공고 정보를 찾을 수 없습니다(삭제되었을 수 있습니다).")
                else:
                    st.session_state.selected_job = full_job
                    _run_apply_flow(full_job)
            if cols[2].button("즐겨찾기 해제", key=f"unsave_{i}"):
                remove_saved_job(s["job_id"])
                st.rerun()


# ── S11. 제외한 공고 ─────────────────────────────────────────────────────
# 공고 찾기 목록에서 ×로 제외한 공고를 다시 볼 수 있는 화면. "영원히 숨김"이
# 아니라 여기서 [제외 취소]로 언제든 일반 추천 후보로 되돌릴 수 있게 한다.
# 저장하는 건 순수 사용자 행동 데이터(dismissed_jobs)뿐 - 추천 알고리즘
# 학습/유사 공고 감점에는 쓰지 않는다.

def render_s11() -> None:
    st.markdown(_SUBLIST_CSS, unsafe_allow_html=True)
    page_header(svg_icon("filter", 40, "var(--navy)"), "제외한 공고",
                "공고 찾기에서 ×로 제외한 공고입니다. [제외 취소]를 누르면 다시 추천 후보로 돌아갑니다.")
    back_button()

    dismissed = list_dismissed_jobs()
    if not dismissed:
        st.markdown(
            empty_state_html("제외한 공고가 없습니다.",
                             "공고 찾기 목록에서 ×를 누르면 그 공고가 여기 모입니다."),
            unsafe_allow_html=True,
        )
        return

    for i, d in enumerate(dismissed):
        with st.container(border=True):
            st.markdown(f"**{d['company']}** - {d['title']}", unsafe_allow_html=True)
            st.caption(f"출처: {d.get('source') or '기업 홈페이지'}"
                       + (f" · 제외 {(d.get('dismissed_at') or '')[:10]}" if d.get("dismissed_at") else ""))

            cols = st.columns([1, 1, 2])
            if d.get("url"):
                cols[0].link_button("공고 원문 보기", d["url"])
            if cols[1].button("제외 취소", key=f"undismiss_{i}", type="primary"):
                # 제외 기록을 지우고 추천 목록 캐시(세션+DB)를 비운다 - 이
                # 공고는 목록 생성 단계에서 제외됐으므로, 다시 후보 풀에
                # 넣어 관련도 순위를 매기려면 재계산이 필요하다(드물게
                # 누르는 명시적 동작이라 1~2분 대기는 감수한다 - 새로고침과 동일).
                remove_dismissed_job(d["job_id"])
                recommendation_cache.clear_all()
                st.session_state.top_cache = {}
                st.rerun()


# ── 라우팅 ────────────────────────────────────────────────────────────

_init_state()

# ── missed-run catch-up (2026-08-30) ──────────────────────────────────
# n8n Workflow B 는 매일 10:00 KST 에 수집을 돌리지만, 개인 노트북이라
# 그 시각에 PC 가 꺼져 있으면 그날 수집이 누락된다. JobAI 최초 실행 시
# "오늘 완료 로그가 하나도 없으면" background 로 1회만 collection.refresh
# 를 돌린다(SUCCESS/PARTIAL 있으면 안 함, 이미 실행 중이면 skip). 즉시
# 반환 - UI 를 block 하지 않는다. 세션당 1회만 판단.
if not IS_DEMO and not st.session_state.get("_catchup_checked"):
    st.session_state["_catchup_checked"] = True
    try:
        from collection.catchup import maybe_run_catchup
        print(f"[catchup] {maybe_run_catchup()}", flush=True)
    except Exception as _e:  # noqa: BLE001 - catch-up 실패가 앱 로딩을 막지 않게
        print(f"[catchup] skip (error): {_e}", flush=True)

# ── 마지막 이력서 자동 복원 (2026-09-03) ──────────────────────────────
# 게스트 세션은 이력서를 세션에 안 들고 시작해서, 새로 들어올 때마다 홈
# 대시보드("분석한 공고"/"최근 분석한 공고")가 비어 보였다. 마지막으로
# 업로드한 PDF(data/last_resume/)를 세션에 이력서가 없을 때만 복원한다.
# 무거운 Understanding 은 캐시에 있을 때만 가져온다(LLM 호출 없음) -
# 없으면 빈 값으로 두고 실제 분석 시 생성된다. career_level 은 규칙 기반.
if not st.session_state.get("_last_resume_restored"):
    st.session_state["_last_resume_restored"] = True
    if not st.session_state.get("resume_raw"):
        try:
            _saved = last_resume.load()
            if _saved:
                _txt = extract_text_from_pdf(io.BytesIO(_saved["pdf_bytes"]))
                st.session_state.resume_raw = _txt
                st.session_state.resume_pdf_bytes = _saved["pdf_bytes"]
                st.session_state.uploaded_file_id = f"{_saved['file_name']}:{_saved['file_size']}"
                st.session_state.resume_file_name = _saved["file_name"]
                st.session_state.resume_file_size = _saved["file_size"]
                st.session_state.resume_uploaded_at = _saved.get("saved_at") or ""
                st.session_state.show_uploader = False
                _det = pipeline.detect_user_career_level(_txt)
                st.session_state.user_career_level = _det["career_level"]
                st.session_state.career_level_confidence = _det["confidence"]
                st.session_state.career_level_reason = _det["reason"]
                st.session_state.career_level_source = "auto"
                _und = get_cached_resume_understanding(_txt)
                st.session_state.resume_semantic_objects = (_und or {}).get("semantic_objects") or []
                print("[last_resume] restored from disk", flush=True)
        except Exception as _e:  # noqa: BLE001 - 복원 실패가 앱 로딩을 막지 않게
            print(f"[last_resume] restore skip: {_e}", flush=True)

# DEMO MODE 전용 - "자소서 준비하기"는 LLM을 여러 번(Planner→Writer→Critic)
# 호출하는 화면이라 실시간 생성은 안 하지만, 버튼과 화면 자체는 숨기지
# 않는다(사용자 확정). 데모 공고 1건(demo-001)에 한해 실제 화면과 같은
# 구조의 예시 결과를 세션에 미리 채워 넣는다 - LLM 호출 0회, 그 외
# 3개 공고는 기존과 동일하게 "소재 준비하기" 버튼을 눌러야 시도되고,
# 그 시도는 call_llm() 가드에서 막혀 안내 메시지로 대체된다.
if IS_DEMO and not st.session_state.get("_demo_composition_seeded"):
    st.session_state["_demo_composition_seeded"] = True
    st.session_state.composition_by_job.setdefault("demo-001", {
        "motivation": {
            "title": "반복을 줄이고 판단에 집중하는 방식으로 기여하고 싶습니다",
            "material": "SQL/Python 기반 데이터 분석과 지표 설계 경험이 이 공고가 강조하는 역할과 맞닿아 있습니다.",
            "writing_point": "지표를 새로 설계해 의사결정에 반영한 경험을 구체적으로 강조",
            "draft": (
                "완료 여부만으로는 실제 효과를 판단하기 어려운 상황에서, 행동을 기준으로 지표를 "
                "새로 설계해 의사결정에 반영한 경험이 있습니다. 이 공고에서도 지표를 먼저 의심하고 "
                "다시 설계하는 방식으로 기여하고 싶습니다."
            ),
        },
        "experience": {
            "title": "리워드 비용 최적화 전략에서 지표를 새로 설계해 의사결정을 바꾼 경험",
            "situation": "완료 구매의 17.5%가 오퍼 확인 없이 발생해, 완료 여부만으로는 오퍼 효과를 판단하기 어려웠음",
            "action": "오퍼 확인 여부와 구매 여부로 행동을 4가지로 분류하고, Net Lift Index·Cannibalization Rate 지표를 설계함",
            "result": "50.0% 고객군에 Opt-in 방식을 제안했고, 발송 축소 시뮬레이션에서 +$24,324의 순효과를 확인함",
            "writing_point": "숫자 기반으로 운영 기준을 다시 세운 과정 강조",
            "draft": (
                "완료 구매의 상당수가 오퍼를 확인하지 않고도 발생한다는 점을 발견하고, 구매 행동을 "
                "4가지로 나누어 새로운 지표(Net Lift Index, Cannibalization Rate)를 설계했습니다. "
                "이 지표를 바탕으로 특정 고객군에는 Opt-in 방식을 제안했고, 발송을 줄이는 시뮬레이션에서 "
                "실제 순효과를 확인해 운영 기준을 다시 세웠습니다."
            ),
        },
        "verification": {"passed": True, "failed_checks": []},
    })

SCREENS = {
    "LOGIN": render_login,
    "S0": render_s0,
    "S1": render_s1,
    "S1D": render_s1d,
    "S2": render_s2,
    "S3": render_s3,
    "S4": render_s4,
    "S6": render_s6,
    "S7": render_s7,
    "S8": render_s8,
    "S9": render_s9,
    "S10": render_s10,
    "S11": render_s11,
    "SETTINGS": render_settings,
}

# 사이드바(2026-08-31, 사용자 확정 - 정보구조 재정리). 역할이 완전히 분리된다:
#  홈       = "오늘 뭘 해야 하지?"  → 지원 현황 대시보드
#  공고 찾기 = "어디 지원하지?"     → 이력서 확인 + 수집 공고 탐색·분석
#  지원 기록 = "내가 어디까지 진행됐지?"
#  관심/제외 = ☆/×로 분류해둔 공고
# "직접 공고 분석"(S2)은 사이드바에서 뺐다(2026-08-31 사용자 확정 - "빼는 것에
# 동의") - 화면은 그대로 두고, 공고 찾기 안의 "+ 공고 직접 붙여넣기" 버튼으로만
# 접근한다. 이력서 맞춤화(S4)도 공고 분석 하위 플로우라 최상위 메뉴엔 없다.
_NAV_GROUPS = [
    [("S0", "홈", "home")],
    [("S1", "공고 찾기", "search")],
    [("S9", "관심 공고", "star"), ("S11", "제외한 공고", "block")],
    [("S7", "지원 기록", "fact_check")],
]
_NAV_ITEMS = [item for group in _NAV_GROUPS for item in group]
# 공고 분석(상세)은 S1 목록 안에서 영역만 바뀌므로 S1D는 "공고 찾기"를 활성
# 표시한다. JD 붙여넣기 판단결과(S3)는 그 진입점인 "직접 공고 분석"(S2)을,
# 커스터마이징 세부(S6)는 "이력서 맞춤화"(S4, 메뉴엔 없지만 화면은 존재)를
# 활성 취급한다.
_NAV_ALIAS = {"S1D": "S1", "S3": "S2", "S6": "S4"}

if st.session_state.screen != "LOGIN":
    with st.sidebar:
        _demo_badge = (
            "<span title='" + DEMO_BANNER_TEXT + "' "
            "style='margin-left:6px;font-size:0.62rem;font-weight:700;color:#1E3A8A;"
            "background:#E7ECFB;border-radius:999px;padding:2px 8px;vertical-align:middle;'>DEMO</span>"
        ) if IS_DEMO else ""
        st.markdown(
            "<div style='display:flex;align-items:center;gap:8px;padding:0.3rem 0.2rem 1.1rem;'>"
            "<span style='font-size:1.3rem;font-weight:800;color:var(--text-primary)'>JobFit AI</span>"
            "<span style='color:var(--accent);font-weight:800'>+</span>"
            f"{_demo_badge}"
            "</div>",
            unsafe_allow_html=True,
        )
        current_screen = _NAV_ALIAS.get(st.session_state.screen, st.session_state.screen)
        for _gi, group in enumerate(_NAV_GROUPS):
            if _gi:
                st.space(10)  # 그룹 사이 약한 간격만
            for screen_id, label, material_icon in group:
                is_active = current_screen == screen_id
                if st.button(
                    label, key=f"nav_{screen_id}", use_container_width=True,
                    icon=f":material/{material_icon}:", type="primary" if is_active else "secondary",
                ):
                    if not is_active:
                        go(screen_id)

        # DEMO MODE 전용 - 원래 nav 구조(_NAV_GROUPS)에는 없는 항목이라
        # 운영 화면의 메뉴 구성을 그대로 둔 채 맨 아래에만 작게 추가한다.
        if IS_DEMO:
            st.space(20)
            if st.button("🔄 데모 데이터 초기화", key="demo-reset-btn", use_container_width=True,
                         help="공고·지원 기록에 변경한 내용을 지우고 처음 예시 데이터로 되돌립니다."):
                from scripts.init_demo_db import reset_demo_data
                with st.spinner("데모 데이터를 초기화하는 중입니다..."):
                    reset_demo_data()
                for k in ("top_cache", "resume_raw", "resume_pdf_bytes", "uploaded_file_id"):
                    st.session_state.pop(k, None)
                st.success("초기화되었습니다.")
                st.rerun()

SCREENS[st.session_state.screen]()
