"""
ui_theme.py

Streamlit 기본 룩을 "JobFit AI" 레퍼런스 목업 스타일로 덮어쓰는 CSS
테마. 순수 프레젠테이션 레이어 - 여기엔 어떤 판단/로직도 없다.
resume_input 패키지는 이 파일을 모르고, 이 파일도 resume_input을
모른다(완전히 독립). 화면에 표시되는 문구(카피)는 이 파일이 건드리지
않는다 - 색/모양/레이아웃만 바꾼다.

배색(2026-07-16 사용자 요청으로 재조정): 메인은 흰색 배경 + 네이비
(구조적 강조 - 버튼/포커스/탭/사이드바 active/링차트 채움). 포인트로
연한 베이비 핑크(소프트 강조 - 뱃지/hover 배경/링차트 트랙)를 쓴다.
추가 포인트로 노랑/빨강/베이지/블루를 뱃지 톤으로 제공한다. 기존
(2026-07-14) 따뜻한 크림 배경 + 코랄 핑크 단일 강조 버전에서 전환.

Streamlit의 내부 DOM 구조(data-testid)에 의존하는 선택자가 있어
Streamlit 버전이 크게 바뀌면 일부 선택자가 안 먹힐 수 있다 - 이
프로젝트는 1.59.1 기준으로 작성했다. 안 먹히는 선택자가 있어도
레이아웃 자체가 깨지지는 않는다(그냥 그 부분만 Streamlit 기본
스타일로 보임) - CSS라 예외를 던지지 않는다.
"""
from __future__ import annotations

import streamlit as st

_FONT_IMPORT = """
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
"""

_CSS = """
:root {
    /* 메인 - 파스텔 배경 + 네이비(2026-07-23 사용자 확정 - "메인 브랜드
       컬러(네이비)는 유지, 배경/카드 톤만 파스텔 블루로" - 순백색보다
       은은한 파스텔 배경이 더 고급스럽다는 방향 확정). */
    --bg: #F6F8FF;
    --card-bg: #FFFFFF;
    --card-border: #E7EAF5;
    --text-primary: #14213D;
    --text-secondary: #5E6472;
    --text-muted: #A7ABBA;

    /* 네이비 - 버튼/포커스/탭/사이드바 active/링차트 채움. 버튼은
       입체감을 위해 3-stop 그라데이션(밝은 위 -> 진한 아래)을 쓴다.
       --navy-light는 사용자가 준 "서브 네이비"(#3146C5), --navy-mid는
       사용자가 준 메인 네이비(#1E3A8A) 그대로, --navy-dark는 그 아래로
       더 짙게 이어지는 그라데이션 끝점이다. */
    --navy: #163A70;
    --navy-hover: #1B4B92;
    --navy-light: #3146C5;
    --navy-mid: #1E3A8A;
    --navy-dark: #152C6B;
    --navy-soft: #E8ECF7;

    /* 포인트(전체의 10~15%만) - 베이비 핑크 */
    --pink: #F7DCE8;
    --pink-dark: #C97FA0;
    --pink-soft: #FFF3F7;

    /* 기존 변수명 유지(하위 호환) - 네이비/핑크로 매핑 */
    --accent: var(--navy);
    --accent-dark: var(--navy-dark);
    --accent-soft: var(--pink-soft);
    --accent-pink: var(--pink);
    --accent-pink-soft: var(--pink-soft);

    /* 시맨틱 톤(성공/경고/정보) - 베이비 그린/옐로우/블루 */
    --mint: #DDF4EA;
    --mint-dark: #1F8F5E;
    --mint-soft: #F2FFF6;
    --peach: #FFE7D2;
    --peach-dark: #B9702E;
    --peach-soft: #FFFBEA;
    --sky: #DCEFFF;
    --sky-dark: #2F6FE0;
    --sky-soft: #EEF5FF;

    /* 추가 포인트 컬러(뱃지/태그용) */
    --yellow: #E8B93D;
    --yellow-soft: #FFFBEA;
    --red: #E0555F;
    --red-soft: #FBE3E5;
    --beige: #B9A47E;
    --beige-soft: #F3ECDD;
    --blue: var(--sky-dark);
    --blue-soft: var(--sky-soft);
    --divider: #E7EAF5;

    --sidebar-bg: #FFFFFF;
    --radius-lg: 16px;
    --radius-md: 13px;
    --radius-sm: 9px;
    /* 카드 공통 그림자 - 사용자가 준 공식 그대로(rgba(30,58,138,.06) =
       새 네이비 #1E3A8A 톤의 그림자). 카드마다 다른 그림자를 쓰지 않는다. */
    /* 2026-08-31 디자인 통일 - 공통 카드 그림자 하나. 강한 그림자 금지. */
    --shadow-sm: 0 4px 20px rgba(30, 58, 138, 0.06);
    --shadow-md: 0 8px 24px rgba(30, 58, 138, 0.12);
    --shadow-lift: 0 8px 24px rgba(30, 58, 138, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── 배경 / 기본 여백 ───────────────────────────────────────────── */
/* 2026-08-31(사용자 확정) - 앱 전체 페이지 최외곽 배경을 하나로 통일한다.
   기준값 = 로그인 화면에 실제 적용되던 --bg(#F6F8FF, 아주 연한 Baby Blue).
   페이지별 _STEP1_CSS/_STEP2_CSS/_SUBLIST_CSS/_ANALYSIS_CSS 가 각자
   .stApp 배경을 흰색/회색으로 덮던 것을 전부 제거했으므로, 여기 한 곳만
   !important 로 강제하면 모든 route 가 동일 배경을 쓴다. 콘텐츠 카드는
   그대로 White(--card-bg). */
.stApp {
    background: var(--bg) !important;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
}

/* ── 공통 카드(2026-08-31, 사용자 확정 - 디자인 시스템 통일) ────────────
   앱 전체의 "네모 박스/카드"를 하나의 규칙으로 통일한다. 기준은 공고 찾기
   화면의 White 카드: Baby Blue 배경(--bg) 위에 흰 카드가 아주 살짝 떠
   있는 형태. 배경 #FFFFFF / radius 12px / soft shadow / 테두리 없음.

   Streamlit 1.59.1 은 st.container(border=True) 를 stVerticalBlock +
   emotion 클래스(테두리+radius 를 그 클래스가 준다)로 렌더한다.
   .st-emotion-cache-3nc66l = "테두리 있는 컨테이너"의 emotion 해시
   (스타일 내용 기반이라 이 스트림릿 버전 + config.toml baseRadius/
   borderColor 조합에서 고정). 스트림릿 업그레이드/테마 radius 변경 시
   이 해시가 바뀌면 갱신 필요. stVerticalBlockBorderWrapper 는 미래 버전 대비. */
[data-testid="stVerticalBlockBorderWrapper"],
.stApp .st-emotion-cache-3nc66l,
[data-testid="stDialog"] .st-emotion-cache-3nc66l {
    background: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(30, 58, 138, 0.06) !important;
    /* 카드 안 글자가 테두리에 붙지 않게 좌우 여백을 넉넉히 + 대칭. */
    padding: 1.15rem 1.5rem !important;
}
/* 카드 안에 카드(중첩)는 그림자를 겹치지 않고 얇은 선으로만 구분. */
.st-emotion-cache-3nc66l .st-emotion-cache-3nc66l {
    box-shadow: none !important;
    border: 1px solid var(--card-border) !important;
}

/* Streamlit 기본 장식 숨기기 */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stDecoration"] { display: none; }

/* ── 타이포그래피 ───────────────────────────────────────────────── */
h1, h2, h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.75rem !important; margin-bottom: 0.25rem !important; }
h2, h3 { font-size: 1.15rem !important; margin-top: 1.25rem !important; }
p, span, label, div { color: var(--text-primary); }
[data-testid="stCaptionContainer"], .stCaption {
    color: var(--text-secondary) !important;
}

/* ── 버튼(2026-07-24 디자인 통일) ────────────────────────────────────
   Primary/Secondary 구분 없이 프로젝트 전체가 하나의 버튼 디자인만
   쓴다 - 흰 배경 + 네이비 테두리/글자, radius 12px, height 40px.
   pill(999px)/그라데이션은 쓰지 않는다. ─────────────────────────── */
.stButton > button, .stLinkButton > a, .stDownloadButton > button {
    height: 40px !important;
    border-radius: 12px !important;
    border: 1px solid var(--navy) !important;
    background: #FFFFFF !important;
    color: var(--navy) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 0 22px !important;
    box-shadow: none !important;
    transition: background .15s ease, border-color .15s ease, color .15s ease;
    white-space: nowrap !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
/* 2026-09-07(버그 수정, 실측 기반) - 실제 문제는 "중앙정렬이 안 됨"이
   아니라 p(텍스트) 박스 자체가 자기 글자 실제 폭(Range.getBoundingClientRect
   실측 약 52px)보다 좁게(약 38px) 잡혀서, 그 좁은 박스는 button 안에서
   정확히 중앙에 오지만 그 안의 글자가 한쪽(오른쪽)으로 삐져나오는 것이었다
   - DevTools로 p/stMarkdownContainer의 실제 렌더 폭을 재서 확인함.
   글자 폭보다 박스가 좁아지지 않게 min-width를 텍스트 실제 폭 기준
   (max-content)으로 고정해 삐져나옴 자체를 막는다. */
.stButton > button [data-testid="stMarkdownContainer"],
.stLinkButton > a [data-testid="stMarkdownContainer"],
.stDownloadButton > button [data-testid="stMarkdownContainer"],
.stButton > button [data-testid="stMarkdownContainer"] p,
.stLinkButton > a [data-testid="stMarkdownContainer"] p,
.stDownloadButton > button [data-testid="stMarkdownContainer"] p {
    min-width: max-content !important;
    margin: 0 !important;
}
.stButton > button p, .stButton > button span,
.stLinkButton > a p, .stLinkButton > a span { color: var(--navy); }
.stButton > button:hover, .stLinkButton > a:hover {
    border-color: var(--navy-hover) !important;
    color: var(--navy-hover) !important;
    background: #F5F8FF !important;
}
.stButton > button:hover p, .stButton > button:hover span,
.stLinkButton > a:hover p, .stLinkButton > a:hover span { color: var(--navy-hover); }
/* Primary도 동일한 디자인 언어를 쓴다(2026-07-24, 사용자 확정 - "로그인
   Guest 시작/파일 선택/이력서 교체/연차 수정 전부 같은 버튼"). 필러+
   그라데이션 대신 흰 배경 + 네이비 테두리인 단일 버튼 스타일 하나만
   프로젝트 전체에서 쓴다. */
.stButton > button[kind="primary"], .stLinkButton > a[kind="primary"] {
    background: #FFFFFF !important;
    border: 1px solid var(--navy) !important;
    color: var(--navy) !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"] p, .stButton > button[kind="primary"] span,
.stLinkButton > a[kind="primary"] p, .stLinkButton > a[kind="primary"] span {
    color: var(--navy) !important;
}
.stButton > button[kind="primary"]:hover, .stLinkButton > a[kind="primary"]:hover {
    background: #F5F8FF !important;
    border-color: var(--navy-hover) !important;
    color: var(--navy-hover) !important;
}
.stButton > button[kind="primary"]:hover p, .stButton > button[kind="primary"]:hover span,
.stLinkButton > a[kind="primary"]:hover p, .stLinkButton > a[kind="primary"]:hover span {
    color: var(--navy-hover) !important;
}
/* Accent(포인트, 전체의 10~15%만 - 랜딩 등 특별히 강조하고 싶은 CTA
   1~2곳에만 선택적으로 쓴다) - container key에 "accent-btn"이 있으면
   적용. 지금은 기본 적용 대상 없음(호출부가 필요할 때 st.container
   (key="...-accent-btn")로 감싸면 켜짐). */
div[class*="accent-btn"] .stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #FBE7F0 0%, var(--pink) 60%, #F0C4D8 100%) !important;
    color: var(--navy) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 8px 18px rgba(201,127,160,0.28) !important;
}
div[class*="accent-btn"] .stButton > button[kind="primary"] p,
div[class*="accent-btn"] .stButton > button[kind="primary"] span { color: var(--navy) !important; }

/* ── 카드(우리 자체 클래스 - st.container(key="card-...")에 부여됨)
   2026-07-16: "하얀 네모"에서 국내 서비스 카드 느낌으로 - 아주 옅은
   테두리 + 아주 약한 그림자 + hover 시 살짝 떠오름(transform+그림자
   심화). 카드 전부에 hover를 걸면 클릭 불가능한 정적 카드까지 어색하게
   반응하므로, 실제로 hover 반응이 자연스러운 상호작용 카드 위주로
   건다(공고 카드/패널류). ─────────────────────────────────────────── */
div[class*="st-key-card-"] > div,
div[class*="st-key-panel-"] > div {
    background: var(--card-bg);
    border: 1px solid var(--divider);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: 1.25rem 1.4rem;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}
div[class*="st-key-card-"] {
    margin-bottom: 0.9rem;
}
div[class*="st-key-card-"]:hover > div {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* Streamlit 기본 bordered container도 카드처럼 (fallback) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--divider) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.18s ease, transform 0.18s ease;
}

/* ── 메트릭 카드 ────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
div[data-testid="stMetricValue"] { color: var(--accent) !important; font-weight: 700 !important; }

/* ── 진행바(2026-07-16 추가 - 기존엔 스타일 없이 Streamlit 기본
   회색 막대였다) - 네이비 그라데이션 채움 + 은은한 glow. ─────────── */
div[data-testid="stProgress"] div[role="progressbar"] > div {
    background: var(--divider) !important;
    border-radius: 999px !important;
    height: 8px !important;
}
div[data-testid="stProgress"] div[role="progressbar"] > div > div {
    background: linear-gradient(90deg, var(--navy-light) 0%, var(--navy) 100%) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 10px rgba(19, 43, 99, 0.35) !important;
}

/* ── 인풋 계열 ──────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--card-border) !important;
    background: var(--card-bg) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* 파일 업로더(홈 화면 이력서 업로드 카드 - 레퍼런스 ②번 패널 재현) -
   앱 전체에서 이 업로더 하나뿐이라 전역으로 덮어써도 안전하다. */
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    justify-content: flex-end;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important; /* "200MB per file..." - 카드 자체 설명 문구로 대체 */
}
/* 파일 선택 = Primary 액션(네이비 채움 + 흰 글씨) - 이력서 교체/연차
   수정(Secondary, 흰 배경+네이비 테두리)과 역할이 다르다. */
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    position: relative !important;
    height: 40px !important;
    min-width: 110px !important;
    border-radius: 12px !important;
    border: none !important;
    background: var(--navy) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    padding: 0 24px !important;
    box-shadow: none !important;
    transition: background .15s ease;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--navy-hover) !important;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
    display: none; /* 업로드 아이콘은 카드 상단 아이콘 배지로 대체 */
}
/* "파일 선택" 텍스트가 실제로는 이 p의 ::after(가상요소)라서, 버튼 자체를
   flex로 만드는 것만으로는 안 맞았다(2026-07-24 확인 1차: 세로는 맞았는데
   가로가 왼쪽 58px/오른쪽 29px로 어긋남 - display:none인 아이콘 자리를
   대신하던 숨은 wrapper가 flex 흐름에서 여전히 자리를 차지해 텍스트
   그룹 전체를 오른쪽으로 밀고 있었다). 형제 요소 영향을 아예 받지
   않도록 이 div를 버튼 전체를 덮는 절대위치로 두고 그 안에서만
   flex 중앙정렬한다. */
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] div[data-testid="stMarkdownContainer"] {
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] div[data-testid="stMarkdownContainer"] p {
    font-size: 0;
    margin: 0 !important;
    line-height: 1 !important;
}
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] div[data-testid="stMarkdownContainer"] p::after {
    content: "파일 선택";
    font-size: 0.95rem;
    line-height: 1;
    color: #FFFFFF !important;
}

/* ── 이력서 업로드 카드 헤더(아이콘+제목+설명) ───────────────────── */
.sc-upload-card { display: flex; align-items: center; gap: 14px; margin-bottom: 0.9rem; }
.sc-upload-icon-wrap { position: relative; display: inline-block; flex-shrink: 0; }
.sc-upload-star {
    position: absolute; top: -3px; right: -3px;
    background: var(--accent); color: white; border-radius: 50%;
    width: 18px; height: 18px; font-size: 10px; line-height: 18px; text-align: center;
    box-shadow: 0 0 0 2px var(--card-bg);
}
.sc-upload-title { font-weight: 700; font-size: 1.05rem; color: var(--text-primary); }
.sc-upload-desc { color: var(--text-secondary); font-size: 0.88rem; margin-top: 2px; }

/* ── 익스팬더 ───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--card-bg);
    box-shadow: none !important;
}

/* ── 탭(원본/맞춤화, 자소서 섹션 등 - 레퍼런스의 필세그먼트 스타일) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--accent-soft);
    padding: 4px;
    border-radius: 999px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    color: var(--text-secondary);
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: var(--shadow-sm);
}

/* ── 사이드바 ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] .stButton > button {
    border: none !important;
    border-radius: var(--radius-md) !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 600 !important;
    /* 2026-08-31 디자인 통일 - 사이드바 텍스트/아이콘은 전부 Navy 계열
       (비활성도 회색 아님). 활성/비활성 구분은 baby blue 배경으로만. */
    color: var(--navy-mid) !important;
    background: transparent !important;
    padding: 0.55rem 0.9rem !important;
}
[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stButton > button svg {
    color: var(--navy-mid) !important;
    fill: var(--navy-mid) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #EEF4FF !important;
    color: var(--accent-dark) !important;
    transform: none;
}
/* 현재 화면(활성) 네비 항목 - render_sidebar_nav()가 type="primary"로 표시.
   2026-08-31(사용자 확정) - 활성 배경을 연한 핑크(--accent-soft)에서
   아주 연한 Baby Blue 로 바꾼다(디자인 가이드: Active/Selected = Baby Blue,
   Navy 텍스트/아이콘). 다른 화면의 --accent-soft 용도(입력 포커스 링,
   expander 등)에는 영향 없게 이 사이드바 규칙만 하드코딩한다. */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #EAF2FF !important;
    color: var(--accent-dark) !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #DDEBFF !important;
    filter: none;
}
/* 버그 수정(2026-07-16): 위 전역 "버튼 kind=primary 내부 텍스트는 흰색"
   규칙이 사이드바의 활성 네비 항목(연한 핑크 배경 + 네이비 텍스트 조합)
   에도 적용돼 텍스트가 안 보이는 문제가 있었다 - 사이드바 안에서는
   다시 네이비로 되돌린다(선택자 등장 순서상 이 규칙이 나중에 오므로
   특이도가 같아도 이 값이 적용됨). */
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span {
    color: var(--accent-dark) !important;
}

/* ── 알림(info/warning/error/success) - 파스텔 톤으로 ───────────── */
div[data-testid="stAlertContainer"] {
    border-radius: var(--radius-md) !important;
    border: none !important;
}

/* ── 키보드 포커스 표시(접근성) ───────────────────────────────── */
.stButton > button:focus-visible, .stLinkButton > a:focus-visible,
.stTextInput input:focus-visible, .stTextArea textarea:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}

/* ── 로그인/시작 화면 전용(게스트 모드 중심 랜딩 - OAuth는 백엔드 없이
   UI만 자리를 잡아두고, 실제로는 게스트로 바로 진입시킨다) ───────── */
div[class*="st-key-auth-card-main"] {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 28px;
    box-shadow: var(--shadow-lift);
    padding: 3.6rem 3.2rem 2.8rem;
}
.sc-auth-logo {
    text-align: center; font-size: 2.4rem; font-weight: 800;
    letter-spacing: -0.02em; color: var(--text-primary); margin-bottom: 10px;
}
.sc-auth-logo span { color: var(--accent); }
.sc-auth-sub {
    text-align: center; color: var(--text-secondary); font-size: 1.1rem;
    margin-bottom: 2.2rem;
}
/* Guest 시작 버튼 - 첫 화면의 Primary CTA라 네이비 채움 + 흰 글씨로
   눈에 띄게 한다(이력서 교체/연차 수정 같은 Secondary 버튼과는
   역할이 다르다 - 2026-07-24 사용자 확정). */
div[class*="st-key-auth-card-main"] .stButton > button[kind="primary"],
div[class*="st-key-google-fallback-guest-btn"] button[kind="primary"] {
    width: 100%;
    height: 44px !important;
    min-height: 44px !important;
    border: none !important;
    border-radius: 12px !important;
    background: var(--navy) !important;
    color: #FFFFFF !important;
    box-shadow: 0 5px 14px rgba(22, 58, 112, 0.22) !important;
}
div[class*="st-key-auth-card-main"] .stButton > button[kind="primary"] p,
div[class*="st-key-auth-card-main"] .stButton > button[kind="primary"] span,
div[class*="st-key-google-fallback-guest-btn"] button[kind="primary"] p,
div[class*="st-key-google-fallback-guest-btn"] button[kind="primary"] span {
    color: #FFFFFF !important;
}
div[class*="st-key-auth-card-main"] .stButton > button[kind="primary"]:hover,
div[class*="st-key-google-fallback-guest-btn"] button[kind="primary"]:hover {
    background: var(--navy-hover) !important;
}
/* Google 버튼 - "작게"(보조 수단) - 텍스트 링크에 가깝게 축소 */
div[class*="st-key-google-btn-login"] { text-align: center; margin-top: 0.7rem; }
div[class*="st-key-google-btn-login"] button {
    background: transparent !important; border: none !important;
    color: var(--text-secondary) !important; font-weight: 500 !important;
    font-size: 0.92rem !important; padding: 0.3rem 0.6rem !important;
    box-shadow: none !important;
}
div[class*="st-key-google-btn-login"] button:hover {
    color: var(--accent) !important; transform: none;
}
.sc-auth-footer {
    text-align: center; color: var(--text-secondary); font-size: 0.92rem;
    margin-top: 1.3rem;
}

/* ── 홈 대시보드 인사말 헤더 ───────────────────────────────────────── */
.sc-home-greeting { font-size: 1.4rem; font-weight: 800; color: var(--text-primary); }
.sc-home-sub { color: var(--text-secondary); font-size: 0.92rem; margin: 2px 0 1.1rem; }

/* ── 진행 방식 선택 카드(전체 카드가 클릭 가능한 버튼처럼 동작) ──── */
.sc-mode-heading { font-weight: 700; font-size: 1.02rem; margin: 0.3rem 0 0.7rem; }
.sc-mode-card { text-align: center; padding: 1.4rem 0.6rem 0.4rem; }
.sc-mode-card .sc-icon-badge { margin: 0 auto 0.9rem; }
.sc-mode-title { font-weight: 700; font-size: 1rem; color: var(--text-primary); }
.sc-mode-desc { color: var(--text-secondary); font-size: 0.85rem; margin-top: 4px; line-height: 1.4; }
div[class*="st-key-mode-card-"] {
    position: relative; transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[class*="st-key-mode-card-"]:hover {
    transform: translateY(-2px); box-shadow: var(--shadow-md);
}
div[class*="st-key-mode-card-"] .stButton {
    position: absolute !important; inset: 0 !important; margin: 0 !important;
    width: 100% !important; height: 100% !important;
}
div[class*="st-key-mode-card-"] div[data-testid="stElementContainer"]:has(.stButton) {
    position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important;
}
div[class*="st-key-mode-card-"] .stButton button {
    width: 100%; height: 100%; opacity: 0; cursor: pointer; border-radius: var(--radius-lg) !important;
}

/* ── 최근 분석한 공고 ─────────────────────────────────────────────── */
.sc-recent-heading { font-weight: 700; font-size: 1.02rem; margin: 1.4rem 0 0.7rem; }

/* ── 스크롤바 ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--pink); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--pink-dark); }

/* ── 유틸리티: 배지 - pill + 작은 dot 마커 + 은은한 그라데이션
   (2026-07-16: "텍스트만 있는 밋밋한 뱃지"에서 폴리시업 - 국내 SaaS
   서비스들의 pill 뱃지는 보통 옅은 그라데이션 배경 + 상태 dot을
   같이 쓴다). 메인 흰색+네이비, 포인트 베이비핑크 + 성공(mint)/
   경고(peach)/정보(sky) + 추가 태그용 노랑/베이지. ────────────────── */
.sc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; white-space: nowrap;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
}
.sc-badge::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; opacity: 0.85; flex-shrink: 0;
}
.sc-badge-green { background: linear-gradient(160deg, var(--mint-soft) 0%, var(--mint) 100%); color: var(--mint-dark); }
.sc-badge-orange { background: linear-gradient(160deg, var(--peach-soft) 0%, var(--peach) 100%); color: var(--peach-dark); }
.sc-badge-red { background: linear-gradient(160deg, var(--red-soft) 0%, #FBD3D6 100%); color: var(--red); }
.sc-badge-gray { background: linear-gradient(160deg, #F7F8FA 0%, #EEF0F3 100%); color: var(--text-secondary); }
.sc-badge-blue { background: linear-gradient(160deg, var(--sky-soft) 0%, var(--sky) 100%); color: var(--sky-dark); }
.sc-badge-lavender { background: linear-gradient(160deg, var(--navy-soft) 0%, #DCE2F2 100%); color: var(--navy); }
.sc-badge-pink { background: linear-gradient(160deg, var(--pink-soft) 0%, var(--pink) 100%); color: var(--pink-dark); }
.sc-badge-yellow { background: linear-gradient(160deg, var(--yellow-soft) 0%, #FBEAB8 100%); color: #9C7A1A; }
.sc-badge-beige { background: linear-gradient(160deg, var(--beige-soft) 0%, #EAE0C8 100%); color: #7A6A48; }
.sc-badge-navy { background: linear-gradient(160deg, var(--navy-light) 0%, var(--navy-dark) 100%); color: white; }
.sc-badge-navy::before { background: white; }

/* ── 유틸리티: 아이콘 배지(2026-07-16 스퀴클 + 글로시 마감) - 폰 테마
   아이콘(둥근 사각형 "스퀴클" + 유리질 광택)의 "느낌"을 참고해서,
   기존 사각 배지보다 모서리를 더 둥글리고(스퀴클), 위쪽 유리 하이라이트
   (::after, 반투명 흰 타원)를 추가했다. 전체 배색 원칙(메인 흰색+
   네이비, 포인트 베이비핑크)은 그대로 유지 - 배지 안 그라데이션
   채도만 살짝 올려서 폰 테마 아이콘처럼 또렷한 파스텔로 보이게 한다
   (ui_icons.py의 _BADGE_GRADIENTS 참고). ─────────────────────────── */
.sc-icon-badge {
    position: relative;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 27%;
    overflow: hidden;
    box-shadow:
        inset 0 1px 1px rgba(255,255,255,0.85),
        inset 0 -8px 12px rgba(19,43,99,0.08),
        0 5px 14px rgba(19, 43, 99, 0.16);
    flex-shrink: 0;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.sc-icon-badge::after {
    content: "";
    position: absolute; top: 6%; left: 12%; width: 60%; height: 38%;
    background: radial-gradient(ellipse at center, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
    pointer-events: none;
}
.sc-icon-badge:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow:
        inset 0 1px 1px rgba(255,255,255,0.9),
        inset 0 -8px 12px rgba(19,43,99,0.08),
        0 8px 20px rgba(19, 43, 99, 0.22);
}

/* ── 유틸리티: 페이지 헤더 ─────────────────────────────────────── */
.sc-page-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 0.4rem;
}
.sc-page-header h1 { margin: 0 !important; }

/* ── 유틸리티: 원형 적합도 게이지(2026-07-16 glass 효과 추가) ───────
   단색 conic-gradient였던 걸 네이비 그라데이션(밝은~진한 톤)으로
   바꾸고, 유리질 하이라이트(::before, 좌상단 radial-gradient)를
   얹어서 "Glass" 느낌을 낸다 - 이미지 없이 순수 CSS. ───────────────── */
.sc-ring-wrap {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 0.2rem;
}
.sc-ring {
    position: relative;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: conic-gradient(from 0deg, var(--navy-light) 0%, var(--navy) calc(var(--pct) * 1%), var(--pink-soft) 0);
    box-shadow: 0 8px 20px rgba(19, 43, 99, 0.22);
}
.sc-ring::before {
    content: "";
    position: absolute; inset: 0; border-radius: 50%;
    background: radial-gradient(circle at 30% 22%, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0) 45%);
    pointer-events: none;
}
.sc-ring-inner {
    background: var(--card-bg);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    box-shadow: inset 0 1px 3px rgba(19, 43, 99, 0.08);
    position: relative;
}
.sc-ring-value {
    color: var(--navy); font-weight: 800; line-height: 1;
}
.sc-ring-label {
    color: var(--text-secondary); font-size: 0.75rem; margin-top: 2px;
}

/* ── 유틸리티: 회사 이니셜 아바타(레퍼런스의 컬러 로고 사각형 근사) ── */
.sc-avatar {
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: var(--radius-sm);
    color: white; font-weight: 700; flex-shrink: 0;
}

/* ── 진행 단계 바(이력서업로드→공고찾기→...→지원기록) ───────────── */
/* 2026-08-16 네 번째 수정(사용자 지적 - "전체적으로 2배 정도 키워줘") -
원/글자/간격을 전부 약 2배로 키운다(32px→60px 원, 0.8rem→1.5rem 숫자,
0.72rem→1.35rem 라벨) - 비율은 유지하고 크기만 키운다, 새 색/레이아웃
구조는 만들지 않는다. */
.sc-steps {
    display: flex; align-items: flex-start; gap: 8px;
    margin: 0.6rem 0 1.5rem;
    overflow-x: auto;
}
.sc-step {
    display: flex; flex-direction: column; align-items: center;
    flex: 1 1 0; min-width: 92px; position: relative;
}
.sc-step-circle {
    /* 2026-08-30(사용자 요청 - "2크기 정도 줄여줘") - 비율 유지, 약 70%로 축소
       (원 60→42px / 라벨 1.35→0.95rem / 선 위치도 함께). */
    width: 42px; height: 42px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; font-weight: 700;
    /* 2026-08-16 세 번째 수정(사용자 스펙 이미지 - "완료/현재/미완료 구분을
    명확히") - 두 번째 수정에서는 done/future를 구분 없이 전부 파스텔
    블루로 통일했었는데, 실제 화면으로 보니 3단계 구분이 안 보인다는
    피드백. 미완료(future) 단계는 흰 배경 + 네이비 테두리로 되돌리고,
    파스텔 블루는 done 전용으로 다시 분리한다. 색은 --accent-dark(=
    --navy-dark, #152C6B)가 아니라 var(--navy)를 직접 쓴다 - app.py가
    이 화면 전용으로 --navy만 #0F1F4A로 오버라이드하고 --navy-dark는
    안 건드려서, accent-dark를 쓰면 스펙이 지정한 정확한 네이비(#0F1F4A)
    대신 미묘하게 다른 색이 나오는 실측 버그가 있었다. */
    background: #FFFFFF; color: var(--navy);
    border: 2px solid var(--navy);
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.sc-step-label {
    font-size: 0.95rem; color: var(--text-secondary);
    margin-top: 7px; text-align: center; line-height: 1.2;
}
.sc-step-line {
    /* 2026-08-15(사용자 지적 - 연결선이 화면에 안 보임) - z-index:-1이
    이 absolute div를 조상 stacking context 뒤로 밀어내 흰 배경에
    가려지던 실측 버그. circle/label이 DOM상 이 다음에 오므로
    z-index 없이도 이미 선 위에 그려진다 - z-index:-1만 제거하면
    된다(다른 값 변경 없음).
    2026-08-16(사용자 지적 - "선도 네이비로") - 연결선 색을 옅은 회색
    대신 네이비로 통일한다(완료/미완료 구분 없이 전부).
    2026-08-16 세 번째 수정(사용자 지적 - "선이 원을 침범") - width:100%
    가 자기 원 중심(50%)에서 다음 원 중심(150%)까지 꽉 채워서, 원의
    반지름(16px)만큼 선이 원 밑으로 파고들어 있었다. 양쪽 다 반지름만큼
    안쪽으로 들여서 원의 테두리에서 시작/끝나도록 한다. */
    position: absolute; top: 20px; left: calc(50% + 21px); width: calc(100% - 42px); height: 2px;
    background: var(--navy);
}
/* 2026-08-31 - .sc-step-line 은 "이 step -> 다음 step" 연결선이다(자기
   원 오른쪽에서 시작). 그래서 :first-child 를 숨기면 1->2 선이 사라지는
   버그가 있었다(_STEP2_CSS 가 화면별로 되돌리던 걸 전역에서 바로잡음).
   숨겨야 하는 건 마지막 step 뒤로 빈 공간에 뻗는 :last-child 선뿐이다. */
.sc-step:last-child .sc-step-line { display: none; }
.sc-step-done .sc-step-circle { background: #E9F0FF; color: var(--navy); border-color: transparent; }
.sc-step-current .sc-step-circle { background: var(--navy); color: white; border-color: transparent; }
.sc-step-current .sc-step-label { color: var(--text-primary); font-weight: 700; }

/* STEP1(이력서 업로드 화면)의 CSS는 이 파일에 없다 - 사용자가 직접
   작성해 전달한 참조 코드를 "그대로" 이식해달라고 명시적으로 요청해서
   (2026-07-23), app.py의 render_s0()에 그 코드의 <style> 블록을 그
   문법·클래스명·값 그대로 옮겨 별도 주입한다. 이 파일의 다른 화면
   공통 시스템과 섞으면 "내 마음대로 바꾼 것"이 되므로 분리해뒀다. */

/* 도전공고 토글(st.toggle) - 기본 Streamlit 테마의 on-색상(#FF4B4B,
   레드)이 이 프로젝트의 네이비 배색과 충돌해서 덮어쓴다. 실측(2026-07-23,
   javascript_tool으로 DOM 확인) - 켜졌을 때 label에 data-selected="true"
   가 붙고, 트랙(배경)은 label의 첫 번째 div 자식이다. emotion 해시
   클래스(ew2p8o5 등)는 빌드마다 바뀔 수 있어 의존하지 않는다. */
div[class*="st-key-challenge-toggle"] [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {
    background: var(--navy) !important;
}

/* "+ 더 보기" / "이력서 전체 요약 보기" 같은 텍스트 링크형 버튼 -
   테두리/배경 없이 파란 텍스트만 보이게(스크린샷 재현). */
div[class*="st-key-linkbtn-"] .stButton > button {
    border: none !important; background: transparent !important;
    color: var(--sky-dark) !important; font-weight: 600 !important;
    padding: 0.1rem 0 !important; box-shadow: none !important;
    font-size: 0.85rem !important;
}
div[class*="st-key-linkbtn-"] .stButton > button p,
div[class*="st-key-linkbtn-"] .stButton > button span { color: var(--sky-dark) !important; }
div[class*="st-key-linkbtn-"] .stButton > button:hover {
    text-decoration: underline; transform: none; background: transparent !important;
}

/* ── 모바일 대응 ────────────────────────────────────────────────── */
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    div[class*="st-key-card-"] > div { padding: 1rem; }
    h1 { font-size: 1.4rem !important; }
}
"""


def inject_theme() -> None:
    st.markdown(f"<style>{_FONT_IMPORT}{_CSS}</style>", unsafe_allow_html=True)


def page_header(icon_html: str, title: str, subtitle: str = "") -> None:
    """아이콘 배지 + 제목을 한 줄에 렌더링한다(주요 화면 상단에만
    사용 - 요청사항: 아이콘은 과도하게 쓰지 않는다).

    2026-09-03(사용자 확정) - 제목 아래 회색 한 줄 설명(subtitle)은 전
    화면에서 제거한다("굳이 필요 없다"). subtitle 인자는 호출부 호환을
    위해 남겨두되 렌더링하지 않는다."""
    st.markdown(
        f'<div class="sc-page-header">{icon_html}<div><h1>{title}</h1></div></div>',
        unsafe_allow_html=True,
    )


def badge_html(text: str, tone: str = "gray") -> str:
    return f'<span class="sc-badge sc-badge-{tone}">{text}</span>'


def chip_html(text: str) -> str:
    """dot 마커 없는 담백한 pill(주요 기술 Chip 전용, 2026-07-23 -
    레퍼런스 사진 재현). badge_html은 상태 뱃지용(dot 마커 있음)이라
    기술 태그 나열에는 안 맞는다."""
    return f'<span class="sc-chip">{text}</span>'


_AVATAR_COLORS = [
    "#EF5D7D", "#5C8DEF", "#7C5CFC", "#12A57E", "#E8A23D",
    "#3DBEEF", "#EF8B5C", "#9B5CEF", "#5CEFAF", "#EF5C9B",
]


def avatar_html(name: str, size: int = 40) -> str:
    """회사명 등 첫 글자를 색상 사각형 아바타로 표시한다(레퍼런스의
    회사 로고 아이콘 자리를 대신함 - 실제 브랜드 로고가 없으므로
    지어내지 않고 이니셜+색상으로 근사한다). 이름 해시로 색을 정해서
    같은 이름은 항상 같은 색이 나온다."""
    initial = (name or "?").strip()[:1].upper()
    color = _AVATAR_COLORS[hash(name or "") % len(_AVATAR_COLORS)]
    font_size = int(size * 0.42)
    return (
        f'<div class="sc-avatar" style="width:{size}px;height:{size}px;'
        f'background:{color};font-size:{font_size}px;">{initial}</div>'
    )


def ring_chart(percent: float, size: int = 120, label: str = "", value_text: str = "") -> str:
    """레퍼런스의 원형 적합도 게이지(도넛 차트)를 CSS conic-gradient로
    근사한다 - 이미지/차트 라이브러리 없이 순수 CSS만 사용. percent는
    0~100 사이 값."""
    pct = max(0, min(100, percent))
    text = value_text or f"{int(round(pct))}%"
    inner_size = int(size * 0.78)
    value_font = int(size * 0.22)
    return (
        f'<div class="sc-ring-wrap">'
        f'<div class="sc-ring" style="width:{size}px;height:{size}px;--pct:{pct};">'
        f'<div class="sc-ring-inner" style="width:{inner_size}px;height:{inner_size}px;">'
        f'<span class="sc-ring-value" style="font-size:{value_font}px;">{text}</span>'
        f'{f"<span class=\'sc-ring-label\'>{label}</span>" if label else ""}'
        f'</div></div></div>'
    )


_STEP_LABELS = ["이력서 업로드", "공고 찾기", "공고 분석", "지원 준비", "지원 기록"]


def step_progress_html(current_index: int) -> str:
    """상단 진행 단계 바. current_index는 0-based(0=이력서 업로드,
    1=공고 찾기, 2=공고 분석, 3=지원 준비, 4=지원 기록).
    2026-08-31(사용자 확정) - 5단계로 정리. 4="이력서 수정"→"지원 준비",
    5="자소서 준비"→"지원 기록", 6(구 "지원 기록") 삭제. 자소서는 지원
    준비의 선택 하위 흐름이라 별도 단계로 두지 않는다. 이 바를 이력서
    업로드~지원 기록 전 화면 최상단에 표시한다(render_s1/s1d/s4/s7/s8).
    2026-08-15 - "매칭 결과" 죽은 단계 제거. 2026-08-16 - "자소서 준비"
    추가했다가 2026-08-31 다시 제거."""
    parts = ['<div class="sc-steps">']
    for i, label in enumerate(_STEP_LABELS):
        state = "sc-step-current" if i == current_index else ("sc-step-done" if i < current_index else "")
        parts.append(
            f'<div class="sc-step {state}"><div class="sc-step-line"></div>'
            f'<div class="sc-step-circle">{i + 1}</div>'
            f'<div class="sc-step-label">{label}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)
