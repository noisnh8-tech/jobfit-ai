"""
resume_input/reason_phrasing.py

**이 파일은 최종 Recommendation Explanation Engine이 아니다 - 지금
있는 Retrieval(matched_keywords, 키워드 겹침 기반) 결과를 보여주기
위한 임시 Presentation Layer일 뿐이다.** 최종 목표는 Problem/
Thinking/Experience/Skill 같은 의미(Meaning) 기반 설명이고, 이
파일은 그 엔진이 준비되기 전까지 화면에 "추천 이유"를 채우기 위한
가림막이다. 그래서 이 파일은 판단 로직(candidate_search.py,
career_filter.py 등)에 대한 의존을 최소화하고 - matched_keywords와
job_sections라는 이미 계산된 값만 받아서 문구만 조립한다 - 나중에
Meaning 기반 엔진으로 통째로 교체될 때 이 파일만 들어내면 되도록
독립성을 유지한다. `format_recommendation_reasons()`를 호출하는
쪽(pipeline.py)도 이 함수 하나만 알면 되고 내부 구현(키워드 매핑
등)에 의존하지 않는다.

matched_keywords(Resume∩JD 토큰 겹침, LLM 미사용)를 키워드 그대로
나열하지 않고 "카테고리 문구"로 치환해서 사용자 친화적으로 보여준다.
새 LLM 호출 없음 - 매핑 테이블 + 규칙 기반.

각 키워드가 JD의 자격요건/우대사항(req/pref) 블록에 등장하면
"✓ {카테고리} 요구사항 충족", 그렇지 않으면(주요업무/기타에서만
겹치는 경우) "✓ {카테고리} 경험 일치"로 구분한다 - 완전한 의미 기반
그룹핑(예: Tableau+PowerBI를 "데이터 시각화" 하나로 묶는 것)은
매핑 테이블 커버리지에 의존한다. 매핑에 없는 키워드는 원문 그대로
폴백한다("경험"을 라벨에 미리 넣지 않는다 - "경험 일치" 문구와
합쳐질 때 "구축 경험 경험 일치"처럼 "경험"이 중복되는 문제를
실측으로 발견해서 라벨에서는 뺐다). 실제 매칭되는 키워드 분포를
보며 넓혀간다(ui_design.md §6).
"""
from __future__ import annotations

import re

# candidate_search._matched_keywords()의 _EXPLAIN_STOPWORDS는 건드리지
# 않는다(Retrieval 관련 로직은 닫혔음) - 대신 이 표시 레이어에서만,
# "요구사항 충족" 문장으로 감싸면 유독 어색해지는 일반 단어를 추가로
# 거른다(실측: "기반 경험 요구사항 충족", "경력 경험 요구사항 충족"
# 같은 문장이 실제로 생성되는 걸 확인하고 추가함).
_DISPLAY_STOPWORDS = {"경력", "기반", "해당", "이런", "가지", "여러", "다양"}

# 순수 숫자 토큰("2022"/"01" 등, 이력서 재직기간 날짜가 JD 본문의
# 다른 숫자와 우연히 겹쳐서 matched_keywords에 섞여 들어온 경우)도
# 추천 이유로 의미가 없다 - 실측으로 발견("2022 경험 일치" 같은
# 문장이 실제로 생성됨).
_NUMERIC_RE = re.compile(r"^\d+$")

_KEYWORD_CATEGORY_MAP: dict[str, str] = {
    "sql": "SQL 활용 능력",
    "python": "Python 활용 능력",
    "tableau": "데이터 시각화",
    "powerbi": "데이터 시각화",
    "looker": "데이터 시각화",
    "kpi": "KPI 분석",
    "대시보드": "대시보드 구축",
    "분석": "데이터 분석",
    "데이터": "데이터 분석",
    "마케팅": "마케팅 분석",
    "crm": "CRM 분석",
    "실험": "실험 설계",
    "지표": "지표 설계",
    "리포트": "리포팅",
    "리포팅": "리포팅",
    "통계": "통계 분석",
}


def _label_for(keyword: str) -> str:
    return _KEYWORD_CATEGORY_MAP.get(keyword.lower(), keyword)


def format_recommendation_reasons(matched_keywords: list[str], job_sections: dict, max_items: int = 4) -> list[str]:
    req_pref_text = ((job_sections.get("req") or "") + " " + (job_sections.get("pref") or "")).lower()

    requirement_hits: list[str] = []
    experience_hits: list[str] = []
    seen_labels: set[str] = set()
    for kw in matched_keywords:
        if kw.lower() in _DISPLAY_STOPWORDS or _NUMERIC_RE.match(kw):
            continue
        label = _label_for(kw)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        if kw.lower() in req_pref_text:
            requirement_hits.append(f"✓ {label} 요구사항 충족")
        else:
            experience_hits.append(f"✓ {label} 경험 일치")

    return (requirement_hits + experience_hits)[:max_items]


def group_recommendation_reasons(matched_keywords: list[str], job_sections: dict) -> dict[str, list[str]]:
    """상세보기용 - matched_keywords를 실제로 매칭이 발견된 JD 섹션
    기준 3개 그룹(자격요건/우대사항/기술 역량)으로 나눈다. 전부 이미
    존재하는 신호(job_sections의 req/pref 분류 - candidate_search.py가
    검색에도 쓰는 바로 그 분류)에서 파생된다.

    "프로젝트 경험"/"문제 해결 경험" 같은 그룹은 만들지 않는다 - 현재
    matched_keywords는 Resume-JD 토큰 겹침일 뿐이라 어떤 매칭이
    "프로젝트 경험"인지 "문제 해결 경험"인지 판단할 근거가 없다.
    UI를 위해 존재하지 않는 판단을 지어내지 않기 위해 이 두 그룹은
    의도적으로 생략했다(Explainability 점검 결과 - ui_design.md 참고).
    """
    req_text = (job_sections.get("req") or "").lower()
    pref_text = (job_sections.get("pref") or "").lower()

    groups: dict[str, list[str]] = {"자격요건": [], "우대사항": [], "기술 역량": []}
    seen_labels: set[str] = set()
    for kw in matched_keywords:
        if kw.lower() in _DISPLAY_STOPWORDS or _NUMERIC_RE.match(kw):
            continue
        label = _label_for(kw)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        if kw.lower() in req_text:
            groups["자격요건"].append(label)
        elif kw.lower() in pref_text:
            groups["우대사항"].append(label)
        else:
            groups["기술 역량"].append(label)

    return {k: v for k, v in groups.items() if v}
