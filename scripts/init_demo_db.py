# -*- coding: utf-8 -*-
"""
scripts/init_demo_db.py — 공개 데모 DB(data/demo.db) 초기화 스크립트.

이 스크립트는 실제 운영 DB(job_ai_v3/data/jobs.db)를 전혀 읽지 않는다.
아래 상수(POSTINGS/APPLICATIONS/RESUME_UNDERSTANDING 등)에 처음부터
새로 작성한 비식별 예시 데이터가 코드에 직접 들어있고, 이 파일 하나만
실행하면 그 데이터로 data/demo.db 를 만든다 — 운영 DB 경로/스키마를
몰라도, 그리고 실제 운영 DB가 아예 없어도 그대로 동작한다.

실행:
    python scripts/init_demo_db.py

무엇을 만드는가:
  - candidate_jobs   : 공개 데모용 공고 4건(회사·직무·본문 전부 새로 작성 - 가상)
  - applications      : 공개 데모용 지원기록 4건(위 공고와 별개로 새로 작성 - 가상)
  - application_status_history
  - resume_understanding_cache : 가상 지원자 프로필 1건
  - semantic_link_cache        : 공고 4건 × 가상 지원자 분석 결과(사전 계산)
  - preparation_sessions/events, collection_logs (화면이 비어 보이지 않게 소량)
  - data/last_resume/resume.pdf : 가상 지원자 이력서 PDF(자동 복원용, data/ 는 git 추적 제외)

공고·회사·직무·지원기록·결과·이력서 전부 이 스크립트에서 처음부터 새로
작성한 가상 데이터다(2026-09-06 방침 변경 - 실제 이력서 파일은 더 이상
공개 저장소/데모 DB 어디에도 사용하지 않는다). 가상 이력서 PDF는
`_build_demo_resume_pdf_bytes()`가 실행 시점에 PyMuPDF 내장 한국어
폰트("korea-s")로 그 자리에서 생성한다 - 저장소에 커밋되는 파일이 아니라
매번 새로 만들어지는 바이트열이다. 프로젝트 3건의 이름(AI 기반 취업
의사결정 시스템/서울 Airbnb 호스트 수익 최적화 가이드/리워드 비용
최적화 전략)만 실제 이력서·포트폴리오·README에 실린 것과 동일하게
재사용한다 - 이 프로젝트명·요약 설명은 README/포트폴리오에도 공개된
내용이라 재사용 가능(사용자 확정, 개인정보 아님).

지원 판단 결과(2-1-1)는 임의로 정한 라벨이 아니라, 실제
judge_engine/eligibility_compare 로직 자체에 위 가상 이력서와 아래 가상
공고 데이터를 직접 넣어 재현되는지 확인한 뒤 만든 값이다.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from resume_input.runtime_mode import DB_PATH, IS_DEMO  # noqa: E402
from resume_input import job_store, application_manager, last_resume  # noqa: E402
from resume_input.quick_analysis import SCHEMA_VERSION  # noqa: E402
from resume_input.understanding import RESUME_REPRESENTATION_VERSION  # noqa: E402

if not IS_DEMO:
    raise SystemExit(
        "JOBFIT_MODE=local 상태에서는 이 스크립트를 실행하지 않습니다 "
        "(실운영 jobs.db를 건드리지 않기 위한 안전장치)."
    )

# 가상 지원자 이력서 원문 - 실제 인물과 무관. "경력" 섹션이 없어
# resume_career.extract_user_career_level()이 자동으로 "신입"으로
# 분류하고(§docstring 참고), "학력" 섹션의 "전문학사"는
# resume_facts.extract_education()이 DEGREE_RANK 1로 인식한다(데모
# 공고 4건 중 "4년제 대학교 학사 학위 이상 필수" 요건에 미달 - 실제
# judge_engine 재현 결과와 동일하게 유지하기 위한 의도적 설정).
DEMO_RESUME_TEXT = """데모 지원자
데이터 분석가

보유 기술
Python, SQL, LLM/AI, n8n, Streamlit, Excel

프로젝트
1. AI 기반 취업 의사결정 시스템
채용공고 탐색부터 지원 판단, 자료 준비, 결과 관리까지 연결한 개인 프로젝트

2. 서울 Airbnb 호스트 수익 최적화 가이드
운영 데이터 분석 및 수익 예측 대시보드 구현 (3인팀)

3. 리워드 비용 최적화 전략
구매 행동 기반 리워드 운영 전략 설계 (5인팀)

학력
전문학사 졸업

교육
데이터 분석 부트캠프 수료 (약 1,080시간)
"""

# resume_input.pdf_parser.extract_text_from_pdf() 로 위 DEMO_RESUME_TEXT를
# 넣어 만든 PDF를 재추출한 결과의 sha256 앞 16자리(고정값 - 재계산해도
# 같은 텍스트면 항상 이 값이 나온다. verify_demo_resume_hash()로 검증 가능).
DEMO_RESUME_HASH = "d6da0efb846c8e09"


def _build_demo_resume_pdf_bytes() -> bytes:
    """가상 지원자 이력서를 그 자리에서 PDF로 렌더링한다. 외부 폰트/
    Chrome/PowerPoint 없이 PyMuPDF 내장 CJK 폰트("korea-s")만 쓰므로
    Windows/Mac/Linux 어디서나 동작한다 - 공개 데모의 핵심 화면(공고
    찾기/분석/지원 판단)은 OS에 상관없이 그대로 재현되어야 하기 때문."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 545, 792)
    page.insert_textbox(rect, DEMO_RESUME_TEXT, fontname="korea-s", fontsize=11, lineheight=1.6)
    return doc.tobytes()


def verify_demo_resume_hash() -> None:
    """DEMO_RESUME_HASH가 실제 생성되는 PDF의 추출 텍스트 해시와
    일치하는지 확인한다(불일치면 resume_understanding_cache/
    semantic_link_cache가 캐시 미스로 LLM을 재호출하려다 데모 모드
    가드에 막혀 에러가 난다 - 조용히 넘어가지 않고 여기서 바로 멈춘다)."""
    import io

    from resume_input.pdf_parser import extract_text_from_pdf
    from resume_input.execution_logger import resume_hash as _compute_hash

    pdf_bytes = _build_demo_resume_pdf_bytes()
    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    actual = _compute_hash(text)
    if actual != DEMO_RESUME_HASH:
        raise SystemExit(
            f"DEMO_RESUME_HASH 불일치: 코드값={DEMO_RESUME_HASH} 실제값={actual} "
            "- DEMO_RESUME_TEXT를 바꿨다면 이 파일 상단의 DEMO_RESUME_HASH도 갱신할 것."
        )

# ── 1) 데모 공고 4건 (회사명·직무명·본문 전부 새로 작성, 실제 채용공고 아님) ──
POSTINGS = [
    {
        "job_id": "demo-001", "company": "A커머스", "title": "데이터 분석가",
        "career_level": "1~3년", "location": "서울",
        "posting_text": (
            "[담당업무]\n"
            "- 커머스 서비스 지표(구매전환/리텐션 등) 분석 및 대시보드 운영\n"
            "- SQL/Python 기반 데이터 추출·가공 및 정기 리포트 작성\n\n"
            "[자격요건]\n"
            "- SQL, Python 활용 데이터 분석 경험\n"
            "- 통계적 가설 검정에 대한 이해\n\n"
            "[우대사항]\n"
            "- 이커머스 도메인 실무 경험\n"
            "- 대시보드/시각화 도구 활용 경험"
        ),
        "judgment_expected": "지원",
    },
    {
        "job_id": "demo-002", "company": "B테크", "title": "AI 서비스 기획",
        "career_level": "신입", "location": "서울",
        "posting_text": (
            "[담당업무]\n"
            "- LLM 기반 신규 기능 기획 및 실험 설계\n"
            "- 반복 업무 자동화를 위한 워크플로 설계·운영\n\n"
            "[자격요건]\n"
            "- LLM/생성형 AI에 대한 이해와 실제 적용 경험\n"
            "- 문제 정의부터 기능 설계까지 주도적으로 수행한 경험\n\n"
            "[우대사항]\n"
            "- 기획 직무 실무 경력"
        ),
        "judgment_expected": "지원",
    },
    {
        "job_id": "demo-003", "company": "C플랫폼", "title": "Product Analyst",
        "career_level": "1~3년", "location": "서울",
        "posting_text": (
            "[담당업무]\n"
            "- 프로덕트 지표 분석 및 실험(A/B 테스트) 설계\n"
            "- SQL/Python 기반 데이터 분석\n\n"
            "[자격요건]\n"
            "- SQL, Python 데이터 분석 역량\n"
            "- 실험 설계 및 결과 해석 경험\n\n"
            "[우대사항]\n"
            "- 영어 커뮤니케이션 가능자 우대(TOEIC 700점 이상)"
        ),
        "judgment_expected": "보류",
    },
    {
        "job_id": "demo-004", "company": "D서비스", "title": "데이터 운영",
        "career_level": "5년+", "location": "서울",
        "posting_text": (
            "[담당업무]\n"
            "- 데이터 파이프라인 운영 및 품질 관리\n"
            "- 데이터 이슈 대응 및 개선\n\n"
            "[자격요건]\n"
            "- 관련 직무 경력 5년 이상 필수\n"
            "- 4년제 대학교 학사 학위 이상 필수\n"
            "- Python/SQL 활용 능력\n\n"
            "[우대사항]\n"
            "- 대용량 데이터 운영 경험"
        ),
        "judgment_expected": "미지원",
    },
]

# quick_analysis(v2) 스키마 그대로 - requirement/type/relation/gap_type/
# short_label/resume_evidence. resume_evidence는 전부 공개 이력서·포트폴리오에
# 이미 있는 프로젝트 서술(JobFit AI/Airbnb/Starbucks 리워드)만 재사용.
ANALYSIS_RESULTS = {
    "demo-001": {
        "job_core": "커머스 서비스의 핵심 지표를 분석하고 대시보드로 관리하며, SQL/Python 기반 데이터 분석 업무를 수행하는 직무입니다.",
        "role_context": {
            "purpose": "서비스 지표를 분석해 운영 의사결정을 지원",
            "domain": "이커머스",
            "main_tasks": ["지표 분석", "대시보드 운영", "정기 리포트 작성"],
            "important_capabilities": ["SQL/Python 분석", "통계적 검증"],
        },
        "requirements": [
            {"requirement": "SQL과 Python을 활용한 데이터 분석 및 지표 설계 경험", "type": "skill",
             "relation": "match", "gap_type": None, "short_label": "SQL·Python 데이터 분석",
             "resume_evidence": "Starbucks 리워드 비용 최적화 프로젝트에서 Python/Pandas로 Net Lift Index 등 지표를 설계·분석함"},
            {"requirement": "가설 수립부터 통계적 검증까지 분석 프로세스 수행 경험", "type": "task",
             "relation": "match", "gap_type": None, "short_label": "통계 기반 분석 프로세스",
             "resume_evidence": "Airbnb 프로젝트에서 운영 요소별 수익 차이를 통계적으로 검증함"},
            {"requirement": "분석 결과를 대시보드로 시각화해 의사결정에 활용한 경험", "type": "task",
             "relation": "match", "gap_type": None, "short_label": "분석 결과 시각화·의사결정 도구화",
             "resume_evidence": "Airbnb 프로젝트에서 진단→전략선택→예상수익비교로 이어지는 의사결정 대시보드를 Streamlit으로 구현함"},
            {"requirement": "이커머스 도메인 실무 경험 우대", "type": "domain",
             "relation": "partial", "gap_type": "domain_gap", "short_label": "이커머스 도메인 신규",
             "resume_evidence": "이커머스 도메인 실무 경험은 없으나 유사한 소비자 행동·운영 데이터 분석 경험 보유"},
        ],
        "resume_order": {
            "project_priority": ["리워드 비용 최적화 전략", "서울 Airbnb 호스트 수익 최적화 가이드", "AI 기반 취업 의사결정 시스템"],
            "skill_priority": ["SQL", "Python / Pandas", "Excel", "n8n", "Streamlit", "LLM / AI"],
        },
    },
    "demo-002": {
        "job_core": "LLM 기반 신규 기능을 기획하고, 반복 업무를 자동화하는 워크플로를 설계·운영하는 직무입니다.",
        "role_context": {
            "purpose": "LLM 기반 기능 기획 및 자동화 워크플로 설계",
            "domain": "AI 서비스 기획",
            "main_tasks": ["기능 기획", "실험 설계", "워크플로 자동화"],
            "important_capabilities": ["LLM 활용", "주도적 기능 설계"],
        },
        "requirements": [
            {"requirement": "LLM 기반 기능 기획 및 실제 서비스 적용 경험", "type": "skill",
             "relation": "match", "gap_type": None, "short_label": "LLM 기반 서비스 기획",
             "resume_evidence": "LLM으로 채용공고 요구사항과 이력서 경험을 의미 단위로 연결하는 기능을 설계·구현함"},
            {"requirement": "반복 업무를 자동화하는 워크플로 설계 경험", "type": "task",
             "relation": "match", "gap_type": None, "short_label": "업무 자동화 워크플로 설계",
             "resume_evidence": "n8n으로 데이터 수집·처리·결과 반영 등 반복 업무를 자동화함"},
            {"requirement": "AI 기능과 규칙 기반 로직의 역할을 구분해 설계한 경험", "type": "task",
             "relation": "match", "gap_type": None, "short_label": "AI·Rule 역할 분리 설계",
             "resume_evidence": "LLM이 지원 여부를 직접 결정하지 않고, 연결 근거와 조건을 바탕으로 Rule이 최종 판단하도록 역할을 분리해 설계함"},
            {"requirement": "기획 실무 경력 우대", "type": "experience",
             "relation": "partial", "gap_type": "experience_gap", "short_label": "기획 직무 경력 없음(개인 프로젝트로 대체)",
             "resume_evidence": "기획 직무로 근무한 경력은 없으나, 개인 프로젝트에서 문제 정의부터 기능 설계·구현까지 기획자 역할을 직접 수행함"},
        ],
        "resume_order": {
            "project_priority": ["AI 기반 취업 의사결정 시스템", "서울 Airbnb 호스트 수익 최적화 가이드", "리워드 비용 최적화 전략"],
            "skill_priority": ["LLM / AI", "n8n", "Streamlit", "Python / Pandas", "SQL", "Excel"],
        },
    },
    "demo-003": {
        "job_core": "프로덕트 지표를 분석하고 실험(A/B 테스트)을 설계하는 직무입니다.",
        "role_context": {
            "purpose": "프로덕트 지표 분석 및 실험 설계",
            "domain": "플랫폼 서비스",
            "main_tasks": ["지표 분석", "실험 설계"],
            "important_capabilities": ["SQL/Python 분석", "실험 설계"],
        },
        "requirements": [
            {"requirement": "SQL/Python 기반 데이터 분석 역량", "type": "skill",
             "relation": "match", "gap_type": None, "short_label": "SQL·Python 분석 역량",
             "resume_evidence": "Python/Pandas/SQL 기반 데이터 분석 프로젝트 다수 수행"},
            {"requirement": "실험 설계 및 A/B 테스트 경험", "type": "task",
             "relation": "partial", "gap_type": "experience_gap", "short_label": "그룹 비교 시뮬레이션 경험으로 대체",
             "resume_evidence": "formal A/B 테스트 실행 경험은 없으나, Starbucks 프로젝트에서 그룹 비교·시뮬레이션 방식으로 효과를 검증함"},
            {"requirement": "영어 커뮤니케이션 가능자 우대(TOEIC 700점 이상)", "type": "qualification",
             "relation": "no_match", "gap_type": "qualification_gap", "short_label": None,
             "resume_evidence": None},
        ],
        "resume_order": {
            "project_priority": ["리워드 비용 최적화 전략", "서울 Airbnb 호스트 수익 최적화 가이드", "AI 기반 취업 의사결정 시스템"],
            "skill_priority": ["SQL", "Python / Pandas", "Excel", "LLM / AI", "n8n", "Streamlit"],
        },
    },
    "demo-004": {
        "job_core": "데이터 파이프라인을 운영하고 품질을 관리하는 직무입니다.",
        "role_context": {
            "purpose": "데이터 운영 및 품질 관리",
            "domain": "데이터 운영",
            "main_tasks": ["파이프라인 운영", "품질 관리"],
            "important_capabilities": ["Python/SQL 활용", "5년 이상 실무 경력"],
        },
        "requirements": [
            {"requirement": "데이터 운영 및 품질 관리 업무 경험", "type": "task",
             "relation": "match", "gap_type": None, "short_label": "데이터 품질 관리 경험",
             "resume_evidence": "데이터 정합성 확인 및 품질 개선 작업을 프로젝트 내에서 수행함"},
            {"requirement": "데이터 분석 도구(Python/SQL) 활용 능력", "type": "skill",
             "relation": "match", "gap_type": None, "short_label": "Python·SQL 활용",
             "resume_evidence": "Python/SQL 기반 분석 프로젝트 다수 수행"},
            {"requirement": "관련 직무 경력 5년 이상 필수", "type": "experience",
             "relation": "no_match", "gap_type": "experience_gap", "short_label": None,
             "resume_evidence": None},
            {"requirement": "4년제 대학교 학사 학위 이상 필수", "type": "qualification",
             "relation": "no_match", "gap_type": "qualification_gap", "short_label": None,
             "resume_evidence": None},
        ],
        "resume_order": {
            "project_priority": ["리워드 비용 최적화 전략", "서울 Airbnb 호스트 수익 최적화 가이드", "AI 기반 취업 의사결정 시스템"],
            "skill_priority": ["SQL", "Python / Pandas", "Excel", "n8n", "Streamlit", "LLM / AI"],
        },
    },
}

# ── 2) 데모 지원 기록 4건 - 위 공고와는 완전히 별개로 새로 작성 ──
# (실제 지원 기록을 회사명만 바꾼 게 아니라, 상태값 예시를 보여주기 위해
#  처음부터 새로 만든 항목. 회사명·직무명·날짜·상태·메모·ID 전부 신규.)
# 주의: 위 POSTINGS(A~D커머스/테크/플랫폼/서비스)와 회사명이 겹치면 "이미
# 지원한 공고 제외" 로직(posting_key = 회사+정규화 제목)이 실제로 작동해서
# 데모 공고 목록에서 사라진다 - 일부러 다른 회사명을 쓴다(지원 기록은
# 공고 목록과 독립적인 완전히 새 예시라는 요구사항과도 일치).
APPLICATIONS = [
    {"job_id": "demo-app-001", "company": "E커머스", "title": "데이터 분석가", "status": "지원 완료",
     "memo": "데모 예시 - 실제 지원 기록 아님", "source": "demo"},
    {"job_id": "demo-app-002", "company": "F테크", "title": "AI 서비스 기획", "status": "서류 진행",
     "memo": "데모 예시 - 실제 지원 기록 아님", "source": "demo"},
    {"job_id": "demo-app-003", "company": "G플랫폼", "title": "Product Analyst", "status": "면접",
     "memo": "데모 예시 - 실제 지원 기록 아님", "source": "demo"},
    {"job_id": "demo-app-004", "company": "H서비스", "title": "데이터 운영", "status": "불합격",
     "memo": "데모 예시 - 실제 지원 기록 아님", "source": "demo"},
]

# ── 3) 가상 지원자 이력서 이해 캐시 (프로젝트명·요약은 README/포트폴리오에
#      공개된 내용만 재사용 - 개인정보 아님, §모듈 docstring 참고) ──
RESUME_SEMANTIC_OBJECTS = [
    {"layer": "skill", "normalized_text": "SQL/Python 데이터 분석", "importance": "core",
     "confidence": "high", "persona_tags": ["분석가"],
     "evidence": {"project": "리워드 비용 최적화 전략", "section": "기술",
                  "source_text": "Python, Pandas 기반 구매 행동 지표 설계 및 분석"}},
    {"layer": "task", "normalized_text": "통계적 가설 검정", "importance": "core",
     "confidence": "high", "persona_tags": ["분석가"],
     "evidence": {"project": "서울 Airbnb 호스트 수익 최적화 가이드", "section": "해결 과정",
                  "source_text": "운영 요소별 수익 차이를 통계적으로 검증"}},
    {"layer": "task", "normalized_text": "의사결정 도구 구현", "importance": "major",
     "confidence": "high", "persona_tags": ["분석가"],
     "evidence": {"project": "서울 Airbnb 호스트 수익 최적화 가이드", "section": "주요 결과",
                  "source_text": "예상 수익을 비교할 수 있는 의사결정 도구 구현"}},
    {"layer": "skill", "normalized_text": "LLM 기반 기능 설계", "importance": "core",
     "confidence": "high", "persona_tags": ["기획자"],
     "evidence": {"project": "AI 기반 취업 의사결정 시스템", "section": "해결 과정",
                  "source_text": "LLM으로 요구사항과 경험의 의미 관계를 연결하는 방식 설계"}},
    {"layer": "task", "normalized_text": "업무 자동화 워크플로 설계", "importance": "major",
     "confidence": "high", "persona_tags": ["기획자"],
     "evidence": {"project": "AI 기반 취업 의사결정 시스템", "section": "해결 과정",
                  "source_text": "n8n 기반 시스템으로 반복 업무를 자동 처리"}},
]
RESUME_SUMMARY = {
    "profile_summary": "데이터 분석과 자동화를 결합해 반복 업무를 줄이고 의사결정을 지원하는 분석가",
    "core_strength": "문제 재정의 후 분석 기준을 새로 세우고, 결과를 실행 가능한 도구로 연결하는 역량",
    "problem_solved": "가격 중심/완료 여부 중심의 단순 분석에서 벗어나 운영 요소·행동 기반 분석으로 전환",
    "thinking_style": "가설 수립 → 통계적 검증 → 실행 가능한 도구화",
}


def _seed_candidate_jobs() -> None:
    job_store._init_table()
    conn = sqlite3.connect(DB_PATH)
    for p in POSTINGS:
        conn.execute(
            """
            INSERT OR REPLACE INTO candidate_jobs
                (job_id, title, company, url, source, origin, posting_text,
                 career_level, career_level_confidence, career_parser_version,
                 career_parsed_at, industry, company_size, location,
                 link_dead, deadline_expired)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, 0, 0)
            """,
            (
                p["job_id"], p["title"], p["company"], "https://example.com/jobs/" + p["job_id"],
                "기업 홈페이지", "demo", p["posting_text"], p["career_level"], "high",
                "demo-seed-v1", "IT/서비스", "중견", p["location"],
            ),
        )
    conn.commit()
    conn.close()
    print(f"candidate_jobs: {len(POSTINGS)}건 시드 완료")


def _seed_resume_understanding() -> None:
    job_store.save_resume_understanding(
        resume_hash=DEMO_RESUME_HASH,
        context="",
        short_semantic_json="{}",
        semantic_graph_json="[]",
        representation_version=RESUME_REPRESENTATION_VERSION,
        semantic_objects_json=json.dumps(RESUME_SEMANTIC_OBJECTS, ensure_ascii=False),
        relations_json="[]",
        summary_json=json.dumps(RESUME_SUMMARY, ensure_ascii=False),
    )
    print("resume_understanding_cache: 1건 시드 완료")


def _seed_semantic_link_cache() -> None:
    for job_id, result in ANALYSIS_RESULTS.items():
        payload = dict(result)
        payload["schema_version"] = SCHEMA_VERSION
        job_store.save_semantic_link_result(
            job_id=job_id, resume_hash=DEMO_RESUME_HASH,
            linking_version=SCHEMA_VERSION, result=payload,
        )
    print(f"semantic_link_cache: {len(ANALYSIS_RESULTS)}건 시드 완료")


def _seed_applications() -> None:
    for a in APPLICATIONS:
        app_id = application_manager.add_application(
            job_id=a["job_id"], company=a["company"], title=a["title"],
            url="https://example.com/jobs/" + a["job_id"], source=a["source"],
            status="지원 완료", memo=a["memo"],
        )
        if a["status"] != "지원 완료":
            # 상태가 더 진행된 건은 두 번째 이력을 남겨 전형 진행 타임라인을 보여준다.
            application_manager.update_status(app_id, a["status"], memo=a["memo"])
    print(f"applications: {len(APPLICATIONS)}건 시드 완료")


def _seed_last_resume() -> None:
    pdf_bytes = _build_demo_resume_pdf_bytes()
    last_resume.save(
        pdf_bytes=pdf_bytes,
        file_name="demo_resume.pdf",
        file_size=len(pdf_bytes),
    )
    print("data/last_resume/: 가상 지원자 이력서 복원용 시드 완료")


def _seed_collection_log() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT, collected_at TEXT, total_count INTEGER,
            new_count INTEGER, updated_count INTEGER, error_count INTEGER, status TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO collection_logs (source, collected_at, total_count, new_count, updated_count, error_count, status) "
        "VALUES ('refresh', datetime('now'), ?, 0, 0, 0, 'SUCCESS')",
        (len(POSTINGS),),
    )
    conn.commit()
    conn.close()
    print("collection_logs: 1건 시드 완료(마지막 업데이트 표시용)")


def _seed_company_research_table() -> None:
    # job_store.get_company_research()는 이 테이블이 이미 있다고 가정한다
    # (운영 DB에는 반자동 리서치로 이미 만들어져 있음). 데모 DB에는 그
    # 리서치 자체가 없어도 되지만, 테이블은 있어야 "no such table" 에러가
    # 안 난다 - 비어 있으면 get_company_research()가 그냥 None을 돌려주고
    # 호출부가 원래 있던 폴백으로 자연스럽게 넘어간다.
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS company_research "
        "(company_name TEXT PRIMARY KEY, data_json TEXT, updated_at TEXT)"
    )
    conn.commit()
    conn.close()
    print("company_research: 빈 테이블 생성 완료(폴백 경로 확인용)")


def main() -> None:
    verify_demo_resume_hash()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _seed_candidate_jobs()
    _seed_resume_understanding()
    _seed_semantic_link_cache()
    _seed_applications()
    _seed_last_resume()
    _seed_collection_log()
    _seed_company_research_table()
    print(f"\n완료: {DB_PATH}")


def reset_demo_data() -> None:
    """DEMO MODE 전용 - data/demo.db 를 지우고 처음 시드값으로 다시 만든다.
    app.py의 설정 화면 "데모 데이터 초기화" 버튼에서 호출한다(운영 DB는
    IS_DEMO=False 면 이 함수 자체를 호출하지 않게 app.py 쪽에서 막는다).
    """
    if not IS_DEMO:
        raise RuntimeError("reset_demo_data()는 JOBFIT_MODE=local 에서 호출할 수 없습니다.")
    if DB_PATH.exists():
        DB_PATH.unlink()
    last_resume_dir = _ROOT / "data" / "last_resume"
    if last_resume_dir.exists():
        shutil.rmtree(last_resume_dir)
    main()


if __name__ == "__main__":
    main()
