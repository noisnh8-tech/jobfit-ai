"""
resume_input/job_prep.py

"지원 준비" 화면(S1-D 하단)에 필요한 정보를 JD 원문에서 정규식으로
추출한다. LLM 미사용 - 새 판단을 만들지 않고, JD에 실제로 적힌
문구만 찾아서 보여준다. 신호가 없으면 그 항목은 아예 표시하지
않는다(추측으로 채우지 않는다 - 이 프로젝트 전체의 원칙).

실측 검증 안 됨(중요, 정직하게 기록): career_filter.py의
extract_jd_career_level()처럼 실제 코퍼스로 패턴 커버리지를 검증하지
않았다 - 이번 라운드에서 새로 만든 추출이라 오탐/누락 비율을 아직
모른다. ui_design.md의 "아직 실측 안 된 것" 목록에 추가해야 한다.
"""
from __future__ import annotations

import re
from datetime import date

from resume_input.candidate_search import _TAG_RE, _ENTITY_RE
from resume_input.header_registry import (
    DUTIES, REQUIRED, PREFERRED, BENEFITS, PROCESS, DOCUMENTS, as_regex,
)

# "전형절차/지원방법/제출서류" 류 안내 블록만 스캔 대상으로 좁힌다 -
# JD 전체에서 "포트폴리오"/"이력서" 같은 단어를 찾으면 지원자의 과거
# 경험 설명("포트폴리오 프로젝트 진행")과 뒤섞여 오탐이 나기 쉽다.
#
# 2026-07-16 리팩터 - 패턴 목록은 header_registry.py(단일 출처)를
# 쓴다. 80건 실제 화면 검증에서 이 정규식이 "5. 합류 과정을
# 소개해요"류(고정 문구와 줄 전체가 정확히 일치해야만 인식) 표현을
# 대량으로 놓쳐서 전형절차/제출서류 카드가 광범위하게 비어 있었다.
# "줄 전체 일치"도, "줄 끝 일치(suffix)"도 이 형태("번호. 헤더 문구 +
# 짧은 설명 문구가 뒤에 붙음" - 표제어가 줄 끝이 아니라 중간에 옴)는
# 못 잡는다. job_detail.py가 이미 "포함(contains) + 20자 미만"
# 조합으로 이 문제를 실측 검증했으므로(_classify_display_sections())
# 여기도 같은 조합으로 통일한다 - "제출 서류에는 연봉 정보 제외
# 부탁드립니다"(23자)는 20자 미만 기준을 넘어서 여전히 걸러진다.
_ADMIN_HDR_RE = as_regex(PROCESS + DOCUMENTS)
_WELFARE_HDR_RE = as_regex(BENEFITS)

_SECTION_STOP_RE = as_regex(PROCESS + DOCUMENTS + BENEFITS + DUTIES + REQUIRED + PREFERRED)
_MAX_HEADER_LINE_LEN = 20


def _extract_block(text: str, header_re: re.Pattern, max_len: int = 500) -> str:
    """header_re와 "줄 전체가 일치하는" 줄(진짜 섹션 헤더로 보이는
    짧은 줄)을 찾아, 그 다음 줄부터 다른 섹션 헤더를 만나거나
    max_len에 도달할 때까지의 텍스트를 반환한다.

    버그 이력(실측으로 발견): 처음엔 헤더 패턴을 텍스트 전체에서
    단순 substring 검색했는데, "복지에 차등은 없습니다" 같은 본문
    문장에 우연히 "복지"가 포함돼 있으면 그 문장 뒤 무관한 텍스트를
    "복리후생 정보"로 잘못 추출했다. 헤더처럼 보이는 "줄 전체"만
    인정하도록(HTML 제거 후 줄 단위로 순회) 고쳐서 해결."""
    plain = _TAG_RE.sub(" ", text)
    plain = _ENTITY_RE.sub(" ", plain)
    lines = re.split(r"[\n\r]+", plain)

    header_idx = None
    for i, line in enumerate(lines):
        stripped = re.sub(r"\s+", " ", line).strip()
        if len(stripped) < _MAX_HEADER_LINE_LEN and header_re.search(stripped):
            header_idx = i
            break
    if header_idx is None:
        return ""

    collected: list[str] = []
    total_len = 0
    for line in lines[header_idx + 1:]:
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue
        if len(stripped) < _MAX_HEADER_LINE_LEN and _SECTION_STOP_RE.search(stripped):
            break
        collected.append(stripped)
        total_len += len(stripped)
        if total_len >= max_len:
            break
    return "\n".join(collected)


_DOCUMENT_KEYWORDS = ["이력서", "자기소개서", "포트폴리오", "경력기술서", "졸업증명서", "성적증명서"]

_PROCESS_KEYWORDS = [
    ("서류", "서류전형"), ("코딩\\s*테스트", "코딩테스트"), ("과제\\s*전형", "과제전형"),
    ("ai\\s*역량\\s*검사", "AI역량검사"), ("1차\\s*면접", "1차 면접"), ("2차\\s*면접", "2차 면접"),
    ("최종\\s*면접", "최종 면접"), ("최종\\s*합격", "최종합격"),
]

# 실측 버그(2026-07-16): "\(.*마감.*\)"의 탐욕적 .*가 같은 줄 안에 있는
# "마감"을 포함한 무관한 문단 전체를 괄호로 착각해 통째로 붙잡는 사례가
# 실제 데이터에서 확인됨(수백 자짜리 안내문 전체가 deadline 값이 됨).
# 괄호 안 길이를 짧게 제한해서 실제 "(D-3 마감)" 같은 짧은 부연 설명만
# 잡히도록 고쳤다.
# 실측 버그(2026-07-16, 필드 수집률 감사): 전체 후보 풀(1993건) 재감사
# 결과 마감일 10.0%로 크게 낮았는데, "본 공고는 모집 완료 시 조기
# 마감될 수 있습니다"(57건)/"본 공고는 상시 모집으로,"(5건)/"채용
# 완료 시 조기 마감"(4건)처럼 "상시채용"/"채용시마감"과 뜻은 같지만
# 표현이 다른 상시·수시채용 안내 문구를 전혀 못 잡고 있었다(공고에
# 실제로 적힌 문구인데 패턴에 없어서 누락 - 날짜를 추측하는 게 아니라
# 이미 있는 문구를 못 찾은 것).
_DEADLINE_RE = re.compile(
    r"D-\d{1,3}|\d{4}[.\-]\s?\d{1,2}[.\-]\s?\d{1,2}\s*(까지|\(.{0,20}마감.{0,10}\))?"
    r"|상시\s*채용|상시\s*모집|수시\s*채용|수시\s*모집"
    r"|채용\s*시\s*마감|채용\s*시\s*까지|모집\s*완료\s*시|채용\s*완료\s*시"
    r"|지원서?\s*접수\s*기간"
)
_HEADCOUNT_RE = re.compile(r"\d+\s*명\s*(모집|채용)")
_SALARY_RE = re.compile(r"(연봉|급여)[^.\n]{0,30}\d[\d,~]*\s*만\s*원")
_EMPLOYMENT_TYPE_RE = re.compile(r"정규직|계약직|인턴(십)?|파견직")
# 실측 버그(2026-07-16, 필드 수집률 감사): 학력 10.7%로 낮았는데,
# "고등학교 졸업 이상"/"전공 무관"/"학력 및 전공 무관"/"전문대졸
# 이상"/"초대졸 이상"/"4년제 대학 졸업"처럼 실제로 자주 쓰이는 표현이
# 기존 목록(고졸 이상/전문학사 이상 등)과 형태만 달라서 누락됐다.
_EDUCATION_RE = re.compile(
    r"학력\s*(및\s*전공\s*)?무관|전공\s*무관"
    r"|고졸\s*이상|고등학교\s*졸업\s*이상|초대졸\s*이상"
    r"|전문학사\s*이상|전문대졸\s*이상|전문대\s*졸업\s*이상"
    r"|4년제\s*(대학교?\s*)?졸업|학사\s*이상|학사\s*학위"
    r"|대학원졸\s*\(?\s*석사\s*\)?\s*이상|석사\s*이상|박사\s*이상"
)


def extract_required_documents(posting_text: str) -> list[str]:
    block = _extract_block(posting_text, _ADMIN_HDR_RE)
    if not block:
        return []
    return [kw for kw in _DOCUMENT_KEYWORDS if kw in block]


def extract_application_process(posting_text: str) -> list[str]:
    """전형절차 안내 블록을 못 찾으면 빈 리스트를 반환한다(JD 전체를
    훑지 않는다) - "서류"/"면접" 같은 단어는 전형 단계가 아닌 문맥
    (예: "관련 서류 검토 경험")에도 흔히 등장해서 오탐 위험이 크다."""
    block = _extract_block(posting_text, _ADMIN_HDR_RE)
    if not block:
        return []
    found = []
    for pattern, label in _PROCESS_KEYWORDS:
        if re.search(pattern, block, re.I):
            found.append(label)
    return found


def extract_deadline(posting_text: str) -> str | None:
    m = _DEADLINE_RE.search(posting_text)
    return m.group(0).strip() if m else None


_DATE_TOKEN_RE = re.compile(r"(\d{4})[.\-]\s?(\d{1,2})[.\-]\s?(\d{1,2})")


def parse_deadline_date(deadline_text: str | None) -> date | None:
    """추출된 마감일 문자열에서 실제 달력 날짜만 뽑는다. "상시채용"/
    "채용 시 마감"처럼 특정 날짜가 없는 표현이나 "D-18"처럼 수집 시점
    기준 상대값(그 시점을 모르면 지금 기준 만료 여부를 계산할 근거가
    없음)은 None을 반환한다 - 날짜를 모르면 지어내지 않는다(이 프로젝트
    원칙)."""
    if not deadline_text:
        return None
    m = _DATE_TOKEN_RE.search(deadline_text)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def is_deadline_passed(deadline_text: str | None, today: date) -> bool | None:
    """반환: True(마감 지남) / False(안 지남) / None(날짜를 알 수 없어
    판단 불가 - "상시채용" 등 포함, 이 경우 만료로 취급하지 않는다)."""
    d = parse_deadline_date(deadline_text)
    if d is None:
        return None
    return d < today


def extract_job_info(posting_text: str) -> dict[str, str]:
    """가능한 항목만 채운다 - 못 찾은 항목은 dict에 아예 넣지 않는다."""
    info: dict[str, str] = {}

    m = _SALARY_RE.search(posting_text)
    if m:
        info["급여"] = m.group(0).strip()

    m = _EMPLOYMENT_TYPE_RE.search(posting_text)
    if m:
        info["근무 형태"] = m.group(0).strip()

    m = _EDUCATION_RE.search(posting_text)
    if m:
        info["학력 조건"] = m.group(0).strip()

    m = _HEADCOUNT_RE.search(posting_text)
    if m:
        info["모집 인원"] = m.group(0).strip()

    welfare_block = _extract_block(posting_text, _WELFARE_HDR_RE, max_len=300)
    if welfare_block.strip():
        info["복리후생"] = welfare_block.strip()[:200]

    return info


_TECH_KEYWORDS = [
    "SQL", "Python", "R", "Java", "Scala", "Tableau", "Power BI", "Looker",
    "Looker Studio", "Amplitude", "Mixpanel", "Google Analytics", "GA4",
    "BigQuery", "Snowflake", "Redshift", "Spark", "Hadoop", "Airflow", "dbt",
    "Kafka", "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Git", "Excel",
    "Google Sheets",
    "pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch", "Superset",
]

# 2026-07-31(Skill Extractor 한국어 표기 인식 - docs/verification/2026-07-31_
# candidate_generation_top100_eval/root_cause_table_top20.md의 Rejected
# Hypothesis - Skill Missing Penalty TODO에서 발견된 실측 버그) - 채용
# 공고가 도구명을 한국어로 적는 경우(예: 로그베이스 JD "엑셀, 스프레드시트
# 등을 활용해") 영어 키워드 정규식만으로는 못 잡는다. 추측성으로 목록을
# 넓히지 않고, 실측으로 확인된 사례만 우선 추가한다 - 새 판단을 지어내지
# 않는다는 이 함수의 기존 원칙과 동일하게 적용."
_KOREAN_SYNONYMS: dict[str, list[str]] = {
    "Excel": ["엑셀", "스프레드시트"],
    "Google Sheets": ["구글시트", "구글 시트"],
}

# 2026-07-31(같은 검증 중 발견된 기존 버그, Excel 관련 아님 - 새 한국어
# 동의어 추가와는 별개 이슈) - "Excel"을 대소문자 무시(re.I)로 찾다 보니
# "to excel in voice interactions"처럼 흔한 영어 동사 "excel"(뛰어나다)
# 까지 Microsoft Excel로 오인식됐다(실측 확인: xAI 채용공고 4건, 전부
# 같은 템플릿 문구). Excel만 대소문자를 구분해서 도구명("Excel")만
# 잡고 동사("excel")는 제외한다 - 다른 키워드는 원래대로 re.I 유지.
_CASE_SENSITIVE_KEYWORDS = {"Excel"}


def _build_tech_keyword_patterns() -> list[tuple[str, re.Pattern]]:
    patterns = []
    for kw in _TECH_KEYWORDS:
        surface_forms = [kw] + _KOREAN_SYNONYMS.get(kw, [])
        alt = "|".join(re.escape(s) for s in surface_forms)
        flags = 0 if kw in _CASE_SENSITIVE_KEYWORDS else re.I
        patterns.append((kw, re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])", flags)))
    return patterns


_TECH_KEYWORD_RE = _build_tech_keyword_patterns()


def extract_tech_stack(posting_text: str) -> list[str]:
    """JD 원문에서 알려진 기술 스택 키워드만 표면적으로 찾는다(LLM 미사용,
    이력서 비교 없음 - JD 자체가 요구하는 기술을 나열하는 용도). 목록에 없는
    기술은 놓칠 수 있다는 한계가 있지만, 새 판단을 지어내는 것보다는
    안전하다(이 프로젝트 전체 원칙과 동일)."""
    plain = _TAG_RE.sub(" ", posting_text or "")
    found = []
    for kw, pattern in _TECH_KEYWORD_RE:
        if pattern.search(plain) and kw not in found:
            found.append(kw)
    return found


def extract_application_prep(job: dict) -> dict:
    """S1-D "지원 준비" 섹션에 필요한 정보를 한 번에 묶어서 반환한다."""
    posting_text = job.get("posting_text", "") or ""
    job_info = extract_job_info(posting_text)
    if job.get("location"):
        # location은 텍스트에서 정규식으로 추측하지 않는다 - 원본
        # 소스(job_ai_v2/da_job_market_2026)에 이미 구조화된 값으로
        # 있던 필드를 candidate_jobs로 그대로 옮겨온 것뿐이다.
        job_info = {"근무 지역": job["location"], **job_info}
    return {
        "required_documents": extract_required_documents(posting_text),
        "application_process": extract_application_process(posting_text),
        "deadline": extract_deadline(posting_text),
        "job_info": job_info,
    }
