"""
resume_input/career_filter.py

Career Filter - Retrieval(BM25/Embedding) 이전에 검색 대상 JD Pool
자체를 사용자 연차 기준으로 줄이는 순수 필터. 점수 계산에 관여하지
않는다 - Career Score/Weight/Bonus/Penalty는 없다(ui_design.md v6
§0-7). 정렬 시 연차상태(compute_career_status 결과)를 1차 키로만
쓴다.

`extract_jd_career_level()`의 정규식은 실제 코퍼스(da_job_market_2026
job_postings, 1985건)에서 실측한 패턴 분포를 근거로 만들었다(추측
아님, 2026-07-12 실측):
- "N년 이상": 1083건(가장 흔함) - 예) "개발 경험 3년 이상"
- "경력 N년"(이상 없이 단독): 462건과 상당 부분 겹침(우선순위로 처리)
- "N~M년"(범위): 135건 + "N년~M년": 41건 - 예) "3~10년 미만"
- "신입"(단독): 94건
- "경력무관": 19건
- "신입/경력"(혼용): 8건
- "N년차 이상": 4건
- `job_postings.experience` 구조화 컬럼은 92.8%가 빈 문자열이라 이
  용도로는 쓸 수 없다(실측 확인) - posting_text 정규식에 의존한다.

버킷 경계 규칙: "N년 이상"/"N~M년" 같은 하한 표현은 그 하한값 N을
기준으로 신입/1~3년/3~5년/5년+ 중 하나로 매핑한다(N<1: 신입,
1<=N<3: 1~3년, 3<=N<5: 3~5년, N>=5: 5년+ - "3년 이상"은 "1~3년"이
아니라 "3~5년"에 해당한다고 봄, 최소 3년을 요구하는 공고이므로).

연차상태(compute_career_status) 매핑은 사용자 연차와 JD 연차의
버킷 순서 차이(ordinal distance)로 계산한다 - 신입 사용자 기준으로
주어진 예시(1~3년→약간상향, 3~5년→상향, 5년+→크게상향)를 일반화한
공식이다. 이 공식으로 원래 사용자가 준 전체 표(4개 사용자 레벨 x
6~7개 상태)를 전부 재현할 수 있는지는 검증되지 않았다 - 원문 표
전체가 지금 컨텍스트에 없어서, 주어진 예시와 정렬 순서
(적합→약간하향→약간상향→확인필요→상향→크게상향, 여기에 순수
하향 2단계 이상 차이인 "하향"도 정렬 순서상 약간하향과 상향 사이
어딘가에 필요하다고 판단해 추가함)에 맞춰 공식으로 일반화했다.
실제 표와 다르면 조정이 필요하다.
"""
from __future__ import annotations

import re

# extract_jd_career_level()의 정규식/우선순위 규칙이 바뀔 때마다 올린다.
# candidate_jobs.career_parser_version에 저장되어, 나중에 이 값이 바뀌면
# "옛 버전으로 계산된 job들"을 식별해 재분석 대상으로 표시할 수 있다
# (job_store.list_jobs_needing_career_reparse() 참고). career_level 자체는
# 파생 데이터(원본 JD 텍스트를 이 정규식으로 파싱한 결과)이므로, 파서가
# 좋아지면 재계산이 필요할 수 있다는 것을 명시적으로 추적하기 위함.
PARSER_VERSION = "v7.1-en-minimum-years-2026-08-30"

_LEVELS = ["신입", "1~3년", "3~5년", "5년+"]
_ORDINAL = {lvl: i for i, lvl in enumerate(_LEVELS)}

CAREER_STATUS_SORT_ORDER = ["적합", "약간하향", "약간상향", "확인필요", "상향", "크게상향"]

_MIXED_RE = re.compile(r"신입\s*[/~\-]\s*경력")
_INDEPENDENT_RE = re.compile(r"경력\s*무관")
# (?<!\d) - 앞에 숫자가 더 있으면 안 잡는다. 실측으로 발견한 버그(2026-07-14):
# \d{1,2}만 쓰면 "100년 이상" 같은 표현에서 뒤 두 자리 "00"만 잘라서 잡아
# "0년 이상" = 신입으로 완전히 잘못 분류하는 사례가 실제 공고에서 확인됨
# (예: "100년 이상 지속 성장하는 기업" 문구가 있는 공고에서 정작 자격요건의
# "경력 5년 이상"은 검사되지도 않고 그 앞의 "100년"에서 멈춰버림).
_YEAR_RANGE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[~\-]\s*(\d{1,2})\s*년")
_YEAR_MIN_RE = re.compile(r"(?<!\d)(\d{1,2})\s*년\s*차?\s*이상")
_YEAR_PLAIN_RE = re.compile(r"경력\s*(?<!\d)(\d{1,2})\s*년")
_JUNIOR_RE = re.compile(r"신입")

# 실측 버그(2026-07-17): 경력 수집률(66.5%)을 감사하다가, 못 찾은
# 케이스 121건을 표본 확인하니 전부 해외법인 이커머스/애드테크류 영문
# 공고였다 - "2+ years of experience", "5-6+ years of experience
# working with..." 처럼 명시적으로 연차가 적혀 있는데도 한글 전용
# 정규식이라 하나도 안 걸렸다. 같은 우선순위(무관 → 범위 → 최소 →
# 단독 → 신입)로 영문 패턴을 추가한다.
_EN_ENTRY_RE = re.compile(r"entry[\s-]level|no\s*experience\s*required|new\s*grad", re.I)
_EN_YEAR_RANGE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[\-–~]\s*(\d{1,2})\s*\+?\s*years?", re.I)
_EN_YEAR_MIN_RE = re.compile(r"(?<!\d)(\d{1,2})\s*\+\s*years?", re.I)
_EN_YEAR_PLAIN_RE = re.compile(r"(?<!\d)(\d{1,2})\s*years?\s*(of\s*)?(work\s*)?experience", re.I)
# 실측 버그(2026-08-30, career=None 57건 전수감사): 특정 AI 반도체 공고(GreetingHR) 영문
# 공고 5건이 "Minimum 3 years in technical roles" / "Minimum of 5 years of
# professional experience" / "Minimum of 8 years of hands-on experience" 처럼
# 명시적 최소 경력을 적었는데 위 정규식들이 못 잡았다 - _EN_YEAR_MIN_RE 는
# "N+ years"(플러스 기호) 필수, _EN_YEAR_PLAIN_RE 는 "years [of] [work] experience"
# 연속형만 매칭해서 "of professional experience" / "in technical roles" 는 불일치.
# "minimum"/"at least" 라는 명시적 한정어가 앞에 붙은 "N years" 만 잡는다(회사
# 연혁·성장 스토리의 "10 years" 류 오탐 없게). 이 감사에서 확인된 표현 패턴으로
# 범위를 제한한다 - 새 NLP/추론 아님, 기존 _EN 계열 coverage 보완.
_EN_YEAR_MINIMUM_RE = re.compile(r"(?:minimum|at\s*least)\s*(?:of\s*)?(\d{1,2})\s*\+?\s*years?", re.I)

# 실측 버그(2026-07-18): "Staff~Sr.Staff Data Scientist", "[시니어]
# 데이터 분석가", "Senior Data Analyst"처럼 제목에 직급이 명시돼 있는데
# 숫자 연차가 없는 공고(주로 해외/시니어 채용 공고)는 위 숫자 패턴
# 전부 놓치고 career_level=None으로 떨어졌다. None은 설계상 "항상
# 포함"이라(_BASE_INCLUDE 주석 참고), 신입 사용자에게도 그대로
# 노출됐다. "Manager"/매니저는 국내 스타트업 신입 공고에도 흔해
# 오탐 위험이 커서 제외한다(사용자 확정). 정확한 연차 숫자를 모르므로
# "3~5년"으로 확정(explicit)하지 않고 confidence="inferred"로 표시해서
# 다른 명시적 연차 매칭과 구분한다.
#
# **title에만 적용한다(본문에는 적용하지 않는다)** - 실측으로 확인된
# 오탐(2026-07-18): "Business Analyst (Taiwan)"(정상적으로 잘 맞는
# 공고, career_level=None이 맞음)이 본문 중 "...will **lead** cross-
# functional..." 같은 동사 용법의 "lead" 때문에 3~5년으로 잘못
# 분류됐다. 본문(posting_text)은 자유 문장이라 "lead"/"staff" 같은
# 단어가 동사·일반명사로도 흔히 쓰여 오탐이 심하지만, title은 명사구
# 직함이라 이런 동사 오탐이 없다.
# 2026-07-20(사용자 확정, 실측 근거: Top50 재검증에서 "퓨쳐스콜레 -
# 사업 총괄"이 career_level=None으로 필터를 그냥 통과함 - qualification
# Object엔 "P&L 책임자 또는 조직장 경력"이 critical로 명시돼 있었는데도
# 이 키워드 목록엔 한국어 조직장급 타이틀이 전혀 없었다). "매니저"와
# 달리 "총괄"/"본부장"/"조직장"은 국내 신입 공고 제목에 흔히 쓰이는
# 표현이 아니라 오탐 위험이 낮다(매니저 제외 원칙과 같은 이유로 여기엔
# 안 넣음).
#
# "총괄(?!\])" - 회귀 검증 중 발견(2026-07-20): "[성수프로젝트총괄]
# 분양기획" 같은 공고는 "총괄"이 직급이 아니라 대괄호로 묶인 프로젝트
# 브랜드명("성수프로젝트총괄" 시리즈)의 일부다. 실제 직무(분양기획/
# 오피스상품기획/분양영업)는 신입 지원자도 볼 수 있어야 하는데, "총괄"
# 단어만 보고 전부 시니어로 분류하면 이 3건이 오탐이 된다. "총괄" 바로
# 뒤에 "]"가 오면(=대괄호 프리픽스 안에 갇혀 있으면) 제외하고, 그 외
# (실제 직급으로 쓰인 경우, 예: "사업 총괄", "마케팅 총괄")만 매치한다.
# 이 필터는 여기서 Freeze한다 - 새 키워드/예외는 추가하지 않는다.
_SENIOR_KEYWORD_RE = re.compile(r"\b(Senior|Sr\.?|Staff|Lead|Principal)\b|시니어|수석|총괄(?!\])|본부장|조직장", re.I)


def _valid_years(n: int) -> bool:
    return 0 <= n <= 30


# ── 원천 구조화 경력 우선 (2026-08-30, Top100 감사 후) ─────────────────────────
# 감사에서 확인된 원칙: 원천(source)이 구조화된 경력조건을 제공하면 그 값을
# authoritative 로 우선하고, 없을 때만 posting_text 자연어를 파싱한다.
#   - Wanted API annual_from/annual_to → job_text_fetcher 가 jobs.experience_level 에
#     "신입"/"경력무관"/"5년 이상"/"3~10년" 형태로 저장
#   - GreetingHR openings jobPositionCareer.careerFrom/careerType → A-6 이 "경력 3년 이상" 등으로
#   - Saramin 리스트 .job_condition → A-1 이 "경력 3~9년"/"경력 2년 이상" 등으로
# 이 정확한 값을 다시 자연어 파싱으로 덮으면 본문 뒤쪽 "(형태) 경력 무관" 같은
# 문구에 밀려 "경력 5년 이상"이 "경력무관"으로 뒤집힌다(감사 #42/#86). 20년 요구가
# 본문 파싱 실패로 None→통과되기도 한다(#76).
_STRUCT_IRRELEVANT_RE = re.compile(r"경력\s*무관|신입\s*[·/,]\s*경력|경력\s*[·/,]\s*신입")
_STRUCT_NEWBIE_RE = re.compile(r"^\s*신입\s*$")
_STRUCT_RANGE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[~\-]\s*(\d{1,2})\s*년?")
_STRUCT_MIN_RE = re.compile(r"(?<!\d)(\d{1,2})\s*년?\s*(?:이상|\+|↑)")
_STRUCT_NUM_RE = re.compile(r"(?<!\d)(\d{1,2})\s*년")


def _parse_structured_career(s: str) -> tuple[str | None, str]:
    """원천 구조화 경력 문자열 → (career_level, confidence).
    숫자를 못 찾으면(바 "경력" 등 애매값) (None, "none") - 본문 파싱에 맡긴다."""
    s = (s or "").strip()
    if not s:
        return None, "none"
    if _STRUCT_IRRELEVANT_RE.search(s):
        return "경력무관", "explicit"
    if _STRUCT_NEWBIE_RE.match(s):
        return "신입", "explicit"
    m = _STRUCT_RANGE_RE.search(s)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"
    m = _STRUCT_MIN_RE.search(s)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"
    m = _STRUCT_NUM_RE.search(s)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"
    return None, "none"


# posting_text 의 "[경력]" 명시 영역만 잘라낸다 (Wanted P0 수집이 본문 최상단에
# "[경력]\n경력 5년 이상\n\n[다음섹션]" 형태로 넣는다). structured 가 없을 때
# 이 영역을 본문 전체보다 우선 파싱한다 - 뒤쪽 자격요건의 "경력 무관"에 안 밀리게.
_CAREER_SECTION_RE = re.compile(r"\[경력\]\s*\n(.+?)(?:\n\s*\n|\n\s*\[|$)", re.S)


def _extract_career_section(posting_text: str) -> str | None:
    m = _CAREER_SECTION_RE.search(posting_text or "")
    return m.group(1).strip() if m else None


def _bucket_from_years(n: int) -> str:
    """JD 요구 연차(범위형이면 최소값 n)를 필터링용 bucket 으로 매핑한다.
    이건 "실제 요구 경력이 몇 년이다"가 아니라 "신입 pool 에 넣을지"를
    가르는 분류다 (career_level 은 파생 필터 데이터, 원문은 별도 보존).

    2026-08-30 (사용자 확정) - 임계값 n<3 → n<2. 최소 요구경력 1년까지는
    신입 challenge 로 허용(1~3년 bucket → 약간상향), 최소 2년 이상은 명시적
    경력직으로 보고 신입 pool 에서 제외한다(3~5년 bucket → _BASE_INCLUDE
    ["신입"] 밖). 기존엔 n<3 이라 "2~7년"·"2년 이상" 같은 경력직이 min=2 만
    보고 "1~3년" 으로 들어와 신입 추천에 섞였다. "2~7년 → 3~5년"은 요구
    경력을 3~5년으로 해석한 게 아니라, 신입 허용범위 밖으로 보내는 bucket
    선택일 뿐이다. min=0(0~N년)은 그대로 "신입"(적합)."""
    if n < 1:
        return "신입"
    if n < 2:
        return "1~3년"
    if n < 5:
        return "3~5년"
    return "5년+"


def _extract_from_text(text: str) -> tuple[str | None, str]:
    if _MIXED_RE.search(text) or _INDEPENDENT_RE.search(text):
        return "경력무관", "explicit"

    m = _YEAR_RANGE_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"

    m = _YEAR_MIN_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"

    m = _YEAR_PLAIN_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "inferred"

    if _JUNIOR_RE.search(text):
        return "신입", "explicit"

    if _EN_ENTRY_RE.search(text):
        return "신입", "explicit"

    m = _EN_YEAR_RANGE_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"

    m = _EN_YEAR_MIN_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"

    m = _EN_YEAR_MINIMUM_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "explicit"

    m = _EN_YEAR_PLAIN_RE.search(text)
    if m and _valid_years(int(m.group(1))):
        return _bucket_from_years(int(m.group(1))), "inferred"

    return None, "none"


def extract_jd_career_level(title: str, posting_text: str,
                            structured_career: str = "") -> tuple[str | None, str]:
    """JD에서 요구 연차를 추출한다. 반환: (career_level, confidence)
    career_level: "신입"|"경력무관"|"1~3년"|"3~5년"|"5년+"|None(미기재)
    confidence: "explicit"|"inferred"|"none"

    우선순위 (2026-08-30, Top100 감사 후):
      1. title 에 명시된 연차 ("… (4년이상)" 처럼 사람이 직접 쓴 신호. 원천
         구조화값이 GreetingHR 기본값(0~3)으로 남아있는 등 어긋날 때의 안전장치)
      2. structured_career - 원천 구조화 경력(jobs.experience_level: Wanted
         annual_from / GreetingHR careerFrom / Saramin 리스트). 자연어 본문
         재파싱으로 덮지 않는다 - 본문 뒤쪽 "(형태) 경력 무관" 등에 안 밀리게.
      3. posting_text 의 [경력] 명시 영역
      4. posting_text 본문 fallback
      5. title 직급 키워드
      6. None

    (1)을 앞세우는 이유: 일부 회사(예: Upstage)는 여러 공고를 한 페이지에서
    스크래핑해 posting_text 에 다른 포지션의 연차 문구가 섞인다. title 은 명사구
    직함이라 이런 오염이 없다."""
    level, confidence = _extract_from_text(title or "")
    if level is not None:
        return level, confidence

    if structured_career:
        level, confidence = _parse_structured_career(structured_career)
        if level is not None:
            return level, confidence

    section = _extract_career_section(posting_text or "")
    if section:
        level, confidence = _extract_from_text(section)
        if level is not None:
            return level, confidence

    level, confidence = _extract_from_text(posting_text or "")
    if level is not None:
        return level, confidence
    # 숫자 연차/명시적 문구를 title·본문 어디에서도 못 찾았을 때만,
    # title에 있는 직급 키워드(Senior/Staff/Lead 등)를 마지막으로
    # 확인한다. title에만 적용하는 이유는 _SENIOR_KEYWORD_RE 주석 참고
    # (본문에서 검사하면 "lead"의 동사 용법 등으로 오탐이 심함).
    if _SENIOR_KEYWORD_RE.search(title or ""):
        return _bucket_from_years(3), "inferred"
    return None, "none"


# 사용자 연차별 "기본 포함" ordinal 집합 + "도전 공고 포함" 옵션 ON일 때
# 추가되는 ordinal 집합. 경력무관/미기재(None)는 항상 포함(별도 처리).
_BASE_INCLUDE: dict[str, set[int]] = {
    "신입": {0, 1},
    "1~3년": {1, 2},
    "3~5년": {1, 2, 3},
    "5년+": {0, 1, 2, 3},
}
_CHALLENGE_ADD: dict[str, set[int]] = {
    "신입": {2},
    "1~3년": {0},
    "3~5년": {0},
    "5년+": set(),
}


def filter_by_career_level(jobs: list[dict], user_level: str, challenge_option: bool = False) -> list[dict]:
    """Retrieval 이전에 호출 - 검색 대상 JD Pool 자체를 줄인다.
    job["career_level"]이 이미 채워져 있다고 가정한다(candidate_jobs
    테이블에 저장된 값을 그대로 씀 - 여기서 텍스트를 다시 분석하지
    않는다)."""
    allowed = set(_BASE_INCLUDE[user_level])
    if challenge_option:
        allowed |= _CHALLENGE_ADD[user_level]

    result = []
    for job in jobs:
        jd_level = job.get("career_level")
        if jd_level is None or jd_level == "경력무관":
            result.append(job)
            continue
        if _ORDINAL.get(jd_level) in allowed:
            result.append(job)
    return result


def compute_career_status(user_level: str, jd_level: str | None, challenge_option: bool = False) -> str:
    """필터를 통과한 공고에 대해 연차상태 뱃지를 부여한다. 점수가
    아니라 표시용 상태다. challenge_option은 필터링에만 영향을 주고
    상태 계산 자체에는 영향을 주지 않는다(필터를 통과했다는 전제 하에
    상태는 순수하게 user_level과 jd_level의 관계로만 결정된다)."""
    if jd_level is None:
        return "확인필요"
    if jd_level == "경력무관":
        return "적합"

    distance = _ORDINAL[jd_level] - _ORDINAL[user_level]
    if distance == 0:
        return "적합"
    if distance == 1:
        return "약간상향"
    if distance == 2:
        return "상향"
    if distance >= 3:
        return "크게상향"
    if distance == -1:
        return "약간하향"
    return "하향"


def format_career_reason(user_level: str, jd_level: str | None) -> str:
    """연차 판단 근거 1줄(LLM 미사용, 템플릿 조립)."""
    if jd_level is None:
        return "공고에 연차 요건이 명시되어 있지 않습니다."
    if jd_level == "경력무관":
        return f"{user_level} 사용자이며 공고는 경력무관입니다."
    return f"{user_level} 사용자이며 공고는 {jd_level} 경력을 요구합니다."
