"""
resume_input/header_registry.py

공고 헤더(주요업무/자격요건/우대사항/복지/제출서류/채용절차/노이즈)
동의어의 단일 출처(single source of truth).

배경(2026-07-16, 80건 실제 화면 검증): 같은 공고 텍스트를 파싱하는
헤더 목록이 서로 다른 4곳에 따로 존재했다 - detail_parser.py(수집
단계, da_job_market_2026), candidate_search.py(검색 Representation),
job_detail.py(상세보기 화면), job_prep.py(지원 준비 화면 정규식
추출). 그래서 detail_parser.py에 새 동의어("합류 과정"/"이런 분을
찾고 있습니다" 등)를 추가해도 나머지 3곳에는 전혀 반영이 안 됐고,
그 결과 수집률 통계는 개선됐는데 실제 화면(job_detail.py 기준)은
그대로 깨져 있는 문제가 확인됐다("오늘 고친 코드를 실제 화면은 안
쓴다"). 이제 이 4곳 전부 여기 목록만 가져다 쓴다 - 새 동의어는
반드시 여기에만 추가한다.

각 모듈은 여전히 자기 목적에 맞는 매칭 방식을 쓴다(이 레지스트리는
"무엇이 헤더 후보 문구인가"만 정의하고, "어떻게 매칭할지"는 관여하지
않는다):
  - detail_parser.py: duties/required/preferred/skills는 "포함"(줄
    안 어디든), benefits/process/documents는 "줄 끝 일치"(suffix -
    회사명·이모지 등 장식이 앞에 붙어도 잡되, 본문 문장 중간 우연한
    일치는 배제).
  - candidate_search.py: 검색 Representation용 - 정규식 search().
  - job_detail.py: 상세보기 화면 표시용 - 20자 이하의 "짧은 줄"만
    헤더 후보로 인정(긴 문장 중간의 우연한 매치 배제).
  - job_prep.py: "지원 준비" 화면의 전형절차/제출서류/복지 블록
    추출용 - 줄 전체가 (장식 제외) 헤더 문구와 일치해야 함(가장
    엄격 - 오탐 시 무관한 안내문 전체를 그 항목으로 잘못 보여주는
    위험이 커서).

da_job_market_2026(별도 프로젝트, 독립 배포)의 detail_parser.py는
sys.path를 통해 이 파일을 직접 import한다(job_ai_v3가 이미
REF_DB_PATH/`.env` 경로처럼 절대경로로 da_job_market_2026을
참조하는 기존 관례의 반대 방향 - 이번이 처음이라 detail_parser.py
상단에 그 경위를 주석으로 남겨둔다). 코드를 복제하지 않는 이유: 이
파일이 다시 두 벌이 되면 오늘 고친 문제가 그대로 재발한다.
"""
from __future__ import annotations

import re


def as_regex(patterns: list[str]) -> re.Pattern:
    """문자열 목록을 하나의 대소문자 무시 정규식으로 합친다(각 문구는
    re.escape로 이스케이프 - 정규식 메타문자가 문구에 있어도 안전).
    candidate_search.py/job_detail.py처럼 정규식 search()로 헤더를
    찾는 소비자가 공통으로 쓴다."""
    return re.compile("|".join(re.escape(p) for p in patterns), re.I)


def as_line_regex(patterns: list[str]) -> re.Pattern:
    """job_prep.py처럼 "줄 전체가 (장식 문자 제외) 헤더 문구와 정확히
    같아야 함"을 요구하는 소비자용 - "제출 서류에는 연봉 정보 제외
    부탁드립니다" 같은 본문 문장이 우연히 헤더 문구로 시작해도 걸리지
    않게 하는 가장 엄격한 모드(오탐 시 무관한 안내문 전체를 그 항목
    내용으로 잘못 보여주는 위험이 커서)."""
    alts = "|".join(re.escape(p) for p in patterns)
    return re.compile(rf"^({alts})\s*[:：]?\s*$", re.I)


def as_suffix_regex(patterns: list[str]) -> re.Pattern:
    """줄 끝이 헤더 문구로 끝나는가(콜론 등 장식만 허용) - "5. 합류
    과정을 소개해요"처럼 번호·안내 문구가 헤더 앞에 붙는 실제 사례를
    잡아내면서, "제출 서류에는 연봉 정보 제외 부탁드립니다"처럼
    표제어로 끝나지 않는 본문 문장은 여전히 걸러낸다(실측으로 검증된
    detail_parser.py의 "suffix" 모드와 같은 원칙). 호출부가 줄 길이를
    별도로 제한해서(예: 40자 미만) 아주 긴 문장 끝에 우연히 표제어가
    오는 경우까지 오탐하지 않도록 해야 한다."""
    alts = "|".join(re.escape(p) for p in patterns)
    return re.compile(rf"({alts})\s*[:：]?\s*$", re.I)

DUTIES: list[str] = [
    "주요 업무", "주요업무", "담당 업무", "담당업무",
    "업무 내용", "업무내용", "하는 일", "핵심 업무",
    "합류하시면 함께 할 업무", "이런 업무를 담당", "어떤 업무를 담당",
    "이런 일을 해요", "어떤 일을 하나요", "업무책임",
    "what you'll do", "what you will do", "responsibilities", "responsibility", "role",
]

REQUIRED: list[str] = [
    "자격 요건", "자격요건", "필수 요건", "필수요건",
    "필수 자격", "지원 자격", "지원자격", "자격 조건",
    "기술 요건", "기술요건", "필수 조건",
    "이런 분을 찾고 있", "이런 역량을 갖춘 분을 찾", "이런 경험을 가진 분을 찾",
    "이런 분",
    "required qualifications", "qualifications", "requirements",
    # 실측(2026-07-25, 특정 해외 테크기업 재검증): "Key Competencies:"가 REQUIRED 어느
    # 항목과도 안 걸려서(qualifications/requirements 부분 문자열이 없음)
    # "자격요건" 섹션 전체가 통째로 안 잡히고 직전 섹션(주요업무)에 흡수됐다.
    # "competencies"/"required experience"는 qualifications/requirements와
    # 겹치지 않는 별개 표현이라 추가한다 - 의미 추론이 아니라 실제 JD에
    # 쓰이는 헤더 표현 추가일 뿐(Rule Parser 원칙 유지).
    "competencies", "required experience", "what you'll need", "what you need",
]

PREFERRED: list[str] = [
    "우대 사항", "우대사항", "우대 조건", "우대조건",
    "우대 요건", "우대요건", "우대", "필요 역량",
    "preferred qualifications", "preferred", "nice to have",
]

SKILLS: list[str] = [
    "기술 스택", "기술스택", "tech stack", "skills",
    "사용 기술", "사용기술", "개발 환경", "개발환경",
    "technical skills", "기술",
]

BENEFITS: list[str] = [
    "복지", "복리 후생", "복리후생", "혜택과 복지", "혜택 과 복지",
    "복지 및 혜택", "혜택 및 복지", "복지와 혜택", "혜택과 복지 안내",
    "benefits", "혜택", "근무 환경", "근무환경", "compensation",
    "복지 제도", "복리후생 제도",
]

PROCESS: list[str] = [
    "전형 절차", "전형절차", "채용 절차", "채용절차",
    "채용 프로세스", "채용프로세스", "진행 절차", "진행절차",
    "채용 과정", "채용과정",
    "hiring process", "recruitment process", "interview process", "how we hire",
    "이렇게 합류해요", "이런 순서로 진행돼요",
    "합류 여정", "합류여정", "합류 과정", "합류과정",
    "채용전형", "채용 전형", "전형 안내", "전형안내",
]

DOCUMENTS: list[str] = [
    "제출 서류", "제출서류", "지원 서류", "지원서류",
    "필요 서류", "필요서류", "준비 서류", "준비서류",
    "required documents", "documents to submit",
    "포트폴리오 제출", "필수 제출",
    # 실측 버그(2026-07-17, 80건 재검증): "지원서 제출"은 원래 여기
    # 있었는데, 실제로는 "지원서 제출 → 인터뷰 → 합격" 같은 전형절차
    # 흐름도의 한 단계 이름으로 훨씬 흔하게 쓰였다(특정 이커머스 등) - 그래서
    # NOISE/_SECTION_STOP_RE에도 이 목록이 들어가면서, "합류여정"(전형
    # 절차) 헤더 바로 다음 줄이 우연히 "지원서 제출 →"이면 그걸 새
    # 섹션 시작으로 오인해 전형절차 블록 전체가 사라졌다. "제출서류"
    # 목록 헤더로서의 의미보다 흐름도 단계로 쓰이는 빈도가 훨씬 높아서
    # 아예 뺀다 - "포트폴리오 제출"/"필수 제출"은 흐름도에 잘 안 쓰여서
    # 남겨둔다.
]

# 상세보기 화면(job_detail.py)에서만 쓰는 폐기 버킷 - PROCESS/BENEFITS와
# 겹치는 항목도 있다(그 카드들은 job_prep.py가 별도로 채우므로, 일반
# 텍스트 버킷에서는 노이즈로 끊어내는 게 맞다). candidate_search.py의
# _HDR_END는 이 중 일부만 쓴다(검색 Representation은 노이즈 필터링
# 목적이 아니라 섹션 종료 지점만 필요하기 때문).
# 실측 버그(2026-07-17, 80건 재검증): 처음엔 NOISE를 PROCESS/DOCUMENTS/
# BENEFITS와 별도로 손으로 다시 나열했는데, 거기 붙여쓰기 변형("채용절차"
# 등, PROCESS엔 있음)을 빠뜨려서 "채용절차"가 resp/req/pref 버킷에
# 계속 새는 게 재확인됐다 - 정확히 오늘 통합하려던 그 문제(동의어가
# 한쪽에만 있고 다른 쪽에 없음)가 이 파일 안에서도 재발한 것. 그래서
# PROCESS/DOCUMENTS/BENEFITS를 직접 이어붙여서 NOISE를 만든다 - 그
# 목록들이 갱신되면 NOISE도 자동으로 같이 갱신된다.
NOISE: list[str] = PROCESS + DOCUMENTS + BENEFITS + [
    "지원 방법",
    "참고 사항", "참고해 주세요",
    "개인정보", "개인정보 처리방침", "개인정보 보호", "개인정보처리방침", "개인정보보호",
    "서류 반환", "서류반환", "근무 조건", "모집 부문",
    "동료의 한 마디", "동료의 한마디", "리더의 한 마디", "리더의 한마디",
    "faq", "자주 묻는 질문", "카카오톡", "카카오 채널", "문의 사항",
    "만나게 될 근무지",
    # 실측(2026-07-17, 특정 제조업 공고 재검증): "이렇게 성장할 수 있어요"/
    # "성장 포인트"류 - 자격요건/우대사항이 아닌 "입사 후 성장 서사"
    # 섹션인데, 이 프로젝트 스키마엔 별도 버킷이 없어서 방치하면 직전
    # 섹션(주요업무 등)에 그대로 흡수됐다.
    "이렇게 성장할 수 있어요", "성장 포인트", "성장포인트", "성장 스토리",
    # 실측 버그(2026-07-25, Rule Representation 실험 - 특정 해외 테크기업 공고
    # 4건 재검증):
    # Greenhouse ATS 지원폼의 필드 라벨 "Preferred First Name"이 "preferred"
    # 부분 문자열을 포함해서 PREFERRED(우대사항) 헤더로 오인식되고, 그 뒤로
    # 이어지는 지원서 입력폼 전체(Email/Phone/Location/Resume/Cover Letter/
    # LinkedIn/Submit application)가 통째로 우대사항 버킷에 흡수됐다. 특정 공고만의
    # 문제가 아니라 Greenhouse ATS를 쓰는 다른 회사 공고에도 반복될 수 있는
    # 표준 템플릿이라 여기 추가한다 - "preferred"보다 먼저 NOISE로 걸러져야
    # 하므로(_classify_header_for_display가 NOISE를 PREF보다 먼저 검사) 순서
    # 문제는 없다.
    "apply for this job", "indicates a required field", "quick apply",
    "preferred first name", "preferred name", "first name", "last name", "legal first name",
    "legal last name", "location (city)", "locate me", "resume/cv",
    "cover letter", "linkedin profile", "submit application",
    "accepted file types", "create a job alert", "equal opportunity",
    "candidate privacy notice",
]
