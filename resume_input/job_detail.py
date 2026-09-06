"""
resume_input/job_detail.py

상세보기 화면 표시용 - 새 LLM 호출 없음. candidate_search.py의
`_classify_sections()`(검색 Representation에도 쓰이는 섹션 분류
로직)를 표시용으로 재사용한다 - 분류 로직은 candidate_search.py에만
있고 여기서는 결과를 그대로 가져다 쓴다.

`build_jd_detail_view()`는 공고 상세보기 전용이다(2026-07-13 설계
변경, 2026-07-14 탭 구조로 재정리) - 여기서부터는 이력서를 절대
참조하지 않는다. 이미 생성돼 캐시된 jd_context/jd_short_semantic
(understanding.py, LLM 1회 생성분)과 job_prep.py의 정규식 추출
결과만 재구성해서 보여준다 - 새 LLM 호출 없이 "공고를 이해시키는"
화면을 만든다. 반환 구조는 요약/상세분석/요구스킬/조직문화 4개 탭에
그대로 대응한다(레퍼런스 UI의 탭 구성과 맞춤).
"""
from __future__ import annotations

import json
import re

from resume_input.candidate_search import _ENTITY_RE, _HDR_PREF, _HDR_REQ, _HDR_RESP, _TAG_RE
from resume_input.header_patterns import find_header_matches
from resume_input.header_registry import NOISE, as_regex
from resume_input.job_prep import extract_application_prep, extract_tech_stack
from resume_input.reason_phrasing import group_recommendation_reasons

# 상세보기 전용 종료+폐기 헤더. 2026-07-16 리팩터 - candidate_search의
# _HDR_END와 서로 다른 별도 목록을 유지하고 있었는데(검색 vs 표시),
# 실측 결과 이 구분 자체가 새 동의어 반영 누락의 원인 중 하나였다 -
# 이제 header_registry.NOISE 하나만 쓴다(검색 쪽에서도 노이즈로 걸러야
# 할 내용이라 넓혀도 안전하다).
_HDR_NOISE = as_regex(NOISE)

# 업무/자격요건이 아닌 안내성 문장 - 전형 절차/제출서류 유의사항/법적
# 고지/고용조건 안내가 resp·req·pref 섹션 "안에"(자기 헤더 없이 마지막
# 줄로) 섞여 들어오는 경우가 실제 데이터에서 확인됐다(2026-07-26,
# jobs.db 138건 전수 스캔 - 30건/약 22%에서 발견). `_HDR_NOISE`는
# 헤더 단위로 통째 버리는 로직이라 이 경우를 못 잡는다(그 문장들은
# 자기 헤더 없이 주요업무/자격요건 헤더 밑에 그냥 이어붙어 있음) -
# 그래서 별도로 "줄 단위" 필터가 필요하다.
#
# 1차(138건 표본, 4개 카테고리): 전형 절차/일정, 제출서류 유의사항,
# 법적/제도 고지, 고용조건 안내.
#
# 2차(2026-07-26, 714개사·2,819건 전체 풀 검증 - Case B로 확정한 것만
# 추가): 1차 필터 적용 후에도 56.8%(1,600건)에 노이즈가 남아있었고,
# 그중 압도적 1위가 "상세 정보 더 보기"(1,243건, 44%) - 채용사이트
# "더보기" 버튼 텍스트가 스크래핑되며 섞여 들어온 것으로 보인다. 그
# 외 이메일 주소, 장식용 구분선, 특정 AI 스타트업식 제출서류/전형 안내,
# 특정 대기업 계열사식 "-요"체 고용조건/전형 안내도 추가한다 - 전부 여러
# 공고에서 반복 확인된 정형 문구다.
#
# 여전히 일부러 안 넣은 것 - 복지 혜택 나열이 resp로 오분류된 사례(예:
# 특정 해외 테크기업 영문 "Medical, dental, and vision coverage..."), 해외 기업류
# 영문 D&I 성명("Our Commitment To Inclusion Belonging" 등), 특정 해외 테크기업의
# "일하는 방식" 자격요건 서술("You demonstrate strong judgment under
# uncertainty." 등)은 패턴이 회사마다 크게 다르고 뒤의 것은 실제
# 자격요건 콘텐츠일 가능성이 높아 Recall 손실 위험이 크다 - 계속 제외.
_BOILERPLATE_LINE_RE = re.compile(
    r"전형\s*>"
    r"|전형\s*결과는"
    r"|전형\s*과정이\s*필요"
    r"|해당\s*전형은\s*포지션에\s*따라"
    r"|인터뷰\s*및\s*직무\s*테스트가\s*추가"
    r"|각\s*전형의\s*세부\s*진행\s*방식"
    r"|상시\s*채용으로.{0,10}조기\s*마감"
    r"|포트폴리오.{0,40}(업로드|공동제작|미공개|기밀)"
    r"|각별히\s*유의하시기\s*바랍니다"
    r"|병역\s*관련\s*사항을\s*확인"
    r"|산업기능요원"
    r"|장애인고용촉진"
    r"|(장애인|국가보훈|국가유공자|보훈\s*대상자).{0,30}(우대|관계\s*법령|관련\s*법령|관련\s*법률|취업보호대상자)"
    r"|법령상\s*자격이\s*갖추어지지\s*않은\s*경우\s*채용이\s*제한"
    r"|수습기간"
    r"|제반사정을\s*고려하여\s*변경될\s*수\s*있습니다"
    r"|최종\s*합격\s*통지"
    r"|최종\s*합격\s*후.{0,20}입사"
    r"|중견기업으로\s*분류되어"
    r"|전형.{0,10}\]?\s*>\s*\["
    r"|순서로\s*진행됩니다"
    r"|상세\s*정보\s*더\s*보기"
    r"|^-{5,}"
    r"|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    r"|제출\s*자료는\s*PDF\s*형식으로\s*업로드"
    r"|민감\s*정보.{0,20}(연봉|주민번호)"
    r"|레퍼런스\s*체크\s*절차가\s*진행될\s*수\s*있습니다"
    r"|최종\s*결과\s*발표"
    r"|절차는\s*상황에\s*따라\s*조정될\s*수\s*있습니다"
    r"|모집\s*절차\s*-\s*전체\s*온라인\s*진행"
    r"|관련\s*문의사항은"
    r"|채용\s*관련\s*문의사항은"
    r"|지원\s*분야가\s*변경될\s*수\s*있어요"
    r"|근무지는\s*회사\s*사정에\s*따라\s*변경될\s*수\s*있어요"
    r"|합격이\s*취소되거나\s*전형\s*진행에서\s*불이익"
    r"|해외여행\s*결격\s*사유"
)


def _strip_boilerplate_lines(segment: str) -> str:
    """resp/req/pref 세그먼트에서 안내성 문장 줄만 제거한다(문장을
    요약/축약하지 않음 - 업무/자격요건이 아닌 줄을 통째로 드롭할 뿐).
    실제 업무 내용이 담긴 줄은 절대 건드리지 않는다."""
    lines = [ln for ln in segment.split("\n") if not _BOILERPLATE_LINE_RE.search(ln)]
    return "\n".join(lines).strip()


# ==========================================================
# Merge v1 (Frozen, 2026-07-31)
#
# Scope
#   - Colon Line Merge
#   - Delimiter Merge
#   - Leading Punctuation Merge
#
# 세 Step 모두 "구조 기반 복원"이다 - 원본 정보는 하나도 소실되지 않았고,
# 줄바꿈 위치만 잘못 놓인 경우만 다룬다(복원이 실측으로 증명된 경우만
# 구현, 추측에 의한 삭제/병합 없음). Golden Regression:
# docs/verification/2026-07-31_merge_v1/golden_cases.py
#
# Do not modify.
#
# New recovery logic must be implemented under Sentence Fragment Recovery
# (원본 정보 일부가 이미 소실된 상태에서 복원 가능 범위를 다루는 별도
# 이니셔티브 - lowercase-start / Sentence Fragment / Single Token) - Merge
# v1 안에 새 Step을 추가하지 않는다.
# ==========================================================

# Colon Line Merge v1.0 (Frozen, 2026-07-29) - "콜론 단독 고아 줄" 정리. 실측(100건+
# 실제 인접 줄 쌍 분석) 근거: "용어:\nGA4, SQL..." 형태로 콜론이 앞줄(용어)과
# 떨어져 뒷줄 맨 앞에 붙어 있는 경우가 반복 확인됐다. 콜론 혼자만 있는
# 줄(1~2자)은 이미 기존 <4자 필터로 안 보이지만, "콜론 + 내용"이 길게
# 이어지는 줄(예: ": GA4, 카페24 통계 툴을 활용한...")은 그 필터를
# 통과해서 화면에 앞에 콜론이 붙은 채로 노출된다. 이건 "병합"이 아니라
# "정리"다 - 앞줄(용어)을 복원하지 않고, 그냥 문장 맨 앞의 불필요한
# 콜론만 뗀다. 순수 구조 신호(콜론 유무)만 쓰고 문법/품사는 안 쓴다.
# 전체 라이브 풀(2,502건) 재검증: 선행 콜론 잔존 0건.
_LEADING_COLON_RE = re.compile(r"^\s*[:：]\s*")


def _strip_leading_colon(segment: str) -> str:
    lines = [_LEADING_COLON_RE.sub("", ln) for ln in segment.split("\n")]
    return "\n".join(lines)


# Merge Step 2 - Delimiter Merge v1.0 (Frozen, 2026-07-31). "짝이 끊어진
# Delimiter" 복원. 실측(전체 라이브 풀 2,501건, [] / () 짝 815건 전수 분류)
# 근거로 범위를 아래 두 가지 "복원이 증명된 경우"로만 한정한다. 추측에 의한
# 삭제/병합은 하지 않는다.
#
# Scope: [] Pair, () Pair (실측 근거 부족한 【】/{}/<>/「」/『』는 제외)
#
# Implemented:
#   (A) Single-line Delimiter Cleanup, 326건 - 줄 맨 앞에 낙오된 닫힘기호가
#       있고, 그 "이전 줄"이 대응하는 여는기호로 끝나지 않는 경우(=진짜
#       orphan, 병합 상대가 없음). 콜론 정리와 동일한 메커니즘 - 그 기호
#       하나만 떼어낸다. 같은 줄 안에 실제 내용이 이미 온전히 있으므로
#       정보 손실이 없다.
#       예) "] 이런 업무를 합니다 - ..." -> "이런 업무를 합니다 - ..."
#   (B) Verified Two-line Delimiter Merge, 297건 - 현재 줄이 여는기호로
#       끝나고, "바로 다음 줄"이 정확히 그 짝 닫힘기호로 시작하는 경우만
#       두 줄을 하나로 합친다(검증된 복원).
#       예) "브랜드 전략 (" + ") CRM 운영" -> "브랜드 전략 ( ) CRM 운영"
#
# Explicitly Excluded (434건, C):
#   - 3+ line chaining(다음 줄까지 봐도 짝이 안 맞는 경우 - 실측 결과 상당수가
#     크롤링 과정의 실제 내용 손실(문구 자체가 사라짐)이거나 버튼/아이콘
#     잔재로, 병합하면 오히려 문장이 깨진다. 예: 특정 게임사 "...그 분입니다! ("
#     + "이면 더 좋습니다." -> 병합 시 "...분입니다! (이면 더 좋습니다."로 파손)
#   - Sentence Fragment Merge (다음 Merge 단계 영역, 별도로 다룬다)
#   - Single Token Merge (보류)
#   - Unverified delimiter reconstruction (추측에 의한 삭제/병합 전면 금지)
#
# 전체 라이브 풀 재검증(2,501건, 서버 재시작 후 실제 화면 확인 포함):
# E2E 크래시 0건, A 잔여 0건, B 잔여 11건(3+ line chaining이라 범위 밖,
# 강제 처리 안 함), C 435건 그대로 미보류. 대표 사례(특정 산업AI 스타트업 Data
# Scientist) 실제 화면에서 정상 노출 확인 완료.
# 주의: 짝 끊김이 자격요건(req) 마지막 줄과 우대사항(pref) 첫 줄 사이,
# 즉 서로 다른 헤더 세그먼트 경계에 걸쳐 있는 실사례(예: 특정 게임사)가 있어서,
# 이 함수는 각 헤더 세그먼트 단위가 아니라 rule_representation.py에서
# task/qualification 필드를 최종 조립한 뒤(req+pref 합친 뒤) 호출한다.
_DELIMITER_PAIRS = {"[": "]", "(": ")"}
_DELIMITER_OPENER_OF = {v: k for k, v in _DELIMITER_PAIRS.items()}


def _delimiter_orphan_flags(line: str):
    """이 줄 안에서 스택 기반으로 짝이 안 맞는 기호를 찾는다.
    반환: (lead, trail) - lead=줄 맨 앞의 낙오된 닫힘기호(또는 None),
    trail=줄이 끝날 때까지 안 닫힌 여는기호(또는 None)."""
    stack = []
    s = line.strip()
    if not s:
        return None, None
    lead = None
    for ch in s:
        if ch in _DELIMITER_PAIRS:
            stack.append(ch)
        elif ch in _DELIMITER_OPENER_OF:
            if stack and stack[-1] == _DELIMITER_OPENER_OF[ch]:
                stack.pop()
            else:
                if lead is None and not stack:
                    lead = ch
    trail = stack[-1] if stack else None
    if s[0] not in _DELIMITER_OPENER_OF or s[0] != lead:
        lead = None
    return lead, trail


def merge_delimiter_lines(segment: str) -> str:
    lines = segment.split("\n")
    result = []
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i]
        cur_stripped = cur.strip()
        if not cur_stripped or len(cur_stripped) < 4:
            result.append(cur)
            i += 1
            continue

        lead, trail = _delimiter_orphan_flags(cur)
        merged_cur = cur
        consumed_next = False

        # (B) 검증된 2줄 병합: 다음 줄이 정확히 짝이 맞는 닫힘기호로 시작할 때만
        if trail and i + 1 < n:
            next_stripped = lines[i + 1].strip()
            if next_stripped and next_stripped[0] == _DELIMITER_PAIRS[trail]:
                merged_cur = cur_stripped + " " + next_stripped
                consumed_next = True

        # (A) 낙오된 선행 닫힘기호 - 이전 줄이 그 짝의 여는기호로 끝나지 않을 때만 제거.
        # B에서 병합이 일어났어도 이 줄 "맨 앞"의 lead는 병합과 무관하게 그대로
        # 유효한 판단 대상이다(예: 특정 커머스 - req 마지막 줄이 "["로 끝나 pref 첫
        # 줄의 "]"과 B로 병합되면서도, 그 req 첫 줄 자체의 선행 "]"는 별개로 A 대상).
        if lead:
            prev_stripped = result[-1].strip() if result else ""
            prev_ends_with_opener = bool(prev_stripped) and prev_stripped[-1] == _DELIMITER_OPENER_OF[lead]
            if not prev_ends_with_opener:
                merged_cur = re.sub(r"^\s*" + re.escape(lead) + r"\s*", "", merged_cur, count=1)

        result.append(merged_cur)
        i += 2 if consumed_next else 1
    return "\n".join(result)


# Merge Step 3-1 - Leading Punctuation Merge v1.0 (Frozen, 2026-07-31).
# 이 Step으로 Colon Line Merge / Delimiter Merge / Leading Punctuation
# Merge 세 가지를 묶어 "구조 기반 Merge"(Merge v1)를 완료한다. 다음 단계
# (lowercase-start, Sentence Fragment, Single Token)는 원본 정보 일부가
# 이미 소실된 상태에서 복원 가능 범위를 다루는 문제라 성격이 다르므로,
# "Sentence Fragment Recovery"라는 별도 단계로 분리해서 나중에 착수한다.
#
# 실측(전체 라이브 풀, task/qualification 인접 줄 12,490쌍 후보 수집) 근거로
# 범위를 순수 구두점 신호 하나로만 한정한다 - 문법/어미 사전은 쓰지 않는다.
#
# 발견 과정(중요): "종결 문장부호로 안 끝나는 줄" 전체를 후보로 모으면
# 12,490건이 나오지만, 대다수(현재 줄 마지막 단어 1위 "분" 3,142건, 이하
# 경험/관리/능력 등)는 한국어 JD 불릿이 마침표 없이 명사형으로 끝나는
# 정상 형태였다 - 진짜 Fragment가 아니라 필터 자체의 오탐. 그중 "다음 줄이
# 쉼표(,)로 시작"하는 48건만 표본 전수 확인 결과 100% 진짜 문장 분리였고
# (영어/한국어 공통, 10개 회사), 병합하면 예외 없이 정상 문장으로 복원됐다.
# 반면 "다음 줄이 소문자로 시작"(471건, lowercase-start)은 같은 신호
# 안에서도 성격이 갈렸다 - 특정 해외 테크기업 "Architect and iterate" + "on
# high-performance models..."는 깨끗이 복원되지만, 같은 공고 "How Do I Know
# if the" + "is Right For Me?"는 합쳐도 "...if the is Right For Me?"로
# 여전히 깨져 있다(원본 자체에서 강조 텍스트가 통째로 소실됨 - Merge로
# 해결 불가). 이 둘을 가르려면 관사/지시사 여부를 봐야 하는데 이건 사실상
# 영어 문법 Rule이 되므로, lowercase-start는 이번 범위에서 제외하고
# Sentence Fragment Recovery(별도 단계, 미착수)로 남긴다.
#
# Scope: 다음 줄이 "," 또는 "，"로 시작하는 경우만(데이터에 세미콜론 등
# 다른 구두점 사례가 실측되면 그때 추가한다 - 미리 확장하지 않는다).
#
# Validation: Candidate Pair Analysis(12,490쌍 전수, comma-lead 48건 표본
# 전수 확인) / Golden Cases PASS(docs/verification/2026-07-31_delimiter_
# merge/golden_cases.py) / Expected Failure 유지(회귀 없음) / E2E PASS
# (2,495건 크래시 0) / DB-level Verification PASS(build_rule_based_
# representation() 직접 호출 결과로 코빗 DevOps Engineer 병합 확인).
#
# Note: 브라우저 확장 프로그램 연결 문제로 이번 검증에는 실화면 스크린샷을
# 포함하지 못했다. 다만 DB 레벨 출력과 Golden Case 결과가 일치함을
# 확인했고(Streamlit 서버 재시작으로 최신 코드 반영도 확인 완료), 이 정도
# 근거로 Freeze 조건을 충족한다고 판단했다.
_LEADING_PUNCT_RE = re.compile(r"^[,，]")


def merge_leading_punctuation_lines(segment: str) -> str:
    lines = segment.split("\n")
    result = []
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i]
        cur_stripped = cur.strip()
        if cur_stripped and i + 1 < n:
            next_stripped = lines[i + 1].strip()
            if next_stripped and _LEADING_PUNCT_RE.match(next_stripped):
                result.append(cur_stripped + next_stripped)
                i += 2
                continue
        result.append(cur)
        i += 1
    return "\n".join(result)


# Task Ending Lexicon v1.0 (Frozen, 2026-07-28) - "이 줄이 Task처럼 생겼는가"를
# 판단하는 양성 신호(Signal). _BOILERPLATE_LINE_RE(제거 대상 블랙리스트)와는
# 반대 방향 - 이건 "지워야 할 나쁜 문장"이 아니라 "업무 내용처럼 보이는
# 문장"을 긍정적으로 인식한다.
#
# 검증 이력: 실제 후보 풀에서 뽑은 443건 라벨링 검증셋(Task/Qualification/
# Preferred/Culture/Process/Noise)으로 측정, Task Precision 0.93 / Recall
# 0.84 / F1 0.88(어간 사전 확장 전엔 Recall 0.30 - 3배 가까이 개선). 동사와
# 명사를 별도 목록으로 관리하지 않는다 - "설계"라는 어간 하나가 "설계합니다/
# 설계/설계하고" 등 활용형과 무관하게 전부 같은 Task 신호로 인식되도록
# 어간 하나로 관리한다(활용형별로 규칙을 따로 만들면 유지보수가 끝없이
# 늘어난다는 게 실측으로 확인됨).
#
# Known Limitation(중요, 아직 미해결):
# 1. 이 신호는 Qualification/Preferred/Culture 신호와 별도로 검증 완료되지
#    않았다 - Task 신호만 Freeze한 것이고 나머지는 그대로 두었다.
# 2. "여러 문장이 줄바꿈 없이 한 줄로 뭉친 경우"(Merge 실패, 검증셋에서
#    18건 발견)는 이 함수로 못 잡는다 - 그 케이스들은 애초에 이 검증셋
#    자체에서 "줄 단위로는 채점 불가능"하다고 판단해 평가에서 제외했을
#    뿐, 실제 파이프라인에도 Merge 로직은 아직 없다. 이 문제는 그대로
#    남아있다.
_TASK_ENDING_STEMS = [
    "수립", "기획", "운영", "개발", "관리", "설계", "분석", "개선", "구축", "제공",
    "실행", "수행", "담당", "진행", "도출", "강화", "확장", "운용",
    "최적화", "고도화", "튜닝", "조율", "평가", "제안", "지원", "처리", "리딩",
    "설정", "정리", "다지", "조정", "협업", "커뮤니케이션", "총괄", "실무",
    "의사결정", "연결", "탐색", "발굴", "전달", "매니징", "검증", "집중",
    "안정화", "효율화", "확보", "점검", "대응", "검토", "제작", "작성", "주도",
    "판단", "결정",
]
_TASK_ENDING_RE = re.compile(
    r"(" + "|".join(_TASK_ENDING_STEMS) + r")"
    r"(하게\s*됩니다|합니다|해요|하고|하며|하기|하는|할|함|임|됩니다)?\s*$"
    r"|만듭니다\s*$|만들고\s*$|만드는\s*$"
)


def is_task_like_line(line: str) -> bool:
    """이 줄이 Task(업무) 내용처럼 보이는지 판단한다(양성 신호, Frozen v1.0).
    마침표/물음표 등 문장부호가 어간 판별을 방해하지 않도록 먼저 제거한다
    (실측 확인 - 이게 없으면 "...설계합니다."처럼 이미 목록에 있는 어간도
    마침표 때문에 매칭에 실패했다)."""
    stripped = re.sub(r"[.!?)\]】」』]+\s*$", "", line.strip())
    return bool(_TASK_ENDING_RE.search(stripped))


# Qualification Signal Lexicon v1.0 (Frozen, 2026-07-29) - "이 줄이
# 자격요건/우대사항 내용처럼 보이는가"를 판단하는 양성 신호. Task와 같은
# 원칙(개별 활용형을 나열하지 않고 일반화된 패턴 하나로 관리)을 따른다.
#
# 검증 이력: 같은 443건 검증셋으로 측정, Qualification Precision 0.62 /
# Recall 0.60 / F1 0.61(1차 8개 문구 나열 방식 대비 Recall 0.40->0.60).
#
# Validation Finding(중요, 코드를 보는 사람이 반드시 알아야 함): FP(오탐)
# 56건을 전수 확인한 결과 100%가 실제 Preferred 문장이었다 - Task/Culture/
# Process/Noise를 Qualification으로 잘못 판정한 사례는 0건이었다. 즉
# Precision 0.62라는 숫자만 보면 낮아 보이지만, 실제로는 "완전히 다른
# 종류의 문장을 잘못 잡은" 오탐이 아니라 "Qualification과 Preferred가
# 문장 형태로는 원천적으로 구분 불가능하다"는 이미 알려진 구조적 한계가
# 그대로 드러난 것뿐이다 - Rule만으로 이 둘을 완전히 분리하는 것은
# 불가능하고, 최종 구분은 이 줄이 어느 헤더(자격요건 vs 우대사항) 아래
# 있었는지(Bucket 정보)로만 가능하다.
#
# 일반화 설계에서 정밀도를 지킨 방법: "(형용사/동사 활용형) + 분/자"
# 패턴은 그 자체로는 신호를 안 준다("성장하고 싶은 분"류 채용 브랜딩
# 문장까지 다 잡혀 Precision이 떨어지는 실패를 피하기 위함) - 핵심 명사
# (경험/능력/역량 등)와 같은 줄에 있을 때만 신호로 인정한다.
_QUAL_PERSON_ENDING_RE = re.compile(r"[가-힣]{2,}\s*(분|자)\s*$")
_QUAL_CORE_NOUNS = [
    "경험", "능력", "역량", "이해", "지식", "자격", "학위", "전공", "보유",
    "자격증", "숙련", "관심", "이해도", "숙련도", "무관", "소유자", "보유자",
]
_QUAL_CORE_NOUN_RE = re.compile("(" + "|".join(_QUAL_CORE_NOUNS) + ")")
_QUAL_CORE_NOUN_END_RE = re.compile("(" + "|".join(_QUAL_CORE_NOUNS) + r")\s*$")
_QUAL_TEMPLATE_RE = re.compile(
    r"분을\s*(찾습니다|원해요|원합니다)\s*$"
    r"|분이\s*(면|라면)\s*좋(아요|습니다)\s*$"
    r"|있다면\s*더?\s*좋습니다\s*$"
    r"|우대합니다\s*$"
    r"|필수입니다\s*$"
)
_QUAL_MEDIUM_RE = re.compile(r"\d+\s*년\s*이상|\d+\s*~\s*\d+\s*년|학사\s*이상")
# 영어 Qualification Signal - 별도 레이어(한국어 사전에 안 섞음). Task와
# 마찬가지로 영어 커버리지는 제한적이며 향후 별도 과제로 남긴다.
_QUAL_ENGLISH_RE = re.compile(
    r"\bexperience\b|\bskills?\b|\bdegree\b|\bunderstanding\b|\bis a plus\b|\byears?\s+(in|of)\b",
    re.I,
)


def is_qualification_like_line(line: str) -> bool:
    """이 줄이 자격요건/우대사항 내용처럼 보이는지 판단한다(양성 신호,
    Frozen v1.0). Known Limitation: Preferred와 구조적으로 분리 불가(위
    Validation Finding 참고), 여러 문장이 한 줄로 뭉친 경우(Merge 미해결)
    못 잡음, 영어 커버리지 제한적."""
    stripped = re.sub(r"[.!?)\]】」』]+\s*$", "", line.strip())
    has_core_noun = bool(_QUAL_CORE_NOUN_RE.search(stripped))
    if _QUAL_CORE_NOUN_END_RE.search(stripped):
        return True
    if _QUAL_PERSON_ENDING_RE.search(stripped) and has_core_noun:
        return True
    if _QUAL_TEMPLATE_RE.search(stripped):
        return True
    if _QUAL_MEDIUM_RE.search(stripped):
        return True
    if _QUAL_ENGLISH_RE.search(stripped):
        return True
    return False


def _classify_header_for_display(header_text: str) -> str:
    if _HDR_NOISE.search(header_text):
        return "discard"
    if _HDR_PREF.search(header_text):
        return "pref"
    if _HDR_RESP.search(header_text):
        return "resp"
    if _HDR_REQ.search(header_text):
        return "req"
    return "company"


def _classify_display_sections(job: dict) -> dict[str, list[str]]:
    """공고 상세보기 전용 섹션 분류 - candidate_search._classify_sections()
    (검색용, 손대지 않음 - Retrieval Freeze)와 완전히 분리된 함수다.

    2026-07-25 Header Detector 교체(실측 근거, docs/verification/
    2026-07-25_m3_representation_ab/summary.md) - 기존엔 "20자 이하
    독립 줄"만 헤더로 인정했는데(오탐 방지 목적, 2026-07-16 설계),
    실측 결과 기업 홈페이지 소스 공고의 43.0%가 "헤더 문구는 원문에
    있지만 그 줄이 20자를 넘는다"는 이유로 통째로 실패했다(task/
    qualification 둘 다 공백) - 독립 줄 조건 자체가 원인이었다.
    header_patterns.find_header_matches()(Strong/Weak 2단계 - 길고
    구체적인 문구는 문장 어디서든, 짧고 일반적인 문구는 콜론/괄호/
    독립줄 조건을 만족할 때만 헤더로 인정)로 교체해서 기업 홈페이지
    실패율을 43.0%->1.2%로 낮췄고, 원티드(기존에도 잘 되던 소스)는
    1.1%->0.0%로 오히려 개선됐다(회귀 없음 실측 확인)."""
    raw = job.get("posting_text", "") or ""
    plain = _TAG_RE.sub(" ", raw)
    plain = _ENTITY_RE.sub(" ", plain)

    buckets: dict[str, list[str]] = {"resp": [], "req": [], "pref": [], "company": [], "discard": []}
    matches = find_header_matches(plain)
    if not matches:
        buckets["company"].append(plain)
        return buckets
    if matches[0].start() > 0:
        buckets["company"].append(plain[: matches[0].start()])
    for i, m in enumerate(matches):
        label = _classify_header_for_display(m.group())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(plain)
        segment = plain[start:end].strip()
        if label in ("resp", "req", "pref"):
            segment = _strip_leading_colon(segment)
            segment = _strip_boilerplate_lines(segment)
        if segment:
            buckets[label].append(segment)
    return buckets

# 산업(業種) - DB의 industry 컬럼은 실측 결과 0% 채워져 있다(수집 단계에서
# 아예 안 들어옴, 2026-07-15 확인). 하지만 JD 본문 자체에 회사가 자기
# 사업 영역을 직접 언급하는 경우가 많다(예: 특정 여행 플랫폼 JD에 "여행
# 산업(Online Travel Agency)의..."라고 직접 나옴) - 그래서 외부
# API/스크래핑 없이, 이미 있는 본문 텍스트에서 업종 키워드를 표면적으로
# 찾는다(LLM 없음, career_filter.py/job_prep.py와 같은 방식 - 추측이
# 아니라 텍스트에 실제로 그 단어가 있을 때만 표시). 애매하면 빈 값을
# 반환한다(지어내지 않는다).
# 실측으로 오탐을 확인하고(2026-07-15: 특정 게임사 채용공고가 "AI"로
# 잘못 분류됨 - 실제로는 게임회사인데, AI 관련 직무/툴 언급이 회사
# 소개보다 먼저 나온 경우) 그 사례를 계기로, "직무 스킬로도 흔히
# 언급되는" 범용 기술 키워드(AI/SaaS/광고/HR테크 등)는 목록에서 뺐다 -
# 이런 단어는 산업을 나타내기보다 담당 업무/사용 기술로 언급되는
# 경우가 실제로 더 많아서, 회사의 실제 업종과 다르게 나올 위험이 크다.
# 아래 키워드들은 상대적으로 "회사의 사업 자체"를 가리킬 때만 쓰이는
# 명사라 오탐 위험이 낮다(그래도 100% 안전하지는 않다 - 애매하면
# 빈 값을 반환하는 게 우선).
_INDUSTRY_KEYWORDS: list[tuple[str, str]] = [
    ("이커머스", "커머스"), ("전자상거래", "커머스"),
    ("중고거래", "커머스"), ("리셀 플랫폼", "커머스"),
    ("여행 산업", "여행/OTA"), (" OTA", "여행/OTA"), ("숙박 예약", "여행/OTA"),
    ("항공권", "여행/OTA"),
    ("게임 개발사", "게임"), ("게이밍", "게임"), ("게임사", "게임"), ("게임 스튜디오", "게임"),
    ("핀테크", "핀테크"), ("간편송금", "핀테크"), ("간편결제", "핀테크"),
    ("증권사", "핀테크"), ("보험사", "핀테크"), ("가계부 서비스", "핀테크"),
    ("디지털 헬스케어", "헬스케어"), ("제약회사", "헬스케어"), ("바이오 기업", "헬스케어"),
    ("병원 정보", "헬스케어"),
    ("에듀테크", "교육"), ("교육 플랫폼", "교육"), ("이러닝", "교육"),
    ("물류 플랫폼", "물류"), ("배달 플랫폼", "물류"), ("풀필먼트", "물류"),
    ("모빌리티 플랫폼", "모빌리티"), ("차량 공유", "모빌리티"),
    ("프롭테크", "부동산"), ("부동산 플랫폼", "부동산"),
    ("OTT 서비스", "미디어/엔터"), ("엔터테인먼트", "미디어/엔터"), ("콘텐츠 플랫폼", "미디어/엔터"),
    ("웹툰", "미디어/엔터"), ("음악 스트리밍", "미디어/엔터"),
    ("반도체", "반도체/제조"),
]


def infer_industry(intro_text: str) -> str:
    """업종 키워드를 표면적으로 찾는다(LLM 없음). 반드시 회사소개 성격의
    텍스트(예: extract_display_sections()["other"])만 넣어야 한다 -
    전체 JD 본문(자격요건 등)을 넣으면 오탐이 난다(실측 확인,
    2026-07-15: 특정 AI 스타트업 채용공고의 지원자격에 "증권사 재직 경력"이
    있어 회사 자체가 "핀테크"로 잘못 분류된 사례, 특정 게임사 채용공고의
    직무 스킬 언급 때문에 "AI"로 잘못 분류된 사례 - 둘 다 회사소개
    텍스트로 범위를 좁히자 사라짐). 여러 후보가 매치되면 텍스트에
    가장 먼저 등장하는 것을 쓴다."""
    if not intro_text:
        return ""
    best_label, best_pos = "", None
    for kw, label in _INDUSTRY_KEYWORDS:
        idx = intro_text.find(kw)
        if idx != -1 and (best_pos is None or idx < best_pos):
            best_pos, best_label = idx, label
    return best_label


def extract_display_sections(job: dict) -> dict:
    """반환: {"resp": str, "req": str, "pref": str, "other": str}
    (주요업무/자격요건/우대사항/회사소개, 각각 줄바꿈으로 이어붙인
    텍스트, 없으면 빈 문자열). `_classify_display_sections()`(표시
    전용, 위 참고)를 쓴다 - candidate_search._classify_sections()가
    아니다(2026-07-16, 동료의 한마디/FAQ/카카오톡 노이즈 누출 수정).
    `discard` 버킷은 애초에 반환하지 않는다 - 호출부가 실수로라도
    노이즈를 표시할 수 없게 한다."""
    buckets = _classify_display_sections(job)
    return {
        "resp": "\n".join(buckets["resp"]),
        "req": "\n".join(buckets["req"]),
        "pref": "\n".join(buckets["pref"]),
        "other": "\n".join(buckets["company"]),
    }


def format_company_intro(job: dict, max_len: int = 150) -> str:
    """회사 소개 한 줄 - "other" 버킷(회사소개/복지 등 낮은 우선순위
    텍스트)의 앞부분을 문장 경계 기준으로 잘라 반환한다. 새 로직이
    아니라 기존 분류 결과를 표시용으로 재활용하는 것 - 헤더가 없는
    공고는 빈 문자열을 반환할 수 있다(§ui_design.md 6 실측 필요)."""
    other = extract_display_sections(job).get("other", "")
    if not other:
        return ""
    snippet = other[:max_len]
    cut = max(snippet.rfind("."), snippet.rfind("다 "), snippet.rfind("다."))
    if cut > 20:
        snippet = snippet[:cut + 1]
    return snippet.strip()


# 처음 등장할 때만 "용어(쉬운 설명)" 형태로 풀어준다(스펙 예시 그대로) -
# 매번 통째로 치환하면 정작 그 용어를 아는 사용자에게는 문장이 부자연스러워
# 지고, 아예 안 바꾸면 모르는 사용자가 못 알아듣는다. 목록에 없는 용어는
# 건드리지 않는다(추측 설명을 지어내지 않는다 - 이 프로젝트 전체 원칙).
_JARGON_MAP: dict[str, str] = {
    "retention": "기존 사용자가 다시 이용하는 비율",
    "funnel": "사용자가 거치는 단계",
    "cohort": "같은 시기에 가입한 사용자 집단",
    "etl": "데이터를 수집·가공·저장하는 과정",
    "data pipeline": "데이터가 자동으로 처리되는 흐름",
    "churn": "사용자가 이탈하는 비율",
    "dau": "하루 동안 서비스를 이용한 사용자 수",
    "mau": "한 달 동안 서비스를 이용한 사용자 수",
    "ltv": "고객 한 명이 평생 발생시키는 매출",
    "cac": "고객 한 명을 데려오는 데 드는 비용",
    "a/b test": "두 가지 버전을 비교해 더 나은 쪽을 찾는 실험",
    "kpi": "성과를 확인하는 핵심 지표",
    "stakeholder": "의사결정에 관련된 사람들",
    "backlog": "앞으로 처리할 작업 목록",
    "attribution": "어떤 채널이 성과에 얼마나 기여했는지 계산하는 것",
}
_JARGON_RE = [(term, re.compile(re.escape(term), re.I)) for term in _JARGON_MAP]


def _simplify_jargon(text: str) -> str:
    if not text:
        return text
    result = text
    for term, pattern in _JARGON_RE:
        def _sub(m, term=term):
            return f"{m.group(0)}({_JARGON_MAP[term]})"
        result = pattern.sub(_sub, result, count=1)
    return result


def _parsed_short_semantic(job: dict) -> dict:
    if job.get("_jd_short_semantic"):
        return job["_jd_short_semantic"]
    raw = job.get("jd_short_semantic")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _bullets(text: str, max_n: int = 5) -> list[str]:
    """줄바꿈 텍스트를 불릿 목록으로 - 앞의 기호(•, -, 숫자.)만 벗겨낸다."""
    out = []
    for line in (text or "").split("\n"):
        s = line.strip().lstrip("•-·").strip()
        if s:
            out.append(s)
        if len(out) >= max_n:
            break
    return out


_IMPORTANCE_ORDER = {"critical": 0, "core": 1, "normal": 2, "low": 3}


def culture_highlights(job: dict, max_n: int = 5) -> list[str]:
    """JD Understanding이 이미 추출한 culture 레이어(조직문화/인재상)
    객체를 그대로 보여준다 - "이 회사가 중요하게 보는 인재상" 정보용
    카드(공고 상세 화면). 새 LLM 호출 없음, 새 판단 없음 - importance
    (critical/core 먼저) 순으로 정렬만 해서 보여준다.

    2026-07-21(persona_tags 채택 이후) - 인재상은 이제 customization_
    rules.py의 ②(핵심 경험 및 어필 전략)에서 persona_tags 기반으로
    자동 강조도 함께 한다(_culture_pairs_via_tags). 이 함수는 그것과
    무관하게 "이 공고가 원하는 인재상이 뭔지"를 상세 화면에서 그냥
    보여주는 별개의 정보 표시 기능이다."""
    raw = job.get("jd_semantic_objects")
    if not raw:
        return []
    try:
        objects = json.loads(raw)
    except (TypeError, ValueError):
        return []
    culture_objs = [o for o in objects if o.get("layer") == "culture" and o.get("normalized_text")]
    culture_objs.sort(key=lambda o: _IMPORTANCE_ORDER.get(o.get("importance"), 4))
    return [o["normalized_text"] for o in culture_objs[:max_n]]


def build_jd_detail_view(job: dict) -> dict | None:
    """공고 상세보기 화면용 데이터. 이력서를 참조하지 않는다(JD Context +
    JD short_semantic + job_prep 정규식 추출만 사용). Representation이
    없는 공고(BM25 fallback 대상 - 드문 경우)는 None을 반환한다 - 호출부가
    원문 섹션(extract_display_sections)만으로 대체 표시해야 한다.

    반환 구조는 4개 탭(요약/상세분석/요구스킬/조직문화)에 그대로 대응한다."""
    sem = _parsed_short_semantic(job)
    context = job.get("jd_context") or ""
    if not sem and not context:
        return None

    sections = extract_display_sections(job)
    prep = extract_application_prep(job)
    tech_stack = extract_tech_stack(job.get("posting_text", ""))

    role = sem.get("핵심역할") or (context.split(".")[0] + "." if context else "")
    problem = sem.get("핵심문제") or ""
    expect = sem.get("핵심기대") or ""
    not_fit = sem.get("부적합한경우")
    not_fit = _simplify_jargon(not_fit) if not_fit and not_fit != "null" else None

    return {
        "요약": {
            "핵심역할": _simplify_jargon(role),
            "핵심업무": _bullets(sections.get("resp", "")),
        },
        "상세분석": {
            "해결하려는_문제": _simplify_jargon(problem),
            "중요하게_보는_역량": _simplify_jargon(expect),
            "신중검토": not_fit,
        },
        "요구스킬": {
            "주요_기술": tech_stack,
            "자격요건": sections.get("req") or "",
            "우대사항": sections.get("pref") or "",
        },
        "조직문화": {
            "회사소개": format_company_intro(job) or "회사 소개 정보가 없습니다.",
            "산업": job.get("industry") or infer_industry(sections.get("other", "")),
            "기업규모": job.get("company_size") or "",
            "근무지역": job.get("location") or "",
            "인재상": culture_highlights(job),
        },
        "지원_정보": {
            "지원절차": prep.get("application_process") or [],
            "필요서류": prep.get("required_documents") or [],
            "마감": prep.get("deadline"),
            "기타": prep.get("job_info") or {},
        },
    }


def derive_strengths_and_gaps(matched_keywords: list[str], posting_text: str, max_n: int = 5) -> tuple[list[str], list[str]]:
    """강점/보완점 태그 - 새 LLM 호출 없이 이미 계산된 신호만 비교한다.
    강점: 이력서-JD 키워드 겹침(matched_keywords, candidate_search.py가
    이미 계산). 보완점: JD가 요구하는 기술(extract_tech_stack, 정규식)
    중 겹치지 않는 것. 순수 표면적 키워드/기술명 비교이지 LLM의 의미
    판단이 아니다 - "부족하다"는 판단이 아니라 "이 기술은 JD에 있는데
    이력서 키워드에는 없다"는 사실만 보여준다."""
    matched_clean = [k for k in (matched_keywords or []) if not k.isdigit()]
    strengths = matched_clean[:max_n]
    matched_lower = {k.lower() for k in matched_clean}
    tech_stack = extract_tech_stack(posting_text or "")
    gaps = [t for t in tech_stack if t.lower() not in matched_lower][:max_n]
    return strengths, gaps


def derive_surface_comparisons(job: dict, max_n: int = 5) -> list[dict]:
    """②AI Meaning Matching 표를 LLM Top30 판단이 없는 공고(31~50위,
    배치 실패 등)에서도 같은 모양(jd_point/resume_evidence/verdict/
    explanation)으로 채운다 - 새 LLM 호출은 하지 않는다. matched_keywords/
    extract_tech_stack(둘 다 이미 계산된 표면 키워드 겹침 신호,
    derive_strengths_and_gaps와 동일한 재료)만으로 "이 기술 키워드가
    이력서에 표면적으로 있는가"만 판정한다 - LLM이 의미를 판단한 것처럼
    보이지 않도록 explanation에 항상 "키워드 표면 일치"라고 명시한다.
    tech_stack이 비어 있으면(알려진 기술 키워드 목록에 없는 JD) JD
    섹션(자격요건/우대사항) 기준 매칭 그룹으로 대체한다."""
    posting_text = job.get("posting_text", "") or ""
    matched = job.get("matched_keywords") or []
    matched_lower = {k.lower() for k in matched if not k.isdigit()}
    tech_stack = extract_tech_stack(posting_text)

    rows = []
    for tech in tech_stack[:max_n]:
        hit = tech.lower() in matched_lower
        rows.append({
            "jd_point": f"{tech} 활용 경험",
            "resume_evidence": f"이력서 키워드에 '{tech}' 등장" if hit else "이력서에서 관련 키워드를 찾지 못함",
            "verdict": "충족" if hit else "보완필요",
            "explanation": "키워드가 이력서·공고 양쪽에 표면적으로 등장하는지만 확인한 결과입니다(LLM 의미 판단 아님).",
        })
    if rows:
        return rows

    sections = extract_display_sections(job)
    groups = group_recommendation_reasons(matched, sections)
    for category, labels in groups.items():
        if not labels:
            continue
        rows.append({
            "jd_point": category,
            "resume_evidence": ", ".join(labels),
            "verdict": "충족",
            "explanation": "이력서와 공고의 키워드가 겹치는 부분만 표시합니다(LLM 의미 판단 아님).",
        })
        if len(rows) >= max_n:
            break
    return rows
