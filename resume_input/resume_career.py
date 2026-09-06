"""
resume_input/resume_career.py

이력서에서 "지원하려는 Job Family와 관련된 실무 경력"만 합산해
연차를 판단하는 범용 엔진. Career Filter가 검색 대상 JD Pool을
줄이는 데 쓰는 값이라, 총 사회 경력이 아니라 관련 실무 경력만
계산한다.

LLM 호출 없음 - 휴리스틱/정규식 기반.

이 파일은 어떤 Job Family가 이번 추천 대상인지 모른다 - 호출부가
`resume_input.job_family`에서 가져온 JobFamily 객체를 인자로
넘겨줘야 한다. Data Analytics 전용 키워드 지식은 전부
`job_family.py`에 있다. Backend/Frontend/PM 등 다른 Job Family를
추가할 때 이 파일은 수정할 필요가 없다 - job_family.py에 새
JobFamily만 추가하면 된다.

판단 순서(중요 - 역할 분리):
    경력 섹션 추출 -> 항목(Experience) 추출 -> **실무경력 여부 판단**
    (resume_career.py의 책임) -> **Job Family 관련 여부 판단**
    (target_job_family.classify()에 위임) -> 실무경력이면서 관련
    있는 항목만 기간 병합 -> 개월수 합산 -> 연차 결정

"관련 여부"보다 "실무경력 여부"를 먼저 본다. 경력 섹션 헤더로 이미
프로젝트/포트폴리오/부트캠프/공모전/자격증 섹션 자체는 걸러지지만,
그 섹션들과 별개로 "경력" 섹션 **안에** 개인 프로젝트/사이드
프로젝트/부트캠프 항목이 섞여 적혀 있는 이력서도 있다 - 이런 항목은
Job Family와 관련된 키워드가 있어도(예: "데이터 분석 개인 프로젝트")
연차 계산에 포함하면 안 된다. 이 판단은 JobFamily에 맡기지 않는다 -
JobFamily의 책임은 "관련 직무인가"만이고, "실무경력인가"는
resume_career.py 자신의 책임으로 분리해서 유지한다(JobFamily의
역할이 커지지 않도록).

confidence 정의(중요 - 판단 "결과"가 아니라 판단 "근거의 신뢰도"):
- "explicit": **관련(included) 실무경력 항목 자체에 "데이터 분석
  경력 3년", "BI 실무 2년"처럼 연차가 명시적으로 적혀 있는 경우만**.
  버그 이력(구현 중 발견해서 수정) - 처음엔 이력서 "아무 곳에나"
  있는 "총 경력 N년" 문구를 찾아서 explicit로 올렸는데, 이러면
  "총 경력 5년"이 전부 무관한 경력(간호조무사 등)이라 관련 실무경력이
  0년(신입)인 이력서도 explicit로 표시되는 문제가 있었다 - "확신이
  높은 판단"처럼 보이지만 실제로는 그 확신의 근거(총 경력 문구)가
  관련 경력과 무관했다. 그래서 "총 경력" 같은 포괄적 문구는 신호로
  쓰지 않고, **이미 관련 실무경력으로 판정된(included=True) 항목의
  본문에서만** 명시적 연차 언급을 찾는다 - 이 신호는 정의상 이미
  관련 있다고 판정된 경력에 대한 것이므로 "무관한데 explicit"가 될
  수 없다.
- "calculated": 재직 기간을 실제로 합산해서 계산한 기본 경우(가장
  흔함) - 관련 경력이 0건이라서 "신입"으로 판단한 것도 포함한다
  ("간호조무사 5년 -> 신입"은 판단 실패가 아니라 확실한 계산 결과다).
- "ambiguous": included(실무경력+관련) 구간끼리 날짜가 겹쳐서
  구간 병합이 실제로 발생한 경우 - 병합 로직 자체는 정상 동작하지만
  원본 이력서의 기간 표기가 깔끔하지 않다는 뜻이라 사용자 확인이
  필요하다.
- "none": 경력 섹션 자체를 못 찾았거나, 섹션은 있는데 재직 기간
  패턴을 하나도 못 찾은 경우 - 판단 근거 자체가 부실하다는 뜻.

"ambiguous"/"none"은 UI에서 사용자 확인을 유도해야 한다(S0의
[변경] 강조 표시) - 기능적으로 차단하지는 않는다(career_level은
이 경우에도 항상 4개 값 중 하나로 확정됨, §0-6 원칙 유지).

버킷 경계 규칙(실측 예시 기준으로 고정): "마케팅 분석 1년 + BI 2년"
= 36개월 = "1~3년"으로 판정되어야 한다는 예시를 기준으로, 36개월까지는
"1~3년"에 포함하고 "3~5년"은 36개월 초과부터 시작한다(경계값은
상한 쪽에 포함시키는 규칙 - n~m년 표기를 "n년 이상 ~ m년 이하"로 해석).

겹치는 재직기간 처리: included(실무경력 + 관련 있음)로 판정된
구간들은 절대 개월수를 단순 합산하지 않고, 구간을 병합(interval
merge)한 뒤 병합된 구간의 길이만 합산한다 - 부서 이동/겸직 등으로
기간이 겹치게 기재된 경우 중복 계산을 방지한다.

experiences 스키마(UI/디버깅/재계산에서 문자열을 다시 파싱할 필요가
없도록 구조화): role/company/start_date("YYYY-MM", 내부 계산용)/
end_date(동일)/period_text(UI 표시용, "2022.01 ~ 현재" 형식)/
months(이 경력 하나의 개월수 - 최종 related_experience_months는
included 항목들을 구간 병합한 값이라 이 값들의 단순 합과 다를 수
있음)/is_current(재직중 여부)/is_work_experience/is_related/
included/decision_reason(포함 또는 제외 이유 - 과거 필드명 "reason"에서
개명, 무엇에 대한 이유인지 명확히 하기 위함).

target_job_family.classify()의 반환 타입에 대한 의존을 낮추기 위해
_classify_relatedness() 어댑터를 거친다 - 지금은 str|None을 받지만,
나중에 classify()가 {"matched":bool, "category":str,
"matched_keywords":[...]} 같은 구조화된 객체를 반환하도록 바뀌어도
이 어댑터 함수만 고치면 되고 나머지 로직은 그대로 둘 수 있다.
"""
from __future__ import annotations

import re

from resume_input.job_family import JobFamily

# 경력 섹션 헤더 - 이 헤더를 만나면 그 아래를 경력 섹션으로 본다.
_CAREER_HDR = re.compile(
    r"^(경력\s*사항|경력|career(\s*history)?|work\s*experience|experience"
    r"|employment(\s*history)?|professional\s*experience"
    r"|직무\s*경험|실무\s*경험|업무\s*이력)\s*[:：]?\s*$",
    re.I,
)

# 경력 섹션이 끝났다고 볼 수 있는 다른 섹션 헤더 - 프로젝트/포트폴리오/
# 부트캠프/공모전/자격증은 명시적으로 연차 계산에서 제외해야 하는
# 항목이고(사용자 지시), 그 외 학력/스킬/자기소개 등도 일반적인 경계다.
_EXCLUDE_HDR = re.compile(
    r"^(프로젝트|포트폴리오|project(s)?|portfolio|부트캠프|bootcamp"
    r"|공모전|contest|자격증|certificat(e|ion)s?|교육(\s*과정)?|education"
    r"|학력|수상(\s*내역)?|award(s)?|스킬|기술\s*스택|skills?"
    r"|자기\s*소개|링크|활동)\s*[:：]?\s*$",
    re.I,
)

# 경력 섹션 "안에" 섞여 있을 수 있는 비실무경력 항목 마커 - 섹션
# 경계(_EXCLUDE_HDR)와 별개로, 개별 항목 단위에서도 한 번 더 걸러야
# 한다(예: "개인 프로젝트: 데이터 분석 사이드 프로젝트"가 경력 섹션
# 안에 그대로 적혀 있는 경우). resume_career.py 자신의 책임 -
# JobFamily에 위임하지 않는다.
_NON_WORK_MARKER_RE = re.compile(
    r"(개인\s*프로젝트|사이드\s*프로젝트|프로젝트\s*진행|포트폴리오"
    r"|부트캠프|bootcamp|공모전|contest|해커톤|hackathon|캡스톤"
    r"|자격증|certificat(e|ion)|교육\s*과정|스터디|동아리|학회)",
    re.I,
)

# 재직 기간 패턴 - "2022.03 - 2024.06" / "2022.03~2024.06" /
# "2022년 3월 ~ 2024년 6월" / "2022.03 - 현재" 등을 인식한다.
_DATE_RANGE_RE = re.compile(
    r"(?P<sy>\d{4})[.\-/년]\s*(?P<sm>\d{1,2})?\s*월?\s*"
    r"[~\-–]\s*"
    r"(?:(?P<ey>\d{4})[.\-/년]\s*(?P<em>\d{1,2})?\s*월?"
    r"|(?P<present>현재|지금|재직\s*중))"
)

# "회사명 | 직무" / "회사명 - 직무" 형식의 제목 줄을 회사/직무로
# 분리하는 구분자. 형식이 다른 이력서는 role에 줄 전체를 넣고
# company는 빈 문자열로 둔다(실제 다양성은 실측 필요 - §ui_design.md 6).
_ROLE_COMPANY_SPLIT_RE = re.compile(r"\s*[|｜/]\s*|\s+-\s+")

# "N년"/"N년차"/"N년 경력"/"N년 실무" 형태의 명시적 연차 언급 - 반드시
# 이미 included=True로 판정된 경력의 본문(날짜 줄 제외)에서만 찾는다
# (모듈 docstring 참고 - "총 경력" 같은 포괄적 문구는 관련 없는 경력도
# 섞여 있을 수 있어 신호로 쓰지 않는다). 1~2자리 숫자로 제한해 "2022년"
# 같은 4자리 연도가 오매치되지 않게 한다.
_RELATED_EXPLICIT_YEAR_RE = re.compile(r"\d{1,2}\s*년\s*(차|경력|실무)?")


def _find_career_section(resume_raw: str) -> str:
    lines = resume_raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _CAREER_HDR.match(line.strip()):
            start = i + 1
            break
    if start is None:
        return ""

    end = len(lines)
    for i in range(start, len(lines)):
        if _EXCLUDE_HDR.match(lines[i].strip()):
            end = i
            break
    return "\n".join(lines[start:end])


def _to_abs_month(y: int, m: int) -> int:
    return y * 12 + m


def _guess_title_line(context_text: str) -> str:
    for line in context_text.splitlines():
        stripped = line.strip(" -•\t")
        if 2 <= len(stripped) <= 50 and not _DATE_RANGE_RE.search(stripped):
            return stripped
    return context_text.strip()[:20] or "미상 경력"


def _guess_role_company(context_text: str) -> tuple[str, str]:
    """경력 항목의 제목 줄에서 회사/직무를 best-effort로 분리한다."""
    title = _guess_title_line(context_text)
    parts = _ROLE_COMPANY_SPLIT_RE.split(title, maxsplit=1)
    if len(parts) == 2:
        company, role = parts[0].strip(), parts[1].strip()
        return role, company
    return title, ""


def _classification_text(context_text: str, role: str) -> str:
    """Job Family 관련 여부 판단에 넘길 텍스트 - 직무명(role, 회사명은
    제외)과 실제 업무 내용(날짜 줄을 제외한 본문)만 포함한다.

    버그 이력(구현 중 발견해서 수정) - 처음엔 context_text(제목 줄
    원문 "회사명 | 직무" 그대로 + 날짜 + 본문)를 통째로 classify()에
    넘겼다. 이러면 회사명이 우연히 Job Family 키워드와 겹칠 때
    잘못 매칭될 위험이 있다(실측 검증: 회사명에 "그로스"가 들어간
    회사의 "총무팀 사원" 같은 무관한 직무가 회사명 때문에 "Growth
    Analyst" 관련 경력으로 오분류되는 것을 재현 테스트로 확인).
    그래서 직무명(회사명 제외)과 실제 업무 설명(날짜 줄 제외 본문)만
    골라서 넘긴다 - "직무명만"이 아니라 "직무명 + 실제 업무내용"
    둘 다 보되, 회사명이라는 무관한 신호는 뺀다."""
    body_lines = [
        line for line in context_text.splitlines()[1:]
        if not _DATE_RANGE_RE.search(line)
    ]
    return "\n".join([role] + body_lines)


def _has_explicit_related_year(context_text: str) -> bool:
    """이 경력 항목의 본문(날짜 줄 제외)에 "N년"/"N년차"/"N년 경력"
    같은 명시적 연차 언급이 있는지 - included=True인 항목에 대해서만
    호출되어야 한다(§ confidence "explicit" 정의 참고)."""
    for line in context_text.splitlines():
        if _DATE_RANGE_RE.search(line):
            continue
        if _RELATED_EXPLICIT_YEAR_RE.search(line):
            return True
    return False


def _is_work_experience(context_text: str) -> bool:
    """경력 섹션 안에 있어도 실무경력이 아닐 수 있는 항목(프로젝트/
    부트캠프/공모전/자격증 등)을 걸러낸다. Job Family 관련 여부와
    무관하게 독립적으로 판단한다 - JobFamily는 이 판단을 하지 않는다."""
    return not _NON_WORK_MARKER_RE.search(context_text)


def _extract_entries(section_text: str) -> list[dict]:
    """경력 섹션 텍스트에서 (기간, 컨텍스트) 단위로 항목을 나눈다.
    항목 경계는 재직기간 패턴을 기준으로 삼는다.

    버그 이력: 처음엔 "다음 항목의 날짜 줄 직전까지"를 현재 항목의
    컨텍스트로 잡았는데, 이러면 다음 항목의 회사/직무 표기 줄(날짜
    줄보다 위에 있는)이 현재 항목의 컨텍스트에 잘못 포함됐다(실측:
    "마케팅 분석가"(1번째 경력) 항목의 컨텍스트에 2번째 경력의
    "BI Analyst" 표기가 섞여 들어가 두 항목 모두 "BI Analyst"로
    오분류됨). 그래서 각 날짜 줄마다 자신의 "제목 시작 줄"을 먼저
    계산해두고, 그 지점을 기준으로 항목 경계를 나눈다 - 다음 항목의
    제목 줄은 절대 이전 항목 컨텍스트에 포함되지 않는다."""
    lines = section_text.splitlines()
    date_line_idx = [i for i, l in enumerate(lines) if _DATE_RANGE_RE.search(l)]
    if not date_line_idx:
        return []

    title_starts = []
    for idx in date_line_idx:
        start = idx
        j = idx - 1
        lookback = 0
        while j >= 0 and lookback < 2 and lines[j].strip() and not _DATE_RANGE_RE.search(lines[j]):
            start = j
            j -= 1
            lookback += 1
        title_starts.append(start)

    entries = []
    for n, idx in enumerate(date_line_idx):
        prev_bound = title_starts[n]
        next_bound = title_starts[n + 1] if n + 1 < len(date_line_idx) else len(lines)
        context = "\n".join(lines[prev_bound:next_bound])

        m = _DATE_RANGE_RE.search(lines[idx])
        sy, sm = int(m.group("sy")), int(m.group("sm") or 1)
        is_present = bool(m.group("present"))
        if is_present:
            from datetime import date
            today = date.today()
            ey, em = today.year, today.month
        else:
            ey, em = int(m.group("ey")), int(m.group("em") or 12)

        entries.append({
            "context": context,
            "sy": sy, "sm": sm, "ey": ey, "em": em,
            "is_present": is_present,
            "start_abs": _to_abs_month(sy, sm),
            "end_abs": _to_abs_month(ey, em),
        })
    return entries


def _merge_and_sum_months(intervals: list[tuple[int, int]]) -> tuple[int, bool]:
    """겹치거나 맞닿은 (start_abs, end_abs) 구간을 병합한 뒤 총
    개월수를 반환한다 - 겹치는 재직기간을 중복 계산하지 않기 위함.
    반환: (총 개월수, 실제로 겹치는 구간이 있었는지) - 후자는
    confidence="ambiguous" 판정에 쓰인다."""
    if not intervals:
        return 0, False
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    had_overlap = False
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            had_overlap = True
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    total = sum(max(0, e - s) for s, e in merged)
    return total, had_overlap


def _bucket(months: int) -> str:
    if months <= 0:
        return "신입"
    if months <= 36:
        return "1~3년"
    if months <= 60:
        return "3~5년"
    return "5년+"


def _format_period_text(e: dict) -> str:
    end = "현재" if e["is_present"] else f"{e['ey']}.{e['em']:02d}"
    return f"{e['sy']}.{e['sm']:02d} ~ {end}"


def _classify_relatedness(target_job_family: JobFamily, context_text: str) -> dict:
    """target_job_family.classify()의 반환 타입에 resume_career.py가
    강하게 의존하지 않도록 감싸는 어댑터. 지금은 str|None을 받아
    {"matched": bool, "label": str|None}로 정규화한다. 나중에
    classify()가 {"matched":bool, "category":str, "matched_keywords":
    [...]} 같은 구조화된 객체를 반환하도록 바뀌어도, 이 함수만
    고치면 되고 아래 extract_user_career_level()의 나머지 로직은
    수정할 필요가 없다."""
    result = target_job_family.classify(context_text)
    if isinstance(result, dict):
        return {"matched": bool(result.get("matched")), "label": result.get("category") or result.get("role")}
    return {"matched": result is not None, "label": result}


def extract_user_career_level(resume_raw: str, target_job_family: JobFamily) -> dict:
    """target_job_family와 관련된 실무 경력만 합산해 연차를
    판단한다. 관련 실무 경력이 하나도 없으면 무조건 "신입"으로
    분류한다(총 사회 경력과 무관)."""
    section = _find_career_section(resume_raw)
    if not section.strip():
        return {
            "career_level": "신입",
            "related_experience_months": 0,
            "confidence": "none",
            "reason": "경력 섹션을 찾지 못했습니다",
            "experiences": [],
        }

    entries = _extract_entries(section)
    if not entries:
        return {
            "career_level": "신입",
            "related_experience_months": 0,
            "confidence": "none",
            "reason": "경력 섹션은 있으나 재직 기간 정보를 추출하지 못했습니다",
            "experiences": [],
        }

    included_intervals: list[tuple[int, int]] = []
    experiences: list[dict] = []
    has_explicit_signal = False

    for e in entries:
        role, company = _guess_role_company(e["context"])
        is_work = _is_work_experience(e["context"])
        # "실무경력 여부"를 먼저 판단하고, 실무경력일 때만 Job Family
        # 관련 여부를 본다(순서가 중요 - 실무경력이 아니면 애초에
        # classify()를 호출할 이유가 없다). classify()에는 회사명을
        # 제외한 직무명+실제 업무내용만 넘긴다(_classification_text
        # 참고 - 회사명이 우연히 키워드와 겹치는 오분류 방지).
        classification = (
            _classify_relatedness(target_job_family, _classification_text(e["context"], role))
            if is_work else {"matched": False, "label": None}
        )
        is_related = classification["matched"]
        label = classification["label"]
        included = is_work and is_related
        months = max(0, e["end_abs"] - e["start_abs"])

        if included:
            included_intervals.append((e["start_abs"], e["end_abs"]))
            decision_reason = f"{target_job_family.display_name} Job Family 관련 실무경력"
            if _has_explicit_related_year(e["context"]):
                has_explicit_signal = True
        elif not is_work:
            decision_reason = "프로젝트/부트캠프/공모전 등 실무경력이 아닌 항목으로 판단되어 제외"
        else:
            decision_reason = f"지원 Job Family({target_job_family.display_name})와 관련 없는 실무경력"

        experiences.append({
            "role": label or role,
            "company": company,
            "start_date": f"{e['sy']:04d}-{e['sm']:02d}",
            "end_date": f"{e['ey']:04d}-{e['em']:02d}",
            "period_text": _format_period_text(e),
            "months": months,  # 이 경력 하나의 개월수 - 최종 related_experience_months(병합 후 합산)와는 다르다
            "is_current": e["is_present"],
            "is_work_experience": is_work,
            "is_related": is_related,
            "included": included,
            "decision_reason": decision_reason,
        })

    matched_months, had_overlap = _merge_and_sum_months(included_intervals)
    level = _bucket(matched_months)

    # confidence는 판단 "결과"가 아니라 판단 "근거의 신뢰도"를 뜻한다.
    # 겹치는 구간이 실제로 있었으면(병합이 발생했으면) 원본 표기가
    # 깔끔하지 않다는 뜻이라 "ambiguous" - 사용자 확인을 유도한다.
    # 그렇지 않고 "관련 있다고 이미 판정된" 경력 항목 자체에 명시적
    # 연차 언급이 있으면 "explicit", 둘 다 아니면(가장 흔한 경우,
    # 관련 경력 0건이라 "신입"인 것도 포함) "calculated"다.
    if had_overlap:
        confidence = "ambiguous"
    elif has_explicit_signal:
        confidence = "explicit"
    else:
        confidence = "calculated"

    if included_intervals:
        y, m = divmod(matched_months, 12)
        parts = (f"{y}년" if y else "") + (f" {m}개월" if m else "")
        reason = f"관련 직무({target_job_family.display_name}) 실무 경력 {parts.strip()}"
    else:
        reason = (
            f"경력 섹션은 있으나 지원 Job Family({target_job_family.display_name})와 "
            "관련된 실무 경력을 찾지 못해 신입으로 분류했습니다"
        )

    return {
        "career_level": level,
        "related_experience_months": matched_months,
        "confidence": confidence,
        "reason": reason,
        "experiences": experiences,
    }
