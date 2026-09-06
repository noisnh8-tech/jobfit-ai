# -*- coding: utf-8 -*-
"""
resume_input/resume_facts.py

Eligibility Facts(2026-08-14 신설, 사용자 확정) - "무엇을 해봤는가"를
묻는 Semantic Understanding(problem/thinking/task/skill/qualification)
과 별개로, "객관적으로 어떤 조건을 가지고 있는가"만 담는 사실값 계층.
LLM 호출 없음 - 전부 정규식/휴리스틱(resume_career.py와 같은 방식).

만든 이유: 특정 교육 플랫폼/특정 헬스케어 공고 실측 검증에서 "석사 이상 학위" JD
요구가 Resume의 "부트캠프 수료"와 Semantic Linking에 의해 partial_match로
잘못 연결되는 걸 확인했다(사회복지과 전문학사가 원본 이력서엔 있는데
Resume Understanding 14개 semantic_objects에서 통째로 누락돼 있었음).
학력/자격증처럼 "충족 여부"만 있고 "의미적 유사도"가 없는 객관적 조건은
Semantic Linking이 아니라 이 모듈의 factual value로 비교해야 한다
(judge_engine.py의 Hard Eligibility Gate가 사용 - eligibility_compare.py
참고).

핵심 원칙: "정보 없음"과 "실제 미충족"을 구분한다. 섹션 자체를 못
찾으면 confidence="none"으로 두고, 절대 기본값(0개월/특정 학력 등)으로
채우지 않는다 - resume_career.py의 D/E 원칙과 동일하다.

raw_text 필드는 판정 근거 텍스트로 쓰거나 UI에 노출하지 않는다(TODO -
표 형태 PDF 레이아웃에서 섹션 경계를 놓쳐 문서 뒷부분까지 통째로 캡처될
수 있음이 실측 확인됨 - 김수빈 샘플 이력서. 최종 factual value 자체는
서로 다른 포맷 2건에서 정확했지만 raw_text 경계 파싱은 아직 못 믿는다.
사용자 노출이 필요해지면 그때 별도로 개선한다)."""
from __future__ import annotations

import re

_BULLET_RE = re.compile(r"^[•\-\*\s]+")

# ── 학력(education) ──────────────────────────────────────────────

_EDU_HDR = re.compile(r"^(학력(\s*사항)?|education)\s*[:：]?\s*", re.I)
_NEXT_HDR = re.compile(
    r"^(학력(\s*사항)?|자격증?|자격\s*/?\s*어학\s*/?\s*수상|어학|수상(\s*내역)?|경력|프로젝트|경험(\s*사항)?|활동|"
    r"포트폴리오|자기\s*소개(서)?|certificat(e|ion)s?|award(s)?|career|experience|project(s)?)"
    r"\s*[:：]?\s*$",
    re.I,
)

# 학위 등급 - 계층 순서(높은 것부터)로 첫 매치를 채택. "학사" 패턴은
# "전문학사" 안의 "학사" 부분 문자열에 오매치하지 않도록 부정 전방탐색을
# 건다(실측 - "사회복지과 전문학사" 같은 표기가 "학사"로 오인식됨).
_DEGREE_PATTERNS = [
    ("박사", re.compile(r"박사|ph\.?d")),
    ("석사", re.compile(r"석사|master")),
    ("학사", re.compile(r"(?<!전문)학사|대학교\s*\(?4\s*년\)?|4년제\s*대학|bachelor")),
    ("전문학사", re.compile(r"전문학사|전문대|대학\s*\(?[23]\s*년\)?|초대졸")),
    ("고졸", re.compile(r"고등학교\s*졸업|고졸")),
]
_GRADUATED_RE = re.compile(r"졸업(?!\s*예정)")
_IN_PROGRESS_RE = re.compile(r"재학|휴학|졸업\s*예정|수료\s*예정")

DEGREE_RANK = {"고졸": 0, "전문학사": 1, "학사": 2, "석사": 3, "박사": 4}


def _find_section(raw: str, hdr_pattern: re.Pattern) -> str | None:
    """헤더 줄을 찾으면, 그 줄에서 헤더 뒤에 바로 붙은 값("학력 대학교(4년)
    졸업")과 그 아래 줄들(다음 헤더 전까지)을 합쳐서 반환한다. 헤더 자체가
    없으면 None(정보 없음) - 절대 빈 문자열이나 기본값으로 대체하지 않는다."""
    lines = raw.splitlines()
    start = None
    same_line_tail = ""
    for i, line in enumerate(lines):
        stripped = _BULLET_RE.sub("", line.strip())
        m = hdr_pattern.match(stripped)
        if m:
            start = i + 1
            same_line_tail = stripped[m.end():].strip()
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start, len(lines)):
        if _NEXT_HDR.match(_BULLET_RE.sub("", lines[i].strip())):
            end = i
            break
    body = "\n".join(lines[start:end]).strip()
    return (same_line_tail + "\n" + body).strip() if body else same_line_tail


def extract_education(resume_raw: str) -> dict:
    """섹션을 못 찾으면 highest_degree=None, confidence="none"(정보
    없음). 찾았는데 등급 키워드를 하나도 못 찾아도 confidence="none"
    (값을 임의로 채우지 않음)."""
    section = _find_section(resume_raw, _EDU_HDR)
    if section is None:
        return {"highest_degree": None, "graduated": None, "confidence": "none",
                "reason": "학력 섹션을 찾지 못했습니다", "raw_text": None}

    found = None
    for label, pat in _DEGREE_PATTERNS:
        if pat.search(section):
            found = label
            break
    if found is None:
        return {"highest_degree": None, "graduated": None, "confidence": "none",
                "reason": "학력 섹션은 있으나 학위 등급을 인식하지 못했습니다",
                "raw_text": section}

    graduated = None
    if _IN_PROGRESS_RE.search(section):
        graduated = False
    elif _GRADUATED_RE.search(section):
        graduated = True

    return {"highest_degree": found, "graduated": graduated, "confidence": "calculated",
            "reason": f"학력 섹션에서 '{found}' 키워드를 확인했습니다", "raw_text": section}


# ── 자격증(certification) ────────────────────────────────────────

_CERT_HDR = re.compile(r"^(자격증?|자격\s*/?\s*어학\s*/?\s*수상|certificat(e|ion)s?)\s*[:：]?\s*", re.I)

# 흔한 자격증 이름 - 화이트리스트 방식(오분류 위험이 큰 "경진대회/수상"
# 텍스트를 자격증으로 잘못 뽑지 않기 위해, 알려진 자격증명만 매칭한다).
# 목록 밖의 자격증은 놓칠 수 있음(재현율 한계 - 확인 안 되면 confidence
# ="none"이 아니라 "calculated"에 certifications=[]로 남아 "그 자격증은
# 없다"는 뜻이 된다는 점에 주의. 화이트리스트 확장은 필요시 추가한다).
KNOWN_CERTS_RE = re.compile(
    r"ADsP|ADP|SQLD|SQLP|정보처리기사|정보처리산업기사|컴퓨터활용능력\s*[12]급|"
    r"TOEIC|OPIc|TOEFL|IELTS|JLPT|JPT|HSK|"
    r"빅데이터분석기사|사회조사분석사|리눅스마스터",
    re.I,
)


def extract_certifications(resume_raw: str) -> dict:
    section = _find_section(resume_raw, _CERT_HDR)
    if section is None:
        return {"certifications": [], "confidence": "none",
                "reason": "자격증 섹션을 찾지 못했습니다", "raw_text": None}

    hits = sorted(set(m.group(0) for m in KNOWN_CERTS_RE.finditer(section)))
    if not hits:
        return {"certifications": [], "confidence": "none",
                "reason": "자격증 섹션은 있으나 알려진 자격증명을 인식하지 못했습니다(화이트리스트 밖일 수 있음)",
                "raw_text": section}
    return {"certifications": hits, "confidence": "calculated",
            "reason": f"{len(hits)}건 인식", "raw_text": section}


def extract_resume_facts(resume_raw: str, career_result: dict) -> dict:
    """judge_engine.judge()에 넘길 Eligibility Facts 묶음. career는 이미
    계산된 resume_career.extract_user_career_level() 결과를 그대로
    받는다(중복 계산하지 않음) - 호출부가 이미 갖고 있는 값이다."""
    return {
        "career": career_result,
        "education": extract_education(resume_raw),
        "certification": extract_certifications(resume_raw),
    }
