# -*- coding: utf-8 -*-
"""
resume_input/eligibility_compare.py

Hard Eligibility(경력연차/학위/자격증/어학/물리행정) 항목을 Semantic
Linking이 아니라 resume_facts.py의 factual value와 deterministic하게
비교한다(2026-08-14 신설, 사용자 확정). LLM 호출 없음, 새 점수/임계값
없음.

왜 Semantic Linking을 우회하는가: 실측(특정 헬스케어 공고)에서 "석사 이상
학위" ↔ "부트캠프 수료"가 의미적 유사성("교육을 받았다")만으로
partial_match가 되는 걸 확인했다 - 학력/경력/자격증 같은 객관적 조건은
"의미가 비슷한가"가 아니라 "조건을 충족하는가"를 봐야 하는 다른 종류의
판단이다(사용자 확정).

3-state 규칙(반드시 지킬 것 - 이게 이 파일의 핵심이다):
    충족   -> Hard Eligibility Gate 통과, 기존 구조적 gap 판단으로 진행
    미충족 -> 비추천(judge_engine이 결정)
    확인불가 -> "지원"으로 자동 확정되지 않는다. judge_engine이 "보류
        (필수 조건 확인 필요)"로 처리한다 - 정보가 없다는 사실을 실제
        미충족이라는 사실로 바꾸지 않는다(absence of evidence ≠
        evidence of absence, 이 프로젝트가 D/E에서 지켜온 것과 같은 원칙).

"A 또는 B" 복합 조건 안전장치: 지금 로직은 한쪽 조건만 검사한다(완전한
AND/OR parser는 이번 범위 밖). 그 결과가 "미충족"이면 확정하지 않고
"확인불가"로 낮춘다 - false rejection(다른 쪽 조건은 충족했을 수도
있는데 비추천 처리) 방지가 목적. "충족"은 OR 중 하나만 맞아도 진짜
충족이므로 그대로 인정한다.
"""
from __future__ import annotations

import re

from resume_input.resume_facts import DEGREE_RANK, KNOWN_CERTS_RE, _DEGREE_PATTERNS

_LANGUAGE_RE = re.compile(r"TOEIC|OPIc|TOEFL|IELTS|JLPT|JPT|HSK", re.I)
_EXPERIENCE_YEAR_RE = re.compile(r"(\d+)\s*년")
_LOGISTIC_RE = re.compile(
    r"출장|운전|결격사유|병역|국적|(?<!소)비자|근무지|교대\s*근무|신원조회|채용\s*결격|현장\s*근무"
)
_OR_MARKER_RE = re.compile(r"또는|혹은")


def _parse_jd_degree(jd_text: str) -> str | None:
    for label, pat in _DEGREE_PATTERNS:
        if pat.search(jd_text):
            return label
    return None


def _parse_jd_experience_months(jd_text: str) -> int | None:
    """"2년 이상"/"1~3년"처럼 여러 숫자가 나오면 가장 작은 값(하한선)을
    최소 요구치로 본다."""
    nums = [int(m.group(1)) for m in _EXPERIENCE_YEAR_RE.finditer(jd_text)]
    return min(nums) * 12 if nums else None


def _parse_jd_certification(jd_text: str) -> str | None:
    m = KNOWN_CERTS_RE.search(jd_text)
    return m.group(0) if m else None


def classify_hard_eligibility_type(jd_text: str) -> str | None:
    """language > certification > degree > experience > logistic 순서로
    확인한다 - "4년제 대학교"의 "4년"이 experience 패턴에도 걸리므로
    degree를 먼저 확인해야 오분류가 안 난다."""
    if _LANGUAGE_RE.search(jd_text):
        return "language"
    if KNOWN_CERTS_RE.search(jd_text):
        return "certification"
    if _parse_jd_degree(jd_text) is not None:
        return "degree"
    if _EXPERIENCE_YEAR_RE.search(jd_text):
        return "experience"
    if _LOGISTIC_RE.search(jd_text):
        return "logistic"
    return None


def _raw_compare(jd_text: str, resume_facts: dict) -> dict:
    etype = classify_hard_eligibility_type(jd_text)

    if etype == "degree":
        required = _parse_jd_degree(jd_text)
        edu = resume_facts.get("education") or {}
        if edu.get("confidence") == "none" or edu.get("highest_degree") is None:
            return {"status": "확인불가", "type": etype,
                    "reason": "이력서에서 학력 정보를 찾지 못했습니다"}
        have_rank = DEGREE_RANK.get(edu["highest_degree"], -1)
        need_rank = DEGREE_RANK.get(required, 99)
        status = "충족" if have_rank >= need_rank else "미충족"
        return {"status": status, "type": etype,
                "reason": f"요구={required}, 보유={edu['highest_degree']}"}

    if etype == "experience":
        required_months = _parse_jd_experience_months(jd_text)
        career = resume_facts.get("career") or {}
        if career.get("confidence") == "none":
            return {"status": "확인불가", "type": etype,
                    "reason": "이력서에서 경력 정보를 찾지 못했습니다"}
        have_months = career.get("related_experience_months", 0)
        status = "충족" if have_months >= required_months else "미충족"
        return {"status": status, "type": etype,
                "reason": f"요구={required_months}개월 이상, 보유={have_months}개월"}

    if etype == "certification":
        required_cert = _parse_jd_certification(jd_text)
        cert = resume_facts.get("certification") or {}
        if cert.get("confidence") == "none":
            return {"status": "확인불가", "type": etype,
                    "reason": "이력서에서 자격증 정보를 찾지 못했습니다"}
        have = [c.lower() for c in cert.get("certifications", [])]
        status = "충족" if required_cert and required_cert.lower() in have else "미충족"
        return {"status": status, "type": etype,
                "reason": f"요구={required_cert}, 보유={cert.get('certifications')}"}

    # language / logistic / 분류 안 됨 - 전부 factual extractor가 없음
    return {"status": "확인불가", "type": etype,
            "reason": "이 유형은 아직 factual 추출기가 없습니다(unknown 처리)"}


def compare(jd_text: str, resume_facts: dict) -> dict:
    """resume_facts는 resume_facts.extract_resume_facts()의 반환값.
    반환: {"status": "충족"|"미충족"|"확인불가", "type": str|None, "reason": str}"""
    result = _raw_compare(jd_text, resume_facts)
    if result["status"] == "미충족" and _OR_MARKER_RE.search(jd_text):
        return {"status": "확인불가", "type": result["type"],
                "reason": f"복합(또는) 조건이라 안전하게 판정할 수 없습니다 - 단일 조건 판정: {result['reason']}"}
    return result
