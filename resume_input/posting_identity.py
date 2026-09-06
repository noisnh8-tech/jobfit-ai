"""
resume_input/posting_identity.py

"같은 채용공고인가?"를 판정하는 안정적인 키. 순수 문자열 유틸 -
resume_input의 다른 모듈을 import 하지 않는다(순환 import 방지).

배경: 같은 자리를 가리키는 공고가 서로 다른 URL(→ 서로 다른 job_id
"url::...")로 여러 건 저장돼 있다(실측 2026-09-01: candidate_jobs 4,186행
중 165개 그룹이 회사+정규화제목이 같은데 job_id가 다름 - Greenhouse
gh_jid 재발행, 원티드↔ATS 크로스포스팅, 특정 이커머스 지역별 중복 등).

그래서 사용자가 "이 공고 제외" / "이 공고 지원함"을 한 번 하면,
그 job_id 하나만이 아니라 **같은 posting_key를 가진 모든 쌍**을
목록에서 빼야 한다. 안 그러면 _dedupe_same_posting()이 다음 수집 때
(본문 길이가 바뀌어) 다른 쪽 쌍을 남기면서 "제외한 공고가 되살아난다".
"""
from __future__ import annotations

import re

# 원티드 등은 회사 자체 채용페이지 공고를 "[회사명] 직무명"으로 다시
# 올린다 - 앞의 "[...]" 프리픽스를 떼야 크로스포스팅 중복이 매칭된다.
# (pipeline._normalize_title_for_dedup 과 동일 규칙 - 그쪽이 이 함수를
# import 해서 쓴다. 규칙을 바꿀 땐 두 곳이 아니라 여기 한 곳만 고친다.)
_BRACKET_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
_WS_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """앞의 "[...]" 프리픽스 제거 + 소문자 + **모든 공백 제거**.
    2026-09-02(사용자 신고 "제외한 공고가 계속 나온다") - 한국어 직무명은
    같은 자리인데도 소스마다 띄어쓰기가 제각각이라("데이터분석가" vs
    "데이터 분석가", "퍼포먼스/콘텐츠 마케터" vs "퍼포먼스 / 콘텐츠 마케터")
    공백을 남겨두면 twin 매칭이 새서 dismiss/지원 억제가 뚫린다. 회사명은
    그대로 두고(다른 회사가 합쳐지면 안 됨) 제목의 공백만 없앤다."""
    t = _BRACKET_PREFIX_RE.sub("", (title or "").strip().lower())
    return _WS_RE.sub("", t)


def posting_key(company: str, title: str) -> tuple[str, str]:
    """회사명 + 정규화 직무명. 같은 자리를 가리키는 공고면 URL/본문이
    달라도 이 키는 같다. 회사 또는 제목이 비면 ("", "") 계열의 무의미한
    키가 되어 오탐(정상 공고까지 제외)을 일으키므로, 호출부는 빈 키를
    억제 집합에 넣지 말 것(list_*_posting_keys 가 걸러낸다)."""
    return ((company or "").strip().lower(), normalize_title(title))
