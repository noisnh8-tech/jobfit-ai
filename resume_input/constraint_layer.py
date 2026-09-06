"""
resume_input/constraint_layer.py

Constraint Layer(2026-07-31 도입) - Ranking(M3+M4 코사인 유사도)이 못 잡는
"지원 가능 여부" 자격 조건을 combined_score에 직접 반영하는 확장 지점.

2026-07-31 Candidate Generation Top20 검증(docs/verification/2026-07-31_
candidate_generation_top100_eval/root_cause_table_top20.md)에서 확인된
핵심 원인: 경력연차/학위 같은 자격요건 문구가 Qualification 코사인
유사도에 흡수되면, "2년 이상 필수"가 오히려 "경력/역량" 관련 어휘와
가까워서 감점이 아니라 오히려 점수를 올리는 경우가 많았다. 이런 조건은
Similarity가 아니라 명시적 Constraint(제약)로 처리해야 한다.

`compute_constraint_penalty()`가 penalty 합산의 단일 진입점이다 - 새
제약 조건(학위/병역특례/비자/근무형태 등)이 추가될 때마다
candidate_search.py를 건드리지 않고 이 함수 안에 한 줄만 추가하면 된다
(예: `penalty += degree_penalty(job)`). candidate_search.py는 이 함수의
반환값(단일 float)만 combined_score에 더하면 되므로, 새 제약이 늘어나도
if-분기가 candidate_search.py 쪽에 누적되지 않는다.

첫 번째 규칙(Career)만 구현한다(2026-07-31) - 학위/병역특례/비자 등은
아직 텍스트 추출 로직 자체가 없어서 이후 별도로 추가한다. Career는
career_filter.compute_career_status()가 이미 계산해주던 신호(정규식
기반, LLM 미사용, 1985건 실측 검증)를 그대로 재사용한다 - 새로 만든
추출이 아니라 기존에 표시용으로만 쓰이던 신호를 점수 계산에 배선한
것뿐이다."""
from __future__ import annotations

from resume_input.career_filter import compute_career_status

# 2026-07-31 실험값(A/B 실험 docs/verification/2026-07-31_candidate_
# generation_top100_eval/ab_experiment_career_skill_m4.py의 Variant B와
# 동일한 값) - 아직 튜닝되지 않은 첫 시도값이다. combined_score가
# m3/m4 각각의 z-score를 0.5:0.5로 합친 값이라 대체로 -3~+3 범위에
# 분포하는 것을 참고해서 정했다. 별도 상수로 분리해둔 이유: 이후
# 라벨링 데이터가 늘어나면 이 값 자체를 재조정할 가능성이 높기
# 때문이다(호출부 코드는 그대로 두고 이 값만 바꾸면 됨).
#
# NOTE(2026-07-31, Top100 분포 검증에서 확인 - docs/verification/2026-07-31_
# candidate_generation_top100_eval/constraint_layer_top100_distribution.md):
# 신입 사용자(challenge_option=False) 기준으로는 "상향"/"크게상향"이
# 실전에서 절대 발생하지 않는다 - career_filter._BASE_INCLUDE["신입"]이
# 애초에 career_level을 {신입, 1~3년}까지만 pool에 들여보내기 때문에
# (Career Filter가 이미 3~5년/5년+ JD를 제거함), compute_career_status가
# 신입 사용자에게 낼 수 있는 결과는 사실상 "적합"/"약간상향"뿐이다.
# 즉 실사용 흐름에서는 -0.5만 발동하고, -1.0/-2.0은 challenge_option=True로
# 3~5년+ JD까지 pool에 들어오는 경우나, user_career_level이 "1~3년"
# 이상인 사용자에게만 의미가 생긴다("왜 -1.0이 안 나오지?"를 나중에 또
# 묻지 않기 위해 여기 남겨둠) - 죽은 코드가 아니라 다른 사용자
# 시나리오를 위해 미리 준비해둔 값이다.
CAREER_PENALTY = {
    "적합": 0.0,
    "확인필요": 0.0,
    "약간하향": 0.0,
    "하향": 0.0,
    "약간상향": -0.5,
    "상향": -1.0,
    "크게상향": -2.0,
}


def career_penalty(user_career_level: str | None, jd_level: str | None, challenge_option: bool = False) -> float:
    """career_status(적합/약간상향/상향/크게상향 등)를 CAREER_PENALTY로
    변환한다. career_status 자체의 판단 로직(연차 거리 계산)은 손대지
    않는다 - career_filter.py가 이미 검증된 방식으로 계산한 상태값만
    받아서 점수로 변환하는 역할만 한다. user_career_level이 None이면
    이 규칙 자체를 안 켠 것으로 보고 0.0을 반환한다(호출부가 매번
    None 체크를 하지 않도록, 이 판단은 규칙 함수 안에서 흡수한다)."""
    if user_career_level is None:
        return 0.0
    status = compute_career_status(user_career_level, jd_level, challenge_option)
    return CAREER_PENALTY.get(status, 0.0)


def compute_constraint_penalty(job: dict, user_career_level: str | None, challenge_option: bool = False) -> float:
    """이 job에 적용할 Constraint penalty 총합(현재는 Career 규칙 하나뿐).
    새 제약이 추가되면 이 함수 안에서만 누적한다 - 호출부
    (candidate_search.py)는 이 함수의 반환값만 combined_score에 더하면
    되고, 수정할 필요가 없다. 호출부는 user_career_level이 None인지
    아닌지 신경 쓸 필요 없이 항상 이 함수를 호출하면 된다(각 규칙이
    자기 입력이 없을 때 0을 반환하는 책임을 진다) - Constraint Layer가
    켜져 있는지 꺼져 있는지를 호출부의 if로 표현하지 않는다."""
    penalty = 0.0
    penalty += career_penalty(user_career_level, job.get("career_level"), challenge_option)
    return penalty


def compute_constraint_debug(job: dict, user_career_level: str | None, challenge_option: bool = False) -> dict:
    """규칙별 세부 내역(career_level/career_status/career_penalty) -
    compute_constraint_penalty()의 반환값(총합 float)만으로는 나중에
    "왜 이 penalty가 나왔지?"를 설명할 수 없어서 별도로 남겨두는 진단용
    정보다. 규칙이 늘어나면 이 dict에 키만 추가하면 된다."""
    if user_career_level is None:
        return {}
    status = compute_career_status(user_career_level, job.get("career_level"), challenge_option)
    return {
        "career_level": job.get("career_level"),
        "career_status": status,
        "career_penalty": CAREER_PENALTY.get(status, 0.0),
    }
