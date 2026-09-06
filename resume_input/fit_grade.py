"""
resume_input/fit_grade.py

combined_score(BM25+Embedding 결합 점수)를 사용자가 오인하기 쉬운
%(합격 확률처럼 보임) 대신 등급(A+/A/B+/B/C)으로 변환한다. 검색
결과 "내" 상대적 백분위로 계산한다 - 절대 점수 기준이 아니다.
Career Filter와 마찬가지로 순수 표시용 변환이지 새로운 점수 체계가
아니다(combined_score 자체는 건드리지 않는다).
"""
from __future__ import annotations

_GRADE_CUTOFFS = [
    (0.10, "A+"),
    (0.30, "A"),
    (0.60, "B+"),
    (0.90, "B"),
    (1.01, "C"),
]

# UI에 등급 의미를 설명할 때 쓴다 - _GRADE_CUTOFFS와 동일한 숫자를
# 그대로 문장으로 옮긴 것뿐이다(별도 판단 추가 없음, 두 곳에 숫자를
# 따로 유지하지 않기 위해 여기 한 곳에 둔다).
GRADE_EXPLANATION: dict[str, str] = {
    "A+": "이번 검색 결과 상위 10% 이내",
    "A": "이번 검색 결과 상위 30% 이내",
    "B+": "이번 검색 결과 상위 60% 이내",
    "B": "이번 검색 결과 상위 90% 이내",
    "C": "이번 검색 결과 하위 10%",
}


def compute_fit_grades(scored_jobs: list[dict]) -> None:
    """scored_jobs를 combined_score 내림차순이라고 가정하지 않는다 -
    이 함수 내부에서 순위를 다시 계산해 각 job에 in-place로
    job["fit_grade"]를 부여한다. 상위 10%=A+, 다음 20%=A, 다음
    30%=B+, 다음 30%=B, 나머지=C (경계값은 실측 후 조정 -
    ui_design.md §6 참고)."""
    n = len(scored_jobs)
    if n == 0:
        return

    ordered = sorted(scored_jobs, key=lambda j: -j.get("combined_score", 0.0))
    for rank, job in enumerate(ordered):
        percentile = (rank + 1) / n
        for cutoff, grade in _GRADE_CUTOFFS:
            if percentile <= cutoff:
                job["fit_grade"] = grade
                break
