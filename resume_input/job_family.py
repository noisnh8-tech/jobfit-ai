"""
resume_input/job_family.py

Job Family 정의 - "어떤 실무 경력이 이 직무군과 관련 있는가"에 대한
지식을 resume_career.py(연차 계산 엔진)에서 분리해서 여기에 둔다.
resume_career.py는 Job Family를 모른다 - 호출부가 JobFamily 객체를
넘겨줘야만 관련 경력을 판단할 수 있다. 나중에 Backend/Frontend/
Data Engineer/PM 등 다른 Job Family를 추가할 때 resume_career.py를
고칠 필요가 없고, 이 파일에 새 JobFamily만 추가하면 된다.

관련 여부 판단은 지금은 키워드 매칭 휴리스틱(JobFamily.classify)
이지만, 이 메서드 뒤로 판단 로직이 감춰져 있으므로 나중에 더 정교한
판단(임베딩 유사도, 별도 Job Family Definition 문서 기반 등)으로
바꿔도 resume_career.py는 전혀 건드릴 필요가 없다. 특히
서비스기획/사업기획/PM/전략기획처럼 키워드만으로는 오판 가능성이
있는 애매한 직무는, 지금은 키워드 목록에 넣지 않고 보수적으로
"관련 없음"으로 분류되게 둔다 - 나중에 이 부분만 더 정교한 판단
로직으로 교체하면 된다(resume_career.py 수정 불필요).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobFamily:
    name: str
    display_name: str
    keywords: dict[str, str]  # lowercase 키워드 -> 표시용 라벨

    def classify(self, context_text: str) -> str | None:
        """context_text 안에 이 Job Family의 키워드가 있으면 표시용
        라벨을, 없으면 None을 반환한다. 지금은 단순 키워드 매칭이지만
        이 메서드 시그니처만 유지되면 내부 구현은 자유롭게 교체 가능."""
        low = context_text.lower()
        for kw, label in self.keywords.items():
            if kw in low:
                return label
        return None


# 이번 프로젝트가 Data Analytics 직무 지원자를 대상으로 한다는 전제를
# 반영한 첫 Job Family. 실제 이력서 표기 다양성은 실측하며 넓혀간다.
DATA_ANALYTICS = JobFamily(
    name="data_analytics",
    display_name="Data Analytics",
    keywords={
        "데이터 분석": "데이터 분석가",
        "데이터분석": "데이터 분석가",
        "데이터 애널리스트": "데이터 분석가",
        "data analyst": "Data Analyst",
        "data analytics": "Data Analyst",
        "데이터 사이언티스트": "Data Scientist",
        "data scientist": "Data Scientist",
        "bi 분석": "BI Analyst",
        "bi analyst": "BI Analyst",
        "business intelligence": "BI Analyst",
        "프로덕트 분석": "Product Analyst",
        "product analytics": "Product Analyst",
        "product analyst": "Product Analyst",
        "crm 분석": "CRM Analyst",
        "crm analytics": "CRM Analyst",
        "마케팅 분석": "Marketing Analyst",
        "marketing analytics": "Marketing Analyst",
        "그로스": "Growth Analyst",
        "growth analytics": "Growth Analyst",
        "growth marketing": "Growth Analyst",
        "비즈니스 전략": "Business Strategy",
        "business strategy": "Business Strategy",
        "오퍼레이션 분석": "Operations Analyst",
        "operations analytics": "Operations Analyst",
        "operation analytics": "Operations Analyst",
    },
)

_REGISTRY: dict[str, JobFamily] = {
    "data_analytics": DATA_ANALYTICS,
}


def get_job_family(name: str) -> JobFamily:
    return _REGISTRY[name]
