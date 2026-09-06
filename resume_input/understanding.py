"""
resume_input/understanding.py

JD/Resume Understanding 생성 - LLM 1회 호출로 사람이 읽는 Context와
기계가 비교하는 Representation(short_semantic + semantic_graph)을
동시에 만든다.

검증 이력(2026-07-13, docs/verification/2026-07-13_understanding_engine_
meaning_matching/summary.md 참고 - 여기 적힌 설계 원칙은 전부 실측
근거가 있다. 다시 실험하지 않는다):

1. Context는 사람이 미리 정의한 스키마(Problem/Thinking/Persona 등)로
   강제 분류하지 않는다 - 무관한 공고도 억지로 채워서 판별력을 해친다
   (실측: JSON 스키마 판별력 +0.125 vs 자유문단 +0.170).
2. short_semantic은 딱 4개 필드로 고정한다(핵심역할/핵심문제/핵심기대/
   부적합한경우) - 필드명을 더 늘리거나 줄이지 않는다. 없으면 null이지
   억지로 채우지 않는다.
3. semantic_graph는 "A -> B -> C" 관계 체인 2~3개 - 이게 Meaning
   Matching(meaning_matching.py)의 M4(Graph) 입력이 된다.
4. Context/short_semantic/semantic_graph는 검색(Retrieval)에 쓰지
   않는다 - Context를 임베딩으로 비교하면 판별력이 붕괴한다(anisotropy,
   3개 임베딩 모델로 반증됨). 검색은 meaning_matching.py의 M3/M4가
   담당하고, Context는 Explainability(추천 이유/인재상/커스터마이징
   등)에만 쓴다.

JD Understanding은 JD 등록 시 1회만 생성하고 job_store에 캐시한다
(career_level과 동일한 원칙). Resume Understanding은 이력서가 바뀔 때만
재생성한다(resume_hash 기준).
"""
from __future__ import annotations

import hashlib
import json

from resume_input.llm_client import call_llm, LLMCallError


def _posting_hash(posting_text: str) -> str:
    """posting_text 내용의 해시(2026-07-16 추가) - ensure_jd_semantic_
    objects()가 재크롤링으로 내용이 바뀐 job을 감지하는 데 쓴다."""
    return hashlib.sha256((posting_text or "").encode("utf-8")).hexdigest()[:16]

# jd_representation 생성 프롬프트/방식이 바뀌면 올린다. job_store의
# representation_version 컬럼에 저장되어, 나중에 이 값이 바뀌면 "옛
# 버전으로 생성된 job들"을 재생성 대상으로 식별할 수 있다
# (job_store.list_jobs_needing_representation() 참고).
# Resume와 완전히 독립된 버전이다(2026-07-16, 사용자 명시적 요청) -
# Resume만 바뀌었는데 JD 전체가 재생성 대상으로 잡히거나, 반대로 JD만
# 바뀌었는데 Resume 캐시가 무효화되는 일이 없어야 한다. resume_
# understanding_cache와 candidate_jobs가 서로 다른 테이블/컬럼이라
# 독립 관리해도 안전하다(RESUME_REPRESENTATION_VERSION 참고).
#
# v3(2026-07-17, 사용자 확정) - skill/preferred 생성 규칙 추가 +
# skill.frame.domain 섹션 헤더 복사 금지 규칙 추가(_JD_PROMPT 참고).
# 이전엔 ensure_jd_semantic_objects()의 스킵 조건이 posting_hash만
# 봐서 이 버전을 올려도 실제로는 아무 job도 재생성 대상이 안 됐다 -
# 스킵 조건에 representation_version 비교를 추가했으므로(같은 파일,
# ensure_jd_semantic_objects 참고) 이제 이 버전을 올리면 실제로
# 재생성 대상이 잡힌다.
#
# v4(2026-07-18, 사용자 확정) - LLM-free 3종 감사(docs/verification/
# 2026-07-18_three_llm_free_audits/summary.md)에서 영문 Greenhouse
# 공고(특정 해외 테크기업들) 다수가 "Hands-on experience with SQL,
# Python..." 처럼 산문 문장으로 스킬을 서술하면 skill Object가 통째로
# 안 만들어지는 걸 확인했다 - 리스트형 "[기술]" 헤더가 있는 공고만
# 정상 추출되고 있었다. 산문 문장 안의 기술명도 리스트와 동일하게
# 취급하라는 규칙을 추가한다(_JD_PROMPT 참고).
#
# v4 1차 검증 실패(2026-07-18) - 해당 해외 테크기업 "Expression of Interest:
# Machine Learning Engineer"로 재생성했는데도 skill 0개 그대로였다.
# 원인은 프롬프트 규칙이 아니라 generate_jd_understanding()이
# posting_text를 3000자로 잘랐던 것 - 이 공고는 10000자짜리였고 SQL/
# Python 언급은 5456번째, BigQuery는 6164번째 글자에 있어서 LLM이
# 애초에 그 문장을 받지도 못했다(같은 파일 generate_jd_understanding
# 참고, 3000->8000자로 수정). 이 truncation을 고친 뒤 같은 공고로
# 재검증하니 skill 2개(Python/SQL, TensorFlow/PyTorch/JAX)가 정상
# 생성됐다 - truncation 문제와 산문형 skill 규칙 둘 다 실제로 필요한
# 수정이었음을 1건으로 함께 확인.
# v5(2026-07-21, Rule Executor 구현 - docs/architecture_rule_executor_
# v1.md 참고) - Object마다 `concepts` 필드 추가 + `resume_customization_
# targets`(project_priority/skill_priority/headline_focus/expression_
# focus/experience_focus/avoid_changes, 각 항목에 reason 포함) 추가.
# project_priority는 problem+task 레이어에서만, experience_focus는
# thinking 레이어에서만 뽑아 두 target이 같은 근거를 중복으로 쓰지
# 않게 한다(Rule1/Rule3 역할 분리, 오늘 실측으로 확정된 문제).
#
# v6(2026-07-21, 실사용 검증 중 발견/수정) - v5의 experience_focus
# "thinking 레이어만" 규칙이 실제 공고(특정 DB전문기업 Data Architect)에서
# thinking 레이어가 0개라 experience_focus가 영구히 빈 배열이 되는
# 문제를 확인(실측: jd_semantic_objects 8개가 전부 problem/task/
# qualification/skill뿐, thinking 0개 - 담당업무/자격요건이 전부
# "무엇을 하는가" 위주로 서술된 공고). experience_focus를 problem+
# thinking+task 3개 레이어로 확장 - project_priority(problem+task)
# 와 소스가 겹칠 수 있지만, 두 target의 소비 목적이 다르므로(프로젝트
# 단위 vs 불릿 단위) 감수한다(사용자 확정).
#
# v7(2026-07-21, 사용자 확정 - 단순 합집합에서 우선순위 폴백으로 변경)
# - v6은 problem+thinking+task를 그냥 다 합쳐서 project_priority와
# 어휘가 과도하게 겹쳤다. thinking → task → problem → qualification
# (실제 경험을 요구하는 것만, 단순 도구 보유는 제외) 순서로 하나라도
# 나오면 그 레이어에서 멈추는 우선순위 폴백으로 교체.
#
# v8(2026-07-21, 사용자 확정 - 원문 커버리지 규칙 추가) - 동일 공고를
# 2회 생성해서 비교한 결과(공고 1건, Resume/Rule Executor 미실행),
# Object 개수(2↔3)가 다른 게 문제가 아니라 실행마다 원문 불릿의 서로
# 다른 부분집합만 골라 담고 나머지는 이유 없이 통째로 누락됨을 확인
# (1회차: 담당업무 6/6 반영, 필수요건 3/7, 우대사항 3/5만 반영 / 2회차:
# 담당업무 3/6, 필수요건 2/7, 우대사항 1/5만 반영). 원인은 "레이어당
# 최대 4개, 가장 중요한 것만" 지시가 "어떤 불릿을 볼지/버릴지"를 LLM
# 주관적 판단에 전적으로 맡긴 것 - 병합은 허용하되 누락은 금지하는
# 원문 커버리지 규칙으로 교체(Object 개수를 고정하는 게 아니라 "같은
# 원문 요구사항이 매번 빠짐없이 살아남게" 하는 게 목적).
#
# v9(2026-07-21, 사용자 확정 - concepts를 "생성"에서 "추출"로 변경) -
# 원문 커버리지는 안정됐지만(v8) concepts 자체는 여전히 실행마다
# 달랐다(예: "데이터 표준사전 구축"이 "데이터 표준"/"기준 설계"/
# "데이터 거버넌스"로 제각각). 원인 확정: "표준 카테고리어를 우선
# 쓰라"는 지시 + 예시 목록(데이터 모델링/기준 설계/지표 설계 등)을
# LLM이 원문과 무관한 "선택 가능한 사전"처럼 취급해 실행마다 다른
# 예시 단어를 무작위로 가져다 씀("기준 설계"는 원문에 전혀 없는
# 단어인데 예시 목록에만 있었음). 예시 목록을 없애고, concepts를
# "원문에 실제로 있는 명사(또는 업계 동일 명칭)만 추출 - 새 카테고리
# 생성 금지"로 강제. Rule Executor가 concepts를 문자열로 정확히
# 비교하므로, "창의적 분류"가 아니라 "같은 입력에서 항상 같은 단어"
# 가 나오는 게 목표(사용자 확정).
#
# v10(2026-07-21, 사용자 확정 - concepts를 "명사 추출"에서 "명사구 Span
# 복사"로 재정의) - v9로 원문에 없는 단어(기준 설계 등)는 사라졌지만,
# "핵심 명사를 추출하라"는 지시만으로는 LLM이 "이 문장에서 명사구
# 경계를 어디까지로 볼지"를 매번 다르게 판단해서 같은 원문에 대해
# "데이터거버넌스"/"데이터거버넌스 구축"/"관리 경험"처럼 실행마다
# 다른 길이로 잘라내는 걸 확인함(2회 재생성 비교). "요약/추론/
# 패러프레이즈/일반화 금지, 원문의 연속된 명사구 span을 그대로 복사"
# 로 더 좁혔다. "레이어당 최대 4개"가 아니라 concepts 자체도 "정확히
# 2~4개, 5개 이상은 규칙 위반"으로 명시(1차 검증에서 LLM이 6개를
# 낸 사례 확인).
#
# v11(2026-07-21, 사용자 확정 - v10 폐기, v9 기반으로 롤백 + 분리
# 규칙 추가) - v10 실측 결과 concepts 안정성은 개선됐지만 원문
# 커버리지가 다시 깨짐을 확인(Object 15개→7개로 급감, 원문 18개 불릿
# 중 6개가 통째로 누락된 실행 발견). 원인: concepts에 너무 많은
# 역할(원문 커버·명사 추출·span 유지·개수 제한·재현성)을 동시에
# 요구해서, LLM이 concepts 제약을 맞추려고 Object 생성/병합 자체를
# 다시 바꿔버림 - Object 생성과 concepts 추출이 프롬프트 안에서
# 서로 영향을 주고 있었다(사용자 확정 진단). v9의 "원문 명사만
# 추출, 새 카테고리 생성 금지"는 유지하되, "concepts는 Object 확정
# 후 부가 정보일 뿐 Object 생성에 영향을 주면 안 된다"는 4가지
# 금지 규칙을 추가하고, 개수 제약을 "정확히 2~4개"에서 "1~4개
# (짧은 원문은 1개도 허용)"로 완화했다. **이게 concepts 프롬프트의
# 마지막 조정이다 - 이후로는 Understanding 프롬프트를 더 튜닝하지
# 않는다(사용자 확정, 실질 대비 시간 소모가 커지는 구간으로 판단).
# 남은 concepts 변동성은 알려진 한계로 기록하고 넘어간다.**
#
# v12(2026-08-01, 사용자 확정 - culture 레이어 정의 수정, concepts와
# 무관한 별도 변경) - 실제 저장된 v11 culture Object 117건(174개 공고
# 중 108건에서 추출)을 직접 대조한 결과, "culture(조직문화/인재상)"라는
# 기존 한 줄 설명이 실제 추출 범위보다 좁았다는 게 확인됐다. 실측
# 데이터에는 두 종류가 섞여 있었다: (a) 회사 고유 가치관/일하는 방식
# (특정 커머스 플랫폼 "One Team/GRIT", 특정 AI 기업 "수평적 문화", 특정 해외 브랜드
# "다양성/포용", 특정 해외 테크기업 "겸손한 리더십") (b) 지원자에게 요구하는
# 태도/소프트스킬(특정 이커머스 "커뮤니케이션·협업", 특정 애드테크 기업
# "꼼꼼함과 책임감"). 같은 회사가 여러
# 공고에서 동일 culture 문구를 반복하는 패턴(특정 스타트업 6건 "성장에 대한
# 집착", 특정 AI 스타트업 4건 "회복 탄력성", 특정 해외 브랜드 3건 "다양성/포용")이 이 레이어가
# 회사 고유 시그니처까지 담고 있다는 근거. LLM의 실제 추출 동작은
# 이미 올바르게 두 종류를 다 뽑고 있었으므로(behavior 변경 아님),
# 설명 문구만 실제 범위에 맞게 넓힌다 - "조직 문화·핵심가치·컬처핏"
# (_JD_PROMPT 참고). v11 comment의 "프롬프트를 더 튜닝하지 않는다"는
# concepts 필드 조정에 대한 결정이었고, 이번 변경은 레이어 정의 설명문
# 수정이라 별개 사안으로 진행한다(사용자 확정). 필드 구조/개수 제한/
# concepts 규칙은 전혀 안 바꿨다 - culture 레이어 한 줄 설명만 교체.
JD_REPRESENTATION_VERSION = "v12-culture-redefined-2026-08-01"

# Resume Understanding 전용 버전(2026-07-16, 체크포인트 1) -
# docs/semantic_object_schema.md의 Semantic Object 스키마로 전면 교체.
# JD는 아직 옛 스키마(JD_REPRESENTATION_VERSION)를 그대로 쓴다(체크포인트
# 2에서 별도로 교체) - resume_understanding_cache와 candidate_jobs는
# 서로 다른 테이블/컬럼이라 버전을 독립적으로 관리해도 안전하다.
#
# v3(2026-07-17, 사용자 확정) - 1:1 원문 대조 감사에서 Task Over-merge/
# Missing, Skill Missing(Streamlit/Tableau), Qualification Missing
# (2급 자격증), Thinking 프로젝트 편중을 확인했다(docs/verification/
# 2026-07-17_resume_understanding_1to1_audit/ 참고 - 이 커밋에는 아직
# 없으면 다음 검증에서 저장). 원인은 "레이어당 최대 4개, 대표적인
# 것만"이 5개 레이어에 무차별 적용된 것 - Task/Skill/Qualification/
# Thinking에 예외 규칙을 추가한다(_RESUME_PROMPT 참고). 버전을 올려서
# 기존 resume_understanding_cache가 자동으로 재생성 대상이 되게 한다
# (get_or_generate_resume_understanding의 캐시 체크가 이 값을 본다).
#
# v4(2026-07-17) - v3 검증 결과 Task는 여전히 2개(안 늘어남) - 재확인해
# 보니 원인은 "모델 vs 대시보드를 억지로 분리 안 해서"가 아니라, 원문
# "해결 과정 및 역할"의 하위 불릿 중 하나(Airbnb "의사결정 구조 설계"
# - 시장 유형 4개 분류)가 problem/thinking/task 어디에도 전혀 반영되지
# 않고 통째로 빠져 있었다(다른 불릿 1개는 problem+thinking에 중복
# 반영됨). "모델/대시보드를 나눠라"는 인위적 분리 지시 대신 "불릿
# 하나도 빠뜨리지 마라"는 완전성 지시로 교체한다.
#
# v5(2026-07-18, 사용자 확정) - LLM-free 3종 감사(docs/verification/
# 2026-07-18_three_llm_free_audits/summary.md)에서 v4가 "불릿 누락"은
# 고쳤지만 다른 세 문제를 재확인했다: (1) Task가 여전히 서로 다른
# 행위(데이터 정제 vs 지표 분해)를 한 Object로 합쳐서 "허브"가 되어
# Matching 단계에서 무관한 채용 요건들과 뭉뚱그려 매칭되는 원인이 됨,
# (2) 이력서 최상단 요약에 적힌 핵심 통찰(프로젝트 상세 섹션에는 다시
# 안 나오는)이 Thinking에서 통째로 빠짐, (3) evidence.source_text가
# 원문 인용이 아니라 여러 문장을 합친 재구성 요약으로 채워진 사례 발견.
# 사용자 판단(2026-07-18): "Task 개수"가 아니라 "Task Granularity(의미
# 단위 분리)"가 핵심 - Matching이 틀린 게 아니라 Understanding이 만든
# Object가 너무 넓다는 진단. 세 가지 모두 프롬프트 규칙으로 대응한다
# (_RESUME_PROMPT 참고).
#
# v5 실측 결과(2026-07-18) - Thinking 누락은 해결됐지만 동시에 Skill이
# 4개->3개(Scikit-learn/SHAP/Pandas 완전 삭제), Task 관련 산출물(Airbnb
# 프로젝트의 실제 대시보드 기획)도 삭제되고 그 자리를 교육과정 커리큘럼의
# 사소한 언급이 대신 차지하는 회귀가 발견됐다(Object 총 15개->12개).
# v4/v5 Semantic Object를 필드 단위로 직접 대조한 결과(LLM 재호출 없이),
# Scikit-learn/SHAP/Pandas는 Task.method로 "이동"한 게 아니라 아예 Task/
# Skill/Thinking 어디에도 없는 순수 삭제였다 - 사용자 판단(2026-07-18):
# "교육과정 문장 하나"로는 이 전체 손실 패턴(Skill 압축 + Task 압축이
# 동시에 일어난 것)을 설명 못 하고, 새로 추가한 "task는 의미 단위로
# 나누세요" 규칙 블록(12줄) 자체가 모델로 하여금 Skill까지 통째로
# Task.method 안으로 압축하게 유도했을 가능성이 더 크다는 가설을 세웠다.
#
# v6(2026-07-18, 최종 확정/Freeze) - 위 가설을 단일 변수로 격리
# 검증하기 위해 Task 세분화 규칙 블록만 제거하고(Thinking 최상단요약
# 규칙/source_text 원문인용 규칙은 유지) 1회만 재생성했다. 결과: Object
# 15개로 복원, Skill 4개로 복원(Streamlit/Tableau 회복), Task 4개로
# 복원, Thinking은 Airbnb/Reward 둘 다 정확히 반영된 채 유지 - 가설이
# 맞았다(Task 세분화 규칙이 범인). 단, Scikit-learn/SHAP는 이 규칙을
# 빼고 재생성해도 여전히 없었다 - 즉 이 둘의 실종은 Task 규칙과
# 무관한 별개 원인으로 반증됐다(원인 미확인). 사용자 판단(2026-07-18):
# 여기서 추가로 파고들면 "고치고-다시깨지고"를 반복할 위험이 매출이 큰
# 구간에 들어섰고, Scikit-learn 1개가 추천 결과에 주는 영향은 SQL/
# Python/Task/Thinking에 비해 미미하다 - 이 버전으로 Freeze하고
# Scikit-learn/SHAP 누락은 Known Issue로 남긴다(docs/KNOWN_ISSUES.md
# 참고, 후속 버전에서 Rule 기반 Skill 후처리 보강으로 다룰 후보).
# 이 이후로는 이 프롬프트를 추가 수정하지 않는다.
#
# (v7 "persona 레이어 추가" 2026-07-21 - JD Culture를 Resume Persona로
# 증명하는 인재상 자동 반영 실험. 실데이터 검증 결과 matching_anchor가
# concept 중복 병합용으로 설계된 필드라 22건 전수 0건 연결 - "인재상
# 판단을 위해 만든 게 아니라 있는 걸 공짜로 빌려 쓴 것"이라는 게
# 드러남. 사용자 판단(2026-07-21): 실질 가치 없이 코드/검증만
# 복잡해지므로 인재상 자동 반영 자체를 제거 - persona 레이어도 그
# 목적으로만 넣었던 것이라 함께 되돌린다. v8로 다시 5-레이어 스키마로
# 복귀. Culture는 JD Understanding에서 계속 추출하고 화면에 표시만
# 한다(job_detail.py) - 자동 판단은 이후 별도 기능으로 재검토.)
#
# v9(2026-07-21, 사용자 확정 - persona_tags) - v7과 다른 접근. 새 레이어를
# 만들지 않고, 기존 5개 레이어 Object 각각에 짧은 태그 속성(예: "주도성",
# "협업")만 추가한다. v7/matching_anchor 실험은 JD culture와 Resume
# 객체의 "문장 전체"를 비교해서(어휘가 겹치기만 해도 걸리는) 오탐이
# 심했다(100건 실측: 오탐 80%+) - persona_tags는 짧은 통제된 라벨이라
# 문자열 포함 검사(용어 치환③과 동일한 안전한 방식, 새 유사도 계산
# 아님)만으로 비교해도 오탐이 덜할 것이라는 가설을 검증하기 위함.
#
# v10(2026-07-21, Rule Executor 구현 - docs/architecture_rule_
# executor_v1.md 참고) - Customization Planner(LLM 판단)를 완전히
# 제거하고 Rule Executor(LLM 0회, concepts 교집합 개수만 계산)로
# 대체하기 위해 Object마다 `concepts`(짧고 재사용 가능한 카테고리어
# 2~4개) 필드를 추가한다. persona_tags와 별개 필드 - persona_tags는
# 인재상(태도/스타일) 전용, concepts는 Rule1(프로젝트)/Rule3(경험)이
# 쓰는 범용 카테고리 태그다. 오늘 실측(docs/verification/2026-07-21_
# planner_ab_test) - "짧고 재사용 가능한 카테고리어"라는 스타일
# 지시를 안 지키면 JD/Resume 양쪽 concepts가 우연히도 안 겹쳐서
# 교집합이 0건 나온다(B안 v1) - 지키면 같은 표본에서 14건 나온다
# (B안 v2). 이 차이가 Rule Executor의 유일한 성패 요인이므로 스타일
# 지시를 정확히 지킨다.
#
# v11(2026-07-21, 사용자 확정 - concepts를 "생성"에서 "추출"로 변경) -
# JD 쪽(JD_REPRESENTATION_VERSION v9)과 동일한 원인·동일한 수정.
# "표준 카테고리어 예시 목록"을 LLM이 원문과 무관한 선택지처럼 써서
# 실행마다 다른 단어가 나오던 문제를 "원문에 실제로 있는 명사만 추출,
# 새 카테고리 생성 금지"로 교체. JD/Resume 양쪽이 같은 규칙을 써야
# concepts 교집합이 의미 있으므로 동시에 올린다.
#
# v12(2026-07-21, 사용자 확정 - concepts를 "명사 추출"에서 "명사구
# Span 복사"로 재정의) - JD 쪽(JD_REPRESENTATION_VERSION v10)과 동일한
# 원인·동일한 수정. 명사구 경계 판단의 실행 간 불일치를 줄이기 위해
# "원문의 연속된 명사구 span을 그대로 복사, 요약/추론/패러프레이즈/
# 일반화 금지"로 좁히고, concepts 개수도 "정확히 2~4개, 5개 이상 금지"
# 로 명시. JD/Resume 양쪽이 같은 규칙을 써야 하므로 동시에 올린다.
#
# v13(2026-07-21, 사용자 확정 - v12 폐기, v11 기반으로 롤백 + 분리
# 규칙 추가) - JD 쪽(JD_REPRESENTATION_VERSION v11)과 동일한 원인·
# 동일한 수정. concepts가 Object 생성/병합에 영향을 주지 않도록
# 4가지 금지 규칙을 추가하고 개수를 1~4개로 완화. **concepts
# 프롬프트의 마지막 조정 - 이후로는 Understanding 프롬프트를 더
# 튜닝하지 않는다(사용자 확정).**
RESUME_REPRESENTATION_VERSION = "v13-concepts-final-2026-07-21"

_JD_PROMPT = """아래 채용공고를 읽고 아래 스키마로 분석하세요.
(docs/semantic_object_schema.md에 확정된 Semantic Object 스키마입니다 -
필드를 추가/삭제/변경하지 마세요.)

레이어는 problem(이 역할이 해결하는 문제) / thinking(요구되는 사고방식) /
task(실제 업무) / skill(요구 기술/도구) / qualification(자격요건) /
culture(조직 문화·핵심가치·컬처핏 - 회사 고유의 가치관/일하는 방식과,
그 문화에 맞는 지원자의 태도·소프트스킬을 모두 포함) 6개입니다.
**Object 개수는 고정하지 않습니다.**
직무 관련 원문 불릿을 빠짐없이 커버하되, 의미가 같은 불릿만 병합하여
불필요한 중복 Object 생성을 줄이세요. 원문에 해당 레이어의 근거가
없을 때만 그 레이어를 빈 배열로 두세요.

**[원문 커버리지 규칙 - 2026-07-21, 실측 근거: 동일 공고를 2회
생성했더니 담당업무/필수자격요건/우대사항 불릿이 실행마다 서로 다른
부분집합만 반영되고 나머지는 이유 없이 통째로 사라지는 것을 확인함.
Object 개수를 몇 개로 나누느냐가 아니라, "원문 불릿이 살아남는가"가
핵심입니다]**:
- 담당업무/필수자격요건/우대사항의 하위 불릿을 하나씩 전부 검토하세요.
- 각 불릿은 아래 중 하나로 반드시 처리하세요:
  1. semantic_object 하나로 독립 추출
  2. 의미가 같은 인접 불릿과 하나의 semantic_object로 병합
  3. 채용 판단과 직접 관련 없는 복지·절차·회사 소개이므로 제외
- 담당업무·필수자격요건·우대사항의 **직무 관련 불릿이 이유 없이
  통째로 누락되면 안 됩니다.**
- 여러 불릿을 하나의 Object로 병합하는 것은 허용합니다 - 단,
  evidence.source_text에는 병합된 원문 불릿을 모두 포함하거나 각
  원문이 추적 가능하도록 그대로 기록하세요.
- **레이어당 최대 개수 같은 것 때문에 원문 요구사항을 삭제하지
  마세요 - 개수 제한보다 원문 요구사항의 완전한 커버리지가
  우선입니다.**
- 같은 의미의 불릿은 병합하되, 서로 다른 요구사항을 하나로 억지로
  합치지 마세요.
- 원문에 없는 problem/thinking/task/skill/qualification/culture를
  만들지 마세요(이건 기존 규칙 그대로 유지).

**skill과 preferred는 예외적으로 반드시 확인하세요(2026-07-17, 361건
실측 감사 근거 - 생략 시 누락률이 각각 11.7%/66.0%로 확인됨)**:
- 원문(특히 자격요건/우대사항 섹션)에 SQL, Python, AWS, Docker, Excel,
  Tableau 같은 구체적인 기술/도구 고유명사가 하나라도 있으면, 그 도구명을
  담은 skill Object를 최소 1개는 반드시 만드세요. task나 qualification
  Object의 frame.method 안에 도구명을 적는 것으로는 대체되지 않습니다 -
  skill Object 자체가 독립적으로 있어야 합니다.
- 원문에 "[우대사항]", "[강력 우대]", "Preferred", "Nice to have" 같은
  명시적인 우대 섹션 헤더가 있으면, "가장 중요한 것만"이라는 원칙을
  이유로 그 섹션 전체를 생략하지 마세요 - requirement_type="preferred"인
  Object를 최소 1개는 반드시 만드세요.
- **skill Object의 frame.domain은 절대 원문 섹션 제목/카테고리명을
  그대로 베끼지 마세요**(2026-07-17 실측 근거 - "RDBMS", "Technical
  Stack" 같은 섹션 헤더를 domain에 그대로 옮겨 적은 사례가 확인됨).
  그 기술이 실제로 쓰이는 업무 맥락을 새로 요약하세요(예: "SQL"의
  domain은 "RDBMS"가 아니라 "데이터 조회/백엔드 데이터 관리", "Python"의
  domain은 "Technical Stack"이 아니라 그 기술이 실제로 쓰이는 업무
  영역).
- **기술/도구명은 불릿 리스트뿐 아니라 산문(prose) 문장 안에도
  있을 수 있습니다**(2026-07-18 실측 근거 - "[기술] Python, SQL"처럼
  리스트로 나열된 공고는 정상 추출되는데, "Hands-on experience with
  SQL, Python, or other data analytics tools", "Strong knowledge of
  Spark and Airflow", "Proficiency across Python, SQL..." 처럼 문장
  안에 자연스럽게 섞여 있으면 skill Object가 통째로 안 만들어지는
  사례가 확인됨). 문장 형태로 언급된 기술/도구 고유명사도 리스트로
  나열된 것과 동일하게 skill Object 후보로 취급하세요 - "기술
  스택처럼 정리된 섹션이 아니다"라는 이유로 건너뛰면 안 됩니다.

**중요 - 세상 지식으로 추정해서 채우지 마세요**: 공고 원문에 없는
세부사항을 "보통 이렇다"는 일반 상식으로 추정해서 넣으면 안 됩니다.
예를 들어 회사 규모, 복지 수준, 자격 요건의 정확한 성격 등이 원문에
명시되지 않았다면 절대 추정해서 채우지 마세요 - 원문에 그 단어나
사실이 명시적으로 있을 때만 쓰세요.

각 Semantic Object는 다음 필드를 가집니다(concept_id/matching_anchor는
여기서 만들지 않습니다 - 시스템이 별도로 채웁니다):
- layer: 위 6개 중 하나
- normalized_text: 이 의미를 짧고 표준화된 명사구로(예: "가격 최적화 모델
  구축"). 문장이 아니라 명사구로 간결하게 쓰세요.
- meaning: 사람이 읽는 설명(1문장). 공고에 없는 내용을 지어내지 마세요.
- frame: {{"action": "핵심 행위(동사)", "object": "행위의 대상", "goal": "목적",
  "method": ["방법/도구 목록"], "domain": "업무 영역", "outcome": "기대 결과"}}
  (모르면 빈 문자열/빈 배열)
- importance: "critical"(핵심 필수) / "core"(중요) / "normal"(보통) / "low"(부차적)
- confidence: 이 판단에 대한 확신도(0~1)
- requirement_type: "required"(필수 자격요건) / "preferred"(우대사항) /
  "responsibility"(수행 업무) / "culture"(조직문화)
- intent: 이 요구사항이 실제로 노리는 목적(1문장)
- evaluation_target: 이걸 평가할 때 실제로 보는 지표/대상(예: "매출", "정확도")
- expected_evidence: 지원자 이력서에서 어떤 근거가 나오면 이 요구사항이
  충족됐다고 볼 수 있는지, 타입 구분해서 나열:
  [{{"type": "project|metric|experience|tool", "value": "구체적 근거 예시"}}]
- concepts: 원문(evidence.source_text)에 실제로 있는 핵심 명사(구)
  1~4개(개수는 유동적입니다 - 원문이 짧으면 1개만 있어도 됩니다,
  억지로 2개 이상 채우지 마세요). 원문에 없는 상위 카테고리/추상화
  단어를 새로 만들면 안 됩니다(2026-07-21 실측 근거 - "표준
  카테고리어를 우선 쓰라"는 지시 + 예시 목록을 줬더니 LLM이 그
  예시 목록을 원문과 무관한 "선택 가능한 사전"처럼 취급해서 실행마다
  다른 예시 단어를 무작위로 가져다 씀). 예를 들어 원문이 "데이터
  표준사전 구축"이면 concepts는 ["데이터 표준사전"]처럼 원문 명사를
  그대로 쓰세요 - "기준 설계", "데이터 거버넌스"처럼 원문에 없는
  단어를 만들면 안 됩니다.

  **아래 4가지는 concepts를 채우는 과정에서 절대 하면 안 됩니다**
  (2026-07-21 실측 근거 - "명사구 경계를 더 엄격히 하라"는 지시를
  추가했더니 concepts 자체는 좀 더 안정됐지만, LLM이 그 제약을
  맞추려고 Object 자체를 다시 병합/재구성해서 원문 커버리지가
  깨지는 부작용이 확인됨 - Object 생성과 concepts 추출은 서로
  영향을 주면 안 되는 별개 단계입니다):
  1. concepts 때문에 evidence.source_text를 바꾸지 마세요 -
     source_text는 이미 확정된 원문 인용이고, concepts는 그 뒤에
     그 인용문을 보고 뽑는 것입니다.
  2. concepts 때문에 Object 병합 여부를 바꾸지 마세요 - "이 불릿들을
     합치면 concepts가 몇 개 나올까"를 먼저 생각하지 마세요.
  3. concepts는 Object 생성이 전부 끝난 뒤, 이미 확정된 각 Object
     안에서만 뽑는다고 생각하세요 - Object를 만드는 동시에 concepts
     개수를 맞추려 하지 마세요.
  4. concepts 개수(1~4개)를 맞추려고 Object를 새로 만들거나, 합치거나,
     삭제하지 마세요. concepts는 이미 정해진 evidence.source_text
     안에서 몇 개가 나오든 그대로 두세요.
- evidence: {{"section": "이 요구사항이 언급된 공고 섹션(예: '자격요건')",
  "source_text": "공고 원문에서 그대로 인용한 한 문장"}}

summary(Job Overview - 사람이 읽는 요약이자 재사용 가능한 구조):
{{"job_summary": "1~2문장 압축 요약", "purpose": "이 직무가 존재하는 목적",
  "problem": "이 직무가 해결하는 문제", "thinking": "요구되는 사고방식",
  "environment": "일하는 환경/조직 특성"}}

relations(이 공고 내부에서 Problem -> Thinking -> Task -> Skill ->
Qualification으로 이어지는 관계, 실제로 이어지는 것만 - 억지로 만들지
마세요): [{{"from": "<from object의 normalized_text>", "relation":
"addressed_by|executed_via|uses|requires", "to": "<to object의 normalized_text>"}}]

resume_customization_targets(위에서 만든 semantic_objects의 concepts를
그대로 집계한 것 - 새 어휘를 여기서 새로 만들지 마세요, semantic_objects
concepts 안에 없는 단어가 여기 나오면 안 됩니다):
- project_priority: **problem + task 레이어**(thinking 레이어는 여기서
  절대 쓰지 마세요)의 critical/core Object들의 concepts를 모아
  중복 제거(최대 6개). 형식: [{{"concept": "...", "reason": "이
  concept를 만든 Object의 importance+layer 그대로, 예: 'Critical
  Task'"}}]
- experience_focus: 아래 우선순위를 **순서대로** 확인해서, critical/
  core concept가 하나라도 나오는 첫 단계에서 멈추고 그 레이어의
  concepts만 씁니다(최대 4개, 중복 제거). 상위 단계에 하나라도 있으면
  하위 단계는 보지 않습니다 - 여러 레이어를 섞어 담지 마세요.
  1. thinking 레이어의 critical/core concepts
  2. (1이 비어있을 때만) task 레이어의 critical/core concepts
  3. (1,2가 모두 비어있을 때만) problem 레이어의 critical/core concepts
  4. (1,2,3이 모두 비어있을 때만) qualification 레이어 중 **실제 수행
     경험을 요구하는 것만**(단순 도구 사용 능력이 아니라 "~경험"/
     "~구축"/"~관리" 같은 실무 이력을 요구하는 것) - 예: "SQL 사용
     능력"은 도구 보유 여부라 제외, "데이터 모델링 경험"/"데이터
     품질 관리 경험"/"Pipeline 구축 경험"은 실무 이력이라 포함.
  이렇게 단계적으로 보는 이유: thinking 레이어만 쓰면 공고에 따라
  (특히 "어떻게 접근하는가"보다 "무엇을 하는가"로만 서술되는 공고,
  2026-07-21 실측 근거 - Data Architect 공고에서 thinking 레이어가
  0개라 experience_focus가 항상 빈 배열로 나온 사례 확인) 이 target이
  영구히 비어서 핵심 경험 강조 Rule이 절대 동작하지 않습니다.
  project_priority와 소스 레이어가 겹칠 수 있다는 점은 감수합니다 -
  project_priority는 "어느 프로젝트를 먼저 보여줄까"(프로젝트 단위),
  experience_focus는 "그 프로젝트 안에서 어느 불릿을 먼저 보여줄까"
  (불릿 단위)로 쓰이는 목적 자체가 다릅니다. concepts는 이미 있는
  semantic_objects의 concepts를 그대로 재사용하세요 - 여기서 새
  concept를 만들지 마세요. 형식은 project_priority와 동일.
- skill_priority: critical/core skill Object의 실제 스킬/도구명(예:
  "SQL", "Python") 목록을 중요도 순으로(최대 6개). concepts가 아니라
  스킬 이름 그대로 문자열 리스트로 쓰세요 - reason 불필요.
- headline_focus: critical problem+thinking Object들의 concepts를
  모아 중복 제거(최대 4개). 형식은 project_priority와 동일.
- expression_focus: critical/core 전체 Object들의 concepts 중 가장
  자주 등장한 것(최대 4개). 형식은 project_priority와 동일.
- avoid_changes: 이 공고와 무관해서 건드리면 안 되는 이력서 영역이
  있으면 문자열로(없으면 빈 배열).

[회사] {company}
[직무] {title}
[공고 원문]
{posting_text}

아래 JSON으로만 답하세요(다른 설명 없이):
{{"summary": {{"job_summary": "...", "purpose": "...", "problem": "...", "thinking": "...", "environment": "..."}}, "semantic_objects": [{{"layer": "...", "normalized_text": "...", "meaning": "...", "frame": {{"action": "", "object": "", "goal": "", "method": [], "domain": "", "outcome": ""}}, "importance": "...", "confidence": 0.0, "requirement_type": "...", "intent": "...", "evaluation_target": "...", "expected_evidence": [], "concepts": [], "evidence": {{"section": "", "source_text": ""}}}}], "relations": [{{"from": "...", "relation": "...", "to": "..."}}], "resume_customization_targets": {{"project_priority": [{{"concept": "", "reason": ""}}], "experience_focus": [{{"concept": "", "reason": ""}}], "skill_priority": [], "headline_focus": [{{"concept": "", "reason": ""}}], "expression_focus": [{{"concept": "", "reason": ""}}], "avoid_changes": []}}}}
"""

_RESUME_PROMPT = """아래 이력서를 읽고 아래 스키마로 분석하세요.
(docs/semantic_object_schema.md에 확정된 Semantic Object 스키마입니다 -
필드를 추가/삭제/변경하지 마세요.)

레이어는 problem(이 사람이 반복적으로 해결해온 문제) / thinking(일하는
방식/사고방식) / task(실제 수행 업무) / skill(사용 기술/도구) /
qualification(자격/경력 사실) 5개입니다. **레이어당 최대 4개까지만**
만드세요(가장 대표적인 것만 - 명확하지 않으면 그 레이어는 비워도
됩니다, 억지로 채우지 마세요).

**task/skill/qualification/thinking은 예외적으로 다음 기준을 반드시
따르세요(2026-07-17, 원문 대조 감사 근거 - "대표적인 것만" 원칙 때문에
서로 다른 업무가 하나로 합쳐지거나 독립된 기술/자격증이 통째로
누락되는 사례가 확인됨)**:
- task/problem/thinking 공통: 이력서의 "해결 과정 및 역할"처럼 여러
  하위 항목(불릿)으로 나뉜 부분이 있으면, **하위 항목 하나하나를 전부
  검토하세요** - 항목 중 하나라도 problem/thinking/task 어디에도 전혀
  반영되지 않고 통째로 빠지면 안 됩니다(2026-07-17 실측: 4개 하위
  항목 중 1개가 완전히 누락되고 다른 1개는 problem/thinking에 중복
  반영되는 사례가 확인됨). 각 하위 항목을 가장 잘 맞는 레이어에
  최소 1개의 Object로는 반드시 반영하세요 - 이미 다른 하위 항목을
  반영한 Object에 욱여넣지 말고, 내용이 명확히 다르면 별도 Object로
  만드세요.
- skill: "보유 기술"/"기술 스택" 섹션에 별도 항목으로 나열된 도구는,
  이미 어느 Task의 method 목록에 적혀 있더라도 그 자체로 독립된 Skill
  Object를 만드세요 - Task의 method 안에 적는 것으로 대체되지 않습니다.
- qualification: 자격증/자격 검정 결과는 원문에 나열된 개수만큼 전부
  각각 별도 Qualification Object로 만드세요(하나만 대표로 뽑지 마세요).
- **thinking은 프로젝트 상세 섹션(문제 정의/해결 과정)만 보지 말고
  이력서 최상단 요약도 반드시 확인하세요**(2026-07-18, 사용자 확정 -
  "프로젝트별로 균형 있게 반영"이라는 지시만으로는 실제로 균형이
  안 맞는 사례가 재확인됨). 최상단 요약이나 자기소개에 "무엇을
  다르게 봤는지 / 어떻게 접근을 바꿨는지"에 해당하는 사고방식 전환이
  적혀 있으면, 그 내용이 해당 프로젝트의 상세 섹션(문제 정의/해결
  과정)에서 똑같은 문장으로 반복되지 않았더라도 그 프로젝트의
  Thinking Object로 반영하세요 - 프로젝트 상세 섹션에만 있는 내용만
  Thinking으로 인정하면 최상단 요약에만 적힌 핵심 통찰이 통째로
  빠집니다. 이력서에 프로젝트가 여러 개면 이렇게 확인한 뒤에도
  실제로 프로젝트별 사고방식이 서로 다른 경우, 한 프로젝트에만
  몰아서 만들지 말고 프로젝트별로 균형 있게 반영하세요.
- **evidence.source_text는 원문 문장을 그대로 복사한 인용이어야
  합니다**(2026-07-18 실측 근거 - 서로 다른 위치의 여러 문장을
  조합하거나 원문에 없는 표현을 덧붙여 하나의 "요약 문장"으로
  재구성한 뒤 그것을 source_text에 넣은 사례가 확인됨. 이러면
  이 Object가 실제로 원문 어디에 근거하는지 검증할 수 없게 됩니다).
  여러 문장에 걸친 내용이면 그중 가장 핵심적인 원문 문장 하나만
  그대로 인용하세요 - 문장을 합치거나 다른 말로 바꿔쓰지 마세요.

**중요 - 세상 지식으로 추정해서 채우지 마세요**: 이력서 원문에 없는
세부사항을 "보통 이렇다"는 일반 상식으로 추정해서 넣으면 안 됩니다.
예를 들어 "OO 교육과정 수료"라고만 적혀 있으면 그 과정이 국비지원인지,
유료인지, 온라인인지 원문에 없으면 절대 추정해서 붙이지 마세요("국비
지원 OO 과정 수료"처럼 쓰면 안 됨 - 실제로 이런 사례가 발견됐습니다).
회사 규모, 자격증 등급, 재직 기간의 정확한 성격 등도 마찬가지입니다 -
원문에 그 단어나 사실이 명시적으로 있을 때만 쓰세요.

각 Semantic Object는 다음 필드를 가집니다(concept_id/matching_anchor는
여기서 만들지 않습니다 - 시스템이 별도로 채웁니다):
- layer: 위 5개 중 하나
- normalized_text: 이 의미를 짧고 표준화된 명사구로(예: "가격 예측 모델
  구축"). 문장이 아니라 명사구로 간결하게 쓰세요 - 이후 시스템이 이
  텍스트로 유사 개념을 찾는 데 씁니다.
- meaning: 사람이 읽는 설명(1문장). 이력서에 없는 내용을 지어내지 마세요.
- frame: {{"action": "핵심 행위(동사)", "object": "행위의 대상", "goal": "목적",
  "method": ["방법/도구 목록"], "domain": "업무 영역", "outcome": "기대 결과"}}
  (모르면 빈 문자열/빈 배열)
- importance: "core"(대표 경험) / "major"(주요 경험) / "minor"(부차적 경험)
- confidence: 이 판단에 대한 확신도(0~1)
- persona_tags: 이 경험/행동이 실제로 보여주는 태도·업무 스타일을
  2단어 이내의 아주 짧은 단어로(예: "주도성", "문제해결", "가설검증",
  "협업", "실행력", "고객중심", "데이터기반") 최대 3개까지. 이력서
  원문/frame에 그 행동이 명확히 드러날 때만 붙이세요 - 명확하지 않으면
  빈 배열([])로 두세요(성격을 짐작해서 채우지 마세요, 실제로 한 행동만
  라벨링합니다).
- concepts: 원문(evidence.source_text)에 실제로 있는 핵심 명사(구)
  1~4개(개수는 유동적입니다 - 원문이 짧으면 1개만 있어도 됩니다,
  억지로 2개 이상 채우지 마세요). persona_tags와 다른 필드입니다
  (persona_tags는 태도/스타일 전용, concepts는 업무/기술 범주 전용).
  원문에 없는 상위 카테고리/추상화 단어를 새로 만들면 안 됩니다.
  예를 들어 이력서 원문이 "데이터 기준 재정립: 비활성 숙소를 제외한
  14,399개 운영 숙소 기준으로 데이터를 재정립"이면 concepts는
  ["데이터 기준", "재정립"]처럼 원문 명사를 그대로 쓰세요 - "데이터
  모델링", "기준 설계"처럼 원문에 없는 단어를 만들면 안 됩니다.

  **아래 4가지는 concepts를 채우는 과정에서 절대 하면 안 됩니다**
  (2026-07-21 실측 근거 - "명사구 경계를 더 엄격히 하라"는 지시를
  추가했더니 concepts 자체는 좀 더 안정됐지만, LLM이 그 제약을
  맞추려고 Object 자체를 다시 병합/재구성해서 원문 커버리지가
  깨지는 부작용이 확인됨 - Object 생성과 concepts 추출은 서로
  영향을 주면 안 되는 별개 단계입니다):
  1. concepts 때문에 evidence.source_text를 바꾸지 마세요.
  2. concepts 때문에 Object 병합 여부를 바꾸지 마세요.
  3. concepts는 Object 생성이 전부 끝난 뒤, 이미 확정된 각 Object
     안에서만 뽑는다고 생각하세요.
  4. concepts 개수(1~4개)를 맞추려고 Object를 새로 만들거나, 합치거나,
     삭제하지 마세요.
- evidence: {{"project": "관련 프로젝트/회사명", "section": "이력서 섹션(예: '경력', '프로젝트')",
  "source_text": "이력서 원문에서 그대로 인용한 한 문장"}}

summary(Profile Overview - 사람이 읽는 요약이자 재사용 가능한 구조):
{{"profile_summary": "1~2문장 압축 요약", "core_strength": "가장 두드러지는 강점",
  "problem_solved": "이 사람이 대표적으로 풀어온 문제", "thinking_style": "일하는 방식/사고방식"}}

relations(이 사람의 경험 내부에서 Problem -> Thinking -> Task -> Skill ->
Qualification으로 이어지는 관계, 실제로 이어지는 것만 - 억지로 만들지
마세요): [{{"from": "<from object의 normalized_text>", "relation":
"addressed_by|executed_via|uses|requires", "to": "<to object의 normalized_text>"}}]

[이력서 원문]
{resume_raw}

아래 JSON으로만 답하세요(다른 설명 없이):
{{"summary": {{"profile_summary": "...", "core_strength": "...", "problem_solved": "...", "thinking_style": "..."}}, "semantic_objects": [{{"layer": "...", "normalized_text": "...", "meaning": "...", "frame": {{"action": "", "object": "", "goal": "", "method": [], "domain": "", "outcome": ""}}, "importance": "...", "confidence": 0.0, "persona_tags": [], "concepts": [], "evidence": {{"project": "", "section": "", "source_text": ""}}}}], "relations": [{{"from": "...", "relation": "...", "to": "..."}}]}}
"""


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_jd_understanding(
    company: str, title: str, posting_text: str, provider: str = "gemini",
    timeout: int = 60, max_retries: int = 2,
) -> dict:
    """JD 1건에 대해 Semantic Object 스키마(docs/semantic_object_
    schema.md)로 summary/semantic_objects/relations를 LLM 1회 호출로
    생성한다. concept_id/matching_anchor는 여기서 만들지 않는다(호출부가
    Concept Resolver로 채운다). 실패하면(JSON 파싱 오류 포함) 한 번만
    재시도하고, 그래도 실패하면 예외를 던진다 - 호출부(backfill 스크립트)
    가 그 JD만 건너뛰고 계속 진행할지 결정한다.

    2026-07-18 - posting_text를 3000자로 자르던 걸 8000자로 늘렸다.
    영문 Greenhouse 공고(특정 해외 테크기업 등)는 상단에 회사소개/베네핏 같은
    보일러플레이트가 길게 붙어서 실제 skill/자격요건 문단이 5000~
    6000자 지점에야 나오는 경우가 확인됨 - 3000자 컷에서는 LLM이 그
    구간을 아예 보지 못했다(산문형 skill 추출 프롬프트 규칙을 추가해도
    입력 자체가 잘려서 검증조차 안 됐던 원인).

    2026-07-19(사용자 요청) - timeout/max_retries를 파라미터로 뺐다.
    기본값(60초 x 3회)은 배치 백필 스크립트(안정성 우선)를 그대로
    유지하고, 대화형 경로(ensure_jd_semantic_objects가 Top50 화면
    렌더링 중 cold job을 lazy-generation할 때)만 호출부에서 짧은 값을
    넘기게 한다 - judge()/judge_for_customization()에 적용한 것과
    같은 원칙(화면 앞에서 기다리는 사용자는 빠른 실패가 낫다)."""
    # 2026-07-25(실측 확인, 홀드아웃 10건 중 7건 JSON 절단 실패) - thinking_budget
    # 억제가 llm_client.py에서 고쳐지긴 했지만, 사고를 억제해도 답변 자체가
    # 7000토큰을 넘는 JD가 실측된 적이 있다(candidates_token_count=7048)
    # - max_tokens에 여유를 둔다(6000 -> 9000).
    prompt = _JD_PROMPT.format(company=company or "", title=title or "", posting_text=(posting_text or "")[:8000])
    raw = call_llm(prompt, provider=provider, max_tokens=9000, timeout=timeout, max_retries=max_retries)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError:
        raw = call_llm(
            prompt + "\n\n(JSON 형식으로만, 잘리지 않게 간결하게 다시 답하세요.)",
            provider=provider, max_tokens=9000, timeout=timeout, max_retries=max_retries,
        )
        return _parse_json(raw)


def generate_resume_understanding(resume_raw: str, provider: str = "gemini") -> dict:
    """Resume 1건에 대해 Semantic Object 스키마(docs/semantic_object_
    schema.md)로 summary/semantic_objects/relations를 LLM 1회 호출로
    생성한다. concept_id/matching_anchor는 여기서 만들지 않는다 - LLM은
    normalized_text까지만 만들고, concept_taxonomy.resolve_concept()가
    저장 직전에 채운다(get_or_generate_resume_understanding 참고)."""
    prompt = _RESUME_PROMPT.format(resume_raw=resume_raw)
    raw = call_llm(prompt, provider=provider, max_tokens=6000)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError:
        raw = call_llm(prompt + "\n\n(JSON 형식으로만, 잘리지 않게 간결하게 다시 답하세요.)", provider=provider, max_tokens=6000)
        return _parse_json(raw)


def _resolve_concepts(semantic_objects: list[dict]) -> list[dict]:
    """각 Semantic Object의 normalized_text를 Concept Resolver(코드,
    concept_taxonomy.py)에 넣어 concept_id/matching_anchor를 채운다.
    새 LLM 호출 없음. 문서 내에서 참조 가능한 id(레이어별 순번, 예:
    "task_001")도 여기서 같이 부여한다 - LLM은 id를 만들지 않는다(다른
    필드와 마찬가지로 안정성을 위해 코드가 결정) - meaning_matching.py의
    Match Object가 이 id로 Semantic Object를 참조한다."""
    from resume_input.concept_taxonomy import resolve_concept

    layer_seq: dict[str, int] = {}
    resolved = []
    for obj in semantic_objects:
        layer = obj.get("layer", "")
        layer_seq[layer] = layer_seq.get(layer, 0) + 1
        r = resolve_concept(
            obj.get("normalized_text", ""), layer,
            (obj.get("frame") or {}).get("domain", ""),
        )
        resolved.append({
            **obj, "id": f"{layer}_{layer_seq[layer]:03d}",
            "concept_id": r["concept_id"], "matching_anchor": r["matching_anchor"],
        })
    return resolved


def _derive_legacy_short_semantic(summary: dict) -> dict:
    """새 summary(Profile Overview)에서 옛 short_semantic 4필드를
    파생한다(새 LLM 호출 없음) - M3+M4가 체크포인트 3에서 교체되기
    전까지 과도기 호환용."""
    return {
        "핵심역할": summary.get("core_strength") or None,
        "핵심문제": summary.get("problem_solved") or None,
        "핵심기대": summary.get("thinking_style") or None,
        "부적합한경우": None,
    }


def _derive_legacy_semantic_graph(relations: list[dict]) -> list[str]:
    """새 relations(Document-internal, 분기가 있는 그래프일 수 있음)에서
    옛 M3+M4가 쓰는 "A -> B -> C" 체인 문자열을 파생한다(새 LLM 호출
    없음). 한 노드에 다음 노드가 여러 개면 마지막 것만 따라간다(과도기
    호환용 단순화 - 체크포인트 3에서 M3+M4가 교체되면 이 파생은
    필요없어진다)."""
    edges = [(r["from"], r["to"]) for r in relations if r.get("from") and r.get("to")]
    if not edges:
        return []
    targets = {t for _, t in edges}
    roots = [f for f, _ in edges if f not in targets]
    roots = list(dict.fromkeys(roots))
    next_of = dict(edges)

    chains = []
    for root in roots[:3]:
        chain, cur, seen = [root], root, {root}
        while cur in next_of and next_of[cur] not in seen:
            cur = next_of[cur]
            chain.append(cur)
            seen.add(cur)
        if len(chain) >= 2:
            chains.append(" -> ".join(chain))
    return chains


def _derive_legacy_context(summary: dict) -> str:
    """새 summary(Profile Overview) 필드를 이어붙여 옛 context(자연
    문단) 형태를 근사한다 - Top30 판단(judge_and_reorder_top30)의
    resume_context 입력으로 계속 쓰인다(체크포인트 4에서 교체 전까지)."""
    parts = [summary.get(k) for k in ("profile_summary", "core_strength", "problem_solved", "thinking_style")]
    return " ".join(p for p in parts if p)


def get_or_generate_resume_understanding(resume_raw: str, provider: str = "gemini") -> dict:
    """이력서 Understanding을 resume_hash 기준으로 캐시 조회하고, 없으면
    생성 후 저장한다. 이력서가 바뀌면 해시가 달라지므로 자연스럽게
    재생성된다(career_level과 동일한 "1회 생성, 캐시" 원칙을 이력서
    쪽에도 적용).

    반환 dict는 새 필드(semantic_objects/relations/summary)와 옛 필드
    (context/short_semantic/semantic_graph, summary에서 파생 - 새 LLM
    호출 없음)를 모두 담는다 - M3+M4(meaning_matching.py)가 체크포인트
    3에서 교체되기 전까지는 옛 필드로 계속 동작해야 한다."""
    from resume_input import execution_logger, job_store

    r_hash = execution_logger.resume_hash(resume_raw)
    cached = job_store.get_resume_understanding(r_hash)
    if cached is not None and cached.get("representation_version") == RESUME_REPRESENTATION_VERSION:
        return {
            "context": cached["context"],
            "short_semantic": json.loads(cached["short_semantic"]),
            "semantic_graph": json.loads(cached["semantic_graph"]),
            "semantic_objects": json.loads(cached["semantic_objects"]),
            "relations": json.loads(cached["relations"]),
            "summary": json.loads(cached["summary"]),
        }

    rep = generate_resume_understanding(resume_raw, provider=provider)
    summary = rep.get("summary", {})
    semantic_objects = _resolve_concepts(rep.get("semantic_objects", []))
    relations = rep.get("relations", [])

    short_semantic = _derive_legacy_short_semantic(summary)
    semantic_graph = _derive_legacy_semantic_graph(relations)
    context = _derive_legacy_context(summary)

    job_store.save_resume_understanding(
        resume_hash=r_hash, context=context,
        short_semantic_json=json.dumps(short_semantic, ensure_ascii=False),
        semantic_graph_json=json.dumps(semantic_graph, ensure_ascii=False),
        representation_version=RESUME_REPRESENTATION_VERSION,
        semantic_objects_json=json.dumps(semantic_objects, ensure_ascii=False),
        relations_json=json.dumps(relations, ensure_ascii=False),
        summary_json=json.dumps(summary, ensure_ascii=False),
    )
    return {
        "context": context, "short_semantic": short_semantic, "semantic_graph": semantic_graph,
        "semantic_objects": semantic_objects, "relations": relations, "summary": summary,
    }


def get_cached_resume_understanding(resume_raw: str) -> dict | None:
    """LLM 호출 없이 캐시에 있을 때만 Resume Understanding 을 반환한다.
    캐시가 없거나 representation_version 이 낮으면 None - 호출부는 빈 값으로
    진행하고, 실제로 필요할 때 get_or_generate_...가 생성한다. 세션 시작 시
    마지막 이력서 자동 복원(app.py)이 여기서 조용히 조회만 하도록 쓴다."""
    from resume_input import execution_logger, job_store

    r_hash = execution_logger.resume_hash(resume_raw)
    cached = job_store.get_resume_understanding(r_hash)
    if cached is not None and cached.get("representation_version") == RESUME_REPRESENTATION_VERSION:
        return {
            "context": cached["context"],
            "short_semantic": json.loads(cached["short_semantic"]),
            "semantic_graph": json.loads(cached["semantic_graph"]),
            "semantic_objects": json.loads(cached["semantic_objects"]),
            "relations": json.loads(cached["relations"]),
            "summary": json.loads(cached["summary"]),
        }
    return None


import re as _re

# 2026-07-17 추가 - LLM 생성이 비결정적이라(실측: 특정 게임사 AI Native
# Full Stack Engineer 공고를 같은 프롬프트로 두 번 호출했더니 1차는
# skill 0개, 2차는 4개) 같은 JD도 호출마다 skill/preferred가 통째로
# 빠질 수 있다. 원문에 명백한 신호(구체적 기술명, 우대사항 헤더)가
# 있는데 결과에 없으면 "불완전"으로 보고 1회 재시도한다 - 프롬프트를
# 더 손보는 대신 재시도(reroll)로 비결정성을 완화하는 방식.
_KNOWN_SKILL_KEYWORDS = _re.compile(
    r"\bSQL\b|\bPython\b|\bJava\b|\bAWS\b|\bGCP\b|\bAzure\b|\bDocker\b|"
    r"\bKubernetes\b|\bReact\b|\bNode\.js\b|\bTypeScript\b|\bExcel\b|"
    r"\bTableau\b|\bFigma\b|\bC\+\+\b|\bSpark\b",
    _re.I,
)
_PREFERRED_HDR_HINT = _re.compile(r"우대\s*(사항|조건|요건)|preferred|nice\s*to\s*have", _re.I)


def _looks_incomplete(posting_text: str, semantic_objects: list[dict]) -> bool:
    """원문에 명백한 신호가 있는데 해당 레이어/타입이 결과에서 통째로
    빠졌으면 True. 신호 자체가 원문에 없으면(예: 공고가 정말 기술스택을
    안 나열함) 불완전으로 보지 않는다 - "없는 걸 억지로 만들라"가 아니라
    "있는데 빠진 것만" 재시도 대상으로 잡는다."""
    text = posting_text or ""
    has_skill_kw = bool(_KNOWN_SKILL_KEYWORDS.search(text))
    has_pref_hdr = bool(_PREFERRED_HDR_HINT.search(text))

    layers = {o.get("layer") for o in semantic_objects}
    req_types = {o.get("requirement_type") for o in semantic_objects}

    if has_skill_kw and "skill" not in layers:
        return True
    if has_pref_hdr and "preferred" not in req_types:
        return True
    return False


def ensure_jd_semantic_objects(
    jobs: list[dict], provider: str = "gemini", max_workers: int = 8,
    timeout: int = 20, max_retries: int = 1,
) -> dict:
    """Hybrid Backfill(2026-07-16, 사용자 확정) - jobs(Top50 확정 후에만
    호출할 것) 중 새 스키마(jd_semantic_objects)가 아직 없거나 **원문이
    바뀐** job만 그 자리에서 1회 생성하고 저장한다. 이미 있고 원문도
    안 바뀌었으면 LLM을 다시 부르지 않는다(캐시 재사용) - `ensure_jd_
    context()`와 같은 "화면에 실제로 필요한 것만, 그것도 없을 때만 생성"
    원칙을 그대로 따른다.

    2026-07-19(사용자 요청) - 이 함수는 문서화된 대로 항상 Top50 확정
    직후, 즉 사용자가 화면 앞에서 기다리는 대화형 경로에서만 불린다
    (실제 프로덕션 호출부는 pipeline.py 1곳뿐 - job_store.py의 언급은
    주석뿐). 그래서 기본 timeout/max_retries를 judge()와 같은 원칙으로
    짧게(20초 x 2회) 낮췄다 - cold job이 여러 건이면 실패한 job마다
    60초씩 쌓이던 게 체감 대기시간의 큰 부분이었다(실측: 크레딧 소진
    상태에서 cold job N건이면 최악의 경우 기존 기본값 기준 N x 180초까지
    가능했음). generate_jd_understanding() 자체의 기본값(60초 x 3회)은
    안 건드렸다 - 배치 백필 스크립트가 그 기본값에 의존한다.

    2026-07-16 posting_hash 추가: 예전엔 `jd_semantic_objects`가 있으면
    무조건 스킵했다 - 재크롤링으로 posting_text가 갱신돼도(예: 오늘
    복지/제출서류 섹션이 새로 채워진 공고) job_id는 안 바뀌므로 예전
    Semantic Object를 계속 쓰는 문제가 있었다. 생성 시점의 posting_text
    해시(`posting_hash` 컬럼)와 지금 posting_text 해시를 비교해서 다르면
    재생성 대상으로 잡는다.

    반드시 **Top50이 이미 확정된 뒤에만** 호출해야 한다 - 그보다 큰
    풀(buffer_n) 전체에 걸면 매 요청마다 수십 건의 불필요한 LLM 호출이
    발생한다(비용/지연 모두 문제). job dict 자체(jd_semantic_objects
    등)도 그 자리에서 갱신해서 호출부가 바로 이어서 쓸 수 있게 한다.

    2026-07-17 병렬화(사용자 확정) - 이전엔 이 for-loop가 순차적이라
    cold job이 여러 건이면 LLM 호출(네트워크 대기)이 그대로 직렬로
    쌓였다(실측: 검색 1회에 cold 27건이면 순차 대기 시간이 전체 실행
    시간의 대부분을 차지). `backfill_jd_representations()`(오프라인
    배치 backfill용, 이미 검증된 패턴 - 1,861건을 8개 동시 요청으로
    613초·100% 성공)와 동일한 원칙을 그대로 재사용한다: LLM 호출
    (`generate_jd_understanding`)만 ThreadPoolExecutor로 동시 처리하고,
    Concept Resolver(DB 읽기/쓰기)와 SQLite 저장(`save_job_representation`)
    은 동시 쓰기 경합을 피하기 위해 메인 스레드에서 순차 처리한다. LLM
    호출 횟수/무엇을 생성하는지/결과값은 전혀 안 바뀐다 - "언제 어떤
    순서로 부르느냐"만 바뀐다."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from resume_input.job_store import save_job_representation

    to_generate: list[tuple[dict, str, bool]] = []  # (job, current_hash, is_incompleteness_retry)
    for job in jobs:
        current_hash = _posting_hash(job.get("posting_text", ""))
        # 2026-07-17(사용자 확정) - posting_hash만 보면 프롬프트를 고쳐도
        # 이미 생성된 job은 원문이 안 바뀌는 한 영원히 재생성되지 않는다
        # (실측 확인: skill/preferred 추가, domain 헤더 복사 금지 규칙을
        # 추가한 뒤에도 기존 465건이 전혀 재생성되지 않았음). 저장 시점의
        # representation_version이 지금 JD_REPRESENTATION_VERSION과
        # 다르면 원문이 안 바뀌었어도 재생성 대상으로 잡는다.
        is_stale_version = job.get("representation_version") != JD_REPRESENTATION_VERSION
        if job.get("jd_semantic_objects") and job.get("posting_hash") == current_hash and not is_stale_version:
            if job.get("semantic_object_regen_attempted"):
                continue  # 이미 한 번 재시도했다 - 더 반복하지 않는다
            try:
                existing_objects = json.loads(job.get("jd_semantic_objects") or "[]")
            except json.JSONDecodeError:
                existing_objects = []
            if not _looks_incomplete(job.get("posting_text", ""), existing_objects):
                continue
            to_generate.append((job, current_hash, True))
            continue
        to_generate.append((job, current_hash, False))

    def _call_llm_only(job: dict) -> tuple[dict | None, Exception | None]:
        try:
            rep = generate_jd_understanding(
                job.get("company", ""), job.get("title", ""), job.get("posting_text", ""),
                provider=provider, timeout=timeout, max_retries=max_retries,
            )
            return rep, None
        except (LLMCallError, json.JSONDecodeError, Exception) as e:  # noqa: BLE001 - 이 job만 건너뛰고 계속 진행
            return None, e

    generated, failed, retried = 0, 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(_call_llm_only, job): (job, current_hash, is_retry)
            for job, current_hash, is_retry in to_generate
        }
        for future in as_completed(future_to_pair):
            job, current_hash, is_retry = future_to_pair[future]
            rep, err = future.result()
            if rep is None:
                print(f"[ensure_jd_semantic_objects] {job.get('job_id')} 생성 실패(건너뜀): {err}")
                failed += 1
                continue

            summary = rep.get("summary", {})
            # resume_customization_targets는 별도 DB 컬럼을 새로 만들지
            # 않고 이미 있는 jd_summary(자유 JSON) 안에 키 하나만
            # 추가한다(2026-07-21, Rule Executor 구현 - DB 스키마 변경
            # 없음 원칙). presentation_layer.py/app.py의 기존 jd_summary
            # 소비 코드는 알려진 키(job_summary/purpose/...)만 .get()으로
            # 읽으므로 이 추가 키에 영향받지 않는다.
            summary = {**summary, "resume_customization_targets": rep.get("resume_customization_targets", {})}
            semantic_objects = _resolve_concepts(rep.get("semantic_objects", []))
            relations = rep.get("relations", [])

            job["jd_semantic_objects"] = json.dumps(semantic_objects, ensure_ascii=False)
            job["jd_relations"] = json.dumps(relations, ensure_ascii=False)
            job["jd_summary"] = json.dumps(summary, ensure_ascii=False)
            job["posting_hash"] = current_hash

            save_job_representation(
                job_id=job["job_id"],
                context=_derive_legacy_jd_context(summary),
                short_semantic_json=json.dumps(_derive_legacy_jd_short_semantic(summary), ensure_ascii=False),
                semantic_graph_json=json.dumps(_derive_legacy_semantic_graph(relations), ensure_ascii=False),
                representation_version=JD_REPRESENTATION_VERSION,
                semantic_objects_json=job["jd_semantic_objects"],
                relations_json=job["jd_relations"],
                summary_json=job["jd_summary"],
                posting_hash=current_hash,
                regen_attempted=True if is_retry else None,
            )
            generated += 1
            if is_retry:
                retried += 1

    return {"generated": generated, "failed": failed, "retried_for_incompleteness": retried,
            "already_had": len(jobs) - len(to_generate)}


def _derive_legacy_jd_context(summary: dict) -> str:
    """JD summary(Job Overview) 필드를 이어붙여 옛 jd_context(자연 문단)
    형태를 근사한다 - Top30 판단의 jd_context 입력으로 계속 쓰인다
    (체크포인트 4에서 교체 전까지)."""
    parts = [summary.get(k) for k in ("job_summary", "purpose", "problem", "thinking", "environment")]
    return " ".join(p for p in parts if p)


def _derive_legacy_jd_short_semantic(summary: dict) -> dict:
    """새 summary(Job Overview)에서 옛 short_semantic 4필드를 파생한다
    (새 LLM 호출 없음) - M3+M4가 체크포인트 3에서 교체되기 전까지 과도기
    호환용."""
    return {
        "핵심역할": summary.get("purpose") or None,
        "핵심문제": summary.get("problem") or None,
        "핵심기대": summary.get("thinking") or None,
        "부적합한경우": None,
    }


def backfill_jd_representations(jobs: list[dict], max_workers: int = 8) -> dict:
    """representation_version이 최신이 아닌 job들에 대해 Understanding을
    생성하고 job_store에 저장한다. LLM 호출(네트워크 대기)만 동시
    요청(ThreadPoolExecutor)으로 처리한다 - 실측(2026-07-13): 1,861건을
    8개 동시 요청으로 613초, 100% 성공. Concept Resolver(DB 읽기/쓰기)는
    동시 쓰기로 인한 SQLite 잠금 경합을 피하기 위해 메인 스레드에서
    순차 처리한다(원래도 save_job_representation은 순차 호출이었음 -
    구조를 바꾸지 않고 그 사이에 concept 해석만 추가). LLM 실패한 개별
    job은 건너뛰고 계속 진행한다(전체가 막히지 않도록) - 그 job은
    Retrieval 단계에서 BM25 fallback으로 처리된다(meaning_matching.py
    참고)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from resume_input.job_store import save_job_representation

    def _process(job: dict) -> tuple[str, dict | None, Exception | None]:
        try:
            rep = generate_jd_understanding(job.get("company", ""), job.get("title", ""), job.get("posting_text", ""))
            return job["job_id"], rep, None
        except (LLMCallError, json.JSONDecodeError, Exception) as e:  # noqa: BLE001 - 개별 job 실패는 전체를 막지 않는다
            return job["job_id"], None, e

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, job) for job in jobs]
        for future in as_completed(futures):
            job_id, rep, err = future.result()
            if rep is None:
                failed += 1
                continue

            summary = rep.get("summary", {})
            summary = {**summary, "resume_customization_targets": rep.get("resume_customization_targets", {})}
            semantic_objects = _resolve_concepts(rep.get("semantic_objects", []))
            relations = rep.get("relations", [])

            save_job_representation(
                job_id=job_id,
                context=_derive_legacy_jd_context(summary),
                short_semantic_json=json.dumps(_derive_legacy_jd_short_semantic(summary), ensure_ascii=False),
                semantic_graph_json=json.dumps(_derive_legacy_semantic_graph(relations), ensure_ascii=False),
                representation_version=JD_REPRESENTATION_VERSION,
                semantic_objects_json=json.dumps(semantic_objects, ensure_ascii=False),
                relations_json=json.dumps(relations, ensure_ascii=False),
                summary_json=json.dumps(summary, ensure_ascii=False),
            )
            success += 1

    return {"total": len(jobs), "success": success, "failed": failed}
