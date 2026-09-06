# JobFit AI

공고 탐색부터 지원 판단·맞춤 자료 준비·결과 관리까지 연결한 개인용 AI 취업 자동화 시스템

> 🚀 [공개 데모 바로가기](https://jobfit-ai-zsdpzufdvzcjagkz5hro2y.streamlit.app/) · Source Code: https://github.com/noisnh8-tech/jobfit-ai

---

## 1. 문제 발견

지원 여부와 관계없이 공고마다 탐색부터 결과 관리까지 같은 작업이 반복됐습니다.

```
공고 탐색 → 내용 확인 → 경험 비교 → 지원 판단 → 지원자료 수정 → 지원 기록 → 결과 확인
```

**목표**

반복되는 지원 작업을 하나의 워크플로로 연결해 지원에 드는 시간과 판단 리소스를 줄인다.

---

## 2. 전체 흐름

```
공고 수집·정제 → 관련 후보 정렬 → 선택 공고 분석·지원 판단 → 맞춤 지원 준비 → 지원 기록·결과 관리
```

### 공개 데모

실제 시스템에는 개인 이력서·포트폴리오·지원 기록·결과 메일과 외부 계정이 연결되어 있어, 공개 데모에서는 처음부터 새로 작성한 가상 이력서와 가상 공고를 사용하고 실제 수집·외부 LLM·이메일 연동·포트폴리오 생성은 비활성화했습니다.

---

## 3. 공고 수집 자동화

매일 공고를 다시 확인하는 작업을 줄이고 일부 출처가 실패해도 전체 수집이 중단되지 않도록, n8n은 실행 시점만 관리하고 Python은 수집부터 저장까지 순차 처리합니다.

```
매일 10시 실행(n8n) → 공고 수집(API·Playwright) → 상세 본문 보강(Python) → 품질·중복·마감·비활성 확인(Rule) → SQLite 저장
```

**입력 경로**

기업 ATS 공개 API · 채용 플랫폼 API·키워드 검색 · 공고 직접 붙여넣기

**정제**

본문 품질 확인 · 중복 공고 통합 · 마감 공고 제외 · 비활성 링크 확인 · 출처별 실패 기록

---

## 4. 공고 목록·관련 후보 정렬

관련성이 낮은 공고까지 LLM으로 분석하는 비용과 대기시간을 줄이면서 먼저 확인할 공고를 좁히기 위해, 전체 공고는 Rule과 로컬 의미 모델로 비교했습니다.

### 이력서 구조화

프로젝트 문장에 문제·접근 방식·수행 업무·기술이 함께 포함되어 있어, 문맥을 기준으로 경험을 구분하기 위한 LLM 구조화입니다.

```
이력서 PDF → 텍스트 추출(PyMuPDF) → 문제·사고·업무·기술·보유조건 구조화(LLM) → 원문 근거 저장 → 이력서 변경 기준 캐시(SQLite)
```

| 구조 | 내용 |
|---|---|
| problem | 해결하려 한 문제 |
| thinking | 접근·가설·검증 방식 |
| task | 실제 수행 업무 |
| skill | 사용 기술 |
| qualification | 학력·경력·자격 등 보유 조건 |

### 공고 목록 Rule 구조화

JD는 담당업무·요구기술·지원조건이 비교적 명확하게 구분되어 있어, 전체 공고 비교에 필요한 세 항목만 Rule로 구성합니다.

```
전체 JD → 담당업무·요구기술·지원조건 추출(Rule) → 공고별 비교 구조
```

| 구조 | 내용 |
|---|---|
| task | 공고의 담당 업무 |
| skill | 공고에 명시된 요구 기술 |
| qualification | 경력·학력·자격 등 지원조건 |

### 이력서–JD 비교

```
명시적 경력 조건 필터(Rule) → 이미 지원한 공고·사용자가 제외한 공고·중복 공고 제거 → JD Rule 구조화 → 업무·기술 의미 비교(MiniLM·BM25) → 경력 차이 감점(Rule) → 관련도순 정렬
```

| 이력서 | JD | 비교 |
|---|---|---|
| task | task | 수행 업무와 담당 업무 |
| skill | skill | 보유 기술과 요구 기술 |
| qualification | qualification | 보유 조건과 지원조건 |

---

## 5. 지원 판단·선택 공고 정밀 분석

### 선택 공고 정밀 분석

공고의 요구사항을 업무·기술·경력·책임·도메인으로 구분하고, 표현이 다르거나 다른 도메인에서 전이 가능한 경험까지 연결하기 위한 LLM 분석입니다.

```
분석할 공고 선택 → JD 원문 + 캐시된 이력서 구조 + Rule 사전검증 결과 → 요구사항 유형 분류·경험 연결(LLM 1회) → 직무 핵심·근거·부족 영역·자료 우선순위
```

| 유형 | 분석 대상 |
|---|---|
| task | 실제 수행 업무 |
| skill | 요구 기술·도구 |
| qualification | 학력·자격 등 객관 조건 |
| experience | 요구 경력·경험 수준 |
| responsibility | 담당 범위·책임 수준 |
| domain | 산업·서비스·업무 도메인 |

| 분석 결과 | 내용 |
|---|---|
| 직무 핵심 | 공고에서 수행할 핵심 업무 |
| 직무 맥락 | 목적·도메인·주요 역량 |
| 연결 상태 | 요구사항별 match·partial·no_match |
| 이력서 근거 | 연결된 경험의 원문·사실 |
| 부족 영역 | 확인이 필요하거나 부족한 조건 |
| 자료 우선순위 | 먼저 보여줄 기술·프로젝트 |

- 근거 범위: 이력서 원문·사실 요약
- 생성 제한: 이력서에 없는 경험·성과 수치

### 지원 판단

동일 JD 재실행 검증에서 30건 중 5건의 판단이 달라져, LLM은 의미 연결까지만 담당하고 최종 판단은 명시된 기준에 따라 Rule로 재검증합니다.

```
LLM 분석 결과 → 경력·학력·자격 비교(Rule) → 구조적 부족 확인(Rule) → 지원·보류·비추천
```

```
필수조건 미충족 → 비추천
미충족 없음 + 확인불가 또는 구조적 부족 → 보류
미충족·확인불가·구조적 부족 없음 → 지원
```

### 결과 재사용

```
공고 ID + 이력서 해시 + 분석 버전 → 분석 결과 캐시(SQLite) → 지원 판단·맞춤 자료·자소서에서 재사용
```

---

## 6. 맞춤 지원 준비

상세 분석 결과를 다시 생성하지 않고 기술·프로젝트 우선순위와 이력서 근거를 각 지원자료에 이어서 사용합니다.

```
상세 분석 결과 → 기술·프로젝트 우선순위 → 맞춤 이력서·포트폴리오 → 선택 시 자소서
```

### 맞춤 이력서

```
기술·프로젝트 순서 반영 → 기존 문장·성과 유지 → HTML·CSS 구성 → PDF 생성(Chrome Headless·PyMuPDF)
```

### 맞춤 포트폴리오

```
관련 프로젝트 우선 배치 → About 카드·슬라이드 블록 이동 → 프로젝트·페이지 번호 재구성 → PPTX 생성(python-pptx)
```

- 문장·프로젝트 내용 변경 없음
- PDF 변환: Windows·Microsoft PowerPoint 필요
- 공개 데모: 포트폴리오 생성 제외

### 자소서

```
사용자 선택 → 이력서 근거·상세 분석 결과 불러오기 → 소재 구성 → Planner → Writer → Critic → 코드 기반 근거 검증
```

| 단계 | 역할 |
|---|---|
| 소재 구성 | 기존 분석 결과 정리·LLM 미사용 |
| Planner | 관점·전개 방향 선택 |
| Writer | 선택한 소재로 초안 작성 |
| Critic | 질문 적합성·근거·과장 검토 |
| Code Verifier | 소재에 없는 숫자·요구사항 검사 |

미선택 시 LLM 호출 없음 · 생성 결과는 Streamlit 세션에서만 유지

---

## 7. 지원 기록·결과 관리 자동화

### 지원 기록·준비시간

지원 과정의 준비시간과 상태를 따로 기록하는 작업을 줄이기 위해 화면 이벤트와 상태 변경을 자동 저장합니다.

```
분석 시작 → 지원 판단 → 맞춤 자료 확정 → 지원 완료 → 지원 기록·준비시간 저장(SQLite)
```

지원 상태 이력 · 단계별 준비시간 · 자소서 사용 여부 · 준비시간 중앙값

### 결과 관리

결과 메일의 반복되는 전형 표현을 같은 기준으로 반영하기 위한 Rule 기반 분류와 지원 기록 매칭입니다.

```
결과 메일 감지(n8n) → 전형 결과 분류(Rule) → 지원 기록 매칭(FastAPI) → 상태·이력·처리 로그 갱신(SQLite)
```

```
1건 확인 → 자동 갱신
여러 건 → 검토
후보 없음 → 미매칭
처리 완료 → 중복 방지
```

---

## 8. 주요 파일 구조

```
jobfit-ai/
├── app.py                              # Streamlit 화면·라우팅
├── ops_api.py                          # 결과 메일과 지원 기록 연결 API
├── ui_theme.py                         # 화면 테마
├── ui_components.py                    # 공통 UI 구성요소
├── ui_icons.py                         # 공개용 아이콘
├── resume_input/
│   ├── pipeline.py                     # 후보 정렬·상세 분석 흐름
│   ├── candidate_search.py             # 후보 공고 검색·필터
│   ├── manual_jd_input.py              # 공고 직접 입력
│   ├── rule_representation.py          # 공고 목록 Rule 구조화
│   ├── layer_m3.py                     # 구조 기반 후보 비교
│   ├── meaning_matching.py             # MiniLM·BM25 의미 비교
│   ├── constraint_layer.py             # 경력 조건 필터·감점
│   ├── quick_analysis.py               # 선택 JD 구조화·LLM 정밀 분석
│   ├── eligibility_compare.py          # 경력·학력·자격 비교
│   ├── judge_engine.py                 # 지원·보류·비추천 판단
│   ├── job_store.py                    # 공고·분석 결과·캐시 저장
│   ├── application_manager.py          # 지원 기록·상태 이력 관리
│   ├── preparation_tracker.py          # 준비시간 이벤트·KPI
│   ├── understanding.py                # 이력서 LLM 구조화
│   ├── resume_facts.py                 # 학력·경력·자격 사실값
│   ├── html_resume.py                  # 맞춤 이력서·PDF 생성
│   ├── resume_template.py              # 이력서 출력 템플릿
│   ├── portfolio_pptx.py               # 포트폴리오 슬라이드 재배치
│   ├── cover_letter_engine.py          # 자소서 생성·검증
│   └── runtime_mode.py                 # 운영·데모 환경 분리
├── scripts/
│   └── init_demo_db.py                 # 가상 데이터 기반 데모 DB 생성
├── .env.example                        # 환경변수 예시
├── .gitignore
├── requirements.txt                    # 기본 실행 패키지
├── requirements-windows.txt            # Windows 전용 패키지
├── COPYRIGHT.md                        # 복제·재배포·상업적 이용 조건
├── THIRD_PARTY_NOTICES.md              # 제3자 라이브러리·자산 고지
└── ASSET_NOTES.md                      # 공개 자산 출처·사용 범위
```

---

## 9. 공개 데모 실행

```bash
git clone https://github.com/noisnh8-tech/jobfit-ai.git
cd jobfit-ai

python -m venv .venv
```

**Windows:**
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/init_demo_db.py
streamlit run app.py
```

**macOS·Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_demo_db.py
streamlit run app.py
```

**FastAPI:**
```bash
uvicorn ops_api:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

---

## 10. 실사용 관찰

실제 지원 20건 적용

| 항목 | 기존 | JobFit 적용 |
|---|---|---|
| 공고 검토 후 미지원 | 6건 | 지원 준비 후 0건 |
| 자소서 미포함 준비시간 | 약 15~20분 | 약 1분 |
| 자소서 포함 준비시간 | 약 40분~1시간 | 약 2분 30초 |

> 개인 실사용 기록을 기반으로 하며, 측정 시점과 지원 공고가 달라 동일 조건에서의 효과 검증에는 한계가 있습니다.

---

## 11. 현재 한계·개선 방향

| 현재 한계 | 개선 방향 |
|---|---|
| 개인 사용 환경 중심 검증 | 적용 공고·지원 결과 데이터 확대 |
| 고정된 기업·직무군 중심 탐색 | 이력서 기반 탐색 직무군 추천 |
| 공고 형식에 따른 Rule 인식 차이 | 섹션·조건 추출 범위 확대 |
| 선택 공고 LLM 결과 변동 | 판단 검증 기준·자동 테스트 보완 |

---

## 이용 조건

프로젝트 코드와 문서의 이용 조건은 `COPYRIGHT.md`를 따르며, 제3자 라이브러리와 자산은 `THIRD_PARTY_NOTICES.md`와 `ASSET_NOTES.md`에서 확인할 수 있습니다.
