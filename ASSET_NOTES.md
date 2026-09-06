# Asset Notes

- `assets/icons/*.png` (20 files, navy/flat_navy/colors 변형 포함) — UI icons were generated
  with ChatGPT specifically for this project. 실제 운영 화면(job_ai_v3)에서 쓰는 32종 중
  이 저장소 코드가 실제로 참조하는 것만 선별해 가져왔습니다.
- `assets/illustrations/*.webp` — Microsoft Fluent Emoji(MIT). 출처와 라이선스는
  `THIRD_PARTY_NOTICES.md` 참고.
- `assets/screens/*.jpg` — 공개 데모 앱(아이콘 복원 후 최신 디자인)을 직접 실행해 캡처한
  화면입니다. 회사명·공고 내용·지원 기록·지원자 이름은 전부 가상 데이터
  (`scripts/init_demo_db.py`)이고, 사진·전화번호·이메일·실제 지원 회사·실제 지원 결과·
  실명은 없습니다.

## 개인 문서(이력서/포트폴리오) 정책 — 2026-09-06 변경

**완성된 개인 문서 파일(PDF/PPTX)은 이 저장소에 포함하지 않습니다.** 이전에는 사진·전화번호·
이메일만 제거한 사본을 `assets/demo/`에 커밋했지만, 채용 담당자가 포트폴리오를 보려면 편집용
원본부터 내려받아야 하는 구조가 되어 이제는 실제 문서 파일 자체를 Git 추적 대상과 README
링크에서 완전히 제외했습니다(`.gitignore`의 `*.pdf`/`*.pptx` 규칙).

- **이력서**: 실제 이력서의 텍스트 콘텐츠(이름/학력/프로젝트/기술/자격증/링크)는
  `resume_input/personal_resume_data.py`(로컬 전용, `.gitignore` 대상)에 있습니다. 구조는
  `resume_input/personal_resume_data.py.example`(저장소에 포함)을 참고하세요. 이 파일이 없으면
  `resume_input/html_resume.py`의 맞춤 이력서 PDF 생성 기능 자체가 비활성화되고 "개인 문서
  템플릿은 공개 저장소에서 제외되었습니다." 안내만 표시됩니다(README §10 참고).
- **포트폴리오**: 실제 포트폴리오 원본(개인 PC 전용 PPTX 파일, 이
  저장소에는 없고 원본 자체도 수정하지 않음)에서 증명사진·전화번호·이메일만 제거한 사본을
  로컬 경로 `assets/demo/portfolio_source.pptx`에 두고 씁니다. 이 경로도 `.gitignore` 대상이라
  공개 저장소에는 파일이 없습니다 - 없으면 `resume_input/portfolio_pptx.py`의 재배치 기능이
  비활성화되고 같은 안내 문구를 보여줍니다.
- **데모(공개 데모 앱이 실제로 쓰는 데이터)**: 위 실제 문서와 완전히 분리된, 처음부터 새로
  작성한 가상 지원자 프로필입니다(`scripts/init_demo_db.py: DEMO_RESUME_TEXT`). 이름이 없고
  "신입"/"전문학사" 수준의 가상 텍스트를 PyMuPDF 내장 한글 폰트로 그 자리에서 PDF로 렌더링해
  씁니다 - 파일로 저장되거나 저장소에 커밋되지 않습니다. 이 가상 프로필이 인용하는 프로젝트
  이름 3개(AI 기반 취업 의사결정 시스템/서울 Airbnb 호스트 수익 최적화 가이드/리워드 비용
  최적화 전략)는 README/포트폴리오에도 이미 공개된 프로젝트명이라 재사용했습니다(개인정보
  아님).
