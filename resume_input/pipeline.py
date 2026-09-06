"""
resume_input/pipeline.py

Service 레이어 - 추천 모드/개별 공고 분석 모드를 하나의 실행 흐름으로
연결한다. UI(Streamlit/API 등)는 이 파일의 함수만 호출한다. 로직을
UI 쪽에 넣지 않는다.

판단(run_judgment_only)과 후속 실행(run_post_judgment_pipeline)을
분리한다 - "보류"/"비추천" 판단이 나왔는데도 커스터마이징/자소서/
지원기록까지 자동으로 끝까지 실행되던 문제(실제 시나리오 테스트로
발견)를 고치기 위함이다. 후속 실행은 호출자(UI)가 사용자 승인을 받은
뒤에만 호출해야 한다 - 이 파일 자체는 "언제 호출할지"는 모르고 "호출되면
무엇을 하는지"만 담당한다(UI 로직과 파이프라인 로직 분리).
"""
from __future__ import annotations

import json
import re

from resume_input.pdf_parser import extract_text_from_pdf
from resume_input import link_engine
# 2026-08-04(Step 8, 사용자 확정) - "분석하기" 경로의 매칭 엔진을
# semantic_matching.py(cosine 기반)에서 link_engine.py(Semantic Linking,
# LLM 1회)로 완전히 교체했다. semantic_matching.py는 이제 어디서도
# import되지 않는다(파일 자체는 삭제하지 않음 - Candidate Generation
# (목록)에서도 원래부터 안 쓰였다는 사실이 이번 교체 과정에서 재확인됨).
# match_validator(Top30 LLM Validator, 2026-07-17 제거)와 같은 원칙 -
# 옛 엔진 파일은 진단/롤백용으로 보존하되 프로덕션 경로에서는 뺀다.
from resume_input.candidate_search import search_candidates_meaning
from resume_input.manual_jd_input import wrap_pasted_jd
from resume_input.understanding import (
    get_or_generate_resume_understanding, ensure_jd_semantic_objects, JD_REPRESENTATION_VERSION,
)
# meaning_judgment.judge_and_reorder_top30(옛 Top30 LLM 재판단)은
# 2026-07-16 제거됨(run_recommendation_mode 내부 주석 참고) - import도
# 같이 제거. 모듈 자체는 삭제하지 않음(judgment_cache 등 기존 저장된
# 판단 조회용으로 다른 코드가 참조할 수 있어 체크포인트 7 정리 대상).
from resume_input.link_check import filter_live_links
from resume_input import judge_engine
# 2026-08-04(Step 8, 사용자 확정) - judge_service.judge()(LLM 1회, decision/
# reason/missing 생성)를 judge_engine.judge()(Rule, LLM 호출 없음, job의
# _semantic_match=Semantic Link 결과만 소비)로 교체했다.
#
# 2026-08-16(사용자 확정) - 이력서 자체 품질 체크리스트(Streamlit S5
# 화면)를 삭제했다. run_post_judgment_pipeline()이 매번 계산해 result
# ["checklist"]로 돌려주던 것을 그 화면 하나가 표시하는 용도였는데,
# 그 화면이 없어졌으니 계산 자체가 아무도 안 읽는 데이터가 됐다 -
# checklist.py/checklist_engine.py 파일 자체는 지우지 않는다(다른
# 화면에서 재사용할 수도 있어 - rewrite_engine.py를 지우지 않은 것과
# 같은 원칙).
#
# 2026-08-20(사용자 확정, 죽은 LLM 호출 제거) - checklist_semantic을
# 만들던 judge_for_checklist() 호출(공고 1건당 LLM 1회, run_apply_flow()
# 안에서 결과를 아무도 안 읽는데도 계속 돌고 있었다)을 제거했다.
# judge_service.judge_for_checklist() 함수 자체는 지우지 않는다(같은
# 원칙) - 이 파일이 더 이상 import하지 않을 뿐이다.
from resume_input import customizer
from resume_input import customization_rules
from resume_input import customization_planner
from resume_input import resume_apply_engine
from resume_input import analysis_engine
from resume_input import pdf_generator
from resume_input import resume_versions
from resume_input import resume_template
from resume_input.application_manager import (
    add_application, list_applied_job_ids, list_applied_posting_keys,
)
from resume_input.dismissed_jobs import list_dismissed_job_ids, list_dismissed_posting_keys
from resume_input.posting_identity import (
    normalize_title as _normalize_title_for_dedup,
    posting_key as _posting_key,
)
from resume_input.job_store import list_all_candidate_jobs
from resume_input import job_store
from resume_input.career_filter import filter_by_career_level, compute_career_status, format_career_reason
# 2026-07-23(서비스 구조 변경, 사용자 확정) - Rule Ranking을
# run_recommendation_mode()에서 제거하면서 fit_grade/reason_phrasing/
# extract_display_sections는 이 파일에서 더 이상 쓰지 않는다(모듈
# 자체는 삭제하지 않음 - fit_grade.py 등은 legacy 비교용으로 보존).
# extract_display_sections는 job_detail.py/app.py가 별도로 계속 쓴다.
from resume_input.resume_career import extract_user_career_level
from resume_input.job_family import DATA_ANALYTICS
from resume_input.resume_facts import extract_resume_facts
from resume_input import execution_logger
from resume_input import preparation_tracker


def _compute_semantic_match(resume_semantic_objects: list[dict], resume_raw: str, job: dict) -> dict | None:
    """2026-08-04(Step 8, 사용자 확정) - link_engine.run_semantic_linking()
    (Semantic Linking, LLM 1회)로 교체했다. 이전에는 semantic_matching.
    match_resume_to_jd()(cosine 엔진, LLM 0회)를 호출했다 - 매칭 자체를
    LLM 판단으로 옮긴 것(대체 이유는 docs/design/semantic_linking_
    schema_v1.md 참고). jd_semantic_objects가 없거나 비어 있으면 None을
    반환한다(기존과 동일 - 호출부가 폴백 화면을 쓴다).

    반환 shape은 옛 Match Object 스키마({"matches", "coverage",
    "ranking_score"})와 다르다({"link_version", "links",
    "unmatched_resume_objects"}) - presentation_layer.py를 화면 섹션
    단위로 순차 마이그레이션하는 동안(Step 1: 매칭근거 → Step 2: Gap →
    ...), 아직 마이그레이션 안 된 함수가 참조하는 옛 키(`matches`,
    각 항목의 `jd_object`/`resume_object`)를 같은 값의 별칭(alias)으로
    같이 채워 넣어 크래시를 막는다 - 새 숫자를 만드는 게 아니라 같은
    문자열 id를 다른 키 이름으로도 읽을 수 있게 하는 것뿐이라 "설명
    안 되는 숫자" 원칙과 무관하다. 마이그레이션이 전부 끝나면 이
    별칭들은 제거 대상이다.

    2026-08-16(Semantic Linking 캐시, 사용자 확정) - 지금까지 Linking은
    DB에 전혀 저장되지 않고 매번 새로 LLM을 호출했다(judgment_cache/
    jd_semantic_objects와 달리). ②~⑤ Customization Planner가 전부 이
    결과(Analysis 5-state의 원천)를 공통 입력으로 쓰는데, 세션이 바뀔
    때마다 다른 A/B/C/D가 나오면 같은 공고의 맞춤화 판단이 날마다
    달라진다 - 성능이 아니라 판단 일관성 문제라 이번 MVP 범위에 넣는다.
    (job_id, resume_hash, link_engine.LINK_SCHEMA_VERSION)이 전부 같으면
    캐시를 그대로 쓰고, 셋 중 하나라도 다르면(이력서 변경/Linking 로직
    변경) 다시 계산한다. job_id가 없는 호출(단위 테스트 등)은 캐시를
    건너뛴다 - 캐시 키를 만들 수 없을 뿐 정상 동작은 그대로 유지한다."""
    raw = job.get("jd_semantic_objects")
    if not raw or not resume_semantic_objects:
        return None
    try:
        jd_objects = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not jd_objects:
        return None

    job_id = job.get("job_id")
    resume_hash = execution_logger.resume_hash(resume_raw)
    if job_id:
        cached = job_store.get_cached_semantic_link(job_id, resume_hash, link_engine.LINK_SCHEMA_VERSION)
        if cached is not None:
            return cached

    user_career_level = extract_user_career_level(resume_raw, DATA_ANALYTICS).get("career_level")
    result, _issues = link_engine.run_semantic_linking(
        resume_semantic_objects, jd_objects, job,
        user_career_level=user_career_level, challenge_option=False,
    )
    for link in result["links"]:
        link.setdefault("jd_object", link["jd_object_id"])  # 임시 호환 별칭
        for m in link.get("matched_resume_objects") or []:
            m.setdefault("resume_object", m["resume_object_id"])  # 임시 호환 별칭
    result["matches"] = result["links"]  # 임시 호환 별칭 - 마이그레이션 완료 후 제거

    if job_id:
        job_store.save_semantic_link_result(job_id, resume_hash, link_engine.LINK_SCHEMA_VERSION, result)
    return result


def _run_judge(job: dict, resume_raw: str) -> dict:
    """job["_semantic_match"](Semantic Link 결과, _compute_semantic_match()
    가 이미 채워둠)만 입력으로 judge_engine.judge()를 호출한다(LLM 호출
    없음). Link 결과가 없으면(jd_semantic_objects 없음/이력서 의미
    객체 없음 등) 판단 근거가 없다는 걸 그대로 알린다 - 임의로 "지원"
    등으로 기본값을 주지 않는다.

    2026-08-14(Hard Eligibility factual comparison, 사용자 확정) -
    resume_facts(학력/자격증/경력 factual value)를 같이 넘긴다. 경력은
    이미 계산해온 extract_user_career_level()을 그대로 재사용 - 새로
    계산하지 않는다."""
    link_result = job.get("_semantic_match")
    if not link_result or not link_result.get("links"):
        return {
            "decision": "보류",
            "decision_reason": ["의미 기반 분석 결과가 없어 판단할 수 없습니다."],
            "blocking_requirements": [], "required_actions": [], "structural_gaps": [],
        }
    # 경량 분석(quick_analysis) 경로면 Hard Eligibility 정규식용 JD object 를
    # job["_qa_jd_objects"] 로 이미 갖고 있다. 구경로(jd_semantic_objects 컬럼)도 지원.
    jd_semantic_objects = job.get("_qa_jd_objects") or json.loads(job.get("jd_semantic_objects") or "[]")
    career_result = extract_user_career_level(resume_raw, DATA_ANALYTICS)
    resume_facts = extract_resume_facts(resume_raw, career_result)
    return judge_engine.judge(link_result, jd_semantic_objects, resume_facts)


def _dedupe_same_posting(jobs: list[dict]) -> list[dict]:
    """회사+직무명이 완전히 같은 공고가 여러 건 있으면(실측 확인, 2026-07-14:
    URL/본문이 서로 달라 dedup_key 기준으로는 별개 공고로 저장돼 있지만,
    같은 회사가 같은 제목으로 공고를 여러 건 올려서 추천 목록에 겹쳐
    보이는 경우) 본문이 더 긴(정보가 더 완전한) 1건만 남긴다. 완전히
    다른 회사/직무는 당연히 여러 건 나와도 된다 - 여기서 줄이는 건
    "회사명+직무명까지 똑같아서 사용자가 구분할 수 없는" 경우뿐이다."""
    best: dict[tuple[str, str], dict] = {}
    for j in jobs:
        key = _posting_key(j.get("company") or "", j.get("title") or "")
        prev = best.get(key)
        if prev is None or len(j.get("posting_text") or "") > len(prev.get("posting_text") or ""):
            best[key] = j
    return list(best.values())


def _get_resume_raw(resume_pdf) -> str:
    """resume_pdf가 이미 추출된 텍스트(str)면 그대로, 파일 객체면 추출한다."""
    if isinstance(resume_pdf, str):
        return resume_pdf
    return extract_text_from_pdf(resume_pdf)


def detect_user_career_level(resume_pdf) -> dict:
    """이력서에서 연차를 자동 추출한다(LLM 미사용). S0에서 사용자에게
    "이력서에서 인식한 연차: {career_level} [변경]"으로 보여줄 값을
    만든다 - career_level은 항상 4개 값 중 하나로 확정된다(§ui_design.md
    v6.3 §0-6). Job Family는 지금 DATA_ANALYTICS로 고정한다(이
    프로젝트가 데이터 분석 직무 지원자만 대상으로 하기 때문 - 나중에
    Job Family 선택 기능이 생겨도 resume_career.py는 수정 불필요)."""
    resume_raw = _get_resume_raw(resume_pdf)
    result = extract_user_career_level(resume_raw, DATA_ANALYTICS)
    return {"resume_raw": resume_raw, **result}


_RECALL_POOL_SIZE = 100

# 사용자가 목록에서 직접 제외(dismiss)한 공고를 대체할 여유분. 화면에는
# _RECALL_POOL_SIZE개만 유지하되, 세션 중 dismiss가 생기면 목록을 다시
# 계산(수십 초)하지 않고 이미 랭킹·링크확인까지 끝난 다음 순위 공고로
# 즉시 채우기 위한 버퍼다. dismiss가 0건이면 결과는 기존과 완전히 동일
# 하다(뒤쪽 _DISMISS_RESERVE개는 계산만 하고 잘라낸다).
# `search_candidates_meaning`의 final_top_n은 순수 tail slice라(점수/정렬은
# 전체 풀 기준으로 이미 계산됨) 이 값을 키워도 상위 100위의 집합·순서는
# 바이트 단위로 동일하다 - 랭킹 알고리즘 변경이 아니다.
_DISMISS_RESERVE = 25


def run_recommendation_mode(
    resume_pdf, user_career_level: str, challenge_option: bool = False, pool_limit: int = 200,
) -> dict:
    """탐색 모드(2026-07-25 재정정, 사용자 확정 - 목록 생성 경로에서
    JD Understanding/LLM을 다시 제거한다).

    같은 날 있었던 두 번의 상반된 결정을 정리한다:
    - (이전 결정, 같은 날 더 이른 시점) "JD Understanding을 검색 경로에
      다시 연결" - live_pool 전체에 대해 ensure_jd_semantic_objects()를
      목록 반환 전에 동기로 돌렸다. 카드가 항상 채워져 보이는 장점은
      있었지만, "크롤링 → LLM → 후보생성" 구조와 실질적으로 같아서
      신규 공고가 많을수록 목록 응답이 LLM 대기시간에 묶이는 문제가 있었다.
    - (최종 결정, 이 버전) Rule Representation(Task/Skill/Qualification,
      LLM 미사용 - resume_input/rule_representation.py)이 이미 M3
      랭킹뿐 아니라 카드 표시(presentation_layer.build_job_card_view)에도
      쓰이므로, 목록 화면은 LLM 없이 완전히 구성 가능하다는 게 확인됐다.
      그래서 JD Understanding 동기 생성 블록을 다시 제거했다 - 목록은
      Rule만 쓰고, LLM은 사용자가 [분석하기]를 눌렀을 때 그 공고 1건에
      대해서만 ensure_job_semantic_match() 경로에서 발동한다.

    흐름: Career Filter -> 지원한 공고 제외 -> M3+M4로 Top100 Recall
    구성(_RECALL_POOL_SIZE - combined_score는 내부 신호일 뿐 화면에
    노출·적합도로 해석하지 않는다) -> 최신순 정렬 -> 죽은 링크 확인 ->
    반환. JD Understanding은 이 함수 안에서 전혀 호출하지 않는다.

    warm job(이미 jd_semantic_objects가 있는 job)을 recall 순위와 무관하게
    강제로 얹던 예전 union(combined_ids | warm_ids)도 그대로 제거된 채다 -
    M3+M4가 jd_semantic_objects 유무와 무관하게(Rule Representation
    유효하면 LayerM3Scorer, 아니면 jd_short_semantic/BM25) 순위를
    매기므로 warm 여부가 Top100 선정에 영향을 준 적이 없다.

    LLM은 이 함수 안에서 딱 한 곳만 호출한다: get_or_generate_resume_
    understanding()의 이력서 캐시 미스 1회뿐. 공고(JD) 쪽 LLM 호출은
    이 함수에 전혀 없다 - ensure_job_semantic_match()가 담당한다."""
    resume_raw = _get_resume_raw(resume_pdf)
    resume_understanding = get_or_generate_resume_understanding(resume_raw)

    all_jobs = list_all_candidate_jobs()
    total_before_filter = len(all_jobs)

    career_filtered = filter_by_career_level(all_jobs, user_career_level, challenge_option)

    # 지원한 공고 / 사용자가 직접 제외한 공고를 후보 풀에서 뺀다. 이건
    # source lifecycle 이 아니라 "사용자 선택"이라 jobs/candidate_jobs
    # 원본은 그대로 두고 별도 테이블만 참조한다.
    #
    # 2026-09-01: job_id(=url::...) 뿐 아니라 (회사, 정규화 직무명) 키로도
    # 매칭한다. 같은 자리를 가리키는 공고가 서로 다른 URL 로 여러 건
    # 저장돼 있어서(실측 165개 그룹 - Greenhouse 재발행/원티드 크로스포스팅
    # 등), job_id 하나만 빼면 _dedupe_same_posting() 이 다음 수집 때 다른
    # 쌍을 남기며 "제외한 공고가 되살아난다". posting_identity 참고.
    applied_ids = list_applied_job_ids()
    dismissed_ids = list_dismissed_job_ids()
    suppressed_keys = list_applied_posting_keys() | list_dismissed_posting_keys()

    def _suppressed(j: dict) -> bool:
        return (
            j["job_id"] in applied_ids
            or j["job_id"] in dismissed_ids
            or _posting_key(j.get("company") or "", j.get("title") or "") in suppressed_keys
        )

    excluded_applied = [j for j in career_filtered if j["job_id"] in applied_ids]
    excluded_dismissed = [j for j in career_filtered if j["job_id"] in dismissed_ids]
    pool = [j for j in career_filtered if not _suppressed(j)]
    pool = _dedupe_same_posting(pool)

    # Recall: M3+M4로 Top100 후보군을 만든다(_RECALL_POOL_SIZE=100,
    # 2026-07-25 - 500에서 축소, 어차피 이 Top100 전체를 아래서 직접
    # JD Understanding까지 끝낸다). jd_semantic_objects 유무는 M3+M4
    # 순위에 전혀 반영되지 않는다 - 그래서 warm job을 순위와 무관하게
    # 강제로 얹던 예전 union은 제거했다(사용자 확인 - Top100을 전부
    # 최신화하므로 "이미 분석된 걸 놓친다"는 문제 자체가 없어짐).
    #
    # 2026-07-25(M3 Production Migration) - resume_semantic_objects를
    # 넘겨서 job별 Rule Representation(Task/Skill/Qualification, LLM
    # 미사용)이 유효하면 LayerM3Scorer를 쓰고, 유효하지 않으면(Header
    # Coverage Gap/오분류 등) 기존 jd_short_semantic 기반 M3로 폴백한다
    # (docs/verification/2026-07-25_m3_representation_ab/summary.md §11
    # 검증 완료 - Top10 관련도 향상, FP 안 늘어남, FN은 폴백으로 해소).
    #
    # 2026-07-31(Constraint Layer) - user_career_level/challenge_option을
    # 넘겨서 Career penalty를 combined_score에 반영한다(docs/verification/
    # 2026-07-31_candidate_generation_top100_eval/root_cause_table_top20.md
    # 검증 완료 - "N년 이상" 같은 자격요건이 Similarity로만 처리되면
    # 감점이 아니라 오히려 점수를 올리는 현상 확인). career_status를
    # 새로 계산하는 게 아니라 아래 221행에서 이미 쓰던 compute_career_status
    # 결과를 점수 계산에도 그대로 재사용하는 것뿐이다.
    combined_top = search_candidates_meaning(
        resume_raw,
        resume_understanding["short_semantic"],
        resume_understanding["semantic_graph"],
        final_top_n=min(_RECALL_POOL_SIZE + _DISMISS_RESERVE, len(pool)),
        jobs=pool,
        resume_semantic_objects=resume_understanding["semantic_objects"],
        user_career_level=user_career_level,
        challenge_option=challenge_option,
    )
    pool_by_id = {j["job_id"]: j for j in pool}
    candidates = [pool_by_id[j["job_id"]] for j in combined_top if j["job_id"] in pool_by_id]

    for j in candidates:
        jd_level = j.get("career_level")
        j["career_status"] = compute_career_status(user_career_level, jd_level, challenge_option)
        j["career_reason"] = format_career_reason(user_career_level, jd_level)

    # 최신순 정렬(기본, 사용자 확정 - Rule Ranking을 대체하는 정렬
    # 기준이 아니라 그냥 "가장 최근에 올라온 공고부터" 보여주는 것뿐).
    candidates.sort(key=lambda j: j.get("created_at") or "", reverse=True)

    # 죽은 링크 확인(link_check.py)은 건당 네트워크 요청이라 전체
    # candidates(최대 Top100)를 매번 검사하면 느리다 - 이미 최신순
    # 정렬했으므로 화면에 실제로 보여줄 pool_limit개(+더보기로 늘어날
    # 몫)만 앞에서부터 검사한다(기존 buffer 원칙과 동일, 정렬 기준만
    # ranking_score에서 최신순으로 바뀜).
    live_pool = filter_live_links(candidates, target_n=min(pool_limit, len(candidates)))

    # 2026-07-25 재설계(사용자 확정) - JD Understanding(LLM)을 목록 생성
    # 경로에서 다시 제거한다. 목록 화면은 Rule Representation(Task/Skill/
    # Qualification, LLM 미사용)만으로 완전히 구성 가능하다(presentation_
    # layer.build_job_card_view 참고) - 신규 공고가 몇 건이 들어오든
    # 목록 단계에서는 LLM 대기가 없다. JD Understanding은 사용자가
    # [분석하기]로 공고 1건을 선택했을 때만 ensure_job_semantic_match()가
    # 그 1건에 대해 트리거한다(과거엔 여기서 live_pool 전체에 동기로
    # 미리 생성해뒀으나, 이는 "크롤링→LLM→후보생성" 구조와 사실상 같은
    # 문제였다 - 목록 단계는 Rule, 상세 분석 단계는 LLM으로 책임을
    # 분리한다).

    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for j in live_pool:
        status_counts[j["career_status"]] = status_counts.get(j["career_status"], 0) + 1
        source_counts[j.get("source") or "기업 홈페이지"] = source_counts.get(j.get("source") or "기업 홈페이지", 0) + 1

    execution_logger.log_stage(
        "retrieval", resume_hash=execution_logger.resume_hash(resume_raw), job_id=None,
        data={"pool_limit": pool_limit, "candidates": [
            {"job_id": j["job_id"], "company": j.get("company"), "title": j.get("title"),
             "career_status": j.get("career_status")}
            for j in live_pool
        ]},
    )

    return {
        "resume_raw": resume_raw,
        "candidates": live_pool,
        "stats": {
            "total_before_filter": total_before_filter,
            "after_career_filter": len(career_filtered),
            "excluded_applied_count": len(excluded_applied),
            "excluded_dismissed_count": len(excluded_dismissed),
            "returned_count": len(live_pool),
            "career_status_distribution": status_counts,
            "source_distribution": source_counts,
        },
    }


def _jd_understanding_needs_regeneration(job: dict) -> bool:
    """JD Understanding(jd_semantic_objects)이 없거나 구버전인지 확인한다.
    ensure_job_semantic_match()에서만 쓴다(목록 생성 경로에서는 이제
    JD Understanding을 아예 호출하지 않는다 - run_recommendation_mode
    참고)."""
    if not job.get("jd_semantic_objects"):
        return True
    return job.get("representation_version") != JD_REPRESENTATION_VERSION


def _run_quick_analysis(resume_raw: str, job: dict) -> dict | None:
    """경량 분석(quick_analysis, LLM 1회) - "분석하기"의 유일한 LLM 지점.
    2026-08-31(사용자 확정) JD Understanding + Semantic Linking 2회 호출을
    이 1회로 압축했다. Resume Understanding(이력서당 1회, resume_hash 캐시,
    추천에서도 재사용)만 앞에서 유지한다. 캐시: semantic_link_cache 테이블을
    linking_version=quick_analysis.SCHEMA_VERSION 으로 재사용(스키마 변경 없음).

    반환: quick_analysis 결과 dict(job_core/role_context/requirements/
    resume_order). 이력서 의미 객체를 못 만들면 None."""
    from resume_input import quick_analysis

    resume_understanding = get_or_generate_resume_understanding(resume_raw)
    resume_semantic_objects = resume_understanding.get("semantic_objects") or []
    if not resume_semantic_objects:
        return None

    job_id = job.get("job_id")
    r_hash = execution_logger.resume_hash(resume_raw)
    if job_id:
        cached = job_store.get_cached_semantic_link(job_id, r_hash, quick_analysis.SCHEMA_VERSION)
        if cached is not None:
            return cached

    user_career_level = extract_user_career_level(resume_raw, DATA_ANALYTICS).get("career_level")
    qual_validation = link_engine.compute_qualification_validation(
        resume_semantic_objects, job, user_career_level, challenge_option=False,
    )
    qa = quick_analysis.run_quick_analysis(
        resume_semantic_objects, resume_raw, job, qual_validation,
    )
    if job_id:
        job_store.save_semantic_link_result(job_id, r_hash, quick_analysis.SCHEMA_VERSION, qa)
    return qa


def _attach_quick_analysis(job: dict, qa: dict) -> None:
    """quick_analysis 결과를 job dict 에 붙인다 - 기존 소비자(judge_engine/
    analysis_engine/cover_letter_engine/presentation)가 그대로 쓰도록
    link_result 호환 shape(`_semantic_match`)도 같이 채운다."""
    from resume_input import quick_analysis
    job["_quick_analysis"] = qa
    job["_semantic_match"] = quick_analysis.to_link_result(qa)
    job["_qa_jd_objects"] = quick_analysis.to_jd_objects(qa)


def ensure_job_semantic_match(resume_pdf, job: dict) -> dict:
    """공고 분석 화면(S1D) 진입 시 공고 1건에 대해 경량 분석(LLM 1회)을
    실행해 job dict 에 붙인다. 이미 있으면(job["_quick_analysis"]) 다시
    호출하지 않는다."""
    if job.get("_quick_analysis") is not None:
        return job
    resume_raw = _get_resume_raw(resume_pdf)
    qa = _run_quick_analysis(resume_raw, job)
    if qa is not None:
        _attach_quick_analysis(job, qa)
    return job


def run_judgment_only(resume_pdf, job: dict) -> dict:
    """판단만 한다 - judge_service 1회 호출. 커스터마이징/자소서/지원기록은
    실행하지 않는다. 두 모드(추천 모드에서 공고 선택 직후 / JD 붙여넣기
    직후) 공통 진입점이다.
    """
    resume_raw = _get_resume_raw(resume_pdf)
    judge_result = _run_judge(job, resume_raw)
    execution_logger.log_stage(
        "judge", resume_hash=execution_logger.resume_hash(resume_raw), job_id=job.get("job_id"),
        data=judge_result,
    )
    return {"resume_raw": resume_raw, "job": job, "judge_result": judge_result}


def run_jd_analysis_mode(
    resume_pdf, jd_text: str, company: str = "", title: str = "", url: str = ""
) -> dict:
    """개별 공고 분석 모드: 붙여넣은 JD를 job dict로 감싼 뒤 판단만 한다.

    2026-07-19(사용자 지적) - 추천 모드(S1D)는 Semantic Object/Match
    Object 기반 새 화면(①②③④, presentation_layer.py)을 쓰는데, 이
    모드는 옛 judge() 판단 결과만 만들어서 같은 화면(app.py의
    _render_job_analysis_panel)에서도 항상 구 폴백 경로(use_semantic=
    False)로 빠졌다 - "붙여넣은 공고는 정밀 분석이 안 붙는다"는 사용자
    확인. 여기서 이 job도 jd_semantic_objects를 생성하고 Semantic
    Match를 계산해서 job dict에 붙이면, has_semantic_data(job)가
    True가 되어 같은 새 화면을 그대로 탄다(새 판단 로직 추가 아님 -
    검색 결과 job에 이미 붙던 것과 동일한 계산을 여기서도 한 번 하는
    것뿐).

    검색(Top50) 경로와 달리 여기는 사용자가 지금 이 순간 "분석" 버튼을
    눌러 명시적으로 요청한 공고 1건뿐이다 - Top50에서 문제였던 "cold
    job 여러 건이 쌓여 검색 자체가 막히는" 구조적 문제가 이 경로에는
    없으므로(항상 1건), 동기 호출을 그대로 유지한다(이미 app.py가
    "판단 중입니다..." 스피너로 감싸서 호출함).

    job_id가 "manual-<해시>"라 candidate_jobs 테이블에 없는 행이다 -
    ensure_jd_semantic_objects() 내부의 DB 저장(save_job_representation,
    job_id로 UPDATE)은 대상 행이 없으면 조용히 0건 갱신되고 끝난다(에러
    아님) - job dict 자체는 그 전에 이미 채워지므로 이번 요청의 화면
    렌더링에는 영향이 없다. 붙여넣은 JD를 candidate_jobs에 새로 끼워
    넣지는 않는다(검색 후보 풀에 섞이면 안 되는 별개의 데이터)."""
    resume_raw = _get_resume_raw(resume_pdf)
    job = wrap_pasted_jd(jd_text, company=company, title=title, url=url)

    # 붙여넣기 모드도 추천 모드와 같은 경량 분석 1회(quick_analysis)를 탄다.
    # job_id 가 "manual-…" 라 캐시 저장은 됨(다음 재분석 시 hit).
    qa = _run_quick_analysis(resume_raw, job)
    if qa is not None:
        _attach_quick_analysis(job, qa)

    return run_judgment_only(resume_raw, job)


def run_post_judgment_pipeline(resume_pdf, job: dict, judge_result: dict | None = None) -> dict:
    """사용자가 [다음 단계 진행]을 명시적으로 선택했을 때만 호출한다.

    _run_judge(지원여부 판단, Rule, LLM 0회) -> customization_planner.
    build_resume_commands(analysis_engine의 5-state Evidence Contract를
    Command로 변환, Rule, LLM 0회) -> resume_apply_engine.apply_commands
    (Command를 실제 이력서 원문에 반영, LLM 0회) -> checklist 순서로
    실행한다. judge_result를 이미 갖고 있으면(run_judgment_only 결과)
    그대로 넘겨서 재조회를 피한다.

    2026-08-16(Customization Planner, 사용자 확정 - 이번 세션 ②③④⑤①
    라우팅/최소변경 검증) - resume_customizing.build_resume_commands()
    (v4, raw link_result의 improvement 제안만 쓰던 구 로직)를
    customization_planner.build_resume_commands()로 교체했다. 새 로직은
    judge_engine의 Hard Eligibility 반영 + analysis_engine의 5-state(A~E)
    + JD layer 라우팅을 그대로 쓴다 - link_result에는 이 정보가 없어서
    구 로직은 애초에 이걸 쓸 수 없었다. analysis는 judge_result(전달받은
    값)를 재사용하지 않고 이 함수 안에서 judge_engine.judge()로 별도
    계산한다(app.py의 _get_analysis()와 동일 패턴) - 전달받은 judge_result가
    judge_for_checklist(LLM, hard_eligibility 없음)에서 온 경우도 있어
    analysis_engine이 요구하는 스키마(hard_eligibility 포함)를 항상
    보장하려면 이 경로가 필요하다. judge_engine.judge()는 Rule(LLM
    호출 없음)이라 매번 다시 불러도 비용이 없다.

    용어 최적화(Rule5, APPROVED_TERM_MAP 기반 승인된 치환)는 새로 만들지
    않고 customization_rules.evaluate_terms()를 그대로 재사용한다 -
    이 Rule은 Semantic Link 결과와 무관하게 독립적으로 이미 잘 동작해서
    재구현할 이유가 없다(기존 결정 유지).

    2026-08-20 - checklist_semantic(이력서 자체 품질, JD 매칭과 무관)을
    보여주던 화면(S5)이 2026-08-16에 이미 삭제됐고, 그 값을 만들던
    judge_for_checklist() 호출도 이제 이 파이프라인 어디서도 하지
    않는다(run_apply_flow가 더 이상 이 함수를 미리 호출해 넘기지 않음 -
    죽은 LLM 호출 제거, 아래 judge_result는 항상 _run_judge()(Rule,
    LLM 0회) 결과다).

    호출 규칙(UI가 지켜야 함, 이 함수 자체는 강제하지 않음):
    - decision == "지원": 바로 호출 가능
    - decision == "보류": 경고를 보여준 뒤 사용자가 진행을 선택했을 때만 호출
    - decision == "비추천": 기본적으로 버튼을 숨기고, "그래도 진행" 선택 시에만 호출
    """
    resume_raw = _get_resume_raw(resume_pdf)
    r_hash = execution_logger.resume_hash(resume_raw)
    job_id = job.get("job_id")

    try:
        preparation_tracker.log_event(job_id, "analysis_started")
    except Exception:
        pass  # 계측 실패가 실제 분석 흐름을 막으면 안 됨

    if judge_result is None:
        judge_result = _run_judge(job, resume_raw)
        execution_logger.log_stage("judge", resume_hash=r_hash, job_id=job_id, data=judge_result)

    resume_understanding = get_or_generate_resume_understanding(resume_raw)
    resume_semantic_objects = resume_understanding.get("semantic_objects") or []

    # 2026-08-31(경량화, 사용자 확정) - 이력서 맞춤은 "기술 순서"만 PDF 에
    # 자동 반영한다. 자기소개 수정 / 용어 치환 / 프로젝트 불릿 재작성·재배치 /
    # 프로젝트 블록 통째 이동은 전부 제거(customization_planner/
    # customization_rules 는 dead path 로 남김). skill_priority 는 경량 분석
    # (quick_analysis)이 이미 준 값을 그대로 쓴다 - 새 LLM 없음.
    qa = job.get("_quick_analysis") or {}
    skill_priority = ((qa.get("resume_order") or {}).get("skill_priority")) or []
    project_priority = ((qa.get("resume_order") or {}).get("project_priority")) or []

    resume_commands: list[dict] = []
    if skill_priority:
        resume_commands.append({"type": "reorder_skills", "skill_order": skill_priority, "priority": 1})

    apply_out = resume_apply_engine.apply_commands(
        resume_raw, resume_commands, resume_semantic_objects, term_replacements=None,
    )
    resume_customized = apply_out["resume_customized"]

    # S4 화면(_s4_build_decisions)이 읽는 최소 planner_notes - "기술 순서"만
    # 실제로 PDF 에 반영되므로 그 카드만 "변경"으로 채운다. "프로젝트 순서"는
    # 별도 추천 섹션(result["project_priority"])으로만 노출하고 planner_notes
    # 에는 넣지 않는다(자동 이동 아님을 화면에서 명확히 하기 위함).
    planner_notes: dict = {}
    if apply_out["applied"].get("skill_order"):
        planner_notes["skill_order"] = {
            "decision": "변경",
            "reason": "이 공고에서 중요한 기술을 이력서 기술 목록 앞쪽으로 옮겼습니다.",
            "skill_order": apply_out["applied"]["skill_order"],
        }
    execution_logger.log_stage(
        "customizer", resume_hash=r_hash, job_id=job_id,
        data={"log": apply_out["log"], "commands": resume_commands, "applied": apply_out["applied"]},
    )

    try:
        preparation_tracker.log_event(job_id, "analysis_completed")
    except Exception:
        pass  # 계측 실패가 실제 분석 흐름을 막으면 안 됨

    return {
        "resume_raw": resume_raw,
        "job": job,
        "judge_result": judge_result,
        "resume_customized": resume_customized,
        "customization_log": apply_out["log"],
        "resume_commands": resume_commands,
        "resume_semantic_objects": resume_semantic_objects,
        "term_replacements": [],
        "applied": apply_out["applied"],
        "planner_notes": planner_notes,
        # 화면 추천용(프로젝트 순서는 PDF 자동 반영 안 함)
        "skill_priority": skill_priority,
        "project_priority": project_priority,
    }


_PDF_SAFE_COMMAND_TYPES = {"move_bullet", "replace_headline"}


def _pdf_safe_commands(resume_commands: list[dict]) -> list[dict]:
    """PDF Patch PoC(docs/verification/2026-08-16_pdf_patch_poc, Freeze)로
    안전성이 실측 검증된 범위는 "실제로 값이 바뀐 개별 필드의 등록된
    bbox만 patch"다 - move_bullet(불릿 1개 단위 이동), replace_headline
    (자기소개, 단일 고정 필드)이 여기 해당한다(2026-08-16, 사용자 확정 -
    "자동 적용: 자기소개, 개별 bullet, 간단한 JD 표현"). move_project
    (프로젝트 블록 전체 이동)/reorder_skills(기술 목록 전체 재배치)는
    여러 필드를 한 번에 다시 그려야 해서 아직 PoC 검증 범위 밖이다 -
    S4 화면 판단 카드에는 그대로 "변경" 판단을 보여주되(사용자 확정),
    PDF에는 자동 반영하지 않고 제안으로만 남긴다. replace_term(⑥ JD
    표현)은 이 필터 대상이 아니다 - Command 목록이 아니라 별도 인자
    (term_replacements)로 전달되어 항상 적용된다(기존 동작 유지).
    highlight_keyword는 애초에 apply_commands()가 실행하지 않는다."""
    return [c for c in resume_commands if c.get("type") in _PDF_SAFE_COMMAND_TYPES]


# 런타임 경로(pdf_layout)가 '나머지는 그대로, 순서만' 안전하게 반영할 수 있는
# Command. move_bullet/replace_headline(고속 경로와 동일) + reorder_skills
# (기술 목록 줄 재배치 - 각 줄을 자기 자리에서 위치만 바꿈). move_project 는
# 블록 높이가 제각각이라 제외.
_RUNTIME_PDF_COMMAND_TYPES = {"move_bullet", "replace_headline", "reorder_skills"}


def _runtime_pdf_commands(resume_commands: list[dict]) -> list[dict]:
    return [c for c in resume_commands if c.get("type") in _RUNTIME_PDF_COMMAND_TYPES]


def generate_resume_pdf(
    resume_raw: str, resume_commands: list[dict], resume_semantic_objects: list[dict],
    term_replacements: list[dict] | None = None, resume_pdf_bytes: bytes | None = None,
) -> dict:
    """S4 화면에서 호출한다 - 판단(customization_planner)과 완전히
    분리된 출력 단계(계획 확정). resume_apply_engine.apply_commands()를
    다시 실행해서(LLM 없음, 순수 텍스트 처리라 재실행 비용 무시 가능)
    "완료 데이터"를 만든 뒤 맞춤 PDF 를 생성한다.

    2026-08-30(범용 입력, 사용자 P0) - 이력서 PDF 자동 맞춤화가 사전
    등록된 파일(resume_template.py)에서만 동작하던 구조를 폐기한다.
    이제 어떤 이력서 PDF 든 사전 등록 없이 처리한다:
      1. resume_versions 캐시 히트 → 재사용
      2. 등록 템플릿 일치 → pdf_generator.generate()(고속 경로, 유지)
      3. 그 외(대부분) → pdf_layout.generate_from_pdf()
         업로드된 PDF 자체에서 좌표·폰트·크기·색을 런타임에 읽어 반영.
    hash/template 은 최적화 캐시일 뿐 - cache miss 가 실패가 아니다.
    resume_pdf_bytes 는 app.py 가 업로드 시점에 보관한 원본 PDF 바이트.

    2026-08-04(Step 8, 사용자 확정) - 입력을 옛 customization_rules.
    evaluate() decisions에서 resume_customizing.build_resume_commands()
    가 만든 Command + resume_semantic_objects + term_replacements로
    바꿨다(run_post_judgment_pipeline이 만든 것과 동일 - Resume Apply
    Engine에서 수정된 결과를 PDF Generator 입력으로 그대로 잇는다).
    호출부(app.py)도 result["resume_commands"]를 넘기도록 같이 바꿨다.

    2026-08-16(PDF Patch PoC를 S4에 연결, 사용자 확정) - resume_commands를
    그대로 다 넘기지 않고 `_pdf_safe_commands()`로 걸러서 PDF에는
    move_bullet(안전 검증됨)만 반영한다 - move_project/reorder_skills는
    화면 판단 카드/텍스트 비교에는 그대로 나오지만 실제 PDF 파일에는
    반영하지 않는다(제안만 표시, 이번 라운드 범위 밖).

    같은 이력서에 대해 이미 만든 버전과 완료 데이터(불릿순서/승인용어
    치환)가 전부 같으면 새로 만들지 않고 기존 버전을 재사용한다
    (resume_versions.py).

    반환 status: "ok"(새로 생성) | "reused"(기존 버전 재사용) |
    "manual_review"(자동 반영 불가, 원인은 reason에) | "no_change"(반영할
    변경 없음) | "error"(원본 PDF 손상/누락)."""
    from resume_input import pdf_layout

    r_hash = execution_logger.resume_hash(resume_raw)
    template_id = resume_template.find_template_id_by_resume_hash(r_hash)

    if template_id is not None:
        # 등록 템플릿 고속 경로 - PoC 로 검증된 범위(개별 불릿/자기소개)만.
        custom_out = resume_apply_engine.apply_commands(
            resume_raw, _pdf_safe_commands(resume_commands), resume_semantic_objects, term_replacements,
        )
        result = pdf_generator.generate(template_id, resume_raw, custom_out)
    else:
        # 런타임 경로(pdf_layout) - '원래 있던 텍스트를 자기 자리에서 이동'만
        # 하는 재배치(불릿·기술 목록 순서)와 자기소개·용어 치환까지 반영한다.
        # 프로젝트 블록 통째 이동(move_project)은 블록 높이가 제각각이고
        # 페이지를 걸칠 수 있어 '나머지 그대로'를 보장하며 자동 반영이
        # 불가능하므로 제외(판단 카드 제안으로만 남긴다).
        custom_out = resume_apply_engine.apply_commands(
            resume_raw, _runtime_pdf_commands(resume_commands), resume_semantic_objects, term_replacements,
        )
        result = pdf_layout.generate_from_pdf(resume_pdf_bytes, resume_raw, custom_out)
    applied = custom_out["applied"]
    if result["status"] != "ok":
        return result

    # find_reusable()는 디스크에 중복 PDF 파일을 새로 쓸지만 결정한다 -
    # changed_fields(Preview 하이라이트용 bbox 목록)는 DB에 저장하지
    # 않으므로(스키마 변경 없음, 이번 범위 밖) reused여도 방금 만든
    # result에서 그대로 쓴다 - 같은 applied 값이면 같은 fields가 나오는
    # 결정적 함수라 재사용 판단과 무관하게 항상 정확하다.
    reusable = resume_versions.find_reusable(r_hash, applied)
    if reusable:
        return {
            "status": "reused", "pdf_path": reusable["pdf_path"], "version_id": reusable["id"],
            "fields_changed": result["fields_changed"], "changed_fields": result["changed_fields"],
        }

    version = resume_versions.save_version(r_hash, applied, result["pdf_bytes"])
    return {
        "status": "ok", "pdf_path": version["pdf_path"], "version_id": version["id"],
        "fields_changed": result["fields_changed"], "changed_fields": result["changed_fields"],
    }


def run_apply_flow(resume_pdf, job: dict) -> dict:
    """추천 모드 [지원하기] 전용 진입점. decision/reason/missing은
    호출부(추천 모드 UI)가 절대 표시하지 않는다(추천 모드에는 그 개념이
    없다 - ui_design.md v3 이후 확정 원칙).

    2026-08-20(사용자 확정, 죽은 LLM 호출 제거) - checklist_semantic을
    만들던 judge_for_checklist() 호출(공고 1건당 LLM 1회)을 없앴다.
    이 값을 보여주던 이력서 품질 체크리스트 화면(S5)이 2026-08-16에
    이미 삭제됐는데 이 호출부만 남아있었다(pipeline.py 자체 주석에
    "화면 없이 계속 돌리면 순수 낭비다"라고 적혀 있던 상태) - 결과를
    아무도 안 읽는데 클릭마다 LLM을 호출하고 있었다. 이제
    run_post_judgment_pipeline(judge_result=None)을 그대로 호출해서
    내부의 _run_judge()(Rule, LLM 0회)가 계산한 judge_result를 쓴다.
    judge_service.judge_for_checklist() 함수 자체는 지우지 않는다
    (다른 코드가 재사용할 수 있어 - rewrite_engine.py 등과 같은 원칙)."""
    resume_raw = _get_resume_raw(resume_pdf)
    return run_post_judgment_pipeline(resume_raw, job)


def finalize_application(job: dict, resume_raw: str, resume_version: str | None = None) -> int:
    """최종 지원 화면의 [지원 완료] 버튼에서만 호출한다 - 지원기록은
    여기서만 생성된다. resume_version(선택) - S4에서 이미 만들어둔
    맞춤 PDF의 version_id를 넘기면 지원기록 상세에서 "그 지원 당시 쓴
    이력서"를 그대로 다시 열어볼 수 있다(2026-08-16 사용자 확정)."""
    application_id = add_application(
        job_id=job["job_id"],
        company=job.get("company", ""),
        title=job.get("title", ""),
        url=job.get("url", ""),
        source=job.get("source", ""),
        status="지원 완료",
        resume_version=resume_version,
    )
    execution_logger.log_stage(
        "application", resume_hash=execution_logger.resume_hash(resume_raw), job_id=job.get("job_id"),
        data={"application_id": application_id, "status": "지원 완료"},
    )
    try:
        preparation_tracker.log_event(job.get("job_id"), "application_created", application_id=application_id)
    except Exception:
        pass  # 계측 실패가 실제 지원 완료 처리를 막으면 안 됨
    return application_id
