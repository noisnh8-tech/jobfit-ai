"""
resume_input/job_store.py

job_ai_v3가 소유하는 병합된 후보 풀 테이블(`candidate_jobs`)을
실체화(materialize)한다. `jd_repository.py`가 매 요청마다 두 원본
DB(job_ai_v2 jobs.db의 `jobs` 테이블 138건 + da_job_market_2026의
`job_postings` 1985건)를 다시 읽고 dedup하던 것을, 여기서는 한 번
동기화해서 로컬 `candidate_jobs` 테이블에 저장해두고 이후 요청은
그 테이블만 SELECT한다.

`jobs`(job_ai_v3 own source table, 138건)와 이름이 겹치지 않도록
`candidate_jobs`로 이름을 다르게 잡았다(ui_design.md v6.3 §0-1 참고
- 구현 중 실제로 `jobs` 테이블이 이미 존재하는 걸 발견해서 결정).

career_level/career_level_confidence는 새로 추가되는 job에 대해서만
1회 계산하고, 이미 candidate_jobs에 있는 job_id는 재계산하지 않는다
(sync_jobs를 여러 번 불러도 안전 - 계산 결과가 뒤집히지 않는다).

career_level은 원본 데이터가 아니라 JD Parsing 결과(파생 데이터)다 -
`career_filter.extract_jd_career_level()`의 정규식이 나중에 개선되면
기존 값을 다시 계산해야 할 수 있다. 그래서 계산 시점의 파서 버전
(`career_filter.PARSER_VERSION`)과 계산 시각을
`career_parser_version`/`career_parsed_at` 컬럼에 같이 저장한다 -
파서가 바뀌었을 때 "재분석이 필요한 job"을 식별할 수 있게
(`list_jobs_needing_career_reparse()` 참고, 지금 이 함수는 식별만
하고 자동으로 재계산을 트리거하지는 않는다 - 트리거는 별도 판단이
필요한 작업이라 이번 범위에 넣지 않았다).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from datetime import date

from resume_input.career_filter import extract_jd_career_level, PARSER_VERSION
from resume_input.jd_repository import list_active_jobs, list_reference_jobs, _dedup_jobs, _dedup_key
from resume_input.job_source import detect_job_source
from resume_input.text_normalize import normalize_invisible_chars
# job_prep은 함수 내부에서 지연 import한다(모듈 최상단에서 import하면
# job_prep -> candidate_search -> job_store 순환 참조가 생긴다).

from resume_input.runtime_mode import DB_PATH  # 운영/데모 모드에 따라 jobs.db 또는 demo.db


def _init_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_jobs (
            job_id TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            url TEXT,
            source TEXT,
            origin TEXT,
            posting_text TEXT,
            career_level TEXT,
            career_level_confidence TEXT,
            career_parser_version TEXT,
            career_parsed_at TEXT,
            industry TEXT,
            company_size TEXT,
            location TEXT,
            jd_context TEXT,
            jd_short_semantic TEXT,
            jd_semantic_graph TEXT,
            representation_version TEXT,
            representation_generated_at TEXT,
            link_dead INTEGER DEFAULT 0,
            link_checked_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            synced_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # 이미 candidate_jobs가 있던 경우(메타데이터 컬럼 추가 이전 버전)
    # ALTER TABLE로 컬럼을 보완한다. location은 새 판단이 아니라 원본
    # DB(jd_repository의 _COLUMNS/_REF_COLUMNS)에 이미 있던 필드를
    # 그냥 옮겨오지 않고 있었던 것 - 이번에 "근무지역" 표시 요청으로
    # 발견해서 추가했다. jd_context/jd_short_semantic/jd_semantic_graph는
    # Understanding Engine 검증(2026-07-13,
    # docs/verification/2026-07-13_understanding_engine_meaning_matching)
    # 결과를 반영한 것 - career_level과 같은 원칙으로 JD당 1회만 LLM으로
    # 생성하고 저장, 재계산하지 않는다.
    for col_def in ("career_parser_version TEXT", "career_parsed_at TEXT", "location TEXT",
                    "jd_context TEXT", "jd_short_semantic TEXT", "jd_semantic_graph TEXT",
                    "representation_version TEXT", "representation_generated_at TEXT",
                    "link_dead INTEGER DEFAULT 0", "link_checked_at TEXT",
                    # 2026-07-16: Semantic Object 스키마 재설계(체크포인트 2,
                    # docs/semantic_object_schema.md) - jd_short_semantic/
                    # jd_semantic_graph는 체크포인트 3까지 과도기 호환용으로
                    # 계속 채워진다(새 데이터에서 파생, 별도 LLM 호출 없음).
                    "jd_semantic_objects TEXT", "jd_relations TEXT", "jd_summary TEXT",
                    # 2026-07-16: 마감된 공고 제외 요청 - link_dead와 동일한
                    # 원칙(하드 DELETE 대신 플래그)으로 구현한다. 하드 DELETE는
                    # sync_jobs()가 원본 DB에 그 job_id가 여전히 있으면 "신규
                    # job"으로 착각해서 그대로 되살려낸다(link_dead 컬럼을 만든
                    # 이유와 동일 - 위 mark_dead_jobs() docstring 참고).
                    "deadline_expired INTEGER DEFAULT 0", "deadline_checked_at TEXT",
                    # 2026-07-16: JD Understanding 캐시 무효화용 - 지금까지
                    # ensure_jd_semantic_objects()는 "jd_semantic_objects가
                    # 있으면 무조건 스킵"이라, 재크롤링으로 posting_text가
                    # 갱신돼도 예전 Semantic Object를 계속 썼다(job_id는 안
                    # 바뀌므로). 생성 시점의 posting_text 해시를 같이 저장해서,
                    # 지금 posting_text의 해시와 다르면 재생성 대상으로 잡는다.
                    "posting_hash TEXT",
                    # 2026-07-17: Semantic Object 불완전 생성 감지 후 1회
                    # 재생성 트리거용(ensure_jd_semantic_objects() 참고) -
                    # LLM 생성이 비결정적이라 같은 JD도 호출마다 skill/
                    # preferred 누락 여부가 달라짐(실측 확인, 특정 게임사
                    # AI Native Full Stack Engineer 공고: 1차 생성 skill
                    # 0개 -> 재생성 skill 4개). 무한 재시도를 막기 위해
                    # 이 job에 대해 이미 한 번 재시도했는지만 기록한다.
                    "semantic_object_regen_attempted INTEGER DEFAULT 0"):
        try:
            conn.execute(f"ALTER TABLE candidate_jobs ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass
    # 컬럼 추가 이전에 이미 계산된 행은 그 값이 현재 파서(PARSER_VERSION)로
    # 계산된 것이 맞으므로 소급 기록한다 - 새로 재계산하는 게 아니라 이미
    # 계산된 값에 이름표만 붙이는 것이다. career_level이 아니라
    # career_level_confidence로 "이미 처리됐는지"를 판단한다 - 연차
    # 미기재(career_level=NULL)로 판정된 job도 confidence="none"으로
    # 이미 처리된 것이지 미처리 상태가 아니다(처음엔 career_level로
    # 조건을 걸었다가, 미기재 판정 610건이 계속 "재분석 필요"로 잘못
    # 남는 버그를 발견해서 수정함).
    conn.execute(
        """
        UPDATE candidate_jobs SET career_parser_version = ?, career_parsed_at = COALESCE(career_parsed_at, synced_at)
        WHERE career_parser_version IS NULL AND career_level_confidence IS NOT NULL
        """,
        (PARSER_VERSION,),
    )
    conn.commit()
    conn.close()


def backfill_posting_hash_for_existing_jobs(db_path: Path = DB_PATH) -> dict:
    """일회성 마이그레이션(2026-07-16) - `posting_hash` 컬럼을 새로
    추가하기 전에 이미 `jd_semantic_objects`가 채워진 job은 `posting_hash`
    가 NULL이다. 그대로 두면 `ensure_jd_semantic_objects()`가 "NULL !=
    지금 posting_text 해시"로 보고 **전부** 재생성 대상으로 착각한다
    (실제로 원문이 안 바뀐 job까지 불필요하게 LLM을 다시 부르게 됨).

    **반드시 전체 재크롤링이 끝난 뒤에 1회만 실행할 것.** 지금 이 시점의
    posting_text를 "이 Semantic Object가 만들어진 원문"으로 소급
    확정하는 것이므로, posting_text가 아직 안정되지 않은 상태(재크롤링
    진행 중)에서 실행하면 이미 재크롤링된 job은 원문이 실제로 바뀌었는데도
    "안 바뀐 것"으로 잘못 도장 찍혀서 정말 필요한 재생성을 건너뛰게 된다.
    재크롤링 완료 후 이 함수를 1회 실행하면, 그 이후부터 진짜 내용이
    바뀔 때만(다음 재크롤링 등) 정상적으로 재생성 대상이 된다."""
    import hashlib

    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT job_id, posting_text FROM candidate_jobs "
        "WHERE jd_semantic_objects IS NOT NULL AND jd_semantic_objects != '' "
        "AND (posting_hash IS NULL OR posting_hash = '')"
    ).fetchall()

    updated = 0
    for row in rows:
        h = hashlib.sha256((row["posting_text"] or "").encode("utf-8")).hexdigest()[:16]
        conn.execute("UPDATE candidate_jobs SET posting_hash = ? WHERE job_id = ?", (h, row["job_id"]))
        updated += 1
    conn.commit()
    conn.close()
    return {"scanned": len(rows), "updated": updated}


def mark_dead_jobs(job_ids: list[str], db_path: Path = DB_PATH) -> int:
    """URL이 실제로 안 열리는(마감/삭제된) 공고를 candidate_jobs에서
    "죽은 링크"로 표시한다(DELETE가 아니라 UPDATE) - sync_jobs()는
    job_id가 이미 candidate_jobs에 있으면 그대로 건너뛰기만 하고 원본
    소스가 지운 공고를 스스로 걷어내지 않는다. 만약 여기서 행 자체를
    지우면, sync_jobs()가 "새 job"으로 착각해서 원본(market_analysis.db)
    에서 그대로 다시 살려낸다(실측으로 확인 - DELETE 대신 link_dead
    플래그로 바꾼 이유). list_all_candidate_jobs()가 link_dead=1인
    행은 결과에서 제외한다.

    실측(2026-07-14): candidate_jobs 1993건 중 174건(8.7%, 대부분
    특정 이커머스 - 마감된 Greenhouse 공고)이 실제로 열리지 않았다."""
    if not job_ids:
        return 0
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(job_ids))
    cur = conn.execute(
        f"UPDATE candidate_jobs SET link_dead = 1, link_checked_at = CURRENT_TIMESTAMP WHERE job_id IN ({placeholders})",
        job_ids,
    )
    marked = cur.rowcount
    conn.commit()
    conn.close()
    return marked


def mark_expired_jobs(today: date | None = None, db_path: Path = DB_PATH) -> dict:
    """마감일이 이미 지난 공고를 candidate_jobs에서 제외 표시한다
    (mark_dead_jobs()와 동일 원칙 - 하드 DELETE 대신 플래그. 이유도
    동일: 원본 DB에 그 job_id가 남아있는 한 sync_jobs()가 "신규 job"
    으로 착각해 되살려낸다).

    두 경로로 마감일을 본다:
      (1) posting_text 에서 찾은 실제 달력 날짜 (기존)
      (2) jobs.deadline 구조화 컬럼 (2026-08-30, A-1/A-6 - 사람인 리스트
          `.job_date`, GreetingHR `dueDate` 등 원천이 구조화해 주는 마감일.
          URL 로 candidate_jobs 와 연결)
    "상시채용"/"채용 시 마감"처럼 특정 날짜가 없거나 "D-18"처럼 수집 시점
    기준 상대값이라 지금 판단할 근거가 없는 공고는 만료로 보지 않는다.
    새 LLM 호출 없음 - 순수 텍스트/날짜 파싱."""
    from resume_input.job_prep import extract_deadline, is_deadline_passed

    if today is None:
        today = date.today()
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT job_id, url, source, posting_text FROM candidate_jobs "
        "WHERE (link_dead IS NULL OR link_dead = 0) AND (deadline_expired IS NULL OR deadline_expired = 0)"
    ).fetchall()

    # jobs.deadline 구조화 컬럼 (URL → 마감일 문자열)
    struct_deadline: dict[str, str] = {
        r[0]: (r[1] or "")
        for r in conn.execute(
            "SELECT url, deadline FROM jobs WHERE COALESCE(deadline, '') != '' AND COALESCE(url, '') != ''"
        ).fetchall()
    }

    expired_ids: list[str] = []
    checked = 0
    for row in rows:
        deadline_text = extract_deadline(row["posting_text"] or "")
        if deadline_text is None:
            # 구조화 컬럼 폴백. 단 원티드는 여기서 처리하지 않는다 -
            # lifecycle.mark_closed_wanted() 가 API status/hidden/due_time 으로
            # authoritative 하게 판단한다(원티드가 마감일을 지나도 status 를
            # active 로 유지하는 경우가 있어 표시 마감일만으론 오은퇴 위험).
            if (row["source"] or "") != "원티드":
                sd = struct_deadline.get(row["url"] or "")
                if sd and not any(k in sd for k in ("상시", "수시", "채용시", "충원")):
                    deadline_text = sd
        if deadline_text is None:
            continue
        checked += 1
        if is_deadline_passed(deadline_text, today):
            expired_ids.append(row["job_id"])

    if expired_ids:
        placeholders = ",".join("?" * len(expired_ids))
        conn.execute(
            f"UPDATE candidate_jobs SET deadline_expired = 1, deadline_checked_at = CURRENT_TIMESTAMP "
            f"WHERE job_id IN ({placeholders})",
            expired_ids,
        )
        conn.commit()
    conn.close()
    return {"scanned": len(rows), "with_parseable_deadline": checked, "newly_expired": len(expired_ids)}


def propagate_inactive_jobs(db_path: Path = DB_PATH) -> int:
    """`jobs.is_active = 0` (L-1 lifecycle - ATS snapshot diff / Wanted status
    로 은퇴된 공고)을 candidate_jobs 로 전파한다. candidate_jobs 는 URL 로
    jobs 와 연결된다(reference job 의 job_id 는 `url::{url}` 이라 직접 join 불가).

    새 LLM 호출 없음 - 순수 로컬 SQL. `list_all_candidate_jobs()` 가 매번
    호출하므로(mark_expired_jobs 와 같은 원칙) 일일 배치와 무관하게 항상
    최신 상태가 반영된다. 반환: 새로 은퇴 표시된 candidate 수."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """
            UPDATE candidate_jobs
               SET link_dead = 1, link_checked_at = CURRENT_TIMESTAMP
             WHERE (link_dead IS NULL OR link_dead = 0)
               AND url IN (
                   SELECT url FROM jobs
                    WHERE is_active = 0 AND COALESCE(url, '') != ''
               )
            """
        )
        n = cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def list_jobs_needing_career_reparse(db_path: Path = DB_PATH) -> list[str]:
    """career_parser_version이 현재 PARSER_VERSION과 다른(또는 없는)
    job_id 목록 - 파서가 개선된 뒤 "재분석이 필요한 job"을 식별하는
    용도. 지금은 조회만 하고 자동 재계산은 하지 않는다."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT job_id FROM candidate_jobs WHERE career_parser_version IS NULL OR career_parser_version != ?",
        (PARSER_VERSION,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def list_jobs_needing_representation(representation_version: str, db_path: Path = DB_PATH) -> list[dict]:
    """representation_version이 현재 버전과 다르거나(또는 없거나) 없는
    job 전체 dict 목록 - Understanding(jd_context/jd_short_semantic/
    jd_semantic_graph) 생성이 필요한 후보. career_level과 같은 원칙으로
    한 번 생성되면 재계산하지 않는다(버전이 바뀌기 전까지)."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candidate_jobs WHERE representation_version IS NULL OR representation_version != ?",
        (representation_version,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_job_representation(
    job_id: str, context: str, short_semantic_json: str, semantic_graph_json: str,
    representation_version: str, semantic_objects_json: str = "[]",
    relations_json: str = "[]", summary_json: str = "{}", posting_hash: str = "",
    regen_attempted: bool | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """LLM 1회 생성 결과를 저장한다(호출부가 이미 생성을 마친 뒤 호출).
    이 함수 자체는 LLM을 호출하지 않는다 - 순수 저장만.

    `posting_hash`(2026-07-16 추가): 생성 시점의 posting_text 해시를
    같이 저장한다 - `ensure_jd_semantic_objects()`가 재크롤링으로
    posting_text가 바뀐 job을 자동으로 재생성 대상으로 잡을 수 있게
    한다(job_id만으로는 내용 변경을 알 수 없음).

    `regen_attempted`(2026-07-17 추가): None이면 이 컬럼을 안 건드린다
    (기본 저장 경로). ensure_jd_semantic_objects()가 "불완전해 보여서
    재생성"한 경우에만 True를 넘겨서 semantic_object_regen_attempted를
    1로 표시한다 - 다음에도 계속 불완전하게 나오더라도 무한 재시도하지
    않기 위함(1회만 재시도)."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    if regen_attempted is None:
        conn.execute(
            """
            UPDATE candidate_jobs
            SET jd_context = ?, jd_short_semantic = ?, jd_semantic_graph = ?,
                jd_semantic_objects = ?, jd_relations = ?, jd_summary = ?,
                representation_version = ?, representation_generated_at = CURRENT_TIMESTAMP,
                posting_hash = ?
            WHERE job_id = ?
            """,
            (
                context, short_semantic_json, semantic_graph_json,
                semantic_objects_json, relations_json, summary_json,
                representation_version, posting_hash, job_id,
            ),
        )
    else:
        conn.execute(
            """
            UPDATE candidate_jobs
            SET jd_context = ?, jd_short_semantic = ?, jd_semantic_graph = ?,
                jd_semantic_objects = ?, jd_relations = ?, jd_summary = ?,
                representation_version = ?, representation_generated_at = CURRENT_TIMESTAMP,
                posting_hash = ?, semantic_object_regen_attempted = ?
            WHERE job_id = ?
            """,
            (
                context, short_semantic_json, semantic_graph_json,
                semantic_objects_json, relations_json, summary_json,
                representation_version, posting_hash, int(regen_attempted), job_id,
            ),
        )
    conn.commit()
    conn.close()


def save_job_context(job_id: str, context: str, db_path: Path = DB_PATH) -> None:
    """jd_context만 갱신한다(jd_short_semantic/jd_semantic_graph는 건드리지
    않음). Top30 LLM Judgment 직전에 Context가 비어있는 job만 지연 생성할 때
    쓴다 - 표시되지 않는 나머지 후보까지 Context를 미리 만들 필요는 없어서
    save_job_representation과 분리했다(그 함수는 3개 필드를 한번에 덮어써서
    이미 검증된 short_semantic/semantic_graph를 재호출로 흔들 수 있다)."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE candidate_jobs SET jd_context = ? WHERE job_id = ?", (context, job_id))
    conn.commit()
    conn.close()


def sync_jobs(db_path: Path = DB_PATH) -> dict:
    """원본 DB를 읽어 dedup한 뒤 candidate_jobs에 UPSERT한다. 이미
    있는 job_id는 career_level 등 계산된 필드를 건드리지 않고 건너뛴다
    (최초 계산 결과를 보존). 반환: 동기화 통계(신규/기존/전체 건수)."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    existing_ids = {row["job_id"] for row in conn.execute("SELECT job_id FROM candidate_jobs")}

    combined = list_active_jobs() + list_reference_jobs()
    deduped, removed = _dedup_jobs(combined)
    if removed:
        print(f"[job_store] 소스 간 중복 제거: {len(combined)}건 -> {len(deduped)}건 ({removed}건 제거)")

    new_count = 0
    for job in deduped:
        job_id = _dedup_key(job)
        if job_id in existing_ids:
            # career_level 등 "판단" 결과는 절대 재계산하지 않지만,
            # location처럼 원본 소스에 이미 있던 단순 필드는 컬럼 추가
            # 이전에 채워지지 않았을 수 있어 비어있으면 채워준다(판단
            # 아님 - 원본 데이터 그대로 옮기는 것).
            conn.execute(
                "UPDATE candidate_jobs SET location = ? WHERE job_id = ? AND (location IS NULL OR location = '')",
                (job.get("location", "") or "", job_id),
            )
            continue

        title = job.get("title", "") or ""
        posting_text = normalize_invisible_chars(job.get("posting_text", "") or "")
        # 원천 구조화 경력(jobs.experience_level)을 authoritative 로 우선한다 (2026-08-30)
        career_level, confidence = extract_jd_career_level(
            title, posting_text, job.get("experience_level", "") or "")

        conn.execute(
            """
            INSERT INTO candidate_jobs
                (job_id, title, company, url, source, origin, posting_text,
                 career_level, career_level_confidence, career_parser_version, career_parsed_at, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """,
            (
                job_id, title, job.get("company", ""), job.get("url", ""),
                detect_job_source(job.get("url", "")), job.get("source", ""),
                posting_text, career_level, confidence, PARSER_VERSION,
                job.get("location", "") or "",
            ),
        )
        new_count += 1

    conn.commit()
    conn.close()
    return {"total_candidates": len(deduped), "new": new_count, "already_synced": len(deduped) - new_count}


_REFRESH_MIN_TEXT_LEN = 1000  # recrawl_full_backfill.py의 _MIN_TEXT_LEN과 동일 기준


def refresh_stale_postings(db_path: Path = DB_PATH) -> dict:
    """일회성/주기적 갱신(2026-07-16 추가) - `sync_jobs()`는 신규 job만
    추가하고 기존 행의 `posting_text`는 의도적으로 절대 안 건드린다
    (career_level 등 계산 결과가 재계산으로 흔들리지 않게 하려는
    설계). 그런데 이러면 재크롤링으로 원본 크롤러가 아무리 개선돼도
    (예: 오늘 고친 JSON-LD 파싱 버그) 이미 동기화된 후보 풀에는 영원히
    반영이 안 되는 문제가 있다 - 오늘 재크롤링 후 실측으로 확인됨.

    원본 `posting_text`가 최소 길이(`_REFRESH_MIN_TEXT_LEN`, 아래 참고)
    이상이면 갱신한다. `posting_text`가 바뀌면 그 안의 경력 정보도
    달라질 수 있으므로 `career_level`도 같은 시점에 다시 계산한다
    (posting_hash 기반 JD Semantic Object 재생성과 동일한 원칙).

    2026-07-16 수정 - 원래는 "기존 저장값보다 더 길 때만" 갱신했으나,
    이게 정확히 오늘 고친 버그의 최대 수혜 케이스를 스스로 걸러내는
    부작용이 있었다: 기존 저장값이 HTML 태그가 안 걸러진 오염 텍스트인
    경우, 태그 바이트 수 때문에 오히려 "더 길어" 보여서 깨끗하게
    재파싱된 (더 짧은) 새 본문이 계속 거부됐다(실측: 특정 공고
    - 새 본문 2354자/HTML 없음 vs 기존 저장값
    5561자/HTML 오염 - 계속 거부되고 있었음). 원본 크롤러 쪽에는 이미
    "본문이 아예 비었으면 SUCCESS 처리 안 함"(detail_collector.py) +
    "재수집 결과가 최소 길이(1000자) 미만이면 반영 안 함"
    (recrawl_full_backfill.py의 upsert_market_db) 가드가 있으므로, 여기
    도달한 시점의 새 본문은 이미 "성공적으로 수집된" 것으로 신뢰하고
    절대 길이 기준으로만 판단한다.

    2026-07-16 추가 수정 - "성공적으로 수집됨"을 신뢰하는 것과, 그
    본문이 "이미 정규화됐다"고 신뢰하는 것은 다른 문제였다. 오늘
    normalize_invisible_chars()를 detail_parser.py에 추가했지만, 원본
    da_job_market_2026 DB 자체는 예전에 수집된 행까지 소급 정규화하지
    않았다 - 그 결과 이 함수가 원본에서 그대로 끌어온 텍스트가 이미
    한 번 정리했던 candidate_jobs 값을 다시 오염시켰다(특정 제조업 공고 등
    재확인됨). 원본을 신뢰하지 않고 저장 직전에 항상 정규화한다."""
    _init_table(db_path)
    combined = list_active_jobs() + list_reference_jobs()
    deduped, _ = _dedup_jobs(combined)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    existing = {
        row["job_id"]: row["posting_text"]
        for row in conn.execute("SELECT job_id, posting_text FROM candidate_jobs")
    }

    updated = 0
    for job in deduped:
        job_id = _dedup_key(job)
        old_text = existing.get(job_id)
        if old_text is None:
            continue  # 신규 job - sync_jobs()가 처리할 대상, 여기서 안 건드림
        new_text = normalize_invisible_chars(job.get("posting_text", "") or "")
        if len(new_text) < _REFRESH_MIN_TEXT_LEN:
            continue

        career_level, confidence = extract_jd_career_level(
            job.get("title", "") or "", new_text, job.get("experience_level", "") or "")
        conn.execute(
            """
            UPDATE candidate_jobs SET
                posting_text = ?, career_level = ?, career_level_confidence = ?,
                career_parser_version = ?, career_parsed_at = CURRENT_TIMESTAMP,
                location = COALESCE(NULLIF(?, ''), location)
            WHERE job_id = ?
            """,
            (new_text, career_level, confidence, PARSER_VERSION, job.get("location", "") or "", job_id),
        )
        updated += 1

    conn.commit()
    conn.close()
    return {"scanned": len(deduped), "updated": updated}


def list_all_candidate_jobs(db_path: Path = DB_PATH) -> list[dict]:
    """candidate_jobs 테이블을 반환한다. 매번 호출 시 sync_jobs()로 원본
    DB(job_ai_v3 자체 jobs 테이블 + da_job_market_2026 참조 코퍼스) 대비
    새로 생긴 공고만 가져온다(2026-07-14 변경 - 이전엔 테이블이 비어있을
    때 딱 1회만 동기화해서, 원본이 나중에 갱신돼도 후보 풀이 그 시점에
    영원히 고정되는 문제가 있었다). sync_jobs()는 이미 있는 job_id는
    건드리지 않고 새 job_id만 추가하므로(career_level 등 계산 결과를
    보존) 매번 불러도 안전하다 - LLM 호출 없이 SQLite 읽기 + 신규분
    dedup만 수행한다. link_dead=1로 표시된(mark_dead_jobs 참고 - URL이
    실제로 안 열려서 걸러진) 공고와 deadline_expired=1로 표시된
    (mark_expired_jobs 참고 - 마감일이 이미 지난) 공고는 여기서
    제외한다.

    mark_expired_jobs()도 매번 같이 실행한다(2026-08-14 추가 - 이전엔
    이 함수를 자동으로 호출하는 경로가 코드 어디에도 없어서 2026-07-17
    수동 실행 이후로 한 번도 안 돌았고, 그 사이 마감일이 지난 공고가
    목록에 계속 남아있던 버그가 실측으로 확인됨). LLM 호출 없이
    posting_text의 이미 저장된 텍스트를 정규식으로 재파싱만 하므로
    매번 불러도 안전하다(sync_jobs와 같은 원칙)."""
    _init_table(db_path)
    from resume_input.runtime_mode import IS_DEMO
    if not IS_DEMO:
        # 데모 모드에서는 raw_postings(=jobs 테이블) 자체가 없다 - 공고
        # 수집이 비활성화된 데모에서는 candidate_jobs 를 그대로 읽기만
        # 하고, 실제 수집기가 채우는 원본 테이블과의 동기화는 건너뛴다.
        sync_jobs(db_path)
        mark_expired_jobs(db_path=db_path)
        propagate_inactive_jobs(db_path=db_path)  # L-1: jobs.is_active=0 → candidate 은퇴

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candidate_jobs WHERE (link_dead IS NULL OR link_dead = 0) "
        "AND (deadline_expired IS NULL OR deadline_expired = 0)"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_company_research(company_name: str, db_path: Path = DB_PATH) -> dict | None:
    """"회사 알아보기" 모달용 Company Research 조회(2026-07-25, 반자동
    리서치 - WebSearch + jd_semantic_objects culture-layer DB 집계로
    채움, LLM 호출 없음). `company_research` 테이블에 해당 회사명이
    없으면 None(아직 리서치 안 된 회사 - 호출부가 폴백 처리)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT data_json FROM company_research WHERE company_name = ?", (company_name,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row["data_json"])
    except (TypeError, ValueError):
        return None


def _init_resume_understanding_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_understanding_cache (
            resume_hash TEXT PRIMARY KEY,
            context TEXT,
            short_semantic TEXT,
            semantic_graph TEXT,
            representation_version TEXT,
            generated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # 2026-07-16: Semantic Object 스키마 재설계(체크포인트 1,
    # docs/semantic_object_schema.md) - semantic_objects/relations/summary
    # 컬럼 추가. short_semantic/semantic_graph는 M3+M4가 대체되는
    # 체크포인트 3까지 과도기 호환용으로 계속 채워진다(새 데이터에서
    # 파생, 별도 LLM 호출 없음).
    for col in ("semantic_objects", "relations", "summary"):
        try:
            conn.execute(f"ALTER TABLE resume_understanding_cache ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def get_resume_understanding(resume_hash: str, db_path: Path = DB_PATH) -> dict | None:
    """이력서 Understanding 캐시 조회 - 같은 이력서(해시 동일)면 재생성
    없이 재사용한다. 이력서가 바뀌면 해시가 달라지므로 자연스럽게 새로
    생성된다(별도의 무효화 로직 불필요)."""
    _init_resume_understanding_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM resume_understanding_cache WHERE resume_hash = ?", (resume_hash,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_resume_understanding(
    resume_hash: str, context: str, short_semantic_json: str, semantic_graph_json: str,
    representation_version: str, semantic_objects_json: str = "[]",
    relations_json: str = "[]", summary_json: str = "{}", db_path: Path = DB_PATH,
) -> None:
    _init_resume_understanding_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO resume_understanding_cache
            (resume_hash, context, short_semantic, semantic_graph,
             semantic_objects, relations, summary, representation_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(resume_hash) DO UPDATE SET
            context = excluded.context, short_semantic = excluded.short_semantic,
            semantic_graph = excluded.semantic_graph, semantic_objects = excluded.semantic_objects,
            relations = excluded.relations, summary = excluded.summary,
            representation_version = excluded.representation_version,
            generated_at = CURRENT_TIMESTAMP
        """,
        (
            resume_hash, context, short_semantic_json, semantic_graph_json,
            semantic_objects_json, relations_json, summary_json, representation_version,
        ),
    )
    conn.commit()
    conn.close()


def _init_judgment_cache_table(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS judgment_cache (
            resume_hash TEXT NOT NULL,
            job_id TEXT NOT NULL,
            meaning_fit_score INTEGER,
            reasoning TEXT,
            evidence TEXT,
            key_reasons TEXT,
            judgment_version TEXT,
            generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (resume_hash, job_id)
        )
        """
    )
    # 2026-07-15: comparisons/summary_bullets 컬럼 추가(구조화 비교표 +
    # 추상 요약). 기존 DB에는 없을 수 있으니 ALTER로 보강한다 - 이미
    # 있으면 sqlite3.OperationalError를 그냥 무시한다.
    for col in ("comparisons", "summary_bullets"):
        try:
            conn.execute(f"ALTER TABLE judgment_cache ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def get_cached_judgments(resume_hash: str, job_ids: list[str], judgment_version: str, db_path: Path = DB_PATH) -> dict[str, dict]:
    """(resume_hash, job_id) 조합으로 Top30 LLM Meaning Judgment 결과를
    캐시 조회한다. judgment_version이 다르면(프롬프트가 바뀐 경우) 캐시
    미스로 취급해서 재호출하게 만든다 - representation_version과 같은
    원칙(career_level의 "1회 계산, 버전 바뀌면 재계산" 패턴을 여기에도
    적용). 같은 이력서로 같은 공고를 다시 보면(앱 재시작/새 세션 포함)
    LLM을 다시 부르지 않는다."""
    if not job_ids:
        return {}
    _init_judgment_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(job_ids))
    rows = conn.execute(
        f"""
        SELECT * FROM judgment_cache
        WHERE resume_hash = ? AND judgment_version = ? AND job_id IN ({placeholders})
        """,
        (resume_hash, judgment_version, *job_ids),
    ).fetchall()
    conn.close()

    result = {}
    for r in rows:
        result[r["job_id"]] = {
            "meaning_fit_score": r["meaning_fit_score"],
            "reasoning": r["reasoning"] or "",
            "evidence": json.loads(r["evidence"]) if r["evidence"] else [],
            "key_reasons": json.loads(r["key_reasons"]) if r["key_reasons"] else [],
            "comparisons": json.loads(r["comparisons"]) if r["comparisons"] else [],
            "summary_bullets": json.loads(r["summary_bullets"]) if r["summary_bullets"] else [],
        }
    return result


def save_judgment(
    resume_hash: str, job_id: str, meaning_fit_score: int | None, reasoning: str,
    evidence_json: str, key_reasons_json: str, judgment_version: str,
    comparisons_json: str = "[]", summary_bullets_json: str = "[]", db_path: Path = DB_PATH,
) -> None:
    _init_judgment_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO judgment_cache
            (resume_hash, job_id, meaning_fit_score, reasoning, evidence, key_reasons,
             comparisons, summary_bullets, judgment_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(resume_hash, job_id) DO UPDATE SET
            meaning_fit_score = excluded.meaning_fit_score, reasoning = excluded.reasoning,
            evidence = excluded.evidence, key_reasons = excluded.key_reasons,
            comparisons = excluded.comparisons, summary_bullets = excluded.summary_bullets,
            judgment_version = excluded.judgment_version, generated_at = CURRENT_TIMESTAMP
        """,
        (
            resume_hash, job_id, meaning_fit_score, reasoning, evidence_json, key_reasons_json,
            comparisons_json, summary_bullets_json, judgment_version,
        ),
    )
    conn.commit()
    conn.close()


def list_recent_judgments(resume_hash: str, limit: int = 2, db_path: Path = DB_PATH) -> list[dict]:
    """홈 대시보드 "최근 분석한 공고" 카드용 - 이 이력서로 Top30 Meaning
    Judgment가 실제로 실행된 공고 중 최신순 N개. judgment_cache는 이미
    실행된 LLM 판단 결과를 저장해두는 캐시라, 여기서는 조회만 한다(새
    LLM 호출 없음 - 표시용 데이터를 지어내지 않고 실제 저장된 판단
    이력만 보여준다)."""
    _init_judgment_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT jc.job_id, jc.meaning_fit_score, jc.generated_at,
               cj.company, cj.title
        FROM judgment_cache jc
        LEFT JOIN candidate_jobs cj ON cj.job_id = jc.job_id
        WHERE jc.resume_hash = ?
        ORDER BY jc.generated_at DESC
        LIMIT ?
        """,
        (resume_hash, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _init_semantic_link_cache_table(db_path: Path = DB_PATH) -> None:
    """Semantic Linking(link_engine.run_semantic_linking(), LLM 1회) 결과
    캐시(2026-08-16, 사용자 확정) - 성능 최적화가 아니라 판단 일관성
    보장이 목적이다. ②~⑤ Customization Planner가 전부 이 Linking
    결과(Analysis 5-state의 원천)를 공통 입력으로 쓰므로, 세션이 바뀔
    때마다 Linking이 재실행되어 다른 A/B/C/D가 나오면 같은 공고인데도
    맞춤화 판단이 날마다 달라진다 - 이게 실측으로 확인됐다(judgment_cache
    와 달리 Linking은 지금까지 DB에 전혀 저장되지 않고 있었다).

    Linking은 JD 단독 결과가 아니라 JD × Resume 결과라 job_id 하나로는
    식별이 안 된다 - resume_hash까지 키에 포함한다(이력서가 바뀌면
    cache miss). linking_version(link_engine.LINK_SCHEMA_VERSION)도 키에
    포함해서 프롬프트/로직이 바뀌면 예전 결과를 계속 쓰지 않게 한다.

    candidate_jobs에 컬럼을 추가하지 않고 별도 테이블로 둔 이유 - 한
    공고에 이력서 버전이 여러 개 대응될 수 있어(resume_hash가 곧
    row 단위 식별자), 공고당 1개 값만 담는 컬럼 방식이 안 맞는다."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_link_cache (
            job_id TEXT NOT NULL,
            resume_hash TEXT NOT NULL,
            linking_version TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (job_id, resume_hash, linking_version)
        )
        """
    )
    conn.commit()
    conn.close()


def get_cached_semantic_link(
    job_id: str, resume_hash: str, linking_version: str, db_path: Path = DB_PATH,
) -> dict | None:
    """(job_id, resume_hash, linking_version)이 전부 일치할 때만 저장된
    link_result를 반환한다 - 셋 중 하나라도 다르면 None(cache miss,
    호출부가 link_engine을 다시 돌려야 한다는 뜻)."""
    _init_semantic_link_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT result_json FROM semantic_link_cache
        WHERE job_id = ? AND resume_hash = ? AND linking_version = ?
        """,
        (job_id, resume_hash, linking_version),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row["result_json"])


def save_semantic_link_result(
    job_id: str, resume_hash: str, linking_version: str, result: dict, db_path: Path = DB_PATH,
) -> None:
    _init_semantic_link_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO semantic_link_cache (job_id, resume_hash, linking_version, result_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(job_id, resume_hash, linking_version) DO UPDATE SET
            result_json = excluded.result_json, created_at = CURRENT_TIMESTAMP
        """,
        (job_id, resume_hash, linking_version, json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def list_recent_analyses(resume_hash: str, limit: int = 5, db_path: Path = DB_PATH) -> list[dict]:
    """홈 대시보드 "최근 확인/분석한 공고"용 - 이 이력서로 "분석하기"가
    실제 실행돼 semantic_link_cache 에 결과가 저장된 공고를 최신순 N개.
    조회만 한다(새 LLM/판단 없음, 표시용 데이터를 지어내지 않는다).
    result_json 은 linking_version 에 따라 quick_analysis 결과(dict) 또는
    구 link_engine 결과 - 호출부가 필요하면 파싱한다."""
    _init_semantic_link_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT s.job_id, s.linking_version, s.result_json, s.created_at,
               cj.company, cj.title, cj.url
        FROM semantic_link_cache s
        LEFT JOIN candidate_jobs cj ON cj.job_id = s.job_id
        WHERE s.resume_hash = ?
        ORDER BY s.created_at DESC
        LIMIT ?
        """,
        (resume_hash, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_analyses(resume_hash: str, db_path: Path = DB_PATH) -> tuple[int, int]:
    """(이 이력서로 분석한 공고 수, 그중 오늘 분석한 수) - 홈 대시보드 카드용."""
    _init_semantic_link_cache_table(db_path)
    conn = sqlite3.connect(db_path)
    total = conn.execute(
        "SELECT COUNT(DISTINCT job_id) FROM semantic_link_cache WHERE resume_hash = ?",
        (resume_hash,),
    ).fetchone()[0]
    today = conn.execute(
        "SELECT COUNT(DISTINCT job_id) FROM semantic_link_cache "
        "WHERE resume_hash = ? AND date(created_at) = date('now', 'localtime')",
        (resume_hash,),
    ).fetchone()[0]
    conn.close()
    return int(total), int(today)


def get_candidate_job(job_id: str, db_path: Path = DB_PATH) -> dict | None:
    """job_id 단건 조회 - 관심공고(saved_jobs)처럼 job_id만 들고 있는
    화면에서 전체 job dict(posting_text 등)가 필요할 때 쓴다."""
    _init_table(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM candidate_jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
