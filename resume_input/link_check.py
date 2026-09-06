"""
resume_input/link_check.py

추천 화면에 실제로 노출될 공고의 원문 링크가 그 순간에 진짜로 열리는지
확인한다. candidate_jobs 캐시에 저장된 URL은 수집 시점엔 살아있었어도
그 뒤로 채용이 마감/삭제됐을 수 있다(실측 2026-07-14: 1993건 중 174건,
8.7%가 이미 죽어있었음 - 대부분 마감된 특정 이커머스 Greenhouse 공고).

전체 풀을 매번 검사하면 느리다(1993건에 약 4분) - 대신 "실제로 화면에
보여줄 후보"만 요청 시점에 검사한다. 죽은 링크가 나오면 순위표의 다음
후보로 채워서 최종적으로 보여줄 개수(target_n)를 항상 채운다. 검사
과정에서 확인된 죽은 링크는 job_store.mark_dead_jobs()로 영구 기록해서
다음 요청부터는 그 공고를 아예 후보에서 제외한다(매번 다시 검사하지
않도록 - career_level/representation과 같은 "한 번 확인하면 재사용"
원칙).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

_DEAD_TEXT_MARKERS = [
    "position has been filled", "채용이 마감", "공고가 마감", "posting has expired",
    "no longer accepting applications", "채용이 종료", "페이지를 찾을 수 없습니다",
    "page not found", "이 채용은 마감", "job not found",
]


def is_link_alive(url: str, timeout: int = 8) -> bool:
    """URL이 실제로 열리는지 확인. 네트워크 오류/타임아웃/404 이상 상태
    코드/명백한 "마감" 문구는 전부 죽은 것으로 취급한다 - 애매하면
    (예: 로그인 필요 페이지처럼 판단 불가) 살아있는 것으로 간주해서
    잘못 걸러내지 않는다(과도한 제외보다 과소 제외가 안전)."""
    if not url:
        return False
    try:
        r = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout, allow_redirects=True)
        if r.status_code >= 400:
            return False
        low = r.text.lower()
        return not any(marker in low for marker in _DEAD_TEXT_MARKERS)
    except Exception:
        return False


def filter_live_links(ranked_jobs: list[dict], target_n: int, max_workers: int = 20) -> list[dict]:
    """ranked_jobs(이미 순위가 매겨진 상태, 앞에서부터 우선)를 순서대로
    검사해서 target_n개의 "실제로 열리는" 공고만 순위를 유지한 채
    반환한다. ranked_jobs가 target_n보다 넉넉히 큰 풀로 주어져야
    죽은 링크를 대체할 다음 후보가 있다(호출부가 버퍼를 두고 넘겨야
    함)."""
    from resume_input import job_store
    from resume_input.runtime_mode import IS_DEMO

    if IS_DEMO:
        # 데모 데이터의 URL은 전부 example.com 자리표시자라 실제로 열어볼
        # 대상이 아니고, 외부로 나가는 요청 자체를 만들지 않기 위해 스킵한다.
        return ranked_jobs[:target_n]

    if len(ranked_jobs) <= target_n:
        candidates = ranked_jobs
    else:
        candidates = ranked_jobs

    live: list[dict] = []
    dead_job_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_job = {ex.submit(is_link_alive, j.get("url", "")): j for j in candidates}
        alive_map: dict[str, bool] = {}
        for fut in as_completed(future_to_job):
            job = future_to_job[fut]
            try:
                alive_map[job["job_id"]] = fut.result()
            except Exception:
                alive_map[job["job_id"]] = False

    for j in candidates:
        if alive_map.get(j["job_id"]):
            live.append(j)
        else:
            dead_job_ids.append(j["job_id"])
        if len(live) >= target_n:
            break

    if dead_job_ids:
        job_store.mark_dead_jobs(dead_job_ids)

    return live
