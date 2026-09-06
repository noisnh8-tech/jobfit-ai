"""
resume_input/checklist_engine.py

Checklist Rule Engine(2026-08-03, v2 - Command 기반) - "지원 전
체크포인트"를 만든다. LLM 호출 없음. JD를 다시 해석하지 않고, Resume를
다시 해석하지 않고, Semantic Link/Judge를 다시 판단하지 않는다 - 이미
만들어진 정보를 정리해서 보여주는 역할만 한다.

- submission_documents/application_process/job_requirements: 새로
  추출하지 않는다 - `job_prep.extract_application_prep()`이 이미 만들어둔
  값(정규식 기반, LLM 미사용, 2026-07-16~31에 걸쳐 실측 검증된 로직)을
  그대로 가져온다.
- resume_actions: `resume_customizing.build_resume_commands()`의 Command
  목록을 받아, **문장을 새로 생성하지 않고** Command 종류를 그대로
  서술하는 짧은 텍스트로만 바꾼다(move_project/move_bullet는 기계적
  사실 서술이라 안전하지만, rewrite_sentence는 아직 문장 품질이
  검증되지 않았다 - resume_customizing.py 상단 docstring 참고 - 그래서
  "이 문장을 자동 적용"이 아니라 "검토가 필요하다"는 안내로만 표시한다).

같은 입력이면 항상 같은 결과다(Deterministic)."""
from __future__ import annotations

from resume_input.job_prep import extract_application_prep

_JOB_INFO_LABEL = {
    "급여": "급여", "근무 형태": "근무형태", "학력 조건": "학력 조건",
    "모집 인원": "모집인원", "근무 지역": "근무지역",
}


def _describe_commands(commands: list[dict]) -> list[str]:
    """Command 목록을 사람이 보는 짧은 텍스트로 바꾼다 - 새 문장을
    만드는 게 아니라 Command 자체(무엇을, 어디로)를 그대로 서술한다.
    같은 project_id/bullet_id를 가리키는 Command가 여러 JD 요구사항에서
    나올 수 있어 (project_id, type) / (bullet_id, type) 기준으로
    중복 제거한다(가장 우선순위 높은 것만 남김 - resume_customizing.py의
    priority 정렬을 그대로 신뢰)."""
    seen: set[tuple] = set()
    lines: list[str] = []
    for c in commands:
        if c["type"] == "move_project":
            key = ("move_project", c["project_id"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"'{c['project_id']}' 프로젝트를 이력서 상단으로 이동 필요")
        elif c["type"] == "move_bullet":
            key = ("move_bullet", c["project_id"], c["bullet_id"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"'{c['project_id']}' 프로젝트 내 관련 항목의 순서 조정 필요")
        elif c["type"] == "highlight_keyword":
            key = ("highlight_keyword", c["target_resume_object_id"])
            if key in seen or not c.get("keywords"):
                continue
            seen.add(key)
            lines.append(f"관련 항목에서 다음 키워드를 강조하세요: {', '.join(c['keywords'])}")
    return lines


def build_checklist(job: dict, resume_commands: list[dict]) -> dict:
    """job(posting_text 포함)과 build_resume_commands()의 결과를 받아
    Checklist 4개 섹션을 만든다. 공고에 없는 항목은 생성하지 않는다."""
    prep = extract_application_prep(job)

    submission_documents = [{"required": True, "name": doc} for doc in prep["required_documents"]]
    application_process = prep["application_process"]

    job_requirements = []
    if prep["deadline"]:
        job_requirements.append(f"마감일: {prep['deadline']}")
    for key, label in _JOB_INFO_LABEL.items():
        if key in prep["job_info"]:
            job_requirements.append(f"{label}: {prep['job_info'][key]}")

    resume_actions = _describe_commands(resume_commands)

    return {
        "submission_documents": submission_documents,
        "application_process": application_process,
        "job_requirements": job_requirements,
        "resume_actions": resume_actions,
    }
