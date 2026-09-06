"""
resume_input/customizer.py

customization_rules.py가 만든 판단 결과(변경/유지/직접확인필요)를
원본 이력서 텍스트에 코드로 적용해서 "커스터마이징 완료 데이터"를
만든다. 여기서는 아무 판단도 하지 않는다(무엇을 바꿀지는 customization_
rules.py가 이미 정함) - 이 파일은 순수하게 "이미 정해진 변경을 원본
텍스트 구조에 반영"만 한다. LLM 호출 없음.

원본 텍스트 파싱(프로젝트 블록 -> 섹션 -> 불릿 단위)도 이 파일이
담당한다 - customization_rules.py가 "어느 프로젝트/어느 불릿이 JD와
연결되는지" 판단하려면 이 파싱 결과가 먼저 있어야 하므로, 판단 쪽에서
이 파일의 파싱 함수를 가져다 쓴다(customization_rules.py -> customizer.py
단방향 의존).

2026-07-21(Rule Executor 구현, docs/architecture_rule_executor_v1.md
참고) - Rule Executor(customization_rules.py)가 APPROVED_TERM_MAP까지
그 자리에서 확인해서 change/keep을 최종 확정한 뒤 넘긴다(후처리
단계 없음) - 이 파일은 이미 결정된 replacements를 그대로 적용만
한다(판단 없음 원칙 그대로 유지).
"""
from __future__ import annotations

import re

_PROJECT_HEADER_RE = re.compile(r"^(?P<title>.+?)\s*\|\s*[\d.]+.*~.*\|.*$", re.MULTILINE)
_PROJECT_SECTION_START = "• 프로젝트"
_PROJECT_SECTION_END = "• 대외활동"
_SKILL_SECTION_START = "[보유 기술]"

# 프로젝트 블록 내부의 "N) 제목" 서브섹션 헤더(예: "3) 해결 과정 및 역할").
_SUBSECTION_HEADER_RE = re.compile(r"^\d\)\s*(.+)$", re.MULTILINE)
# 불릿 줄(다음 불릿/헤더 전까지 이어지는 줄바꿈 문장도 같은 불릿으로 묶는다).
_BULLET_START_RE = re.compile(r"^•\s*")


def _split_projects(resume_raw: str) -> tuple[str, list[tuple[str, str]], str]:
    """(preamble, [(title, block_text), ...], postamble) 반환. 원문 내용 불변."""
    start = resume_raw.find(_PROJECT_SECTION_START)
    if start == -1:
        return resume_raw, [], ""
    end = resume_raw.find(_PROJECT_SECTION_END, start)
    if end == -1:
        end = len(resume_raw)

    preamble = resume_raw[: start + len(_PROJECT_SECTION_START)]
    projects_text = resume_raw[start + len(_PROJECT_SECTION_START) : end]
    postamble = resume_raw[end:]

    headers = list(_PROJECT_HEADER_RE.finditer(projects_text))
    if not headers:
        return preamble + projects_text, [], postamble

    blocks: list[tuple[str, str]] = []
    for i, m in enumerate(headers):
        block_start = m.start()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(projects_text)
        title = m.group("title").strip()
        blocks.append((title, projects_text[block_start:block_end]))

    lead = projects_text[: headers[0].start()]
    return preamble + lead, blocks, postamble


def _reorder_projects(resume_raw: str, project_order: list[str]) -> str:
    """project_order: 프로젝트 title 리스트(원본에 있는 문자열 그대로,
    또는 부분 일치). 블록 전체(제목+기간+기술+본문)를 통째로 이동만
    한다 - 내용은 한 글자도 안 바꾼다."""
    preamble, blocks, postamble = _split_projects(resume_raw)
    if not blocks or not project_order:
        return resume_raw

    remaining = list(blocks)
    ordered: list[tuple[str, str]] = []
    for title in project_order:
        for b in list(remaining):
            if title.lower() in b[0].lower() or b[0].lower() in title.lower():
                ordered.append(b)
                remaining.remove(b)
                break
    ordered.extend(remaining)  # 지정 안 된 나머지는 원래 순서 유지, 맨 뒤로

    return preamble + "".join(block for _, block in ordered) + postamble


def split_project_subsections(project_block: str) -> list[tuple[str, str]]:
    """프로젝트 블록 하나를 "N) 제목" 서브섹션 단위로 나눈다. 반환:
    [(섹션제목, 섹션본문), ...] - 헤더 앞부분(제목|기간|팀, [기술] 줄)은
    포함 안 함(재배치 대상이 아니므로)."""
    headers = list(_SUBSECTION_HEADER_RE.finditer(project_block))
    if not headers:
        return []
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(headers):
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(project_block)
        sections.append((m.group(1).strip(), project_block[body_start:body_end]))
    return sections


def split_bullets(section_body: str) -> list[str]:
    """서브섹션 본문을 "•"로 시작하는 불릿 단위로 나눈다. 불릿 안에서
    줄바꿈된 이어지는 문장(다음 불릿이 아닌 줄)은 같은 불릿에 붙인다.
    "•"가 없는(불릿형이 아닌) 섹션은 빈 리스트를 반환한다(재배치 대상
    아님 - 목적/문제정의처럼 통짜 문단인 섹션)."""
    lines = [ln for ln in section_body.split("\n")]
    bullets: list[str] = []
    current: list[str] = []
    for ln in lines:
        if _BULLET_START_RE.match(ln.strip()):
            if current:
                bullets.append("\n".join(current))
            current = [ln]
        elif ln.strip():
            if current:
                current.append(ln)
    if current:
        bullets.append("\n".join(current))
    return bullets


def _replace_subsection_bullets(project_block: str, section_title: str, new_bullet_order: list[str]) -> str:
    """project_block 안의 section_title 서브섹션 본문을, 기존 불릿을
    new_bullet_order 순서로 재배열한 텍스트로 통째로 교체한다. 불릿
    내용 자체는 한 글자도 안 바꾼다 - 순서만 바뀐다."""
    headers = list(_SUBSECTION_HEADER_RE.finditer(project_block))
    for i, m in enumerate(headers):
        if m.group(1).strip() != section_title:
            continue
        body_start = m.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(project_block)
        new_body = "\n" + "\n".join(new_bullet_order) + "\n"
        return project_block[:body_start] + new_body + project_block[body_end:]
    return project_block


def _split_skill_lines(resume_raw: str) -> tuple[str, list[str], str]:
    start = resume_raw.find(_SKILL_SECTION_START)
    if start == -1:
        return resume_raw, [], ""
    body_start = start + len(_SKILL_SECTION_START)
    end = resume_raw.find(_PROJECT_SECTION_START, body_start)
    if end == -1:
        end = len(resume_raw)

    preamble = resume_raw[:body_start]
    body = resume_raw[body_start:end]
    postamble = resume_raw[end:]

    lines = [ln for ln in body.split("\n")]
    return preamble, lines, postamble


def _reorder_skills(resume_raw: str, skill_order: list[str]) -> str:
    preamble, lines, postamble = _split_skill_lines(resume_raw)
    if not lines or not skill_order:
        return resume_raw

    non_empty = [ln for ln in lines if ln.strip()]
    remaining = list(non_empty)
    ordered: list[str] = []
    for skill in skill_order:
        for ln in list(remaining):
            name = ln.split(":", 1)[0].lower()
            if skill.lower() in name:
                ordered.append(ln)
                remaining.remove(ln)
                break
    ordered.extend(remaining)

    return preamble + "\n" + "\n".join(ordered) + "\n" + postamble


_MATCHABLE_LAYERS = ("problem", "thinking", "task", "skill", "qualification")


def _obj_id(obj: dict) -> str:
    return obj.get("id") or obj.get("normalized_text") or ""


def build_structured_resume(resume_raw: str, resume_objects: list[dict]) -> dict:
    """Link Engine이 "이 Task가 실제 이력서의 어느 문장/불릿인지" 바로
    찾을 수 있도록, 원문을 문장/불릿 단위로 나누고 각 Semantic Object의
    evidence.source_text를 문자열 포함 검사로 매칭해 참조(refs)를
    붙인다(2026-07-23, 사용자 확정). **새 LLM 호출 없음** - 원문 분리는
    기존 결정적 파서(_split_projects/split_project_subsections/
    split_bullets)를 그대로 쓰고, ref는 이미 확정된 evidence.source_text
    (원문 그대로 인용이어야 한다는 제약이 이미 있음 - _RESUME_PROMPT
    참고)를 원문에서 찾아 연결한다.

    LLM에게 "문장을 그대로 분리해라"를 새로 시키지 않는 이유: 같은
    "원문 그대로 인용" 요구를 이미 evidence.source_text에 걸어놨는데도
    LLM이 여러 문장을 조합해 재구성한 사례가 실측으로 확인된 바 있다
    (understanding.py의 _RESUME_PROMPT 주석, 2026-07-18). 결정적 파서 +
    문자열 매칭이면 이 위험이 원천적으로 없다.

    반환: {"summary": [{"sentence_id", "text", "refs": [object_id,...]}],
    "projects": [{"project_id": <프로젝트 제목 원문 - evidence.project와
    동일한 값>, "bullets": [{"bullet_id", "text", "refs": [...]}]}]}.
    refs는 레이어별로 나누지 않는다 - object_id 자체가 레이어를 담고
    있어(예: "task_001") 필요하면 접두어로 걸러 쓰면 된다(같은 정보를
    두 형태로 중복 저장하지 않는다)."""
    matchable = [o for o in resume_objects if o.get("layer") in _MATCHABLE_LAYERS]

    def _find_refs(text: str) -> list[str]:
        return [
            _obj_id(o) for o in matchable
            if (o.get("evidence") or {}).get("source_text") and (o["evidence"]["source_text"] in text)
        ]

    headline = extract_current_headline(resume_raw)
    summary = [{"sentence_id": "s1", "text": headline, "refs": _find_refs(headline)}] if headline else []

    _, blocks, _ = _split_projects(resume_raw)
    projects = []
    for title, block in blocks:
        bullets = []
        idx = 0
        for _section_title, section_body in split_project_subsections(block):
            for bullet_text in split_bullets(section_body):
                idx += 1
                bullets.append({
                    "bullet_id": f"b{idx}", "text": bullet_text, "refs": _find_refs(bullet_text),
                })
        projects.append({"project_id": title, "bullets": bullets})

    return {"summary": summary, "projects": projects}


def extract_current_headline(resume_raw: str) -> str | None:
    """지금 이력서 원문의 첫 줄(자기소개)을 찾는다 - _replace_headline과
    같은 판정 기준(이름/연락처 줄이 아닌 첫 줄)."""
    for ln in resume_raw.split("\n")[:6]:
        stripped = ln.strip()
        if stripped and "|" not in stripped and "@" not in stripped:
            return ln
    return None


def _replace_headline(resume_raw: str, headline: str) -> tuple[str, str | None]:
    """반환: (결과 텍스트, 교체된 원래 줄 - Before/After UI용, 못 찾으면 None)."""
    if not headline:
        return resume_raw, None
    lines = resume_raw.split("\n")
    for i, ln in enumerate(lines[:6]):
        stripped = ln.strip()
        if not stripped or "|" in stripped or "@" in stripped:
            continue
        original = lines[i]
        lines[i] = headline
        return "\n".join(lines), original
    return resume_raw, None  # 헤드라인 위치를 못 찾으면 원문 그대로(왜곡 방지)


def _apply_term_replacements(text: str, replacements: list[dict]) -> tuple[str, list[dict]]:
    """replacements: [{"before": "...", "after": "...", "source_text": "..."}].
    customization_rules.py(Rule5)가 APPROVED_TERM_MAP으로 이미 검증해서
    넘긴 것만 적용한다(여기서는 판단하지 않음). source_text가 있으면
    치환 범위를 그 문장 안으로 한정한다 - 전체 텍스트에서 첫 occurrence
    를 바꾸면, 이 치환과 무관한 다른 문장에 우연히 같은 단어가 있을 때
    엉뚱한 문장이 바뀌는 버그가 생긴다(구현 검증 중 실측 확인, 2026-
    07-21)."""
    applied: list[dict] = []
    for r in replacements:
        before, after = r.get("before", ""), r.get("after", "")
        source_text = r.get("source_text", "")
        if not before:
            continue
        if source_text and source_text in text:
            new_source = source_text.replace(before, after, 1)
            if new_source != source_text:
                text = text.replace(source_text, new_source, 1)
                applied.append(r)
        elif before in text:
            text = text.replace(before, after, 1)
            applied.append(r)
    return text, applied


def apply(resume_raw: str, decisions: list[dict]) -> dict:
    """decisions: customization_rules.evaluate()의 반환값(4~5개 항목,
    각 {item, status, reason_code, reason, evidence, before, after}).
    status=="change"인 항목만 적용한다. 반환: "커스터마이징 완료
    데이터"(resume_customized 텍스트 + 항목별 적용 로그 + decisions
    그대로) - PDF 생성 단계(pdf_generator.py)는 이 반환값만 소비한다."""
    text = resume_raw
    log: list[str] = []
    by_item = {d["item"]: d for d in decisions}

    struct = by_item.get("이력서 구조 최적화")
    project_order: list[str] | None = None
    skill_order: list[str] | None = None
    if struct and struct["status"] == "change":
        after = struct.get("after") or {}
        project_order = after.get("project_order")
        skill_order = after.get("skill_order")
        if project_order:
            text = _reorder_projects(text, project_order)
            log.append(f"프로젝트 순서 적용: {project_order}")
        if skill_order:
            text = _reorder_skills(text, skill_order)
            log.append(f"기술 순서 적용: {skill_order}")

    emphasis = by_item.get("핵심 경험 및 어필 전략")
    bullet_changes: list[dict] = []
    if emphasis and emphasis["status"] == "change":
        for change in (emphasis.get("after") or {}).get("bullet_reorders") or []:
            project_title = change["project_title"]
            section_title = change["section_title"]
            new_order = change["new_order"]
            _, blocks, _ = _split_projects(text)
            for idx, (title, block) in enumerate(blocks):
                if title != project_title:
                    continue
                new_block = _replace_subsection_bullets(block, section_title, new_order)
                if new_block != block:
                    preamble, all_blocks, postamble = _split_projects(text)
                    all_blocks[idx] = (title, new_block)
                    text = preamble + "".join(b for _, b in all_blocks) + postamble
                    bullet_changes.append(change)
                break
        log.append(f"경험/성과 강조 순서 적용: {len(bullet_changes)}건")

    term = by_item.get("표현 및 JD 용어 최적화")
    applied_terms: list[dict] = []
    if term and term["status"] == "change":
        replacements = (term.get("after") or {}).get("replacements") or []
        text, applied_terms = _apply_term_replacements(text, replacements)
        log.append(f"용어 치환 적용: {len(applied_terms)}건")

    headline_decision = by_item.get("자기소개 최적화")
    chosen_headline = None
    if headline_decision and headline_decision["status"] == "change":
        chosen_headline = (headline_decision.get("after") or {}).get("text")
        if chosen_headline:
            text, original_headline = _replace_headline(text, chosen_headline)
            log.append(f"자기소개 교체: {chosen_headline!r} (원본: {original_headline!r})")

    return {
        "resume_customized": text,
        "log": log,
        "decisions": decisions,
        "applied": {
            "project_order": project_order,
            "skill_order": skill_order,
            "bullet_reorders": bullet_changes,
            "term_replacements": applied_terms,
            "headline": chosen_headline,
        },
    }
