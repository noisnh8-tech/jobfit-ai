"""
resume_input/resume_apply_engine.py

Resume Apply Engine(2026-08-03) - Resume Customizing이 만든 Command만
실행한다. LLM 호출 없음, 새 판단 없음(무엇을 바꿀지는 이미 정해져
있음 - 이 파일은 "어떻게 원문 텍스트에 반영하는가"만 담당). customizer.py
의 기존 파싱/치환 함수(_split_projects/_reorder_projects/split_project_
subsections/split_bullets/_replace_subsection_bullets/_apply_term_
replacements)를 그대로 재사용한다 - 원문 파싱 로직을 새로 만들지 않는다.

지원 Command(v1):
- move_project: `customizer._reorder_projects()`를 그대로 호출.
- move_bullet: Resume Object의 `evidence.source_text`로 실제 불릿을 찾아
  (customization_rules._bullet_matches_object()와 동일한 매칭 방식을
  이 파일 안에 그대로 재구현 - import하기엔 private 함수라 동일 로직만
  복제, 매칭 기준 자체는 바꾸지 않음) 그 서브섹션 안에서 맨 앞으로 옮긴다.
- reorder_skills(2026-08-16 추가, customization_planner.py ③ 전용):
  `customizer._reorder_skills()`를 그대로 호출 - 이 함수는 이미 있었지만
  Command 스키마에는 연결이 안 돼 있었다(구 customizer.apply() 전용).
  새 파싱/치환 로직을 만들지 않고 기존 함수만 Command 경로에 잇는다.
- replace_term: `customizer._apply_term_replacements()`를 그대로 호출
  (customization_rules.evaluate_terms()가 만든 replacements를 받는다 -
  이 엔진이 직접 만들지 않는다).
- replace_headline(2026-08-16 추가, ① 전용) - `customizer._replace_headline()`
  을 그대로 호출. 사용자가 S4 화면에서 [추천안 적용]을 눌렀을 때만 이
  Command가 생긴다(rewrite_engine.generate_headline_proposal()의 LLM
  제안 + 자동 검증을 통과한 것만, 자동 적용 없음). pdf_generator.generate()
  는 applied["headline"]이 있고 resume_customized의 headline이 resume_raw
  와 다를 때만 PDF 헤드라인 필드를 패치한다(기존에 이미 있던 로직) -
  이 Command가 바로 그 조건을 만든다.
- highlight_keyword: **실행하지 않는다.** pdf_generator.py를 확인한 결과
  좌표 기반 텍스트 채우기만 지원하고 볼드/색 등 인라인 서식 기능이
  없다(2026-08-03 확인, `_slot_fields`/`generate()` 참고) - 이 Command는
  로그에만 남기고 텍스트는 그대로 둔다. 실제로 강조하려면 pdf_generator.py
  확장이 먼저 필요하다(별도 작업, 이번 범위 아님) - 없는 기능을 실행한
  것처럼 조용히 넘어가지 않는다.

레이아웃/사진/폰트/PDF 템플릿은 건드리지 않는다 - customizer.py의 기존
함수들도 전부 "텍스트 블록을 통째로 이동/치환"만 하지 새 디자인을
만들지 않는다(이 원칙을 그대로 승계)."""
from __future__ import annotations

import re

from resume_input import customizer

_MATCHABLE_LAYERS = ("problem", "thinking", "task", "skill", "qualification")


def _obj_id(obj: dict) -> str:
    return obj.get("id") or obj.get("normalized_text") or ""


def _norm_ws(text: str) -> str:
    return re.sub(r"[\s•]+", "", text or "")


def _bullet_matches(bullet: str, source_text: str) -> bool:
    """customization_rules._bullet_matches_object()와 동일 로직(매칭
    기준을 새로 만들지 않고 그대로 복제) - 이 불릿이 이 Resume Object의
    evidence.source_text에 대응하는가."""
    b, s = _norm_ws(bullet), _norm_ws(source_text)
    if not b or not s:
        return False
    if b in s or s in b:
        return True
    return any(_norm_ws(line) and _norm_ws(line) in b for line in (source_text or "").split("\n"))


def _project_order_from_commands(commands: list[dict]) -> list[str]:
    order: list[str] = []
    for c in sorted(commands, key=lambda c: c["priority"]):
        if c["type"] == "move_project" and c.get("project_id") and c["project_id"] not in order:
            order.append(c["project_id"])
    return order


def _apply_move_bullets(text: str, commands: list[dict], resume_objects: list[dict]) -> tuple[str, list[dict]]:
    """move_bullet Command를 project_id 단위로 묶어서, 그 프로젝트 안의
    맞는 서브섹션에서 실제 불릿을 찾아 맨 앞으로 옮긴다. 여러 Command가
    같은 (project, section)을 가리키면 priority 순서대로 앞쪽에 쌓는다
    (customization_rules._evaluate_bullet_reorders()와 동일 원칙 - 매칭된
    불릿을 앞으로, 나머지는 원래 순서 유지한 채 뒤로)."""
    obj_by_id = {_obj_id(o): o for o in resume_objects}
    applied: list[dict] = []

    by_project: dict[str, list[dict]] = {}
    for c in sorted(commands, key=lambda c: c["priority"]):
        if c["type"] == "move_bullet" and c.get("project_id"):
            by_project.setdefault(c["project_id"], []).append(c)

    for project_title, cmds in by_project.items():
        _, blocks, _ = customizer._split_projects(text)
        block_idx = next((i for i, (t, _b) in enumerate(blocks) if t == project_title), None)
        if block_idx is None:
            continue
        title, block = blocks[block_idx]

        for section_title, section_body in customizer.split_project_subsections(block):
            bullets = customizer.split_bullets(section_body)
            if len(bullets) < 2:
                continue
            target_bullets = []
            for c in cmds:
                obj = obj_by_id.get(c["bullet_id"])
                if not obj:
                    continue
                source_text = (obj.get("evidence") or {}).get("source_text", "")
                match = next((b for b in bullets if _bullet_matches(b, source_text)), None)
                if match and match not in target_bullets:
                    target_bullets.append(match)
            if not target_bullets or (len(target_bullets) == 1 and bullets[0] == target_bullets[0]):
                continue  # 근거 없거나 이미 맨 앞
            new_order = target_bullets + [b for b in bullets if b not in target_bullets]
            if new_order == bullets:
                continue
            new_block = customizer._replace_subsection_bullets(block, section_title, new_order)
            if new_block != block:
                _, all_blocks, _ = customizer._split_projects(text)
                preamble, _, postamble = customizer._split_projects(text)
                all_blocks[block_idx] = (title, new_block)
                text = preamble + "".join(b for _, b in all_blocks) + postamble
                applied.append({
                    "project": project_title, "section": section_title,
                    "new_order": new_order, "new_order_count": len(new_order),
                })
    return text, applied


def apply_commands(
    resume_raw: str, commands: list[dict], resume_objects: list[dict],
    term_replacements: list[dict] | None = None,
) -> dict:
    """Command 목록(+선택적으로 기존 Rule5의 term_replacements)을 실행해서
    수정된 이력서 텍스트를 만든다. LLM 호출 없음. 레이아웃/사진/폰트는
    건드리지 않는다 - 텍스트 블록 이동과 치환만 한다."""
    text = resume_raw
    log: list[str] = []

    project_order = _project_order_from_commands(commands)
    if project_order:
        text = customizer._reorder_projects(text, project_order)
        log.append(f"프로젝트 순서 적용: {project_order}")

    skill_order = next(
        (c["skill_order"] for c in sorted(commands, key=lambda c: c["priority"])
         if c["type"] == "reorder_skills" and c.get("skill_order")),
        None,
    )
    if skill_order:
        text = customizer._reorder_skills(text, skill_order)
        log.append(f"기술 순서 적용: {skill_order}")

    text, bullet_changes = _apply_move_bullets(text, commands, resume_objects)
    if bullet_changes:
        log.append(f"불릿 순서 적용: {len(bullet_changes)}건")

    applied_terms: list[dict] = []
    if term_replacements:
        text, applied_terms = customizer._apply_term_replacements(text, term_replacements)
        log.append(f"용어 치환 적용: {len(applied_terms)}건")

    headline_text = next(
        (c["text"] for c in sorted(commands, key=lambda c: c["priority"]) if c["type"] == "replace_headline" and c.get("text")),
        None,
    )
    if headline_text:
        text, _original_line = customizer._replace_headline(text, headline_text)
        log.append(f"자기소개 교체: {headline_text!r}")

    skipped = [c["type"] for c in commands if c["type"] == "highlight_keyword"]
    if skipped:
        log.append(
            f"highlight_keyword {len(skipped)}건은 실행하지 않음 "
            "(pdf_generator.py가 인라인 서식을 지원하지 않음 - 별도 작업 필요)"
        )

    return {
        "resume_customized": text,
        "log": log,
        "applied": {
            "project_order": project_order,
            "skill_order": skill_order,
            "bullet_reorders": bullet_changes,
            "term_replacements": applied_terms,
            "headline": headline_text,
        },
        "skipped_commands": skipped,
    }
