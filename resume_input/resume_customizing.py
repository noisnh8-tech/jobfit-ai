"""
resume_input/resume_customizing.py

Resume Customizing(2026-08-03, v4 - Command Schema, rewrite_sentence 제거)
- Semantic Link 결과만 입력받아, Resume Apply Engine(customizer.apply()
확장판, 별도 작업 예정)이 그대로 실행할 Command 목록을 생성한다. LLM
호출 없음. 문장을 새로 생성하지 않는다 - "무엇을 바꿀지"만 결정하고,
"어떻게 자연스러운 문장을 쓸지"는 이 파일의 책임이 아니다(사용자 확정).

v3 -> v4 변경(2026-08-03, 사용자 확정, 실측 근거): `rewrite_sentence`
(action=rephrase를 frame.action/object/goal/outcome 이어붙이기 템플릿으로
문장 생성)를 완전히 제거했다. 실제 Resume Object 3건에 적용해본 결과
전부 비문이 나왔다(조사 오류 + 술어 없는 명사구 종결) - 이건 Rule
Template의 한계가 아니라 애초에 "자연스러운 문장 생성"이 Resume
Customizing의 책임이 아니었다는 뜻이다. 문장 재작성이 필요하면 이건
Resume Rewrite Engine이라는 완전히 별도의 기능(LLM 재도입 또는 훨씬
정교한 NLG 필요)이고, 지금 파이프라인 범위 밖이다 - 억지로 살리지
않는다.

Command 종류(v4, Apply Engine이 지원할 4종 중 이 파일이 만드는 것):
- move_project: action=reposition, 대상이 프로젝트 섹션 소속일 때.
- move_bullet: action=reposition, 대상이 프로젝트 안의 개별 항목일 때.
- highlight_keyword: relation=match 이거나 action=add_emphasis일 때.
  **새 문구를 만들지 않는다** - JD Semantic Object 자신의 concepts(이미
  Understanding 단계에서 추출된 원문 명사구, link_engine.py가 각 link에
  복사해둔 값)를 그대로 "강조할 키워드 후보"로 넘긴다. 실제로 그 단어가
  해당 Resume Object의 원문에 문자 그대로 있는지 확인하고 강조(볼드 등)
  하는 건 Apply Engine의 책임이다 - 없는 단어를 강조하라고 지어내지
  않는다(PROJECT_HANDBOOK.md "과장 없이 사실에 기반" 원칙).

이 파일이 만들지 않는 것:
- replace_term: `customization_rules.py`의 Rule5(`evaluate_terms`)가 이미
  concepts 기반 승인된 동의어 치환을 하고 있다 - 중복 재구현하지 않는다.
  pipeline 연결 시 이 파일의 Command 목록과 별도로 합쳐서 Apply Engine에
  전달한다.
- rewrite_sentence(문장 재작성): 위 설명대로 이번 범위에서 제외.

priority는 importance만으로 정한다(critical=1 ... low=4) - 새 가중치
아님."""
from __future__ import annotations

_PRIORITY_BY_IMPORTANCE = {"critical": 1, "core": 2, "normal": 3, "low": 4}

# 이 섹션 이름이면 "프로젝트"가 아니라 스킬/자격증/대외활동 목록이다
# (customization_rules.py의 _NON_PROJECT_SECTIONS와 동일 기준 재사용 -
# 새 판단 기준을 만들지 않는다).
_NON_PROJECT_SECTIONS = {"보유 기술", "자격증", "대외활동", "", None}


def build_resume_commands(link_result: dict) -> list[dict]:
    """Semantic Link 결과에서 Command 목록을 만든다. 각 Command는
    {"type", "priority", ...실행에 필요한 필드}만 담는다 - 사람이 읽는
    문장이나 새로 지어낸 텍스트는 없다."""
    commands: list[dict] = []
    for link in link_result.get("links") or []:
        importance = link.get("importance", "normal")
        priority = _PRIORITY_BY_IMPORTANCE.get(importance, 3)
        relation = link.get("relation")
        keywords = link.get("concepts") or []

        if relation == "match":
            for m in link.get("matched_resume_objects") or []:
                commands.append({
                    "type": "highlight_keyword", "target_resume_object_id": m["resume_object_id"],
                    "keywords": keywords, "priority": priority,
                    "source_jd_object_id": link.get("jd_object_id"),
                })
            continue

        improvement = link.get("improvement")
        if not improvement or improvement.get("type") != "rewrite":
            continue  # genuine_gap - 건드리지 않음

        action = improvement["action"]
        target_id = improvement["target_resume_object_id"]
        matched = next(
            (m for m in link.get("matched_resume_objects") or [] if m["resume_object_id"] == target_id),
            None,
        )
        project = matched.get("source_project") if matched else None
        is_project_item = project not in _NON_PROJECT_SECTIONS

        if action == "reposition" and is_project_item:
            commands.append({
                "type": "move_project", "project_id": project, "priority": priority,
                "source_jd_object_id": link.get("jd_object_id"),
            })
            commands.append({
                "type": "move_bullet", "project_id": project, "bullet_id": target_id, "priority": priority,
                "source_jd_object_id": link.get("jd_object_id"),
            })
        elif action == "reposition":
            commands.append({
                "type": "move_bullet", "project_id": project, "bullet_id": target_id, "priority": priority,
                "source_jd_object_id": link.get("jd_object_id"),
            })
        elif action == "add_emphasis":
            commands.append({
                "type": "highlight_keyword", "target_resume_object_id": target_id,
                "keywords": keywords, "priority": priority,
                "source_jd_object_id": link.get("jd_object_id"),
            })
        # rephrase: Command 없음(v4, 위 docstring 참고 - Resume Customizing 범위 밖)

    commands.sort(key=lambda c: c["priority"])
    return commands
