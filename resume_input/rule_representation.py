"""
resume_input/rule_representation.py

Rule 기반 JD Representation - LLM 호출 없이 Task/Skill/Qualification을
만든다. 2026-07-25 검증 완료(docs/verification/2026-07-25_m3_
representation_ab/summary.md) - A/B Top20 비교에서 기존 LLM 기반
Representation(jd_short_semantic, 핵심역할/핵심문제/핵심기대 3필드)보다
Top10 관련도가 높고, FP가 늘지 않았으며, 추천 근거를 SQL/Python 등
구체적 도구 매칭으로 설명할 수 있음을 확인했다.

Section(job_detail.extract_display_sections)을 그대로 쓴다 - 2026-07-25
Header Detector 개선(header_patterns.py, Strong/Weak Pattern Registry)을
별도 연결 작업 없이 자동으로 이어받는다.

Problem/Thinking 레이어는 규칙으로 안 만든다(의미 추론이 필요해서) -
빈 문자열로 두면 LayerM3Scorer가 그 레이어를 자동으로 건너뛴다(0점
처리가 아니라 skip - 자원 없는 쪽을 불리하게 만들지 않음).
"""
from __future__ import annotations

from resume_input.job_detail import (
    extract_display_sections,
    is_qualification_like_line,
    is_task_like_line,
    merge_delimiter_lines,
    merge_leading_punctuation_lines,
)
from resume_input.job_prep import extract_tech_stack


def is_rule_representation_valid(rep: dict, min_len: int = 30) -> bool:
    """Rule Representation이 M3 입력으로 신뢰할 만한지 판단하는 단일
    기준(2026-07-25, 실측 확정 - FN Root Cause Audit). 원인이 서로 달라도
    (예: 특정 부동산 플랫폼 - Header Coverage Gap, Task 헤더 자체가 원문에 없음 /
    특정 해외 테크기업 - Header 오분류, "Role"이 DUTIES로 잡혀 실제 자격요건 내용이 task
    버킷에 섞여 들어감) 관찰되는 증상은 같다(task 또는 qualification이
    비어있거나 정보가 부족함) - 이 함수 하나로 원인과 무관하게 판단한다.

    단순 비어있음(!= "")이 아니라 길이 기준(30자)을 쓰는 이유: 자격요건이
    원래 짧고 업무만 긴 정상 JD도 있어서, "비어있는가"보다 "이 레이어가
    M3가 쓸 만큼 정보를 담고 있는가"를 봐야 한다. Header Detector/Rule
    Parser가 나중에 바뀌어도 M3 쪽(candidate_search.py)은 이 함수 결과만
    보면 되고 수정할 필요가 없다.

    설계 의도(2026-07-25): 지금은 길이 기준의 heuristic이다 - 나중에
    Representation 품질 판단이 더 중요해지면(예: 섹션이 진짜 헤더에서
    왔는지/오분류로 흘러들어왔는지, 문장 형태인지 키워드 나열인지 등을
    반영) True/False가 아니라 confidence(0~1) 점수로 확장할 수 있다.
    지금 바꿀 필요는 없고, 이 자리에 그대로 발전시키면 된다 - M3 쪽
    호출부는 이 함수의 반환값만 보므로 내부 판단 로직이 바뀌어도 영향
    없다."""
    return len(rep["task"].strip()) > min_len and len(rep["qualification"].strip()) > min_len


def build_rule_based_representation(job: dict) -> dict[str, str]:
    """JD의 Task/Skill/Qualification을 규칙 기반으로 만든다(새 LLM 호출
    없음). Task/Qualification은 Section(주요업무/자격요건+우대사항)에서,
    Skill은 posting_text 전체에서 알려진 기술 키워드를 표면 스캔한다
    (기술 스택은 섹션 경계와 무관하게 문서 어디서든 언급될 수 있어서
    전체 스캔이 맞다 - Task/Qualification과는 다른 성격의 필드).

    task_lines(2026-07-29, Task Signal v1.0 Frozen - job_detail.
    is_task_like_line() 검증 완료, 443건 검증셋 F1 0.88)와 qualification_
    lines(같은 날, Qualification Signal v1.0 Frozen - is_qualification_
    like_line() 검증 완료, F1 0.61): task/qualification 텍스트를 삭제하거나
    골라내지 않는다 - "task"/"qualification" 필드는 그대로 두고, 줄마다
    신호(bool)만 추가 정보로 붙인다. 제거 필터도, UI 전용 선택기도
    아니다 - M3/Ranking/Display 등 후속 단계가 참고 신호로만 쓸 수 있게
    하는 것이 목적이다(정보 손실 없음 원칙). 이 필드들을 아직 아무
    소비처도 사용하지 않는다 - 사용 여부/방식은 별도로 결정한다.

    Qualification Validation Finding(중요): FP의 100%가 실제로는
    Preferred 문장이었다 - Task/Culture/Process/Noise 오탐은 0건. 즉
    Qualification과 Preferred는 문장 형태로 원천적 구분이 불가능하고,
    최종 구분은 이 줄이 어느 헤더(자격요건 vs 우대사항) 아래 있었는지로만
    가능하다 - Rule로 더 파고들 수 있는 부분이 아니다."""
    sections = extract_display_sections(job)
    posting_text = job.get("posting_text", "") or ""
    skills = extract_tech_stack(posting_text)
    rep = {
        "task": merge_leading_punctuation_lines(
            merge_delimiter_lines(sections.get("resp", "").strip())
        ),
        "skill": ", ".join(skills),
        "qualification": merge_leading_punctuation_lines(
            merge_delimiter_lines(
                (sections.get("req", "") + "\n" + sections.get("pref", "")).strip()
            )
        ),
    }
    rep["is_valid"] = is_rule_representation_valid(rep)
    rep["task_lines"] = [
        {"text": ln, "task_signal": is_task_like_line(ln)}
        for ln in rep["task"].split("\n") if ln.strip()
    ]
    rep["qualification_lines"] = [
        {"text": ln, "qualification_signal": is_qualification_like_line(ln)}
        for ln in rep["qualification"].split("\n") if ln.strip()
    ]
    return rep


def build_rule_based_m4_nodes(rep: dict) -> list[str]:
    """M4(meaning_matching.m4_score)에 넣을 JD 쪽 노드 리스트를
    build_rule_based_representation()의 출력에서 그대로 뽑는다(새로
    다시 파싱하지 않음) - Task 줄 + Qualification 줄 + Skill 키워드를
    중복 제거해 이어붙인 평문 리스트다(2026-08-13, docs/verification/
    2026-07-25_m3_representation_ab/audit_final_rule_node.py의
    rule_set_only_nodes() 실측 검증 그대로 production化).

    "->" 구분자가 없는 평문이라 meaning_matching._nodes_of()가 각
    항목을 노드 1개로만 처리하고 _edges_of()는 빈 리스트를 반환한다 -
    m4_score()가 그 경우 자동으로 node_score만 쓰도록(SET-ONLY) 이미
    분기돼 있어서(meaning_matching.py 무수정) 이 함수만 새로 추가하면
    된다. 관계(Edge) 자체가 M4 성능에 기여하지 않는다는 것도 같은
    실험(§7 Relation Ablation Audit)에서 이미 확인됐다."""
    task_texts = [ln["text"] for ln in rep.get("task_lines") or [] if ln["text"].strip()]
    qual_texts = [ln["text"] for ln in rep.get("qualification_lines") or [] if ln["text"].strip()]
    skill_texts = [s.strip() for s in (rep.get("skill") or "").split(",") if s.strip()]
    return list(dict.fromkeys(task_texts + qual_texts + skill_texts))
