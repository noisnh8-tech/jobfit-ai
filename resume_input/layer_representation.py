"""
resume_input/layer_representation.py

Layer 기반 Representation Builder(2026-07-25, M3 Production Migration
리팩터링) - Scorer(layer_m3.py)와 책임을 분리한다. 이 모듈은 "무엇을
비교할 텍스트로 쓸지"만 만들고, "어떻게 비교할지"는 모른다.

Resume 쪽(LLM이 만든 semantic_objects)을 레이어별 문자열로 합치는
용도로 쓴다. JD 쪽은 Rule Representation(rule_representation.
build_rule_based_representation)이 이미 이 형태(task/skill/qualification
키)로 직접 만들어서 이 모듈을 거치지 않는다.
"""
from __future__ import annotations

SHARED_LAYERS = ["problem", "thinking", "task", "skill", "qualification"]


def build_layer_representation(semantic_objects: list[dict]) -> dict[str, str]:
    """semantic_objects(레이어별 Object 목록)를 레이어당 문자열 1개로
    합친다."""
    by_layer: dict[str, list[str]] = {}
    for obj in semantic_objects or []:
        layer = obj.get("layer")
        if layer not in SHARED_LAYERS:
            continue
        text = (obj.get("normalized_text") or "").strip()
        meaning = (obj.get("meaning") or "").strip()
        if not text:
            continue
        combined = f"{text}: {meaning}" if meaning else text
        by_layer.setdefault(layer, []).append(combined)
    return {layer: " / ".join(texts) for layer, texts in by_layer.items()}
