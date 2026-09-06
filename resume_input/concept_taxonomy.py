"""
resume_input/concept_taxonomy.py

Concept Resolver - Semantic Object의 concept_id/matching_anchor를 LLM이
아니라 코드로 결정한다(설계 원칙: docs/semantic_object_schema.md).
LLM은 normalized_text까지만 만들고, 이 모듈이 그 텍스트를 받아 Canonical
Taxonomy(Concept 고정 + Alias 증가 2단계)에서 concept_id를 찾거나 새로
등록한다. 새 LLM 호출 없음 - 로컬 임베딩 모델(candidate_search와 동일
모델, 이미 프로세스에 캐시됨) + DB 조회/삽입만.

해석(resolve) 절차(체크포인트 0 실측으로 확정, docs/semantic_object_
schema.md 참고):
1. concept_alias에 정확히 일치하는 표현이 있으면 그 concept_id 재사용
   (가장 빠르고 안정적 - 반복 표현은 대부분 여기서 걸림).
2. 없으면 같은 layer의 기존 Concept(canonical_name)들과 임베딩 비교 -
   최댓값이 CONCEPT_PROMOTE_THRESHOLD 이상이면 그 concept_id 재사용 +
   새 Alias 등록.
3. 임계치 미만이면 새 Concept 발급 + 최초 Alias로 등록.
4. matching_anchor: 자기 자신 + 더 느슨한 ANCHOR_THRESHOLD로 찾은 인접
   Concept 상위 ANCHOR_TOP_K개.

Concept은 최소한으로 유지되고(레이어당 개념 자체는 적음), Alias만
계속 늘어난다 - 문서가 1900건->20만 건으로 늘어도 Concept 비교 비용은
거의 늘지 않는다(CompanyStore.upsert_by_name()과 같은 "이름으로 먼저
찾고 없으면 등록" 골격을 따르되, Concept은 동의어 판별을 위해 임베딩
비교 단계를 추가한 것).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

from resume_input.candidate_search import _load_embedding_model
from resume_input.job_store import DB_PATH

CONCEPT_PROMOTE_THRESHOLD = 0.85
ANCHOR_THRESHOLD = 0.6
ANCHOR_TOP_K = 3

_LAYER_PREFIX = {
    "problem": "PROB", "thinking": "THINK", "task": "TASK",
    "skill": "SKILL", "qualification": "QUAL", "culture": "CULT",
}

# 2026-07-17(사용자 확정, 성능 병목 1순위) - resolve_concept()이 호출될
# 때마다 그 레이어의 canonical_name **전체**를 매번 새로 임베딩하고 있었다
# (임계치 비교용 1회 + _find_anchors 안에서 또 1회, 총 2회) - 캐시가
# 전혀 없어서 Semantic Object 하나당 그 레이어의 taxonomy 전체가 반복
# 재계산됐다(실측: JD 6건 backfill 중 Concept Resolve 단계가 ~110초
# 소비 - LLM 호출(병렬화된 구간, ~26초)보다 훨씬 컸다). Concept은
# "최소한으로 유지되고 Alias만 늘어난다"는 설계 원칙(모듈 docstring)
# 그대로, (db_path, layer)별 임베딩을 프로세스 메모리에 캐시하고 새
# Concept이 추가될 때만 그 자리에서 이어붙인다(전체 재계산 없음). 새
# Concept이 등록되는 시점(같은 프로세스 안에서 순차 실행되는 Concept
# Resolver 호출부, understanding.py의 _resolve_concepts() 참고)에는
# 그 즉시 캐시에 반영되므로 다음 resolve_concept() 호출이 최신 상태를
# 정확히 본다 - 매번 DB를 다시 읽어오는 것과 결과상 동일하다(값 자체는
# 전혀 안 바뀜, 계산을 안 되풀이할 뿐).
_taxonomy_cache_lock = threading.Lock()
_taxonomy_cache: dict[tuple[str, str], dict] = {}
_tables_initialized: set[str] = set()


def _load_layer_cache(conn: sqlite3.Connection, db_path: Path, layer: str) -> dict:
    key = (str(db_path), layer)
    with _taxonomy_cache_lock:
        cached = _taxonomy_cache.get(key)
        if cached is not None:
            return cached
        rows = conn.execute(
            "SELECT concept_id, canonical_name FROM concept_taxonomy WHERE layer = ?", (layer,)
        ).fetchall()
        names = [r[1] for r in rows]
        entry = {
            "concept_ids": [r[0] for r in rows],
            "names": names,
            "embeddings": _embed(names) if names else np.zeros((0, 384)),
        }
        _taxonomy_cache[key] = entry
        return entry


def _add_to_layer_cache(db_path: Path, layer: str, concept_id: str, name: str) -> None:
    key = (str(db_path), layer)
    with _taxonomy_cache_lock:
        entry = _taxonomy_cache.get(key)
        if entry is None:
            return  # 아직 캐시가 없으면 다음 _load_layer_cache 호출이 DB에서 새로 읽어옴(자동으로 최신 상태)
        emb = _embed([name])
        entry["concept_ids"].append(concept_id)
        entry["names"].append(name)
        entry["embeddings"] = emb if entry["embeddings"].shape[0] == 0 else np.vstack([entry["embeddings"], emb])


def _find_anchors_cached(entry: dict, concept_id: str) -> list[str]:
    """_find_anchors()와 동일한 로직(자기 자신 + ANCHOR_THRESHOLD 이상인
    상위 ANCHOR_TOP_K개) - DB 재조회/재임베딩 없이 캐시된 레이어
    임베딩만으로 계산한다."""
    concept_ids = entry["concept_ids"]
    if concept_id not in concept_ids or len(concept_ids) <= 1:
        return [concept_id]
    self_idx = concept_ids.index(concept_id)
    self_emb = entry["embeddings"][self_idx]
    sims = entry["embeddings"] @ self_emb
    ranked = sorted(
        ((cid, float(sim)) for cid, sim in zip(concept_ids, sims) if cid != concept_id),
        key=lambda x: -x[1],
    )
    anchors = [concept_id]
    for other_id, sim in ranked[:ANCHOR_TOP_K]:
        if sim >= ANCHOR_THRESHOLD:
            anchors.append(other_id)
    return anchors


def _init_tables(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_taxonomy (
            concept_id TEXT PRIMARY KEY,
            layer TEXT NOT NULL,
            canonical_type TEXT,
            canonical_name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_alias (
            alias_text TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (alias_text, concept_id)
        )
        """
    )
    conn.commit()
    conn.close()


def _embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384))
    model = _load_embedding_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _next_concept_id(layer: str, conn: sqlite3.Connection) -> str:
    prefix = _LAYER_PREFIX.get(layer, layer.upper()[:4])
    n = conn.execute("SELECT COUNT(*) FROM concept_taxonomy WHERE layer = ?", (layer,)).fetchone()[0]
    return f"{prefix}_{n + 1:03d}"


def resolve_concept(normalized_text: str, layer: str, canonical_type: str = "", db_path: Path = DB_PATH) -> dict:
    """normalized_text -> {"concept_id": str, "matching_anchor": list[str]}.
    새 LLM 호출 없음 - 임베딩 비교 + DB 조회/삽입만. 계산 방식은 그대로
    지만(알고리즘/임계치 전부 동일), 레이어 taxonomy 임베딩을 프로세스
    캐시에서 재사용한다(_load_layer_cache/_add_to_layer_cache 참고) -
    매 호출마다 그 레이어 전체를 다시 임베딩하지 않는다."""
    key = str(db_path)
    if key not in _tables_initialized:
        _init_tables(db_path)
        _tables_initialized.add(key)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT concept_id FROM concept_alias WHERE alias_text = ?", (normalized_text,)
        ).fetchone()
        entry = _load_layer_cache(conn, db_path, layer)

        if row:
            concept_id = row[0]
            anchors = _find_anchors_cached(entry, concept_id)
            return {"concept_id": concept_id, "matching_anchor": anchors}

        best_idx, best_sim = -1, -1.0
        if entry["names"]:
            cand_emb = _embed([normalized_text])[0]
            sims = entry["embeddings"] @ cand_emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

        if entry["names"] and best_sim >= CONCEPT_PROMOTE_THRESHOLD:
            concept_id = entry["concept_ids"][best_idx]
            conn.execute(
                "INSERT OR IGNORE INTO concept_alias (alias_text, concept_id) VALUES (?, ?)",
                (normalized_text, concept_id),
            )
        else:
            concept_id = _next_concept_id(layer, conn)
            conn.execute(
                "INSERT INTO concept_taxonomy (concept_id, layer, canonical_type, canonical_name) VALUES (?, ?, ?, ?)",
                (concept_id, layer, canonical_type, normalized_text),
            )
            conn.execute(
                "INSERT OR IGNORE INTO concept_alias (alias_text, concept_id) VALUES (?, ?)",
                (normalized_text, concept_id),
            )
            _add_to_layer_cache(db_path, layer, concept_id, normalized_text)

        conn.commit()
        anchors = _find_anchors_cached(entry, concept_id)
        return {"concept_id": concept_id, "matching_anchor": anchors}
    finally:
        conn.close()
