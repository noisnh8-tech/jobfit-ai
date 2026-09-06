"""
resume_input/meaning_matching.py

M3(Rule+Embedding Hybrid) + M4(Graph Node+Edge Matching) - Understanding
Representation(short_semantic + semantic_graph)을 LLM 재호출 없이
비교하는 순수 로컬 계산. 임베딩 모델 forward pass만 쓴다.

검증 이력(2026-07-13, docs/verification/2026-07-13_understanding_engine_
meaning_matching/summary.md - 다시 실험하지 않는다):
- Context 전체를 벡터 하나로 압축해서 비교하는 방식은 실패했다
  (anisotropy, MiniLM/BGE-M3/e5-large 3개 모델 전부 판별력 붕괴).
- 8개 매칭 방식(M1~M7) 비교 결과 M3+M4 결합이 LLM Gold Standard 대비
  Spearman 0.820으로 최고, 기존 BM25+Embedding(0.708)을 확실히 앞섬.
- M4는 Node+Edge까지만 확장한다 - Path/위치기반 Weighted는 오히려
  손해였다(m4_internal_decomposition.py 실측).
- 쿼리 시점 속도: 1,917건 48.8ms, 10,000건 225.9ms - BM25(1,917건
  1,300ms+)보다 훨씬 빠르다.
- 서로 다른 이력서 프로필 2개(SQL/Tableau 실무형, 신입 개발자 출신)에서
  반복 검증됨 - Current(BM25+Embedding)는 키워드 겹침으로 완전히 무관한
  직군(반도체 엔지니어링, 자율주행 엔지니어링 등)을 Top20에 올리는 반면
  M3+M4는 직군 정확도를 지켰다.

M3/M4는 자격요건(연차/학력)을 전혀 모델링하지 않는다 - 그건
career_filter.py가 이미 앞단에서 처리한다는 전제로 설계됐다.
"""
from __future__ import annotations

import json

import numpy as np

from resume_input.candidate_search import _load_embedding_model

# M3의 필드쌍 가중치 - 사람이 "이 필드끼리 비교하라"고 지정한 규칙
# (m3_vs_m4b_case_analysis.py에서 확인: 추상적/전략적 개념 대응에 강함,
# 특히 problem<->expect 교차비교가 "숨은 의도"를 잘 잡아낸다).
_RULE_PAIRS = [
    ("핵심역할", "핵심역할", 1.0),
    ("핵심문제", "핵심기대", 1.2),
    ("핵심기대", "핵심문제", 1.0),
    ("핵심역할", "핵심기대", 0.8),
]


def _field(d: dict, name: str) -> str | None:
    v = d.get(name)
    return v if v and v != "null" else None


def _parse_chain(chain: str) -> list[str]:
    return [p.strip() for p in chain.split("->")]


def _nodes_of(chains: list[str]) -> list[str]:
    out: list[str] = []
    for c in chains or []:
        out.extend(_parse_chain(c))
    return out


def _edges_of(chains: list[str]) -> list[str]:
    out: list[str] = []
    for c in chains or []:
        parts = _parse_chain(c)
        for i in range(len(parts) - 1):
            out.append(f"{parts[i]} -> {parts[i + 1]}")
    return out


class MeaningMatchScorer:
    """이력서 1건에 대해 여러 JD를 반복 비교할 때 이력서 쪽 임베딩을
    한 번만 계산해서 재사용하기 위한 클래스. JD 쪽 필드/그래프 임베딩은
    호출부(candidate_search.py)가 배치로 미리 계산해서 캐시에 넣어두고,
    이 클래스는 그 캐시를 조회만 한다(중복 임베딩 계산 방지)."""

    def __init__(self, resume_short_semantic: dict, resume_semantic_graph: list[str]):
        self._model = _load_embedding_model()
        self._emb_cache: dict[str, np.ndarray] = {}

        self.resume_sem = resume_short_semantic
        self.resume_graph = resume_semantic_graph or []
        self.resume_nodes = _nodes_of(self.resume_graph)
        self.resume_edges = _edges_of(self.resume_graph)

        self.r_role_e = self._emb(_field(self.resume_sem, "핵심역할"))
        self.r_prob_e = self._emb(_field(self.resume_sem, "핵심문제"))
        self.r_exp_e = self._emb(_field(self.resume_sem, "핵심기대"))
        self.r_node_embs = self._emb_many(self.resume_nodes)
        self.r_edge_embs = self._emb_many(self.resume_edges)

    def _emb(self, text: str | None) -> np.ndarray | None:
        if not text:
            return None
        if text not in self._emb_cache:
            self._emb_cache[text] = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return self._emb_cache[text]

    def _emb_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384))
        missing = [t for t in texts if t not in self._emb_cache]
        if missing:
            # 2026-07-19(실측 근거: 프로파일링 결과 with_rep 621건의
            # preload()가 23.69초 소요 - JD의 노드/엣지 텍스트는 job의
            # jd_semantic_graph가 안 바뀌는 한 결정적인데도 매 검색마다
            # 처음부터 encode()했다. candidate_search.py가 job 본문
            # 텍스트에 이미 쓰고 있는 것과 같은 텍스트 해시 기준 영구
            # 캐시를 재사용한다 - 점수식은 그대로, encode() 호출만
            # 캐시 미스일 때만 발생한다.
            from resume_input.candidate_search import _load_cached_job_text_embeddings, _save_job_text_embeddings

            cached = _load_cached_job_text_embeddings(missing)
            self._emb_cache.update(cached)
            still_missing = [t for t in missing if t not in self._emb_cache]
            if still_missing:
                embs = self._model.encode(still_missing, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
                new_cache = {t: e for t, e in zip(still_missing, embs)}
                self._emb_cache.update(new_cache)
                _save_job_text_embeddings(new_cache)
        return np.array([self._emb_cache[t] for t in texts])

    def preload(self, jd_semantics: list[dict], jd_graphs: list[list[str]]) -> None:
        """여러 JD의 필드/그래프 텍스트를 한 번에 배치 임베딩한다(호출부가
        전체 후보 풀을 순회하기 전에 1회 호출) - 건별로 encode()를 부르면
        건당 오버헤드가 커진다(실측: 배치 임베딩이 10배 이상 빠름)."""
        texts: list[str] = []
        for sem in jd_semantics:
            for fname in ("핵심역할", "핵심문제", "핵심기대"):
                v = _field(sem, fname)
                if v:
                    texts.append(v)
        for graph in jd_graphs:
            texts.extend(_nodes_of(graph))
            texts.extend(_edges_of(graph))
        self._emb_many(list(dict.fromkeys(texts)))  # dedup 유지 순서

    @staticmethod
    def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    @staticmethod
    def _set_max_sim(embs_a: np.ndarray, embs_b: np.ndarray) -> float:
        if len(embs_a) == 0 or len(embs_b) == 0:
            return 0.0
        sim = embs_a @ embs_b.T
        return float((sim.max(axis=1).mean() + sim.max(axis=0).mean()) / 2)

    def m3_score(self, jd_sem: dict) -> float:
        """Rule + Embedding Hybrid - 사람이 정한 필드쌍을 임베딩으로 비교."""
        total_w = total_s = 0.0
        resume_embs = {"핵심역할": self.r_role_e, "핵심문제": self.r_prob_e, "핵심기대": self.r_exp_e}
        for r_field, jd_field, w in _RULE_PAIRS:
            s = self._cosine(resume_embs[r_field], self._emb(_field(jd_sem, jd_field)))
            total_s += w * s
            total_w += w
        return total_s / total_w if total_w else 0.0

    def m4_score(self, jd_graph: list[str]) -> float:
        """Graph Node+Edge Matching - semantic_graph 노드/엣지 집합 최대매칭."""
        jd_nodes = _nodes_of(jd_graph)
        node_score = self._set_max_sim(self.r_node_embs, self._emb_many(jd_nodes))
        jd_edges = _edges_of(jd_graph)
        if self.resume_edges and jd_edges:
            edge_score = self._set_max_sim(self.r_edge_embs, self._emb_many(jd_edges))
        else:
            edge_score = node_score
        return 0.5 * node_score + 0.5 * edge_score

    def score(self, jd_sem: dict, jd_graph: list[str]) -> dict:
        """M3, M4를 모두 계산해서 반환한다. 결합(M3+M4)은 전체 후보
        풀에 대해 정규화가 필요하므로 여기서 하지 않는다 - 호출부
        (candidate_search.py)가 전체 점수 배열을 모은 뒤 정규화+평균한다."""
        return {"m3_score": self.m3_score(jd_sem), "m4_score": self.m4_score(jd_graph)}


def combine_m3_m4(m3_scores: list[float], m4_scores: list[float]) -> list[float]:
    """M3, M4 점수를 각각 표준화(z-score)한 뒤 0.5:0.5로 평균한다.
    전체 후보 풀 단위로 한 번에 호출해야 한다(개별 건마다 정규화하면
    안 됨 - 정규화 기준이 후보 풀 전체 분포여야 의미가 있다)."""
    m3 = np.array(m3_scores)
    m4 = np.array(m4_scores)
    m3_n = (m3 - m3.mean()) / (m3.std() + 1e-8)
    m4_n = (m4 - m4.mean()) / (m4.std() + 1e-8)
    return list(0.5 * m3_n + 0.5 * m4_n)


def has_valid_representation(job: dict) -> bool:
    """이 job이 M3+M4로 채점 가능한 Representation을 갖고 있는지 확인.
    없거나(생성 실패/누락) 파싱이 안 되면 False - 호출부는 이 경우 BM25
    fallback을 써야 한다(candidate_search.py 참고)."""
    sem_json = job.get("jd_short_semantic")
    graph_json = job.get("jd_semantic_graph")
    if not sem_json or not graph_json:
        return False
    try:
        sem = json.loads(sem_json)
        graph = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(sem, dict) or not isinstance(graph, list):
        return False
    # 4필드가 전부 null/누락이면 사실상 빈 Representation - fallback 대상.
    if not any(_field(sem, f) for f in ("핵심역할", "핵심문제", "핵심기대")):
        return False
    return True
