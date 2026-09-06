"""
resume_input/layer_m3.py

Rule/Layer Representation 전용 M3 스코어러(Adapter, 2026-07-25 M3
Production Migration). 기존 meaning_matching.MeaningMatchScorer(핵심역할/
핵심문제/핵심기대 3필드를 __init__/m3_score()/preload()에 하드코딩)는
건드리지 않는다 - Rule Representation(task/skill/qualification)은
스키마가 완전히 달라서, 기존 클래스를 억지로 확장하면 위험하다(실측 감사,
docs/verification/2026-07-25_m3_representation_ab/summary.md §11 -
"핵심역할" 등 키로 _field()가 None을 반환해 에러 없이 점수가 조용히
0이 되는 실패 모드를 확인함). 그래서 별도 파일·클래스로 둔다 -
candidate_search.py가 Rule Representation이 유효한 job에만 이 클래스를
쓰고, 나머지는 기존 MeaningMatchScorer.m3_score()로 폴백한다.

책임 분리(2026-07-25 리팩터링, 사용자 지적 - meaning_matching.py가
Scorer+Representation Builder+Embedding Cache 접근까지 한 파일에 다
모이면 유지보수가 어려워지고, candidate_search.py<->meaning_matching.py
순환 의존 위험도 커진다):
- 레이어별 텍스트를 "무엇으로 만들지"는 layer_representation.py 책임.
- 임베딩 캐시를 "어떻게 읽고 쓰는지"는 candidate_search.py 책임(이
  모듈은 그 함수를 호출만 함, MeaningMatchScorer._emb_many와 동일한
  기존 패턴 재사용 - 새로 만든 의존이 아니다).
- 이 모듈은 오직 "레이어별 코사인 유사도를 어떻게 결합할지"만 안다.
"""
from __future__ import annotations

import numpy as np

from resume_input.candidate_search import _load_embedding_model
from resume_input.layer_representation import SHARED_LAYERS


class LayerM3Scorer:
    """레이어별 문자열 1개씩, 동일 레이어끼리만 비교(task<->task 등),
    균등 가중치(단순 평균)로 코사인 유사도를 결합한다. 기존
    MeaningMatchScorer.m3_score()의 _RULE_PAIRS(3필드 비대칭 가중치)는
    예전 3필드 구조에 맞춰 사람이 튜닝한 규칙이라 5-레이어 구조에 그대로
    못 옮긴다 - 교차 레이어 가중치 같은 새 튜닝 변수를 추가하지 않기
    위해 균등 가중치로 시작한다(2026-07-25 실측 확정, Recall@100/A-B
    Top20 검증 통과)."""

    def __init__(self, resume_layer_repr: dict[str, str]):
        self._model = _load_embedding_model()
        self._cache: dict[str, np.ndarray] = {}
        self.resume_layer_repr = resume_layer_repr
        self.r_embs = {layer: self._emb(resume_layer_repr.get(layer)) for layer in SHARED_LAYERS}

    def _emb(self, text: str | None) -> np.ndarray | None:
        if not text:
            return None
        if text not in self._cache:
            self._cache[text] = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return self._cache[text]

    def preload(self, jd_layer_reprs: list[dict[str, str]]) -> None:
        """여러 JD의 레이어 텍스트를 한 번에 배치 임베딩한다(호출부가
        전체 후보 풀을 순회하기 전에 1회 호출) - MeaningMatchScorer.
        preload()와 같은 원칙(건별 encode() 호출은 오버헤드가 크다,
        2026-07-19 실측). job 본문 텍스트 영구 캐시(candidate_search.py,
        MeaningMatchScorer._emb_many가 쓰는 것과 동일)를 그대로
        재사용한다 - 검색마다 매번 새로 encode()하지 않는다."""
        from resume_input.candidate_search import _load_cached_job_text_embeddings, _save_job_text_embeddings

        texts: list[str] = []
        for rep in jd_layer_reprs:
            for layer in SHARED_LAYERS:
                v = rep.get(layer)
                if v:
                    texts.append(v)
        texts = list(dict.fromkeys(texts))
        missing = [t for t in texts if t not in self._cache]
        if not missing:
            return
        cached = _load_cached_job_text_embeddings(missing)
        self._cache.update(cached)
        still_missing = [t for t in missing if t not in self._cache]
        if still_missing:
            embs = self._model.encode(still_missing, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            new_cache = {t: e for t, e in zip(still_missing, embs)}
            self._cache.update(new_cache)
            _save_job_text_embeddings(new_cache)

    @staticmethod
    def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None:
            return None
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def score(self, jd_layer_repr: dict[str, str]) -> float:
        """레이어별로 양쪽 다 텍스트가 있을 때만 비교 대상에 넣는다(한쪽
        레이어가 없으면 skip - 0점으로 편향시키지 않음, Problem/Thinking
        처럼 Rule Representation에 아예 없는 레이어는 항상 skip된다)."""
        total_s = total_w = 0.0
        for layer in SHARED_LAYERS:
            r_e = self.r_embs.get(layer)
            j_e = self._emb(jd_layer_repr.get(layer))
            s = self._cosine(r_e, j_e)
            if s is None:
                continue
            total_s += s
            total_w += 1.0
        return total_s / total_w if total_w else 0.0
