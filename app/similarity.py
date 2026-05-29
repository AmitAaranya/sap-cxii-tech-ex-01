from app.model import SearchResponse, SimilarProduct
from app.store import feature_vector_store, semantic_store

_RRF_K = 60


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = _RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def find_similar_products(product_id: str, num_similar: int = 10) -> SearchResponse:
    feat_ids: list[str] = feature_vector_store.search(product_id, top_k=num_similar * 5)  # type: ignore[assignment]
    sem_ids: list[str] = semantic_store.search(product_id, top_k=num_similar * 5)  # type: ignore[assignment]

    merged = _reciprocal_rank_fusion([feat_ids, sem_ids])
    return SearchResponse(
        data=[
            SimilarProduct(id=pid, score=round(score, 6))
            for pid, score in merged[:num_similar]
        ]
    )
