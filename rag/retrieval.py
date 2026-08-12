from .ingestion import Chunk
from .indexing import DenseIndex
from .bm25_index import SparseIndex


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    sparse_results: list[tuple[Chunk, float]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Fuse dense and sparse retrieval results using Reciprocal Rank Fusion.

    RRF score for a document d:
        score(d) = sum over all systems: 1 / (k + rank_of_d_in_system)

    k=60 is the empirically validated smoothing constant that prevents
    top-ranked documents from completely dominating the fused ranking.

    Documents appearing in both systems receive contributions from both,
    rewarding cross-system consistency.
    """
    rrf_scores: dict[int, float] = {}
    # Map chunk_id to chunk object for result assembly
    chunk_map: dict[int, Chunk] = {}

    def _add_results(results: list[tuple[Chunk, float]], system_weight: float = 1.0):
        for rank, (chunk, _score) in enumerate(results, start=1):
            chunk_id = id(chunk)  # use object id as unique key
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + system_weight * (1.0 / (k + rank))
            )

    _add_results(dense_results)
    _add_results(sparse_results)

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids]


def hybrid_retrieve(
    query: str,
    dense_index: DenseIndex,
    sparse_index: SparseIndex,
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[tuple[Chunk, float]]:
    """
    Full hybrid retrieval pipeline:
    1. Dense retrieval (semantic similarity via FAISS)
    2. Sparse retrieval (keyword matching via BM25)
    3. RRF fusion of both ranked lists
    Returns top_k fused results.
    """
    dense_results = dense_index.retrieve(query, top_k=top_k)
    sparse_results = sparse_index.retrieve(query, top_k=top_k)
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k)
    return fused[:top_k]