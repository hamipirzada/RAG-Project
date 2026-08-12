from dataclasses import dataclass
from .ingestion import Chunk


@dataclass
class RetrievalResult:
    """Structured result from the full retrieval pipeline."""
    chunks: list[Chunk]
    scores: list[float]
    query: str
    retrieval_succeeded: bool
    failure_reason: str = ""


CONFIDENCE_THRESHOLD = 0.0  # cross-encoder scores are unbounded; tune empirically
# For ms-marco-MiniLM: scores > 0 generally indicate relevance;
# scores < -5 indicate very low relevance. Tune based on your data.
CROSS_ENCODER_THRESHOLD = -2.0


def apply_confidence_threshold(
    reranked: list[tuple[Chunk, float]],
    threshold: float = CROSS_ENCODER_THRESHOLD,
) -> list[tuple[Chunk, float]]:
    """
    Filter out chunks whose cross-encoder score falls below the threshold.
    A score below threshold indicates the model judged the chunk as not
    relevant to the query — including such chunks would degrade generation
    quality and risk hallucination from poor context.
    """
    return [(chunk, score) for chunk, score in reranked if score >= threshold]


def assemble_context(
    reranked: list[tuple[Chunk, float]],
    query: str,
    threshold: float = CROSS_ENCODER_THRESHOLD,
) -> RetrievalResult:
    """
    Assemble the final retrieval result for generation.
    Applies confidence thresholding and returns a structured result
    indicating whether retrieval succeeded (found relevant chunks) or
    whether the system should abstain from answering.
    """
    filtered = apply_confidence_threshold(reranked, threshold)

    if not filtered:
        return RetrievalResult(
            chunks=[],
            scores=[],
            query=query,
            retrieval_succeeded=False,
            failure_reason="No retrieved chunks exceeded the relevance threshold. "
                           "The answer may not be in the indexed documents."
        )

    chunks = [chunk for chunk, _ in filtered]
    scores = [score for _, score in filtered]

    return RetrievalResult(
        chunks=chunks,
        scores=scores,
        query=query,
        retrieval_succeeded=True,
    )