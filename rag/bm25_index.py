import re
from rank_bm25 import BM25Okapi
from .ingestion import Chunk


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, split on non-alphanumeric characters,
    remove empty tokens. Production systems would use stemming/lemmatization.
    """
    return [
        token for token in re.split(r"[^a-zA-Z0-9]", text.lower())
        if token
    ]


class SparseIndex:
    """
    BM25-backed sparse retrieval index.
    BM25Okapi is the standard BM25 variant with document-length normalization.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Tokenize all chunks and build the BM25 index."""
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """
        Retrieve top_k chunks with highest BM25 scores for the query.
        Returns list of (chunk, bm25_score) tuples.
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built — call add() first")

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get indices of top_k scores (argsort descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # exclude zero-score results
        ]