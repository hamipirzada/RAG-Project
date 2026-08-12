from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .ingestion import Chunk


class CrossEncoderReranker:
    """
    Cross-encoder re-ranker using a sequence classification model.
    Takes (query, document) pairs and scores them jointly,
    capturing fine-grained query-document interactions missed by bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Score all (query, candidate_chunk) pairs with the cross-encoder.
        Returns top_k candidates sorted by cross-encoder score descending.

        This is the expensive step — O(n) model forward passes where n = len(candidates).
        Only call after first-pass retrieval has reduced candidates to a manageable set
        (typically 20-50 documents, not millions).
        """
        if not candidates:
            return []

        chunks = [chunk for chunk, _ in candidates]
        texts = [chunk.text for chunk in chunks]

        # Tokenize all (query, document) pairs as a batch
        # The cross-encoder processes query + document jointly via [CLS] q [SEP] d [SEP]
        encoded = self.tokenizer(
            [query] * len(texts),  # repeat query once per document
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**encoded).logits
            # ms-marco models output a single relevance logit — squeeze to scalar
            scores = logits.squeeze(-1).tolist()

        # Normalize from list if single item
        if isinstance(scores, float):
            scores = [scores]

        # Pair chunks with their cross-encoder scores and sort descending
        reranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(reranked[:top_k])