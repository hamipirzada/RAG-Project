import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .ingestion import Chunk


class DenseIndex:
    """FAISS backed dense vector index using bi-encoder embeddings. 
    Supports add, save, load, and top-k retrieval"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

        # IndexFlatIP: exact inner product (cosine similarity when normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: list[Chunk] = []


    def add(self, chunks: list[Chunk]) ->None:
        """Embed all chunks and add to the FAISS index"""

        texts = [chunk.text for chunk in chunks]

        # normalize_embeddings = True converts inner product to cosine similarity
        embeddings = self.model.encode(
            texts,
            normalize_embeddings = True,
            show_progress_bar = True,
            batch_size = 32,
        )
        self.index.add(np.array(embeddings, dtype = "float32"))
        self.chunks.extend(chunks)


    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """Retrieve top_k most similar chunks to the query.
        Returns list of (chunk, cosine_similarity_score) tuples"""

        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings = True
        )
        scores, indices = self.index.search(
            np.array(query_embedding, dtype = "float32"),
            top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                # FAISS returns -1 for empty slots
                results.append((self.chunks[idx], float(score)))
        return results


    def embed_query(self, query: str) -> np.ndarray:
        """Returns the normalized query embedding vector"""

        return self.model.encode([query], normalized_embeddings = True)[0]