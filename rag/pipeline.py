from groq import Groq
from .ingestion import load_text_file, load_pdf, recursive_chunk, Chunk
from .indexing import DenseIndex
from .bm25_index import SparseIndex
from .retrieval import hybrid_retrieve
from .reranker import CrossEncoderReranker
from .context import assemble_context
from .generation import generate_answer


class RAGPipeline:
    """
    Production-shaped RAG pipeline with hybrid retrieval, re-ranking,
    confidence thresholding, and grounded generation.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(["doc1.pdf", "doc2.txt"])
        result = pipeline.query("What is the refund policy?")
        print(result["answer"])
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        groq_api_key: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_top_k: int = 20,
        reranker_top_k: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_top_k = retrieval_top_k
        self.reranker_top_k = reranker_top_k

        # Initialize all pipeline components
        self.dense_index = DenseIndex(model_name=embedding_model)
        self.sparse_index = SparseIndex()
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self.llm_client = Groq(api_key=groq_api_key)

        self._all_chunks: list[Chunk] = []

    def ingest(self, file_paths: list[str]) -> None:
        """
        Load, chunk, and index all documents.
        Builds both dense (FAISS) and sparse (BM25) indexes.
        """
        all_chunks = []
        for path in file_paths:
            print(f"Loading: {path}")
            if path.endswith(".pdf"):
                doc = load_pdf(path)
            else:
                doc = load_text_file(path)
            chunks = recursive_chunk(doc, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks from {path}")

        print(f"\nTotal chunks: {len(all_chunks)}")
        print("Building dense index...")
        self.dense_index.add(all_chunks)
        print("Building sparse index...")
        self.sparse_index.add(all_chunks)
        self._all_chunks = all_chunks
        print("Ingestion complete.")

    def query(
        self,
        question: str,
        model: str = "llama-3.1-8b-instant",
        verbose: bool = False,
    ) -> dict:
        """
        Full retrieval → re-ranking → generation pipeline for a single question.

        Steps:
        1. Hybrid retrieval (dense + sparse + RRF)
        2. Cross-encoder re-ranking of top-k candidates
        3. Confidence thresholding
        4. Grounded generation with citations

        Returns structured dict with answer, sources, and pipeline metadata.
        """
        if verbose:
            print(f"\nQuery: {question}")

        # Step 1: Hybrid retrieval
        fused_results = hybrid_retrieve(
            query=question,
            dense_index=self.dense_index,
            sparse_index=self.sparse_index,
            top_k=self.retrieval_top_k,
        )

        if verbose:
            print(f"Hybrid retrieval returned {len(fused_results)} candidates")

        # Step 2: Cross-encoder re-ranking
        reranked = self.reranker.rerank(
            query=question,
            candidates=fused_results,
            top_k=self.reranker_top_k,
        )

        if verbose:
            print(f"Re-ranked to top {len(reranked)} candidates")
            for i, (chunk, score) in enumerate(reranked, 1):
                print(f"  [{i}] score={score:.3f} | {chunk.text[:80]}...")

        # Step 3: Confidence thresholding + context assembly
        retrieval_result = assemble_context(reranked, question)

        if not retrieval_result.retrieval_succeeded and verbose:
            print("Warning: No chunks passed confidence threshold — abstaining")

        # Step 4: Grounded generation
        answer = generate_answer(
            result=retrieval_result,
            client=self.llm_client,
            model=model,
        )

        return {
            **answer,
            "num_chunks_retrieved": len(fused_results),
            "num_chunks_reranked": len(reranked),
            "num_chunks_used": len(retrieval_result.chunks),
        }