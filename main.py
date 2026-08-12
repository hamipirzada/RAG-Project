import os
from groq import Groq
from rag.pipeline import RAGPipeline
from rag.evaluation import EvalSample, run_evaluation

# ── 1. Initialize pipeline ──────────────────────────────────────────────────
pipeline = RAGPipeline(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    chunk_size=500,
    chunk_overlap=50,
    retrieval_top_k=20,
    reranker_top_k=5,
)

# ── 2. Ingest documents ─────────────────────────────────────────────────────
pipeline.ingest([
    "data/GenAI_Engineers_Handbook.pdf",
    "data/GenAI_Interview_Roadmap_Hamid.pdf",
])

# ── 3. Test a single query ───────────────────────────────────────────────────
result = pipeline.query(
    "What is the Chinchilla rule of thumb for training tokens per parameter?",
    verbose=True,
)
print("\nAnswer:", result["answer"])
print("Sources:", result["sources"])
print(f"Pipeline: {result['num_chunks_retrieved']} retrieved → "
      f"{result['num_chunks_reranked']} reranked → "
      f"{result['num_chunks_used']} used")

# ── 4. Golden dataset ───────────────────────────────────────
golden_dataset = [
    EvalSample(
        query="What is the Chinchilla rule of thumb for training tokens per parameter?",
        ground_truth_answer="Approximately 20 training tokens per model parameter.",
        relevant_doc_ids=["GenAI_Engineers_Handbook.pdf"],
    ),
    EvalSample(
        query="What are the four RAGAS metrics?",
        ground_truth_answer="Faithfulness, answer relevance, context precision, and context recall.",
        relevant_doc_ids=["GenAI_Engineers_Handbook.pdf"],
    ),
    EvalSample(
        query="What is RRF and what problem does it solve in hybrid retrieval?",
        ground_truth_answer="Reciprocal Rank Fusion fuses dense and sparse retrieval results using ranks rather than scores, avoiding the score-scale incompatibility between cosine similarity and BM25.",
        relevant_doc_ids=["GenAI_Engineers_Handbook.pdf"],
    ),
]

# ── 5. Run evaluation ────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
eval_results = run_evaluation(pipeline, golden_dataset, client)