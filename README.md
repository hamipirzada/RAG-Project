# Production-Grade RAG System

A ground-up implementation of a hybrid retrieval-augmented generation (RAG) pipeline — built without frameworks to demonstrate genuine understanding of every component, from chunking strategy through evaluation.

Built as Project 1 of the [GenAI Engineer Interview Mastery Program](https://github.com/hamipirzada).

---

## What This Is

Most RAG tutorials wrap LangChain around an OpenAI API call. This project implements every layer from scratch:

- **Recursive chunker** with configurable size and overlap
- **Dual-index retrieval**: dense (FAISS + sentence-transformers) and sparse (BM25) running in parallel
- **Reciprocal Rank Fusion** to merge ranked lists without score-scale incompatibility
- **Cross-encoder re-ranking** for precise relevance scoring on the fused candidate set
- **Confidence thresholding** — the system abstains rather than generating from low-relevance context
- **Grounded generation** with numbered citations and explicit refuse-if-unsure instruction
- **RAGAS-style evaluation** — faithfulness and context recall scored by LLM-as-judge against a golden dataset

---

## Architecture

```
INGESTION (offline)
  Raw documents (PDF, TXT)
      → Recursive chunker (paragraph → sentence → word fallback)
      → Bi-encoder embedding  (all-MiniLM-L6-v2, 384-dim)
      → FAISS dense index     (IndexFlatIP, exact cosine)
      → BM25 sparse index     (BM25Okapi)

RETRIEVAL (per query)
  User query
      → Dense retrieval       top-20 by cosine similarity
      → Sparse retrieval      top-20 by BM25 score
      → RRF fusion            rank-based merge, k=60
      → Cross-encoder rerank  top-5 by joint (query, chunk) scoring
      → Confidence threshold  abstain if no chunk passes minimum score

GENERATION
  Filtered context + query
      → Grounded prompt       cite-or-refuse instruction
      → LLM generation        temperature=0.1 for factual grounding
      → Structured response   answer + source citations

EVALUATION
  Golden dataset (query, ground truth, relevant doc IDs)
      → Faithfulness          LLM-as-judge: fraction of claims grounded in context
      → Context Recall        set overlap: fraction of relevant docs retrieved
```

---

## Design Decisions

### Why hybrid retrieval?

Dense and sparse retrieval fail in complementary ways. BM25 misses semantic matches ("heart attack" vs "myocardial infarction"). Dense retrieval misses exact rare-term matches (product codes, legal citations). Running both and fusing with RRF captures what either alone would miss.

### Why RRF instead of score averaging?

Dense similarity scores (cosine, bounded [-1, 1]) and BM25 scores (unbounded positive floats) live on incompatible scales. Averaging them is meaningless. RRF works purely on rank position, making it scale-agnostic. Documents retrieved by both systems receive contributions from both — rewarding cross-system consistency.

### Why a cross-encoder for re-ranking?

Bi-encoders embed query and document independently — they can't capture token-level interactions between them. A cross-encoder processes the full (query, document) pair jointly, letting attention heads match specific query terms to specific document phrases. This is more accurate but too slow for first-pass retrieval over millions of documents. The two-stage pattern (bi-encoder for breadth, cross-encoder for precision) gets near-cross-encoder quality at bi-encoder speed.

### Why confidence thresholding?

A RAG system that generates from low-relevance context is worse than one that admits ignorance — it produces confident-sounding hallucinations. If the best-matched chunk scores below the threshold, the system returns a structured refusal rather than a fabricated answer.

### Why temperature=0.1 for generation?

Low temperature keeps the model close to its highest-probability completions, which for a well-prompted grounded generation task means staying closer to the retrieved context rather than drifting toward parametric memory. Not 0.0 — a small amount of stochasticity prevents pathological repetition.

---

## Project Structure

```
project1_rag/
├── rag/
│   ├── __init__.py
│   ├── ingestion.py      Document loading and recursive chunking
│   ├── indexing.py       Dense FAISS index (bi-encoder embeddings)
│   ├── bm25_index.py     Sparse BM25 index
│   ├── retrieval.py      Hybrid retrieval with Reciprocal Rank Fusion
│   ├── reranker.py       Cross-encoder re-ranking
│   ├── context.py        Confidence thresholding and context assembly
│   ├── generation.py     Grounded generation with citations
│   ├── evaluation.py     RAGAS-style faithfulness and recall evaluation
│   └── pipeline.py       End-to-end assembled pipeline
├── data/                 Documents go here (PDF, TXT)
├── main.py               Usage example and golden dataset evaluation
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hamipirzada/project1-rag
cd project1-rag

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
sentence-transformers==2.7.0
faiss-cpu==1.8.0
rank-bm25==0.2.2
transformers==4.41.0
groq==0.9.0
PyPDF2==3.0.1
numpy==1.26.4
pydantic==2.7.0
torch
```

---

## Usage

### Set API Key

```bash
export GROQ_API_KEY="your-groq-api-key"
# Get a free key at: https://console.groq.com
```

### Add Documents

```bash
mkdir -p data
cp your_documents.pdf data/
```

### Run

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    groq_api_key="your-key",
    chunk_size=500,
    chunk_overlap=50,
    retrieval_top_k=20,
    reranker_top_k=5,
)

pipeline.ingest(["data/your_document.pdf"])

result = pipeline.query(
    "Your question here?",
    verbose=True,
)

print(result["answer"])
print("Sources:", result["sources"])
```

### Run Evaluation

```python
from rag.evaluation import EvalSample, run_evaluation
from groq import Groq

golden_dataset = [
    EvalSample(
        query="What is the Chinchilla rule of thumb?",
        ground_truth_answer="Approximately 20 training tokens per model parameter.",
        relevant_doc_ids=["GenAI_Engineers_Handbook.pdf"],
    ),
    # Add more samples...
]

client = Groq(api_key="your-key")
results = run_evaluation(pipeline, golden_dataset, client)
```

---

## Evaluation Results

Evaluated over 3 golden samples from the GenAI Engineer's Handbook (272 pages).

| Metric | Score | Notes |
|---|---|---|
| Context Recall | **1.000** | All queries retrieved from the correct source document |
| Faithfulness | 0.517 | Claims grounded in retrieved context — limited by mixed-topic chunks |

**Known limitation:** the current 500-token chunker produces chunks that mix topics (e.g., a chunk about temperature defaults that also contains Chinchilla scaling content mid-way through). This causes the cross-encoder to rank these chunks low for Chinchilla queries, and the LLM supplements retrieved context with parametric memory — reducing faithfulness. Smaller chunks (200-300 tokens) or semantic chunking would isolate topics and improve this metric.

---

## Retrieval Pipeline in Detail

### Reciprocal Rank Fusion (RRF)

For a document ranked at position `r` in retrieval system `i`:

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))
```

With `k=60` (empirically validated smoothing constant), a document ranked #2 in dense and #4 in BM25 scores:

```
1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03175
```

Higher than a document ranked #1 in dense only:
```
1/(60+1) = 0.01639
```

Documents retrieved by both systems consistently outrank those retrieved by one — RRF rewards cross-system agreement.

### Cross-Encoder Re-ranking

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) processes each (query, chunk) pair jointly:

```
Input:  [CLS] query tokens [SEP] document tokens [SEP]
Output: scalar relevance score
```

Unlike bi-encoders (which embed query and document independently), the cross-encoder allows every query token to attend to every document token — capturing whether the document contains the specific answer, not just whether it's about the same topic.

---

## Interview Talking Points

**"Walk me through your RAG project."**

Four design decisions drove this system:

1. **Hybrid retrieval with RRF** — dense and sparse retrieval fail in complementary ways; combining them via rank-based fusion improves recall over either alone without score-incompatibility issues.

2. **Two-stage retrieval** — retrieve top-20 cheaply with bi-encoder + FAISS (O(log n)), then re-rank top-20 precisely with cross-encoder (O(20) forward passes). Near-cross-encoder quality at bi-encoder speed for the first stage.

3. **Confidence thresholding** — if no retrieved chunk passes the relevance threshold, the system abstains with a structured refusal rather than generating from poor context. Prevents hallucination from low-quality retrieval.

4. **Evaluation-driven development** — faithfulness (are claims grounded in context?) and context recall (did we retrieve the right documents?) run against a golden dataset. Any change to prompts, chunking, or models requires metrics to hold or improve before deployment.

---

## Candidate
