# RAG Project

A hybrid retrieval-augmented generation pipeline built from scratch — no LangChain, no LlamaIndex. Every component is implemented directly so I actually understand what's happening under the hood.

Tested on a 272-page technical handbook (GenAI Engineer's Handbook) and a 19-page study roadmap — 716 chunks total.

---

## Why I built this

Most RAG tutorials are essentially: call an embedding API, store in a vector DB, retrieve, prompt GPT. That works, but you don't learn anything about why retrieval fails or how to fix it.

I wanted to understand the actual failure modes — why dense retrieval misses exact keyword matches, why BM25 can't handle synonyms, why a high cosine similarity score doesn't mean the chunk actually answers the question. So I built each layer manually, including a simple evaluation loop to measure what's actually happening.

---

## What's in here

```
rag/
├── ingestion.py      PDF/text loading, recursive chunking with overlap
├── indexing.py       FAISS dense index (sentence-transformers embeddings)
├── bm25_index.py     BM25 sparse index
├── retrieval.py      Hybrid retrieval + Reciprocal Rank Fusion
├── reranker.py       Cross-encoder re-ranking
├── context.py        Confidence thresholding, context assembly
├── generation.py     Grounded generation with citations
├── evaluation.py     Faithfulness + context recall scoring
└── pipeline.py       Assembled end-to-end pipeline
```

---

## How it works

**Ingestion:** documents are split with a recursive chunker (paragraph → sentence → word fallback), embedded with `all-MiniLM-L6-v2`, and stored in two indexes simultaneously — FAISS for dense retrieval and BM25 for keyword retrieval.

**Retrieval:** both indexes run independently for each query. Dense retrieval handles semantic matches ("myocardial infarction" finding documents about "heart attack"). BM25 handles exact terms that dense retrieval misses (rare names, product codes, technical jargon). The two ranked lists are merged using Reciprocal Rank Fusion — rank-based, so there's no score-incompatibility problem between cosine similarities and BM25 scores.

**Re-ranking:** the top 20 fused candidates go through a cross-encoder (`ms-marco-MiniLM-L-6-v2`), which scores each (query, chunk) pair jointly. This is more accurate than cosine similarity because the query and document attend to each other's tokens directly. Too slow to run on the whole index — that's why the first stage narrows it to 20 first.

**Generation:** chunks that don't pass a relevance threshold are dropped. The model is instructed to cite passage numbers and refuse if the answer isn't in the context. Temperature is set to 0.1 — low enough to stay grounded, not 0 because that causes repetition issues.

**Evaluation:** a small golden dataset (query + expected answer + which document should be retrieved) runs through the full pipeline. Faithfulness is scored by an LLM judge that checks whether each claim in the answer is actually supported by the retrieved context. Context recall is a simple set-overlap check — did we retrieve the right document?

---

## Actual output

```
Loading: data/GenAI_Engineers_Handbook.pdf
  → 697 chunks from data/GenAI_Engineers_Handbook.pdf
Loading: data/GenAI_Interview_Roadmap_Hamid.pdf
  → 19 chunks from data/GenAI_Interview_Roadmap_Hamid.pdf

Total chunks: 716
Building dense index...
Building sparse index...
Ingestion complete.

Query: What is the Chinchilla rule of thumb for training tokens per parameter?
Hybrid retrieval returned 20 candidates
Re-ranked to top 5 candidates

Answer: The Chinchilla rule of thumb is Doptimal ≈ 20 × N, where Doptimal is the
optimal number of training tokens and N is the number of model parameters.
This means that to train a model of N parameters compute-optimally, feed it
roughly 20N tokens. [2]
Sources: ['GenAI_Engineers_Handbook.pdf']
Pipeline: 20 retrieved → 5 reranked → 1 used

EVALUATION RESULTS (3 samples)
Average Faithfulness:    0.267
Average Context Recall:  1.000
```

Context recall is 1.0 — it's finding the right documents every time. Faithfulness is 0.267 — lower than I'd like.

The faithfulness issue is a chunking problem, not a generation problem. The Chinchilla content sits inside a 500-token chunk that's mostly about temperature defaults — the chunk's embedding reflects the dominant topic (temperature), so the cross-encoder ranks it low for Chinchilla queries. Only 1 chunk passes the relevance threshold, giving the model limited context to cite, so it supplements with parametric memory. Smaller chunks (200-300 tokens) or semantic chunking would isolate topics and fix this.

I'm leaving the current scores in rather than tuning them away — the gap between recall (1.0) and faithfulness (0.267) is itself informative. It shows the retrieval is finding the right documents but the chunking strategy is making it hard for the re-ranker to surface the relevant passage cleanly.

---

## Setup

```bash
git clone https://github.com/hamipirzada/RAG-Project.git
cd RAG-Project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — the free tier is enough for testing.

```bash
export GROQ_API_KEY="your-key-here"
```

Add your own documents to `data/` (PDF or TXT), update the file paths in `main.py`, and run:

```bash
python main.py
```

---

## What I'd change with more time

- Semantic chunking based on embedding similarity between adjacent sentences — would eliminate the mixed-topic chunk problem that's hurting faithfulness
- Contextual retrieval: prepend a short LLM-generated summary to each chunk before embedding, so chunks like "this shall apply as described in Section 4.2" have enough context to be retrievable
- Langfuse tracing on every query — right now the pipeline logs to stdout, which isn't useful in production
- A/B evaluation comparing different chunking strategies against the same golden dataset rather than tuning by intuition

---

## Stack

- `sentence-transformers` — bi-encoder embeddings
- `faiss-cpu` — dense vector index
- `rank-bm25` — sparse BM25 index
- `transformers` — cross-encoder re-ranking
- `groq` — LLM generation and evaluation (free tier)
- `PyPDF2` — PDF text extraction

---

Hamid Mujtaba · [LinkedIn](https://linkedin.com/in/hamid-mujtaba) · [hamipirzada@gmail.com](mailto:hamipirzada@gmail.com)
