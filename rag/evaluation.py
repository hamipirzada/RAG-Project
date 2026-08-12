from dataclasses import dataclass
from groq import Groq
from .context import RetrievalResult


@dataclass
class EvalSample:
    """A single evaluation sample from the golden dataset."""
    query: str
    ground_truth_answer: str
    relevant_doc_ids: list[str]  # filenames of documents containing the answer


@dataclass
class EvalResult:
    """Evaluation scores for a single query."""
    query: str
    generated_answer: str
    faithfulness: float       # 0.0-1.0: claims supported by context
    answer_relevance: float   # 0.0-1.0: answer addresses the question
    context_recall: float     # 0.0-1.0: relevant docs retrieved
    retrieved_sources: list[str]


def score_faithfulness(
    answer: str,
    context_chunks: list[str],
    client: Groq,
    model: str = "llama-3.1-8b-instant",
) -> float:
    """
    Score faithfulness: fraction of answer claims supported by retrieved context.
    Uses LLM-as-judge to check each atomic claim against the context.
    """
    if not answer or not context_chunks:
        return 0.0

    context_text = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""Given the following context passages and an answer, evaluate faithfulness.

Context:
{context_text}

Answer to evaluate:
{answer}

Task: List each factual claim in the answer, then for each claim indicate whether 
it is supported by the context (YES) or not supported / contradicted (NO).

Respond in this exact format:
CLAIM 1: [claim text] | SUPPORTED: YES/NO
CLAIM 2: [claim text] | SUPPORTED: YES/NO
...
FAITHFULNESS_SCORE: [fraction of supported claims, e.g. 0.75]"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic for evaluation
    )

    content = response.choices[0].message.content

    # Extract score from last line
    for line in content.split("\n"):
        if "FAITHFULNESS_SCORE:" in line:
            try:
                score = float(line.split(":")[-1].strip())
                return min(max(score, 0.0), 1.0)  # clamp to [0,1]
            except ValueError:
                pass
    return 0.5  # default if parsing fails


def score_context_recall(
    retrieved_sources: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """
    Score context recall: fraction of relevant documents that were retrieved.
    This is a simple set-overlap metric — no LLM needed.

    relevant_doc_ids: ground-truth list of docs containing the answer
    retrieved_sources: filenames of docs whose chunks were actually retrieved
    """
    if not relevant_doc_ids:
        return 1.0  # no relevant docs expected = perfect recall vacuously

    retrieved_set = set(retrieved_sources)
    relevant_set = set(relevant_doc_ids)
    overlap = retrieved_set & relevant_set

    return len(overlap) / len(relevant_set)


def run_evaluation(
    pipeline,
    golden_dataset: list[EvalSample],
    client: Groq,
) -> list[EvalResult]:
    """
    Run the full evaluation suite over a golden dataset.
    Returns per-sample EvalResult objects and prints aggregate metrics.
    """
    results = []

    for i, sample in enumerate(golden_dataset, 1):
        print(f"Evaluating sample {i}/{len(golden_dataset)}: {sample.query[:50]}...")

        # Get pipeline response
        response = pipeline.query(sample.query, verbose=False)

        # Score faithfulness (LLM-as-judge)
        context_texts = [
            chunk.text for chunk in
            (pipeline._all_chunks[:response["num_chunks_used"]]
             if response["retrieval_succeeded"] else [])
        ]
        faithfulness = score_faithfulness(
            response["answer"],
            context_texts,
            client,
        )

        # Score context recall (set overlap)
        context_recall = score_context_recall(
            response.get("sources", []),
            sample.relevant_doc_ids,
        )

        result = EvalResult(
            query=sample.query,
            generated_answer=response["answer"],
            faithfulness=faithfulness,
            answer_relevance=0.0,  # simplified: implement similarly to faithfulness
            context_recall=context_recall,
            retrieved_sources=response.get("sources", []),
        )
        results.append(result)

    # Aggregate metrics
    avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
    avg_recall = sum(r.context_recall for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({len(results)} samples)")
    print(f"{'='*50}")
    print(f"Average Faithfulness:    {avg_faithfulness:.3f}")
    print(f"Average Context Recall:  {avg_recall:.3f}")
    print(f"{'='*50}")

    return results