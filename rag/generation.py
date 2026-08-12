from groq import Groq
from .context import RetrievalResult


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY using the numbered context passages provided below.
For every claim in your answer, cite the passage number(s) that support it using [1], [2], etc.
If the answer cannot be found in the provided passages, respond with exactly:
"I don't have sufficient information in the provided documents to answer this question."
Do not use any knowledge outside the provided passages."""


def build_context_string(result: RetrievalResult) -> str:
    passages = []
    for i, chunk in enumerate(result.chunks, start=1):
        source = chunk.source.split("/")[-1]
        passages.append(f"[{i}] (Source: {source})\n{chunk.text}")
    return "\n\n".join(passages)


def generate_answer(
    result: RetrievalResult,
    client: Groq,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
) -> dict:
    if not result.retrieval_succeeded:
        return {
            "answer": result.failure_reason,
            "sources": [],
            "retrieval_succeeded": False,
            "model_used": None,
        }

    context_string = build_context_string(result)
    user_message = f"Context passages:\n\n{context_string}\n\nQuestion: {result.query}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )

    answer = response.choices[0].message.content
    sources = list({chunk.source.split("/")[-1] for chunk in result.chunks})

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_succeeded": True,
        "model_used": model,
    }