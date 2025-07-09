from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_tensor=True)

def cosine_sim(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    return float(cosine_similarity([a], [b])[0][0])

def semantic_context_recall(retrieved: List[str], reference: List[str], threshold=0.75) -> float:
    """How many reference docs were matched by any retrieved doc."""
    if not reference:
        return 0.0
    retrieved_emb = embed(retrieved)
    reference_emb = embed(reference)

    matches = 0
    for ref_emb in reference_emb:
        sims = cosine_similarity([ref_emb], retrieved_emb)[0]
        if np.max(sims) >= threshold:
            matches += 1
    return matches / len(reference)

def semantic_precision(retrieved: List[str], reference: List[str], threshold=0.75) -> float:
    """How many retrieved docs matched at least one reference doc."""
    if not retrieved:
        return 0.0
    retrieved_emb = embed(retrieved)
    reference_emb = embed(reference)

    matches = 0
    for ret_emb in retrieved_emb:
        sims = cosine_similarity([ret_emb], reference_emb)[0]
        if np.max(sims) >= threshold:
            matches += 1
    return matches / len(retrieved)

def context_relevance(retrieved: List[str], reference: List[str]) -> float:
    """
    Global similarity between retrieved context and reference content.
    Simulates whether the overall context brings relevant info.
    """
    if not reference or not retrieved:
        return 0.0
    retrieved_text = "\n".join(retrieved)
    reference_text = "\n".join(reference)
    return cosine_sim(embed([retrieved_text])[0], embed([reference_text])[0])

def evaluate_rag(
    query: str,
    generated_answer: str,  # Ignored if no reference answer
    retrieved_docs: List[str],
    reference_docs: List[str],
    reference_answer: str = ""
) -> dict:
    return {
        "semantic_context_recall": semantic_context_recall(retrieved_docs, reference_docs),
        "semantic_precision": semantic_precision(retrieved_docs, reference_docs),
        "context_relevance": context_relevance(retrieved_docs, reference_docs)
    }
