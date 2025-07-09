from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# filepath: /home/tasniml/RAG/modules/metrics.py
def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        # Return an empty 2D array with the correct embedding dimension
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(texts, convert_to_tensor=False)

# filepath: /home/tasniml/RAG/modules/metrics.py
def cosine_sim(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0  # Return 0 similarity if any input is empty
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

# filepath: /home/tasniml/RAG/modules/metrics.py
def evaluate_rag(
    query: str,
    generated_answer: str,
    retrieved_docs: List[str],
    reference_docs: List[str],
    reference_answer: str = ""
) -> dict:
    if not retrieved_docs or not reference_docs:
        return {
            "semantic_context_recall": 0.0,
            "semantic_precision": 0.0,
            "context_relevance": 0.0
        }
    return {
        "semantic_context_recall": semantic_context_recall(retrieved_docs, reference_docs),
        "semantic_precision": semantic_precision(retrieved_docs, reference_docs),
        "context_relevance": context_relevance(retrieved_docs, reference_docs)
    }