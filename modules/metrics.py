from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    return model.encode(texts, convert_to_tensor=False)

def cosine_sim(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(cosine_similarity([a], [b])[0][0])

def query_document_relevance(query: str, retrieved_docs: List[str], threshold=0.3) -> float:
    """
    Measure how relevant the retrieved documents are to the query.
    Returns the average similarity between query and retrieved docs.
    """
    if not retrieved_docs or not query.strip():
        return 0.0
    
    query_emb = embed([query])[0]
    doc_embeddings = embed(retrieved_docs)
    
    if doc_embeddings.size == 0:
        return 0.0
    
    similarities = []
    for doc_emb in doc_embeddings:
        sim = cosine_sim(query_emb, doc_emb)
        similarities.append(sim)
    
    # Return average similarity
    avg_similarity = np.mean(similarities)
    return float(avg_similarity)

def answer_faithfulness(generated_answer: str, retrieved_docs: List[str]) -> float:
    """
    Measure how well the generated answer is supported by the retrieved context.
    Higher score means the answer is more faithful to the retrieved documents.
    """
    if not generated_answer.strip() or not retrieved_docs:
        return 0.0
    
    answer_emb = embed([generated_answer])[0]
    context_text = " ".join(retrieved_docs)
    context_emb = embed([context_text])[0]
    
    return cosine_sim(answer_emb, context_emb)

def answer_relevance(query: str, generated_answer: str) -> float:
    """
    Measure how relevant the generated answer is to the original query.
    """
    if not query.strip() or not generated_answer.strip():
        return 0.0
    
    query_emb = embed([query])[0]
    answer_emb = embed([generated_answer])[0]
    
    return cosine_sim(query_emb, answer_emb)

def context_diversity(retrieved_docs: List[str]) -> float:
    """
    Measure diversity of retrieved documents (avoid redundancy).
    Higher score means more diverse content.
    """
    if not retrieved_docs or len(retrieved_docs) <= 1:
        return 1.0
    
    doc_embeddings = embed(retrieved_docs)
    if doc_embeddings.size == 0:
        return 0.0
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(doc_embeddings)):
        for j in range(i + 1, len(doc_embeddings)):
            sim = cosine_sim(doc_embeddings[i], doc_embeddings[j])
            similarities.append(sim)
    
    # Diversity is inverse of average similarity
    avg_similarity = np.mean(similarities) if similarities else 0.0
    return 1.0 - avg_similarity

def answer_completeness(generated_answer: str) -> float:
    """
    Simple heuristic to measure answer completeness based on length and structure.
    """
    if not generated_answer.strip():
        return 0.0
    
    # Basic completeness indicators
    word_count = len(generated_answer.split())
    sentence_count = len(re.split(r'[.!?]+', generated_answer))
    
    # Normalize scores
    word_score = min(word_count / 100, 1.0)  # Expect at least 100 words for complete answer
    sentence_score = min(sentence_count / 5, 1.0)  # Expect at least 5 sentences
    
    return (word_score + sentence_score) / 2

def evaluate_rag(
    query: str,
    generated_answer: str,
    retrieved_docs: List[str],
    reference_docs: List[str] = None  # Optional, not used in this version
) -> dict:
    """
    Comprehensive RAG evaluation without requiring reference documents.
    Returns a dictionary of metric scores between 0 and 1.
    """
    if not retrieved_docs:
        return {
            "query_document_relevance": 0.0,
            "answer_faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_diversity": 0.0,
            "answer_completeness": 0.0,
            "overall_score": 0.0
        }
    
    # Calculate individual metrics
    query_doc_relevance = query_document_relevance(query, retrieved_docs)
    faithfulness = answer_faithfulness(generated_answer, retrieved_docs)
    relevance = answer_relevance(query, generated_answer)
    diversity = context_diversity(retrieved_docs)
    completeness = answer_completeness(generated_answer)
    
    # Calculate overall score (weighted average)
    overall_score = (
        query_doc_relevance * 0.25 +  # How relevant are retrieved docs to query
        faithfulness * 0.25 +          # How faithful is answer to context
        relevance * 0.25 +             # How relevant is answer to query
        diversity * 0.125 +            # How diverse is the retrieved context
        completeness * 0.125           # How complete is the answer
    )
    
    return {
        "query_document_relevance": round(query_doc_relevance, 3),
        "answer_faithfulness": round(faithfulness, 3),
        "answer_relevance": round(relevance, 3),
        "context_diversity": round(diversity, 3),
        "answer_completeness": round(completeness, 3),
        "overall_score": round(overall_score, 3)
    }