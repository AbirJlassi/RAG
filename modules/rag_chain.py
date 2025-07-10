import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.vectorstore import create_vectorstore, create_bm25_vectorstore
from modules.loader import load_and_tag_documents
from modules.llm import get_llm
from modules.storage import store_generation
from modules.metrics import evaluate_rag

import traceback  
# 🔐 Charger les variables d'environnement (.env)
load_dotenv()

# 🎯 Initialiser le reranker CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 📥 Charger les documents et les enrichir
docs = load_and_tag_documents("data/")

# ✂️ Chunking (chunks courts pour + de pertinence)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 🧠 Création des vecteurs
faiss_vectorstore = create_vectorstore(chunks)
bm25_vectorstore = create_bm25_vectorstore(chunks)  # BM25 n'a pas .as_retriever()

# 🔁 Fusion manuelle des résultats FAISS + BM25
def hybrid_retrieve(query, retriever1, retriever2, top_k=12):
    """
    Combine les résultats de FAISS (dense) et BM25 (sparse) de façon manuelle
    """
    try:
        results1 = retriever1.get_relevant_documents(query)
    except Exception as e:
        print(f"Erreur retriever1: {e}")
        results1 = []
    
    try:
        results2 = retriever2.get_relevant_documents(query)
    except Exception as e:
        print(f"Erreur retriever2: {e}")
        results2 = []

    # Suppression des doublons
    seen = set()
    combined = []
    for doc in results1 + results2:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)
        if len(combined) >= top_k:
            break
    return combined

# 🤖 Chargement du modèle LLM
llm = get_llm()

# 🚀 Fonction principale : RAG + évaluation



def generate_response(query, filters=None):
    try:
        # === 1. Récupération hybride ===
        context_docs = hybrid_retrieve(query, 
                               faiss_vectorstore.as_retriever(search_kwargs={"k": 12}), 
                               bm25_vectorstore, 
                               top_k=8)

        valid_context_docs = [doc for doc in context_docs if doc.page_content and doc.page_content.strip()]
        if not valid_context_docs:
            print("No valid context documents found.")
            return "Aucun contenu pertinent trouvé dans la base de connaissance."

        # === 2. Reranking CrossEncoder (si documents présents) ===
        if context_docs:
            valid_docs = [doc for doc in context_docs if doc.page_content and doc.page_content.strip()]
            if valid_docs:
                pairs = [[query, doc.page_content] for doc in valid_docs]
                if pairs:
                    try:
                        scores = reranker.predict(pairs)
                        scored_docs = list(zip(scores, valid_docs))
                        context_docs = [doc for _, doc in sorted(scored_docs, key=lambda x: -x[0])]
                    except Exception as e:
                        print(f"Erreur lors du reranking: {e}")
                        context_docs = valid_docs

        # === 3. Fallback de filtre manuel ===
        if filters:
            filtered_docs = [
                doc for doc in context_docs
                if all(doc.metadata.get(k) == v for k, v in filters.items() if v)
            ]
            if filtered_docs:
                context_docs = filtered_docs

        if not context_docs:
            return "Aucun contenu pertinent trouvé dans la base de connaissance."

        # === 4. Construction du contexte pour le prompt ===
        valid_context_docs = [doc for doc in context_docs if doc.page_content and doc.page_content.strip()]
        if not valid_context_docs:
            return "Aucun contenu pertinent trouvé dans la base de connaissance."

        context = "\n\n".join([doc.page_content for doc in valid_context_docs])
        tags = valid_context_docs[0].metadata if valid_context_docs else {}
        if filters:
            tags.update({"filtres_appliques": filters})

        # === 5. Prompt structuré ===
        filter_info = f"\n🎯 Filtres appliqués : {filters}" if filters else ""
        prompt = f"""
        Tu es le consultant expert virtuel de l'entreprise SKILLIA, chargé de générer une proposition commerciale structurée, écrite en FRANCAIS à partir des documents internes de l'entreprise.

        📌 Demande utilisateur :
        {query}{filter_info}

        📚 Contexte extrait :
        {context}

        🔖 Métadonnées associées :
        {tags}

        ✍️ La propale doit inclure :
        1. Contexte client
        2. Objectifs et enjeux identifiés
        3. Démarche ou méthodologie recommandée
        4. Livrables attendus
        5. Planning estimé
        6. Budget indicatif ou TJM
        7. Valeur ajoutée de l'approche proposée

        🧩 Utilise la taxonomie interne pour structurer au mieux ta réponse.
        """

        # === 6. Génération via LLM ===
        response = llm.invoke(prompt).content

        # === 7. Sauvegarde pour apprentissage continu ===
        store_generation(
            query=query,
            context=context,
            metadata={**tags, **(filters or {})},
            response=response
        )

        # === 8. Évaluation (RAG metrics) - SIMPLIFIED ===
        try:
            retrieved_texts = [doc.page_content for doc in valid_context_docs]
            print(f"Evaluating with {len(retrieved_texts)} retrieved documents")
            
            # Use the new metrics that don't require reference documents
            metrics_report = evaluate_rag(
                query=query,
                generated_answer=response,
                retrieved_docs=retrieved_texts
            )
            
            print("=== RAG EVALUATION RESULTS ===")
            for metric, score in metrics_report.items():
                print(f"{metric}: {score}")
            print("===============================")
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation des métriques: {traceback.format_exc()}")
            metrics_report = {
                "query_document_relevance": 0.0,
                "answer_faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_diversity": 0.0,
                "answer_completeness": 0.0,
                "overall_score": 0.0
            }

        return {
            "response": response,
            "metrics": metrics_report
        }

    except Exception as e:
        return f"❌ Erreur dans generate_response: {str(e)}\nTraceback: {traceback.format_exc()}"