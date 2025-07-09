import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever

from modules.vectorstore import create_vectorstore, create_bm25_vectorstore
from modules.loader import load_and_tag_documents
from modules.llm import get_llm
from modules.storage import store_generation
from modules.metrics import evaluate_rag

# 🔐 Charger les variables d'environnement (.env)
load_dotenv()

# 🎯 Reranker CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 📥 Charger et enrichir les documents
docs = load_and_tag_documents("data/")

# ✂️ Chunking des documents (chunks plus courts pour plus de précision)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 🧠 Création des vecteurs FAISS (denses) et BM25 (sparse)
faiss_vectorstore = create_vectorstore(chunks)
bm25_vectorstore = create_bm25_vectorstore(chunks)

# 🔍 Création des retrievers
retriever_faiss = faiss_vectorstore.as_retriever(search_kwargs={"k": 8})
retriever_bm25 = bm25_vectorstore

# 🔁 Fusion manuelle des résultats FAISS + BM25
def hybrid_retrieve(query, retriever1, retriever2, top_k=8):
    results1 = retriever1.get_relevant_documents(query)
    results2 = retriever2.get_relevant_documents(query)

    # Fusionner les documents sans doublons
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

# 🚀 Fonction principale : génère réponse + évalue la qualité
def generate_response(query, filters=None):
    """
    Génère une propale structurée à partir des documents internes,
    puis évalue la qualité de la réponse RAG.
    """
    try:
        # === 1. Application éventuelle des filtres ===
        search_kwargs = {"k": 8}
        if filters:
            filter_conditions = {k: v for k, v in filters.items() if v}
            if filter_conditions:
                search_kwargs["filter"] = filter_conditions

        # === 2. Récupération hybride FAISS + BM25 ===
        context_docs = hybrid_retrieve(query, retriever_faiss, retriever_bm25, top_k=8)

        # === 3. Reranking avec CrossEncoder ===
        if context_docs:
            pairs = [[query, doc.page_content] for doc in context_docs]
            scores = reranker.predict(pairs)
            context_docs = [doc for _, doc in sorted(zip(scores, context_docs), key=lambda x: -x[0])]

        # === 4. Filtrage post-retrieval (fallback) ===
        if filters:
            filtered_docs = [
                doc for doc in context_docs
                if all(doc.metadata.get(k) == v for k, v in filters.items() if v)
            ]
            context_docs = filtered_docs if filtered_docs else context_docs

        if not context_docs:
            return "Aucun contenu pertinent trouvé dans la base de connaissance."

        # === 5. Construction du contexte ===
        context = "\n\n".join([doc.page_content for doc in context_docs])
        tags = context_docs[0].metadata if context_docs else {}
        if filters:
            tags.update({"filtres_appliques": filters})

        # === 6. Prompt pour génération de propale ===
        filter_info = f"\n🎯 Filtres appliqués : {filters}" if filters else ""
        prompt = f"""
Tu es le consultant expert virtuel de l'entreprise SKILLIA, chargé de générer une proposition commerciale structurée, écrite en FRANCAIS à partir des documents internes de l’entreprise.

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

        # === 7. Génération de réponse via LLM ===
        response = llm.invoke(prompt).content

        # === 8. Sauvegarde de la génération ===
        store_generation(
            query=query,
            context=context,
            metadata={**tags, **(filters or {})},
            response=response
        )

        # === 9. Évaluation RAG (qualité du retrieval et de la génération) ===
        reference_docs = []  # Mettre ici tes documents de vérité si disponibles
        retrieved_texts = [doc.page_content for doc in context_docs]

        metrics_report = evaluate_rag(
            query=query,
            generated_answer=response,
            retrieved_docs=retrieved_texts,
            reference_docs=reference_docs
        )

        # ✅ 10. Retourner la réponse + les métriques
        return {
            "response": response,
            "metrics": metrics_report
        }

    except Exception as e:
        return f"❌ Erreur dans generate_response: {str(e)}"
