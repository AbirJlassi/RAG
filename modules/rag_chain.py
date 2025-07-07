# 📁 modules/rag_chain.py

from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.vectorstore import create_vectorstore
from modules.loader import load_and_tag_documents
from modules.llm import get_llm
from modules.storage import store_generation

load_dotenv()

# Chargement + enrichissement des documents
docs = load_and_tag_documents("data/")

# Chunking avec split logique
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# Création du vecteur index
vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = get_llm()

def generate_response(query, filters=None):
    """
    Génère une réponse propale enrichie et structurée, avec tracking pour apprentissage continu
    
    Args:
        query (str): Question/demande de l'utilisateur
        filters (dict, optional): Filtres taxonomiques (secteur, domaine, etc.)
    """
    try:
        # Configuration de la recherche avec filtres
        search_kwargs = {"k": 4}
        
        # Application des filtres si fournis
        if filters:
            # Conversion des filtres en critères de recherche
            filter_conditions = {}
            for key, value in filters.items():
                if value:  # Ignore les valeurs vides
                    filter_conditions[key] = value
            
            if filter_conditions:
                search_kwargs["filter"] = filter_conditions

        # Recherche des chunks pertinents (avec ou sans filtres)
        context_docs = retriever.get_relevant_documents(query)
        
        # Filtrage post-recherche si nécessaire (fallback)
        if filters and context_docs:
            filtered_docs = []
            for doc in context_docs:
                doc_metadata = doc.metadata
                match = True
                for key, value in filters.items():
                    if value and doc_metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            
            # Utiliser les docs filtrés si disponibles, sinon garder tous
            context_docs = filtered_docs if filtered_docs else context_docs
        
        if not context_docs:
            return "Aucun contenu pertinent trouvé dans la base de connaissance."

        # Construction du contexte et métadonnées
        context = "\n\n".join([doc.page_content for doc in context_docs])
        tags = context_docs[0].metadata if context_docs else {}
        
        # Enrichissement avec les filtres appliqués
        if filters:
            tags.update({"filtres_appliques": filters})

        # Construction du prompt professionnel structuré
        filter_info = f"\n🎯 Filtres appliqués : {filters}" if filters else ""
        
        prompt = f"""
Tu es le consultant expert virtuel de l'entreprise SKILLIA, chargé de générer une proposition commerciale structurée, écrite en FRANCAIS et pas en anglais,  à partir des documents internes de l’entreprise.

📌 Demande utilisateur :
{query}{filter_info}

📚 Contexte extrait :
{context}

🔖 Métadonnées associées (secteur, domaine, sous-domaine, livrables, client, durée, TJM) :
{tags}

✍️ La propale doit inclure :
1. Contexte client
2. Objectifs et enjeux identifiés
3. Démarche ou méthodologie recommandée (avec références à la taxonomie si possible)
4. Livrables attendus ou livrables similaires observés
5. Planning estimé (phases, charges)
6. Budget indicatif ou TJM (si détecté)
7. Valeur ajoutée de l'approche proposée

🧩 Utilise la taxonomie interne (domaines, livrables, méthodes) pour structurer au mieux ta réponse.
Rédige en langage clair, professionnel et adapté au secteur d'activité du client.
"""

        response = llm.invoke(prompt)

        # Stockage pour apprentissage continu (feedforward)
        store_generation(
            query=query, 
            context=context, 
            metadata={**tags, **(filters or {})}, 
            response=response
        )

        return response

    except Exception as e:
        return f"❌ Erreur dans generate_response: {str(e)}"