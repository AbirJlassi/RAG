# modules/enhanced_rag_chain.py

from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.vectorstore import create_vectorstore
from modules.loader import load_and_tag_documents
from modules.llm import get_llm
from modules.storage import store_generation
from modules.query_classifier import query_classifier
from modules.reranker import document_reranker
import json
from typing import Dict, List, Optional

load_dotenv()

# Chargement + enrichissement des documents
docs = load_and_tag_documents("data/")

# Chunking avec split logique
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# Création du vecteur index
vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # Récupération plus large pour le reranking
llm = get_llm()

def generate_enhanced_response(query: str, filters: Optional[Dict] = None, 
                              debug_mode: bool = False) -> Dict:
    """
    Génère une réponse avec classification de requête et reranking avancé
    
    Args:
        query (str): Question/demande de l'utilisateur
        filters (dict, optional): Filtres taxonomiques (secteur, domaine, etc.)
        debug_mode (bool): Active le mode debug avec informations détaillées
    
    Returns:
        Dict: Réponse avec informations de debug optionnelles
    """
    try:
        # 📊 ÉTAPE 1: Classification de la requête (Pre-Retrieval)
        print("🔍 Classification de la requête...")
        query_classification = query_classifier.classify_query(query)
        
        # 🎯 ÉTAPE 2: Optimisation de la stratégie de recherche
        search_strategy = query_classification['search_strategy']
        enhanced_query = query_classification['enhanced_query']
        
        # Configuration de recherche adaptée
        search_kwargs = {"k": search_strategy['search_k']}
        
        # Application des filtres (combinés avec ceux de la classification)
        combined_filters = filters.copy() if filters else {}
        
        # Ajout des filtres automatiques si détectés
        if search_strategy['use_filters']:
            entities = query_classification['entities']
            if entities['secteurs'] and not combined_filters.get('secteur'):
                combined_filters['secteur'] = entities['secteurs'][0]
            if entities['domaines'] and not combined_filters.get('domaine'):
                combined_filters['domaine'] = entities['domaines'][0]
        
        if combined_filters:
            search_kwargs["filter"] = combined_filters
        
        # 📚 ÉTAPE 3: Retrieval initial (élargi)
        print("📚 Recherche des documents pertinents...")
        
        # Utilisation de la requête enrichie pour une meilleure recherche
        search_query = enhanced_query if search_strategy['expand_query'] else query
        initial_docs = retriever.get_relevant_documents(search_query)
        
        # Fallback si pas de résultats avec filtres
        if not initial_docs and combined_filters:
            print("🔄 Recherche sans filtres (fallback)...")
            initial_docs = retriever.get_relevant_documents(query)
        
        if not initial_docs:
            return {
                'response': "Aucun contenu pertinent trouvé dans la base de connaissance.",
                'debug_info': {
                    'query_classification': query_classification,
                    'search_strategy': search_strategy,
                    'documents_found': 0
                } if debug_mode else None
            }
        
        # 🎯 ÉTAPE 4: Reranking avancé (Post-Retrieval)
        print("🎯 Reranking des documents...")
        
        # Détermination du nombre final de documents selon le type de requête
        if 'exemple' in query_classification['query_type']:
            final_k = min(6, len(initial_docs))
        elif 'devis' in query_classification['query_type']:
            final_k = min(5, len(initial_docs))
        else:
            final_k = min(4, len(initial_docs))
        
        # Reranking des documents
        
        reranked_docs = document_reranker.rerank_documents(
            query=query,
            documents=initial_docs,
            query_classification=query_classification,
            top_k=final_k
        )
        
        # 📝 ÉTAPE 5: Construction du contexte enrichi
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        # Métadonnées combinées
        all_metadata = {}
        for doc in reranked_docs:
            all_metadata.update(doc.metadata)
        
        # Ajout des informations de classification
        all_metadata.update({
            "query_type": query_classification['query_type'],
            "query_complexity": query_classification['complexity']['level'],
            "ai_classification": query_classification['ai_classification'],
            "reranking_applied": True,
            "documents_analyzed": len(initial_docs),
            "documents_selected": len(reranked_docs)
        })
        
        if combined_filters:
            all_metadata["filtres_appliques"] = combined_filters
        
        # 🤖 ÉTAPE 6: Génération de la réponse avec prompt enrichi
        print("🤖 Génération de la réponse...")
        
        # Adaptation du prompt selon le type de requête
        prompt = _build_adaptive_prompt(
            query, context, all_metadata, query_classification, combined_filters
        )
        
        response = llm.invoke(prompt)
        final_response = response.content
        
        # 💾 ÉTAPE 7: Stockage pour apprentissage
        store_generation(
            query=query,
            context=context,
            metadata=all_metadata,
            response=final_response
        )
        
        # 📊 Préparation des informations de debug
        debug_info = None
        if debug_mode:
            debug_info = {
                'query_classification': query_classification,
                'search_strategy': search_strategy,
                'initial_documents': len(initial_docs),
                'reranked_documents': len(reranked_docs),
                'combined_filters': combined_filters,
                'reranking_explanation': document_reranker.explain_reranking(
                    query, reranked_docs, query_classification
                ),
                'final_metadata': all_metadata
            }
        
        return {
            'response': final_response,
            'debug_info': debug_info,
            'classification': query_classification,
            'documents_used': len(reranked_docs)
        }
        
    except Exception as e:
        error_msg = f"❌ Erreur dans generate_enhanced_response: {str(e)}"
        return {
            'response': error_msg,
            'debug_info': {'error': str(e)} if debug_mode else None
        }

def _build_adaptive_prompt(query: str, context: str, metadata: Dict, 
                          query_classification: Dict, filters: Dict) -> str:
    """
    Construit un prompt adaptatif selon le type de requête
    """
    
    # Informations sur la classification
    query_type = query_classification['query_type']
    ai_class = query_classification['ai_classification']
    complexity = query_classification['complexity']['level']
    
    # Base du prompt
    base_prompt = f"""
Tu es le consultant expert virtuel de l'entreprise SKILLIA, spécialisé dans la génération de propositions commerciales personnalisées.

📌 ANALYSE DE LA REQUÊTE :
- Requête : {query}
- Type détecté : {', '.join(query_type)}
- Complexité : {complexity}
- Intention : {ai_class.get('intention', 'Non définie')}
- Urgence : {ai_class.get('urgence', 'Medium')}
- Type de projet : {ai_class.get('type_projet', 'Non spécifié')}
"""
    
    # Ajout des filtres si présents
    if filters:
        base_prompt += f"\n🎯 Filtres appliqués : {filters}"
    
    # Contexte et métadonnées
    base_prompt += f"""

📚 CONTEXTE EXTRAIT (reranké et optimisé) :
{context}

🔖 MÉTADONNÉES ASSOCIÉES :
{json.dumps(metadata, indent=2, ensure_ascii=False)}
"""
    
    # Instructions adaptées selon le type de requête
    if 'devis' in query_type:
        base_prompt += """
✍️ FOCUS DEVIS - La proposition doit particulièrement inclure :
1. 💰 Estimation budgétaire détaillée (avec TJM si disponible)
2. 📊 Répartition des coûts par phase
3. 🎯 Facteurs impactant le budget
4. 📈 Options d'optimisation des coûts
5. 🔄 Modalités de facturation proposées
"""
    elif 'propale' in query_type:
        base_prompt += """
✍️ FOCUS PROPALE - La proposition doit inclure :
1. 🎯 Contexte et besoins du client
2. 📋 Solution proposée (résumé de l'approche)
3. 🛠️ Méthodologie adaptée
4. 📅 Planning estimé
5. 💰 Estimation du budget
6. 🧠 Valeur ajoutée de Skillia
7. 📎 Exemples ou cas similaires si disponibles
"""
 
    elif 'methodologie' in query_type:
        base_prompt += """
✍️ FOCUS MÉTHODOLOGIE - La proposition doit particulièrement inclure :
1. 🔄 Démarche étape par étape
2. 🛠️ Outils et méthodes utilisés
3. 👥 Rôles et responsabilités
4. 📋 Livrables à chaque phase
5. 🎯 Critères de succès et indicateurs
"""
    
    elif 'exemple' in query_type:
        base_prompt += """
✍️ FOCUS EXEMPLES - La proposition doit particulièrement inclure :
1. 📚 Références similaires détaillées
2. 🎯 Cas d'usage concrets
3. 📊 Résultats obtenus
4. 🔄 Adaptations possibles au contexte
5. 💡 Leçons apprises et bonnes pratiques
"""
    
    else:
        base_prompt += """
✍️ STRUCTURE STANDARD - La proposition doit inclure :
1. 🎯 Contexte et enjeux client
2. 🔍 Objectifs identifiés
3. 🛠️ Approche méthodologique
4. 📋 Livrables attendus
5. 📅 Planning et organisation
6. 💰 Budget indicatif
7. 🚀 Valeur ajoutée proposée
"""
    
    # Instructions finales
    base_prompt += """

🎨 CONSIGNES DE RÉDACTION :
- Rédige en français professionnel et structuré
- Utilise les données métier de SKILLIA (IA, Data, Cybersécurité, Automatisation)
- Exploite le savoir-faire capitalisé dans les documents existants
- Personnalise selon le secteur et le contexte client
- Propose des solutions concrètes et actionnables

🔧 RÉUTILISATION INTELLIGENTE :
- Adapte les méthodes éprouvées aux nouveaux contextes
- Référence les approches similaires déjà menées
- Capitalise sur l'expertise accumulée dans les livrables précédents
- Propose des évolutions/améliorations basées sur les retours d'expérience
"""
    
    return base_prompt