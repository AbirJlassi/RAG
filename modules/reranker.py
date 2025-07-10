# modules/reranker.py
"""
Module de reranking des documents (Post-Retrieval)
Améliore la pertinence des documents récupérés
"""

import numpy as np
from typing import List, Dict, Tuple
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from modules.llm import get_llm
import re
from collections import Counter

class DocumentReranker:
    def __init__(self):
        self.llm = get_llm()
        # Modèle de sentence embeddings pour le reranking sémantique
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Poids pour les différents critères de scoring
        self.weights = {
            'semantic_similarity': 0.4,
            'keyword_overlap': 0.2,
            'metadata_relevance': 0.2,
            'document_quality': 0.2
        }
    
    def rerank_documents(self, query: str, documents: List[Document], 
                        query_classification: Dict, top_k: int = 5) -> List[Document]:
        """
        Reranke les documents selon plusieurs critères de pertinence
        
        Args:
            query (str): Requête originale
            documents (List[Document]): Documents à reranker
            query_classification (Dict): Classification de la requête
            top_k (int): Nombre de documents à retourner
            
        Returns:
            List[Document]: Documents reranked et triés
        """
        if not documents:
            return []
        
        # Calcul des scores pour chaque document
        document_scores = []
        
        for doc in documents:
            score = self._calculate_document_score(query, doc, query_classification)
            document_scores.append((doc, score))
        
        # Tri par score décroissant
        document_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retour des top_k documents
        reranked_docs = [doc for doc, score in document_scores[:top_k]]
        
        # Ajout des scores dans les métadonnées pour debugging
        for i, (doc, score) in enumerate(document_scores[:top_k]):
            doc.metadata['rerank_score'] = score
            doc.metadata['rerank_position'] = i + 1
        
        return reranked_docs
    
    def _calculate_document_score(self, query: str, document: Document, 
                                 query_classification: Dict) -> float:
        """Calcule le score de pertinence d'un document"""
        
        # 1. Similarité sémantique
        semantic_score = self._calculate_semantic_similarity(query, document)
        
        # 2. Correspondance de mots-clés
        keyword_score = self._calculate_keyword_overlap(query, document)
        
        # 3. Pertinence des métadonnées
        metadata_score = self._calculate_metadata_relevance(document, query_classification)
        
        # 4. Qualité du document
        quality_score = self._calculate_document_quality(document)
        
#         
        # Score final pondéré
        final_score = (
            semantic_score * self.weights['semantic_similarity'] +
            keyword_score * self.weights['keyword_overlap'] +
            metadata_score * self.weights['metadata_relevance'] +
            quality_score * self.weights['document_quality'] 
        )
        
        return final_score
    
    def _calculate_semantic_similarity(self, query: str, document: Document) -> float:
        """Calcule la similarité sémantique entre requête et document"""
        try:
            # Embeddings
            query_embedding = self.sentence_model.encode([query])
            doc_embedding = self.sentence_model.encode([document.page_content])
            
            # Similarité cosinus
            similarity = np.dot(query_embedding[0], doc_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding[0])
            )
            
            return float(similarity)
        except Exception:
            return 0.0
    
    def _calculate_keyword_overlap(self, query: str, document: Document) -> float:
        """Calcule la correspondance de mots-clés"""
        # Nettoyage et tokenisation
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        doc_words = set(re.findall(r'\b\w+\b', document.page_content.lower()))
        
        # Mots vides français
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'à', 'dans', 'pour', 'sur', 'avec', 'par', 'ce', 'qui', 'que', 'dont', 'où'}
        
        # Filtrage des mots vides
        query_words = query_words - stop_words
        doc_words = doc_words - stop_words
        
        if not query_words:
            return 0.0
        
        # Calcul de l'intersection
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_metadata_relevance(self, document: Document, 
                                    query_classification: Dict) -> float:
        """Calcule la pertinence basée sur les métadonnées"""
        score = 0.0
        metadata = document.metadata
        
        # Correspondance de secteur
        if query_classification.get('ai_classification', {}).get('secteur_probable'):
            expected_secteur = query_classification['ai_classification']['secteur_probable']
            if metadata.get('secteur') == expected_secteur:
                score += 0.3
        
        # Correspondance de domaine
        if query_classification.get('entities', {}).get('domaines'):
            expected_domaines = query_classification['entities']['domaines']
            if metadata.get('domaine') in expected_domaines:
                score += 0.3
        
        # Correspondance de type de projet
        if query_classification.get('ai_classification', {}).get('type_projet'):
            expected_type = query_classification['ai_classification']['type_projet']
            if metadata.get('type_projet') == expected_type:
                score += 0.2
        
        # Bonus pour les documents avec TJM si budget mentionné
        if query_classification.get('ai_classification', {}).get('budget_mentionne'):
            if metadata.get('tjm') or metadata.get('budget'):
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_document_quality(self, document: Document) -> float:
        """Évalue la qualité du document"""
        content = document.page_content
        
        # Longueur du contenu (ni trop court ni trop long)
        length = len(content)
        if 100 <= length <= 2000:
            length_score = 1.0
        elif length < 100:
            length_score = length / 100
        else:
            length_score = max(0.5, 2000 / length)
        
        # Richesse du contenu (présence de mots-clés métier)
        business_keywords = ['client', 'projet', 'livrable', 'méthodologie', 'équipe', 'expertise', 'solution']
        keyword_count = sum(1 for keyword in business_keywords if keyword in content.lower())
        richness_score = min(keyword_count / len(business_keywords), 1.0)
        
        # Structure (présence de sections, listes, etc.)
        structure_indicators = ['\n-', '\n•', '\n1.', '\n2.', '##', '###']
        structure_score = min(sum(1 for indicator in structure_indicators if indicator in content) / 3, 1.0)
        
        return (length_score + richness_score + structure_score) / 3
    

    
    def explain_reranking(self, query: str, documents: List[Document], 
                         query_classification: Dict) -> Dict:
        """
        Fournit une explication détaillée du reranking
        
        Returns:
            Dict: Explication avec scores détaillés
        """
        explanations = []
        
        for i, doc in enumerate(documents):
            semantic_score = self._calculate_semantic_similarity(query, doc)
            keyword_score = self._calculate_keyword_overlap(query, doc)
            metadata_score = self._calculate_metadata_relevance(doc, query_classification)
            quality_score = self._calculate_document_quality(doc)
            
            final_score = (
                semantic_score * self.weights['semantic_similarity'] +
                keyword_score * self.weights['keyword_overlap'] +
                metadata_score * self.weights['metadata_relevance'] +
                quality_score * self.weights['document_quality'] 
            )
            
            explanations.append({
                'document_index': i,
                'final_score': final_score,
                'detail_scores': {
                    'semantic_similarity': semantic_score,
                    'keyword_overlap': keyword_score,
                    'metadata_relevance': metadata_score,
                    'document_quality': quality_score,
                },
                'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata
            })
        
        return {
            'query': query,
            'query_classification': query_classification,
            'reranking_weights': self.weights,
            'document_explanations': explanations
        }

# Instance globale
document_reranker = DocumentReranker()