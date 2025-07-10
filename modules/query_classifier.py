# modules/query_classifier.py
"""
Module de classification des requêtes (Pre-Retrieval)
Analyse la requête utilisateur pour optimiser la recherche
"""

import re
from typing import Dict, List, Tuple
from modules.llm import get_llm
from utils.taxonomy_loader import load_taxonomy

class QueryClassifier:
    def __init__(self):
        self.llm = get_llm()
        self.taxonomy = load_taxonomy()
        
        # Patterns pour détecter les types de requêtes
        self.patterns = {
            'devis': r'(devis|budget|coût|prix|tarif|estimation|combien)',
            'methodologie': r'(méthode|approche|démarche|process|étapes|comment)',
            'delivrables': r'(livrable|résultat|output|documentation|rapport)',
            'planning': r'(planning|délai|durée|temps|quand|calendrier)',
            'expertise': r'(expert|compétence|skill|profil|qui|équipe)',
            'secteur': r'(secteur|industrie|domaine|vertical|marché)',
            'comparative': r'(compare|différence|versus|vs|alternative)',
            'exemple': r'(exemple|cas|illustration|référence|similaire)',
            'propale': r'(propale|proposition|propose-moi|offre|réponse à appel d\'offres|réponds.*appel|répond.*offre|proposition commerciale)'

        }
        
    def classify_query(self, query: str) -> Dict:
        """
        Classifie une requête selon plusieurs dimensions
        
        Args:
            query (str): Requête utilisateur
            
        Returns:
            Dict: Classification complète avec recommandations
        """
        query_lower = query.lower()
        
        # 1. Classification par type de demande
        query_type = self._classify_query_type(query_lower)
        
        # 2. Détection d'entités métier
        entities = self._extract_entities(query_lower)
        
        # 3. Analyse de complexité
        complexity = self._analyze_complexity(query)
        
        # 4. Recommandations de recherche
        search_strategy = self._recommend_search_strategy(query_type, entities, complexity)
        
        # 5. Classification intelligente via LLM
        ai_classification = self._ai_classify_query(query)
        
        return {
            'query_type': query_type,
            'entities': entities,
            'complexity': complexity,
            'search_strategy': search_strategy,
            'ai_classification': ai_classification,
            'enhanced_query': self._enhance_query(query, entities)
        }
    
    def _classify_query_type(self, query: str) -> List[str]:
        """Classifie le type de requête selon les patterns"""
        detected_types = []
        
        for type_name, pattern in self.patterns.items():
            if re.search(pattern, query):
                detected_types.append(type_name)
        
        return detected_types if detected_types else ['general']
    
    def _extract_entities(self, query: str) -> Dict:
        """Extrait les entités métier de la requête"""
        entities = {
            'secteurs': [],
            'domaines': [],
            'technologies': [],
            'livrables': []
        }
        
        # Extraction des secteurs
        for secteur in self.taxonomy.get('secteurs', []):
            if secteur.lower() in query:
                entities['secteurs'].append(secteur)
        
        # Extraction des domaines
        for domaine in self.taxonomy.get('domaines', []):
            if domaine.get('nom', '').lower() in query:
                entities['domaines'].append(domaine['nom'])
        
        # Extraction des technologies courantes
        tech_keywords = ['python', 'java', 'react', 'angular', 'cloud', 'aws', 'azure', 'gcp']
        for tech in tech_keywords:
            if tech in query:
                entities['technologies'].append(tech)
        
        return entities
    
    def _analyze_complexity(self, query: str) -> Dict:
        """Analyse la complexité de la requête"""
        word_count = len(query.split())
        question_marks = query.count('?')
        specific_terms = len(re.findall(r'\b[A-Z][a-z]+\b', query))
        
        complexity_score = (
            (word_count * 0.1) +
            (question_marks * 2) +
            (specific_terms * 0.5)
        )
        
        if complexity_score > 10:
            level = 'high'
        elif complexity_score > 5:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': complexity_score,
            'word_count': word_count,
            'specificity': specific_terms
        }
    
    def _recommend_search_strategy(self, query_type: List[str], entities: Dict, complexity: Dict) -> Dict:
        """Recommande une stratégie de recherche optimisée"""
        strategy = {
            'search_k': 4,  # Nombre par défaut
            'use_filters': False,
            'expand_query': False,
            'prioritize_recent': False,
            'focus_areas': []
        }
        
        # Ajustement selon le type de requête
        if 'devis' in query_type:
            strategy['search_k'] = 6
            strategy['focus_areas'] = ['budget', 'tjm', 'estimation']
        
        if 'methodologie' in query_type:
            strategy['search_k'] = 5
            strategy['focus_areas'] = ['methodologie', 'process', 'etapes']
        
        if 'exemple' in query_type:
            strategy['search_k'] = 8
            strategy['focus_areas'] = ['cas_client', 'reference', 'exemple']
        
        # Ajustement selon les entités détectées
        if entities['secteurs'] or entities['domaines']:
            strategy['use_filters'] = True
        
        # Ajustement selon la complexité
        if complexity['level'] == 'high':
            strategy['search_k'] = max(strategy['search_k'], 7)
            strategy['expand_query'] = True
        
        return strategy
    
    def _ai_classify_query(self, query: str) -> Dict:
        """Classification intelligente via LLM"""
        prompt = f"""
        Analyse cette requête commerciale et classe-la selon ces critères :
        
        Requête: "{query}"
        
        Réponds au format JSON avec :
        {{
            "intention": "description_intention",
            "urgence": "low|medium|high",
            "secteur_probable": "secteur_detecte_ou_null",
            "type_projet": "conseil|formation|audit|developpement|autre",
            "budget_mentionne": true/false,
            "mots_cles": ["mot1", "mot2", "mot3"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parsing simple (à améliorer avec un parser JSON robuste)
            import json
            return json.loads(response.content.strip())
        except Exception as e:
            return {
                "intention": "non_classifiee",
                "urgence": "medium",
                "secteur_probable": None,
                "type_projet": "autre",
                "budget_mentionne": False,
                "mots_cles": [],
                "error": str(e)
            }
    
    def _enhance_query(self, original_query: str, entities: Dict) -> str:
        """Enrichit la requête avec le contexte détecté"""
        enhanced_parts = [original_query]
        
        # Ajout d'informations contextuelles
        if entities['secteurs']:
            enhanced_parts.append(f"secteur: {', '.join(entities['secteurs'])}")
        
        if entities['domaines']:
            enhanced_parts.append(f"domaine: {', '.join(entities['domaines'])}")
        
        if entities['technologies']:
            enhanced_parts.append(f"technologies: {', '.join(entities['technologies'])}")
        
        return " | ".join(enhanced_parts)

# Instance globale
query_classifier = QueryClassifier()