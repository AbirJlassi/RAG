# modules/skillia_document_classifier.py
"""
Classificateur spécialisé pour les documents SKILLIA
Identifie automatiquement le type de document et extrait les métadonnées métier
"""

import re
from typing import Dict, List
from langchain.schema import Document

class SkilliaDocumentClassifier:
    def __init__(self):
        # Types de documents SKILLIA
        self.document_types = {
            'propale': {
                'keywords': ['proposition', 'devis', 'offre', 'commercial', 'tarif', 'tjm', 'budget'],
                'patterns': [r'proposition\s+commerciale', r'devis\s+détaillé', r'offre\s+de\s+service']
            },
            'diagnostic': {
                'keywords': ['diagnostic', 'audit', 'analyse', 'état des lieux', 'évaluation'],
                'patterns': [r'diagnostic\s+technique', r'audit\s+de\s+sécurité', r'analyse\s+des\s+risques']
            },
            'feuille_route': {
                'keywords': ['feuille de route', 'roadmap', 'plan', 'stratégie', 'recommandations'],
                'patterns': [r'feuille\s+de\s+route', r'plan\s+stratégique', r'roadmap\s+technique']
            },
            'rapport_technique': {
                'keywords': ['rapport', 'étude', 'analyse technique', 'documentation', 'spécifications'],
                'patterns': [r'rapport\s+technique', r'étude\s+de\s+faisabilité', r'documentation\s+technique']
            },
            'livrable_formation': {
                'keywords': ['formation', 'cours', 'module', 'apprentissage', 'compétences'],
                'patterns': [r'support\s+de\s+formation', r'module\s+pédagogique', r'programme\s+de\s+formation']
            }
        }
        
        # Domaines d'expertise SKILLIA
        self.skillia_domains = {
            'ia': {
                'keywords': ['intelligence artificielle', 'machine learning', 'deep learning', 'nlp', 'computer vision', 'ia générative'],
                'technologies': ['tensorflow', 'pytorch', 'sklearn', 'langchain', 'openai', 'huggingface']
            },
            'data': {
                'keywords': ['data science', 'analytics', 'big data', 'datawarehouse', 'bi', 'données'],
                'technologies': ['python', 'sql', 'spark', 'hadoop', 'tableau', 'power bi', 'databricks']
            },
            'cybersecurite': {
                'keywords': ['cybersécurité', 'sécurité', 'pentest', 'vulnerability', 'rgpd', 'iso27001'],
                'technologies': ['nessus', 'metasploit', 'wireshark', 'kali', 'splunk', 'siem']
            },
            'automatisation': {
                'keywords': ['automatisation', 'rpa', 'workflow', 'orchestration', 'devops', 'ci/cd'],
                'technologies': ['ansible', 'terraform', 'docker', 'kubernetes', 'jenkins', 'gitlab']
            }
        }
        
        # Secteurs clients typiques
        self.client_sectors = [
            'banque', 'assurance', 'retail', 'industrie', 'santé', 'télécoms', 
            'énergie', 'transport', 'secteur public', 'startup', 'grand groupe'
        ]
    
    def classify_document(self, document: Document) -> Dict:
        """
        Classifie un document selon les standards SKILLIA
        
        Args:
            document (Document): Document à classifier
            
        Returns:
            Dict: Classification complète avec métadonnées enrichies
        """
        content = document.page_content.lower()
        
        # Classification du type de document
        doc_type = self._classify_document_type(content)
        
        # Identification du domaine d'expertise
        domain = self._identify_domain(content)
        
        # Extraction des métadonnées métier
        business_metadata = self._extract_business_metadata(content)
        
        # Détection du secteur client
        client_sector = self._detect_client_sector(content)
        
        # Analyse de la valeur commerciale
        commercial_value = self._analyze_commercial_value(content, doc_type)
        
        # Enrichissement des métadonnées existantes
        enhanced_metadata = document.metadata.copy()
        enhanced_metadata.update({
            'document_type': doc_type,
            'skillia_domain': domain,
            'client_sector': client_sector,
            'commercial_value': commercial_value,
            **business_metadata
        })
        
        return {
            'document_type': doc_type,
            'skillia_domain': domain,
            'client_sector': client_sector,
            'business_metadata': business_metadata,
            'commercial_value': commercial_value,
            'enhanced_metadata': enhanced_metadata,
            'classification_confidence': self._calculate_confidence(content, doc_type, domain)
        }
    
    def _classify_document_type(self, content: str) -> str:
        """Classifie le type de document"""
        scores = {}
        
        for doc_type, config in self.document_types.items():
            score = 0
            
            # Score basé sur les mots-clés
            for keyword in config['keywords']:
                score += content.count(keyword) * 2
            
            # Score basé sur les patterns
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, content))
                score += matches * 3
            
            scores[doc_type] = score
        
        # Retourne le type avec le score le plus élevé
        if scores:
            return max(scores, key=scores.get)
        return 'document_general'
    
    def _identify_domain(self, content: str) -> str:
        """Identifie le domaine d'expertise SKILLIA"""
        scores = {}
        
        for domain, config in self.skillia_domains.items():
            score = 0
            
            # Score basé sur les mots-clés du domaine
            for keyword in config['keywords']:
                score += content.count(keyword) * 2
            
            # Score basé sur les technologies
            for tech in config['technologies']:
                score += content.count(tech) * 1.5
            
            scores[domain] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'domaine_general'
    
    def _extract_business_metadata(self, content: str) -> Dict:
        """Extrait les métadonnées métier importantes"""
        metadata = {}
        
        # Extraction des montants/TJM
        tjm_pattern = r'(\d+)\s*€?\s*(?:€|euros?)?(?:/|\s+par\s+|\s+)\s*(?:jour|j|day)'
        tjm_matches = re.findall(tjm_pattern, content, re.IGNORECASE)
        if tjm_matches:
            metadata['tjm_detected'] = [int(match) for match in tjm_matches]
        
        # Extraction des durées
        duration_pattern = r'(\d+)\s*(?:jours?|semaines?|mois|j|sem|m)\s*(?:homme|/homme)?'
        duration_matches = re.findall(duration_pattern, content, re.IGNORECASE)
        if duration_matches:
            metadata['duration_detected'] = duration_matches
        
        # Extraction des livrables
        deliverable_keywords = ['livrable', 'rapport', 'présentation', 'documentation', 'formation']
        deliverables = []
        for keyword in deliverable_keywords:
            pattern = f'{keyword}[^.]*'
            matches = re.findall(pattern, content, re.IGNORECASE)
            deliverables.extend(matches[:3])  # Max 3 par type
        
        if deliverables:
            metadata['deliverables_detected'] = deliverables
        
        # Extraction des méthodologies
        methodology_keywords = ['agile', 'scrum', 'waterfall', 'lean', 'devops', 'design thinking']
        methodologies = [method for method in methodology_keywords if method in content]
        if methodologies:
            metadata['methodologies_detected'] = methodologies
        
        return metadata
    
    def _detect_client_sector(self, content: str) -> str:
        """Détecte le secteur du client"""
        for sector in self.client_sectors:
            if sector in content:
                return sector
        return 'secteur_non_identifie'
    
    def _analyze_commercial_value(self, content: str, doc_type: str) -> Dict:
        """Analyse la valeur commerciale du document"""
        value = {
            'reusability_score': 0,
            'commercial_potential': 'low',
            'key_differentiators': []
        }
        
        # Score de réutilisabilité selon le type
        type_scores = {
            'propale': 0.9,
            'diagnostic': 0.7,
            'feuille_route': 0.8,
            'rapport_technique': 0.6,
            'livrable_formation': 0.8
        }
        
        value['reusability_score'] = type_scores.get(doc_type, 0.5)
        
        # Détection des éléments différenciants
        differentiators = ['innovation', 'expertise', 'méthode propriétaire', 'retour sur investissement', 'roi']
        for diff in differentiators:
            if diff in content:
                value['key_differentiators'].append(diff)
        
        # Évaluation du potentiel commercial
        if len(value['key_differentiators']) > 2 and value['reusability_score'] > 0.7:
            value['commercial_potential'] = 'high'
        elif len(value['key_differentiators']) > 1 or value['reusability_score'] > 0.6:
            value['commercial_potential'] = 'medium'
        
        return value
    
    def _calculate_confidence(self, content: str, doc_type: str, domain: str) -> float:
        """Calcule la confiance dans la classification"""
        # Logique simplifiée de calcul de confiance
        type_confidence = 0.8 if doc_type != 'document_general' else 0.4
        domain_confidence = 0.8 if domain != 'domaine_general' else 0.4
        
        return (type_confidence + domain_confidence) / 2
    
    def get_reusable_templates(self, query_type: str, domain: str) -> List[str]:
        """
        Retourne les types de documents les plus pertinents pour une requête
        
        Args:
            query_type (str): Type de requête (devis, methodologie, etc.)
            domain (str): Domaine d'expertise
            
        Returns:
            List[str]: Types de documents recommandés
        """
        recommendations = []
        
        # Mapping requête -> types de documents
        query_to_docs = {
            'devis': ['propale', 'diagnostic'],
            'methodologie': ['feuille_route', 'rapport_technique'],
            'formation': ['livrable_formation'],
            'exemple': ['propale', 'diagnostic', 'feuille_route'],
            'delivrable': ['rapport_technique', 'livrable_formation']
        }
        
        # Recommandations basées sur le type de requête
        if query_type in query_to_docs:
            recommendations.extend(query_to_docs[query_type])
        else:
            # Recommandations par défaut
            recommendations = ['propale', 'diagnostic', 'feuille_route']
        
        return recommendations

# Instance globale
skillia_classifier = SkilliaDocumentClassifier()