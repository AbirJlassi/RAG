# ui/enhanced_app.py
import streamlit as st
from modules.enhanced_rag_chain import generate_enhanced_response
from modules.storage import store_generation
from utils.taxonomy_loader import load_taxonomy
import json

# ---------- 🎨 Configuration de la page ----------
st.set_page_config(
    page_title="SKILLIA - Plateforme RAG Intelligente",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 🎨 Style CSS enrichi ----------
def inject_enhanced_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            :root {
                --skillia-primary: #2f2f9b;
                --skillia-secondary: #f92a88;
                --skillia-accent: #00d4ff;
                --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --card-bg: rgba(255, 255, 255, 0.95);
                --text-primary: #1a1a1a;
                --text-secondary: #666;
                --success: #10b981;
                --warning: #f59e0b;
                --error: #ef4444;
            }
            
            .main {
                background: var(--bg-gradient);
                min-height: 100vh;
                font-family: 'Inter', sans-serif;
            }
            
            .skillia-header {
                background: var(--card-bg);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
                text-align: center;
                backdrop-filter: blur(10px);
            }
            
            .skillia-header h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: linear-gradient(45deg, var(--skillia-primary), var(--skillia-secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .skillia-header .subtitle {
                color: var(--text-secondary);
                font-size: 1.2rem;
                margin-bottom: 1rem;
            }
            
            .platform-description {
                background: linear-gradient(45deg, var(--skillia-accent), var(--skillia-secondary));
                color: white;
                padding: 1rem 2rem;
                border-radius: 15px;
                margin: 1rem 0;
                font-weight: 500;
            }
            
            .feature-card {
                background: var(--card-bg);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border-left: 4px solid var(--skillia-primary);
                transition: all 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.15);
            }
            
            .feature-card h3 {
                color: var(--skillia-primary);
                margin-bottom: 0.5rem;
                font-weight: 600;
            }
            
            .mode-selector {
                background: var(--card-bg);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .debug-info {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                border-left: 4px solid var(--skillia-accent);
                font-family: 'Monaco', monospace;
                font-size: 0.85rem;
            }
            
            .classification-badge {
                display: inline-block;
                background: var(--skillia-primary);
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                margin: 0.2rem;
            }
            
            .rerank-info {
                background: linear-gradient(45deg, var(--skillia-accent), var(--skillia-primary));
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            
            .stButton > button {
                background: linear-gradient(45deg, var(--skillia-primary), var(--skillia-secondary));
                color: white;
                border: none;
                padding: 0.8rem 2rem;
                border-radius: 25px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }
            
            .metric-card {
                background: var(--card-bg);
                border-radius: 10px;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--skillia-primary);
            }
            
            .metric-label {
                color: var(--text-secondary);
                font-size: 0.9rem;
            }
        </style>
    """, unsafe_allow_html=True)

# ---------- 🚀 Interface principale ----------
def run_enhanced_app():
    inject_enhanced_css()
    
    # Header principal SKILLIA
    st.markdown("""
        <div class="skillia-header">
            <h1>SKILLIA</h1>
            <p class="subtitle">Plateforme RAG Intelligente</p>
            <div class="platform-description">
                 Valorisation du capital intellectuel • Génération automatique de propales • Réutilisation des livrables
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations sur les fonctionnalités
    with st.sidebar:
        st.markdown("## 🎯 Fonctionnalités ")
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Classification Intelligente</h3>
            <p>Analyse automatique des requêtes pour optimiser la recherche</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Reranking Avancé</h3>
            <p>Amélioration de la pertinence des résultats par post-traitement</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📚 Réutilisation Intelligente</h3>
            <p>Capitalisation sur les propales et livrables existants</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mode debug
        st.markdown("## 🔧 Options")
        debug_mode = st.checkbox("Mode Debug", help="Affiche les informations de traitement détaillées")
        
        # Sélection du type de génération
        st.markdown("## 📝 Type de génération")
        generation_type = st.selectbox(
            "Que souhaitez-vous générer ?",
            ["Proposition commerciale", "Ébauche de livrable", "Recherche de contenu existant"],
            help="Choisissez le type de document à générer"
        )
    
    # Chargement de la taxonomie
    try:
        taxonomy = load_taxonomy()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la taxonomie : {e}")
        taxonomy = {}
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Votre demande")
        
        # Adaptation du placeholder selon le type
        placeholders = {
            "Proposition commerciale": "Décrivez le besoin client et le contexte de la mission...",
            "Ébauche de livrable": "Précisez le type de livrable et les objectifs...",
            "Recherche de contenu existant": "Décrivez ce que vous recherchez dans les documents existants..."
        }
        
        query = st.text_area(
            "Votre requête",
            placeholder=placeholders[generation_type],
            height=150,
            help="Soyez précis pour obtenir des résultats optimaux"
        )
    
    with col2:
        st.markdown("### 🎯 Filtres")
        
        # Filtres selon les domaines SKILLIA
        skillia_domain = st.selectbox(
            "Domaine d'expertise",
            ["", "IA", "Data", "Cybersécurité", "Automatisation"],
            help="Domaine d'expertise SKILLIA"
        )
        
        secteur = st.selectbox(
            "Secteur client",
            [""] + taxonomy.get("secteurs", []),
            help="Secteur d'activité du client"
        )
        
        # Filtres avancés
        with st.expander("🔧 Filtres avancés"):
            domaines = [d["nom"] for d in taxonomy.get("domaines", [])] if taxonomy.get("domaines") else []
            domaine = st.selectbox("Domaine technique", [""] + domaines)
            
            document_type = st.selectbox(
                "Type de document source",
                ["", "Propale", "Diagnostic", "Feuille de route", "Rapport technique", "Formation"],
                help="Type de document à privilégier comme source"
            )
    
    # Construction des filtres
    filters = {}
    if secteur:
        filters["secteur"] = secteur
    if domaine:
        filters["domaine"] = domaine
    if skillia_domain:
        filters["skillia_domain"] = skillia_domain.lower()
    if document_type:
        filters["document_type"] = document_type.lower().replace(" ", "_")
    
    # Bouton de génération
    if st.button(f"🚀 Générer {generation_type.lower()}", key="generate_enhanced"):
        if query:
            with st.spinner("🔄 Traitement intelligent en cours..."):
                try:
                    # Génération avec le système amélioré
                    result = generate_enhanced_response(
                        query=query,
                        filters=filters if filters else None,
                        debug_mode=debug_mode
                    )
                    
                    # Affichage des résultats
                    st.markdown("---")
                    st.markdown(f"## 📄 {generation_type} générée")
                    
                    # Métriques en haut
                    if result.get('debug_info'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result['documents_used']}</div>
                                    <div class="metric-label">Documents utilisés</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            classification_info = result['classification']
                            complexity = classification_info['complexity']['level']
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{complexity.upper()}</div>
                                    <div class="metric-label">Complexité</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            query_types = classification_info['query_type']
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{len(query_types)}</div>
                                    <div class="metric-label">Types détectés</div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Résultat principal
                    st.markdown("### 📋 Résultat")
                    st.write(result['response'])
                    
                    # Informations de debug si activées
                    if debug_mode and result.get('debug_info'):
                        st.markdown("---")
                        st.markdown("## 🔍 Informations de Debug")
                        
                        # Classification de la requête
                        st.markdown("### 🎯 Classification de la requête")
                        classification = result['classification']
                        
                        # Badges pour les types détectés
                        st.markdown("**Types détectés :**")
                        for qtype in classification['query_type']:
                            st.markdown(f'<span class="classification-badge">{qtype}</span>', unsafe_allow_html=True)
                        
                        # Entités extraites
                        entities = classification['entities']
                        if any(entities.values()):
                            st.markdown("**Entités détectées :**")
                            for entity_type, entity_list in entities.items():
                                if entity_list:
                                    st.write(f"- {entity_type}: {', '.join(entity_list)}")
                        
                        # Stratégie de recherche
                        st.markdown("### 🔍 Stratégie de recherche")
                        strategy = classification['search_strategy']
                        st.json(strategy)
                        
                        # Informations de reranking
                        debug_info = result['debug_info']
                        if debug_info.get('reranking_explanation'):
                            st.markdown("### 🎯 Explication du Reranking")
                            st.markdown(f"""
                                <div class="rerank-info">
                                    <strong>Documents analysés :</strong> {debug_info['initial_documents']}<br>
                                    <strong>Documents sélectionnés :</strong> {debug_info['reranked_documents']}<br>
                                    <strong>Amélioration :</strong> {debug_info['reranked_documents']}/{debug_info['initial_documents']} documents les plus pertinents
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Détails complets en JSON
                        with st.expander("📊 Détails complets"):
                            st.json(result['debug_info'])
                    
                    # Feedback utilisateur
                    st.markdown("---")
                    st.markdown("### 💭 Feedback")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("👍 Résultat satisfaisant"):
                            st.success("Merci pour votre retour ! Cela aide à améliorer le système.")
                    
                    with col2:
                        if st.button("👎 À améliorer"):
                            feedback = st.text_area("Commentaires (optionnel)", key="feedback")
                            if st.button("Envoyer le feedback"):
                                # Ici, vous pouvez stocker le feedback pour améliorer le système
                                st.info("Feedback enregistré. Merci !")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération : {e}")
                    if debug_mode:
                        st.exception(e)
        else:
            st.warning("⚠️ Veuillez saisir une requête avant de générer.")
    
    # Footer informatif
    st.markdown("---")
    st.markdown("""
        <footer style="text-align: center; padding: 1rem; color: var(--text-secondary);">
            <p>© 2025 SKILLIA - Tous droits réservés</p>
            <p>Développé par l'équipe SKILLIA </p>
        </footer>
    """, unsafe_allow_html=True)