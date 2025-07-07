# ui/app.py
import streamlit as st
from modules.rag_chain import generate_response
from modules.storage import store_generation
from utils.taxonomy_loader import load_taxonomy

# ---------- 🎨 Configuration de la page ----------
st.set_page_config(
    page_title="Skillia - Générateur RAG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- 🎨 Style CSS avancé ----------
def inject_custom_css():
    st.markdown("""
        <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Variables CSS */
            :root {
                --primary-color: #6366f1;
                --secondary-color: #8b5cf6;
                --accent-color: #f59e0b;
                --success-color: #10b981;
                --error-color: #ef4444;
                --warning-color: #f59e0b;
                --text-primary: #1f2937;
                --text-secondary: #6b7280;
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --border-color: #e5e7eb;
                --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
                --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            }
            
            /* Corps principal */
            .main {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Inter', sans-serif;
            }
            
            .block-container {
                padding: 2rem 1rem;
                max-width: 1000px;
                margin: 0 auto;
            }
            
            /* Header principal */
            .main-header {
                background: var(--bg-primary);
                padding: 2rem;
                border-radius: 16px;
                box-shadow: var(--shadow-lg);
                margin-bottom: 2rem;
                text-align: center;
                border: 1px solid var(--border-color);
            }
            
            .main-header h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .main-header .skill-text {
                color: #2f2f9b;
            }
            
            .main-header .ia-text {
                color: #f92a88;
            }
            
            .main-header p {
                color: var(--text-secondary);
                font-size: 1.1rem;
                margin: 0;
            }
            
            /* Cards */
            .card {
                background: var(--bg-primary);
                border-radius: 12px;
                padding: 2rem;
                box-shadow: var(--shadow-md);
                border: 1px solid var(--border-color);
                margin-bottom: 2rem;
            }
            
            .card h3 {
                color: var(--text-primary);
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            /* Inputs personnalisés */
            .stTextInput > div > div > input {
                border: 2px solid var(--border-color);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                font-size: 1rem;
                transition: all 0.3s ease;
                background: var(--bg-primary);
            }
            
            .stTextInput > div > div > input:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
                outline: none;
            }
            
            .stSelectbox > div > div > div {
                border: 2px solid var(--border-color);
                border-radius: 8px;
                background: var(--bg-primary);
                transition: all 0.3s ease;
            }
            
            .stSelectbox > div > div > div:focus-within {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
            }
            
            /* Boutons */
            .stButton > button {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                font-weight: 600;
                border: none;
                padding: 0.75rem 2rem;
                border-radius: 8px;
                font-size: 1rem;
                transition: all 0.3s ease;
                box-shadow: var(--shadow-sm);
                width: 100%;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-lg);
                background: linear-gradient(135deg, #5b5fd8, #7c3aed);
            }
            
            .stButton > button:active {
                transform: translateY(0);
            }
            
            /* Expander */
            .st-expander {
                background: var(--bg-primary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                box-shadow: var(--shadow-sm);
                margin-bottom: 2rem;
            }
            
            .st-expander > summary {
                color: var(--text-primary);
                font-weight: 600;
                padding: 1rem;
                background: var(--bg-secondary);
                border-radius: 8px 8px 0 0;
            }
            
            /* Messages d'état */
            .stSuccess {
                background: linear-gradient(135deg, var(--success-color), #059669);
                color: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            
            .stError {
                background: linear-gradient(135deg, var(--error-color), #dc2626);
                color: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            
            .stWarning {
                background: linear-gradient(135deg, var(--warning-color), #d97706);
                color: white;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            
            /* Résultat */
            .result-container {
                background: var(--bg-primary);
                border-radius: 12px;
                padding: 2rem;
                box-shadow: var(--shadow-lg);
                border: 1px solid var(--border-color);
                margin-top: 2rem;
            }
            
            .result-header {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 1rem;
                color: var(--text-primary);
                font-weight: 600;
                font-size: 1.25rem;
            }
            
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .fade-in {
                animation: fadeIn 0.5s ease-out;
            }
            
            /* Responsive */
            @media (max-width: 768px) {
                .block-container {
                    padding: 1rem 0.5rem;
                }
                
                .main-header h1 {
                    font-size: 2rem;
                }
                
                .card {
                    padding: 1.5rem;
                }
            }
        </style>
    """, unsafe_allow_html=True)

# ---------- 🚀 App principale ----------
def run_app():
    inject_custom_css()
    
    # Header principal
    st.markdown("""
        <div class="main-header fade-in">
            <h1><span class="skill-text">SKILL</span><span class="ia-text">IA</span></h1>
            <h2>Générateur RAG de propositions commerciales</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Chargement de la taxonomie
    try:
        taxonomy = load_taxonomy()
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la taxonomie : {e}")
        taxonomy = {}

    # Section principale - Votre demande
    st.markdown("### 💬 Votre demande")
    query = st.text_input(
        "Posez votre question",
        placeholder="Décrivez votre besoin commercial...",
        help="Soyez précis dans votre demande pour obtenir une réponse adaptée"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Filtres facultatifs
    with st.expander("🔎 Filtres facultatifs"):
        col1, col2 = st.columns(2)
        
        with col1:
            secteur = st.selectbox(
                "🏢 Secteur d'activité", 
                [""] + taxonomy.get("secteurs", []),
                help="Sélectionnez le secteur de votre client"
            )
        
        with col2:
            domaines = [d["nom"] for d in taxonomy.get("domaines", [])] if taxonomy.get("domaines") else []
            domaine = st.selectbox(
                "📚 Domaine", 
                [""] + domaines,
                help="Choisissez le domaine technique concerné"
            )

    # Construction des filtres
    filters = {}
    if secteur:
        filters["secteur"] = secteur
    if domaine:
        filters["domaine"] = domaine

    # Bouton de génération
    if st.button(" Générer la proposition", key="generate_btn"):
        if query:
            with st.spinner("✏️ Génération en cours..."):
                try:
                    response = generate_response(query, filters=filters if filters else None)

                    # Affichage du résultat
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("""
                        <div class="result-container fade-in">
                            <div class="result-header">
                                📄 Proposition générée
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(response)

                    # Stockage optionnel (commenté comme dans l'original)
                    # store_generation(query, response, metadata=filters, context="(filtré)" if filters else "")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération : {e}")
        else:
            st.warning("⚠️ Veuillez saisir une question avant de générer.")
    
    if not st.session_state.get("generate_btn", False):
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run_app()