

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Import corrigé

# Utilisation du nouveau import pour éviter le warning
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(docs):
    """
    Crée un vectorstore FAISS avec les documents fournis
    """
    if not docs:
        raise ValueError("Aucun document fourni pour créer le vectorstore")
    
    return FAISS.from_documents(docs, embedding)
