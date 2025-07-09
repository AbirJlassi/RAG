

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Utilisation du nouveau import pour éviter le warning
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vectorstore(docs):
    """
    Crée un vectorstore FAISS avec les documents fournis
    """
    if not docs:
        raise ValueError("Aucun document fourni pour créer le vectorstore")
    
    return FAISS.from_documents(docs, embedding)
def create_bm25_vectorstore(docs):
    """
    Crée un retriever BM25 à partir des documents fournis
    (non vectoriel, basé sur la fréquence des mots clés)
    """
    if not docs:
        raise ValueError("Aucun document fourni pour créer le retriever BM25")
    
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 8  # Nombre de documents à retourner (modifiable)
    return retriever
