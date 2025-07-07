# 3. modules/loader.py (chargement + enrichissement docs)

import os
from langchain_community.document_loaders import PyMuPDFLoader
from utils.taxonomy import enrich_with_taxonomy

def load_and_tag_documents(folder_path):
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            enriched_docs = [enrich_with_taxonomy(doc) for doc in docs]
            all_docs.extend(enriched_docs)

    return all_docs
