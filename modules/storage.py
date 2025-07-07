import json
from datetime import datetime
import os

LOG_FILE = "storage/generated_responses.jsonl"

def store_generation(query, context, metadata, response):
    """
    Enregistre chaque génération dans un fichier .jsonl
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    generation = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "context": context[:1000],  # pour éviter les logs trop lourds
        "metadata": metadata,
        "response": response
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(generation, ensure_ascii=False) + "\n")
