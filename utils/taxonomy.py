import re
import yaml

# 📌 Chargement de la taxonomie depuis config/taxonomie.yaml
with open("config/taxonomie.yaml", "r", encoding="utf-8") as f:
    TAXONOMY = yaml.safe_load(f)["taxonomie"]

def enrich_with_taxonomy(doc):
    text = doc.page_content
    lower_text = text.lower()

    doc.metadata["titre"] = extract_title(text)
    doc.metadata["secteur"] = guess_value_in_list(lower_text, TAXONOMY["secteurs"])
    doc.metadata["domaine"], doc.metadata["sous_domaine"] = guess_domain(lower_text)
    doc.metadata["livrables"] = guess_multiple_matches(lower_text, TAXONOMY["livrables"])
    doc.metadata["méthodologies"] = guess_multiple_matches(lower_text, TAXONOMY["méthodologies"])
    doc.metadata.update(extract_dynamic_vars(text))
    
    return doc

# 🔍 Cherche le premier match dans une liste (secteur, format, etc.)
def guess_value_in_list(text, liste):
    for item in liste:
        if item.lower() in text:
            return item
    return "Inconnu"

# 🧠 Détection du domaine/sous-domaine le plus probable
def guess_domain(text):
    for bloc in TAXONOMY["domaines"]:
        domaine = bloc["nom"]
        for sd in bloc["sous-domaines"]:
            if sd.lower() in text:
                return domaine, sd
    return "Inconnu", "Inconnu"

# 🔎 Liste des livrables ou méthodologies mentionnés
def guess_multiple_matches(text, items):
    return [item for item in items if item.lower() in text]

# 🧠 Titre = première ligne informative
def extract_title(text):
    for line in text.split("\n"):
        if len(line.strip()) > 10:
            return line.strip()
    return "Titre inconnu"

# 🏷 Variables dynamiques (client, durée, TJM)
def extract_dynamic_vars(text):
    return {
        "client": extract_regex(text, r"(?:client|nom du client)[ :]*([A-Z][a-zA-Z0-9 &-]{2,})"),
        "duration": extract_regex(text, r"(?:durée|duration)[ :]*([0-9]+ ?(?:jours|semaines|mois))"),
        "tjm": extract_regex(text, r"([0-9]{3,5} ?(?:TND|EUR|€))")
    }

def extract_regex(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Non spécifié"
