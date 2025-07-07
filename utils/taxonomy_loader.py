import yaml

def load_taxonomy():
    with open("config/taxonomie.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["taxonomie"]
