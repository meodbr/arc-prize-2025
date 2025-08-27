import json

def load_arc_challenges(filename: str):
    return json.load(filename)

def save_arc_challenges(data, filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)