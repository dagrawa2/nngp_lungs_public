import json

def save_json(D, path):
	with open(path, "w") as f:
		json.dump(D, f, indent=2)

def load_json(path):
	with open(path, "r") as f:
		D = json.load(f)
	return D
