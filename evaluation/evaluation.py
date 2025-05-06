import json, argparse
from sklearn.metrics import accuracy_score, f1_score

p = argparse.ArgumentParser()
p.add_argument("--pred", required=True, help="Path to predictions JSON")
args = p.parse_args()

true, preds = [], []
with open(args.pred) as f:
    for row in json.load(f):
        if row["true_answer"] == "":
            continue
        true.append(row["true_answer"].lower())
        preds.append(row["model_answer"].lower())

print("Accuracy :", accuracy_score(true, preds))
print("F1‑macro :", f1_score(true, preds, average="macro"))
