import json, argparse
from sklearn.metrics import accuracy_score, f1_score

p = argparse.ArgumentParser()
p.add_argument("--pred", required=True, help="Path to outputjson")
args = p.parse_args()

true, preds = [], []
with open(args.pred) as f:
    for row in json.load(f):
        true.append(row["gold_answer"])
        preds.append(row["model_answer"])

print("Accuracy :", accuracy_score(true, preds))
print("F1‑macro :", f1_score(true, preds, average="macro"))
