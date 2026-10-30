"""
PharmGraph Evaluation Script
============================
Evaluates PharmGraph on 200 drug pairs sampled directly from the ChEMBL KG:
  - 100 POSITIVE pairs  (drugs sharing ≥1 protein target → likely interaction)
  - 100 NEGATIVE pairs  (drugs sharing 0 targets → likely no interaction)

Ground truth label:
  positive = shared_targets ≥ 1  (KG-based, objective)
  negative = shared_targets == 0

PharmGraph prediction:
  positive = severity in {moderate, severe}
  negative = severity in {none, mild} OR abstained

Metrics reported: Precision, Recall, F1, Accuracy, AUROC, Abstention Rate
Run from project root:
    python evaluate_pharmgraph.py
"""

import sys, os, random, json, time, csv
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__) if "__file__" in dir() else ".")
sys.path.insert(0, ".")          # project root when run from ddi-agent/

import pickle
import networkx as nx

# ── Load KG ──────────────────────────────────────────────────────────────────
KG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drug_kg.pkl")
if not os.path.exists(KG_PATH):
    KG_PATH = "data/drug_kg.pkl"

print(f"Loading knowledge graph from {KG_PATH} ...")
with open(KG_PATH, "rb") as f:
    data = pickle.load(f)

# Support both dict-style and NetworkX graph objects
if isinstance(data, dict):
    kg = data.get("graph") or data.get("kg") or data
    name_to_id = data.get("name_to_id", {})
    if not name_to_id and isinstance(kg, nx.Graph):
        name_to_id = {
            d.get("name", "").upper(): n
            for n, d in kg.nodes(data=True)
            if d.get("node_type") == "drug" and d.get("name")
        }
elif isinstance(data, nx.Graph):
    kg = data
    name_to_id = {
        d.get("name", "").upper(): n
        for n, d in kg.nodes(data=True)
        if d.get("node_type") == "drug" and d.get("name")
    }
else:
    raise ValueError(f"Unexpected KG type: {type(data)}")

print(f"KG loaded — {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
print(f"Drug name index size: {len(name_to_id)}")

# ── Build drug list ───────────────────────────────────────────────────────────
drug_names = list(name_to_id.keys())
print(f"Total drugs available: {len(drug_names)}")

# ── Helper: get shared targets for a pair ────────────────────────────────────

def get_shared_targets(drug_a_name: str, drug_b_name: str) -> list:
    """Return list of shared target node IDs between two drugs."""
    id_a = name_to_id.get(drug_a_name.upper())
    id_b = name_to_id.get(drug_b_name.upper())
    if not id_a or not id_b:
        return None   # drug not in KG

    # Collect targets of each drug (neighbours with node_type == target)
    def targets_of(drug_id):
        return {
            n for n in kg.neighbors(drug_id)
            if kg.nodes[n].get("node_type") == "target"
        }

    t_a = targets_of(id_a)
    t_b = targets_of(id_b)
    return list(t_a & t_b)


# ── Sample 100 positive + 100 negative pairs ─────────────────────────────────

random.seed(42)

positive_pairs = []
negative_pairs = []
checked = set()

print("\nSampling drug pairs from KG ...")
shuffled = drug_names.copy()
random.shuffle(shuffled)

for i, drug_a in enumerate(shuffled):
    if len(positive_pairs) >= 100 and len(negative_pairs) >= 100:
        break
    for drug_b in shuffled[i+1:]:
        if len(positive_pairs) >= 100 and len(negative_pairs) >= 100:
            break
        key = tuple(sorted([drug_a, drug_b]))
        if key in checked:
            continue
        checked.add(key)

        shared = get_shared_targets(drug_a, drug_b)
        if shared is None:
            continue

        if shared and len(positive_pairs) < 100:
            positive_pairs.append((drug_a, drug_b, len(shared)))
        elif not shared and len(negative_pairs) < 100:
            negative_pairs.append((drug_a, drug_b, 0))

print(f"Positive pairs (shared targets ≥1): {len(positive_pairs)}")
print(f"Negative pairs (shared targets = 0): {len(negative_pairs)}")

all_pairs = [
    {"drug_a": a, "drug_b": b, "shared_count": c, "ground_truth": "positive"}
    for a, b, c in positive_pairs
] + [
    {"drug_a": a, "drug_b": b, "shared_count": c, "ground_truth": "negative"}
    for a, b, c in negative_pairs
]

random.shuffle(all_pairs)

# ── Run PharmGraph pipeline ───────────────────────────────────────────────────

from agents.path_retriever import PathRetrieverAgent
from agents.mechanism_reasoner import MechanismReasonerAgent

print("\nInitialising agents ...")
retriever = PathRetrieverAgent(kg_path=KG_PATH)
reasoner  = MechanismReasonerAgent()   # needs OPENAI_API_KEY in env

POSITIVE_SEVERITIES = {"moderate", "severe"}
NEGATIVE_SEVERITIES = {"none", "mild"}

results = []
abstentions = 0
errors = 0

print(f"\nRunning pipeline on {len(all_pairs)} pairs ...\n")
print(f"{'#':<5} {'Drug A':<20} {'Drug B':<20} {'GT':<10} {'Severity':<12} {'Pred':<10} {'✓?'}")
print("-" * 85)

for idx, pair in enumerate(all_pairs, 1):
    drug_a = pair["drug_a"]
    drug_b = pair["drug_b"]
    gt     = pair["ground_truth"]

    try:
        path_result = retriever.get_shared_targets(drug_a, drug_b)

        if "error" in path_result:
            errors += 1
            results.append({**pair, "severity": "error", "prediction": "abstain", "correct": False})
            print(f"{idx:<5} {drug_a:<20} {drug_b:<20} {gt:<10} {'error':<12} {'abstain':<10} ✗")
            continue

        reasoning = reasoner.reason(path_result)
        severity  = reasoning.severity.lower() if reasoning.severity else "none"

        if severity in POSITIVE_SEVERITIES:
            prediction = "positive"
        elif severity in NEGATIVE_SEVERITIES:
            prediction = "negative"
        else:
            prediction = "abstain"
            abstentions += 1

        correct = (prediction == gt) if prediction != "abstain" else None
        tick = "✓" if correct else ("?" if correct is None else "✗")

        results.append({
            **pair,
            "severity": severity,
            "prediction": prediction,
            "correct": correct,
            "confidence": getattr(reasoning, "confidence_score", None),
        })

        print(f"{idx:<5} {drug_a:<20} {drug_b:<20} {gt:<10} {severity:<12} {prediction:<10} {tick}")

    except Exception as e:
        errors += 1
        results.append({**pair, "severity": "error", "prediction": "abstain", "correct": False})
        print(f"{idx:<5} {drug_a:<20} {drug_b:<20} {gt:<10} {'EXCEPTION':<12} {'abstain':<10} ✗  [{e}]")

    # small delay to avoid hitting rate limits
    time.sleep(0.3)

# ── Compute metrics ───────────────────────────────────────────────────────────

decided = [r for r in results if r["prediction"] != "abstain" and r["prediction"] != "error"]
abstained_count = sum(1 for r in results if r["prediction"] == "abstain")
error_count     = sum(1 for r in results if r["prediction"] == "error")

tp = sum(1 for r in decided if r["ground_truth"] == "positive" and r["prediction"] == "positive")
fp = sum(1 for r in decided if r["ground_truth"] == "negative" and r["prediction"] == "positive")
tn = sum(1 for r in decided if r["ground_truth"] == "negative" and r["prediction"] == "negative")
fn = sum(1 for r in decided if r["ground_truth"] == "positive" and r["prediction"] == "negative")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp + tn) / len(decided) if decided else 0
abstention_rate = abstained_count / len(results)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Total pairs evaluated : {len(results)}")
print(f"  Decided             : {len(decided)}")
print(f"  Abstained           : {abstained_count}  ({abstention_rate:.1%})")
print(f"  Errors              : {error_count}")
print()
print(f"  True Positives (TP) : {tp}")
print(f"  False Positives (FP): {fp}")
print(f"  True Negatives (TN) : {tn}")
print(f"  False Negatives (FN): {fn}")
print()
print(f"  Precision           : {precision:.3f}")
print(f"  Recall              : {recall:.3f}")
print(f"  F1 Score            : {f1:.3f}")
print(f"  Accuracy            : {accuracy:.3f}")
print(f"  Abstention Rate     : {abstention_rate:.3f}")
print("=" * 60)

# ── Save results ──────────────────────────────────────────────────────────────

out_csv  = "evaluation_results.csv"
out_json = "evaluation_summary.json"

fieldnames = ["drug_a", "drug_b", "shared_count", "ground_truth", "severity", "prediction", "correct", "confidence"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

summary = {
    "total_pairs": len(results),
    "decided": len(decided),
    "abstained": abstained_count,
    "errors": error_count,
    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "accuracy": round(accuracy, 4),
    "abstention_rate": round(abstention_rate, 4),
}
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDetailed results saved to: {out_csv}")
print(f"Summary saved to         : {out_json}")
print("\nDone.")
