"""
DDI-Agent Evaluation Script — 100 drug pairs
"""
import sys, os, json, time, csv
sys.path.append(".")

from agents.path_retriever import PathRetrieverAgent
from agents.mechanism_reasoner import MechanismReasonerAgent
from agents.evidence_validator import EvidenceValidatorAgent


def run_evaluation(pairs_csv="data/eval_pairs.csv"):
    print("Loading agents...")
    retriever = PathRetrieverAgent()
    reasoner  = MechanismReasonerAgent()
    validator = EvidenceValidatorAgent()
    print("Ready.\n")

    # Load pairs
    pairs = []
    with open(pairs_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((
                row["drug_a"],
                row["drug_b"],
                row["known_interaction"].lower() == "true"
            ))
    print(f"Loaded {len(pairs)} pairs to evaluate.\n")

    results = []
    correct = 0
    total   = 0

    os.makedirs("data", exist_ok=True)

    for idx, (drug_a, drug_b, known_interaction) in enumerate(pairs, 1):
        print(f"[{idx:03d}/{len(pairs)}] {drug_a} + {drug_b}")
        try:
            path_result = retriever.get_shared_targets(drug_a, drug_b)
            if "error" in path_result:
                print(f"       ⚠  Skipped: {path_result['error']}")
                continue

            n_shared = len(path_result["shared_targets"])
            chain    = reasoner.reason(path_result)
            val      = validator.validate(chain)
            conf     = val["confidence_score"]

            predicted = (n_shared > 0 and conf >= 50)
            is_correct = (predicted == known_interaction)
            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"       {status}  shared={n_shared} | severity={chain.severity} | conf={conf} | pred={predicted} | known={known_interaction}")

            results.append({
                "drug_a": drug_a, "drug_b": drug_b,
                "known_interaction": known_interaction,
                "shared_targets": n_shared,
                "severity": chain.severity,
                "confidence": conf,
                "predicted_interaction": predicted,
                "correct": is_correct
            })

            time.sleep(1.5)

        except Exception as e:
            print(f"       ERROR: {e}")
            continue

    # Metrics
    accuracy  = correct / total * 100 if total else 0
    tp = sum(1 for r in results if     r["predicted_interaction"] and     r["known_interaction"])
    fp = sum(1 for r in results if     r["predicted_interaction"] and not r["known_interaction"])
    fn = sum(1 for r in results if not r["predicted_interaction"] and     r["known_interaction"])
    tn = sum(1 for r in results if not r["predicted_interaction"] and not r["known_interaction"])
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "="*55)
    print("  EVALUATION RESULTS  (100-pair benchmark)")
    print("="*55)
    print(f"  Total pairs tested : {total}")
    print(f"  Correct            : {correct}")
    print(f"  Accuracy           : {accuracy:.1f}%")
    print(f"  Precision          : {precision:.1f}%")
    print(f"  Recall             : {recall:.1f}%")
    print(f"  F1 Score           : {f1:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  : {tp}")
    print(f"    False Positives : {fp}")
    print(f"    True Negatives  : {tn}")
    print(f"    False Negatives : {fn}")
    print("="*55)

    # Save CSV
    with open("data/evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Save JSON summary
    summary = {
        "total": total, "correct": correct,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }
    with open("data/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to data/evaluation_results.csv")
    print(f"Summary saved to data/evaluation_summary.json")
    return summary


if __name__ == "__main__":
    run_evaluation()
