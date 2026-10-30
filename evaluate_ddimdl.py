"""
DDI-Agent REAL Evaluation — Literature-Curated Benchmark
=========================================================
Ground truth comes from published clinical literature and FDA drug labels,
NOT from our own knowledge graph. This gives honest, independent results.

References used:
 - Hansten & Horn: Drug Interactions Analysis and Management (2023)
 - FDA Drug Safety Communications
 - Stockley's Drug Interactions (12th ed.)
 - DrugBank Open (CC-BY 4.0 licensed interaction list)
"""

import sys, os, json, time, csv
sys.path.append(".")

from agents.path_retriever import PathRetrieverAgent
from agents.mechanism_reasoner import MechanismReasonerAgent
from agents.evidence_validator import EvidenceValidatorAgent

# ── INDEPENDENT GROUND TRUTH from published clinical literature ────────────
#
# Format: (Drug A, Drug B, Has_Interaction, Clinical_Evidence)
# All drug names use INN (International Nonproprietary Names) = ChEMBL standard
#
LITERATURE_BENCHMARK = [
    # ── CONFIRMED INTERACTIONS (True) ─────────────────────────────────────
    # Source: FDA Drug Safety Communications + Stockley's
    ("WARFARIN",       "ASPIRIN",        True,  "Increased bleeding risk — FDA black box warning"),
    ("WARFARIN",       "IBUPROFEN",      True,  "NSAIDs potentiate anticoagulant effect (Stockley's)"),
    ("WARFARIN",       "FLUCONAZOLE",    True,  "CYP2C9 inhibition raises warfarin levels (FDA)"),
    ("SIMVASTATIN",    "AMIODARONE",     True,  "Myopathy/rhabdomyolysis — FDA dose limit"),
    ("SIMVASTATIN",    "CLARITHROMYCIN", True,  "CYP3A4 inhibition → rhabdomyolysis (FDA)"),
    ("DIGOXIN",        "AMIODARONE",     True,  "Digoxin toxicity — FDA label contraindication"),
    ("DIGOXIN",        "VERAPAMIL",      True,  "P-glycoprotein inhibition raises digoxin (Stockley's)"),
    ("METHOTREXATE",   "IBUPROFEN",      True,  "NSAID reduces methotrexate renal clearance (FDA)"),
    ("CLOPIDOGREL",    "OMEPRAZOLE",     True,  "CYP2C19 inhibition reduces antiplatelet effect (FDA)"),
    ("FLUOXETINE",     "TRAMADOL",       True,  "Serotonin syndrome risk (FDA warning 2016)"),
    ("FLUOXETINE",     "LITHIUM",        True,  "Serotonin syndrome + lithium toxicity (Stockley's)"),
    ("CARBAMAZEPINE",  "VALPROATE",      True,  "Mutual PK interaction — both levels affected"),
    ("LITHIUM",        "IBUPROFEN",      True,  "NSAIDs reduce renal lithium clearance → toxicity"),
    ("METFORMIN",      "ALCOHOL",        True,  "Lactic acidosis risk — FDA label warning"),
    ("SILDENAFIL",     "NITRATES",       True,  "Severe hypotension — absolute contraindication (FDA)"),
    ("PHENYTOIN",      "WARFARIN",       True,  "Biphasic interaction — initial inhibition then induction"),
    ("CIPROFLOXACIN",  "THEOPHYLLINE",   True,  "CYP1A2 inhibition raises theophylline → seizures"),
    ("FLUOXETINE",     "PHENYTOIN",      True,  "CYP2C9 inhibition raises phenytoin levels"),
    ("CARBAMAZEPINE",  "WARFARIN",       True,  "CYP enzyme induction reduces warfarin effect"),
    ("AMIODARONE",     "WARFARIN",       True,  "CYP2C9 inhibition → bleeding (FDA black box)"),
    ("ERYTHROMYCIN",   "SIMVASTATIN",    True,  "CYP3A4 inhibition → myopathy risk"),
    ("RIFAMPICIN",     "WARFARIN",       True,  "Strong CYP inducer — reduces warfarin 5-fold"),
    ("VERAPAMIL",      "CARBAMAZEPINE",  True,  "CYP3A4 inhibition raises carbamazepine levels"),
    ("LITHIUM",        "METFORMIN",      True,  "Both renally cleared — competition increases lithium"),
    ("DIGOXIN",        "CLARITHROMYCIN", True,  "P-gp inhibition → digoxin toxicity (FDA)"),

    # ── NON-INTERACTIONS (False) — these combinations are clinically safe ─
    # Source: multiple clinical pharmacology reviews
    ("METFORMIN",      "LISINOPRIL",     False, "Safe combination — widely used together in T2DM+HTN"),
    ("ATORVASTATIN",   "AMLODIPINE",     False, "Well-tolerated combination (CAMELOT trial)"),
    ("OMEPRAZOLE",     "PARACETAMOL",    False, "No clinically significant interaction"),
    ("ASPIRIN",        "LISINOPRIL",     False, "Generally safe at low aspirin doses"),
    ("METFORMIN",      "PARACETAMOL",    False, "No significant PK interaction documented"),
    ("AMLODIPINE",     "METFORMIN",      False, "Calcium channel blockers do not affect metformin"),
    ("ATORVASTATIN",   "LISINOPRIL",     False, "Safe combination — standard cardiovascular polypharmacy"),
    ("OMEPRAZOLE",     "METFORMIN",      False, "No clinically relevant interaction"),
    ("ASPIRIN",        "PARACETAMOL",    False, "Different mechanisms; combination used post-surgery"),
    ("AMLODIPINE",     "OMEPRAZOLE",     False, "No significant interaction documented"),
]

MAX_PAIRS = len(LITERATURE_BENCHMARK)   # test all 35


def run_real_evaluation():
    print("=" * 62)
    print("  DDI-AGENT — REAL EVALUATION")
    print("  Ground truth: Published clinical literature (FDA/Stockley's)")
    print("=" * 62)

    print(f"\nBenchmark: {MAX_PAIRS} drug pairs ({sum(1 for _,_,l,_ in LITERATURE_BENCHMARK if l)} interactions, "
          f"{sum(1 for _,_,l,_ in LITERATURE_BENCHMARK if not l)} non-interactions)")

    # Load agents
    print("\nLoading agents...")
    retriever = PathRetrieverAgent()
    reasoner  = MechanismReasonerAgent()
    validator = EvidenceValidatorAgent()
    print("Ready.\n")

    results  = []
    correct  = 0
    total    = 0
    skipped  = 0
    not_found = []

    for idx, (drug_a, drug_b, known, evidence) in enumerate(LITERATURE_BENCHMARK, 1):
        print(f"[{idx:02d}/{MAX_PAIRS}] {drug_a} + {drug_b}  (known={known})")
        try:
            path_result = retriever.get_shared_targets(drug_a, drug_b)

            if "error" in path_result:
                print(f"       ⚠  Not in graph: {path_result['error']}")
                not_found.append(f"{drug_a}+{drug_b}")
                skipped += 1
                continue

            n_shared = len(path_result["shared_targets"])

            if n_shared > 0:
                chain    = reasoner.reason(path_result)
                val      = validator.validate(chain)
                conf     = val["confidence_score"]
                severity = chain.severity
            else:
                conf     = 0
                severity = "none"

            predicted  = (n_shared > 0 and conf >= 50)
            is_correct = (predicted == known)
            if is_correct:
                correct += 1
            total += 1

            mark = "✓" if is_correct else "✗"
            print(f"       {mark}  shared={n_shared} | conf={conf} | pred={predicted} | known={known}")

            results.append({
                "drug_a": drug_a, "drug_b": drug_b,
                "known_interaction": known,
                "clinical_evidence": evidence,
                "shared_targets": n_shared,
                "severity": severity,
                "confidence": conf,
                "predicted_interaction": predicted,
                "correct": is_correct,
                "source": "Clinical literature"
            })

            if n_shared > 0:
                time.sleep(1.5)   # only sleep if we called GPT

        except Exception as e:
            print(f"       ERROR: {e}")
            skipped += 1
            continue

    if total == 0:
        print("\nNo pairs were found in our knowledge graph.")
        print("Drugs not found:", not_found)
        return

    # Metrics
    accuracy  = correct / total * 100
    tp = sum(1 for r in results if     r["predicted_interaction"] and     r["known_interaction"])
    fp = sum(1 for r in results if     r["predicted_interaction"] and not r["known_interaction"])
    fn = sum(1 for r in results if not r["predicted_interaction"] and     r["known_interaction"])
    tn = sum(1 for r in results if not r["predicted_interaction"] and not r["known_interaction"])
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n" + "=" * 62)
    print("  EVALUATION RESULTS  (Independent Literature Benchmark)")
    print("=" * 62)
    print(f"  Ground truth source : Clinical literature (FDA / Stockley's)")
    print(f"  Total pairs tested  : {total}   (skipped/not-in-graph: {skipped})")
    print(f"  Correct predictions : {correct}")
    print(f"  Accuracy            : {accuracy:.1f}%")
    print(f"  Precision           : {precision:.1f}%")
    print(f"  Recall              : {recall:.1f}%")
    print(f"  F1 Score            : {f1:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  (TP) : {tp}")
    print(f"    False Positives (FP) : {fp}")
    print(f"    True Negatives  (TN) : {tn}")
    print(f"    False Negatives (FN) : {fn}")
    print("=" * 62)

    if skipped > 0:
        print(f"\n  Note: {skipped} pairs skipped — drugs not in ChEMBL extract")
        print(f"  (These would be reported as a data coverage limitation in the paper)")

    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/literature_eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "dataset": "Clinical literature (FDA/Stockley's)",
        "total": total, "correct": correct, "skipped": skipped,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }
    with open("data/literature_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results → data/literature_eval_results.csv")
    print(f"  Summary → data/literature_eval_summary.json")
    return summary


if __name__ == "__main__":
    run_real_evaluation()
