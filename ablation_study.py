"""
DDI-Agent Ablation Study
========================
Compares four methods on the same 29-pair clinical benchmark:

  Method 1 — Shared-target rule only  (no LLM)
  Method 2 — Single LLM call          (no graph, no agents)
  Method 3 — Graph + single LLM       (graph info, one prompt)
  Method 4 — Full 4-agent pipeline    (our complete system)

This proves (or disproves) that each stage adds value.
"""

import sys, os, json, time, csv
sys.path.append(".")

from agents.path_retriever import PathRetrieverAgent
from agents.mechanism_reasoner import MechanismReasonerAgent
from agents.evidence_validator import EvidenceValidatorAgent
from openai import OpenAI

client = OpenAI()

# ── Same benchmark as evaluate_ddimdl.py ──────────────────────────────────
BENCHMARK = [
    ("WARFARIN",       "ASPIRIN",        True),
    ("WARFARIN",       "IBUPROFEN",      True),
    ("WARFARIN",       "FLUCONAZOLE",    True),
    ("SIMVASTATIN",    "AMIODARONE",     True),
    ("SIMVASTATIN",    "CLARITHROMYCIN", True),
    ("DIGOXIN",        "AMIODARONE",     True),
    ("DIGOXIN",        "VERAPAMIL",      True),
    ("METHOTREXATE",   "IBUPROFEN",      True),
    ("CLOPIDOGREL",    "OMEPRAZOLE",     True),
    ("FLUOXETINE",     "TRAMADOL",       True),
    ("FLUOXETINE",     "LITHIUM",        True),
    ("CARBAMAZEPINE",  "VALPROATE",      True),
    ("LITHIUM",        "IBUPROFEN",      True),
    ("METFORMIN",      "ALCOHOL",        True),
    ("SILDENAFIL",     "NITRATES",       True),
    ("PHENYTOIN",      "WARFARIN",       True),
    ("CIPROFLOXACIN",  "THEOPHYLLINE",   True),
    ("FLUOXETINE",     "PHENYTOIN",      True),
    ("CARBAMAZEPINE",  "WARFARIN",       True),
    ("AMIODARONE",     "WARFARIN",       True),
    ("ERYTHROMYCIN",   "SIMVASTATIN",    True),
    ("RIFAMPICIN",     "WARFARIN",       True),
    ("VERAPAMIL",      "CARBAMAZEPINE",  True),
    ("LITHIUM",        "METFORMIN",      True),
    ("DIGOXIN",        "CLARITHROMYCIN", True),
    ("METFORMIN",      "LISINOPRIL",     False),
    ("ATORVASTATIN",   "AMLODIPINE",     False),
    ("ASPIRIN",        "LISINOPRIL",     False),
    ("AMLODIPINE",     "METFORMIN",      False),
    ("ATORVASTATIN",   "LISINOPRIL",     False),
    ("OMEPRAZOLE",     "METFORMIN",      False),
    ("AMLODIPINE",     "OMEPRAZOLE",     False),
]


def metrics(results):
    """Compute accuracy, precision, recall, F1 from list of (predicted, known)."""
    tp = sum(1 for p, k in results if p and k)
    fp = sum(1 for p, k in results if p and not k)
    fn = sum(1 for p, k in results if not p and k)
    tn = sum(1 for p, k in results if not p and not k)
    total = tp + fp + fn + tn
    acc  = (tp + tn) / total * 100 if total else 0
    prec = tp / (tp + fp) * 100   if (tp + fp) else 0
    rec  = tp / (tp + fn) * 100   if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total}


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: Shared-target rule only  (no LLM at all)
# ══════════════════════════════════════════════════════════════════════════════

def run_method1(retriever, pairs):
    """Predict interaction if and only if shared_targets > 0."""
    print("\n── Method 1: Shared-target rule only ──────────────────────")
    preds = []
    skipped = []
    for drug_a, drug_b, known in pairs:
        r = retriever.get_shared_targets(drug_a, drug_b)
        if "error" in r:
            skipped.append(f"{drug_a}+{drug_b}")
            continue
        predicted = len(r["shared_targets"]) > 0
        mark = "✓" if predicted == known else "✗"
        print(f"  {mark}  {drug_a}+{drug_b}  shared={len(r['shared_targets'])} pred={predicted} known={known}")
        preds.append((predicted, known))
    if skipped:
        print(f"  Skipped (not in graph): {skipped}")
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: Single LLM call — no graph, no agents
# ══════════════════════════════════════════════════════════════════════════════

def run_method2(pairs):
    """Ask GPT directly 'do these drugs interact?' — no graph info at all."""
    print("\n── Method 2: Single LLM call (no graph) ───────────────────")
    preds = []
    for drug_a, drug_b, known in pairs:
        prompt = (
            f"Do {drug_a} and {drug_b} have a clinically significant drug-drug interaction? "
            "Answer with exactly one word: YES or NO."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        answer = resp.choices[0].message.content.strip().upper()
        predicted = "YES" in answer
        mark = "✓" if predicted == known else "✗"
        print(f"  {mark}  {drug_a}+{drug_b}  llm={answer} pred={predicted} known={known}")
        preds.append((predicted, known))
        time.sleep(0.5)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: Graph info + single LLM call (retrieval-augmented, no agents)
# ══════════════════════════════════════════════════════════════════════════════

def run_method3(retriever, pairs):
    """Give GPT the shared-target list and ask it to decide — one prompt."""
    print("\n── Method 3: Graph + single LLM (no agent chain) ──────────")
    preds = []
    skipped = []
    for drug_a, drug_b, known in pairs:
        r = retriever.get_shared_targets(drug_a, drug_b)
        if "error" in r:
            skipped.append(f"{drug_a}+{drug_b}")
            continue
        shared = r["shared_targets"]
        if shared:
            names = [t.get('target_name', str(t)) if isinstance(t, dict) else str(t) for t in shared[:5]]
            context = f"They share {len(shared)} protein targets: {', '.join(names)}."
        else:
            context = "They share no protein targets in the knowledge graph."

        prompt = (
            f"Drug A: {drug_a}\nDrug B: {drug_b}\n"
            f"Knowledge graph context: {context}\n\n"
            "Based on this, do these drugs have a clinically significant interaction? "
            "Answer YES or NO only."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        answer = resp.choices[0].message.content.strip().upper()
        predicted = "YES" in answer
        mark = "✓" if predicted == known else "✗"
        print(f"  {mark}  {drug_a}+{drug_b}  shared={len(shared)} llm={answer} pred={predicted} known={known}")
        preds.append((predicted, known))
        time.sleep(0.5)
    if skipped:
        print(f"  Skipped (not in graph): {skipped}")
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4: Full 4-agent pipeline  (our system)
# ══════════════════════════════════════════════════════════════════════════════

def run_method4(retriever, reasoner, validator, pairs):
    """Full pipeline: graph → mechanism reasoning → evidence validation → predict."""
    print("\n── Method 4: Full 4-agent pipeline (our system) ───────────")
    preds = []
    skipped = []
    for drug_a, drug_b, known in pairs:
        r = retriever.get_shared_targets(drug_a, drug_b)
        if "error" in r:
            skipped.append(f"{drug_a}+{drug_b}")
            continue
        n_shared = len(r["shared_targets"])
        if n_shared > 0:
            chain = reasoner.reason(r)
            val   = validator.validate(chain)
            conf  = val["confidence_score"]
        else:
            conf = 0
        predicted = (n_shared > 0 and conf >= 50)
        mark = "✓" if predicted == known else "✗"
        print(f"  {mark}  {drug_a}+{drug_b}  shared={n_shared} conf={conf} pred={predicted} known={known}")
        preds.append((predicted, known))
        if n_shared > 0:
            time.sleep(1.5)
    if skipped:
        print(f"  Skipped (not in graph): {skipped}")
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation():
    print("=" * 62)
    print("  DDI-AGENT ABLATION STUDY")
    print("  Comparing 4 methods on identical benchmark pairs")
    print("=" * 62)

    print("\nLoading agents...")
    retriever = PathRetrieverAgent()
    reasoner  = MechanismReasonerAgent()
    validator = EvidenceValidatorAgent()
    print("Ready.\n")

    pairs = BENCHMARK

    # Run all 4 methods
    r1 = run_method1(retriever, pairs)
    r2 = run_method2(pairs)
    r3 = run_method3(retriever, pairs)
    r4 = run_method4(retriever, reasoner, validator, pairs)

    # Compute metrics for each
    m1 = metrics(r1)
    m2 = metrics(r2)
    m3 = metrics(r3)
    m4 = metrics(r4)

    # Print comparison table
    print("\n\n" + "=" * 70)
    print("  ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"  {'Method':<40} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 70)
    print(f"  {'1. Shared-target rule (no LLM)':<40} {m1['accuracy']:>5.1f}% {m1['precision']:>6.1f}% {m1['recall']:>6.1f}% {m1['f1']:>6.1f}%")
    print(f"  {'2. Single LLM call (no graph)':<40} {m2['accuracy']:>5.1f}% {m2['precision']:>6.1f}% {m2['recall']:>6.1f}% {m2['f1']:>6.1f}%")
    print(f"  {'3. Graph + single LLM':<40} {m3['accuracy']:>5.1f}% {m3['precision']:>6.1f}% {m3['recall']:>6.1f}% {m3['f1']:>6.1f}%")
    print(f"  {'4. Full 4-agent pipeline (ours)':<40} {m4['accuracy']:>5.1f}% {m4['precision']:>6.1f}% {m4['recall']:>6.1f}% {m4['f1']:>6.1f}%")
    print("=" * 70)

    # Save results
    os.makedirs("data", exist_ok=True)
    summary = {
        "method_1_shared_target_rule": m1,
        "method_2_single_llm": m2,
        "method_3_graph_plus_llm": m3,
        "method_4_full_pipeline": m4,
    }
    with open("data/ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  Results saved to data/ablation_results.json")

    return summary


if __name__ == "__main__":
    run_ablation()
