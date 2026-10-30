"""
PharmGraph Ablation Study
=========================
Compares three conditions on the same 200 drug pairs used in the original eval:

  1. Rule-only      : shared_targets >= 1  →  positive  (no LLM, free)
  2. LLM-only       : GPT-4o-mini with no graph context  (~200 API calls)
  3. Full pipeline  : KG + LLM (loaded from existing evaluation_results.csv)

Also performs Error Analysis on the full pipeline's 41 false negatives.

Prerequisites:
  - evaluation_results.csv must exist (run evaluate_pharmgraph.py first)
  - OPENAI_API_KEY must be set in your environment

Run from project root:
    python evaluate_baselines.py
"""

import sys, os, csv, json, time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

RESULTS_CSV  = "evaluation_results.csv"
KG_PATH      = "data/drug_kg.pkl"
OUT_JSON     = "ablation_summary.json"

# ── load existing full-pipeline results ───────────────────────────────────────

if not Path(RESULTS_CSV).exists():
    print(f"ERROR: {RESULTS_CSV} not found. Run evaluate_pharmgraph.py first.")
    sys.exit(1)

print(f"Loading existing evaluation results from {RESULTS_CSV} ...")
pipeline_rows = []
with open(RESULTS_CSV, newline="") as f:
    for row in csv.DictReader(f):
        pipeline_rows.append(row)

print(f"  Loaded {len(pipeline_rows)} rows.")

# ── load KG for rule-only baseline ───────────────────────────────────────────

import pickle
import networkx as nx

print(f"\nLoading knowledge graph from {KG_PATH} ...")
with open(KG_PATH, "rb") as f:
    data = pickle.load(f)

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

print(f"  KG loaded — {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

def kg_shared_count(drug_a: str, drug_b: str) -> int:
    id_a = name_to_id.get(drug_a.upper())
    id_b = name_to_id.get(drug_b.upper())
    if not id_a or not id_b:
        return -1  # unknown
    def targets_of(nid):
        return {n for n in kg.neighbors(nid) if kg.nodes[n].get("node_type") == "target"}
    return len(targets_of(id_a) & targets_of(id_b))

# ── helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(results):
    """results is a list of dicts with ground_truth and prediction keys."""
    decided = [r for r in results if r["prediction"] not in ("abstain", "error")]
    tp = sum(1 for r in decided if r["ground_truth"] == "positive" and r["prediction"] == "positive")
    fp = sum(1 for r in decided if r["ground_truth"] == "negative" and r["prediction"] == "positive")
    tn = sum(1 for r in decided if r["ground_truth"] == "negative" and r["prediction"] == "negative")
    fn = sum(1 for r in decided if r["ground_truth"] == "positive" and r["prediction"] == "negative")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
    accuracy  = (tp + tn) / len(decided) if decided else 0
    abstain   = sum(1 for r in results if r["prediction"] in ("abstain", "error"))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(precision,3), recall=round(recall,3),
                f1=round(f1,3), accuracy=round(accuracy,3),
                abstain=abstain, n=len(results))

def print_metrics(label, m):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}  Abstain={m['abstain']}")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1        : {m['f1']:.3f}")
    print(f"  Accuracy  : {m['accuracy']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 1 — Rule Only (KG shared-target threshold)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 1 — Rule Only (shared targets ≥ 1 → positive)")
print("="*60)

rule_results = []
for row in pipeline_rows:
    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    gt     = row["ground_truth"]
    count  = kg_shared_count(drug_a, drug_b)
    if count < 0:
        pred = "abstain"
    elif count >= 1:
        pred = "positive"
    else:
        pred = "negative"
    rule_results.append({"ground_truth": gt, "prediction": pred,
                          "drug_a": drug_a, "drug_b": drug_b, "shared_count": count})

rule_m = compute_metrics(rule_results)
print_metrics("BASELINE 1 — Rule Only", rule_m)

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 2 — LLM Only (no graph context)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("BASELINE 2 — LLM Only (GPT-4o-mini, no graph context)")
print("="*60)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

LLM_SYSTEM = (
    "You are a clinical pharmacology expert. "
    "Given two drug names, decide if they have a clinically significant drug-drug interaction. "
    "Reply with exactly one line in this format:\n"
    "SEVERITY: <none|mild|moderate|severe>\n"
    "Do not explain. Only output the SEVERITY line."
)

POSITIVE_SEVERITIES = {"moderate", "severe"}
NEGATIVE_SEVERITIES = {"none", "mild"}

llm_results = []
print(f"\n{'#':<5} {'Drug A':<20} {'Drug B':<20} {'GT':<10} {'Severity':<12} {'Pred'}")
print("-" * 80)

for idx, row in enumerate(pipeline_rows, 1):
    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    gt     = row["ground_truth"]

    try:
        prompt = f"Drug A: {drug_a}\nDrug B: {drug_b}"
        resp = llm.invoke([SystemMessage(content=LLM_SYSTEM),
                           HumanMessage(content=prompt)])
        text = resp.content.strip()
        # parse SEVERITY: <value>
        severity = "unknown"
        for line in text.splitlines():
            if line.upper().startswith("SEVERITY:"):
                severity = line.split(":", 1)[1].strip().lower()
                break

        if severity in POSITIVE_SEVERITIES:
            pred = "positive"
        elif severity in NEGATIVE_SEVERITIES:
            pred = "negative"
        else:
            pred = "abstain"

        llm_results.append({"ground_truth": gt, "prediction": pred,
                             "drug_a": drug_a, "drug_b": drug_b, "severity": severity})
        tick = "✓" if pred == gt else ("?" if pred == "abstain" else "✗")
        print(f"{idx:<5} {drug_a:<20} {drug_b:<20} {gt:<10} {severity:<12} {pred}  {tick}")

    except Exception as e:
        llm_results.append({"ground_truth": gt, "prediction": "error",
                             "drug_a": drug_a, "drug_b": drug_b, "severity": "error"})
        print(f"{idx:<5} {drug_a:<20} {drug_b:<20} {gt:<10} {'ERROR':<12} error  [{e}]")

    time.sleep(0.3)

llm_m = compute_metrics(llm_results)
print_metrics("BASELINE 2 — LLM Only", llm_m)

# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — load from existing CSV
# ═══════════════════════════════════════════════════════════════════════════════

pipeline_results = []
for row in pipeline_rows:
    pred = row.get("prediction", "abstain")
    if pred not in ("positive", "negative"):
        pred = "abstain"
    pipeline_results.append({"ground_truth": row["ground_truth"], "prediction": pred,
                              "drug_a": row["drug_a"], "drug_b": row["drug_b"],
                              "severity": row.get("severity", "")})

pipe_m = compute_metrics(pipeline_results)
print_metrics("FULL PIPELINE (KG + LLM)", pipe_m)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*72)
print("  ABLATION COMPARISON TABLE")
print("="*72)
print(f"  {'Condition':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")

def row_str(label, m):
    return (f"  {label:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} "
            f"{m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4}")

print(row_str("Baseline 1: Rule Only",       rule_m))
print(row_str("Baseline 2: LLM Only",        llm_m))
print(row_str("Full Pipeline (KG + LLM)",    pipe_m))
print("="*72)

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR ANALYSIS — False Negatives in Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  ERROR ANALYSIS — False Negatives (Full Pipeline)")
print("="*60)

false_negatives = [
    row for row in pipeline_rows
    if row.get("ground_truth") == "positive"
    and row.get("prediction") == "negative"
]

print(f"\n  Total FN: {len(false_negatives)}")

# Categorise by severity returned
severity_counts = {}
for row in false_negatives:
    sev = row.get("severity", "unknown").lower()
    severity_counts[sev] = severity_counts.get(sev, 0) + 1

print("\n  Severity returned for FN cases (LLM said 'no interaction' but KG says positive):")
for sev, cnt in sorted(severity_counts.items(), key=lambda x: -x[1]):
    print(f"    {sev:<15}: {cnt:>3} cases")

# Categorise by shared target count (from our rule baseline)
count_buckets = {"1 shared target": 0, "2–3 shared targets": 0, "4+ shared targets": 0}
for row in false_negatives:
    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    count = int(row.get("shared_count", 0))
    if count == 0:
        # Re-compute from KG (shared_count in CSV is from the pipeline, may differ)
        count = kg_shared_count(drug_a, drug_b)
    if count == 1:
        count_buckets["1 shared target"] += 1
    elif 2 <= count <= 3:
        count_buckets["2–3 shared targets"] += 1
    elif count >= 4:
        count_buckets["4+ shared targets"] += 1

print("\n  Shared target count for FN cases (how much graph evidence existed):")
for bucket, cnt in count_buckets.items():
    print(f"    {bucket:<25}: {cnt:>3} cases")

print("\n  Sample FN cases (first 10):")
print(f"  {'Drug A':<20} {'Drug B':<20} {'Shared':>6} {'Severity':<12}")
print(f"  {'-'*20} {'-'*20} {'-'*6} {'-'*12}")
for row in false_negatives[:10]:
    drug_a = row["drug_a"]
    drug_b = row["drug_b"]
    count  = row.get("shared_count", "?")
    sev    = row.get("severity", "?")
    print(f"  {drug_a:<20} {drug_b:<20} {count:>6} {sev:<12}")

# ── save summary ──────────────────────────────────────────────────────────────

summary = {
    "rule_only":      rule_m,
    "llm_only":       llm_m,
    "full_pipeline":  pipe_m,
    "false_negative_analysis": {
        "total_fn": len(false_negatives),
        "severity_breakdown": severity_counts,
        "shared_target_count_breakdown": count_buckets,
    }
}
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Ablation summary saved to: {OUT_JSON}")
print("\nDone.")
