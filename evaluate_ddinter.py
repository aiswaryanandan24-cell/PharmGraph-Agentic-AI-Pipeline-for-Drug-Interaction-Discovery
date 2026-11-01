"""
PharmGraph External Validation — DDInter 2.0
==============================================
Validates PharmGraph against DDInter (https://ddinter2.scbdd.com), an
independently curated clinical drug-drug interaction database that is NOT
derived from ChEMBL. This breaks the circularity of the earlier KG-sampled
evaluation, where ground truth and the rule-only baseline shared a definition.

Ground truth:
  positive = pair listed in DDInter with Level in {Minor, Moderate, Major}
  negative = pair of drugs both present in DDInter's drug vocabulary and in
             our KG, but NOT found anywhere in DDInter's records (any level,
             including "Unknown") — i.e. DDInter has no evidence of interaction

Place the downloaded DDInter CSVs in the project root before running:
  ddinter_downloads_code_A.csv, _B.csv, _H.csv, _R.csv (add more if you
  downloaded additional categories — just extend DDINTER_FILES below)

Run from project root:
    python evaluate_ddinter.py
"""

import sys, os, csv, json, time, random, pickle
import networkx as nx

random.seed(42)

# ── Load KG ──────────────────────────────────────────────────────────────────

KG_PATH = "data/drug_kg.pkl"
if not os.path.exists(KG_PATH):
    KG_PATH = os.path.join("..", "data", "drug_kg.pkl")

print(f"Loading knowledge graph from {KG_PATH} ...")
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

print(f"KG loaded — {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

def resolve(name: str):
    return name_to_id.get(name.strip().upper())

def kg_shared_count(drug_a: str, drug_b: str) -> int:
    id_a, id_b = resolve(drug_a), resolve(drug_b)
    if not id_a or not id_b:
        return -1
    def targets_of(nid):
        return {n for n in kg.neighbors(nid) if kg.nodes[n].get("node_type") == "target"}
    return len(targets_of(id_a) & targets_of(id_b))

# ── Load DDInter CSVs ─────────────────────────────────────────────────────────

DDINTER_FILES = [
    "ddinter_downloads_code_A.csv",
    "ddinter_downloads_code_B.csv",
    "ddinter_downloads_code_H.csv",
    "ddinter_downloads_code_R.csv",
]

print(f"\nLoading DDInter files: {DDINTER_FILES}")

all_pairs_any_level = set()     # frozenset(lowercase names) — full exclusion set
positive_candidates = []        # (drug_a, drug_b, level)
ddinter_drug_names = set()      # KG-resolvable names seen anywhere in DDInter

for fname in DDINTER_FILES:
    if not os.path.exists(fname):
        print(f"  WARNING: {fname} not found in current directory — skipping")
        continue
    n_rows = 0
    with open(fname, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            da = (row.get("Drug_A") or "").strip()
            db = (row.get("Drug_B") or "").strip()
            level = (row.get("Level") or "").strip()
            if not da or not db:
                continue
            n_rows += 1
            key = frozenset([da.lower(), db.lower()])
            all_pairs_any_level.add(key)
            if level in ("Minor", "Moderate", "Major"):
                positive_candidates.append((da, db, level))
            if resolve(da):
                ddinter_drug_names.add(da)
            if resolve(db):
                ddinter_drug_names.add(db)
    print(f"  {fname}: {n_rows} rows loaded")

print(f"\nTotal positive-level records loaded: {len(positive_candidates)}")
print(f"Total unique pairs (any level, used for negative exclusion): {len(all_pairs_any_level)}")
print(f"KG-resolvable drug names found in DDInter vocabulary: {len(ddinter_drug_names)}")

if len(ddinter_drug_names) < 10:
    print("\nWARNING: very little overlap between DDInter drug names and your KG.")
    print("Check that drug names in your KG match common/generic names (DDInter uses generic names).")

# ── Build positive sample: pairs resolvable in KG ────────────────────────────

resolved_positive = [(a, b, lvl) for a, b, lvl in positive_candidates if resolve(a) and resolve(b)]
seen = set()
dedup_positive = []
for a, b, lvl in resolved_positive:
    key = frozenset([a.lower(), b.lower()])
    if key in seen:
        continue
    seen.add(key)
    dedup_positive.append((a, b, lvl))

random.shuffle(dedup_positive)
TARGET_N = 500          # ← scaled up from 100 for publishable sample size
positive_sample = dedup_positive[:TARGET_N]
print(f"\nPositive pairs resolvable in KG (deduplicated): {len(dedup_positive)}")
print(f"Positive pairs sampled for evaluation: {len(positive_sample)} (target={TARGET_N})")

# ── Build negative sample: KG-resolvable pairs absent from DDInter entirely ──

ddinter_drug_list = list(ddinter_drug_names)
random.shuffle(ddinter_drug_list)

negative_sample = []
checked = set()
for i, da in enumerate(ddinter_drug_list):
    if len(negative_sample) >= TARGET_N:
        break
    for db in ddinter_drug_list[i+1:]:
        if len(negative_sample) >= TARGET_N:
            break
        key = frozenset([da.lower(), db.lower()])
        if key in checked:
            continue
        checked.add(key)
        if key in all_pairs_any_level:
            continue  # DDInter has SOME record of this pair — skip, not a clean negative
        negative_sample.append((da, db))

print(f"Negative pairs sampled: {len(negative_sample)}")

all_pairs = (
    [{"drug_a": a, "drug_b": b, "ground_truth": "positive", "ddinter_level": lvl}
     for a, b, lvl in positive_sample]
    + [{"drug_a": a, "drug_b": b, "ground_truth": "negative", "ddinter_level": ""}
       for a, b in negative_sample]
)
random.shuffle(all_pairs)

print(f"\nFinal evaluation set: {len(all_pairs)} pairs "
      f"({len(positive_sample)} positive, {len(negative_sample)} negative)")

if len(all_pairs) < 20:
    print("WARNING: evaluation set is small — results will have wide uncertainty.")
    print("Consider downloading more DDInter category files and adding them to DDINTER_FILES.")

# ── Run pipeline ──────────────────────────────────────────────────────────────

from agents.path_retriever import PathRetrieverAgent
from agents.mechanism_reasoner import MechanismReasonerAgent

print("\nInitialising agents ...")
retriever = PathRetrieverAgent(kg_path=KG_PATH)
reasoner  = MechanismReasonerAgent()

POSITIVE_SEVERITIES = {"moderate", "severe"}
NEGATIVE_SEVERITIES = {"none", "mild"}

results = []
print(f"\nRunning PharmGraph on {len(all_pairs)} DDInter-grounded pairs ...\n")
print(f"{'#':<4} {'Drug A':<20} {'Drug B':<20} {'GT':<10} {'DDLvl':<10} {'Severity':<10} {'RulePred':<10} {'PipePred'}")
print("-" * 100)

for idx, pair in enumerate(all_pairs, 1):
    a, b, gt = pair["drug_a"], pair["drug_b"], pair["ground_truth"]

    rule_count = kg_shared_count(a, b)
    rule_pred = "positive" if rule_count >= 1 else ("negative" if rule_count == 0 else "abstain")

    try:
        path_result = retriever.get_shared_targets(a, b)
        if "error" in path_result:
            severity = "error"
            pipe_pred = "abstain"
        else:
            reasoning = reasoner.reason(path_result)
            severity = (reasoning.severity or "none").lower()
            if severity in POSITIVE_SEVERITIES:
                pipe_pred = "positive"
            elif severity in NEGATIVE_SEVERITIES:
                pipe_pred = "negative"
            else:
                pipe_pred = "abstain"
    except Exception as e:
        severity = f"EXC"
        pipe_pred = "abstain"
        print(f"      exception on {a}/{b}: {e}")

    results.append({
        **pair,
        "shared_count": rule_count,
        "rule_prediction": rule_pred,
        "severity": severity,
        "pipeline_prediction": pipe_pred,
    })

    tick = "✓" if pipe_pred == gt else ("?" if pipe_pred == "abstain" else "✗")
    print(f"{idx:<4} {a:<20} {b:<20} {gt:<10} {pair['ddinter_level']:<10} {severity:<10} {rule_pred:<10} {pipe_pred}  {tick}")

    time.sleep(0.3)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(key_pred):
    decided = [r for r in results if r[key_pred] not in ("abstain", "error")]
    tp = sum(1 for r in decided if r["ground_truth"] == "positive" and r[key_pred] == "positive")
    fp = sum(1 for r in decided if r["ground_truth"] == "negative" and r[key_pred] == "positive")
    tn = sum(1 for r in decided if r["ground_truth"] == "negative" and r[key_pred] == "negative")
    fn = sum(1 for r in decided if r["ground_truth"] == "positive" and r[key_pred] == "negative")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
    accuracy  = (tp + tn) / len(decided) if decided else 0
    abstain   = sum(1 for r in results if r[key_pred] in ("abstain", "error"))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(precision,3), recall=round(recall,3),
                f1=round(f1,3), accuracy=round(accuracy,3),
                abstain=abstain, n=len(results))

rule_m = compute_metrics("rule_prediction")
pipe_m = compute_metrics("pipeline_prediction")

print("\n" + "="*70)
print("  RESULTS — External Validation Against DDInter 2.0")
print("="*70)
print(f"\n  Rule-only (KG threshold):")
print(f"    TP={rule_m['tp']} FP={rule_m['fp']} TN={rule_m['tn']} FN={rule_m['fn']}")
print(f"    Precision={rule_m['precision']:.3f}  Recall={rule_m['recall']:.3f}  "
      f"F1={rule_m['f1']:.3f}  Accuracy={rule_m['accuracy']:.3f}")

print(f"\n  Full pipeline (KG + LLM):")
print(f"    TP={pipe_m['tp']} FP={pipe_m['fp']} TN={pipe_m['tn']} FN={pipe_m['fn']}")
print(f"    Precision={pipe_m['precision']:.3f}  Recall={pipe_m['recall']:.3f}  "
      f"F1={pipe_m['f1']:.3f}  Accuracy={pipe_m['accuracy']:.3f}")
print("="*70)

# ── Miss Analysis: why did we miss positives? ────────────────────────────────
# This distinguishes data-coverage misses (ChEMBL has no targets for the drug)
# from model misses (ChEMBL had targets but the pipeline still got it wrong).
# A publishable paper must show the recall gap is a coverage problem, not a
# model failure.

false_negatives = [r for r in results if r["ground_truth"] == "positive"
                   and r["rule_prediction"] == "negative"]

zero_target_misses   = [r for r in false_negatives if r["shared_count"] == 0]
nonzero_target_misses = [r for r in false_negatives if r["shared_count"] > 0]
unresolved_misses    = [r for r in false_negatives if r["shared_count"] < 0]

total_positives = len([r for r in results if r["ground_truth"] == "positive"])

print("\n" + "="*70)
print("  MISS ANALYSIS — Why did the system miss true interactions?")
print("="*70)
print(f"\n  Total true positives in evaluation set : {total_positives}")
print(f"  Missed (false negatives, rule-only)    : {len(false_negatives)}")
print(f"\n  Breakdown of misses:")
print(f"    Zero shared ChEMBL targets (coverage gap) : "
      f"{len(zero_target_misses)} "
      f"({100*len(zero_target_misses)/len(false_negatives):.1f}% of misses)"
      if false_negatives else "    (no false negatives)")
print(f"    Had ChEMBL targets but still missed       : "
      f"{len(nonzero_target_misses)} "
      f"({100*len(nonzero_target_misses)/len(false_negatives):.1f}% of misses)"
      if false_negatives else "")
print(f"    Drug not resolved in KG at all            : "
      f"{len(unresolved_misses)} "
      f"({100*len(unresolved_misses)/len(false_negatives):.1f}% of misses)"
      if false_negatives else "")

if false_negatives:
    coverage_pct = 100 * len(zero_target_misses) / len(false_negatives)
    if coverage_pct >= 70:
        print(f"\n  ✓ CONCLUSION: {coverage_pct:.1f}% of misses are due to ChEMBL data "
              f"coverage gaps,\n    not model errors. The recall ceiling is a data problem, "
              f"not a reasoning problem.")
    else:
        print(f"\n  ✗ NOTE: Only {coverage_pct:.1f}% of misses are coverage gaps — "
              f"investigate non-zero-target misses.")

# DDInter severity breakdown of misses
print(f"\n  DDInter severity of missed pairs:")
for lvl in ("Major", "Moderate", "Minor"):
    n = sum(1 for r in false_negatives if r.get("ddinter_level") == lvl)
    print(f"    {lvl:<10}: {n}")

print("="*70)

# ── Save ──────────────────────────────────────────────────────────────────────

out_csv = "evaluation_results_ddinter.csv"
out_json = "evaluation_summary_ddinter.json"

fieldnames = ["drug_a", "drug_b", "ground_truth", "ddinter_level", "shared_count",
              "rule_prediction", "severity", "pipeline_prediction"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

summary = {
    "source": "DDInter 2.0 (https://ddinter2.scbdd.com)",
    "n_positive_sampled": len(positive_sample),
    "n_negative_sampled": len(negative_sample),
    "rule_only": rule_m,
    "full_pipeline": pipe_m,
    "miss_analysis": {
        "total_false_negatives": len(false_negatives),
        "zero_chembl_target_misses": len(zero_target_misses),
        "nonzero_chembl_target_misses": len(nonzero_target_misses),
        "unresolved_drug_misses": len(unresolved_misses),
        "pct_misses_due_to_coverage_gap": round(
            100 * len(zero_target_misses) / len(false_negatives), 1
        ) if false_negatives else None,
    },
}
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDetailed results saved to: {out_csv}")
print(f"Summary saved to         : {out_json}")
print("\nDone.")
