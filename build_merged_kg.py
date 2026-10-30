"""
Build a merged knowledge graph: original ChEMBL drug-target graph PLUS
direct drug-drug "known_interaction" edges sourced from DDInter 2.0.

Does NOT touch the original drug_kg.pkl — writes a new drug_kg_merged.pkl.

To keep evaluation honest, DDInter's resolvable interaction pairs are split
80/20: 80% are merged into the graph as known_interaction edges (the system
can use these at inference time), 20% are held out and NEVER added to the
graph, so they can be used afterward as a clean, unseen test set.

Run from project root (same folder as data/drug_kg.pkl):
    python build_merged_kg.py
"""

import os, csv, json, pickle, random
from datetime import date, datetime
import networkx as nx

random.seed(42)
BUILD_DATE = date.today().isoformat()

KG_IN  = "data/drug_kg.pkl"
KG_OUT = "data/drug_kg_merged.pkl"

DDINTER_DIR = "ddinter"
DDINTER_FILES = [
    "ddinter_downloads_code_A.csv",
    "ddinter_downloads_code_B.csv",
    "ddinter_downloads_code_D.csv",
    "ddinter_downloads_code_H.csv",
    "ddinter_downloads_code_L.csv",
    "ddinter_downloads_code_P.csv",
    "ddinter_downloads_code_R.csv",
    "ddinter_downloads_code_V.csv",
]

SEVERITY_RANK = {"Major": 3, "Moderate": 2, "Minor": 1}  # higher = more severe
TRAIN_FRACTION = 0.8

# ── Load graph ────────────────────────────────────────────────────────────────

print(f"Loading {KG_IN} ...")
with open(KG_IN, "rb") as f:
    G = pickle.load(f)

print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, directed={G.is_directed()}")

name_to_id = {}
for node, d in G.nodes(data=True):
    if d.get("node_type") == "drug" and d.get("name"):
        name_to_id[d["name"].upper()] = node

print(f"  {len(name_to_id)} drug nodes indexed by name")

# Manually curated, non-exhaustive INN/common-name synonym map. DDInter and
# ChEMBL sometimes name the same drug differently (e.g. DDInter's
# "Acetylsalicylic acid" vs ChEMBL's "Aspirin"). Without this, an exact-string
# match silently drops the pair even when both databases record the same
# drug. This list covers well-established US-adopted-name <-> INN pairs; it
# is not comprehensive — extend it as more mismatches are found.
ALIASES = {
    "ACETYLSALICYLIC ACID": "ASPIRIN", "ASPIRIN": "ACETYLSALICYLIC ACID",
    "ACETAMINOPHEN": "PARACETAMOL", "PARACETAMOL": "ACETAMINOPHEN",
    "EPINEPHRINE": "ADRENALINE", "ADRENALINE": "EPINEPHRINE",
    "NOREPINEPHRINE": "NORADRENALINE", "NORADRENALINE": "NOREPINEPHRINE",
    "ALBUTEROL": "SALBUTAMOL", "SALBUTAMOL": "ALBUTEROL",
    "FUROSEMIDE": "FRUSEMIDE", "FRUSEMIDE": "FUROSEMIDE",
    "MEPERIDINE": "PETHIDINE", "PETHIDINE": "MEPERIDINE",
    "ACETYLCYSTEINE": "N-ACETYLCYSTEINE", "N-ACETYLCYSTEINE": "ACETYLCYSTEINE",
    "VITAMIN C": "ASCORBIC ACID", "ASCORBIC ACID": "VITAMIN C",
    "VITAMIN B12": "CYANOCOBALAMIN", "CYANOCOBALAMIN": "VITAMIN B12",
}

alias_hits = 0

def resolve(name):
    global alias_hits
    key = name.strip().upper()
    direct = name_to_id.get(key)
    if direct:
        return direct
    alias = ALIASES.get(key)
    if alias:
        hit = name_to_id.get(alias)
        if hit:
            alias_hits += 1
        return hit
    return None

# ── Load all DDInter records ─────────────────────────────────────────────────

print(f"\nLoading DDInter files from {DDINTER_DIR}/ ...")

all_pairs_any_level = {}   # frozenset(lower names) -> best level seen (any level incl. Unknown)
raw_positive = {}          # frozenset(lower names) -> (drug_a_name, drug_b_name, best_level, ddinter_id_a, ddinter_id_b)
original_casing = {}       # lowercase name -> first-seen properly-cased spelling

for fname in DDINTER_FILES:
    path = os.path.join(DDINTER_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping")
        continue
    n = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            da = (row.get("Drug_A") or "").strip()
            db = (row.get("Drug_B") or "").strip()
            level = (row.get("Level") or "").strip()
            id_a_ddinter = (row.get("DDInterID_A") or "").strip()
            id_b_ddinter = (row.get("DDInterID_B") or "").strip()
            if not da or not db:
                continue
            n += 1
            original_casing.setdefault(da.lower(), da)
            original_casing.setdefault(db.lower(), db)
            key = frozenset([da.lower(), db.lower()])

            # track best (highest-rank) level seen for this pair, any level
            prev = all_pairs_any_level.get(key)
            if prev is None or SEVERITY_RANK.get(level, 0) > SEVERITY_RANK.get(prev, 0):
                all_pairs_any_level[key] = level

            if level in SEVERITY_RANK:
                prev_pos = raw_positive.get(key)
                if prev_pos is None or SEVERITY_RANK[level] > SEVERITY_RANK[prev_pos[2]]:
                    raw_positive[key] = (da, db, level, id_a_ddinter, id_b_ddinter)
    print(f"  {fname}: {n} rows")

print(f"\nTotal unique pairs seen (any level): {len(all_pairs_any_level)}")
print(f"Total unique positive pairs (Minor/Moderate/Major): {len(raw_positive)}")

# ── Resolve against KG vocabulary ────────────────────────────────────────────

resolved_positive = [
    (a, b, lvl, ida, idb) for (a, b, lvl, ida, idb) in raw_positive.values()
    if resolve(a) and resolve(b)
]
print(f"Positive pairs resolvable in our KG: {len(resolved_positive)}")
print(f"  (of which {alias_hits} drug-name lookups needed the alias map to resolve)")

random.shuffle(resolved_positive)
split_idx = int(len(resolved_positive) * TRAIN_FRACTION)
train_pairs = resolved_positive[:split_idx]
holdout_pairs = resolved_positive[split_idx:]

print(f"  Train (merged into graph): {len(train_pairs)}")
print(f"  Holdout (never merged, for evaluation): {len(holdout_pairs)}")

# ── Merge train pairs into graph as known_interaction edges ─────────────────

added = 0
for a, b, lvl, ddinter_id_a, ddinter_id_b in train_pairs:
    id_a, id_b = resolve(a), resolve(b)
    attrs_ab = {"edge_type": "known_interaction", "severity": lvl.lower(), "source": "DDInter",
                "source_version": "DDInter 2.0", "retrieved_date": BUILD_DATE,
                "ddinter_record_id_a": ddinter_id_a, "ddinter_record_id_b": ddinter_id_b,
                "ddinter_name_a": a, "ddinter_name_b": b}
    attrs_ba = {**attrs_ab, "ddinter_record_id_a": ddinter_id_b, "ddinter_record_id_b": ddinter_id_a,
                "ddinter_name_a": b, "ddinter_name_b": a}
    if not G.has_edge(id_a, id_b):
        G.add_edge(id_a, id_b, **attrs_ab)
        added += 1
    if not G.has_edge(id_b, id_a):
        G.add_edge(id_b, id_a, **attrs_ba)
        added += 1

print(f"\nAdded {added} new directed known_interaction edges "
      f"({len(train_pairs)} pairs x up to 2 directions)")
print(f"Merged graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Save merged graph ─────────────────────────────────────────────────────────

with open(KG_OUT, "wb") as f:
    pickle.dump(G, f)
print(f"\nSaved merged graph to {KG_OUT}")

# ── Save holdout set + full exclusion set for honest evaluation later ───────

with open("ddinter_holdout_positive.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["drug_a", "drug_b", "level", "ddinter_id_a", "ddinter_id_b"])
    for a, b, lvl, ida, idb in holdout_pairs:
        w.writerow([a, b, lvl, ida, idb])
print(f"Saved {len(holdout_pairs)} holdout positive pairs to ddinter_holdout_positive.csv")

# all pairs ever seen (any level, across ALL 8 files) — used so the evaluation
# script never accidentally samples a "negative" that DDInter actually has
# some record of, even an unknown-severity one
exclusion_list = [sorted(list(k)) for k in all_pairs_any_level.keys()]
with open("ddinter_all_pairs_any_level.json", "w") as f:
    json.dump(exclusion_list, f)
print(f"Saved {len(exclusion_list)} pairs (any level) to ddinter_all_pairs_any_level.json")

# KG-resolvable drug name vocabulary seen anywhere in DDInter (for negative sampling pool)
lower_names_seen = {n for pair in all_pairs_any_level.keys() for n in pair}
vocab = sorted({original_casing.get(n, n) for n in lower_names_seen if resolve(n)})
with open("ddinter_kg_vocab.json", "w") as f:
    json.dump(vocab, f)
print(f"Saved {len(vocab)} KG-resolvable drug names to ddinter_kg_vocab.json")

print("\nDone. Original data/drug_kg.pkl was not modified.")
