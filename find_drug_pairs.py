"""
Scan the knowledge graph to find 100 drug pairs that actually share targets.
Saves them to data/eval_pairs.csv for the evaluation script.
"""
import sys, pickle, csv, random
sys.path.append(".")

print("Loading graph...")
with open("data/drug_kg.pkl", "rb") as f:
    G = pickle.load(f)

drug_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
print(f"  {len(drug_nodes)} drugs in graph")

# Find pairs with shared targets
interacting_pairs = []
non_interacting_pairs = []

checked = 0
for i, (drug_a, data_a) in enumerate(drug_nodes):
    targets_a = set(G.successors(drug_a))
    if not targets_a:
        continue
    for drug_b, data_b in drug_nodes[i+1:]:
        targets_b = set(G.successors(drug_b))
        shared = targets_a & targets_b
        name_a = data_a.get("name", drug_a)
        name_b = data_b.get("name", drug_b)
        if len(shared) >= 3:
            interacting_pairs.append((name_a, name_b, len(shared), True))
        elif len(shared) == 0 and targets_b:
            non_interacting_pairs.append((name_a, name_b, 0, False))
    checked += 1
    if checked % 100 == 0:
        print(f"  Checked {checked} drugs... found {len(interacting_pairs)} interacting, {len(non_interacting_pairs)} non-interacting")
    if len(interacting_pairs) >= 75 and len(non_interacting_pairs) >= 75:
        break

print(f"\nFound {len(interacting_pairs)} interacting pairs")
print(f"Found {len(non_interacting_pairs)} non-interacting pairs")

# Take 70 interacting + 30 non-interacting = 100 pairs
random.seed(42)
selected_interacting    = random.sample(interacting_pairs,    min(70, len(interacting_pairs)))
selected_non_interacting = random.sample(non_interacting_pairs, min(30, len(non_interacting_pairs)))
all_pairs = selected_interacting + selected_non_interacting
random.shuffle(all_pairs)

with open("data/eval_pairs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["drug_a", "drug_b", "shared_targets", "known_interaction"])
    writer.writerows(all_pairs)

print(f"\nSaved {len(all_pairs)} pairs to data/eval_pairs.csv")
print("Top 10 pairs:")
for row in all_pairs[:10]:
    print(f"  {row[0]} + {row[1]}  →  {row[2]} shared targets  |  interaction={row[3]}")
