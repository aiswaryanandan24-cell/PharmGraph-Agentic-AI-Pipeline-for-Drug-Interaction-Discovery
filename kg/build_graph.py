import pandas as pd
import networkx as nx
import pickle
import os

print("Loading CSVs...")
drugs_df = pd.read_csv("data/drugs.csv")
interactions_df = pd.read_csv("data/drug_target_interactions.csv")

print(f"  {len(drugs_df)} drugs loaded")
print(f"  {len(interactions_df)} interactions loaded")

# Build a directed graph
G = nx.DiGraph()

# Add drug nodes
for _, row in drugs_df.iterrows():
    G.add_node(row["chembl_id"],
               name=row["pref_name"],
               node_type="drug",
               max_phase=row["max_phase"])

print(f"  Added {G.number_of_nodes()} drug nodes")

# Add target nodes + edges (drug → target)
for _, row in interactions_df.iterrows():
    drug_id = row["drug_chembl_id"]
    target_id = row["target_chembl_id"]

    # Add target node if not already there
    if not G.has_node(target_id):
        G.add_node(target_id,
                   name=row["target_name"],
                   node_type="target",
                   target_type=row["target_type"])

    # Add edge: drug → target
    G.add_edge(drug_id, target_id,
               activity_type=row["activity_type"],
               activity_value=row["activity_value"],
               units=row["activity_units"])

print(f"  Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Save the graph
os.makedirs("data", exist_ok=True)
with open("data/drug_kg.pkl", "wb") as f:
    pickle.dump(G, f)

print("\nKnowledge graph saved to data/drug_kg.pkl")

# Quick check: find shared targets between two random drugs
print("\n--- Quick test: finding shared targets ---")
drug_list = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]

if len(drug_list) >= 2:
    drug_a = drug_list[0]
    drug_b = drug_list[1]

    targets_a = set(G.successors(drug_a))
    targets_b = set(G.successors(drug_b))
    shared = targets_a & targets_b

    name_a = G.nodes[drug_a].get("name", drug_a)
    name_b = G.nodes[drug_b].get("name", drug_b)

    print(f"  Drug A: {name_a}")
    print(f"  Drug B: {name_b}")
    print(f"  Shared targets: {len(shared)}")
    if shared:
        for t in list(shared)[:3]:
            print(f"    → {G.nodes[t].get('name', t)}")

print("\nDone! Knowledge graph is ready.")
