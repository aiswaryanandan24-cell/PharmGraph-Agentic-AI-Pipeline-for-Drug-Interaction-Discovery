import pickle
import networkx as nx

class PathRetrieverAgent:
    """
    Agent 1: Path Retriever
    Given two drug names, finds all protein paths connecting them
    in the knowledge graph.
    """

    def __init__(self, kg_path="data/drug_kg.pkl"):
        print("Loading knowledge graph...")
        with open(kg_path, "rb") as f:
            self.G = pickle.load(f)

        # Build a name → chembl_id lookup
        self.name_to_id = {}
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "drug" and data.get("name"):
                self.name_to_id[data["name"].upper()] = node

        print(f"  Ready. {len(self.name_to_id)} drugs indexed.")

    def find_drug_id(self, drug_name):
        return self.name_to_id.get(drug_name.upper())

    def get_shared_targets(self, drug_a_name, drug_b_name):
        id_a = self.find_drug_id(drug_a_name)
        id_b = self.find_drug_id(drug_b_name)

        if not id_a:
            return {"error": f"Drug not found in graph: {drug_a_name}"}
        if not id_b:
            return {"error": f"Drug not found in graph: {drug_b_name}"}

        targets_a = set(self.G.successors(id_a))
        targets_b = set(self.G.successors(id_b))
        shared = targets_a & targets_b

        result = {
            "drug_a": drug_a_name,
            "drug_b": drug_b_name,
            "drug_a_id": id_a,
            "drug_b_id": id_b,
            "drug_a_target_count": len(targets_a),
            "drug_b_target_count": len(targets_b),
            "shared_targets": []
        }

        for target_id in shared:
            target_data = self.G.nodes[target_id]
            edge_a = self.G.edges[id_a, target_id]
            edge_b = self.G.edges[id_b, target_id]
            result["shared_targets"].append({
                "target_id": target_id,
                "target_name": target_data.get("name", target_id),
                "target_type": target_data.get("target_type", "unknown"),
                "drug_a_activity": f"{edge_a.get('activity_type')} = {edge_a.get('activity_value')} {edge_a.get('units','')}",
                "drug_b_activity": f"{edge_b.get('activity_type')} = {edge_b.get('activity_value')} {edge_b.get('units','')}",
            })

        return result

    def find_pairs_with_shared_targets(self, min_shared=2, limit=5):
        """Scan graph to find drug pairs that actually share targets."""
        print(f"\nSearching for drug pairs with >={min_shared} shared targets...")
        drug_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "drug"]

        found = []
        checked = 0
        for i, drug_a in enumerate(drug_nodes):
            targets_a = set(self.G.successors(drug_a))
            if not targets_a:
                continue
            for drug_b in drug_nodes[i+1:]:
                targets_b = set(self.G.successors(drug_b))
                shared = targets_a & targets_b
                if len(shared) >= min_shared:
                    name_a = self.G.nodes[drug_a].get("name", drug_a)
                    name_b = self.G.nodes[drug_b].get("name", drug_b)
                    found.append((name_a, name_b, len(shared)))
                    if len(found) >= limit:
                        return found
            checked += 1
            if checked > 200:   # stop early to keep it fast
                break
        return found


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = PathRetrieverAgent()

    # First find real pairs that share targets in our graph
    pairs = agent.find_pairs_with_shared_targets(min_shared=3, limit=5)

    print("\nDrug pairs with shared targets found:")
    for a, b, count in pairs:
        print(f"  {a} + {b}  →  {count} shared targets")

    if pairs:
        # Test the first real pair
        drug_a, drug_b, _ = pairs[0]
        print(f"\n=== Detailed test: {drug_a} + {drug_b} ===")
        result = agent.get_shared_targets(drug_a, drug_b)

        if "error" in result:
            print("Error:", result["error"])
        else:
            print(f"{drug_a} has {result['drug_a_target_count']} targets")
            print(f"{drug_b} has {result['drug_b_target_count']} targets")
            print(f"Shared: {len(result['shared_targets'])}")
            for t in result["shared_targets"][:5]:
                print(f"\n  Target: {t['target_name']}")
                print(f"    {drug_a}: {t['drug_a_activity']}")
                print(f"    {drug_b}: {t['drug_b_activity']}")
