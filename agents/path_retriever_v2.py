"""
Agent 1 v2: Path Retriever with dual evidence (ChEMBL shared targets + DDInter
known-interaction edges) and full provenance tracking.

Does not modify or import the original PathRetrieverAgent — it is a
self-contained class so the original pipeline, and its 125 passing tests,
are completely unaffected. Save this file as agents/path_retriever_v2.py.
"""

import pickle
from datetime import datetime

CHEMBL_VERSION = "ChEMBL 37"


class PathRetrieverAgentV2:
    """
    Loads the MERGED knowledge graph (ChEMBL drug-target edges + DDInter
    known_interaction drug-drug edges) and returns combined evidence for a
    drug pair, with full provenance: database versions, retrieval date,
    normalized drug names, DDInter drug IDs, and ChEMBL target IDs.
    """

    def __init__(self, kg_path="data/drug_kg_full_merged.pkl"):
        print("Loading merged knowledge graph (ChEMBL + DDInter)...")
        with open(kg_path, "rb") as f:
            self.G = pickle.load(f)

        self.name_to_id = {}
        self.id_to_display_name = {}
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == "drug" and data.get("name"):
                self.name_to_id[data["name"].upper()] = node
                self.id_to_display_name[node] = data["name"]

        print(f"  Ready. {len(self.name_to_id)} drugs indexed.")

    def find_drug_id(self, drug_name):
        return self.name_to_id.get(drug_name.upper())

    def get_evidence(self, drug_a_name, drug_b_name):
        """
        Returns combined evidence for a drug pair with explicit source
        attribution and provenance. Structure:
          {
            "drug_a": <input name>, "drug_b": <input name>,
            "drug_a_normalized": <ChEMBL node name>, "drug_b_normalized": ...,
            "drug_a_id": <ChEMBL ID>, "drug_b_id": <ChEMBL ID>,
            "shared_targets": [ {target_id, target_name, target_type,
                                  drug_a_activity, drug_b_activity}, ... ],
            "ddinter_interaction": {
                "severity": "major"/"moderate"/"minor",
                "source_db": "DDInter", "source_version": "DDInter 2.0",
                "retrieved_date": "YYYY-MM-DD",
                "record_id_a": "DDInterNNN", "record_id_b": "DDInterNNN",
                "name_a": <DDInter's own spelling>, "name_b": ...,
            } or None,
            "source": "both" | "chembl_only" | "ddinter_only" | "none",
            "provenance": {
                "chembl_version": "ChEMBL 37",
                "query_timestamp": ISO datetime of this lookup,
            },
          }
        """
        id_a = self.find_drug_id(drug_a_name)
        id_b = self.find_drug_id(drug_b_name)

        if not id_a:
            return {"error": f"Drug not found in graph: {drug_a_name}"}
        if not id_b:
            return {"error": f"Drug not found in graph: {drug_b_name}"}

        # IMPORTANT: filter successors to node_type == "target" only.
        # The merged graph also has drug->drug known_interaction edges, so an
        # unfiltered successors() call would wrongly include other drugs as
        # "shared targets" — this filter keeps that computation correct.
        targets_a = {n for n in self.G.successors(id_a)
                     if self.G.nodes[n].get("node_type") == "target"}
        targets_b = {n for n in self.G.successors(id_b)
                     if self.G.nodes[n].get("node_type") == "target"}
        shared = targets_a & targets_b

        shared_targets = []
        for target_id in shared:
            target_data = self.G.nodes[target_id]
            edge_a = self.G.edges[id_a, target_id]
            edge_b = self.G.edges[id_b, target_id]
            shared_targets.append({
                "target_id": target_id,   # ChEMBL target identifier, e.g. CHEMBL2094253
                "target_name": target_data.get("name", target_id),
                "target_type": target_data.get("target_type", "unknown"),
                "drug_a_activity": f"{edge_a.get('activity_type')} = {edge_a.get('activity_value')} {edge_a.get('units','')}",
                "drug_b_activity": f"{edge_b.get('activity_type')} = {edge_b.get('activity_value')} {edge_b.get('units','')}",
            })

        # Check for a direct DDInter known_interaction edge (either direction)
        ddinter_interaction = None
        edge_data = None
        if self.G.has_edge(id_a, id_b) and self.G.edges[id_a, id_b].get("edge_type") == "known_interaction":
            edge_data = self.G.edges[id_a, id_b]
        elif self.G.has_edge(id_b, id_a) and self.G.edges[id_b, id_a].get("edge_type") == "known_interaction":
            edge_data = self.G.edges[id_b, id_a]

        if edge_data is not None:
            ddinter_interaction = {
                "severity": edge_data.get("severity", "unknown"),
                "source_db": edge_data.get("source", "DDInter"),
                "source_version": edge_data.get("source_version", "DDInter 2.0"),
                "retrieved_date": edge_data.get("retrieved_date", "unknown"),
                "record_id_a": edge_data.get("ddinter_record_id_a", "unknown"),
                "record_id_b": edge_data.get("ddinter_record_id_b", "unknown"),
                "name_a": edge_data.get("ddinter_name_a", drug_a_name),
                "name_b": edge_data.get("ddinter_name_b", drug_b_name),
            }

        has_chembl = len(shared_targets) > 0
        has_ddinter = ddinter_interaction is not None
        if has_chembl and has_ddinter:
            source = "both"
        elif has_chembl:
            source = "chembl_only"
        elif has_ddinter:
            source = "ddinter_only"
        else:
            source = "none"

        return {
            "drug_a": drug_a_name,
            "drug_b": drug_b_name,
            "drug_a_normalized": self.id_to_display_name.get(id_a, drug_a_name),
            "drug_b_normalized": self.id_to_display_name.get(id_b, drug_b_name),
            "drug_a_id": id_a,
            "drug_b_id": id_b,
            "drug_a_target_count": len(targets_a),
            "drug_b_target_count": len(targets_b),
            "shared_targets": shared_targets,
            "ddinter_interaction": ddinter_interaction,
            "source": source,
            "provenance": {
                "chembl_version": CHEMBL_VERSION,
                "query_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        }
