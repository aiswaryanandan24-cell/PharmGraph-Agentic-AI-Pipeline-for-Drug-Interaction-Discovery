"""
Tests for PathRetrieverAgent — knowledge graph queries.
These tests require the knowledge graph file (data/drug_kg.pkl) to be present.
They do NOT call the OpenAI API.
Run with: pytest tests/test_path_retriever.py -v
"""

import sys
import os
import pickle
import pytest
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixture: load agent once for the whole test session ──────────────────────

@pytest.fixture(scope="session")
def agent():
    from agents.path_retriever import PathRetrieverAgent
    return PathRetrieverAgent()


# ── name_to_id index tests ────────────────────────────────────────────────────

class TestNameIndex:
    def test_index_is_nonempty(self, agent):
        assert len(agent.name_to_id) > 0

    def test_index_contains_warfarin(self, agent):
        assert "WARFARIN" in agent.name_to_id

    def test_index_contains_aspirin(self, agent):
        assert "ASPIRIN" in agent.name_to_id

    def test_index_contains_atorvastatin(self, agent):
        assert "ATORVASTATIN" in agent.name_to_id

    def test_find_drug_id_returns_id_for_known_drug(self, agent):
        drug_id = agent.find_drug_id("WARFARIN")
        assert drug_id is not None

    def test_find_drug_id_case_insensitive(self, agent):
        id_upper = agent.find_drug_id("WARFARIN")
        id_lower = agent.find_drug_id("warfarin")
        assert id_upper == id_lower

    def test_find_drug_id_returns_none_for_unknown(self, agent):
        assert agent.find_drug_id("NOTADRUG_XYZ_123") is None


# ── get_shared_targets: error handling ───────────────────────────────────────

class TestGetSharedTargetsErrors:
    def test_unknown_drug_a_returns_error(self, agent):
        result = agent.get_shared_targets("NOTADRUG_XYZ_123", "WARFARIN")
        assert "error" in result

    def test_unknown_drug_b_returns_error(self, agent):
        result = agent.get_shared_targets("WARFARIN", "NOTADRUG_XYZ_123")
        assert "error" in result

    def test_both_unknown_returns_error(self, agent):
        result = agent.get_shared_targets("FAKE_A", "FAKE_B")
        assert "error" in result

    def test_error_message_mentions_drug_name(self, agent):
        result = agent.get_shared_targets("NOTADRUG_XYZ_123", "WARFARIN")
        assert "NOTADRUG_XYZ_123" in result["error"]


# ── get_shared_targets: result structure ──────────────────────────────────────

class TestGetSharedTargetsStructure:
    def test_result_has_drug_a_key(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert "drug_a" in result

    def test_result_has_drug_b_key(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert "drug_b" in result

    def test_result_has_shared_targets_key(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert "shared_targets" in result

    def test_shared_targets_is_list(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert isinstance(result["shared_targets"], list)

    def test_result_preserves_drug_names(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert result["drug_a"] == "WARFARIN"
        assert result["drug_b"] == "ASPIRIN"

    def test_target_entries_have_target_name(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        for target in result["shared_targets"]:
            assert "target_name" in target

    def test_target_entries_have_target_type(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        for target in result["shared_targets"]:
            assert "target_type" in target

    def test_result_has_target_count_fields(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert "drug_a_target_count" in result
        assert "drug_b_target_count" in result

    def test_target_counts_are_nonnegative_integers(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert result["drug_a_target_count"] >= 0
        assert result["drug_b_target_count"] >= 0


# ── get_shared_targets: pharmacological correctness ──────────────────────────

class TestGetSharedTargetsPharmacology:
    def test_warfarin_has_at_least_one_target(self, agent):
        result = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        assert result["drug_a_target_count"] >= 1

    def test_no_shared_targets_returns_empty_list_not_error(self, agent):
        # METFORMIN + LEVOTHYROXINE should have no shared targets (demo case)
        result = agent.get_shared_targets("METFORMIN", "LEVOTHYROXINE")
        assert "error" not in result
        assert "shared_targets" in result
        assert isinstance(result["shared_targets"], list)

    def test_drug_with_itself_shares_all_targets(self, agent):
        # A drug vs itself should share 100% of its targets
        result = agent.get_shared_targets("WARFARIN", "WARFARIN")
        assert "error" not in result
        # shared count should equal individual target count
        assert len(result["shared_targets"]) == result["drug_a_target_count"]

    def test_symmetric_query_same_shared_count(self, agent):
        # shared(A, B) == shared(B, A)
        ab = agent.get_shared_targets("WARFARIN", "ASPIRIN")
        ba = agent.get_shared_targets("ASPIRIN", "WARFARIN")
        if "error" not in ab and "error" not in ba:
            assert len(ab["shared_targets"]) == len(ba["shared_targets"])


# ── Graph integrity ───────────────────────────────────────────────────────────

class TestGraphIntegrity:
    def test_graph_is_loaded(self, agent):
        assert agent.G is not None

    def test_graph_has_nodes(self, agent):
        assert agent.G.number_of_nodes() > 0

    def test_graph_has_edges(self, agent):
        assert agent.G.number_of_edges() > 0

    def test_all_index_keys_are_uppercase(self, agent):
        for name in agent.name_to_id:
            assert name == name.upper(), f"name_to_id has lowercase key: {name!r}"
