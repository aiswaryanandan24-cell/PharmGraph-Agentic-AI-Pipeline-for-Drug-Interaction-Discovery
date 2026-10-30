"""
Tests for MechanismReasonerAgent — LLM reasoning over shared targets.
All OpenAI API calls are mocked so no API key or credits are required.
Run with: pytest tests/test_mechanism_reasoner.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.mechanism_reasoner import MechanismReasonerAgent, DDIReasoningChain, ReasoningStep


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent(monkeypatch):
    """Create agent with a fake API key so __init__ doesn't raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    return MechanismReasonerAgent()


@pytest.fixture
def path_result_with_targets():
    return {
        "drug_a": "WARFARIN",
        "drug_b": "ASPIRIN",
        "drug_a_id": "CHEMBL1",
        "drug_b_id": "CHEMBL2",
        "drug_a_target_count": 3,
        "drug_b_target_count": 5,
        "shared_targets": [
            {
                "target_id": "CHEMBL2094253",
                "target_name": "Prostaglandin G/H synthase 1",
                "target_type": "SINGLE PROTEIN",
                "drug_a_activity": "IC50 = 12.5 nM",
                "drug_b_activity": "IC50 = 8.3 nM",
            }
        ]
    }


@pytest.fixture
def path_result_no_targets():
    return {
        "drug_a": "METFORMIN",
        "drug_b": "LEVOTHYROXINE",
        "drug_a_id": "CHEMBL3",
        "drug_b_id": "CHEMBL4",
        "drug_a_target_count": 2,
        "drug_b_target_count": 3,
        "shared_targets": []
    }


def _make_fake_llm_response(text):
    """Build a mock LangChain LLM response with .content set."""
    mock_response = MagicMock()
    mock_response.content = text
    return mock_response


GOOD_LLM_RESPONSE = """\
INTERACTION_TYPE: pharmacodynamic
SEVERITY: moderate
MECHANISM: Both drugs inhibit platelet function through different mechanisms, leading to additive hemorrhagic risk.
RECOMMENDATION: Monitor closely for bleeding; avoid combination at high aspirin doses.

STEP 1:
CLAIM: WARFARIN inhibits clotting factor synthesis via Vitamin K antagonism.
EVIDENCE: Well-established anticoagulant mechanism documented in literature.

STEP 2:
CLAIM: ASPIRIN irreversibly inhibits COX-1, reducing thromboxane A2 and platelet aggregation.
EVIDENCE: Confirmed by platelet function studies.

STEP 3:
CLAIM: Combined use significantly increases bleeding risk beyond either drug alone.
EVIDENCE: Clinical studies show additive hemorrhagic effect.
"""


# ── No-targets early exit ─────────────────────────────────────────────────────

class TestNoTargetsEarlyExit:
    def test_returns_ddi_chain_when_no_targets(self, agent, path_result_no_targets):
        result = agent.reason(path_result_no_targets)
        assert isinstance(result, DDIReasoningChain)

    def test_severity_none_when_no_targets(self, agent, path_result_no_targets):
        result = agent.reason(path_result_no_targets)
        assert result.severity == "none"

    def test_interaction_type_none_when_no_targets(self, agent, path_result_no_targets):
        result = agent.reason(path_result_no_targets)
        assert result.interaction_type == "none"

    def test_steps_empty_when_no_targets(self, agent, path_result_no_targets):
        result = agent.reason(path_result_no_targets)
        assert result.steps == []

    def test_drug_names_preserved_when_no_targets(self, agent, path_result_no_targets):
        result = agent.reason(path_result_no_targets)
        assert result.drug_a == "METFORMIN"
        assert result.drug_b == "LEVOTHYROXINE"

    def test_no_llm_call_when_no_targets(self, agent, path_result_no_targets):
        # Patch at class level — Pydantic v2 blocks instance-level attribute patching
        with patch.object(type(agent.llm), "invoke") as mock_invoke:
            agent.reason(path_result_no_targets)
            mock_invoke.assert_not_called()


# ── Mocked LLM reasoning ──────────────────────────────────────────────────────
# We patch type(agent.llm) (the class) rather than agent.llm (the instance)
# because ChatOpenAI is a Pydantic v2 model whose __setattr__ rejects unknown
# fields — patching the class bypasses that restriction entirely.

class TestReasoningWithMockedLLM:
    def test_returns_ddi_chain(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert isinstance(result, DDIReasoningChain)

    def test_severity_is_moderate(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert result.severity == "moderate"

    def test_interaction_type_is_pharmacodynamic(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert result.interaction_type == "pharmacodynamic"

    def test_mechanism_summary_is_nonempty(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert len(result.mechanism_summary) > 0

    def test_clinical_recommendation_is_nonempty(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert len(result.clinical_recommendation) > 0

    def test_steps_are_list_of_reasoning_steps(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert isinstance(result.steps, list)
        for step in result.steps:
            assert isinstance(step, ReasoningStep)

    def test_steps_have_claim_and_evidence(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        for step in result.steps:
            assert len(step.claim) > 0
            assert len(step.evidence) > 0

    def test_drug_names_preserved_in_result(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)):
            result = agent.reason(path_result_with_targets)
        assert result.drug_a == "WARFARIN"
        assert result.drug_b == "ASPIRIN"

    def test_llm_is_called_once(self, agent, path_result_with_targets):
        with patch.object(type(agent.llm), "invoke", return_value=_make_fake_llm_response(GOOD_LLM_RESPONSE)) as mock:
            agent.reason(path_result_with_targets)
        mock.assert_called_once()


# ── _parse_response unit tests ────────────────────────────────────────────────

class TestParseResponse:
    def test_parse_extracts_severity(self, agent):
        chain = agent._parse_response("WARFARIN", "ASPIRIN", GOOD_LLM_RESPONSE)
        assert chain.severity == "moderate"

    def test_parse_extracts_interaction_type(self, agent):
        chain = agent._parse_response("WARFARIN", "ASPIRIN", GOOD_LLM_RESPONSE)
        assert chain.interaction_type == "pharmacodynamic"

    def test_parse_extracts_mechanism(self, agent):
        chain = agent._parse_response("WARFARIN", "ASPIRIN", GOOD_LLM_RESPONSE)
        assert "inhibit" in chain.mechanism_summary.lower()

    def test_parse_extracts_recommendation(self, agent):
        chain = agent._parse_response("WARFARIN", "ASPIRIN", GOOD_LLM_RESPONSE)
        assert len(chain.clinical_recommendation) > 0

    def test_parse_extracts_three_steps(self, agent):
        chain = agent._parse_response("WARFARIN", "ASPIRIN", GOOD_LLM_RESPONSE)
        assert len(chain.steps) == 3

    def test_parse_steps_numbered_correctly(self, agent):
        chain = agent._parse_response("WARFARIN", "ASPIRIN", GOOD_LLM_RESPONSE)
        for i, step in enumerate(chain.steps, start=1):
            assert step.step_number == i

    def test_parse_malformed_response_returns_chain_not_exception(self, agent):
        bad_response = "This is not a structured response at all."
        chain = agent._parse_response("DRUGX", "DRUGY", bad_response)
        assert isinstance(chain, DDIReasoningChain)
        assert chain.drug_a == "DRUGX"
        assert chain.drug_b == "DRUGY"

    def test_parse_severe_severity(self, agent):
        response = GOOD_LLM_RESPONSE.replace("SEVERITY: moderate", "SEVERITY: severe")
        chain = agent._parse_response("WARFARIN", "ASPIRIN", response)
        assert chain.severity == "severe"

    def test_parse_mild_severity(self, agent):
        response = GOOD_LLM_RESPONSE.replace("SEVERITY: moderate", "SEVERITY: mild")
        chain = agent._parse_response("WARFARIN", "ASPIRIN", response)
        assert chain.severity == "mild"
