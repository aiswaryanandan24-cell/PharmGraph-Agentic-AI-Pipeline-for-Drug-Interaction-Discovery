"""
Tests for pipeline decision logic, polypharmacy pair generation,
severity ordering, and abstention thresholds.
These tests are pure Python — no API calls, no knowledge graph required.
Run with: pytest tests/test_pipeline_logic.py -v
"""

import sys
import os
import pytest
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import BRAND_TO_INN, CONFIDENCE_THRESHOLD, MAX_LOOP, _resolve


# ── Abstention / confidence threshold logic ───────────────────────────────────
# These replicate the logic from the _run() decision loop in app.py

def should_proceed(confidence: int, attempts_used: int) -> str:
    """
    Mirror of the decision logic inside _run():
      - confidence >= CONFIDENCE_THRESHOLD → 'proceed'
      - all attempts used AND confidence < 30 → 'abstain'
      - all attempts used AND 30 <= confidence < threshold → 'warn'
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        return "proceed"
    if attempts_used >= MAX_LOOP:
        if confidence < 30:
            return "abstain"
        return "warn"
    return "retry"


class TestAbstentionLogic:
    def test_high_confidence_proceeds(self):
        assert should_proceed(80, 1) == "proceed"

    def test_exactly_at_threshold_proceeds(self):
        assert should_proceed(CONFIDENCE_THRESHOLD, 1) == "proceed"

    def test_just_below_threshold_retries_if_attempts_remain(self):
        assert should_proceed(CONFIDENCE_THRESHOLD - 1, 0) == "retry"

    def test_very_low_confidence_abstains_after_max_attempts(self):
        assert should_proceed(20, MAX_LOOP) == "abstain"

    def test_low_confidence_warns_after_max_attempts(self):
        assert should_proceed(45, MAX_LOOP) == "warn"

    def test_zero_confidence_abstains(self):
        assert should_proceed(0, MAX_LOOP) == "abstain"

    def test_29_abstains(self):
        assert should_proceed(29, MAX_LOOP) == "abstain"

    def test_30_warns(self):
        assert should_proceed(30, MAX_LOOP) == "warn"

    def test_59_warns_at_max_attempts(self):
        assert should_proceed(59, MAX_LOOP) == "warn"

    def test_100_always_proceeds(self):
        assert should_proceed(100, 0) == "proceed"
        assert should_proceed(100, MAX_LOOP) == "proceed"

    def test_confidence_threshold_constant_is_reasonable(self):
        assert 50 <= CONFIDENCE_THRESHOLD <= 80, \
            f"CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD} seems misconfigured"

    def test_max_loop_is_at_least_1(self):
        assert MAX_LOOP >= 1


# ── Polypharmacy pair generation ──────────────────────────────────────────────

class TestPolypharmacyPairs:
    def _all_pairs(self, drugs):
        return list(combinations(drugs, 2))

    def test_three_drugs_gives_three_pairs(self):
        drugs = ["WARFARIN", "ASPIRIN", "IBUPROFEN"]
        assert len(self._all_pairs(drugs)) == 3

    def test_four_drugs_gives_six_pairs(self):
        drugs = ["A", "B", "C", "D"]
        assert len(self._all_pairs(drugs)) == 6

    def test_five_drugs_gives_ten_pairs(self):
        drugs = ["A", "B", "C", "D", "E"]
        assert len(self._all_pairs(drugs)) == 10

    def test_eight_drugs_gives_28_pairs(self):
        drugs = [f"DRUG{i}" for i in range(8)]
        assert len(self._all_pairs(drugs)) == 28

    def test_pairs_are_unique(self):
        drugs = ["WARFARIN", "ASPIRIN", "IBUPROFEN", "METFORMIN"]
        pairs = self._all_pairs(drugs)
        pair_set = set(frozenset(p) for p in pairs)
        assert len(pair_set) == len(pairs)

    def test_pairs_are_not_self_referential(self):
        drugs = ["WARFARIN", "ASPIRIN", "METFORMIN"]
        for a, b in self._all_pairs(drugs):
            assert a != b

    def test_brand_names_in_list_get_resolved(self):
        raw_list = ["COUMADIN", "ADVIL", "GLUCOPHAGE"]
        resolved = [_resolve(d)[0] for d in raw_list]
        assert resolved == ["WARFARIN", "IBUPROFEN", "METFORMIN"]


# ── Severity ordering ─────────────────────────────────────────────────────────

class TestSeverityOrdering:
    SEV_ORDER = {
        "Severe": 0,
        "Moderate": 1,
        "Mild": 2,
        "Uncertain": 3,
        "No interaction": 4,
        "Not found": 5,
    }

    def test_severe_ranks_first(self):
        assert self.SEV_ORDER["Severe"] == 0

    def test_moderate_after_severe(self):
        assert self.SEV_ORDER["Moderate"] > self.SEV_ORDER["Severe"]

    def test_mild_after_moderate(self):
        assert self.SEV_ORDER["Mild"] > self.SEV_ORDER["Moderate"]

    def test_no_interaction_before_not_found(self):
        assert self.SEV_ORDER["No interaction"] < self.SEV_ORDER["Not found"]

    def test_sort_results_by_severity(self):
        results = [
            {"pair": ("A", "B"), "severity": "Mild"},
            {"pair": ("C", "D"), "severity": "Severe"},
            {"pair": ("E", "F"), "severity": "Moderate"},
        ]
        results.sort(key=lambda r: self.SEV_ORDER.get(r["severity"], 9))
        assert results[0]["severity"] == "Severe"
        assert results[1]["severity"] == "Moderate"
        assert results[2]["severity"] == "Mild"

    def test_all_expected_severities_in_order(self):
        order = sorted(self.SEV_ORDER.keys(), key=lambda k: self.SEV_ORDER[k])
        assert order == ["Severe", "Moderate", "Mild", "Uncertain", "No interaction", "Not found"]


# ── DEMOS data integrity ──────────────────────────────────────────────────────

class TestDemosData:
    def test_demos_has_three_entries(self):
        from app import DEMOS
        assert len(DEMOS) == 3

    def test_each_demo_has_required_keys(self):
        from app import DEMOS
        required = {"label", "description", "drug_a", "drug_b"}
        for demo in DEMOS:
            assert required.issubset(demo.keys()), f"Demo missing keys: {demo}"

    def test_demo_drug_names_are_strings(self):
        from app import DEMOS
        for demo in DEMOS:
            assert isinstance(demo["drug_a"], str)
            assert isinstance(demo["drug_b"], str)

    def test_demo_drug_a_and_b_differ(self):
        from app import DEMOS
        for demo in DEMOS:
            assert demo["drug_a"] != demo["drug_b"], \
                f"Demo has same drug for A and B: {demo['drug_a']}"

    def test_demo_labels_are_nonempty(self):
        from app import DEMOS
        for demo in DEMOS:
            assert len(demo["label"]) > 0

    def test_warfarin_aspirin_demo_exists(self):
        from app import DEMOS
        pairs = {(d["drug_a"], d["drug_b"]) for d in DEMOS}
        assert ("WARFARIN", "ASPIRIN") in pairs
