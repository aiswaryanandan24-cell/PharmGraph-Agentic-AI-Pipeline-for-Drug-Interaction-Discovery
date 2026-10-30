"""
Tests for brand name → INN resolution.
These are fully deterministic and require no API key or knowledge graph.
Run with: pytest tests/test_brand_resolution.py -v
"""

import sys
import os

# Allow importing from project root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import BRAND_TO_INN, _resolve


# ── BRAND_TO_INN dictionary tests ────────────────────────────────────────────

class TestBrandToINNDict:
    def test_analgesic_tylenol(self):
        assert BRAND_TO_INN["TYLENOL"] == "ACETAMINOPHEN"

    def test_analgesic_panadol(self):
        assert BRAND_TO_INN["PANADOL"] == "ACETAMINOPHEN"

    def test_nsaid_advil(self):
        assert BRAND_TO_INN["ADVIL"] == "IBUPROFEN"

    def test_nsaid_motrin(self):
        assert BRAND_TO_INN["MOTRIN"] == "IBUPROFEN"

    def test_nsaid_aleve(self):
        assert BRAND_TO_INN["ALEVE"] == "NAPROXEN"

    def test_anticoagulant_coumadin(self):
        assert BRAND_TO_INN["COUMADIN"] == "WARFARIN"

    def test_anticoagulant_plavix(self):
        assert BRAND_TO_INN["PLAVIX"] == "CLOPIDOGREL"

    def test_anticoagulant_eliquis(self):
        assert BRAND_TO_INN["ELIQUIS"] == "APIXABAN"

    def test_anticoagulant_xarelto(self):
        assert BRAND_TO_INN["XARELTO"] == "RIVAROXABAN"

    def test_statin_lipitor(self):
        assert BRAND_TO_INN["LIPITOR"] == "ATORVASTATIN"

    def test_statin_zocor(self):
        assert BRAND_TO_INN["ZOCOR"] == "SIMVASTATIN"

    def test_statin_crestor(self):
        assert BRAND_TO_INN["CRESTOR"] == "ROSUVASTATIN"

    def test_calcium_channel_blocker_norvasc(self):
        assert BRAND_TO_INN["NORVASC"] == "AMLODIPINE"

    def test_calcium_channel_blocker_calan(self):
        assert BRAND_TO_INN["CALAN"] == "VERAPAMIL"

    def test_antidepressant_prozac(self):
        assert BRAND_TO_INN["PROZAC"] == "FLUOXETINE"

    def test_antidepressant_zoloft(self):
        assert BRAND_TO_INN["ZOLOFT"] == "SERTRALINE"

    def test_antidepressant_lexapro(self):
        assert BRAND_TO_INN["LEXAPRO"] == "ESCITALOPRAM"

    def test_anxiolytic_xanax(self):
        assert BRAND_TO_INN["XANAX"] == "ALPRAZOLAM"

    def test_thyroid_synthroid(self):
        assert BRAND_TO_INN["SYNTHROID"] == "LEVOTHYROXINE"

    def test_diabetes_glucophage(self):
        assert BRAND_TO_INN["GLUCOPHAGE"] == "METFORMIN"

    def test_cardiac_lanoxin(self):
        assert BRAND_TO_INN["LANOXIN"] == "DIGOXIN"

    def test_cardiac_cordarone(self):
        assert BRAND_TO_INN["CORDARONE"] == "AMIODARONE"

    def test_opioid_ultram(self):
        assert BRAND_TO_INN["ULTRAM"] == "TRAMADOL"

    def test_antibiotic_cipro(self):
        assert BRAND_TO_INN["CIPRO"] == "CIPROFLOXACIN"

    def test_antifungal_diflucan(self):
        assert BRAND_TO_INN["DIFLUCAN"] == "FLUCONAZOLE"

    def test_ppi_prilosec(self):
        assert BRAND_TO_INN["PRILOSEC"] == "OMEPRAZOLE"

    def test_unknown_brand_not_present(self):
        assert "FAKEBRAND" not in BRAND_TO_INN

    def test_inn_names_not_in_dict_as_keys(self):
        # INN names should NOT be keys — the dict maps brand → INN
        assert "WARFARIN" not in BRAND_TO_INN
        assert "IBUPROFEN" not in BRAND_TO_INN
        assert "METFORMIN" not in BRAND_TO_INN

    def test_dict_is_nonempty(self):
        assert len(BRAND_TO_INN) >= 50

    def test_all_values_are_strings(self):
        for brand, inn in BRAND_TO_INN.items():
            assert isinstance(inn, str), f"{brand} maps to non-string: {inn!r}"

    def test_all_values_uppercase(self):
        for brand, inn in BRAND_TO_INN.items():
            assert inn == inn.upper(), f"{brand} → {inn!r} is not uppercase"

    def test_all_keys_uppercase(self):
        for brand in BRAND_TO_INN:
            assert brand == brand.upper(), f"Key {brand!r} is not uppercase"


# ── _resolve() function tests ─────────────────────────────────────────────────

class TestResolveFunction:
    def test_brand_name_resolves_to_inn(self):
        resolved, original = _resolve("TYLENOL")
        assert resolved == "ACETAMINOPHEN"
        assert original == "TYLENOL"

    def test_brand_name_case_insensitive(self):
        resolved, original = _resolve("tylenol")
        assert resolved == "ACETAMINOPHEN"
        assert original == "TYLENOL"

    def test_mixed_case_brand(self):
        resolved, original = _resolve("Tylenol")
        assert resolved == "ACETAMINOPHEN"
        assert original == "TYLENOL"

    def test_inn_name_passes_through_unchanged(self):
        resolved, original = _resolve("WARFARIN")
        assert resolved == "WARFARIN"
        assert original is None  # not a brand name

    def test_lowercase_inn_passes_through(self):
        resolved, original = _resolve("warfarin")
        # Not in BRAND_TO_INN (which has brand keys), so passes through as-is
        assert original is None

    def test_empty_string_returns_empty(self):
        resolved, original = _resolve("")
        assert resolved == ""
        assert original is None

    def test_none_returns_none(self):
        resolved, original = _resolve(None)
        assert resolved is None
        assert original is None

    def test_coumadin_resolves(self):
        resolved, original = _resolve("COUMADIN")
        assert resolved == "WARFARIN"
        assert original == "COUMADIN"

    def test_unknown_drug_passes_through(self):
        resolved, original = _resolve("UNKNOWNDRUG")
        assert resolved == "UNKNOWNDRUG"
        assert original is None

    def test_resolve_strips_whitespace(self):
        resolved, original = _resolve("  TYLENOL  ")
        assert resolved == "ACETAMINOPHEN"
        assert original == "TYLENOL"
