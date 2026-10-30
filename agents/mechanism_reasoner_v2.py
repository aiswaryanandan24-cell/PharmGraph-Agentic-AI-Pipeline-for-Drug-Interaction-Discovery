"""
Agent 2 v2: Mechanism Reasoner — DDInter-first, grounded, provenance-aware.

Self-contained — does not modify or import the original MechanismReasonerAgent,
so the original pipeline and its 125 passing tests are unaffected.
Save this file as agents/mechanism_reasoner_v2.py.

Design, per the dashboard-role spec this version implements:

  DDInter  -> the recorded interaction + severity (authoritative; "DDInter-reported")
  ChEMBL   -> supporting assay / shared-target observations (not a severity source)
  LLM      -> summarizes the retrieved evidence and flags gaps/disagreements;
              never asked to invent a mechanism beyond what was retrieved

Key rules:

1. DDInter severity is authoritative whenever a DDInter record exists — for
   BOTH "ddinter_only" and "both" cases. It is labeled "DDInter-reported" and
   is never overridden or averaged with an LLM guess. ChEMBL shared targets
   never independently set severity; they only support the mechanism text.

2. When shared ChEMBL targets exist (chembl_only or both), the LLM is given
   the exact retrieved target names and is explicitly instructed not to
   introduce a named mechanism or clinical syndrome unless it is a direct,
   well-established consequence of one of those specific named targets. A
   post-hoc lexical grounding check verifies the LLM's explanation actually
   references the retrieved evidence; if it doesn't, the result is marked
   mechanism_status="unresolved" and the UI should show that rather than the
   model's unverified claim as a confident conclusion.

3. "No DDInter record found" and "No shared ChEMBL targets retrieved" are
   reported as what they are — absence of evidence — never phrased as
   "unlikely" or implying the combination is safe.
"""

import os, re
from typing import List, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class ReasoningStepV2(BaseModel):
    step_number: int
    claim: str
    evidence: str


class DDIReasoningChainV2(BaseModel):
    drug_a: str
    drug_b: str
    interaction_type: str
    severity: str                        # "none" | "mild" | "moderate" | "severe"
    severity_label: str                  # "DDInter-reported" | "Model-predicted" | "Not applicable"
    mechanism_summary: str
    mechanism_status: str                # "grounded" | "unresolved" | "not_applicable"
    steps: List[ReasoningStepV2]
    clinical_recommendation: str
    source: str                          # "chembl_only" | "ddinter_only" | "both" | "none"
    ddinter_level: Optional[str] = None  # raw DDInter level if present, e.g. "major"


DDINTER_TO_SEVERITY = {"minor": "mild", "moderate": "moderate", "major": "severe"}

# words too generic to count as "grounding" on their own
_STOPWORDS = {"receptor", "protein", "enzyme", "channel", "transporter", "single",
              "human", "type", "subunit", "dependent"}


def _grounding_terms(target_name: str):
    words = re.split(r"[\s\-/]+", target_name.lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


def _is_grounded(text: str, shared_targets: list) -> bool:
    """Heuristic lexical check: does the LLM's own text reference at least one
    specific retrieved target name, rather than inventing an unrelated mechanism?
    This is a lexical check, not semantic verification — it catches the clearest
    case (the model naming a syndrome/mechanism with zero connection to any
    retrieved target), not every possible subtle overreach."""
    if not shared_targets:
        return False
    text_l = text.lower()
    for t in shared_targets:
        terms = _grounding_terms(t.get("target_name", ""))
        if any(term in text_l for term in terms):
            return True
    return False


class MechanismReasonerAgentV2:
    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Run: set OPENAI_API_KEY=your_key")
        self.llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    def reason(self, evidence: dict) -> DDIReasoningChainV2:
        drug_a = evidence["drug_a"]
        drug_b = evidence["drug_b"]
        shared_targets = evidence.get("shared_targets", [])
        ddinter = evidence.get("ddinter_interaction")
        source = evidence.get("source", "none")

        # ── Case 1: no evidence at all — explicit, not reassuring ──
        if source == "none":
            return DDIReasoningChainV2(
                drug_a=drug_a, drug_b=drug_b,
                interaction_type="unknown", severity="unknown",
                severity_label="Not applicable",
                mechanism_summary=(
                    "No DDInter record found. No shared ChEMBL targets retrieved. "
                    "Neither finding means this combination is safe — it means no "
                    "evidence was located in these two specific sources."
                ),
                mechanism_status="not_applicable",
                steps=[], clinical_recommendation="Consult a clinical pharmacist or an "
                                                   "independent interaction reference before co-administration.",
                source="none", ddinter_level=None,
            )

        # ── Case 2: DDInter record, no shared ChEMBL target ──
        # Severity is DDInter's, deterministically. No LLM call for severity —
        # the mechanism note is explicit that DDInter provides no mechanism data.
        if source == "ddinter_only":
            ddinter_level = ddinter["severity"]
            severity = DDINTER_TO_SEVERITY.get(ddinter_level, "moderate")
            return DDIReasoningChainV2(
                drug_a=drug_a, drug_b=drug_b,
                interaction_type="listed_in_ddinter",
                severity=severity,
                severity_label="DDInter-reported",
                mechanism_summary=(
                    f"DDInter 2.0 lists {ddinter['name_a']} (DDInter drug ID "
                    f"{ddinter['record_id_a']}) and {ddinter['name_b']} (DDInter drug ID "
                    f"{ddinter['record_id_b']}) as interacting, with severity recorded as "
                    f"{ddinter_level}. DDInter does not publish a mechanism or reference list "
                    f"in the data available to this system, and no shared ChEMBL target was "
                    f"retrieved, so no mechanism can be supported from either source. "
                    f"DDInter severity retrieved; mechanism explanation withheld."
                ),
                mechanism_status="not_applicable",
                steps=[ReasoningStepV2(
                    step_number=1,
                    claim=f"{ddinter['name_a']} and {ddinter['name_b']} are recorded as interacting in DDInter 2.0.",
                    evidence=f"DDInter severity level: {ddinter_level}. "
                             f"DDInter drug IDs: {ddinter['record_id_a']} ({ddinter['name_a']}) / "
                             f"{ddinter['record_id_b']} ({ddinter['name_b']}).",
                )],
                clinical_recommendation="Consult a clinical interaction reference or pharmacist for "
                                         "the underlying mechanism before co-administration.",
                source="ddinter_only", ddinter_level=ddinter_level,
            )

        # ── Case 3 & 4: shared ChEMBL targets exist (chembl_only or both) ──
        target_names = [t["target_name"] for t in shared_targets]
        targets_text = ""
        for t in shared_targets[:5]:
            targets_text += f"- {t['target_name']} ({t['target_type']})\n"
            targets_text += f"    {drug_a}: {t['drug_a_activity']}\n"
            targets_text += f"    {drug_b}: {t['drug_b_activity']}\n"

        ddinter_note = ""
        if ddinter is not None:
            ddinter_note = (f"\nDDInter 2.0 independently records this pair as a "
                             f"{ddinter['severity']}-severity clinical interaction "
                             f"(DDInter drug IDs {ddinter['record_id_a']} / {ddinter['record_id_b']}). "
                             f"That severity label takes display priority and will be used regardless "
                             f"of what you write below — do not restate or contradict it; focus only on "
                             f"explaining mechanism from the shared targets listed.\n")

        system_prompt = """You are a clinical pharmacologist summarizing retrieved evidence about a
possible drug-drug interaction. You must ground every claim in the SPECIFIC protein targets listed
below. Do NOT introduce a named mechanism or clinical syndrome (e.g. "serotonin syndrome", "QT
prolongation") unless it is a direct, well-established consequence of one of the SPECIFIC targets
listed — not a generic association with the drug classes involved. If the listed targets do not
clearly support a specific named mechanism, describe the interaction only in terms of the targets'
actual known function, and say explicitly that the broader clinical mechanism is not established by
this evidence. Be concise and accurate."""

        user_prompt = f"""Summarize the retrieved evidence for a potential interaction between {drug_a} and {drug_b}.

They share the following protein targets, retrieved from ChEMBL 37:
{targets_text}{ddinter_note}
Provide your analysis in this EXACT format:

INTERACTION_TYPE: [pharmacokinetic OR pharmacodynamic]
SEVERITY: [mild OR moderate OR severe — your best read of the shared-target evidence alone; may be overridden]
MECHANISM: [one sentence, referencing the SPECIFIC target(s) listed above by name]
RECOMMENDATION: [what a prescribing doctor should do]

STEP 1:
CLAIM: [first reasoning step — must reference a specific listed target]
EVIDENCE: [what supports this, from the data above]

STEP 2:
CLAIM: [second reasoning step]
EVIDENCE: [what supports this]

STEP 3:
CLAIM: [conclusion about the interaction]
EVIDENCE: [clinical basis — only if directly supported by the listed targets]"""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        chain = self._parse_response(drug_a, drug_b, response.content, source,
                                      ddinter["severity"] if ddinter else None)

        # Grounding check: does the model's own text reference the actual retrieved targets?
        full_text = chain.mechanism_summary + " " + " ".join(
            s.claim + " " + s.evidence for s in chain.steps)
        grounded = _is_grounded(full_text, shared_targets)
        chain.mechanism_status = "grounded" if grounded else "unresolved"
        if not grounded:
            chain.mechanism_summary = (
                "Mechanism unresolved — the model's explanation could not be verified against the "
                "specific retrieved ChEMBL targets and is not shown as a conclusion. "
                "Retrieved targets: " + ", ".join(target_names[:5]) + "."
            )

        # DDInter severity is authoritative whenever a DDInter record exists — always,
        # never reconciled/averaged with the model's guess.
        if ddinter is not None:
            chain.severity = DDINTER_TO_SEVERITY.get(ddinter["severity"], "moderate")
            chain.severity_label = "DDInter-reported"
            chain.ddinter_level = ddinter["severity"]
        else:
            chain.severity_label = "Model-predicted" if grounded else "Model-predicted (unresolved)"

        return chain

    def _parse_response(self, drug_a, drug_b, text, source, ddinter_level) -> DDIReasoningChainV2:
        lines = text.strip().split("\n")
        interaction_type = "unknown"
        severity = "unknown"
        mechanism = ""
        recommendation = ""
        steps = []
        current_step = None
        current_claim = ""
        current_evidence = ""

        for line in lines:
            line = line.strip()
            if line.startswith("INTERACTION_TYPE:"):
                interaction_type = line.split(":", 1)[1].strip().lower()
            elif line.startswith("SEVERITY:"):
                severity = line.split(":", 1)[1].strip().lower()
            elif line.startswith("MECHANISM:"):
                mechanism = line.split(":", 1)[1].strip()
            elif line.startswith("RECOMMENDATION:"):
                recommendation = line.split(":", 1)[1].strip()
            elif line.startswith("STEP "):
                if current_step and current_claim:
                    steps.append(ReasoningStepV2(step_number=current_step, claim=current_claim, evidence=current_evidence))
                try:
                    current_step = int(line.replace("STEP", "").replace(":", "").strip())
                except Exception:
                    current_step = len(steps) + 1
                current_claim = ""
                current_evidence = ""
            elif line.startswith("CLAIM:"):
                current_claim = line.split(":", 1)[1].strip()
            elif line.startswith("EVIDENCE:"):
                current_evidence = line.split(":", 1)[1].strip()

        if current_step and current_claim:
            steps.append(ReasoningStepV2(step_number=current_step, claim=current_claim, evidence=current_evidence))

        return DDIReasoningChainV2(
            drug_a=drug_a, drug_b=drug_b,
            interaction_type=interaction_type, severity=severity,
            severity_label="Model-predicted",
            mechanism_summary=mechanism, mechanism_status="grounded",  # provisional; overwritten by caller
            steps=steps,
            clinical_recommendation=recommendation,
            source=source, ddinter_level=ddinter_level,
        )
