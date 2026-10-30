import os
from typing import List, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Structured output models ───────────────────────────────────────────────

class ReasoningStep(BaseModel):
    step_number: int
    claim: str
    evidence: str

class DDIReasoningChain(BaseModel):
    drug_a: str
    drug_b: str
    interaction_type: str        # e.g. "pharmacokinetic" or "pharmacodynamic"
    severity: str                # "mild" | "moderate" | "severe"
    mechanism_summary: str       # one sentence plain-English explanation
    steps: List[ReasoningStep]   # step-by-step reasoning
    clinical_recommendation: str # what a doctor should do

# ── Agent 2: Mechanism Reasoner ────────────────────────────────────────────

class MechanismReasonerAgent:
    """
    Agent 2: Mechanism Reasoner
    Takes shared protein targets from Agent 1 and uses GPT to explain
    WHY those shared targets cause a drug-drug interaction.
    """

    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Run: set OPENAI_API_KEY=your_key")
        self.llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    def reason(self, path_retriever_result: dict) -> DDIReasoningChain:
        """
        Takes the output from PathRetrieverAgent and returns a structured
        reasoning chain explaining the interaction.
        """
        drug_a = path_retriever_result["drug_a"]
        drug_b = path_retriever_result["drug_b"]
        shared_targets = path_retriever_result["shared_targets"]

        if not shared_targets:
            return DDIReasoningChain(
                drug_a=drug_a,
                drug_b=drug_b,
                interaction_type="none",
                severity="none",
                mechanism_summary="No shared protein targets found — interaction unlikely via direct mechanism.",
                steps=[],
                clinical_recommendation="No specific precaution needed based on available data."
            )

        # Format shared targets for the prompt
        targets_text = ""
        for t in shared_targets[:5]:
            targets_text += f"- {t['target_name']} ({t['target_type']})\n"
            targets_text += f"    {drug_a}: {t['drug_a_activity']}\n"
            targets_text += f"    {drug_b}: {t['drug_b_activity']}\n"

        system_prompt = """You are a clinical pharmacologist expert in drug-drug interactions.
You reason step-by-step about how shared protein targets between two drugs
can lead to interactions in the human body.
Be concise, accurate, and clinically relevant."""

        user_prompt = f"""Analyze the potential drug-drug interaction between {drug_a} and {drug_b}.

They share the following protein targets in the body:
{targets_text}

Provide your analysis in this EXACT format:

INTERACTION_TYPE: [pharmacokinetic OR pharmacodynamic]
SEVERITY: [mild OR moderate OR severe]
MECHANISM: [one sentence explaining the core mechanism]
RECOMMENDATION: [what a prescribing doctor should do]

STEP 1:
CLAIM: [first reasoning step]
EVIDENCE: [what supports this]

STEP 2:
CLAIM: [second reasoning step]
EVIDENCE: [what supports this]

STEP 3:
CLAIM: [conclusion about the interaction]
EVIDENCE: [clinical basis]"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        print(f"  Asking GPT about {drug_a} + {drug_b}...")
        response = self.llm.invoke(messages)
        return self._parse_response(drug_a, drug_b, response.content)

    def _parse_response(self, drug_a, drug_b, text) -> DDIReasoningChain:
        """Parse GPT's text response into structured format."""
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
                    steps.append(ReasoningStep(
                        step_number=current_step,
                        claim=current_claim,
                        evidence=current_evidence
                    ))
                try:
                    current_step = int(line.replace("STEP", "").replace(":", "").strip())
                except:
                    current_step = len(steps) + 1
                current_claim = ""
                current_evidence = ""
            elif line.startswith("CLAIM:"):
                current_claim = line.split(":", 1)[1].strip()
            elif line.startswith("EVIDENCE:"):
                current_evidence = line.split(":", 1)[1].strip()

        if current_step and current_claim:
            steps.append(ReasoningStep(
                step_number=current_step,
                claim=current_claim,
                evidence=current_evidence
            ))

        return DDIReasoningChain(
            drug_a=drug_a,
            drug_b=drug_b,
            interaction_type=interaction_type,
            severity=severity,
            mechanism_summary=mechanism,
            steps=steps,
            clinical_recommendation=recommendation
        )


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from agents.path_retriever import PathRetrieverAgent

    # Step 1: get paths from Agent 1
    retriever = PathRetrieverAgent()
    path_result = retriever.get_shared_targets("PRAZOSIN", "NICOTINE")

    print(f"\nShared targets found: {len(path_result['shared_targets'])}")

    # Step 2: reason with Agent 2
    reasoner = MechanismReasonerAgent()
    chain = reasoner.reason(path_result)

    print("\n" + "="*50)
    print(f"DDI ANALYSIS: {chain.drug_a} + {chain.drug_b}")
    print("="*50)
    print(f"Type:       {chain.interaction_type}")
    print(f"Severity:   {chain.severity}")
    print(f"Mechanism:  {chain.mechanism_summary}")
    print(f"\nReasoning steps:")
    for step in chain.steps:
        print(f"\n  Step {step.step_number}: {step.claim}")
        print(f"  Evidence: {step.evidence}")
    print(f"\nRecommendation: {chain.clinical_recommendation}")
