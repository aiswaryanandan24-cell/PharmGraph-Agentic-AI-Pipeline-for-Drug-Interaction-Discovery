import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class EvidenceValidatorAgent:
    """
    Agent 3: Evidence Validator
    Takes the reasoning chain from Agent 2 and scores how confident
    we should be in the interaction prediction.
    Outputs: confidence score, flags any weak points in the reasoning.
    """

    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        self.llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)

    def validate(self, reasoning_chain) -> dict:
        """
        Takes a DDIReasoningChain from Agent 2 and validates the evidence.
        Returns a validation report with confidence score.
        """
        drug_a = reasoning_chain.drug_a
        drug_b = reasoning_chain.drug_b
        severity = reasoning_chain.severity
        mechanism = reasoning_chain.mechanism_summary

        steps_text = ""
        for step in reasoning_chain.steps:
            steps_text += f"Step {step.step_number}: {step.claim}\n"
            steps_text += f"  Evidence: {step.evidence}\n\n"

        system_prompt = """You are a critical pharmacology reviewer.
Your job is to evaluate the quality of drug interaction reasoning
and assign a confidence score. Be honest about weaknesses."""

        user_prompt = f"""Review this drug-drug interaction analysis:

DRUGS: {drug_a} + {drug_b}
SEVERITY: {severity}
MECHANISM: {mechanism}

REASONING:
{steps_text}

Evaluate this analysis and respond in EXACTLY this format:

CONFIDENCE_SCORE: [0-100]
EVIDENCE_QUALITY: [strong OR moderate OR weak]
BIGGEST_STRENGTH: [one sentence]
BIGGEST_WEAKNESS: [one sentence]
VALIDATED: [yes OR no]
NOTES: [any important clinical caveats in one sentence]"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        print(f"  Validating reasoning for {drug_a} + {drug_b}...")
        response = self.llm.invoke(messages)
        return self._parse_response(response.content)

    def _parse_response(self, text) -> dict:
        result = {
            "confidence_score": 0,
            "evidence_quality": "unknown",
            "biggest_strength": "",
            "biggest_weakness": "",
            "validated": False,
            "notes": ""
        }
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CONFIDENCE_SCORE:"):
                try:
                    result["confidence_score"] = int(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("EVIDENCE_QUALITY:"):
                result["evidence_quality"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("BIGGEST_STRENGTH:"):
                result["biggest_strength"] = line.split(":", 1)[1].strip()
            elif line.startswith("BIGGEST_WEAKNESS:"):
                result["biggest_weakness"] = line.split(":", 1)[1].strip()
            elif line.startswith("VALIDATED:"):
                result["validated"] = line.split(":", 1)[1].strip().lower() == "yes"
            elif line.startswith("NOTES:"):
                result["notes"] = line.split(":", 1)[1].strip()
        return result


# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from agents.path_retriever import PathRetrieverAgent
    from agents.mechanism_reasoner import MechanismReasonerAgent

    # Agent 1
    retriever = PathRetrieverAgent()
    path_result = retriever.get_shared_targets("PRAZOSIN", "NICOTINE")

    # Agent 2
    reasoner = MechanismReasonerAgent()
    chain = reasoner.reason(path_result)

    # Agent 3
    validator = EvidenceValidatorAgent()
    validation = validator.validate(chain)

    print("\n" + "="*50)
    print("VALIDATION REPORT")
    print("="*50)
    print(f"Confidence Score : {validation['confidence_score']}/100")
    print(f"Evidence Quality : {validation['evidence_quality']}")
    print(f"Validated        : {'YES' if validation['validated'] else 'NO'}")
    print(f"Strength         : {validation['biggest_strength']}")
    print(f"Weakness         : {validation['biggest_weakness']}")
    print(f"Notes            : {validation['notes']}")
