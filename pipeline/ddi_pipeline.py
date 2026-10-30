"""
DDI-Agent: Full LangGraph Pipeline
Wires all 4 agents into a stateful agentic graph.

Flow: retrieve_paths → reason → validate → report → END
"""

import sys
sys.path.append(".")

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.path_retriever import PathRetrieverAgent
from agents.mechanism_reasoner import MechanismReasonerAgent, DDIReasoningChain
from agents.evidence_validator import EvidenceValidatorAgent
from agents.report_generator import ReportGeneratorAgent


# ── Shared state passed between agents ────────────────────────────────────

class DDIState(TypedDict):
    drug_a: str
    drug_b: str
    path_result: Optional[dict]
    reasoning_chain: Optional[object]
    validation: Optional[dict]
    report: Optional[str]
    error: Optional[str]


# ── Load all agents once ───────────────────────────────────────────────────

print("Initialising agents...")
retriever  = PathRetrieverAgent()
reasoner   = MechanismReasonerAgent()
validator  = EvidenceValidatorAgent()
reporter   = ReportGeneratorAgent()
print("All agents ready.\n")


# ── Node functions (one per agent) ────────────────────────────────────────

def node_retrieve(state: DDIState) -> DDIState:
    print(f"[Agent 1] Finding shared targets: {state['drug_a']} + {state['drug_b']}")
    result = retriever.get_shared_targets(state["drug_a"], state["drug_b"])
    if "error" in result:
        return {**state, "error": result["error"]}
    print(f"[Agent 1] Found {len(result['shared_targets'])} shared targets")
    return {**state, "path_result": result}


def node_reason(state: DDIState) -> DDIState:
    if state.get("error"):
        return state
    print(f"[Agent 2] Reasoning about mechanism...")
    chain = reasoner.reason(state["path_result"])
    print(f"[Agent 2] Severity: {chain.severity} | Type: {chain.interaction_type}")
    return {**state, "reasoning_chain": chain}


def node_validate(state: DDIState) -> DDIState:
    if state.get("error"):
        return state
    print(f"[Agent 3] Validating evidence...")
    validation = validator.validate(state["reasoning_chain"])
    print(f"[Agent 3] Confidence: {validation['confidence_score']}/100")
    return {**state, "validation": validation}


def node_report(state: DDIState) -> DDIState:
    if state.get("error"):
        return state
    print(f"[Agent 4] Generating clinical report...")
    report = reporter.generate(
        state["path_result"],
        state["reasoning_chain"],
        state["validation"]
    )
    print(f"[Agent 4] Report ready.")
    return {**state, "report": report}


# ── Build the LangGraph ────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(DDIState)

    graph.add_node("retrieve_paths", node_retrieve)
    graph.add_node("reason",         node_reason)
    graph.add_node("validate",       node_validate)
    graph.add_node("report",         node_report)

    graph.set_entry_point("retrieve_paths")
    graph.add_edge("retrieve_paths", "reason")
    graph.add_edge("reason",         "validate")
    graph.add_edge("validate",       "report")
    graph.add_edge("report",         END)

    return graph.compile()


# ── Main: run a drug pair through the full pipeline ───────────────────────

def run_ddi_check(drug_a: str, drug_b: str) -> dict:
    pipeline = build_pipeline()

    initial_state: DDIState = {
        "drug_a": drug_a.upper(),
        "drug_b": drug_b.upper(),
        "path_result": None,
        "reasoning_chain": None,
        "validation": None,
        "report": None,
        "error": None
    }

    print("=" * 55)
    print(f"  DDI-AGENT PIPELINE")
    print(f"  Checking: {drug_a.upper()} + {drug_b.upper()}")
    print("=" * 55)

    final_state = pipeline.invoke(initial_state)

    if final_state.get("error"):
        print(f"\nError: {final_state['error']}")
        return final_state

    print("\n" + final_state["report"])

    # Save report
    filename = f"data/report_{drug_a.lower()}_{drug_b.lower()}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_state["report"])
    print(f"\nReport saved to {filename}")

    return final_state


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with PRAZOSIN + INDOMETHACIN (5 shared targets — strongest pair we found)
    run_ddi_check("PRAZOSIN", "INDOMETHACIN")
