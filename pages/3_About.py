"""DDI-Agent — About page"""
import streamlit as st

st.set_page_config(
    page_title="About · DDI-Agent",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""<style>
.block-container{max-width:1050px;padding-top:1rem;padding-bottom:2.5rem}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:2rem;margin-bottom:0.2rem">
  <span style="font-size:1.5rem;font-weight:700;color:#0f2557">ℹ️ About DDI-Agent</span>
</div>
<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem">
  Project overview, motivation, and contribution
</div>
""", unsafe_allow_html=True)
st.divider()

st.subheader("Project")
st.markdown("""
**DDI-Agent: An Agentic AI Workflow for Drug Interaction Analysis**

This is a research prototype exploring agentic AI for drug-drug interaction (DDI) detection.
The system combines knowledge graph retrieval over ChEMBL 37 with a multi-agent reasoning
pipeline (GPT-4o-mini) to produce mechanism-level explanations of potential drug interactions.

**Research question:** Can a bounded, decision-loop agentic pipeline produce more reliable
DDI characterizations than a single-pass LLM call — particularly by detecting and abstaining
from low-confidence predictions rather than hallucinating an answer?
""")

st.subheader("Agentic workflow")
st.markdown("""
The pipeline implements a **4-stage agentic workflow** with an embedded decision loop:

```
Drug A + Drug B
       ↓
[PathRetrieverAgent]  ← ChEMBL 37 knowledge graph
       ↓ shared protein targets (or: not found → stop)
[MechanismReasonerAgent]  ← GPT-4o-mini chain-of-thought
       ↓ candidate mechanism explanation
[EvidenceValidatorAgent]  ← GPT-4o-mini self-validation
       ↓
  confidence ≥ 60?
    Yes → [ReportGeneratorAgent] → structured report
    No  → retry (up to 2 attempts)
          still low → ABSTAIN (insufficient evidence)
```

The decision loop is the core agentic contribution: instead of always generating a report,
the system can **abstain** when evidence quality falls below a threshold, signalling
uncertainty rather than producing a potentially unreliable answer.
""")

st.subheader("Data sources")
st.markdown("""
| Source | Role | Version |
|--------|------|---------|
| **ChEMBL** | Drug-target binding knowledge graph | ChEMBL 37 |
| **GPT-4o-mini** | Mechanism reasoning and validation | OpenAI API |
""")

st.subheader("What this system is not")
st.markdown("""
- **Not a clinical decision support tool.** Results must not be used for patient care.
- **Not a validated interaction database.** Severity labels are model-predicted.
- **Not a real-time literature search.** No PubMed or FDA label querying is performed.
- **Not peer-reviewed.** This is an academic research prototype.
""")

st.subheader("Disclaimer")
st.warning(
    "This tool is a research prototype for academic study of agentic AI methods. "
    "It has not been validated for clinical use. All output — including severity labels, "
    "mechanism explanations, and clinical notes — is machine-generated and unverified. "
    "Always consult a qualified clinical pharmacist or authoritative drug interaction "
    "resource (e.g. Lexicomp, Micromedex, Stockley's) for patient-related decisions."
)

st.divider()
st.caption("DDI-Agent · ChEMBL 37 · GPT-4o-mini · Research prototype · Not for clinical use")
