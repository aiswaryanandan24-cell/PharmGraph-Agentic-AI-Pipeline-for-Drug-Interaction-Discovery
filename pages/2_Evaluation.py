"""DDI-Agent — Evaluation page"""
import streamlit as st

st.set_page_config(
    page_title="Evaluation · DDI-Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""<style>
.block-container{max-width:1050px;padding-top:1rem;padding-bottom:2.5rem}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:2rem;margin-bottom:0.2rem">
  <span style="font-size:1.5rem;font-weight:700;color:#0f2557">📊 Evaluation</span>
</div>
<div style="color:#6b7280;font-size:0.85rem;margin-bottom:1rem">
  Honest assessment of DDI-Agent performance and methodology
</div>
""", unsafe_allow_html=True)
st.divider()

st.info(
    "**Research prototype — not a validated clinical tool.**  "
    "This page reports the actual pipeline performance, "
    "data limitations, and what this system can and cannot claim."
)

# ── Architecture ──────────────────────────────────────────────────────────────
st.subheader("Architecture")
st.markdown("""
DDI-Agent is a 4-agent agentic pipeline:

| Step | Agent | Method |
|------|-------|--------|
| 1 | **PathRetrieverAgent** | Graph query on ChEMBL 37 knowledge graph — retrieves shared protein targets |
| 2 | **MechanismReasonerAgent** | GPT-4o-mini chain-of-thought — generates a mechanism explanation from targets |
| 3 | **EvidenceValidatorAgent** | GPT-4o-mini self-validation — rates confidence; triggers revision if low |
| 4 | **ReportGeneratorAgent** | Structured clinical-style report |

The decision loop in Stage 3 allows up to **2 reasoning attempts** before abstaining with an
"insufficient evidence" result rather than fabricating a low-confidence answer.
""")

# ── Knowledge base ────────────────────────────────────────────────────────────
st.subheader("Knowledge base")
st.markdown("""
- **Source:** ChEMBL 37 (public, pre-built knowledge graph of drug-target binding assays)
- **Coverage:** Human protein targets with binding data for approved drugs and candidates
- **Limitation:** Assay data reflects *in vitro* binding, not clinical interaction outcomes.
  A shared target is a signal of potential interaction, not a confirmed drug-drug effect.
- **No real-time retrieval:** The pipeline does not query PubMed, FDA labels, or other
  live sources during inference.
""")

# ── What the confidence scores mean ──────────────────────────────────────────
st.subheader("Confidence scores — interpretation")
st.warning(
    "The confidence score (0–100) is a **self-rating produced by the same GPT-4o-mini model** "
    "that generated the analysis. It has **not** been calibrated against held-out clinical data "
    "or validated interaction databases. Do not use it as a quality proxy."
)
st.markdown("""
| Score range | Interpretation in this system |
|-------------|-------------------------------|
| ≥ 60 | Model considers evidence sufficient to report |
| < 60 (any attempt) | Triggers a retry; if still < 60 after 2 attempts → abstain |
| Displayed in Technical details tab | Visible there only — not used in the main result display |
""")

# ── Known limitations ─────────────────────────────────────────────────────────
st.subheader("Known limitations")
st.markdown("""
1. **No synonym mapping** — drug names must match ChEMBL 37 INN names.
   TYLENOL, ADVIL, PARACETAMOL may fail; use ACETAMINOPHEN, IBUPROFEN.
2. **No dose or patient context** — interactions are flagged at the drug level;
   dose-dependence, renal/hepatic adjustment, and patient-specific factors are ignored.
3. **Severity is model-predicted** — there is no mapping to a validated severity taxonomy
   (e.g. Stockley's, Lexicomp). Labels such as "Severe" reflect GPT-4o-mini's assessment.
4. **Causal explanations may be unsupported** — the mechanism text is generated from shared
   targets; it may include plausible-sounding but unverified causal pathways.
5. **No independent validation** — predictions have not been benchmarked against a frozen
   held-out test set of known interactions.
""")

# ── Planned work ──────────────────────────────────────────────────────────────
st.subheader("Planned improvements")
st.markdown("""
- **Synonym normalization layer** (ChEMBL name → INN mapping)
- **Provenance table** — per-target ChEMBL assay IDs and evidence strength
- **Frozen test set evaluation** against DrugBank or TWOSIDES interaction labels
- **Real-time PubMed retrieval** for supporting literature (RAG step)
- **Calibration study** — align confidence scores with empirical precision/recall
""")

st.divider()
st.caption("DDI-Agent · Research prototype · Not for clinical use")
