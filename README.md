# PharmGraph
### Drug Interaction & Molecular Target Knowledge Graph Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://github.com/langchain-ai/langchain)
[![OpenAI](https://img.shields.io/badge/LLM-GPT--4o--mini-412991.svg)](https://openai.com/)
[![NetworkX](https://img.shields.io/badge/graph-NetworkX-informational.svg)](https://networkx.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![ChEMBL 37](https://img.shields.io/badge/data-ChEMBL%2037-orange.svg)](https://www.ebi.ac.uk/chembl/)
[![DDInter 2.0](https://img.shields.io/badge/validation-DDInter%202.0-red.svg)](https://ddinter2.scbdd.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Overview

Current clinical decision support systems (CDSS) and EHR platforms flag drug-drug interactions with a severity label — but provide no mechanistic context. Understanding *why* an interaction occurs (which protein targets are shared, which pathway is implicated) requires manually cross-referencing two separate databases: DDInter for severity and ChEMBL for molecular target data. This is time-consuming and is not currently within the scope of most CDSS tools.

PharmGraph fills that gap. It is a **knowledge graph-powered, multi-agent AI platform** that queries both databases in a single workflow and returns a complete, downloadable interaction report at one click — designed to complement EHR integration and clinical trial pharmacovigilance workflows.

**Each report includes:**
- DDInter severity classification (Major / Moderate / Minor)
- Shared protein binding targets with assay data (from ChEMBL 37)
- LLM-generated mechanism of action explaining *why* the interaction occurs
- Evidence-weighted confidence score
- Generic and brand names for both drugs
- Downloadable clinical note (for trial documentation or EHR attachment)

> **Scope:** PharmGraph retrieves and explains documented interactions. It does not predict undocumented ones. "No interaction found" means no evidence in these two sources — it does not guarantee safety.

---

## Architecture

```
Drug A ──┐
         ├──► [Agent 1] PathRetriever
Drug B ──┘    ChEMBL KG traversal + DDInter lookup
                      │
                      ▼
         [Agent 2] MechanismReasoner
         GPT-4o-mini, grounded to retrieved evidence only
                      │
                      ▼
         [Agent 3] EvidenceValidator
         Evidence-weighted confidence scoring
                      │
                      ▼
         [Agent 4] ReportGenerator
         Structured clinical report (PDF + JSON)
```

Orchestrated via LangGraph — each agent writes typed outputs to a shared state object.

---

## Knowledge Graph

Built from two sources via a reproducible ETL pipeline:

| Component | Count |
|---|---|
| Drug nodes | 10,000 |
| Protein target nodes | 4,527 |
| Total nodes | 15,557 |
| Total edges | 96,763 |
| DDInter interaction pairs | 11,727 |

Of 10,000 drugs, 3,736 (37.4%) have ChEMBL target data and receive full mechanistic enrichment. The remaining 6,264 return DDInter severity without molecular explanation. This is reported explicitly in every generated report.

---

## Confidence Scoring

Rather than relying on LLM self-reported confidence, PharmGraph applies a deterministic evidence floor schema:

| Evidence | DDInter Severity | Floor |
|---|---|---|
| DDInter + shared ChEMBL targets | Major | 90 |
| DDInter + shared ChEMBL targets | Moderate | 85 |
| DDInter + shared ChEMBL targets | Minor/Unknown | 80 |
| DDInter only | Major | 79 |
| DDInter only | Moderate | 75 |
| DDInter only | Minor/Unknown | 70 |
| ChEMBL only (≥3 shared targets) | — | 67 |
| ChEMBL only (1–2 shared targets) | — | 60 |

Scores are capped at 90 to reflect the absence of prospective clinical validation. DDInter severity is never overridden by the LLM.

---

## Evaluation

Validated on **1,000 drug pairs** (500 DDInter-confirmed positives + 500 presumed negatives — drugs in DDInter vocabulary with no recorded interaction).

| | Rule-Only | Full Pipeline |
|---|---|---|
| Precision | 99.2% | 98.7% |
| False Positives | 1 / 500 | 2 / 500 |

**Miss analysis:** 100% of false negatives had zero shared ChEMBL targets. All missed interactions are due to ChEMBL coverage gaps, not logic errors.

### Ablation (200-pair benchmark)

| Method | Precision | Recall |
|---|---|---|
| Rule-only (KG) | 100% | 59% |
| LLM-only | 92.6% | 25% |
| Full pipeline | 100% | 59% |

The KG rule layer drives precision. The LLM layer adds mechanistic explanation without introducing false positives.

---

## Setup

### 1. Clone and install
```bash
git clone https://github.com/aiswaryanandan24-cell/pharmgraph.git
cd pharmgraph
pip install -r requirements.txt
```

### 2. Set API key
```bash
# Windows
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```

### 3. Build the knowledge graph
```bash
python kg/build_graph.py
```
Requires ChEMBL 37 and DDInter 2.0 source data in `data/`. Runtime: ~45 minutes.

### 4. Run
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

125 tests covering path retrieval, mechanism reasoning, brand name resolution, confidence scoring, and full pipeline integration.

---

## Data Sources

| Source | Version | Role |
|---|---|---|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 37 | Drug-target bioactivity (IC50, Ki, Kd, EC50) |
| [DDInter](https://ddinter2.scbdd.com) | 2.0 | Clinically curated DDI severity labels |

---

## Limitations

- ChEMBL coverage: 62.6% of drugs have no target data and receive severity only
- Static snapshot — not updated in real time
- No dosage, pharmacogenomics, or patient-specific factors
- Not validated in a clinical setting with patient data
- Requires OpenAI API key

---

## License

MIT — see [LICENSE](LICENSE)
