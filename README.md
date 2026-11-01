# PharmGraph
### A Multi-Agent AI System for Mechanistic Drug-Drug Interaction Discovery

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![ChEMBL 37](https://img.shields.io/badge/data-ChEMBL%2037-orange.svg)](https://www.ebi.ac.uk/chembl/)
[![DDInter 2.0](https://img.shields.io/badge/validation-DDInter%202.0-red.svg)](https://ddinter2.scbdd.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Overview

PharmGraph is a **knowledge-grounded, multi-agent AI pipeline** that detects and mechanistically explains drug interactions (DDIs) by fusing biomedical knowledge graph traversal with structured large language model reasoning.

Unlike black-box ML classifiers that predict DDIs without explanation, PharmGraph produces **mechanistically interpretable reports** grounded in real bioactivity data — every claim is traceable to a specific ChEMBL assay or DDInter clinical record. The system is designed around a core principle: **LLM reasoning should augment evidence, never replace it.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PharmGraph Pipeline                    │
│                                                             │
│  Drug A ──┐                                                 │
│           ├──► [Agent 1] PathRetriever                      │
│  Drug B ──┘    ChEMBL KG traversal + DDInter lookup         │
│                        │                                    │
│                        ▼                                    │
│           [Agent 2] MechanismReasoner                       │
│           GPT-4o-mini, grounded to retrieved evidence only  │
│                        │                                    │
│                        ▼                                    │
│           [Agent 3] EvidenceValidator                       │
│           Evidence-weighted confidence scoring              │
│                        │                                    │
│                        ▼                                    │
│           [Agent 4] ReportGenerator                         │
│           Structured clinical interaction report            │
└─────────────────────────────────────────────────────────────┘
```

**Orchestration:** Stateful agent graph via LangGraph — each agent writes typed outputs to a shared state object, enabling downstream agents to build on upstream results without requerying.

---

## Key Design Decisions

### 1. Evidence-Weighted Confidence Scoring
Rather than relying on LLM self-reported confidence (which clusters at 65–75 due to temperature=0 calibration bias), PharmGraph applies a **deterministic evidence floor schema**:

| Evidence Type | DDInter Severity | Confidence Floor |
|---|---|---|
| DDInter + ChEMBL shared targets | Major | 90 |
| DDInter + ChEMBL shared targets | Moderate | 85 |
| DDInter + ChEMBL shared targets | Minor/Unknown | 80 |
| DDInter only | Major | 79 |
| DDInter only | Moderate | 75 |
| DDInter only | Minor/Unknown | 70 |
| ChEMBL only (≥3 shared targets) | — | 67 |
| ChEMBL only (2 shared targets) | — | 63 |
| ChEMBL only (1 shared target) | — | 60 |

This eliminates LLM-as-judge bias while preserving GPT-4o-mini for qualitative interpretation (mechanism explanation, clinical caveats, biggest strength/weakness analysis).

### 2. DDInter Severity is Always Authoritative
The LLM **never overrides** a recorded DDInter severity label. Clinical records from DDInter 2.0 propagate directly to the final report; the LLM's role is to explain the mechanism, not to re-classify the severity.

### 3. Adaptive Confidence Threshold Routing
Interaction detection uses a shared-target-count-aware threshold:
- **≥2 shared ChEMBL targets** → threshold lowered to 40 (mechanistic signal is strong)
- **1 shared target or DDInter-only** → standard threshold (60)

This avoids false negatives on well-evidenced interactions where the LLM may hedge.

### 4. Grounding Verification
Agent 2 output is lexically checked to confirm it references the actual retrieved protein targets. Responses that introduce ungrounded targets are flagged and the confidence score is penalized automatically.

### 5. Brand Name Resolution
Accepts trade names (Lipitor → atorvastatin, Coumadin → warfarin) via a curated synonym map, lowering the barrier for clinical use cases.

---

## Evaluation

### External Validation Against DDInter 2.0

PharmGraph was validated on **1,000 drug pairs** (500 positive DDInter-confirmed interactions + 500 negative pairs — drugs present in DDInter's vocabulary but with no recorded interaction). This design breaks the circularity of KG-sampled benchmarks.

| Metric | Rule-Only Baseline | Full PharmGraph Pipeline |
|---|---|---|
| Precision | 0.992 | 0.987 |
| False Positives | 1 / 500 | ~2 / 500 |

**Miss analysis:** 100% of false negatives had **zero shared ChEMBL targets** for the interacting drug pair. The system makes no reasoning errors — it only misses interactions that are genuinely invisible to the ChEMBL knowledge graph. This is a coverage gap, not a model failure.

### Ablation Study (32-pair clinical benchmark)

| Method | Description | Notes |
|---|---|---|
| Method 1 | Shared-target rule only | No LLM |
| Method 2 | Single LLM call | No knowledge graph |
| Method 3 | Graph + single LLM prompt | No multi-agent |
| Method 4 | Full PharmGraph pipeline | Four agents + grounding |

Full ablation results available in the Evaluation page of the Streamlit app.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/aiswaryanandan24-cell/pharmgraph.git
cd pharmgraph
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Or set it in your terminal:

```bash
# Windows
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```

> ⚠️ Never commit your API key. `.env` is in `.gitignore`.

### 3. Build the knowledge graph

The `.pkl` KG files are not included in the repository (size). Build them locally:

```bash
python kg/build_graph.py
```

This requires ChEMBL 37 and DDInter 2.0 source data. See `data/` for expected file paths.

### 4. Launch the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
pharmgraph/
├── app.py                          # Streamlit UI — main analysis interface
├── pages/
│   ├── 2_Evaluation.py             # Evaluation dashboard (metrics, miss analysis)
│   └── 3_About.py                  # Methodology and design rationale
├── agents/
│   ├── path_retriever.py           # Agent 1: ChEMBL KG traversal
│   ├── path_retriever_v2.py        # Agent 1 v2: merged ChEMBL + DDInter KG
│   ├── mechanism_reasoner.py       # Agent 2: grounded LLM mechanism explanation
│   ├── mechanism_reasoner_v2.py    # Agent 2 v2: DDInter-first reasoning
│   ├── evidence_validator.py       # Agent 3: evidence-weighted confidence scoring
│   └── report_generator.py         # Agent 4: structured clinical report
├── pipeline/
│   └── ddi_pipeline.py             # LangGraph stateful orchestration
├── kg/
│   ├── build_graph.py              # KG construction from ChEMBL + DDInter
│   └── parse_chembl.py             # ChEMBL bioactivity parser
├── data/
│   ├── drug_kg.pkl                 # ChEMBL-only KG (not in repo)
│   ├── drug_kg_merged.pkl          # ChEMBL + DDInter merged KG (not in repo)
│   └── drug_kg_full_merged.pkl     # Full merged KG (not in repo)
├── tests/                          # 125 pytest unit tests
├── ablation_study.py               # 4-method ablation comparison
├── evaluate_ddinter.py             # External DDInter 2.0 validation (1,000 pairs)
├── evaluate_pharmgraph.py          # PharmGraph evaluation script
├── requirements.txt
└── requirements-test.txt
```

---

## Running Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

125 tests covering path retrieval, mechanism reasoning, brand name resolution, confidence scoring, and full pipeline integration.

---

## Data Sources

| Source | Version | Role |
|---|---|---|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | 37 | Drug-target bioactivity KG (IC50, Ki, Kd, EC50 assays) |
| [DDInter](https://ddinter2.scbdd.com) | 2.0 | Clinically curated DDI records + severity labels (76,000+ interactions) |
| [DDIMDL](https://github.com/YifanDengWHU/DDIMDL) | — | Deep learning baseline for ablation comparison |

---

## Tech Stack

| Component | Technology |
|---|---|
| Multi-agent orchestration | LangGraph (stateful graph) |
| LLM reasoning | LangChain + OpenAI GPT-4o-mini |
| Knowledge graph | NetworkX |
| Structured LLM outputs | Pydantic |
| UI | Streamlit |
| Testing | pytest (125 tests) |

---

## Limitations

- Requires an OpenAI API key (usage costs apply)
- KG coverage is bounded by ChEMBL 37 and DDInter 2.0 — novel drugs or rare interactions outside these sources will not be detected
- "No interaction found" means no evidence in the integrated sources — it does **not** guarantee safety
- Grounding check is lexical; semantic drift in LLM outputs may not always be caught
- Recall is intentionally conservative to minimise false positives in a clinical context

---

## License

MIT — see [LICENSE](LICENSE)
