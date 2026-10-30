# PharmGraph 
### An Agentic AI Pipeline for Drug-Drug Interaction Discovery

PharmGraph is a multi-agent AI system that detects and explains **drug-drug interactions (DDIs)** by combining a biomedical knowledge graph with grounded LLM reasoning. It cross-references **ChEMBL 37** drug-target data with the **DDInter 2.0** clinical interaction database, then uses a four-agent LangGraph pipeline to retrieve evidence, reason about mechanisms, validate confidence, and generate a clinical report.

---

## How It Works

```
User inputs two drugs
        ↓
[Agent 1] PathRetriever   — queries ChEMBL + DDInter KG for shared targets & known interactions
        ↓
[Agent 2] MechanismReasoner — LLM explains mechanism grounded in retrieved evidence only
        ↓
[Agent 3] EvidenceValidator — scores confidence; flags weak or unresolved reasoning
        ↓
[Agent 4] ReportGenerator  — produces a structured clinical report
```

- **DDInter severity is always authoritative** — the LLM never overrides a recorded clinical severity
- **Grounding check** — LLM output is verified to reference actual retrieved targets; unresolved mechanisms are flagged, not shown as confident conclusions
- **Brand name support** — accepts trade names (e.g. Lipitor, Coumadin) and maps them to INN names automatically

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/aiswaryanandan24-cell/pharmgraph.git
cd pharmgraph
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the project root (never commit this file):

```
OPENAI_API_KEY=sk-...
```

Or set it directly in your terminal session:

```bash
# Windows
set OPENAI_API_KEY=sk-...

# macOS / Linux
export OPENAI_API_KEY=sk-...
```

> ⚠️ Never hardcode your API key in source files or commit it to version control.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
pharmgraph/
├── app.py                        # Streamlit UI (main analysis page)
├── pages/
│   ├── 2_Evaluation.py           # Evaluation results dashboard
│   └── 3_About.py                # About / methodology page
├── agents/
│   ├── path_retriever.py         # Agent 1: ChEMBL KG query
│   ├── path_retriever_v2.py      # Agent 1 v2: ChEMBL + DDInter merged KG
│   ├── mechanism_reasoner.py     # Agent 2: LLM mechanism reasoning
│   ├── mechanism_reasoner_v2.py  # Agent 2 v2: DDInter-first, grounded
│   ├── evidence_validator.py     # Agent 3: Confidence scoring
│   └── report_generator.py      # Agent 4: Clinical report generation
├── pipeline/
│   └── ddi_pipeline.py           # LangGraph stateful pipeline
├── kg/
│   ├── build_graph.py            # KG construction from ChEMBL
│   └── parse_chembl.py           # ChEMBL data parser
├── data/
│   ├── drug_kg.pkl               # ChEMBL knowledge graph (not in repo — build locally)
│   ├── drug_kg_merged.pkl        # ChEMBL + DDInter merged KG (not in repo — build locally)
│   └── drug_kg_full_merged.pkl   # Full merged KG (not in repo — build locally)
├── tests/                        # 125 pytest unit tests
├── ablation_study.py             # 4-method ablation comparison
├── evaluate_ddinter.py           # External validation against DDInter 2.0
├── evaluate_pharmgraph.py        # PharmGraph evaluation script
├── requirements.txt
└── requirements-test.txt
```

> **Note on data files:** The `.pkl` knowledge graph files are not included in this repository due to size. Build them locally by running `python kg/build_graph.py` after downloading the ChEMBL and DDInter source data.

---

## Evaluation

PharmGraph was evaluated using two independent methods:

**Ablation study** — four methods compared on a 32-pair clinical benchmark:

| Method | Description |
|--------|-------------|
| Method 1 | Shared-target rule only (no LLM) |
| Method 2 | Single LLM call (no graph) |
| Method 3 | Graph + single LLM prompt |
| Method 4 | Full 4-agent pipeline (PharmGraph) |

**External validation** — evaluated against DDInter 2.0, a clinically curated DDI database that is entirely independent of ChEMBL, breaking the circularity of KG-sampled evaluation. Positive pairs are DDInter-listed interactions; negatives are drug pairs present in DDInter's vocabulary but with no recorded interaction.

Run evaluations:

```bash
python ablation_study.py
python evaluate_ddinter.py
python evaluate_pharmgraph.py
```

---

## Running Tests

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

125 tests covering path retrieval, mechanism reasoning, brand name resolution, and pipeline logic.

---

## Data Sources

| Source | Use |
|--------|-----|
| [ChEMBL 37](https://www.ebi.ac.uk/chembl/) | Drug-target bioactivity data for KG construction |
| [DDInter 2.0](https://ddinter2.scbdd.com) | Clinically curated DDI records + severity labels |
| [DDIMDL](https://github.com/YifanDengWHU/DDIMDL) | Baseline comparison model |

---

## Tech Stack

- **LangGraph** — stateful multi-agent orchestration
- **LangChain + OpenAI GPT-4o-mini** — grounded LLM reasoning
- **NetworkX** — knowledge graph storage and traversal
- **Streamlit** — interactive web UI
- **Pydantic** — structured, typed LLM outputs
- **pytest** — unit testing

---

## Limitations

- Requires an OpenAI API key (paid)
- KG coverage limited to drugs present in ChEMBL 37 and DDInter 2.0
- "No interaction found" means no evidence in these two sources — it does **not** mean the combination is safe
- LLM grounding check is lexical, not semantic

---

## License

MIT
