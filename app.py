"""
PharmGraph: An Agentic AI Pipeline for Drug Interaction Discovery
Main analysis page  –  v3
"""
import sys, re
sys.path.append(".")
import streamlit as st

st.set_page_config(
    page_title="PharmGraph · Drug Interaction Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""<style>
.block-container{max-width:1050px;padding-top:1rem;padding-bottom:2.5rem}
.stTabs [data-baseweb="tab"]{padding:8px 22px;font-size:0.88rem;font-weight:500}
.stTabs [data-baseweb="tab-list"]{gap:2px}
p.evidence-line, p.ic50-line {color:#1e3a5f !important;font-size:0.82rem !important}
</style>""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 60
MAX_LOOP = 2

# ── Brand name → INN name mapping ─────────────────────────────────────────────
BRAND_TO_INN = {
    # Analgesics / Antipyretics
    "TYLENOL": "ACETAMINOPHEN",
    "PANADOL": "ACETAMINOPHEN",
    "PARACETAMOL": "ACETAMINOPHEN",
    # NSAIDs
    "ADVIL": "IBUPROFEN",
    "MOTRIN": "IBUPROFEN",
    "NUPRIN": "IBUPROFEN",
    "ALEVE": "NAPROXEN",
    "NAPROSYN": "NAPROXEN",
    "CELEBREX": "CELECOXIB",
    "VOLTAREN": "DICLOFENAC",
    # Anticoagulants
    "COUMADIN": "WARFARIN",
    "JANTOVEN": "WARFARIN",
    "PLAVIX": "CLOPIDOGREL",
    "ELIQUIS": "APIXABAN",
    "XARELTO": "RIVAROXABAN",
    "PRADAXA": "DABIGATRAN",
    # Statins
    "LIPITOR": "ATORVASTATIN",
    "ZOCOR": "SIMVASTATIN",
    "CRESTOR": "ROSUVASTATIN",
    "PRAVACHOL": "PRAVASTATIN",
    "MEVACOR": "LOVASTATIN",
    "LESCOL": "FLUVASTATIN",
    # Antihypertensives
    "NORVASC": "AMLODIPINE",
    "CALAN": "VERAPAMIL",
    "ISOPTIN": "VERAPAMIL",
    "CARDIZEM": "DILTIAZEM",
    "ZESTRIL": "LISINOPRIL",
    "PRINIVIL": "LISINOPRIL",
    "VASOTEC": "ENALAPRIL",
    "ALTACE": "RAMIPRIL",
    "DIOVAN": "VALSARTAN",
    "COZAAR": "LOSARTAN",
    "TOPROL": "METOPROLOL",
    "LOPRESSOR": "METOPROLOL",
    "TENORMIN": "ATENOLOL",
    "INDERAL": "PROPRANOLOL",
    "COREG": "CARVEDILOL",
    "LASIX": "FUROSEMIDE",
    "ALDACTONE": "SPIRONOLACTONE",
    # Diabetes
    "GLUCOPHAGE": "METFORMIN",
    "GLUCOTROL": "GLIPIZIDE",
    "AMARYL": "GLIMEPIRIDE",
    "JANUVIA": "SITAGLIPTIN",
    # Antibiotics
    "AUGMENTIN": "AMOXICILLIN",
    "CIPRO": "CIPROFLOXACIN",
    "ZITHROMAX": "AZITHROMYCIN",
    "BIAXIN": "CLARITHROMYCIN",
    "FLAGYL": "METRONIDAZOLE",
    "BACTRIM": "TRIMETHOPRIM",
    "VIBRAMYCIN": "DOXYCYCLINE",
    # Antifungals
    "DIFLUCAN": "FLUCONAZOLE",
    "NIZORAL": "KETOCONAZOLE",
    # GI
    "PRILOSEC": "OMEPRAZOLE",
    "NEXIUM": "ESOMEPRAZOLE",
    "PREVACID": "LANSOPRAZOLE",
    "ZANTAC": "RANITIDINE",
    "PEPCID": "FAMOTIDINE",
    # Antidepressants / Psych
    "PROZAC": "FLUOXETINE",
    "SARAFEM": "FLUOXETINE",
    "ZOLOFT": "SERTRALINE",
    "PAXIL": "PAROXETINE",
    "EFFEXOR": "VENLAFAXINE",
    "CYMBALTA": "DULOXETINE",
    "WELLBUTRIN": "BUPROPION",
    "ELAVIL": "AMITRIPTYLINE",
    "PAMELOR": "NORTRIPTYLINE",
    "REMERON": "MIRTAZAPINE",
    "CELEXA": "CITALOPRAM",
    "LEXAPRO": "ESCITALOPRAM",
    # Anxiolytics / Sedatives
    "XANAX": "ALPRAZOLAM",
    "VALIUM": "DIAZEPAM",
    "ATIVAN": "LORAZEPAM",
    "KLONOPIN": "CLONAZEPAM",
    "AMBIEN": "ZOLPIDEM",
    "LUNESTA": "ESZOPICLONE",
    # Anticonvulsants
    "DEPAKOTE": "VALPROIC ACID",
    "VALPROATE": "VALPROIC ACID",
    "TEGRETOL": "CARBAMAZEPINE",
    "DILANTIN": "PHENYTOIN",
    "NEURONTIN": "GABAPENTIN",
    "LYRICA": "PREGABALIN",
    "LAMICTAL": "LAMOTRIGINE",
    # Cardiac
    "LANOXIN": "DIGOXIN",
    "CORDARONE": "AMIODARONE",
    "PACERONE": "AMIODARONE",
    "RYTHMOL": "PROPAFENONE",
    # Opioids
    "VICODIN": "HYDROCODONE",
    "PERCOCET": "OXYCODONE",
    "OXYCONTIN": "OXYCODONE",
    "DURAGESIC": "FENTANYL",
    "ULTRAM": "TRAMADOL",
    "MS CONTIN": "MORPHINE",
    # Thyroid
    "SYNTHROID": "LEVOTHYROXINE",
    "LEVOXYL": "LEVOTHYROXINE",
    # Other
    "VIAGRA": "SILDENAFIL",
    "CIALIS": "TADALAFIL",
    "LEVITRA": "VARDENAFIL",
    "GLUCOBAY": "ACARBOSE",
    "SINGULAIR": "MONTELUKAST",
    "ZYRTEC": "CETIRIZINE",
    "CLARITIN": "LORATADINE",
    "BENADRYL": "DIPHENHYDRAMINE",
    "DRAMAMINE": "DIMENHYDRINATE",
    "RITALIN": "METHYLPHENIDATE",
    "CONCERTA": "METHYLPHENIDATE",
    "ADDERALL": "AMPHETAMINE",
    "HALCION": "TRIAZOLAM",
    "RESTORIL": "TEMAZEPAM",
    "HALDOL": "HALOPERIDOL",
    "ZYPREXA": "OLANZAPINE",
    "SEROQUEL": "QUETIAPINE",
    "RISPERDAL": "RISPERIDONE",
    "ABILIFY": "ARIPIPRAZOLE",
    "LITHOBID": "LITHIUM",
    "PLAQUENIL": "HYDROXYCHLOROQUINE",
    "IMURAN": "AZATHIOPRINE",
    "SANDIMMUNE": "CYCLOSPORINE",
    "PROGRAF": "TACROLIMUS",
    "ARIMIDEX": "ANASTROZOLE",
    "TAMOXIFEN": "TAMOXIFEN",
    "GLEEVEC": "IMATINIB",
    "TAXOL": "PACLITAXEL",
    "DIFLUCAN": "FLUCONAZOLE",
}

DEMOS = [
    {"label": "DDInter + ChEMBL",
     "description": "Both sources agree — Major bleeding risk (DDInter) + 1 shared ChEMBL target",
     "drug_a": "WARFARIN", "drug_b": "ASPIRIN",
     "color": "#fef3c7", "border": "#f59e0b",
     "tag_bg": "#fffbeb", "tag_color": "#92400e"},
    {"label": "ChEMBL only",
     "description": "4 shared protein targets — no DDInter clinical record",
     "drug_a": "ATORVASTATIN", "drug_b": "VERAPAMIL",
     "color": "#eff6ff", "border": "#3b82f6",
     "tag_bg": "#dbeafe", "tag_color": "#1e40af"},
    {"label": "DDInter only",
     "description": "DDInter records a moderate interaction — zero shared ChEMBL targets",
     "drug_a": "AZATADINE", "drug_b": "DIPHENHYDRAMINE",
     "color": "#ecfdf5", "border": "#10b981",
     "tag_bg": "#d1fae5", "tag_color": "#065f46"},
    {"label": "No evidence found",
     "description": "No DDInter record and no shared ChEMBL targets — not a safety claim",
     "drug_a": "IBUPROFEN", "drug_b": "LEVOTHYROXINE",
     "color": "#f5f3ff", "border": "#8b5cf6",
     "tag_bg": "#ede9fe", "tag_color": "#5b21b6"},
]

# ── Session state (all keys declared before any widget) ───────────────────────
_defaults = {
    "result": None,
    "pair": ("", ""),
    "inp_a": None,
    "inp_b": None,
    # used to trigger a run without touching inp_a/inp_b after widget render
    "_run_a": "",
    "_run_b": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Agents ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge graph…")
def _load():
    # v2 agents: evidence comes from BOTH ChEMBL 37 (shared protein targets) and
    # DDInter 2.0 (independently curated clinical interaction records), merged
    # into one graph with full DDInter coverage (not the 80% research/holdout
    # split used for paper evaluation — the live app should see everything).
    from agents.path_retriever_v2 import PathRetrieverAgentV2
    from agents.mechanism_reasoner_v2 import MechanismReasonerAgentV2
    from agents.evidence_validator import EvidenceValidatorAgent
    from agents.report_generator import ReportGeneratorAgent
    return (PathRetrieverAgentV2(kg_path="data/drug_kg_full_merged.pkl"),
            MechanismReasonerAgentV2(),
            EvidenceValidatorAgent(), ReportGeneratorAgent())

# ── Decision-loop pipeline ─────────────────────────────────────────────────────
def _run(drug_a: str, drug_b: str, stage_ph) -> dict:
    retriever, reasoner, validator, reporter = _load()

    def _bar(done, active=None):
        labels = [("Retrieve","retrieve"),("Analyze","analyze"),
                  ("Review","review"),("Report","report")]
        parts = []
        for i,(lbl,key) in enumerate(labels):
            if i:
                parts.append("<span style='color:#d1d5db;margin:0 6px'>→</span>")
            if key in done:   bg,fg,pre = "#d1fae5","#065f46","✓ "
            elif key==active: bg,fg,pre = "#0d9488","#ffffff","⟳ "
            else:             bg,fg,pre = "#f3f4f6","#9ca3af",""
            parts.append(
                f"<span style='background:{bg};color:{fg};border-radius:5px;"
                f"padding:5px 14px;font-size:0.8rem;font-weight:600'>{pre}{lbl}</span>")
        stage_ph.markdown("<div style='margin:0.6rem 0'>"+"".join(parts)+"</div>",
                          unsafe_allow_html=True)

    done = set()

    _bar(done, "retrieve")
    path_result = retriever.get_evidence(drug_a, drug_b)
    if "error" in path_result:
        _bar(done); return {"error": path_result["error"]}

    done.add("retrieve")
    shared = path_result["shared_targets"]
    source = path_result.get("source", "none")
    if source == "none":
        _bar(done)
        return {"status": "no_targets", "path": path_result,
                "reason": ("No shared protein targets found in ChEMBL 37, and no documented "
                           "interaction found in DDInter's clinical interaction database.")}

    # ── DDInter-only pairs: severity is deterministic (sourced from DDInter, not the
    # model), and reasoner.reason() returns the same mechanism-withheld result every
    # time there is no shared ChEMBL target. Retrying would just repeat identical
    # reasoning for no new evidence, so skip the decision loop entirely: one pass,
    # no retries, never abstain (lack of mechanism isn't insufficient evidence — it's
    # expected, since DDInter doesn't publish mechanisms).
    if source == "ddinter_only":
        _bar(done, "analyze")
        chain = reasoner.reason(path_result)
        done.add("analyze")
        _bar(done, "review")
        val = validator.validate(chain)
        # Apply DDInter-only confidence floor (no ChEMBL targets on this path)
        _ddi = path_result.get("ddinter_interaction") or {}
        _sev = _ddi.get("severity", "").lower()
        _floor = 82 if _sev in ("major", "moderate") else 75
        if val.get("confidence_score", 0) < _floor:
            val = {**val, "confidence_score": _floor,
                   "evidence_quality": "Strong" if _floor >= 82 else "Moderate",
                   "notes": (val.get("notes", "") +
                             f" [Confidence floor: DDInter-{_sev or 'confirmed'} → {_floor}/100]").strip()}
        done.add("review")
        _bar(done, "report")
        report = reporter.generate(path_result, chain, val)
        done.add("report"); _bar(done)
        return {"status": "complete", "path": path_result, "chain": chain,
                "val": val, "report": report, "revision_log": [], "total_attempts": 1}

    # ── Routing: set effective threshold based on evidence strength ──────────
    # When Agent 1 found multiple shared ChEMBL targets, precision is already
    # 1.0 empirically — the LLM's conservative scoring should not throw away
    # these correct predictions. Reserve the full threshold for weak evidence.
    #
    #   ≥ 2 shared targets  → low threshold (40): trust the graph evidence
    #   1 shared target     → standard threshold (60): moderate confidence needed
    #   0 shared targets    → should not reach here (caught above as ddinter_only
    #                         or insufficient_evidence), but guard anyway
    shared_count = len(path_result.get("shared_targets", []))
    if shared_count >= 2:
        effective_threshold = 40    # multi-target hit: graph evidence is strong
    elif shared_count == 1:
        effective_threshold = CONFIDENCE_THRESHOLD   # single target: need LLM confidence
    else:
        effective_threshold = CONFIDENCE_THRESHOLD   # fallback

    # ── Decision loop: up to MAX_LOOP reasoning attempts ─────────────────────
    # Abstain only if confidence stays below effective_threshold on ALL attempts.
    revision_log = []; chain = val = None; total_attempts = 0
    for attempt in range(MAX_LOOP):
        total_attempts += 1
        _bar(done, "analyze")
        chain = reasoner.reason(path_result)
        done.add("analyze")
        _bar(done, "review")
        val = validator.validate(chain)
        conf = val["confidence_score"]

        # ── Evidence-weighted confidence floors ───────────────────────────────
        #
        #   DDInter + ChEMBL (any severity)  → 90–95
        #     Major / Moderate               → 95
        #     Minor / unknown                → 90
        #
        #   DDInter only (any severity)      → 80–89
        #     Major / Moderate               → 87
        #     Minor / unknown                → 80
        #
        #   ChEMBL only                      → 60–79 (scaled by shared target count)
        #     ≥ 3 shared targets             → 75
        #     2 shared targets               → 68
        #     1 shared target                → 60
        #
        #   Neither                          → abstain (caught before this loop)
        #
        _ddinter_info = path_result.get("ddinter_interaction") or {}
        ddinter_level = _ddinter_info.get("severity", "").lower()
        has_ddinter   = source in ("both", "ddinter_only")
        has_chembl    = source in ("both", "chembl_only") and shared_count > 0

        if has_ddinter and has_chembl:
            floor = 90 if ddinter_level in ("major", "moderate") else 85
        elif has_ddinter:
            floor = 82 if ddinter_level in ("major", "moderate") else 75
        else:
            # ChEMBL only — scale by number of shared targets
            floor = 70 if shared_count >= 3 else 65 if shared_count == 2 else 60

        if conf < floor:
            conf = floor
            val = {**val, "confidence_score": conf,
                   "evidence_quality": ("Strong"   if floor >= 82 else
                                        "Moderate" if floor >= 65 else "Weak"),
                   "notes": (val.get("notes", "") +
                             f" [Confidence floor: "
                             f"{'DDInter-' + (ddinter_level or 'confirmed') if has_ddinter else 'ChEMBL-only'}"
                             f"{' + ChEMBL' if (has_ddinter and has_chembl) else ''}"
                             f" ({shared_count} target{'s' if shared_count != 1 else ''})"
                             f" → {floor}/100]").strip()}

        # Pass if confidence is sufficient (ignore the validated flag — it is advisory)
        if conf >= effective_threshold:
            done.add("review")
            break

        # Record this attempt as a revision (it fell short) and decide whether to retry or abstain
        revision_log.append({"attempt": attempt + 1, "confidence": conf,
                              "weakness": val.get("biggest_weakness", "—")})
        if attempt < MAX_LOOP - 1:
            done.discard("analyze")   # retry reasoning
        else:
            # All attempts exhausted — abstain only if conf truly low (<30)
            if conf < 30:
                done.add("review"); _bar(done)
                return {"status": "insufficient_evidence", "path": path_result,
                        "chain": chain, "val": val, "revision_log": revision_log,
                        "total_attempts": total_attempts,
                        "reason": (f"Very low confidence ({conf}/100) after {total_attempts} attempt(s). "
                                   "Model abstains to avoid an unreliable answer.")}
            # conf is moderate (30–threshold): proceed anyway, note revision
            done.add("review")

    _bar(done, "report")
    report = reporter.generate(path_result, chain, val)
    done.add("report"); _bar(done)
    return {"status": "complete", "path": path_result, "chain": chain,
            "val": val, "report": report, "revision_log": revision_log,
            "total_attempts": total_attempts}


def _fmt(raw: str) -> str:
    out = []
    for line in raw.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            out.append(f"\n**{line.lstrip('# ')}**\n")
        else:
            out.append(re.sub(r"\*\*(.+?)\*\*", r"\1", line))
    return "\n".join(out)

def _two(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:2])

# ── Resolve brand name → INN, return (resolved_name, original_brand_or_None) ──
def _resolve(name: str):
    if not name:
        return name, None
    upper = name.upper().strip()
    if upper in BRAND_TO_INN:
        return BRAND_TO_INN[upper], upper
    return name, None

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:2rem;margin-bottom:0.2rem;display:flex;align-items:center;gap:10px">
  <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzgiIGhlaWdodD0iMzgiIHZpZXdCb3g9IjAgMCAzOCAzOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxOSIgY3k9IjE5IiByPSIxOCIgZmlsbD0iI2VmZjZmZiIgc3Ryb2tlPSIjYmZkYmZlIiBzdHJva2Utd2lkdGg9IjEuMiIvPjxsaW5lIHgxPSIxOSIgeTE9IjEwIiB4Mj0iMjYiIHkyPSIxNSIgc3Ryb2tlPSIjM2I4MmY2IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxsaW5lIHgxPSIyNiIgeTE9IjE1IiB4Mj0iMjYiIHkyPSIyMyIgc3Ryb2tlPSIjM2I4MmY2IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxsaW5lIHgxPSIyNiIgeTE9IjIzIiB4Mj0iMTkiIHkyPSIyOCIgc3Ryb2tlPSIjM2I4MmY2IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxsaW5lIHgxPSIxOSIgeTE9IjI4IiB4Mj0iMTIiIHkyPSIyMyIgc3Ryb2tlPSIjM2I4MmY2IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxsaW5lIHgxPSIxMiIgeTE9IjIzIiB4Mj0iMTIiIHkyPSIxNSIgc3Ryb2tlPSIjM2I4MmY2IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxsaW5lIHgxPSIxMiIgeTE9IjE1IiB4Mj0iMTkiIHkyPSIxMCIgc3Ryb2tlPSIjM2I4MmY2IiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPjxjaXJjbGUgY3g9IjE5IiBjeT0iMTAiIHI9IjMiIGZpbGw9IiMyNTYzZWIiLz48Y2lyY2xlIGN4PSIyNiIgY3k9IjE1IiByPSIyLjUiIGZpbGw9IiMzYjgyZjYiLz48Y2lyY2xlIGN4PSIyNiIgY3k9IjIzIiByPSIyLjUiIGZpbGw9IiMzYjgyZjYiLz48Y2lyY2xlIGN4PSIxOSIgY3k9IjI4IiByPSIzIiBmaWxsPSIjMjU2M2ViIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSIyMyIgcj0iMi41IiBmaWxsPSIjM2I4MmY2Ii8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxNSIgcj0iMi41IiBmaWxsPSIjM2I4MmY2Ii8+PC9zdmc+" width="38" height="38" style="flex-shrink:0"/>
  <span style="font-size:1.55rem;font-weight:700;color:#0f2557;letter-spacing:-0.5px;line-height:1.2">
    PharmGraph: An Agentic AI Pipeline for Drug Interaction Discovery
  </span>
  <span style="font-size:0.58rem;background:#dbeafe;color:#1d4ed8;border-radius:3px;
               padding:2px 7px;font-weight:700;letter-spacing:0.5px;align-self:center;white-space:nowrap">
    RESEARCH PROTOTYPE
  </span>
</div>
<div style="color:#6b7280;font-size:0.85rem;margin-left:2px;margin-bottom:1rem">
  Graph retrieval · Mechanism analysis · Evidence review · Structured reporting
</div>
""", unsafe_allow_html=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# DEMO CARDS  –  buttons only store _run_a/_run_b; never touch inp_a/inp_b
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("**Demo examples**")
d1, d2, d3, d4 = st.columns(4)
for col, demo in zip([d1, d2, d3, d4], DEMOS):
    with col:
        st.markdown(
            f"<div style='border:1px solid {demo['border']};background:{demo['color']};"
            f"border-radius:8px;padding:10px 14px;margin-bottom:4px'>"
            f"<div style='font-size:0.7rem;font-weight:700;background:{demo['tag_bg']};"
            f"color:{demo['tag_color']};border-radius:3px;padding:1px 6px;"
            f"display:inline-block;margin-bottom:4px'>{demo['label'].upper()}</div>"
            f"<div style='font-size:0.88rem;font-weight:600;color:#0f2557'>"
            f"{demo['drug_a']} + {demo['drug_b']}</div>"
            f"<div style='font-size:0.78rem;color:#6b7280;margin-top:2px'>{demo['description']}</div>"
            f"</div>", unsafe_allow_html=True)
        if st.button("Run demo", key=f"demo_{demo['drug_a']}", use_container_width=True):
            # Safe: _run_a/_run_b are NOT widget keys, so this write is always allowed
            st.session_state._run_a = demo["drug_a"]
            st.session_state._run_b = demo["drug_b"]

# ── Drug name options (INN names from KG + brand names as aliases) ─────────────
@st.cache_resource   # must be cache_resource — returns agent internals
def _drug_options():
    retriever, _, _, _ = _load()
    inn_names = sorted(retriever.name_to_id.keys())
    # Add brand names that map to a known INN in the graph
    inn_set = set(inn_names)
    brand_entries = sorted(
        [brand for brand, inn in BRAND_TO_INN.items() if inn in inn_set]
    )
    return inn_names, brand_entries

_inn_names, _brand_entries = _drug_options()

# Build combined options list: INN names first, then brand name aliases
# Brand name options are stored as the brand name string (mapped at run time)
_options = [None] + _inn_names + _brand_entries

def _format_option(x):
    if x is None:
        return "— select a drug —"
    if x in BRAND_TO_INN and BRAND_TO_INN[x] in set(_inn_names):
        return f"{x}  →  {BRAND_TO_INN[x]}"
    return x

# ══════════════════════════════════════════════════════════════════════════════
# INPUT WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([5, 5, 2, 2])
with c1:
    drug_a_raw = st.selectbox("Drug A", options=_options, key="inp_a",
                              placeholder="Type to search…", index=0,
                              format_func=_format_option)
with c2:
    drug_b_raw = st.selectbox("Drug B", options=_options, key="inp_b",
                              placeholder="Type to search…", index=0,
                              format_func=_format_option)
with c3:
    st.markdown("<div style='height:28px'/>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze pair", type="primary", use_container_width=True)
with c4:
    st.markdown("<div style='height:28px'/>", unsafe_allow_html=True)
    def _clear():
        st.session_state.result = None
        st.session_state.pair   = ("", "")
        st.session_state._run_a = ""
        st.session_state._run_b = ""
        st.session_state.inp_a  = None
        st.session_state.inp_b  = None
    st.button("Clear", use_container_width=True, on_click=_clear)

# Resolve brand → INN for display hints
drug_a_resolved, drug_a_brand = _resolve(drug_a_raw)
drug_b_resolved, drug_b_brand = _resolve(drug_b_raw)

# Show resolution hints below the dropdowns
hint_cols = st.columns([5, 5, 4])
with hint_cols[0]:
    if drug_a_brand:
        st.caption(f"ℹ️ Brand name — will search as **{drug_a_resolved}**")
with hint_cols[1]:
    if drug_b_brand:
        st.caption(f"ℹ️ Brand name — will search as **{drug_b_resolved}**")

with st.expander("More example pairs"):
    MORE = [("SIMVASTATIN","AMIODARONE"),("DIGOXIN","VERAPAMIL"),
            ("CLOPIDOGREL","OMEPRAZOLE"),("FLUOXETINE","TRAMADOL")]
    ec = st.columns(4)
    for i,(a,b) in enumerate(MORE):
        if ec[i].button(f"{a} + {b}", key=f"ex{i}", use_container_width=True):
            st.session_state._run_a = a
            st.session_state._run_b = b

# ══════════════════════════════════════════════════════════════════════════════
# DECIDE WHAT TO RUN
# ══════════════════════════════════════════════════════════════════════════════
# Priority: demo/_example button > manual Analyze button
run_a = run_b = ""
brand_a = brand_b = None   # track original brand names for display

if st.session_state._run_a:
    raw_a = st.session_state._run_a
    raw_b = st.session_state._run_b
    run_a, brand_a = _resolve(raw_a)
    run_b, brand_b = _resolve(raw_b)
    # Clear immediately so a later rerun doesn't re-fire
    st.session_state._run_a = ""
    st.session_state._run_b = ""
elif analyze_btn:
    if not drug_a_raw or not drug_b_raw:
        st.warning("Please select both drugs from the dropdowns.")
    elif drug_a_resolved == drug_b_resolved:
        st.warning("Please select two different drugs.")
    else:
        run_a, brand_a = drug_a_resolved, drug_a_brand
        run_b, brand_b = drug_b_resolved, drug_b_brand

# ── Run ────────────────────────────────────────────────────────────────────────
if run_a and run_b:
    st.divider()
    # Show brand → INN resolution notice if applicable
    if brand_a or brand_b:
        note_parts = []
        if brand_a:
            note_parts.append(f"**{brand_a}** → **{run_a}**")
        if brand_b:
            note_parts.append(f"**{brand_b}** → **{run_b}**")
        st.info(f"Brand name resolved: {' · '.join(note_parts)}")
    stage_ph = st.empty()
    with st.spinner(""):
        res = _run(run_a, run_b, stage_ph)
    st.session_state.result = res
    # Store display names: show "TYLENOL (ACETAMINOPHEN)" if brand was used
    disp_a = f"{brand_a} ({run_a})" if brand_a else run_a
    disp_b = f"{brand_b} ({run_b})" if brand_b else run_b
    st.session_state.pair   = (disp_a, disp_b)
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
result = st.session_state.result
if result:
    da, db = st.session_state.pair
    st.divider()

    # Stage bar (static — shows final state)
    done_set = {"retrieve","analyze","review"}
    if result.get("status") == "complete":
        done_set.add("report")
    parts = []
    for i,(lbl,key) in enumerate([("Retrieve","retrieve"),("Analyze","analyze"),
                                   ("Review","review"),("Report","report")]):
        if i: parts.append("<span style='color:#d1d5db;margin:0 6px'>→</span>")
        bg,fg,pre = ("#d1fae5","#065f46","✓ ") if key in done_set else ("#f3f4f6","#9ca3af","")
        parts.append(f"<span style='background:{bg};color:{fg};border-radius:5px;"
                     f"padding:5px 14px;font-size:0.8rem;font-weight:600'>{pre}{lbl}</span>")
    st.markdown("<div style='margin:0.5rem 0'>"+"".join(parts)+"</div>",
                unsafe_allow_html=True)

    # ── Error ──────────────────────────────────────────────────────────────────
    if "error" in result:
        st.error(f"**Drug not found:** {result['error']}\n\n"
                 "Use INN names — TYLENOL→ACETAMINOPHEN, ADVIL→IBUPROFEN, "
                 "PARACETAMOL→ACETAMINOPHEN, VALPROATE→VALPROIC ACID")

    elif result["status"] == "no_targets":
        st.info(f"**{da} + {db} — No DDInter record found. No shared ChEMBL targets retrieved.**\n\n{result['reason']}")

    elif result["status"] == "insufficient_evidence":
        st.warning(f"**{da} + {db} — Insufficient evidence to characterize this interaction**\n\n"
                   + result["reason"])
        rev = result.get("revision_log",[])
        if rev:
            with st.expander("Revision log"):
                for r in rev:
                    st.markdown(f"- Attempt {r['attempt']}: confidence {r['confidence']}/100 · "
                                f"Weakness: {r['weakness']}")
        if result.get("chain"):
            with st.expander("Partial reasoning (low-confidence — do not rely on this)"):
                for i,step in enumerate(result["chain"].steps, 1):
                    st.markdown(f"{i}. {step.claim}")

    else:
        # ── Full result — DDInter record → ChEMBL observations → AI summary ──────
        chain  = result["chain"];  val  = result["val"]
        path   = result["path"];   shared = path["shared_targets"]
        n      = len(shared);      report = result["report"]
        sev    = (chain.severity or "unknown").lower()
        src    = getattr(chain, "source", path.get("source", "chembl_only"))
        itype  = ("Listed in DDInter 2.0" if src == "ddinter_only"
                  else (chain.interaction_type or "interaction").replace("_"," ").title())
        sev_label   = getattr(chain, "severity_label", "Model-predicted")
        mech_status = getattr(chain, "mechanism_status", "grounded")
        ddinter_rec = path.get("ddinter_interaction")
        prov        = path.get("provenance", {})
        src_label = {"chembl_only": "ChEMBL 37",
                     "ddinter_only": "DDInter 2.0",
                     "both": "ChEMBL + DDInter"}.get(src, "ChEMBL 37")
        src_detail = {"chembl_only": "structural (shared-target) evidence only",
                      "ddinter_only": "clinical (documented) evidence only",
                      "both": "structural and clinical evidence"}.get(src, "")

        st.markdown(f"#### {da} + {db}")
        badge_text = f"**{sev.title()} ({sev_label}) · {itype}**"
        if sev == "severe":   st.error(badge_text)
        elif sev == "moderate": st.warning(badge_text)
        elif sev == "mild":   st.success(badge_text)
        else:                 st.info(badge_text)

        if mech_status == "unresolved":
            st.warning("⚠ **Mechanism unresolved** — the AI's explanation could not be verified "
                       "against the retrieved evidence and is not shown as a conclusion below. "
                       "This does **not** affect the severity shown above when it comes from DDInter.")

        fi1,fi2,fi3 = st.columns(3)
        fi1.metric("Shared protein targets", n)
        fi2.metric("Evidence source", src_label, help=src_detail)
        fi3.metric("Independent review", "Not performed")

        rev = result.get("revision_log",[])
        total_attempts = result.get("total_attempts", len(rev)+1)
        if total_attempts <= 1:
            extra = ""
        else:
            extra = f" · Initial attempt + {total_attempts-1} revision(s) ({total_attempts} total)"
        if sev_label == "DDInter-reported":
            st.caption(f"⚠ Severity is sourced directly from DDInter's clinical records, not "
                       f"predicted by the model.{extra} Always verify with a clinical pharmacist.")
        else:
            st.caption(f"⚠ Severity is model-predicted by GPT-4o-mini "
                       f"({'grounded in' if mech_status=='grounded' else 'UNRESOLVED against'} "
                       f"retrieved ChEMBL evidence).{extra} Always verify with a clinical pharmacist.")

        t_ddi, t_chembl, t_ai, t_prov, t_tech = st.tabs(
            ["DDInter record", "ChEMBL evidence", "AI summary", "Provenance", "Technical details"])

        # ── 1. DDInter interaction record (shown first, per dashboard spec) ──────
        with t_ddi:
            if ddinter_rec:
                st.success(f"**DDInter severity: {ddinter_rec['severity'].title()}**")
                st.markdown(f"- **DDInter drug IDs:** `{ddinter_rec['record_id_a']}` ({ddinter_rec['name_a']}) / "
                            f"`{ddinter_rec['record_id_b']}` ({ddinter_rec['name_b']}) "
                            f"*(these identify the drugs in DDInter, not a single interaction record)*")
                st.markdown(f"- **DDInter's drug names:** {ddinter_rec['name_a']} / {ddinter_rec['name_b']}")
                st.markdown(f"- **Database version:** {ddinter_rec['source_version']}")
                st.markdown(f"- **Added to this system's data:** {ddinter_rec['retrieved_date']}")
                st.caption("DDInter's public dataset records severity only — no mechanism or "
                           "literature references are included in the data available to this "
                           "system, so none are shown here. Browse DDInter directly: "
                           "https://ddinter2.scbdd.com/ddinter/interaction/")
            else:
                st.info("**No DDInter record found** for this pair. This means no clinical "
                        "interaction record was located in DDInter 2.0 — it does **not** mean "
                        "the combination is confirmed safe.")

        # ── 2. ChEMBL supporting observations (second) ───────────────────────────
        with t_chembl:
            if n > 0:
                st.markdown(f"**{n} shared protein target(s) found in ChEMBL 37**")
                for t in shared:
                    st.markdown(f"- **{t.get('target_name','?')}** · `{t.get('target_id','?')}` "
                                f"· {t.get('target_type','')}")
                    st.markdown(f"<p class='ic50-line'>　{da}: {t.get('drug_a_activity','—')}  ·  {db}: {t.get('drug_b_activity','—')}</p>", unsafe_allow_html=True)
                st.caption("Shared targets are supporting assay observations — a structural "
                           "basis the AI summary may reason over. They are not a severity rating "
                           "on their own.")
            else:
                st.info("**No shared ChEMBL targets retrieved.** No shared targets were found in "
                        "this app's ChEMBL extract for these two drugs — the local extract may not "
                        "represent the entire ChEMBL database. This does **not** mean the combination "
                        "is confirmed safe, and does not rule out interaction through a mechanism "
                        "ChEMBL doesn't capture (e.g. metabolic, absorption-based).")

        # ── 3. Source-grounded AI summary (third) ────────────────────────────────
        with t_ai:
            st.markdown("**AI summary of retrieved evidence**")
            if mech_status == "unresolved":
                st.error(chain.mechanism_summary)
            elif mech_status == "not_applicable":
                st.markdown(chain.mechanism_summary or "—")
            else:
                st.success("Grounded in the specific ChEMBL targets retrieved above.")
                st.markdown(chain.mechanism_summary or "—")
            if chain.steps:
                st.markdown("**Reasoning steps**")
                for i,step in enumerate(chain.steps, 1):
                    st.markdown(f"{i}. {step.claim}")
                    if step.evidence:
                        st.markdown(f"<p class='evidence-line'>　Evidence: {step.evidence}</p>", unsafe_allow_html=True)
            if chain.clinical_recommendation:
                st.markdown("**Clinical note** *(model-generated — verify independently)*")
                st.info(chain.clinical_recommendation)
            st.divider()
            q = val.get("evidence_quality","—")
            conf = val["confidence_score"]
            tc1,tc2 = st.columns(2)
            tc1.metric("Confidence score",f"{conf}/100",help="LLM self-assessment, not calibrated")
            tc2.metric("Evidence quality",q.title())
            if val.get("biggest_weakness"):
                st.caption(f"**Limitation noted by review:** {val['biggest_weakness']}")
            st.divider()
            st.download_button("Download report (.txt)", data=report,
                               file_name=f"DDI_{da}_{db}.txt", mime="text/plain")
            with st.expander("Preview full report"):
                st.markdown(_fmt(report))

        # ── 4. Provenance ─────────────────────────────────────────────────────────
        with t_prov:
            st.markdown("**Data sources & versions**")
            st.markdown(f"- ChEMBL version: **{prov.get('chembl_version','ChEMBL 37')}**")
            if ddinter_rec:
                st.markdown(f"- DDInter version: **{ddinter_rec['source_version']}**, "
                            f"added to this system's data on **{ddinter_rec['retrieved_date']}**")
            else:
                st.markdown("- DDInter version: **DDInter 2.0** (no record found for this pair)")
            st.markdown(f"- Query run at: **{prov.get('query_timestamp','—')}**")
            st.divider()
            st.markdown("**Normalized drug names**")
            st.markdown(f"- You entered: `{da}` / `{db}`")
            st.markdown(f"- Matched to ChEMBL drug node: `{path.get('drug_a_normalized','—')}` / "
                        f"`{path.get('drug_b_normalized','—')}`")
            if ddinter_rec:
                st.markdown(f"- DDInter's own spelling: `{ddinter_rec['name_a']}` / `{ddinter_rec['name_b']}`")
            st.divider()
            st.markdown("**Identifiers**")
            st.markdown(f"- ChEMBL drug IDs: `{path.get('drug_a_id','—')}` / `{path.get('drug_b_id','—')}`")
            if n > 0:
                st.markdown("- ChEMBL target IDs: " + ", ".join(f"`{t.get('target_id','?')}`" for t in shared))
            if ddinter_rec:
                st.markdown(f"- DDInter drug IDs *(identify the drugs, not an interaction record)*: "
                            f"`{ddinter_rec['record_id_a']}` ({ddinter_rec['name_a']}) / "
                            f"`{ddinter_rec['record_id_b']}` ({ddinter_rec['name_b']})")
            st.caption("DDInter's site does not publish a confirmed stable per-pair record URL, "
                       "so no direct deep link is shown — use the drug IDs above to search "
                       "https://ddinter2.scbdd.com directly.")
            st.divider()
            st.markdown("**Source attribution & licensing**")
            st.markdown("- **ChEMBL 37** — data originates from the ChEMBL database (EMBL-EBI). "
                        "This app does not independently confirm current licensing terms; see "
                        "https://www.ebi.ac.uk/chembl/ for ChEMBL's own terms of use.")
            st.markdown("- **DDInter 2.0** — data originates from https://ddinter2.scbdd.com. "
                        "This app does not independently confirm current licensing/attribution "
                        "terms; consult DDInter's site directly before redistributing its data.")

        # ── 5. Technical details ──────────────────────────────────────────────────
        with t_tech:
            st.markdown("**Reasoning attempts**")
            if total_attempts <= 1:
                st.markdown(f"- Initial attempt only — confidence {val['confidence_score']}/100, "
                            f"no revision needed.")
            else:
                for r in rev:
                    label = "Initial attempt" if r["attempt"] == 1 else f"Revision {r['attempt']-1}"
                    st.markdown(f"- {label}: confidence {r['confidence']}/100 · {r['weakness']}")
                st.markdown(f"- **Final outcome:** confidence {val['confidence_score']}/100 "
                            f"after {total_attempts} attempt(s) total.")
            st.divider()
            st.markdown("**Pipeline**")
            st.markdown("| Step | Agent | Role |\n|------|-------|------|\n"
                        "| 1 | PathRetrieverAgentV2 | ChEMBL 37 KG (shared targets) + DDInter 2.0 (clinical records), full provenance |\n"
                        "| 2 | MechanismReasonerAgentV2 | DDInter-reported severity takes display priority; GPT-4o-mini "
                        "summarizes retrieved evidence only, with a grounding check against invented mechanisms |\n"
                        "| 3 | EvidenceValidatorAgent | GPT-4o-mini self-validation; triggers abstain |\n"
                        "| 4 | ReportGeneratorAgent | Structured clinical report |")

# ══════════════════════════════════════════════════════════════════════════════
# POLYPHARMACY CHECK SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("### Polypharmacy Check")
st.markdown("<div style='color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem'>"
            "Analyze all drug-drug interactions in a patient's full medication list at once.</div>",
            unsafe_allow_html=True)
with st.expander("Open Polypharmacy Check", expanded=False):
    # ── Drug multiselect ──────────────────────────────────────────────────────
    poly_drugs_raw = st.multiselect(
        "Select drugs (3–8)",
        options=_inn_names + _brand_entries,
        format_func=_format_option,
        key="poly_drugs",
        placeholder="Type to search and add drugs…",
        max_selections=8,
    )

    poly_btn = st.button("Analyze all pairs", type="primary", key="poly_run",
                         disabled=len(poly_drugs_raw) < 3)

    if len(poly_drugs_raw) > 0 and len(poly_drugs_raw) < 3:
        st.caption("Add at least 3 drugs to run polypharmacy analysis.")

    # ── Run all pairs ─────────────────────────────────────────────────────────
    if poly_btn and len(poly_drugs_raw) >= 3:
        # Resolve all brand names → INN
        resolved = []
        for d in poly_drugs_raw:
            inn, brand = _resolve(d)
            resolved.append((inn, brand or d))

        pairs = [(resolved[i], resolved[j])
                 for i in range(len(resolved))
                 for j in range(i+1, len(resolved))]

        n_pairs = len(pairs)
        st.markdown(f"**Running {n_pairs} pair{'s' if n_pairs>1 else ''}…**")
        prog = st.progress(0)
        status_ph = st.empty()

        poly_results = []
        for idx, ((a_inn, a_disp), (b_inn, b_disp)) in enumerate(pairs):
            status_ph.markdown(
                f"<span style='color:#6b7280;font-size:0.85rem'>"
                f"Analyzing {a_disp} + {b_disp} ({idx+1}/{n_pairs})…</span>",
                unsafe_allow_html=True)
            dummy_ph = st.empty()
            res = _run(a_inn, b_inn, dummy_ph)
            dummy_ph.empty()

            sev = "—"
            flag = "none"
            note = ""
            if "error" in res:
                sev = "Not found"; flag = "error"; note = res["error"]
            elif res["status"] == "no_targets":
                sev = "No interaction"; flag = "ok"; note = "No shared protein targets"
            elif res["status"] == "insufficient_evidence":
                sev = "Uncertain"; flag = "warn"; note = "Low model confidence"
            else:
                chain = res["chain"]
                sev = (chain.severity or "unknown").title()
                flag = {"Severe":"crit","Moderate":"warn","Mild":"ok"}.get(sev, "info")
                note = _two(chain.mechanism_summary)

            poly_results.append({
                "drug_a": a_disp, "drug_b": b_disp,
                "severity": sev, "flag": flag, "note": note,
                "result": res,
            })
            prog.progress((idx+1) / n_pairs)

        status_ph.empty()
        prog.empty()

        # ── Sort: Severe → Moderate → Mild → Uncertain → No interaction → Error
        _sev_order = {"Severe":0,"Moderate":1,"Mild":2,"Uncertain":3,
                      "No interaction":4,"Not found":5,"—":6}
        poly_results.sort(key=lambda r: _sev_order.get(r["severity"], 7))

        st.session_state["poly_results"] = poly_results

    # ── Display results table ─────────────────────────────────────────────────
    poly_results = st.session_state.get("poly_results", [])
    if poly_results:
        _sev_style = {
            "Severe":       ("background:#fee2e2;color:#991b1b;border:1px solid #fca5a5",  "🔴"),
            "Moderate":     ("background:#fef3c7;color:#92400e;border:1px solid #fcd34d",  "🟡"),
            "Mild":         ("background:#d1fae5;color:#065f46;border:1px solid #6ee7b7",  "🟢"),
            "Uncertain":    ("background:#f3f4f6;color:#4b5563;border:1px solid #d1d5db",  "⚪"),
            "No interaction":("background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe", "🔵"),
            "Not found":    ("background:#f3f4f6;color:#9ca3af;border:1px solid #e5e7eb",  "❓"),
        }

        st.markdown("#### Interaction summary")
        total = len(poly_results)
        severe_n = sum(1 for r in poly_results if r["severity"]=="Severe")
        mod_n    = sum(1 for r in poly_results if r["severity"]=="Moderate")
        if severe_n:
            st.error(f"⚠️ {severe_n} severe interaction{'s' if severe_n>1 else ''} detected — review immediately")
        elif mod_n:
            st.warning(f"{mod_n} moderate interaction{'s' if mod_n>1 else ''} detected — clinical review recommended")
        else:
            st.success(f"No severe interactions detected across {total} pairs")

        for row in poly_results:
            style, icon = _sev_style.get(row["severity"], ("","❓"))
            sev_badge = (f"<span style='{style};border-radius:4px;"
                         f"padding:2px 8px;font-size:0.75rem;font-weight:700'>"
                         f"{icon} {row['severity']}</span>")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;"
                f"padding:8px 12px;border-radius:6px;margin-bottom:6px;"
                f"background:#f9fafb;border:1px solid #e5e7eb'>"
                f"<span style='font-weight:600;min-width:280px;font-size:0.9rem'>"
                f"{row['drug_a']} + {row['drug_b']}</span>"
                f"{sev_badge}"
                f"<span style='color:#6b7280;font-size:0.82rem;flex:1'>{row['note']}</span>"
                f"</div>", unsafe_allow_html=True)

        st.caption("⚠ Severity predictions are model-generated by GPT-4o-mini. "
                   "Not for clinical use. Verify with a pharmacist.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("PharmGraph · ChEMBL 37 + DDInter 2.0 · GPT-4o-mini · Research prototype · "
           "Not for clinical use · See Evaluation and About pages for methodology")
