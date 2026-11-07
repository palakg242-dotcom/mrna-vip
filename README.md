# mRNA-VIP - Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform

Student engineering project (case study). Output is **not** real clinical guidance.

Status: **All modules complete** (Week 1 risk classifier, Week 2 self-attention, Week 3 RAG, Week 4 LLM report generator and dashboard). The final write-up is in [`outputs/reports/portfolio_report.md`](outputs/reports/portfolio_report.md).

## Pipeline overview

| Module | What it does | Main file |
|---|---|---|
| 1 | Cleans patient data, trains Logistic Regression and Random Forest, saves the better model | `src/risk_model.py` |
| 2 | Self-attention from scratch in NumPy, heatmaps | `src/attention.py` |
| 3 | Chunks the guidelines, embeds them, stores them in ChromaDB, retrieves the top-3 chunks | `src/rag_index.py` |
| 4 | Combines risk score + retrieved guidelines in an LLM prompt and returns a validated JSON report | `src/report_generator.py` |
| 5 | 3-panel dashboard (ROC curve, attention heatmap, latency chart) | `src/dashboard.py` |

## Setup

Tested on Windows 11 with Python 3.12 (the PRD needs 3.10+).

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins every version used. If the installed scikit-learn differs from the one that saved `outputs/risk_model.joblib`, a version warning appears when loading the model. It is harmless; running `python src/risk_model.py` again re-creates the model with your version.

### Environment variables (LLM API key)

Only Week 4 needs an LLM. Copy `.env.example` to a new file named `.env` **in the project root** (next to `README.md`) and fill it in:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `mock` (no key, no internet, for testing; reports say clearly that it is not a real model), `gemini` (free tier of Google AI Studio), or `ollama` (local model) |
| `GEMINI_API_KEY` | Key from aistudio.google.com. Never commit it. |
| `GEMINI_MODEL` | Model name. List the models your key can use with `python src/report_generator.py --list-models`. |

`.env` is listed in `.gitignore`, so it is never uploaded. Do not send real patient data to a free-tier API; this project uses the synthetic data provided with the PRD.

## Reproduce Week 1

```bash
python src/risk_model.py          # cleans data, trains both models, saves outputs
python tests/test_week1.py        # checks FR-1.1 to FR-1.7
jupyter notebook notebooks/01_data_eda.ipynb   # EDA (5 visualizations)
```

## Week 1 outputs

| File | Purpose |
|---|---|
| `data/patient_records_clean.csv` | Cleaned data, 0 missing values |
| `outputs/risk_model.joblib` | Selected model (Logistic Regression, chosen by High-risk F1) |
| `outputs/preprocessor.joblib` | Scaler + category order, needed in Week 4 to score a new patient |
| `outputs/figures/` | Classification reports and confusion matrices (both models), EDA figures |
| `outputs/metrics.json`, `outputs/test_predictions.csv` | Metrics and test scores (used by the dashboard ROC panel) |
| `outputs/week1_model_comparison.md` | Written model comparison |

## Reproduce Week 2

```bash
python src/attention.py           # heatmaps + saved attention weights
python tests/test_week2.py        # checks FR-2.1 to FR-2.5, 3-token hand example, multi-head bonus
jupyter notebook notebooks/02_attention_math.ipynb
```

## Week 2 outputs

| File | Purpose |
|---|---|
| `src/attention.py` | Scaled dot-product self-attention from scratch (NumPy), input validation, heatmap, multi-head bonus |
| `outputs/figures/attention_heatmap.png` | Labeled heatmap (template scale 0.1) |
| `outputs/figures/attention_heatmap_scale1.png` | Same sequence with 10x larger projection weights, for comparison |
| `outputs/figures/attention_multihead.png` | Bonus: 3-head attention |
| `outputs/attention_weights.npy`, `outputs/attention_meta.json` | Saved result used by the dashboard |
| `outputs/reports/week2_attention_explanation.md` | Written explanation of Q/K/V and the heatmap |

## Reproduce Week 3

```bash
python src/rag_index.py           # builds the ChromaDB index, answers 7 test questions, writes the retrieval log
python tests/test_week3.py        # checks FR-3.1 to FR-3.5
```

The first run downloads the embedding model `all-MiniLM-L6-v2` (about 90 MB), so it needs internet once.

## Week 3 outputs

| File | Purpose |
|---|---|
| `data/mrna_safety_guidelines.txt` | Plain-text version of the provided guidelines document (title page and student note removed) |
| `src/rag_index.py` | Token-aware chunking (50-token overlap), embeddings, ChromaDB index, `query_guidelines()` |
| `outputs/chroma_store/` | The vector index. Not committed (it is in `.gitignore`); `rag_index.py` rebuilds it. |
| `outputs/reports/week3_retrieval_log.md` | 7 test questions with top chunk, similarity and latency |
| `outputs/retrieval_latency_log.csv` | Per-query retrieval latency |
| `outputs/reports/week3_rag_writeup.md` | Chunking choices and failure case |

## Reproduce Week 4

```bash
# Without an API key (mock provider)
python src/report_generator.py --provider mock

# With Gemini (needs the .env file from the setup section)
python src/report_generator.py --list-models                       # optional: see usable model names
python src/report_generator.py --provider gemini --patient P001    # one patient
python src/report_generator.py --provider gemini                   # all 5 sample patients

python src/dashboard.py            # outputs/figures/dashboard.png
python tests/test_week4.py         # checks FR-4.1 to FR-5.3
```

If Gemini answers with `503 UNAVAILABLE` or `429`, the server is busy or the free-tier limit was reached. Wait a minute and retry, or switch to a lighter model in `GEMINI_MODEL`.

## Week 4 outputs

| File | Purpose |
|---|---|
| `src/schemas.py` | Pydantic report schema (5 fields, extra fields rejected) |
| `src/report_generator.py` | Risk model + RAG + LLM, validation, exactly one retry, code-level 100 mcg dose check |
| `src/dashboard.py` | 3-panel dashboard (ROC curve, attention heatmap, latency chart) |
| `data/sample_patients.json` | 5 sample patients |
| `outputs/reports/report_P00X_gemini.json` / `.md` | Gemini reports (one pair per patient) |
| `outputs/reports/report_P00X_mock.json` / `.md` | Reports from the mock provider |
| `outputs/reports/sample_report.json` / `.md` | One example report |
| `outputs/llm_latency_log.csv` | LLM latency per report |
| `outputs/figures/dashboard.png` | Final dashboard (150 DPI) |
| `outputs/reports/portfolio_report.md` | Final portfolio report |

## Run all tests

```bash
python tests/test_week1.py
python tests/test_week2.py
python tests/test_week3.py
python tests/test_week4.py
```

## Data notes

- The text `"None"` in `Prior_Reaction_History` is a real category (no prior reaction), not a missing value. The CSV is read with `keep_default_na=False`; only 17 truly empty cells are imputed (mode).
- `Biomarker_Level`: 20 missing values, imputed with the median.
- The scaler is fitted on the training split only (no test-set leakage).
- Dose is handled in **mcg**, because the data and the guidelines use mcg. The PRD schema field `Recommended_Dose_mg` is kept and filled by converting (100 mcg = 0.1 mg).

## Data sourcing

`patient_records.csv` and the guidelines document were provided with the PRD; no real patient data.

## Week 2 notes

- `encode_sequence` rejects anything outside A, C, G, U. The 5-50 length rule (FR-2.1) is enforced by `analyze_sequence`, because the PRD's own acceptance example calls `encode_sequence('ACGU')`, which has only 4 bases.
- With the template's small projection weights (x0.1), attention is almost uniform (0.065-0.069 against a uniform value of 0.067). The heatmap title states the range so the colour scale is not misleading.
- Embeddings and weights use a seeded random generator, so results are identical on every run.

## Differences from the PRD template (and why)

| PRD template | What this project does | Reason |
|---|---|---|
| Scaler fitted before the split | Fitted on the training split only | Avoids test-set leakage |
| Pandas default `NaN` handling | `keep_default_na=False` | 291 `"None"` values would otherwise become "Mild" |
| Chunk size in characters | Chunk size in tokens (300-500, 50 overlap) | FR-3.1 is defined in tokens |
| LangChain + OpenAI | Direct Gemini SDK (free tier) | No cost, fewer version problems |
| Dose in mg | Dose validated in mcg | Data and guidelines use mcg |

## Known limitations

See Section 8 of the [portfolio report](outputs/reports/portfolio_report.md). In short: the guideline document is tiny (3 chunks), retrieval never answers "not covered", "elevated biomarker" has no numeric definition in the guidelines, and free-tier LLM latency varies a lot.
