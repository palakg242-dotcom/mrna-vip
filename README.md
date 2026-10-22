# mRNA-VIP - Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform

Student engineering project (case study). Output is **not** real clinical guidance.

Status: **Week 1 and Week 2 complete** (ML risk classifier, self-attention engine). Weeks 3-4 pending.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Weeks 1-2 only need: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib.

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
| `outputs/metrics.json`, `outputs/test_predictions.csv` | Metrics and test scores (used by the Week 4 ROC panel) |
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
| `outputs/figures/attention_multihead.png` | Bonus: 2-head attention |
| `outputs/attention_weights.npy`, `outputs/attention_meta.json` | Saved result used by the Week 5 dashboard |
| `outputs/reports/week2_attention_explanation.md` | Written explanation of Q/K/V and the heatmap |

## Data notes
- The text `"None"` in `Prior_Reaction_History` is a real category (no prior reaction), not a missing value. The CSV is read with `keep_default_na=False`; only 17 truly empty cells are imputed (mode).
- `Biomarker_Level`: 20 missing values, imputed with the median.
- The scaler is fitted on the training split only (no test-set leakage).

## Data sourcing
`patient_records.csv` and the guidelines document were provided with the PRD; no real patient data.

## Week 2 notes
- `encode_sequence` rejects anything outside A, C, G, U. The 5-50 length rule (FR-2.1) is enforced by `analyze_sequence`, because the PRD's own acceptance example calls `encode_sequence('ACGU')`, which has only 4 bases.
- With the template's small projection weights (x0.1), attention is almost uniform (0.065-0.069 against a uniform value of 0.067). The heatmap title states the range so the colour scale is not misleading.
- Embeddings and weights use a seeded random generator, so results are identical on every run.
