# mRNA-VIP - Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform

Student engineering project (case study). Output is **not** real clinical guidance.

Status: **Week 1 complete** (data pipeline + ML risk classifier). Weeks 2-4 pending.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Week 1 only needs: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib.

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

## Data notes
- The text `"None"` in `Prior_Reaction_History` is a real category (no prior reaction), not a missing value. The CSV is read with `keep_default_na=False`; only 17 truly empty cells are imputed (mode).
- `Biomarker_Level`: 20 missing values, imputed with the median.
- The scaler is fitted on the training split only (no test-set leakage).

## Data sourcing
`patient_records.csv` and the guidelines document were provided with the PRD; no real patient data.
