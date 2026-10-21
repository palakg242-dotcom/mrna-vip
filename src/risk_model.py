"""Week 1 - train, evaluate, compare and save the vaccine-reaction risk classifier.

Satisfies FR-1.5 to FR-1.7. Run from the repo root:  python src/risk_model.py
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_pipeline as dp  # noqa: E402

CLASS_NAMES = ["Low risk (0)", "High risk (1)"]


def train_models(X_train, y_train) -> dict:
    """Train Logistic Regression and Random Forest on identical data (FR-1.5)."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=dp.RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=dp.RANDOM_SEED),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate(model, X_test, y_test) -> dict:
    """Predictions, report, confusion matrix and F1 on the held-out test set."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "preds": preds,
        "proba": proba,
        "report_text": classification_report(y_test, preds, target_names=CLASS_NAMES),
        "report_dict": classification_report(y_test, preds, target_names=CLASS_NAMES, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds),
        # F1 of the High-risk class: missing a high-risk patient is the costly error.
        "f1_high_risk": f1_score(y_test, preds, pos_label=1),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Low", "High"])
    ax.set_yticks([0, 1], labels=["Low", "High"])
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Actual risk")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_pipeline(data_path: str = "data/patient_records.csv",
                 out_dir: str = "outputs",
                 clean_csv_path: str = "data/patient_records_clean.csv",
                 verbose: bool = True) -> dict:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load, validate, clean
    raw = dp.load_raw(data_path)
    if verbose:
        print("Missing values before cleaning:\n", raw.isnull().sum(), "\n")
    df = dp.clean(raw)
    assert df.isnull().sum().sum() == 0
    if clean_csv_path:
        df.to_csv(clean_csv_path, index=False)

    # 2. Encode, split, scale (scaler fitted on train only)
    df_enc = dp.encode_categorical(df)
    X_train, X_test, y_train, y_test = dp.split_data(df_enc)
    scaler = dp.fit_scaler(X_train)
    X_train_s = dp.apply_scaler(X_train, scaler)
    X_test_s = dp.apply_scaler(X_test, scaler)

    # 3. Train + evaluate both models
    models = train_models(X_train_s, y_train)
    results = {}
    for name, model in models.items():
        res = evaluate(model, X_test_s, y_test)
        results[name] = res
        with open(os.path.join(fig_dir, f"classification_report_{name}.txt"), "w") as f:
            f.write(res["report_text"])
        plot_confusion_matrix(res["confusion_matrix"], f"Confusion matrix - {name}",
                              os.path.join(fig_dir, f"confusion_matrix_{name}.png"))
        if verbose:
            print(f"--- {name} ---\n{res['report_text']}\n{res['confusion_matrix']}\n")

    # 4. Select best by test-set F1 and serialize (FR-1.7)
    best_name = max(results, key=lambda n: results[n]["f1_high_risk"])
    joblib.dump(models[best_name], os.path.join(out_dir, "risk_model.joblib"))
    # Week 4 needs the same preprocessing to score a new patient.
    joblib.dump({"scaler": scaler, "reaction_order": dp.REACTION_ORDER,
                 "feature_cols": dp.FEATURE_COLS, "numeric_cols": dp.NUMERIC_COLS},
                os.path.join(out_dir, "preprocessor.joblib"))

    # Saved so the Week 4 dashboard can draw the ROC curve from real outputs (FR-5.2).
    pd.DataFrame({"y_true": y_test.values, "y_score": results[best_name]["proba"],
                  "y_pred": results[best_name]["preds"]}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)

    metrics = {
        "selected_model": best_name,
        "selection_metric": "F1 of High-risk class on held-out test set",
        "models": {
            n: {"f1_high_risk": round(r["f1_high_risk"], 4),
                "accuracy": round(r["report_dict"]["accuracy"], 4),
                "macro_f1": round(r["report_dict"]["macro avg"]["f1-score"], 4),
                "confusion_matrix": r["confusion_matrix"].tolist()}
            for n, r in results.items()
        },
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"Selected model: {best_name}")
    return metrics


if __name__ == "__main__":
    run_pipeline()
