"""Week 4 - Three-panel dashboard (FR-5.1 to FR-5.3).

Panel A: ROC curve of the Week 1 risk classifier   (outputs/test_predictions.csv)
Panel B: Week 2 self-attention heatmap             (outputs/attention_weights.npy + attention_meta.json)
Panel C: RAG vs LLM latency                        (outputs/retrieval_latency_log.csv, outputs/llm_latency_log.csv)

Run from the repo root:  python src/dashboard.py
"""
import csv
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

OUT_DIR = "outputs"
SAVE_PATH = "outputs/figures/dashboard.png"
DPI = 200   # FR-5.3 asks for at least 150


def _mean_column(path: str, column: str, only_provider=None):
    values, providers = [], set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if only_provider is not None and row.get("provider") != only_provider:
                continue
            values.append(float(row[column]))
            providers.add(row.get("provider", ""))
    if not values:
        raise ValueError(f"No rows found in {path}.")
    return float(np.mean(values)), providers


def load_latencies(rag_log: str, llm_log: str) -> dict:
    """Average latency per stage from the logged timings. A real LLM is preferred over the mock."""
    for path, hint in ((rag_log, "python src/rag_index.py"), (llm_log, "python src/report_generator.py")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Create it first with:  {hint}")
    rag_mean, _ = _mean_column(rag_log, "latency_seconds")
    with open(llm_log, newline="", encoding="utf-8") as f:
        providers = {row["provider"] for row in csv.DictReader(f)}
    provider = next((p for p in sorted(providers) if p != "mock"), "mock")
    llm_mean, _ = _mean_column(llm_log, "llm_latency_seconds", only_provider=provider)
    return {"RAG retrieval": rag_mean, f"LLM report ({provider})": llm_mean}


def build_dashboard(y_test, y_scores, attention_weights, sequence, latencies: dict,
                    save_path: str = SAVE_PATH, dpi: int = DPI) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # Panel A: ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    axes[0].plot(fpr, tpr, color="#1F3A5F", linewidth=2, label=f"AUC = {auc(fpr, tpr):.2f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("Risk Classifier ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # Panel B: attention heatmap
    n = len(sequence)
    labels = [f"{base}{i + 1}" for i, base in enumerate(sequence)]
    im = axes[1].imshow(attention_weights, cmap="viridis")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_title("mRNA Self-Attention Weights")
    axes[1].set_xlabel("Attending TO position")
    axes[1].set_ylabel("Attending FROM position")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel C: latency comparison (log scale when the two stages differ by more than 50x)
    names, values = list(latencies.keys()), list(latencies.values())
    bars = axes[2].bar(names, values, color=["#0F6E6E", "#1F3A5F", "#7A8B99"][:len(names)])
    if min(values) > 0 and max(values) / min(values) > 50:
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Seconds (log scale)")
    else:
        axes[2].set_ylabel("Seconds")
    for bar, value in zip(bars, values):
        axes[2].annotate(f"{value:.3f} s", (bar.get_x() + bar.get_width() / 2, value),
                         ha="center", va="bottom", fontsize=10)
    axes[2].set_title("Pipeline Stage Latency (seconds)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


def main() -> None:
    import pandas as pd

    preds = pd.read_csv(os.path.join(OUT_DIR, "test_predictions.csv"))
    weights = np.load(os.path.join(OUT_DIR, "attention_weights.npy"))
    with open(os.path.join(OUT_DIR, "attention_meta.json"), encoding="utf-8") as f:
        sequence = json.load(f)["sequence"]
    latencies = load_latencies(os.path.join(OUT_DIR, "retrieval_latency_log.csv"),
                               os.path.join(OUT_DIR, "llm_latency_log.csv"))
    path = build_dashboard(preds["y_true"], preds["y_score"], weights, sequence, latencies)
    print(f"Dashboard saved to {path}")
    print("Latencies used:", {k: round(v, 4) for k, v in latencies.items()})


if __name__ == "__main__":
    main()
