"""Week 2 - Scaled dot-product self-attention on mRNA sequences, from scratch (NumPy).

    Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_k)) . V

Satisfies FR-2.1 to FR-2.5. Run from the repo root:  python src/attention.py
"""
import json
import os

import matplotlib.pyplot as plt  # backend is NOT forced, so notebooks can display charts
import numpy as np

VOCAB = ["A", "C", "G", "U"]
EMBED_DIM = 8
SEED = 42
MIN_LEN, MAX_LEN = 5, 50  # allowed sequence length (FR-2.1)

# Toy "learnable" embedding table: one random vector per nucleotide.
# A seeded generator is used (instead of the global np.random state) so that
# results are identical on every run and on every machine.
_rng = np.random.default_rng(SEED)
embedding_table = {base: _rng.standard_normal(EMBED_DIM) for base in VOCAB}


# --------------------------------------------------------------------------
# Input validation and encoding (FR-2.1, FR-2.2)
# --------------------------------------------------------------------------
def validate_sequence(sequence: str, enforce_length: bool = False) -> None:
    """Raise ValueError unless `sequence` only contains A, C, G, U (FR-2.2).

    With enforce_length=True the length must also be 5-50 (FR-2.1).
    """
    if not isinstance(sequence, str) or len(sequence) == 0:
        raise ValueError("Sequence must be a non-empty string of A, C, G, U.")
    bad = [(i + 1, ch) for i, ch in enumerate(sequence) if ch not in VOCAB]
    if bad:
        shown = ", ".join(f"'{ch}' at position {pos}" for pos, ch in bad[:5])
        raise ValueError(
            f"Invalid character(s) in mRNA sequence: {shown}. "
            f"Only {VOCAB} (uppercase) are allowed."
        )
    if enforce_length and not (MIN_LEN <= len(sequence) <= MAX_LEN):
        raise ValueError(
            f"Sequence length must be between {MIN_LEN} and {MAX_LEN}, got {len(sequence)}."
        )


def encode_sequence(sequence: str) -> np.ndarray:
    """Turn a string like 'ACGU...' into a (seq_len, embed_dim) matrix.

    Invalid characters raise ValueError instead of silently producing output.
    (The 5-50 length rule is enforced by analyze_sequence(); the PRD's own
    acceptance test calls encode_sequence('ACGU'), which has 4 characters.)
    """
    validate_sequence(sequence)
    return np.stack([embedding_table[base] for base in sequence])


# --------------------------------------------------------------------------
# Attention maths (FR-2.3, FR-2.4)
# --------------------------------------------------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


WEIGHT_SCALE = 0.1  # PRD template value: small random weights


def make_projection_weights(embed_dim: int, d_k: int, seed: int = SEED, scale: float = WEIGHT_SCALE):
    """Random W_q, W_k, W_v of shape (embed_dim, d_k), reproducible via `seed`.

    `scale` multiplies the random numbers. Small scales give small dot products,
    so softmax(scores) stays close to uniform (every position gets ~1/seq_len).
    """
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal((embed_dim, d_k)) * scale for _ in range(3))


def attention_details(X: np.ndarray, d_k: int = EMBED_DIM, seed: int = SEED, W=None,
                      weight_scale: float = WEIGHT_SCALE) -> dict:
    """Run self-attention and return every intermediate matrix (useful for teaching).

    X: (seq_len, embed_dim). W: optional (W_q, W_k, W_v) to use instead of random ones.
    """
    seq_len, embed_dim = X.shape
    W_q, W_k, W_v = W if W is not None else make_projection_weights(embed_dim, d_k, seed, weight_scale)

    Q = X @ W_q                                   # what each position looks for
    K = X @ W_k                                   # what each position offers
    V = X @ W_v                                   # the content each position contributes
    scores = (Q @ K.T) / np.sqrt(d_k)             # (seq_len, seq_len) raw similarity
    weights = softmax(scores)                     # each row sums to 1
    output = weights @ V                          # (seq_len, d_k)
    return {"W_q": W_q, "W_k": W_k, "W_v": W_v, "Q": Q, "K": K, "V": V,
            "scores": scores, "weights": weights, "output": output}


def self_attention(X: np.ndarray, d_k: int = EMBED_DIM, seed: int = SEED,
                   weight_scale: float = WEIGHT_SCALE):
    """X: (seq_len, embed_dim) -> (attention_weights (seq_len, seq_len), output (seq_len, d_k))."""
    d = attention_details(X, d_k=d_k, seed=seed, weight_scale=weight_scale)
    return d["weights"], d["output"]


def multi_head_attention(X: np.ndarray, num_heads: int = 2, d_k: int = EMBED_DIM, seed: int = SEED,
                         weight_scale: float = WEIGHT_SCALE):
    """Bonus: run `num_heads` independent attention heads and concatenate their outputs.

    Returns (list of per-head weight matrices, concatenated output of shape
    (seq_len, num_heads * d_k)). Each head has its own random Q/K/V weights.
    """
    if num_heads < 1:
        raise ValueError("num_heads must be at least 1.")
    all_weights, outputs = [], []
    for h in range(num_heads):
        w, out = self_attention(X, d_k=d_k, seed=seed + 1000 * h, weight_scale=weight_scale)
        all_weights.append(w)
        outputs.append(out)
    return all_weights, np.concatenate(outputs, axis=-1)


# --------------------------------------------------------------------------
# Plotting (FR-2.5)
# --------------------------------------------------------------------------
def position_labels(sequence: str) -> list:
    """['A1', 'C2', 'G3', ...] so that both axes show base AND position."""
    return [f"{base}{i + 1}" for i, base in enumerate(sequence)]


def plot_attention(attention_weights: np.ndarray, sequence: str, save_path: str,
                   title: str = "mRNA Sequence Self-Attention Weights", ax=None) -> None:
    """Save a labeled heatmap (rows = attending FROM, columns = attending TO)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = len(sequence)
    labels = position_labels(sequence)
    size = max(6.0, 0.45 * n + 3)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(attention_weights, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8 if n > 25 else 10)
    ax.set_yticklabels(labels, fontsize=8 if n > 25 else 10)
    ax.set_xlabel("Attending TO position (base + position)")
    ax.set_ylabel("Attending FROM position (base + position)")
    lo, hi = float(attention_weights.min()), float(attention_weights.max())
    ax.set_title(f"{title}\nweights range {lo:.3f}-{hi:.3f}  (perfectly uniform attention = {1 / n:.3f})",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="Attention weight", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_multi_head(all_weights: list, sequence: str, save_path: str) -> None:
    """Bonus: one heatmap per head, side by side."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n, heads = len(sequence), len(all_weights)
    labels = position_labels(sequence)
    fig, axes = plt.subplots(1, heads, figsize=(6.2 * heads, 5.6))
    axes = np.atleast_1d(axes)
    for h, (ax, w) in enumerate(zip(axes, all_weights)):
        im = ax.imshow(w, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Attending TO position")
        ax.set_ylabel("Attending FROM position")
        ax.set_title(f"Head {h + 1}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Multi-head self-attention (independent Q/K/V weights per head)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------
def analyze_sequence(sequence: str, out_dir: str = "outputs", seed: int = SEED,
                     weight_scale: float = WEIGHT_SCALE, heatmap_name: str = "attention_heatmap.png") -> dict:
    """Validate (incl. length 5-50) -> encode -> attention -> sanity check -> heatmap + saved outputs."""
    validate_sequence(sequence, enforce_length=True)
    X = encode_sequence(sequence)
    weights, output = self_attention(X, seed=seed, weight_scale=weight_scale)
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise AssertionError("Softmax rows must sum to 1")

    heatmap_path = os.path.join(out_dir, "figures", heatmap_name)
    plot_attention(weights, sequence, heatmap_path)

    # Saved so the Week 5 dashboard can draw Panel B from this module's real output (FR-5.2).
    np.save(os.path.join(out_dir, "attention_weights.npy"), weights)
    with open(os.path.join(out_dir, "attention_meta.json"), "w") as f:
        json.dump({"sequence": sequence, "seed": seed, "embed_dim": EMBED_DIM,
                   "weight_scale": weight_scale, "weights_shape": list(weights.shape)}, f, indent=2)
    return {"X": X, "weights": weights, "output": output, "heatmap_path": heatmap_path}


if __name__ == "__main__":
    sequence = "ACGUACGUACGGCUA"
    result = analyze_sequence(sequence)
    print(f"Sequence: {sequence} (length {len(sequence)})")
    print("Embedding matrix shape:", result["X"].shape)
    print("Attention weights shape:", result["weights"].shape)
    print("Row sums all 1.0:", np.allclose(result["weights"].sum(axis=1), 1.0))
    print("Saved:", result["heatmap_path"])

    heads_w, heads_out = multi_head_attention(result["X"], num_heads=2)
    plot_multi_head(heads_w, sequence, "outputs/figures/attention_multihead.png")
    print("Multi-head output shape:", heads_out.shape)

    # Comparison: same sequence, same seed, but 10x larger projection weights.
    # (Plotted directly so it does NOT overwrite outputs/attention_weights.npy.)
    big_w, _ = self_attention(result["X"], weight_scale=1.0)
    plot_attention(big_w, sequence, "outputs/figures/attention_heatmap_scale1.png",
                   title="Self-attention weights (projection scale 1.0)")
    print("Scale 1.0 weights range:", round(float(big_w.min()), 3), "-", round(float(big_w.max()), 3))
