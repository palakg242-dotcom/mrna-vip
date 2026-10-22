"""Tests for Week 2 functional requirements FR-2.1 to FR-2.5.

Run from the repo root:  python tests/test_week2.py   (or: pytest tests/)
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import attention as att  # noqa: E402

TEST_SEQUENCES = [
    "ACGUA",                                              # shortest allowed (5)
    "ACGUACGUACGGCUA",                                    # the PRD's 15-base example
    "GGGGGGGGGG",                                         # all identical bases
    "AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA"[:50],  # longest allowed (50)
]


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


def test_fr_2_1_encode_shape():
    assert att.encode_sequence("ACGU").shape == (4, att.EMBED_DIM)  # PRD acceptance example
    assert att.encode_sequence("ACGUACGUACGGCUA").shape == (15, att.EMBED_DIM)


def test_fr_2_1_length_rule_5_to_50():
    with tempfile.TemporaryDirectory() as d:
        assert _raises(ValueError, att.analyze_sequence, "ACGU", d)          # 4 is too short
        assert _raises(ValueError, att.analyze_sequence, "A" * 51, d)        # 51 is too long
        att.analyze_sequence("A" * 5, d)                                     # boundaries are OK
        att.analyze_sequence("C" * 50, d)


def test_fr_2_2_rejects_invalid_characters():
    assert _raises(ValueError, att.encode_sequence, "ACGX")
    assert _raises(ValueError, att.encode_sequence, "ACGT")   # DNA base T is not allowed in mRNA
    assert _raises(ValueError, att.encode_sequence, "acgu")   # lowercase is not silently accepted
    assert _raises(ValueError, att.encode_sequence, "")
    try:
        att.encode_sequence("ACGX")
    except ValueError as e:
        assert "'X'" in str(e) and "position 4" in str(e)     # error names the culprit


def test_fr_2_3_shapes():
    X = att.encode_sequence("ACGUACGUACGGCUA")
    weights, output = att.self_attention(X)
    assert weights.shape == (15, 15)
    assert output.shape == (15, att.EMBED_DIM)


def test_fr_2_4_rows_sum_to_one():
    for seq in TEST_SEQUENCES:
        for scale in (0.1, 1.0, 5.0):  # also when scores are large (numerical stability)
            w, _ = att.self_attention(att.encode_sequence(seq), weight_scale=scale)
            assert np.allclose(w.sum(axis=1), 1.0), (seq, scale)
            assert np.all(w >= 0)


def test_fr_2_5_heatmap_saved():
    with tempfile.TemporaryDirectory() as d:
        res = att.analyze_sequence("ACGUACGUACGGCUA", d)
        path = os.path.join(d, "figures", "attention_heatmap.png")
        assert res["heatmap_path"] == path
        assert os.path.getsize(path) > 10_000
        with open(path, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"   # a real PNG file
        assert np.load(os.path.join(d, "attention_weights.npy")).shape == (15, 15)


def test_deterministic_and_seed_sensitive():
    X = att.encode_sequence("ACGUACGUACGGCUA")
    a, _ = att.self_attention(X, seed=42, weight_scale=1.0)
    b, _ = att.self_attention(X, seed=42, weight_scale=1.0)
    c, _ = att.self_attention(X, seed=7, weight_scale=1.0)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_hand_computed_three_token_example():
    """The 3-token toy example from notebooks/02_attention_math.ipynb, worked on paper."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    I = np.eye(2)
    d = att.attention_details(X, d_k=2, W=(I, I, I))
    expected = np.array([[0.4011, 0.1978, 0.4011],
                         [0.1978, 0.4011, 0.4011],
                         [0.2483, 0.2483, 0.5035]])
    assert np.allclose(d["weights"], expected, atol=1e-4)
    assert np.allclose(d["output"][0], [0.8022, 0.5989], atol=1e-3)


def test_bonus_multi_head():
    X = att.encode_sequence("ACGUACGUACGGCUA")
    heads, out = att.multi_head_attention(X, num_heads=3)
    assert len(heads) == 3 and all(h.shape == (15, 15) for h in heads)
    assert out.shape == (15, 3 * att.EMBED_DIM)
    assert not np.allclose(heads[0], heads[1])   # heads really are independent


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} Week 2 tests passed.")
