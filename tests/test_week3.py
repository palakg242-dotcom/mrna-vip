"""Tests for Week 3 functional requirements FR-3.1 to FR-3.5.

Run from the repo root:  python tests/test_week3.py   (or: pytest tests/)

Part A (chunking) needs nothing but the standard library and runs everywhere.
Part B (ChromaDB + embedding model) needs `chromadb` and `sentence-transformers`; the first
run downloads the model (about 90 MB, internet needed). Part B is skipped if unavailable.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import rag_index as rag  # noqa: E402

DOC = os.path.join(ROOT, "data", "mrna_safety_guidelines.txt")


def proxy_tokens(text: str) -> int:
    """Stand-in tokenizer for Part A: words and punctuation, long words count extra (word-pieces)."""
    return sum(1 + len(w) // 7 for w in re.findall(r"\w+|[^\w\s]", text))


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


# ---------------------------- Part A: chunking ----------------------------
def test_fr_3_1_chunks_within_token_range():
    chunks = rag.chunk_text(open(DOC, encoding="utf-8").read(), proxy_tokens)
    lengths = [proxy_tokens(c) for c in chunks]
    assert len(chunks) >= 2
    assert all(rag.MIN_TOKENS <= n <= rag.MAX_TOKENS for n in lengths), lengths


def test_fr_3_1_consecutive_chunks_overlap():
    chunks = rag.chunk_text(open(DOC, encoding="utf-8").read(), proxy_tokens)
    for a, b in zip(chunks, chunks[1:]):
        shared = set(rag.split_sentences(a)) & set(rag.split_sentences(b))
        assert shared, "consecutive chunks share no sentence"
        assert sum(proxy_tokens(s) for s in shared) >= 1


def test_chunks_cover_whole_document_and_do_not_end_on_a_heading():
    text = open(DOC, encoding="utf-8").read()
    chunks = rag.chunk_text(text, proxy_tokens)
    covered = {s for c in chunks for s in rag.split_sentences(c)}
    assert covered == set(rag.split_sentences(text)), "some text was lost"
    assert not any(rag.is_heading(rag.split_sentences(c)[-1]) for c in chunks[:-1])


def test_chunking_is_deterministic_and_edge_cases():
    text = open(DOC, encoding="utf-8").read()
    assert rag.chunk_text(text, proxy_tokens) == rag.chunk_text(text, proxy_tokens)
    assert len(rag.chunk_text("A short note. Only two sentences.", proxy_tokens)) == 1
    assert _raises(ValueError, rag.chunk_text, "word " * 600 + ".", proxy_tokens)


# ----------------------- Part B: ChromaDB + embeddings -----------------------
def _heavy_libs_available() -> bool:
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def test_fr_3_2_to_3_5_index_persistence_and_retrieval():
    if not _heavy_libs_available():
        print("SKIP  part B (chromadb / sentence-transformers not installed)")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        store, log = os.path.join(d, "chroma"), os.path.join(d, "latency.csv")
        info = rag.build_index(DOC, persist_dir=store, verbose=False)

        # FR-3.1 again, with the real tokenizer
        assert all(rag.MIN_TOKENS <= n <= rag.MAX_TOKENS for n in info["token_lengths"]), info["token_lengths"]

        # FR-3.2: one stored vector per chunk
        index = rag.load_index(store, model=info["model"])
        assert index.collection.count() == info["n_chunks"]

        # FR-3.3: a brand-new Python process can reload the index and answer a query
        code = ("import sys; sys.path.insert(0, r'%s'); import rag_index as r; "
                "i = r.load_index(r'%s'); c, t = r.query_guidelines(i, 'storage temperature', k=2, log_path=None); "
                "print(len(c))" % (os.path.join(ROOT, "src"), store))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.strip().endswith("2"), out.stderr[-500:]

        # FR-3.4 + FR-3.5: 5+ questions return k chunks, latency is measured, logged, and under 5 s
        hits_ok = 0
        for question, keyword in rag.TEST_QUESTIONS[:6]:
            chunks, latency = rag.query_guidelines(index, question, k=3, log_path=log)
            assert len(chunks) == min(3, info["n_chunks"])
            assert 0 < latency < 5, latency
            hits_ok += any(keyword in c.lower() for c in chunks)
        assert hits_ok >= 5, f"only {hits_ok}/6 questions had the expected passage in the top-3"
        with open(log, encoding="utf-8") as f:
            assert len(f.readlines()) == 1 + 6   # header + one row per query


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} Week 3 tests passed.")
