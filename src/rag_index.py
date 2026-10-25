"""Week 3 - Retrieval-Augmented Generation (RAG) index over the mRNA safety guidelines.

Pipeline:  text file -> sentence-aware token chunks (300-500 tokens, ~50 overlap)
           -> sentence embeddings (all-MiniLM-L6-v2) -> persistent ChromaDB index
           -> query_guidelines(index, question, k) -> top-k chunks + latency

Satisfies FR-3.1 to FR-3.5. Run from the repo root:
    python src/rag_index.py                      # build index + run the test questions
    python src/rag_index.py --query "..." --k 3  # ask one question against the saved index
"""
import argparse
import csv
import os
import re
import time
from datetime import datetime

DOC_PATH = "data/mrna_safety_guidelines.txt"
PERSIST_DIR = "outputs/chroma_store"
COLLECTION_NAME = "mrna_safety_guidelines"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 512        # MiniLM defaults to 256; chunks here are up to 500 tokens
MIN_TOKENS, MAX_TOKENS = 300, 500   # allowed chunk size (FR-3.1)
START_TARGET = 400                  # preferred chunk size; raised if the last chunk would be too small
OVERLAP_TOKENS = 50                 # ~50-token overlap, rounded up to whole sentences
LATENCY_LOG = "outputs/retrieval_latency_log.csv"
REPORT_PATH = "outputs/reports/week3_retrieval_log.md"

# (question, keyword the correct chunk should contain). None = deliberately out of scope.
TEST_QUESTIONS = [
    ("What temperature should unopened mRNA vials be stored at?", "-25c"),
    ("Can a vial be refrozen after it has thawed?", "refrozen"),
    ("Which dose needs secondary clinician review?", "100 micrograms"),
    ("What should be done for a patient aged 65 or older with pre-existing conditions?", "two or more"),
    ("Within what time must a moderate adverse event be documented?", "24 hours"),
    ("How long should a patient be monitored after the injection?", "15 minutes"),
    ("Is the vaccine safe to give during pregnancy?", None),  # not in the document
]


# --------------------------------------------------------------------------
# Chunking (FR-3.1)
# --------------------------------------------------------------------------
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\-])|\n+")
_HEADING = re.compile(r"^(Section \d+:|\d+\.\d+ )")


def split_sentences(text: str) -> list:
    """Split into sentences. Headings and bullet lines count as their own pieces."""
    return [p.strip() for p in _SENTENCE_BREAK.split(text) if p and p.strip()]


def is_heading(sentence: str) -> bool:
    return bool(_HEADING.match(sentence)) and not sentence.endswith(".")


def _join(sentences: list) -> str:
    out = ""
    for i, s in enumerate(sentences):
        if i:
            prev = sentences[i - 1]
            # keep line breaks after headings and after lines that do not end a sentence
            out += "\n" if (is_heading(s) or is_heading(prev) or prev[-1] not in ".!?") else " "
        out += s
    return out


def _spans(sizes: list, sentences: list, target: int, overlap: int) -> list:
    """Greedily pack sentences up to `target` tokens; start each new chunk with the last
    sentence(s) of the previous one (>= `overlap` tokens) so context carries across."""
    n, spans, start = len(sizes), [], 0
    while start < n:
        end, total = start, 0
        while end < n and total + sizes[end] <= target:
            total += sizes[end]
            end += 1
        if end == start:                     # one sentence longer than target
            end += 1
        while end - start > 1 and is_heading(sentences[end - 1]):
            end -= 1                         # never end a chunk on a dangling heading
        spans.append((start, end))
        if end >= n:
            break
        back, acc = end, 0
        while back > start + 1 and acc < overlap:
            back -= 1
            acc += sizes[back]
        start = max(back, start + 1)
    return spans


def chunk_text(text: str, count_tokens, min_tokens: int = MIN_TOKENS, max_tokens: int = MAX_TOKENS,
               start_target: int = START_TARGET, overlap_tokens: int = OVERLAP_TOKENS) -> list:
    """Return overlapping chunks, each at most `max_tokens` tokens (and at least `min_tokens`
    whenever the document allows it). `count_tokens(str) -> int` is the tokenizer length.

    The LangChain template splitter counts characters and cannot guarantee a token range or
    a visible overlap on a document this short, so chunking is done here sentence by sentence.
    """
    sentences = split_sentences(text)
    sizes = [count_tokens(s) for s in sentences]
    if any(sz > max_tokens for sz in sizes):
        raise ValueError(f"A single sentence is longer than {max_tokens} tokens.")

    best = None
    for target in range(start_target, max_tokens + 1, 10):
        spans = _spans(sizes, sentences, target, overlap_tokens)
        chunks = [_join(sentences[a:b]) for a, b in spans]
        lengths = [count_tokens(c) for c in chunks]
        if max(lengths) > max_tokens:
            continue
        score = min(lengths)
        if best is None or score > best[0]:
            best = (score, chunks)
        if score >= min_tokens:
            break
    if best is None:
        raise ValueError("Could not build chunks within the token limit.")
    return best[1]


# --------------------------------------------------------------------------
# Embedding model and index (FR-3.2, FR-3.3)
# --------------------------------------------------------------------------
class GuidelineIndex:
    """A loaded ChromaDB collection together with the model used to embed queries."""

    def __init__(self, collection, model):
        self.collection = collection
        self.model = model


def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def token_counter(model):
    tokenizer = model.tokenizer
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def read_document(path: str = DOC_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_index(doc_path: str = DOC_PATH, persist_dir: str = PERSIST_DIR, model=None,
                verbose: bool = True) -> dict:
    """Chunk the document, embed every chunk and save it in a persistent ChromaDB store."""
    import chromadb

    model = model or load_model()
    count = token_counter(model)
    chunks = chunk_text(read_document(doc_path), count)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(COLLECTION_NAME)      # rebuild from scratch
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[f"chunk_{i:03d}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"chunk_index": i, "n_tokens": count(c)} for i, c in enumerate(chunks)],
    )
    lengths = [count(c) for c in chunks]
    if verbose:
        print(f"Built index: {len(chunks)} chunks, token lengths {lengths}, saved to {persist_dir}")
    return {"n_chunks": len(chunks), "token_lengths": lengths, "chunks": chunks, "model": model}


def load_index(persist_dir: str = PERSIST_DIR, model=None) -> GuidelineIndex:
    """Reload a saved index from disk (no re-embedding of the document)."""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"No saved index found in '{persist_dir}'. Run build_index() first.") from e
    model = model or load_model()
    model.encode(["warm up"], show_progress_bar=False)   # first call is slow; keep it out of the timing
    return GuidelineIndex(collection, model)


# --------------------------------------------------------------------------
# Retrieval (FR-3.4, FR-3.5)
# --------------------------------------------------------------------------
def _log_latency(log_path: str, question: str, k: int, latency: float, top_chunk) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "question", "k", "latency_seconds", "top_chunk_index"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), question, k,
                         round(latency, 4), top_chunk])


def query_details(index: GuidelineIndex, question: str, k: int = 3, log_path=LATENCY_LOG):
    """Return (hits, latency). Each hit has the chunk text, its index and a cosine similarity.

    Embeddings are normalised, and ChromaDB reports squared L2 distance, so
    cosine similarity = 1 - distance / 2. Latency covers embedding the question plus the search.
    """
    n_results = max(1, min(k, index.collection.count()))
    t0 = time.perf_counter()
    q = index.model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    res = index.collection.query(query_embeddings=[q[0].tolist()], n_results=n_results,
                                 include=["documents", "distances", "metadatas"])
    latency = time.perf_counter() - t0

    hits = [{"chunk": doc, "chunk_index": meta["chunk_index"], "similarity": 1 - dist / 2}
            for doc, dist, meta in zip(res["documents"][0], res["distances"][0], res["metadatas"][0])]
    if log_path:
        _log_latency(log_path, question, k, latency, hits[0]["chunk_index"] if hits else "")
    return hits, latency


def query_guidelines(index: GuidelineIndex, question: str, k: int = 3, log_path=LATENCY_LOG):
    """Return ([top-k chunk texts], latency_in_seconds)."""
    hits, latency = query_details(index, question, k, log_path)
    return [h["chunk"] for h in hits], latency


# --------------------------------------------------------------------------
# Test-question log (Definition of Done: 5+ questions, checked by hand)
# --------------------------------------------------------------------------
def run_test_questions(index: GuidelineIndex, k: int = 3, report_path: str = REPORT_PATH,
                       log_path=LATENCY_LOG) -> str:
    rows, latencies = [], []
    for i, (question, keyword) in enumerate(TEST_QUESTIONS, 1):
        hits, latency = query_details(index, question, k, log_path)
        latencies.append(latency)
        rank = next((r for r, h in enumerate(hits, 1) if keyword and keyword in h["chunk"].lower()), None)
        found = "n/a (out of scope)" if keyword is None else (f"yes, rank {rank}" if rank else "NO")
        top = hits[0]["chunk"].replace("\n", " ")
        rows.append(f"| {i} | {question} | chunk {hits[0]['chunk_index']}: {top[:90]}... | "
                    f"{hits[0]['similarity']:.2f} | {found} | {latency:.3f} |  |")

    lines = ["# Week 3 - Retrieval test log", "",
             f"Model: `{MODEL_NAME}`, top-k = {k}, {index.collection.count()} chunks in the index.", "",
             "| # | Question | Top-1 chunk | Similarity | Keyword in top-k | Latency (s) | Relevant? (check by hand) |",
             "|---|---|---|---|---|---|---|", *rows, "",
             f"Mean latency: {sum(latencies) / len(latencies):.3f} s, "
             f"max: {max(latencies):.3f} s.", "",
             "The 'Keyword in top-k' column is only an automatic hint. The last column must be "
             "filled in by reading the returned chunks.", ""]
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build / query the mRNA guidelines RAG index.")
    parser.add_argument("--query", help="ask one question against the saved index")
    parser.add_argument("--k", type=int, default=3, help="number of chunks to return")
    args = parser.parse_args()

    if args.query:
        index = load_index()
        chunks, latency = query_guidelines(index, args.query, k=args.k)
        print(f"Question: {args.query}\nLatency: {latency:.3f} s\n")
        for rank, chunk in enumerate(chunks, 1):
            print(f"--- Result {rank} ---\n{chunk}\n")
        return

    info = build_index()
    index = load_index(model=info["model"])
    path = run_test_questions(index)
    print(f"Test questions answered. Log written to {path}")
    print(f"Latency log: {LATENCY_LOG}")


if __name__ == "__main__":
    main()
