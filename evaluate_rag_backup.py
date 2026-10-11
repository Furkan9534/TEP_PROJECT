
from pathlib import Path
import csv
import re
import time
import requests
import numpy as np

# ---------------- Configuration ----------------

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "embeddinggemma"

PROJECT_DIR = Path(
    r"C:\Users\Furkan-KALKAN\Desktop\TEP_PROJECT"
)
KB_DIR = PROJECT_DIR / "knowledge_base"
OUTPUT_CSV = PROJECT_DIR / "rag_evaluation_results.csv"

# ---------------- Load knowledge base ----------------

chunks = []

for file_path in sorted(KB_DIR.glob("*.txt")):
    content = file_path.read_text(encoding="utf-8")
    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            chunks.append({
                "text": paragraph,
                "source": file_path.name
            })

if not chunks:
    raise RuntimeError(
        f"No text documents found in {KB_DIR}"
    )


def get_embeddings(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=180
    )
    response.raise_for_status()
    return response.json()["embeddings"]


print("Creating embeddings...")
vectors = np.asarray(
    get_embeddings([c["text"] for c in chunks]),
    dtype=np.float32
)
vectors /= np.maximum(
    np.linalg.norm(vectors, axis=1, keepdims=True),
    1e-12
)


# ---------------- Retrieval ----------------

def retrieve(question, top_k=3):
    q = np.asarray(
        get_embeddings([question])[0],
        dtype=np.float32
    )
    q /= max(np.linalg.norm(q), 1e-12)

    scores = vectors @ q

    # Boost exact fault IDs appearing in the question.
    ids = set(re.findall(r"\bIDV\d+\b", question.upper()))

    if ids:
        for i, chunk in enumerate(chunks):
            chunk_ids = set(
                re.findall(r"\bIDV\d+\b", chunk["text"].upper())
            )
            if ids.intersection(chunk_ids):
                scores[i] += 1.0

    indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in indices]


# ---------------- LLM calls ----------------

def call_llm(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def answer_baseline(question):
    prompt = f"""
Answer this Tennessee Eastman Process question
using your existing knowledge. Do not invent facts.
If you do not know, say so.

Question: {question}
"""
    start = time.perf_counter()
    answer = call_llm(prompt)
    elapsed = time.perf_counter() - start
    return answer, elapsed


def answer_rag(question):
    results = retrieve(question, top_k=3)

    evidence = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item, score in results
    )

    prompt = f"""
    Answer only using the evidence below.
    State the fault ID, variable description and fault type
    when provided.
    Copy the variable description faithfully from the evidence.
    Do not omit qualifiers, conditions, stream numbers,
    or other details that are part of the variable description.
    Do not invent a fault type or physical cause.
    If the evidence does not specify a detail, say so.
    Cite the source filename.

    Evidence:
    {evidence}

    Question: {question}
    """

    start = time.perf_counter()
    answer = call_llm(prompt)
    elapsed = time.perf_counter() - start

    sources = "; ".join(
        sorted(set(item["source"] for item, score in results))
    )
    retrieved_text = "\n".join(
        item["text"] for item, score in results
    )

    return answer, elapsed, sources, retrieved_text


# ---------------- Test questions ----------------
# Expected answers are grounded in the starter knowledge base.
# Unknown scenarios are intentionally tested for abstention.

tests = [
    (1, "What variable and fault type are listed for IDV1?",
     ["a/c feed ratio", "step"], True),
    (2, "What variable and fault type are listed for IDV2?",
     ["b composition", "step"], True),
    (3, "What variable and fault type are listed for IDV3?",
     ["d feed temperature", "step"], True),
    (4, "What variable and fault type are listed for IDV4?",
     ["reactor cooling-water inlet temperature", "step"], True),
    (5, "What variable and fault type are listed for IDV5?",
     ["condenser cooling-water inlet temperature", "step"], True),
    (6, "What variable and fault type are listed for IDV6?",
     ["a feed loss", "step"], True),
    (7, "What variable and fault type are listed for IDV7?",
     ["c header pressure loss", "step"], True),
    (8, "What variable and fault type are listed for IDV8?",
     ["feed composition", "random"], True),
    (9, "What variable and fault type are listed for IDV9?",
     ["d feed temperature", "random"], True),
    (10, "What variable and fault type are listed for IDV10?",
     ["c feed temperature", "random"], True),
    (11, "What variable and fault type are listed for IDV11?",
     ["reactor cooling-water inlet temperature", "random"], True),
    (12, "What variable and fault type are listed for IDV12?",
     ["condenser cooling-water inlet temperature", "random"], True),
    (13, "What variable and fault type are listed for IDV13?",
     ["reaction kinetics", "slow drift"], True),
    (14, "What variable and fault type are listed for IDV14?",
     ["reactor cooling-water valve", "sticking"], True),
    (15, "What variable and fault type are listed for IDV15?",
     ["condenser cooling-water valve", "sticking"], True),
    (16, "Does the current reference specify the physical fault type for IDV16?",
     ["unknown"], False),
    (17, "Does the current reference specify the physical fault type for IDV17?",
     ["unknown"], False),
    (18, "Does the current reference specify the physical fault type for IDV18?",
     ["unknown"], False),
    (19, "Does the current reference specify the physical fault type for IDV19?",
     ["unknown"], False),
    (20, "Does the current reference specify the physical fault type for IDV20?",
     ["unknown"], False),
    (21, "What variable is listed for IDV21, and is its fault type specified?",
     ["valve position constant", "not specify"], True),
]


# ---------------- Evaluation ----------------


def normalize(text):
    text = text.lower()
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_term(text, term):
    normalized_text = normalize(text)
    normalized_term = normalize(term)

    # Explicit equivalent expressions used in our test set.
    aliases = {
        "b composition": ["b composition", "composition b"],
        "a feed loss": [
            "a feed loss",
            "loss of a feed",
            "a feed lost"
        ],
        "c header pressure loss": [
            "c header pressure loss",
            "header pressure loss",
            "loss of c header pressure"
        ],
        "feed composition": [
            "feed composition",
            "composition of the feed"
        ],
        "valve position constant": [
            "valve position constant",
            "valve position in stream 4"
        ],
        "not specify": [
            "not specify",
            "not specified",
            "does not specify",
            "not provided"
        ],
        "unknown": [
            "unknown",
            "not specified",
            "does not specify",
            "not provided"
        ],
    }

    candidates = aliases.get(normalized_term, [normalized_term])

    return any(
        re.search(
            r"(?<!\w)" + re.escape(candidate) + r"(?!\w)",
            normalized_text
        )
        for candidate in candidates
    )


def terms_match(text, expected_terms):
    return all(contains_term(text, term) for term in expected_terms)



rows = []

for fault_id, question, expected_terms, answer_known in tests:
    print(f"\nEvaluating IDV{fault_id}...")

    baseline, baseline_time = answer_baseline(question)
    rag, rag_time, sources, retrieved_text = answer_rag(question)

    baseline_norm = normalize(baseline)
    rag_norm = normalize(rag)
    evidence_norm = normalize(retrieved_text)

    baseline_terms_found = terms_match(
        baseline, expected_terms
    )
    rag_terms_found = terms_match(
        rag, expected_terms
    )
    evidence_terms_found = terms_match(
        retrieved_text, expected_terms
    )

    # A simple reproducible proxy, not a full semantic judge.
    # For unknown cases, require the retrieved evidence to say unknown
    # and check whether the answer avoids confidently inventing a cause.
    unknown_markers = [
        "unknown", "not specified", "does not specify",
        "cannot determine", "not provided", "insufficient information"
    ]

    if not answer_known:
        rag_abstains = any(
            marker in rag_norm for marker in unknown_markers
        )
        baseline_score = ""
        rag_score = int(rag_abstains and evidence_terms_found)
    else:
        baseline_score = int(baseline_terms_found)
        rag_score = int(rag_terms_found and evidence_terms_found)

    rows.append({
        "fault_id": fault_id,
        "question": question,
        "expected_terms": " | ".join(expected_terms),
        "answer_known": answer_known,
        "baseline_answer": baseline,
        "baseline_latency_seconds": round(baseline_time, 3),
        "baseline_keyword_match": baseline_score,
        "rag_answer": rag,
        "rag_latency_seconds": round(rag_time, 3),
        "rag_keyword_or_abstention_match": rag_score,
        "evidence_keyword_match": int(evidence_terms_found),
        "retrieved_sources": sources,
        "retrieved_text": retrieved_text
    })

    print("Baseline:", baseline)
    print("RAG:", rag)
    print("RAG match:", rag_score)


fieldnames = list(rows[0].keys())

with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

known_rows = [r for r in rows if r["answer_known"]]
unknown_rows = [r for r in rows if not r["answer_known"]]

print("\n" + "=" * 55)
print("EVALUATION SUMMARY")
print("=" * 55)
print("Total questions:", len(rows))
print("Known-answer questions:", len(known_rows))
print("Unknown-answer questions:", len(unknown_rows))

print(
    "Baseline keyword-match rate on known questions:",
    round(
        sum(r["baseline_keyword_match"] for r in known_rows)
        / max(len(known_rows), 1), 3
    )
)
print(
    "RAG keyword-and-evidence match rate on known questions:",
    round(
        sum(r["rag_keyword_or_abstention_match"] for r in known_rows)
        / max(len(known_rows), 1), 3
    )
)
print(
    "Unknown-case abstention/evidence match rate:",
    round(
        sum(r["rag_keyword_or_abstention_match"] for r in unknown_rows)
        / max(len(unknown_rows), 1), 3
    )
)
print("Results saved to:", OUTPUT_CSV)
