
from pathlib import Path
import requests
import numpy as np
import  re

# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "embeddinggemma"

PROJECT_DIR = Path(
    r"C:\Users\Furkan-KALKAN\Desktop\TEP_PROJECT"
)
KB_DIR = PROJECT_DIR / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_URL = (
    "https://huggingface.co/spaces/jkitchin/"
    "tennessee-eastman-process/blob/main/data/README.md"
)

# --------------------------------------------------
# 2. Create starter knowledge documents
# --------------------------------------------------

documents = {
    "tep_overview.txt": f"""
Source: Tennessee Eastman Process reference README
URL: {SOURCE_URL}

The Tennessee Eastman Process (TEP) is a simulated
industrial process used in process monitoring and
fault detection research.

The reference dataset contains 52 variables per
observation. The first 41 variables are measurements
(XMEAS 1-41). The remaining 11 variables are
manipulated variables (XMV 1-11).

The sampling interval is 180 seconds, equivalent
to three minutes.

The dataset contains normal-operation files and
fault-scenario files. Training fault files contain
480 samples, while test fault files contain
960 samples according to the reference README.

A fault ID identifies a simulated fault scenario.
IDV0 is used in this project to represent normal
operation; IDV1 through IDV21 represent fault
scenarios in the available dataset.
""",
    "tep_faults.txt": f"""
Source: Tennessee Eastman Process reference README
URL: {SOURCE_URL}

IDV1: A/C feed ratio, with B composition constant
in Stream 4. Type: step.

IDV2: B composition, with A/C ratio constant
in Stream 4. Type: step.

IDV3: D feed temperature in Stream 2.
Type: step.

IDV4: Reactor cooling-water inlet temperature.
Type: step.

IDV5: Condenser cooling-water inlet temperature.
Type: step.

IDV6: A feed loss in Stream 1.
Type: step.

IDV7: C header pressure loss in Stream 4.
Type: step.

IDV8: A, B and C feed composition in Stream 4.
Type: random.

IDV9: D feed temperature in Stream 2.
Type: random.

IDV10: C feed temperature in Stream 4.
Type: random.

IDV11: Reactor cooling-water inlet temperature.
Type: random.

IDV12: Condenser cooling-water inlet temperature.
Type: random.

IDV13: Reaction kinetics. Type: slow drift.

IDV14: Reactor cooling-water valve.
Type: sticking.

IDV15: Condenser cooling-water valve.
Type: sticking.

IDV16, IDV17, IDV18, IDV19 and IDV20:
The reference README lists these fault types as unknown.
Do not invent their physical causes.

IDV21: Valve position constant in Stream 4.
The README does not specify a fault type.
"""
}

for filename, content in documents.items():
    (KB_DIR / filename).write_text(
        content.strip(), encoding="utf-8"
    )

# --------------------------------------------------
# 3. Read and chunk the knowledge base
# --------------------------------------------------

chunks = []

for file_path in sorted(KB_DIR.glob("*.txt")):
    content = file_path.read_text(encoding="utf-8")
    paragraphs = [
        p.strip()
        for p in content.split("\n\n")
        if p.strip()
    ]

    for paragraph in paragraphs:
        chunks.append({
            "text": paragraph,
            "source": file_path.name
        })

if not chunks:
    raise RuntimeError("Knowledge base is empty.")

# --------------------------------------------------
# 4. Create embeddings using Ollama
# --------------------------------------------------

def get_embeddings(texts):
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["embeddings"]


print("Creating document embeddings...")

vectors = np.array(
    get_embeddings([chunk["text"] for chunk in chunks]),
    dtype=np.float32
)

# Normalize vectors for cosine similarity
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.maximum(norms, 1e-12)

print(f"Indexed {len(chunks)} text chunks.")

# --------------------------------------------------
# 5. Retrieve relevant passages
# --------------------------------------------------


def retrieve(query, top_k=3):
    query_vector = np.array(
        get_embeddings([query])[0],
        dtype=np.float32
    )
    query_vector /= max(np.linalg.norm(query_vector), 1e-12)

    # Semantic similarity
    scores = vectors @ query_vector

    # Exact technical identifier matching, e.g. IDV4
    query_ids = set(
        re.findall(r"\bIDV\d+\b", query.upper())
    )

    if query_ids:
        for i, chunk in enumerate(chunks):
            text_upper = chunk["text"].upper()

            chunk_ids = set(
                re.findall(r"\bIDV\d+\b", text_upper)
            )

            # Strongly prioritize passages mentioning
            # the exact requested fault identifier.
            if query_ids.intersection(chunk_ids):
                scores[i] += 1.0

    best_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (chunks[i], float(scores[i]))
        for i in best_indices
    ]


# --------------------------------------------------
# 6. Compare baseline LLM with RAG
# --------------------------------------------------

def call_llm(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def ask_baseline(question):
    prompt = f"""
You answer questions about industrial processes.

Answer the question using your existing knowledge.
Do not claim to have consulted external documents.
If you are unsure, explicitly say so.

Question: {question}
"""
    return call_llm(prompt)


def ask_rag(question):
    results = retrieve(question, top_k=3)

    evidence = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}"
        for item, score in results
    )

    prompt = f"""
You are an assistant for industrial fault diagnosis.

Answer using the supplied evidence.
Do not contradict information explicitly stated in it.
If the evidence states a fault ID, variable and type,
report those facts directly and clearly.
Do not invent additional physical causes.
If a detail is absent, say only that this detail is
not specified. Do not say a known detail is unknown.
Cite the source filename for your answer.

EVIDENCE:
{evidence}

QUESTION:
{question}
"""
    answer = call_llm(prompt)

    print("\nRETRIEVED PASSAGES:")
    for item, score in results:
        print(f"- {item['source']} | score={score:.3f}")
        print(item["text"], "\n")

    return answer


if __name__ == "__main__":
    question = (
        "What is IDV4 in the Tennessee Eastman Process, "
        "and what type of fault is it?"
    )

    print("=" * 60)
    print("A. BASELINE LLM")
    print("=" * 60)
    print(ask_baseline(question))

    print("\n" + "=" * 60)
    print("B. RETRIEVAL-AUGMENTED LLM")
    print("=" * 60)
    print(ask_rag(question))
