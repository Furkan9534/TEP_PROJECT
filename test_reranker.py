
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Reranker modeli yükleniyor...")

model = CrossEncoder(MODEL_NAME)

question = "What is the fault type of IDV13?"

candidates = [
    "IDV13: Reaction kinetics. Fault type: slow drift.",
    "IDV4: Reactor cooling-water inlet temperature. Fault type: step.",
    "IDV14: Reactor cooling-water valve. Fault type: sticking.",
]

pairs = [(question, text) for text in candidates]
scores = model.predict(pairs)

ranked_results = sorted(
    zip(scores, candidates),
    key=lambda item: item[0],
    reverse=True,
)

print("\nReranking results:")
for score, text in ranked_results:
    print(f"Score: {float(score):.4f} | {text}")

print("\nReranker test completed.")
