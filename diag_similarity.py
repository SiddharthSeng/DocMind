from vector_store import VectorStore

store = VectorStore()

print("=== QUERY 1: 'What happens if uptime drops below 95%?' ===")
chunks = store.retrieve("What happens if uptime drops below 95%?", top_k=5)
for i, c in enumerate(chunks):
    print(f"Chunk {i}: score={c['similarity_score']:.4f}  source={c['source_file']}")
    print("Full text:")
    print(c["text"])
    print("-" * 60)

print()
print("=== QUERY 2: 'Do you use customer documents to train the LLM?' ===")
chunks2 = store.retrieve("Do you use customer documents to train the LLM?", top_k=5)
for i, c in enumerate(chunks2):
    print(f"Chunk {i}: score={c['similarity_score']:.4f}  source={c['source_file']}")
    print("Full text:")
    print(c["text"])
    print("-" * 60)
