import os
from vector_store import VectorStore
from generation import generate_answer

def main():
    print("Initializing VectorStore...")
    store = VectorStore()
    
    print("Loading chunks into vector store from processed_chunks.json...")
    store.load_and_embed_chunks('processed_chunks.json')
    
    questions = [
        "What happens if uptime drops below 95%?",
        "Do you use customer documents to train the LLM?"
    ]
    
    for q in questions:
        print(f"\n--- QUERY: {q} ---")
        chunks = store.retrieve(q, top_k=3)
        for i, c in enumerate(chunks):
            print(f"CHUNK {i} [{c['similarity_score']:.3f}]: {c['text'][:100]}...")
        res = generate_answer(chunks, q)
        print(f"ANSWER:\n{res['answer']}\n")
        print(f"SOURCES CITED:\n{res['sources_cited']}\n")
        print(f"CONFIDENCE:\n{res['confidence']}\n")

if __name__ == '__main__':
    main()
