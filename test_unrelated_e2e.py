"""
End-to-end test for the three unrelated queries.
Runs the full pipeline: retrieve -> generation.py -> print answer/sources/warning.
"""
import json
from vector_store import VectorStore
from generation import generate_answer

UNRELATED_QUERIES = [
    "How do I reset my password for the DocMind dashboard?",
    "What are the exact monthly prices for the Enterprise and Pro tiers?",
    "Does the DocMind API support streaming responses via Server-Sent Events?",
]

def main():
    store = VectorStore()
    
    for i, query in enumerate(UNRELATED_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/3] QUERY: {query}")
        print('='*70)
        
        chunks = store.retrieve(query, top_k=5)
        
        # Show what the retriever returned
        print(f"\n--- Retrieval (top 3 shown) ---")
        for c in chunks[:3]:
            print(f"  [{c['similarity_score']:.4f}] {c['source_file']} | embed={c['embed_score']:.4f} bm25_rrf={c['bm25_rrf']:.6f}")
            print(f"    \"{c['text'][:80].strip().replace(chr(10),' ')}...\"")
        
        print(f"\n--- Generation ---")
        result = generate_answer(chunks, query)
        
        print(f"ANSWER:       {result['answer']}")
        print(f"SOURCES:      {result['sources_cited']}")
        print(f"CONFIDENCE:   {result['confidence']}")
        print(f"WARNING:      {result.get('warning', '(none)')}")

if __name__ == "__main__":
    main()
