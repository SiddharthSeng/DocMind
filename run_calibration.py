import json
from vector_store import VectorStore
from generation import ABSTAIN_THRESHOLD, MIN_SIMILARITY_THRESHOLD

def main():
    store = VectorStore()
    
    with open('test_cases.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    print(f"--- Running Calibration Batch ({len(test_cases)} queries) ---\n")
    print(f"Current MIN_SIMILARITY_THRESHOLD: {MIN_SIMILARITY_THRESHOLD}")
    print(f"Current ABSTAIN_THRESHOLD: {ABSTAIN_THRESHOLD}\n")
    
    for idx, tc in enumerate(test_cases, 1):
        print(f"[{idx}/{len(test_cases)}] Type: {tc['type']} | Query: '{tc['query']}'")
        
        chunks = store.retrieve(tc['query'], top_k=5)

        if not chunks:
            print("  [!] No chunks returned at all from ChromaDB.")
            continue

        top_score = chunks[0]['similarity_score']

        if tc['expected_chunk_id'] is None:
            # Unrelated query: we want the abstain threshold to fire
            would_abstain = top_score < ABSTAIN_THRESHOLD
            print(f"  -> Top fused score: {top_score:.4f}  (embed={chunks[0]['embed_score']:.4f}, bm25_rrf={chunks[0]['bm25_rrf']:.6f})")
            print(f"  -> Would abstain?   {'YES (Correct)' if would_abstain else 'NO — threshold not triggered'}")
            if not would_abstain:
                print(f"     (Top chunk: {chunks[0]['source_file']}, fused={top_score:.4f})")
        else:
            expected_id = tc['expected_chunk_id']
            found_rank  = -1
            found_chunk = None

            for i, chunk in enumerate(chunks):
                if chunk['chunk_id'] == expected_id:
                    found_rank  = i + 1
                    found_chunk = chunk
                    break

            if found_chunk is not None:
                fused  = found_chunk['similarity_score']
                embed  = found_chunk['embed_score']
                bm25r  = found_chunk['bm25_rrf']
                passes = fused >= MIN_SIMILARITY_THRESHOLD
                print(f"  -> Expected chunk at rank {found_rank}")
                print(f"     fused={fused:.4f}  embed={embed:.4f}  bm25_rrf={bm25r:.6f}")
                print(f"     Passes MIN_SIMILARITY_THRESHOLD ({MIN_SIMILARITY_THRESHOLD})? {'YES' if passes else 'NO'}")
            else:
                print(f"  -> Expected chunk {expected_id} NOT IN TOP 5")
                print(f"     Top chunk: {chunks[0]['source_file']}  fused={top_score:.4f}  embed={chunks[0]['embed_score']:.4f}  bm25_rrf={chunks[0]['bm25_rrf']:.6f}")
        print()

if __name__ == "__main__":
    main()

