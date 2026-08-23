import json
import re
import chromadb
import httpx
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# A retrieval-optimized embedding model running locally via Ollama.
# nomic-embed-text is specifically trained for RAG and handles semantic paraphrasing well.
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)
# Ollama local inference can sometimes exceed the default 5-second httpx timeout, so we increase it.
embedding_function._session = httpx.Client(timeout=120)

# ---------------------------------------------------------------------------
# RRF constant — the key tuning parameter for Reciprocal Rank Fusion.
# ---------------------------------------------------------------------------
# How RRF works:
#   Each retriever (embedding, BM25) independently ranks all candidate chunks.
#   The fused score for a chunk is the *sum* of 1/(K + rank) across each list
#   it appears in. A chunk missing from a list contributes 0 for that list.
#
# The K constant dampens rank differences. Smaller K = bigger gap between
# rank 1 and rank 5; larger K = ranks matter less and all chunks bunch together.
#
# Standard IR papers use K=60 for large corpora. For a small corpus (5 chunks),
# K=60 makes all ranks nearly identical (1/61 vs 1/65 = negligible). We use
# K=10 to keep rank differences meaningful at this scale:
#   Rank 1:  1/(10+1) = 0.0909
#   Rank 5:  1/(10+5) = 0.0667
#   No BM25: 0        (critical — zero keyword overlap = zero BM25 contribution)
#
# This means a genuine answer (high embedding AND BM25 rank) scores ~0.15+,
# while domain-adjacent noise (high embedding, zero BM25) scores only ~0.07.
# That gap is the clean threshold boundary we couldn't get from embeddings alone.
#
# TUNING: if you add more documents and the corpus grows >50 chunks, increase
# K toward 30-60 to prevent rank 1 from dominating too aggressively.
RRF_K = 10


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25. Lowercase only."""
    return re.findall(r"[a-z0-9]+", text.lower())


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        chunks_json: str = "processed_chunks.json",
    ):
        """
        Initialises the ChromaDB client, builds an in-memory BM25 index from
        processed_chunks.json, and prepares both retrievers for hybrid search.

        Args:
            persist_directory: Where ChromaDB persists the vector index.
            collection_name:   Name of the ChromaDB collection.
            chunks_json:       Path to the JSON produced by data_ingestion.py.
                               Used to build the BM25 index at startup.
        """
        # ---- ChromaDB (embedding retriever) --------------------------------
        # Create a persistent client that saves our vector database to disk.
        self.client = chromadb.PersistentClient(path=persist_directory)

        # ---------------------------------------------------------
        # Why Cosine Similarity vs. Euclidean Distance?
        # ---------------------------------------------------------
        # Euclidean distance measures the straight-line distance between two points in space.
        # This makes it highly sensitive to the magnitude (length) of the vectors.
        # Two documents might be identical in meaning, but if one uses more words or slightly
        # different phrasing that scales the vector, their Euclidean distance could be large.
        #
        # Cosine similarity measures the *angle* between two vectors. It completely ignores
        # the magnitude (length) and only cares about the direction. In NLP, the direction
        # represents the semantic meaning. Because RAG is about capturing the semantic intent
        # regardless of document length, Cosine Similarity is the industry standard for text matching.
        # Note: nomic-embed-text outputs normalized vectors, so Cosine Similarity mathematically
        # becomes equivalent to the Inner Product, making it very fast to compute!
        #
        # Here we tell ChromaDB to use Cosine distance (which is 1 - Cosine Similarity).
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        # ---- BM25 (keyword retriever) --------------------------------------
        # BM25 is a probabilistic keyword-frequency scorer (an improved TF-IDF).
        # It does NOT understand semantics -- it only counts token overlap between
        # the query and each document. This is its strength: a query that shares
        # zero words with a chunk scores exactly 0.0, regardless of how closely
        # their embeddings cluster due to shared domain vocabulary.
        #
        # The index is built in-memory from processed_chunks.json each time
        # VectorStore initialises. It is not persisted -- BM25 is O(n) to rebuild
        # and the JSON is always the source of truth.
        self._bm25_corpus: list[dict] = []  # list of raw chunk dicts
        self._bm25_index: BM25Okapi | None = None
        self._load_bm25_index(chunks_json)

    def _load_bm25_index(self, chunks_json: str) -> None:
        """Loads processed_chunks.json and builds the in-memory BM25 index."""
        try:
            with open(chunks_json, "r", encoding="utf-8") as f:
                self._bm25_corpus = json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: {chunks_json} not found. BM25 index will be empty. "
                "Run data_ingestion.py first, then reinitialise VectorStore."
            )
            return

        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized_corpus)

    def load_and_embed_chunks(self, json_file: str):
        """
        Loads chunks from a JSON file, embeds them, and stores them in ChromaDB.
        Also rebuilds the BM25 index so both retrievers stay in sync.
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except FileNotFoundError:
            print(f"Error: {json_file} not found. Please run the data ingestion script first.")
            return

        if not chunks:
            print("No chunks found in JSON file.")
            return

        print(f"Loading {len(chunks)} chunks into ChromaDB (this might take a moment to embed)...")

        # Prepare lists for ChromaDB ingestion
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            # Store everything else as metadata so we can filter/retrieve it later
            metadatas.append({
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
            })

        # Add everything to the collection.
        # We batch in sizes of 5000 to avoid hitting SQLite payload limits on large datasets.
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )
            print(f"Ingested batch {i // batch_size + 1}")

        print("Finished loading chunks into vector store!")

        # Keep BM25 in sync with what was just embedded.
        self._load_bm25_index(json_file)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion of embedding + BM25 signals.

        Both retrievers independently rank all chunks. The fused score for each
        chunk is:

            fused(chunk) = embed_rrf(chunk) + bm25_rrf(chunk)

        where:
            embed_rrf(chunk) = 1 / (RRF_K + embedding_rank)   [always present]
            bm25_rrf(chunk)  = 1 / (RRF_K + bm25_rank)        [0 if BM25 score == 0]

        A chunk with no keyword overlap at all (BM25 score == 0) contributes
        nothing from the BM25 channel, suppressing domain-clustered false matches
        that look similar only because they share product vocabulary.

        Returns:
            List of chunk dicts sorted by fused score descending, capped at top_k.
            Each dict has: chunk_id, text, source_file, chunk_index,
                           similarity_score (= fused RRF score),
                           embed_score (raw cosine), bm25_rrf (BM25 channel contribution).
        """
        n_corpus = self.collection.count()
        if n_corpus == 0:
            return []

        # ---- 1. Embedding ranking (ChromaDB) --------------------------------
        # Retrieve ALL chunks so we have a complete ranking for RRF, not just top_k.
        embed_results = self.collection.query(
            query_texts=[query],
            n_results=n_corpus,
        )

        if not embed_results["documents"] or not embed_results["documents"][0]:
            return []

        # Build a lookup: chunk_id -> {rank (1-based), cosine similarity, text, metadata}
        embed_lookup: dict[str, dict] = {}
        for rank, (chunk_id, doc, meta, dist) in enumerate(
            zip(
                embed_results["ids"][0],
                embed_results["documents"][0],
                embed_results["metadatas"][0],
                embed_results["distances"][0],
            ),
            start=1,
        ):
            embed_lookup[chunk_id] = {
                "rank": rank,
                "text": doc,
                "source_file": meta["source_file"],
                "chunk_index": meta["chunk_index"],
                "cosine_similarity": 1.0 - dist,
            }

        # ---- 2. BM25 ranking ------------------------------------------------
        # Get raw BM25 scores for every chunk in the corpus.
        # Chunks with score == 0 have zero keyword overlap and contribute nothing.
        bm25_rrf_map: dict[str, float] = {}
        if self._bm25_index is not None:
            tokenized_query = _tokenize(query)
            raw_scores = self._bm25_index.get_scores(tokenized_query)

            # Only rank chunks that actually scored > 0 (real keyword matches).
            scored = [
                (chunk["chunk_id"], score)
                for chunk, score in zip(self._bm25_corpus, raw_scores)
                if score > 0
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            for bm25_rank, (chunk_id, _) in enumerate(scored, start=1):
                bm25_rrf_map[chunk_id] = 1.0 / (RRF_K + bm25_rank)
            # Chunks absent from `scored` stay at 0 -- no BM25 contribution.

        # ---- 3. RRF fusion --------------------------------------------------
        all_chunk_ids = set(embed_lookup.keys())
        fused_chunks = []

        for chunk_id in all_chunk_ids:
            embed_rrf   = 1.0 / (RRF_K + embed_lookup[chunk_id]["rank"])
            bm25_rrf    = bm25_rrf_map.get(chunk_id, 0.0)
            fused_score = embed_rrf + bm25_rrf

            fused_chunks.append({
                "chunk_id":         chunk_id,
                "text":             embed_lookup[chunk_id]["text"],
                "source_file":      embed_lookup[chunk_id]["source_file"],
                "chunk_index":      embed_lookup[chunk_id]["chunk_index"],
                # similarity_score is the fused RRF value -- this is what generation.py
                # filters on. The structural gate still operates; the signal has improved.
                "similarity_score": round(fused_score, 6),
                "embed_score":      round(embed_lookup[chunk_id]["cosine_similarity"], 4),
                "bm25_rrf":         round(bm25_rrf, 6),
            })

        fused_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
        return fused_chunks[:top_k]


if __name__ == "__main__":
    # Example usage:
    # 1. Initialise our vector store (also builds BM25 index automatically)
    store = VectorStore()

    # 2. Load the JSON we generated in step 1.
    # We check if collection is empty before adding to avoid duplicate errors on multiple runs.
    if store.collection.count() == 0:
        store.load_and_embed_chunks("processed_chunks.json")
    else:
        print(f"Collection already contains {store.collection.count()} chunks. Skipping ingestion.")

    # 3. Test retrieval
    test_query = "What is the main topic of these documents?"
    print(f"\nSearching for: '{test_query}'")

    matches = store.retrieve(test_query, top_k=2)

    if not matches:
        print("No matches found. Did you add documents to the docs folder before running the ingestion script?")

    for idx, match in enumerate(matches):
        print(
            f"\n--- Match {idx+1} "
            f"(Fused: {match['similarity_score']:.4f} | "
            f"Embed: {match['embed_score']:.4f} | "
            f"BM25-RRF: {match['bm25_rrf']:.4f}) ---"
        )
        print(f"Source: {match['source_file']} (Chunk {match['chunk_index']})")
        snippet = match["text"][:150].replace("\n", " ")
        print(f"Snippet: {snippet}...")
