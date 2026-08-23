import json
import chromadb
from chromadb.utils import embedding_functions

# A retrieval-optimized embedding model running locally via Ollama.
# nomic-embed-text is specifically trained for RAG and handles semantic paraphrasing well.
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text",
)
# Ollama local inference can sometimes exceed the default 5-second httpx timeout, so we increase it
import httpx
embedding_function._session = httpx.Client(timeout=120)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "rag_documents"):
        """
        Initializes the ChromaDB client and gets or creates a collection.
        """
        # Create a persistent client that saves our vector database to the local disk
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
            metadata={"hnsw:space": "cosine"} # "cosine" space in Chroma calculates cosine distance
        )

    def load_and_embed_chunks(self, json_file: str):
        """
        Loads chunks from a JSON file, embeds them, and stores them in ChromaDB.
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
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
                "chunk_index": chunk["chunk_index"]
            })

        # Add everything to the collection. 
        # The SentenceTransformerEmbeddingFunction will automatically embed the documents.
        # We batch this in sizes of 5000 to avoid hitting SQLite payload limits on large datasets.
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )
            print(f"Ingested batch {i // batch_size + 1}")

        print("Finished loading chunks into vector store!")

    def retrieve(self, query: str, top_k: int = 5):
        """
        Embeds the query and returns the top_k most similar chunks.
        """
        # The collection automatically uses the same embedding function to embed the query
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Chroma returns a complex dictionary of parallel lists. 
        # Let's parse it into a friendly list of dictionaries format.
        retrieved_chunks = []
        
        # results["documents"][0] contains the matched texts for the first query
        # results["metadatas"][0] contains the metadata for those matches
        # results["distances"][0] contains the distance score (lower distance = higher similarity)
        
        if not results["documents"] or not results["documents"][0]:
            return []

        for i in range(len(results["documents"][0])):
            retrieved_chunks.append({
                "text": results["documents"][0][i],
                "source_file": results["metadatas"][0][i]["source_file"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "similarity_score": 1.0 - results["distances"][0][i]  # Convert distance back to similarity
            })

        return retrieved_chunks


if __name__ == "__main__":
    # Example usage:
    # 1. Initialize our vector store
    store = VectorStore()
    
    # 2. Load the JSON we generated in step 1
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
        print(f"\n--- Match {idx+1} (Similarity: {match['similarity_score']:.4f}) ---")
        print(f"Source: {match['source_file']} (Chunk {match['chunk_index']})")
        # Print first 150 characters of the matched chunk
        snippet = match['text'][:150].replace('\n', ' ')
        print(f"Snippet: {snippet}...")
