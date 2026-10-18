"""
rag_pipeline.py — DocMind's retrieval-augmented answering pipeline
===================================================================
The single place that decides how a document is indexed and how a question
is answered.  api.py serves it, and the gold-set evaluation scores it
(eval/pipelines.py, pipeline "structured"), so the numbers in eval/ describe
exactly what the app runs.

    index_documents()  extract (PDFium / DOCX tables) -> sentence-aware chunks
                       sized with bge's tokenizer -> embed into ChromaDB + BM25
    retrieve()         hybrid bge + BM25 (RRF) candidates -> fine-tuned
                       cross-encoder reranker (when its model is present)
    answer()           LLM over the top chunks; no cosine gate — the gate was
                       calibrated on the 5-chunk demo corpus and refused
                       answerable questions on real documents

Gold set (261 questions, 8 documents): 93.5% answer accuracy with llama3.1 via
Ollama, against 78.5% for the previous pipeline.

Configuration (environment):
    RERANKER_DIR        exported ONNX reranker (default models/reranker-finetuned);
                        if the folder is missing, retrieval runs without reranking
    RETRIEVAL_TOP_K     chunks passed to the LLM (default 5)
    RERANK_CANDIDATES   hybrid-retrieval candidates the reranker re-orders (default 30)
    OLLAMA_NUM_CTX      Ollama context window (default 8192; Ollama's own 4096
                        default can silently truncate the prompt)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from data_ingestion import ingest_documents
from generation import generate_answer
from vector_store import VectorStore, embedding_token_counter

BASE_DIR = Path(__file__).parent.absolute()

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
RERANKER_DIR = Path(os.getenv("RERANKER_DIR", BASE_DIR / "models" / "reranker-finetuned"))
CHUNK_MAX_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

os.environ.setdefault("OLLAMA_NUM_CTX", "8192")


def index_documents(
    docs_dir: str | Path,
    chunks_json: str | Path,
    chroma_dir: str | Path,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    **store_options,
) -> tuple[VectorStore, list[dict]]:
    """Chunks every document in docs_dir, embeds the chunks, and returns the store and the chunks."""
    chunks = ingest_documents(
        str(docs_dir), str(chunks_json),
        count_tokens=embedding_token_counter(),
        max_tokens=max_tokens, overlap_tokens=overlap_tokens,
    )
    store = open_store(chroma_dir, chunks_json, **store_options)
    if chunks:
        store.load_and_embed_chunks(str(chunks_json))
    return store, chunks


def open_store(chroma_dir: str | Path, chunks_json: str | Path, **store_options) -> VectorStore:
    """store_options (query_instruction, rrf_k) exist for evaluation ablations; the app uses the defaults."""
    return VectorStore(persist_directory=str(chroma_dir), chunks_json=str(chunks_json), **store_options)


_default_reranker = None


def default_reranker():
    """The reranker at RERANKER_DIR, loaded once; None when the model isn't installed."""
    global _default_reranker
    if _default_reranker is None and (RERANKER_DIR / "model.onnx").exists():
        from reranker import get_reranker
        _default_reranker = get_reranker(RERANKER_DIR)
    return _default_reranker


_USE_DEFAULT = object()


def retrieve(store: VectorStore, question: str, top_k: int = TOP_K,
             reranker=_USE_DEFAULT, candidates: int = RERANK_CANDIDATES) -> list[dict]:
    """Hybrid retrieval, re-ordered by the reranker when one is available (pass reranker=None to skip)."""
    if reranker is _USE_DEFAULT:
        reranker = default_reranker()
    if reranker is None:
        return store.retrieve(question, top_k=top_k)
    pool = store.retrieve(question, top_k=max(candidates, top_k))
    return reranker.rerank(question, pool, top_k=top_k)


def answer(question: str, chunks: list[dict]) -> dict:
    return generate_answer(chunks, question, use_gate=False)


def load_chunks(chunks_json: str | Path) -> list[dict]:
    with open(chunks_json, encoding="utf-8") as f:
        return json.load(f)
