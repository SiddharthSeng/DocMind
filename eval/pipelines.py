"""
pipelines.py — Pluggable RAG pipelines for the gold-set evaluation
===================================================================
Every pipeline implements the same three methods, so run_eval.py can score
the current production stack and each improved variant on identical inputs:

    build_index(doc_path, workdir) -> index   # one index per document, like an /upload session
    retrieve(index, question)      -> list[dict] with at least a "text" key, best first
    answer(index, question, chunks) -> dict with at least an "answer" key

Register new variants in PIPELINES at the bottom of this file.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class BaselinePipeline:
    """
    The pipeline exactly as api.py runs it today:
      data_ingestion.process_documents  (500-token tiktoken windows, 50 overlap — api.py defaults)
      VectorStore.retrieve              (bge-small + BM25, RRF K=10, top_k=5)
      generation.generate_answer        (cosine gate + LLM_BACKEND from .env)
    """

    name = "baseline"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def build_index(self, doc_path: Path, workdir: Path):
        from data_ingestion import process_documents
        from vector_store import VectorStore

        docs_dir = workdir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(doc_path, docs_dir / doc_path.name)
        chunks_json = workdir / "processed_chunks.json"

        process_documents(input_dir=str(docs_dir), output_file=str(chunks_json))
        store = VectorStore(
            persist_directory=str(workdir / "chroma_db"),
            chunks_json=str(chunks_json),
        )
        store.load_and_embed_chunks(str(chunks_json))
        return store

    def retrieve(self, index, question: str) -> list[dict]:
        return index.retrieve(question, top_k=self.top_k)

    def answer(self, index, question: str, chunks: list[dict]) -> dict:
        from generation import generate_answer

        return generate_answer(chunks, question)


class StructuredPipeline:
    """
    Baseline with the no-training fixes found by the gold-set analysis:
      ingestion   data_ingestion.ingest_documents (PDFium extraction, DOCX
                  tables, sentence-aware chunks sized with bge's tokenizer)
      retrieval   bge query instruction, configurable RRF K and top_k
      generation  cosine gate off by default, Ollama context raised to 8192

    Every knob is a constructor argument so run_eval.py --opt key=value can
    ablate one change at a time.
    """

    name = "structured"

    def __init__(self, top_k: int = 5, max_tokens: int = 300, overlap_tokens: int = 50,
                 query_instruction: bool = False, rrf_k: int = 10, use_gate: bool = False,
                 num_ctx: int = 8192, reranker: str = "", candidates: int = 30):
        # query_instruction defaults off: on the gold set it changed hit@1 by
        # -0.4 points (66.7% vs 67.1%), i.e. no measurable benefit.
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.query_instruction = query_instruction
        self.rrf_k = rrf_k
        self.use_gate = use_gate
        # reranker: path to an exported ONNX reranker (e.g. models/reranker-finetuned);
        # the hybrid retriever then supplies `candidates` chunks for it to re-order.
        self.reranker = None
        if reranker:
            from reranker import get_reranker
            self.reranker = get_reranker(reranker)
        self.candidates = candidates
        import os
        os.environ["OLLAMA_NUM_CTX"] = str(num_ctx)

    def build_index(self, doc_path: Path, workdir: Path):
        import rag_pipeline

        docs_dir = workdir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(doc_path, docs_dir / doc_path.name)
        store, _ = rag_pipeline.index_documents(
            docs_dir, workdir / "processed_chunks.json", workdir / "chroma_db",
            max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens,
            query_instruction=self.query_instruction, rrf_k=self.rrf_k,
        )
        return store

    def retrieve(self, index, question: str) -> list[dict]:
        import rag_pipeline

        return rag_pipeline.retrieve(index, question, top_k=self.top_k,
                                     reranker=self.reranker, candidates=self.candidates)

    def answer(self, index, question: str, chunks: list[dict]) -> dict:
        import rag_pipeline
        from generation import generate_answer

        if self.use_gate:
            return generate_answer(chunks, question, use_gate=True)
        return rag_pipeline.answer(question, chunks)


class AppPipeline(StructuredPipeline):
    """
    Exactly what api.py runs: rag_pipeline with its defaults, including the
    reranker at RERANKER_DIR when it is installed.  The answer model is
    whatever LLM_BACKEND / OLLAMA_MODEL select.
    """

    name = "app"

    def __init__(self):
        import rag_pipeline

        super().__init__(top_k=rag_pipeline.TOP_K, candidates=rag_pipeline.RERANK_CANDIDATES)
        self.reranker = rag_pipeline.default_reranker()


PIPELINES = {
    BaselinePipeline.name: BaselinePipeline,
    StructuredPipeline.name: StructuredPipeline,
    AppPipeline.name: AppPipeline,
}
