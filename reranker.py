"""
reranker.py — Cross-encoder reranking for DocMind retrieval
============================================================
The hybrid retriever (bge-small + BM25, see vector_store.py) is good at
getting the answering chunk into its top candidates but weaker at putting it
first.  A cross-encoder reads the question and each candidate chunk TOGETHER
and scores whether the chunk answers the question, which is far more precise
than comparing two independently computed embeddings.

The model is DocMind's own fine-tune of cross-encoder/ms-marco-MiniLM-L-12-v2
(training/01_reranker_finetune.ipynb), exported to ONNX.  It runs on
onnxruntime + tokenizers, so serving needs no PyTorch — the same reason the
embedder runs through fastembed.

Expected model directory (as produced by the notebook):
    model.onnx               exported sequence-classification model (1 logit)
    tokenizer.json           fast tokenizer
    docmind_reranker.json    {"max_length": ..., "input_names": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_rerankers: dict[str, "Reranker"] = {}


class Reranker:
    def __init__(self, model_dir: str | Path, batch_size: int = 16):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        config = json.loads((model_dir / "docmind_reranker.json").read_text(encoding="utf-8"))
        self.input_names = config["input_names"]
        self.batch_size = batch_size

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # Long chunks lose their tail, never the question ("only_second").
        self.tokenizer.enable_truncation(max_length=config["max_length"], strategy="only_second")
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.label = config.get("label", model_dir.name)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Returns one relevance logit per passage (higher = more likely to answer the query)."""
        scores = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encodings = self.tokenizer.encode_batch([(query, p) for p in batch])
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            logits = self.session.run(None, {k: feeds[k] for k in self.input_names})[0]
            scores.append(logits.reshape(-1))
        return np.concatenate(scores) if scores else np.array([])

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """
        Re-orders retrieved chunks by cross-encoder score and keeps the best top_k.
        Each returned chunk gains a "rerank_score" (raw logit; sigmoid gives a
        0-1 probability that the chunk answers the query).
        """
        if not chunks:
            return []
        scores = self.score(query, [c["text"] for c in chunks])
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        return [{**chunk, "rerank_score": round(float(score), 4)} for score, chunk in ranked[:top_k]]


def get_reranker(model_dir: str | Path) -> Reranker:
    """Process-wide cache: loading the ONNX session takes ~1 s, so do it once per model."""
    key = str(Path(model_dir).resolve())
    if key not in _rerankers:
        _rerankers[key] = Reranker(model_dir)
    return _rerankers[key]
