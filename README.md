# DocMind — RAG-based Document Q&A System

A privacy-preserving **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions about your own documents (PDF, DOCX, TXT). Everything runs on your machine: an in-process embedder, a **fine-tuned reranker**, and an open LLM served by **Ollama** — no hosted LLM API required.

**Answer accuracy: 93.5%** on a 261-question gold set over 8 documents (the previous pipeline scored 78.5% on the same set). See [Evaluation](#evaluation).

---

## How it works

All of this lives in [`rag_pipeline.py`](rag_pipeline.py); `api.py` serves it and the evaluation harness scores the same code.

1. **Extraction** ([`data_ingestion.py`](data_ingestion.py) `extract_pages`) — PDFium (`pypdfium2`) for PDFs, which keeps reading order and word spacing on multi-column layouts; DOCX paragraphs *and tables* in document order; page numbers kept for citations.
2. **Chunking** (`chunk_pages`) — whole sentences packed up to 300 tokens, counted with the embedder's own tokenizer so nothing overflows its 512-token window; new chunks start at section headings; each chunk is labelled with its section.
3. **Hybrid retrieval** ([`vector_store.py`](vector_store.py)) — `BAAI/bge-small-en-v1.5` (fastembed, ONNX) embeddings in ChromaDB plus BM25 keyword scores, fused with Reciprocal Rank Fusion into 30 candidates.
4. **Reranking** ([`reranker.py`](reranker.py)) — DocMind's fine-tuned cross-encoder reads the question with each candidate and keeps the best 5 (ONNX, no PyTorch).
5. **Answering** ([`generation.py`](generation.py)) — the LLM answers only from the passages, cites its sources, and replies *"I don't have enough information in the provided documents to answer this question."* when they don't contain the answer. Cited file names are verified against the retrieved chunks.

## Custom models

Training runs on Google Colab (free T4) from the notebooks in [`training/`](training/); none of the evaluation documents are used for training.

| Model | Notebook | Base | Training | Result |
|---|---|---|---|---|
| Reranker | `01_reranker_finetune.ipynb` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | SQuAD 2.0 with *same-document* negatives (DocMind always searches inside one document), 146k pairs | right passage ranked 1st: 67.1% → 77.8% (base reranker 76.4%) |
| Answer model | `02_answer_model_finetune.ipynb` | `Qwen2.5-7B-Instruct` (QLoRA → GGUF → Ollama) | DocMind's exact prompt format; grounded answers, refusals, tables | v1: 88.5% (base Qwen 90.4%) — learned terse fragments; **v2** (full-sentence targets, table examples) pending |

The app's default answer model is `llama3.1` via Ollama, the best measured option so far. A fine-tuned answer model is registered with `ollama create docmind-qwen -f training/Modelfile` and selected with `OLLAMA_MODEL=docmind-qwen`.

## Evaluation

[`eval/`](eval/) scores **answer accuracy** — not just retrieval — on a gold set of 261 questions over 8 documents: three student reports (PDF/DOCX), the two demo policies, and three public PDFs (Qwen2.5 technical report, NIST AI 600-1, the two-column RAGAS paper). Question types: factoid, numeric, list, paraphrase, multi-hop, table, summary, and 34 unanswerable decoys that must be refused.

| Pipeline (llama3.1 via Ollama) | Accuracy | Unanswerable refused | False refusals |
|---|---|---|---|
| Previous (pdfplumber, 500-token windows, cosine gate) | 78.5% | 97.1% | 15.9% |
| Current (`rag_pipeline.py`) | **93.5%** | 100% | 1.8% |

```bash
python eval/run_eval.py --validate                     # check gold evidence exists in each document
python eval/run_eval.py --pipeline app --retrieval-only # retrieval metrics, no LLM
python eval/run_eval.py --pipeline app --llm-backend ollama
python eval/run_eval.py --rescore eval/results/*.json   # re-score saved runs after a gold-set fix
```

The gold set, results and downloaded test PDFs are gitignored because they quote private documents. Scoring rules (normalised token recall, list/key-point/summary handling, refusal detection) are documented in [`eval/scoring.py`](eval/scoring.py).

## Getting started

> **Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) with a model pulled (`ollama pull llama3.1`).

```bash
pip install -r requirements.txt
cp .env.example .env          # defaults: EMBEDDING_BACKEND=bge_fastembed, LLM_BACKEND=ollama
python -m uvicorn api:app --host 0.0.0.0 --port 8000
python -m streamlit run app.py   # separate terminal
```

Upload a **PDF, DOCX or TXT** file in the sidebar to start a private session. To enable the reranker, run `training/01_reranker_finetune.ipynb` and unzip its output into `models/` (`models/reranker-finetuned/`); without it, retrieval skips the reranking step. `LLM_BACKEND=groq` with a `GROQ_API_KEY` still works if you prefer a hosted model.

## Project structure

```
DocMind - Rag Model/
├── api.py              # FastAPI backend: /upload, /chat, /health, demo session
├── app.py              # Streamlit frontend
├── rag_pipeline.py     # The pipeline: index_documents(), retrieve(), answer()
├── data_ingestion.py   # Extraction + sentence-aware chunking
├── vector_store.py     # bge-small embeddings (ChromaDB) + BM25, RRF fusion
├── reranker.py         # ONNX cross-encoder reranker
├── generation.py       # Prompting, Ollama/Groq call, citation checks
├── eval/               # Gold-set evaluation harness
├── training/           # Colab notebooks for the reranker and answer model, Ollama Modelfile
├── models/             # Exported models (gitignored)
└── processed_chunks.json  # Baked demo corpus (DocMind's SLA + privacy policy)
```

## Deployment

`Dockerfile.backend` / `Dockerfile.frontend` deploy to Render (see `deployment_guide.md`). The free tier's 512 MB of RAM cannot host a 7B LLM or comfortably add the reranker, so a free-tier deployment uses `LLM_BACKEND=groq` and runs without `models/`. The full self-hosted stack needs a machine that can run Ollama (≈6 GB of GPU or system memory for a 7B Q4 model).

## Why not just use LangChain?

DocMind is intentionally built from the raw libraries so every step — extraction, chunking, retrieval fusion, reranking, prompting — can be measured and explained, rather than hidden behind a framework abstraction.
