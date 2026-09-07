# DocMind — RAG-based Document Q&A System

A local, privacy-preserving **Retrieval-Augmented Generation (RAG)** system that lets you ask natural-language questions over a set of your own documents (PDFs and text files). Instead of fine-tuning a model on your data, DocMind retrieves the most relevant passages from your documents at query time and feeds them to a language model — giving you accurate, source-grounded answers.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Text Extraction | `pdfplumber` | Parse raw text from PDF and TXT files |
| Tokenization | `tiktoken` | Split text into chunks by token count (not characters) |
| Embedding | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | Convert text chunks and queries into semantic vectors (in-process, no external API) |
| Vector Store | `ChromaDB` | Store, index, and retrieve embeddings with cosine similarity |
| Keyword Retrieval | `rank-bm25` (BM25Okapi) | Hybrid retrieval: RRF fusion of embedding + keyword signals |
| Generation | Groq Cloud API (`llama-3.1-8b-instant`) | Answer generation from retrieved context (deployed); Ollama locally |
| API | `FastAPI` + `uvicorn` | Backend HTTP API with session management |
| Frontend | `Streamlit` | Document upload and conversational UI |
| Language | Python 3.11 | Core runtime |

---

## Project Structure

```
DocMind - Rag Model/
├── data_ingestion.py       # Step 1: Load docs → extract text → chunk → save JSON
├── vector_store.py         # Step 2: Embed chunks → store in ChromaDB → expose retrieve()
├── requirements.txt        # All Python dependencies
└── docs/                   # Drop your source PDFs and TXT files here (gitignored)
```

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your documents
Drop your `.pdf` or `.txt` files into the `docs/` folder.

### 3. Run data ingestion
```bash
python data_ingestion.py
```
This creates `processed_chunks.json` with all tokenized text chunks.

### 4. Build the vector store & test retrieval
```bash
python vector_store.py
```
This embeds all chunks into a local ChromaDB database and runs a sample query.

---

## ⚠️ Work in Progress

This project is being built incrementally:

- [x] **Step 1 — Data Ingestion:** Load PDFs/TXT files, chunk by token count with overlap, export to JSON
- [x] **Step 2 — Vector Retrieval:** Embed chunks with `BAAI/bge-small-en-v1.5` (sentence-transformers), store in ChromaDB, hybrid RRF retrieval (embedding + BM25)
- [x] **Step 3 — Generation:** Groq Cloud API (`llama-3.1-8b-instant`) for production; Ollama locally. Includes calibrated two-tier abstention gate and post-generation citation verification.
- [x] **Step 4 — UI:** Streamlit frontend with document upload, session management, and confidence display. FastAPI backend with per-session vector stores.

---

## Why not just use LangChain?

DocMind is intentionally built from scratch using the raw libraries. The goal is to understand exactly what happens at every step — chunking strategy, embedding model choice, similarity metrics — rather than hiding it behind a framework abstraction. This makes it easier to debug, tune, and explain in interviews.

---

## License

MIT
