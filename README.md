# DocMind — RAG-based Document Q&A System

A privacy-preserving **Retrieval-Augmented Generation (RAG)** system that lets you ask natural-language questions over your own documents (PDFs and text files). DocMind retrieves the most relevant passages at query time and feeds them to a language model — giving you accurate, source-grounded answers without any fine-tuning.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Text Extraction | `pdfplumber` | Parse raw text from PDF and TXT files |
| Tokenization | `tiktoken` | Split text into chunks by token count (not characters) |
| Embedding | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | Convert text chunks and queries into semantic vectors (in-process, no external API) |
| Vector Store | `ChromaDB` | Store, index, and retrieve embeddings with cosine similarity |
| Keyword Retrieval | `rank-bm25` (BM25Okapi) | Hybrid retrieval: RRF fusion of embedding + keyword signals |
| Generation | Groq Cloud API (`openai/gpt-oss-20b`) | Answer generation from retrieved context (deployed); Ollama locally |
| API | `FastAPI` + `uvicorn` | Backend HTTP API with session management |
| Frontend | `Streamlit` | Document upload and conversational UI |
| Language | Python 3.11 | Core runtime |

---

## Project Structure

```
DocMind - Rag Model/
├── api.py                  # FastAPI backend: /upload, /chat, /health, demo session seeding
├── app.py                  # Streamlit frontend: upload flow + demo shortcut
├── data_ingestion.py       # Step 1: Load docs → extract text → chunk → save JSON
├── vector_store.py         # Step 2: Embed chunks → store in ChromaDB → expose retrieve()
├── generation.py           # Step 3: Two-tier abstention gate + Groq LLM generation
├── Dockerfile.backend      # Backend image (bge-small baked in, corpus baked in)
├── Dockerfile.frontend     # Frontend image
├── processed_chunks.json   # Baked SLA + Privacy Policy corpus (5 chunks)
└── docs/                   # Drop your source PDFs and TXT files here (gitignored)
```

---

## Getting Started

### Option A — Try the live demo (no setup needed)

Deploy to Render (see `deployment_guide.md`) and click **"🚀 Try the demo"** in the sidebar. DocMind's own SLA and Privacy Policy are baked into the Docker image and instantly available without uploading anything.

Example questions to try:
- *"What uptime does DocMind guarantee?"*
- *"Who are DocMind's data sub-processors?"*
- *"What service credit do I get if uptime drops below 95%?"*

### Option B — Local development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start the backend
uvicorn api:app --host 0.0.0.0 --port 8000

# 4. Start the frontend (separate terminal)
streamlit run app.py
```

Upload any PDF or TXT file via the sidebar to start a private session. Each upload creates an isolated session — your data is never mixed with other users' documents.

### Option C — Run the pipeline directly (scripting / evaluation)

```bash
# Chunk your documents
python data_ingestion.py

# Build the vector store and test retrieval
python vector_store.py

# Evaluate retrieval quality on the test suite
python evaluate.py
```

---

## ⚠️ Work in Progress

This project is being built incrementally:

- [x] **Step 1 — Data Ingestion:** Load PDFs/TXT files, chunk by token count with overlap, export to JSON
- [x] **Step 2 — Vector Retrieval:** Embed chunks with `BAAI/bge-small-en-v1.5` (sentence-transformers), store in ChromaDB, hybrid RRF retrieval (embedding + BM25)
- [x] **Step 3 — Generation:** Groq Cloud API (`openai/gpt-oss-20b`) for production; Ollama locally. Includes calibrated two-tier abstention gate (cosine-similarity-based, decoupled from RRF ranking) and post-generation citation verification.
- [x] **Step 4 — UI:** Streamlit frontend with document upload, demo shortcut, session management, and confidence display. FastAPI backend with per-session vector stores and seeded demo session.

---

## Why not just use LangChain?

DocMind is intentionally built from scratch using the raw libraries. The goal is to understand exactly what happens at every step — chunking strategy, embedding model choice, similarity metrics — rather than hiding it behind a framework abstraction. This makes it easier to debug, tune, and explain in interviews.

---

## License

MIT
