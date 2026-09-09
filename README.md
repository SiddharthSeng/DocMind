# DocMind — RAG-based Document Q&A System

A privacy-preserving **Retrieval-Augmented Generation (RAG)** system that lets you ask natural-language questions over your own documents (PDFs, TXT files, and DOCX files). DocMind retrieves the most relevant passages at query time and feeds them to a language model — giving you accurate, source-grounded answers without any fine-tuning.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Text Extraction | `pdfplumber`, `python-docx` | Parse raw text from PDF, TXT, and DOCX files |
| Tokenization | `tiktoken` | Split text into chunks by token count (not characters) |
| Embedding | `fastembed` (`BAAI/bge-small-en-v1.5`, ONNX runtime) | Convert text chunks and queries into semantic vectors in-process. ~40–60 MB RAM — replaces sentence-transformers/PyTorch which caused OOM kills on Render's 512 MB free tier |
| Vector Store | `ChromaDB` | Store, index, and retrieve embeddings with cosine similarity |
| Keyword Retrieval | `rank-bm25` (BM25Okapi) | Hybrid retrieval: RRF fusion of embedding + keyword signals for chunk *ranking* |
| Generation | Groq Cloud API (`openai/gpt-oss-20b`) | Answer generation from retrieved context. Free tier: 30 req/min, 14,400 req/day |
| Backend API | `FastAPI` + `uvicorn` | Async HTTP API with per-session vector stores; blocking operations offloaded via `asyncio.to_thread` |
| Frontend | `Streamlit` | Document upload and conversational UI |
| Language | Python 3.11 | Core runtime |

> **Local dev alternative:** You can run the LLM locally via Ollama (`LLM_BACKEND=ollama`) instead of Groq. This is supported but not what runs in production. See Option B below.

---

## Project Structure

```
DocMind - Rag Model/
├── api.py                  # FastAPI backend: /upload, /chat, /health, demo session seeding
├── app.py                  # Streamlit frontend: upload flow + demo shortcut
├── data_ingestion.py       # Step 1: Load docs → extract text → chunk → save JSON
├── vector_store.py         # Step 2: Embed chunks (fastembed/ONNX) → store in ChromaDB → expose retrieve()
├── generation.py           # Step 3: Two-tier abstention gate + Groq LLM generation
├── Dockerfile.backend      # Backend image (bge-small ONNX baked in, baked corpus included)
├── Dockerfile.frontend     # Frontend image
├── deployment_guide.md     # Full Render deployment walkthrough
├── processed_chunks.json   # Baked SLA + Privacy Policy corpus (5 chunks)
└── docs/                   # Drop your source PDFs, TXTs, or DOCX files here (gitignored)
```

---

## How It Works

### Retrieval

DocMind uses **hybrid retrieval** combining two signals:

1. **Dense embedding similarity** — `BAAI/bge-small-en-v1.5` via fastembed (ONNX) encodes both the query and each stored chunk into a semantic vector space. Cosine similarity measures how semantically close the query is to each chunk.
2. **BM25 keyword matching** — sparse keyword overlap score using BM25Okapi.

Both signals are fused into a single ranking using **Reciprocal Rank Fusion (RRF)**, which combines the rank positions (not the raw scores) from each retriever. This is more robust than either signal alone.

### Two-Tier Abstention Gate

Before calling the LLM, DocMind runs a two-tier gate to avoid generating hallucinated answers on off-topic queries:

**Tier 1 — Hard floor (`ABSTAIN_THRESHOLD = 0.55`, cosine similarity):**
If the top retrieved chunk's raw cosine similarity (`embed_score`) is below this threshold, the query is completely off-topic. The LLM is not called at all — DocMind abstains immediately.

**Tier 2 — Context quality filter (`MIN_SIMILARITY_THRESHOLD = 0.60`, cosine similarity):**
After passing the hard floor, any chunk with `embed_score < 0.60` is dropped from the LLM context (too weak to ground a reliable answer). If no chunk passes this filter → a "weak match" abstention is returned without calling the LLM.

> **Important design note:** The gate uses raw **cosine similarity** (`embed_score`), not fused RRF scores. RRF scores are rank-based and compress into a very narrow band (~0.13–0.18) on small corpora, regardless of actual semantic relevance. Cosine similarity retains a meaningful spread that discriminates in-domain from OOD queries.
>
> **The gate is a coarse pre-filter for obvious noise.** The LLM's own system-prompt-driven abstention is the primary defense against domain-adjacent OOD queries that share vocabulary with the corpus (e.g., a query about DocMind that isn't answered in the SLA). This is an intentional two-layer design: retrieval-level gating catches garbage-domain queries; LLM-level abstention catches residual vocabulary-adjacent queries. Both layers are empirically verified working.

### Post-Generation Hallucination Checks

After every LLM response:
- **Citation verification:** Every source file the model cites is cross-checked against the retrieved chunks. Any hallucinated citation (a filename not actually in context) is stripped and flagged.
- **Confidence scoring:** Derived from raw cosine similarity of the retrieved chunks — not from the LLM's self-reported certainty, which is unreliable.

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
# Edit .env — add GROQ_API_KEY and set LLM_BACKEND=groq
# (or set LLM_BACKEND=ollama and run `ollama serve` locally)

# 3. Start the backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# 4. Start the frontend (separate terminal)
python -m streamlit run app.py
```

Upload any **PDF, TXT, or DOCX** file via the sidebar to start a private session. Each upload creates an isolated session — your data is never mixed with other users' documents.

> **Note:** The backend loads environment variables from `.env` via `python-dotenv`. Ensure `.env` contains `LLM_BACKEND=groq` (and a valid `GROQ_API_KEY`) or `LLM_BACKEND=ollama` (with Ollama running locally). If `LLM_BACKEND` is not set, it defaults to `ollama`.

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

## Supported File Formats

| Format | Extension | Library |
|---|---|---|
| PDF | `.pdf` | `pdfplumber` |
| Plain Text | `.txt` | built-in |
| Word Document | `.docx` | `python-docx` |

The pipeline is **format-agnostic after extraction** — once text is extracted from any format, chunking, embedding, retrieval, and generation are identical regardless of source format.

---

## Deployment

DocMind runs as **two separate Docker-based Web Services on Render**:
- **Backend** (`Dockerfile.backend`): FastAPI + ChromaDB + fastembed (ONNX). ~150–180 MB RAM steady-state.
- **Frontend** (`Dockerfile.frontend`): Streamlit UI. Minimal RAM.

The frontend connects to the backend via the `API_BASE_URL` environment variable. See `deployment_guide.md` for the full setup walkthrough.

---

## ⚙️ Build Status

- [x] **Step 1 — Data Ingestion:** Load PDF/TXT/DOCX files, chunk by token count with overlap, export to JSON
- [x] **Step 2 — Vector Retrieval:** Embed chunks with `BAAI/bge-small-en-v1.5` (fastembed/ONNX), store in ChromaDB, hybrid RRF retrieval (embedding + BM25)
- [x] **Step 3 — Generation:** Groq Cloud API (`openai/gpt-oss-20b`) for production; Ollama locally. Calibrated two-tier abstention gate (cosine-similarity-based, decoupled from RRF ranking) + post-generation citation verification
- [x] **Step 4 — UI:** Streamlit frontend with PDF/TXT/DOCX upload, demo shortcut, session management, and confidence display. FastAPI backend with per-session vector stores, seeded demo session, and async offloading of all blocking operations

---

## Why not just use LangChain?

DocMind is intentionally built from scratch using the raw libraries. The goal is to understand exactly what happens at every step — chunking strategy, embedding model choice, hybrid retrieval fusion, abstention thresholds — rather than hiding it behind a framework abstraction. This makes it easier to debug, tune, and explain in interviews.

