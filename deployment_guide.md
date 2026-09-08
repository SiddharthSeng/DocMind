# DocMind — Render Deployment Guide

> **Audience:** Developer deploying DocMind to Render's free tier for the first time.
> **Last verified:** September 2026 (Render free-tier UI).

---

## Overview

DocMind uses two separate services on Render:

| Service | Source | Render type | Port |
|---|---|---|---|
| **Backend** (FastAPI + ChromaDB + bge-small) | `Dockerfile.backend` | Web Service (Docker) | 8000 |
| **Frontend** (Streamlit) | `Dockerfile.frontend` | Web Service (Docker) | 8501 |

Both can be deployed on the **free tier** (512 MB RAM, 0.1 CPU each). Deploy the **backend first** so you have its URL before configuring the frontend.

### What users see after deployment

New users land on the Streamlit UI and see two options:

- **🚀 Try the demo** — immediately starts a chat against DocMind's own SLA and Privacy Policy. No upload required. This works because the baked corpus is seeded into `sessions/demo/` at container startup, before any user arrives.
- **Upload your own document** — creates an isolated session scoped to that document. Sessions are ephemeral (wiped on Render's free-tier restart/spin-down). Data is never shared between sessions.

---

## Prerequisites

- GitHub repo with `Dockerfile.backend` and `Dockerfile.frontend` at the root.
- A free Groq API key from [console.groq.com](https://console.groq.com) (no credit card needed).
- A Render account at [render.com](https://render.com).

---

## Step 1 — Deploy the Backend

### 1.1 Create a new Web Service

1. In the Render Dashboard, click **New +** → **Web Service**.
2. Connect your GitHub repository and select the correct branch (`main`).
3. Under **Language / Runtime**, select **Docker**.
4. Set the **Dockerfile path** to `Dockerfile.backend`.

### 1.2 Build & Start commands

Render reads these from the Dockerfile automatically when using Docker mode. You do **not** need to set a Build Command or Start Command manually — the `CMD` in `Dockerfile.backend` is used:

```
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

If Render shows blank command fields, you can leave them empty for Docker deployments. Do **not** paste the uvicorn command into the Start Command field — it is already in the Dockerfile.

### 1.3 Instance Type

Select **Free** (512 MB RAM, 0.1 CPU). Click **Create Web Service**.

### 1.4 Health Check Path

Render performs HTTP health checks on your service. DocMind's FastAPI backend exposes a dedicated `/health` endpoint that returns `{"status": "ok"}` immediately — without loading the embedding model. This is important: on Render's free tier the health check fires within seconds of deploy, before PyTorch's ~300 MB model load completes. The lazy-loading design ensures the process binds its port and passes health checks first.

In Render's **Settings → Health Check Path**, set:
```
/health
```

> **Important — Cold-start sequence:** After deploying, the startup sequence is:
> 1. uvicorn binds the port and passes Render's health check (`/health`) — **no PyTorch yet**.
> 2. `startup_event()` seeds `sessions/demo/` from the baked `./chroma_db` (a fast `shutil.copytree`, no embedding work). This only happens once — subsequent restarts skip the copy if the directory exists.
> 3. The embedding model (bge-small, ~300 MB PyTorch) is lazy-loaded on the **first actual `/upload` or `/chat` request** (~5–10s). The demo session uses the pre-baked ChromaDB vectors, so it triggers a model load on the first `/chat demo` call.

### 1.5 Environment Variables

In the Render dashboard for the backend service, go to **Environment** and add:

| Variable | Value | Required? |
|---|---|---|
| `GROQ_API_KEY` | Your key from console.groq.com | **Required** |
| `EMBEDDING_BACKEND` | `hf_bge_small` | Set by Dockerfile (override only if changing backend) |
| `LLM_BACKEND` | `groq` | Set by Dockerfile |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Set by Dockerfile (alternatives: `openai/gpt-oss-120b`) |
| `MIN_SIMILARITY_THRESHOLD` | `0.60` | Set by Dockerfile (cosine similarity, not fused RRF) |
| `ABSTAIN_THRESHOLD` | `0.55` | Set by Dockerfile (cosine similarity, not fused RRF) |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | Set by Dockerfile |
| `OMP_NUM_THREADS` | `1` | Set by Dockerfile |
| `TOKENIZERS_PARALLELISM` | `false` | Set by Dockerfile |

> **Security:** `GROQ_API_KEY` must be set in the Render dashboard. It must **never** be committed to the repository or baked into the Dockerfile. All other variables have safe defaults in the Dockerfile.

> **Note on thresholds:** The abstention gate uses **cosine similarity** (embed_score) for gating decisions, not fused RRF scores. This was changed in Sept 2026 because RRF scores compress into an indistinguishable band on small corpora. Retrieval *ranking* still uses fused RRF.

### 1.6 Note the backend URL

Once deployed, Render assigns a URL like `https://docmind-backend-xxxx.onrender.com`. Copy this — you'll need it for the frontend.

---

## Step 2 — Deploy the Frontend

### 2.1 Create a second Web Service

1. Click **New +** → **Web Service** again.
2. Connect the **same** GitHub repository.
3. Under **Language / Runtime**, select **Docker**.
4. Set the **Dockerfile path** to `Dockerfile.frontend`.

### 2.2 Instance Type

Select **Free**.

### 2.3 Environment Variables

| Variable | Value | Required? |
|---|---|---|
| `API_BASE_URL` | `https://your-backend-url.onrender.com` | **Required** — use the URL from Step 1.6 |

> This is the single most important variable. Without it, the frontend points to `localhost:8000` which doesn't exist in the container and every request will fail with "Backend not reachable."

### 2.4 Health Check Path

Streamlit's default route is `/`. Set:
```
/
```

---

## Step 3 — Verify the Deployment

1. Open the backend URL directly: `https://your-backend.onrender.com/` — should return `{"title":"DocMind API"}` or similar JSON.
2. Open the frontend URL — should show the DocMind UI with a file upload sidebar.
3. Upload a `.pdf` or `.txt` file and ask a question — the first request will be slow (~10–15s) because the embedding model loads on first use. Subsequent requests in the same session are fast.

---

## Known Limitations of the Free Tier

### Ephemeral Filesystem

> **Important:** Render's free tier uses an **ephemeral filesystem**. Any files written at runtime (uploaded documents, session ChromaDB directories under `sessions/`) are **lost when the service restarts or spins down**.

This means:
- Uploaded user documents are **not** persisted across cold starts.
- The **pre-baked corpus** (`chroma_db/` in the image) is read-only and always present because it is part of the Docker image layer. It persists correctly.
- User chat sessions are ephemeral by design (the session limit of 5 is enforced in code).

### Cold-Start Latency (Spin-Down)

Free tier services **spin down after 15 minutes of inactivity**. The next request triggers a cold start:

- Uvicorn starts: ~2–4 seconds
- First request (model loads): ~8–15 seconds additional
- **Total first-request latency after idle:** ~10–20 seconds

This is expected behaviour. The Streamlit frontend's "Thinking..." spinner will show during this time. Inform users if deploying publicly.

To avoid spin-down, upgrade to Render's **Starter plan** (~$7/month) which keeps the service always-on.

### Rate Limits (Groq Free Tier)

Groq's free tier for `llama-3.1-8b-instant`:
- 30 requests/minute
- 14,400 requests/day

For a personal or portfolio demo these limits are not a concern. If you exceed them, Groq returns HTTP 429 and the backend will surface an error to the frontend.

---

## Environment Variable Checklist (Pre-Launch)

Run through this before going live:

- [ ] `GROQ_API_KEY` is set in the **backend** service environment (not in the frontend, not in the repo).
- [ ] `API_BASE_URL` is set in the **frontend** service environment and points to the actual backend URL (not localhost).
- [ ] Backend URL returns a 200 at `/` before the frontend is tested.
- [ ] Frontend URL loads without a "Backend not reachable" error.
- [ ] At least one test upload + question completes successfully.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Backend not reachable" in UI | `API_BASE_URL` not set or wrong | Check frontend env var in Render dashboard |
| Backend 502 Bad Gateway on deploy | Startup crashed before port bound | Check Render logs; look for import errors |
| "GROQ_API_KEY not set" error | Forgot to add key in dashboard | Add `GROQ_API_KEY` to backend env vars |
| First request times out | Normal cold-start model load | Wait 15–20s and retry |
| Uploaded file not found after redeploy | Ephemeral filesystem reset | Re-upload the file — this is expected |
| 429 Too Many Requests from Groq | Rate limit hit | Wait 1 minute; consider upgrading Groq plan |
