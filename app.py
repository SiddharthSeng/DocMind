"""
app.py — Streamlit Frontend for DocMind
========================================
Provides a clean, intuitive UI for uploading documents and chatting with the RAG pipeline.
Communicates with the FastAPI backend (api.py).
"""

import os
import streamlit as st
import requests

# Backend URL — set API_BASE_URL in the Render environment variables dashboard.
# Falls back to localhost:8000 for local development.
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Render free-tier services spin down after ~15 min of inactivity and can take
# 50-90 seconds to cold-start.  Use a generous timeout so the user sees a
# friendly message rather than a raw exception.
BACKEND_TIMEOUT = 90  # seconds


def _safe_error_detail(response: "requests.Response", fallback: str = "Unknown error") -> str:
    """
    Try to extract a human-readable detail string from a non-200 response.
    Falls back gracefully if the body is not valid JSON or is empty.
    """
    try:
        return response.json().get("detail", fallback)
    except Exception:
        snippet = response.text[:200].strip() if response.text else ""
        if snippet:
            return f"{fallback} — server said: {snippet}"
        return fallback


def _ping_backend() -> None:
    """
    Fire-and-forget GET to /health so the Render instance warms up before the
    user's first real request.  Any error is silently swallowed.
    """
    try:
        requests.get(f"{API_URL}/health", timeout=5)
    except Exception:
        pass

st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preview_text" not in st.session_state:
    st.session_state.preview_text = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "backend_pinged" not in st.session_state:
    st.session_state.backend_pinged = False

# Pre-warm the Render backend on first page load (non-blocking best-effort ping).
# This gives the instance time to wake up before the user's first real request.
if not st.session_state.backend_pinged:
    import threading
    threading.Thread(target=_ping_backend, daemon=True).start()
    st.session_state.backend_pinged = True

st.title("🧠 DocMind")
st.markdown("Your private, local Document AI. Upload a file — or try the built-in demo.")

# --- Sidebar: File Upload & Controls ---
with st.sidebar:
    st.header("Document Setup")

    # ── Demo shortcut ──────────────────────────────────────────────────────
    st.markdown("**Quick start**")
    if st.button("🚀 Try the demo", help="Chat with DocMind's own SLA & Privacy Policy — no upload required."):
        st.session_state.session_id = "demo"
        st.session_state.demo_mode  = True
        st.session_state.filename   = "DocMind SLA & Privacy Policy (demo)"
        st.session_state.preview_text = None
        st.session_state.messages   = []
        st.success("Demo loaded! Ask about DocMind's SLA or Privacy Policy below.")

    st.divider()
    st.markdown("**Or upload your own document**")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt", "docx"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
        # A new file was uploaded, trigger ingestion
        with st.spinner("Processing & Embedding Document..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=BACKEND_TIMEOUT,
                )

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id   = data["session_id"]
                    st.session_state.filename      = data["filename"]
                    st.session_state.preview_text  = data["preview"]
                    st.session_state.demo_mode     = False
                    st.session_state.messages      = []  # Clear chat for new document
                    st.success(f"Successfully processed {data['chunks_processed']} chunks!")
                else:
                    detail = _safe_error_detail(response, "Upload failed")
                    st.error(
                        f"Upload failed (HTTP {response.status_code}): {detail}"
                    )
                    st.session_state.session_id = None

            except requests.exceptions.Timeout:
                st.error(
                    "⏳ The server is waking up from idle — this can take up to a minute "
                    "on the free tier. Please wait a moment and try again."
                )
            except requests.exceptions.ConnectionError:
                st.error(
                    "🔌 Could not reach the backend. "
                    "Check that the API service is deployed and the API_BASE_URL is correct."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# --- Main Area: Preview & Chat ---
if st.session_state.session_id:
    # Demo mode banner
    if st.session_state.demo_mode:
        st.info(
            "🚀 **Demo mode** — You're chatting with DocMind's own SLA and Privacy Policy. "
            "Try asking: *\"What uptime does DocMind guarantee?\"* or "
            "*\"Who are DocMind's data sub-processors?\"*  "
            "Upload your own document in the sidebar to start a private session."
        )
    else:
        # Optional Preview Pane (only shown for real uploads)
        with st.expander(f"📄 Previewing: {st.session_state.filename}"):
            st.text_area("Extracted Text (first few chunks)", st.session_state.preview_text, height=200, disabled=True)

    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # Display sources and confidence metrics if present
                if msg.get("sources"):
                    st.caption(f"**Sources:** {', '.join(msg['sources'])}")
                st.caption(f"**Confidence:** {msg.get('confidence', 0):.2f}")
                if msg.get("warning"):
                    st.warning(f"Note: {msg['warning']}")

    # Chat Input — placeholder text adapts to demo vs. real session
    chat_placeholder = (
        "Ask about DocMind's SLA or Privacy Policy..."
        if st.session_state.demo_mode
        else "Ask a question about your document..."
    )
    if prompt := st.chat_input(chat_placeholder):
        # 1. Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call backend for answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "query": prompt
                    }
                    response = requests.post(
                        f"{API_URL}/chat",
                        json=payload,
                        timeout=BACKEND_TIMEOUT,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer     = data["answer"]
                        sources    = data["sources_cited"]
                        confidence = data["confidence"]
                        warning    = data["warning"]

                        st.markdown(answer)
                        if sources:
                            st.caption(f"**Sources:** {', '.join(sources)}")
                        st.caption(f"**Confidence:** {confidence:.2f}")
                        if warning:
                            st.warning(f"Note: {warning}")

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "confidence": confidence,
                            "warning": warning
                        })

                    elif response.status_code == 404:
                        st.error("Session expired or invalid. Please re-upload your document.")
                        st.session_state.session_id = None
                    else:
                        detail = _safe_error_detail(response, "Generation failed")
                        st.error(
                            f"Generation failed (HTTP {response.status_code}): {detail}"
                        )

                except requests.exceptions.Timeout:
                    st.error(
                        "⏳ The server is waking up from idle — this can take up to a minute "
                        "on the free tier. Please wait a moment and try again."
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "🔌 Could not reach the backend. "
                        "Check that the API service is deployed and the API_BASE_URL is correct."
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

else:
    st.info("👈 **Upload a document** in the sidebar to begin — or click **Try the demo** to explore DocMind's own SLA and Privacy Policy.")

