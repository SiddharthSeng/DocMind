"""
app.py — Streamlit Frontend for DocMind
========================================
Provides a clean, distinctive UI for uploading documents and chatting
with the RAG pipeline.  Communicates with the FastAPI backend (api.py).

Visual identity: deep indigo (#818cf8) primary + amber (#fbbf24) citation accent.
Theme defined in .streamlit/config.toml.  Custom CSS injected once at startup
for page-level elements (header, sidebar, banner).  Metadata badges use
fully inline styles to guarantee rendering inside st.chat_message containers.
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

# ── Avatars ───────────────────────────────────────────────────────────────────
USER_AVATAR      = "💬"
ASSISTANT_AVATAR = "🔍"


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


def _render_answer_metadata(sources: list, confidence: float, warning: str | None) -> None:
    """
    Renders compact inline metadata badges below an assistant answer:
      - A small confidence pill (color-coded green/amber/red)
      - Source file tag(s) in amber accent

    All styles are INLINE — no reliance on external <style> blocks — so they
    render correctly inside st.chat_message() containers regardless of
    Streamlit's internal isolation / shadow DOM boundaries.
    """
    conf_pct = int(confidence * 100)
    if conf_pct >= 70:
        conf_bg     = "rgba(34,197,94,0.15)"
        conf_color  = "#22c55e"
        conf_border = "rgba(34,197,94,0.3)"
    elif conf_pct >= 50:
        conf_bg     = "rgba(245,158,11,0.15)"
        conf_color  = "#f59e0b"
        conf_border = "rgba(245,158,11,0.3)"
    else:
        conf_bg     = "rgba(248,113,113,0.15)"
        conf_color  = "#f87171"
        conf_border = "rgba(248,113,113,0.3)"

    # Confidence pill
    parts = [
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'background:{conf_bg};color:{conf_color};'
        f'border:1px solid {conf_border};border-radius:12px;'
        f'padding:2px 10px;font-size:0.73rem;font-weight:600;'
        f'white-space:nowrap;">'
        f'<span style="font-size:0.5rem;">●</span> {conf_pct}%</span>'
    ]

    # Source badges
    for s in (sources or []):
        parts.append(
            f'<span style="display:inline-block;'
            f'background:rgba(251,191,36,0.10);color:#fbbf24;'
            f'border:1px solid rgba(251,191,36,0.25);border-radius:4px;'
            f'padding:1px 8px;font-size:0.71rem;'
            f'font-family:ui-monospace,SFMono-Regular,monospace;'
            f'white-space:nowrap;">{s}</span>'
        )

    badge_row = (
        f'<div style="margin-top:0.5rem;display:flex;align-items:center;'
        f'flex-wrap:wrap;gap:0.35rem;">'
        + "".join(parts)
        + '</div>'
    )

    warning_html = ""
    if warning:
        warning_html = (
            f'<div style="margin-top:0.3rem;color:#64748b;font-style:italic;'
            f'font-size:0.71rem;line-height:1.4;">{warning}</div>'
        )

    st.markdown(badge_row + warning_html, unsafe_allow_html=True)


# ── Page config must come before any other Streamlit call ─────────────────────
st.set_page_config(page_title="DocMind", page_icon="🔍", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
# Only for page-level elements (header, sidebar labels, banners) — NOT for
# metadata badges, which use inline styles to survive chat_message isolation.
st.markdown("""
<style>
/* ── Brand header ──────────────────────────────────────── */
.dm-header {
    padding: 0.5rem 0 1.25rem 0;
    border-bottom: 1px solid rgba(129, 140, 248, 0.2);
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(129,140,248,0.06) 0%, transparent 60%);
    border-radius: 0 0 12px 12px;
    padding-left: 0.5rem;
}
.dm-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #e2e8f0;
    margin: 0 0 0.2rem 0;
    line-height: 1.15;
}
.dm-header h1 span {
    color: #818cf8;
}
.dm-header p {
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0;
    font-weight: 400;
}

/* ── Document context bar ──────────────────────────────── */
.dm-ctx-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0.9rem;
    background: rgba(129, 140, 248, 0.06);
    border: 1px solid rgba(129, 140, 248, 0.15);
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.82rem;
}
.dm-ctx-bar .dm-ctx-file {
    color: #c7d2fe;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 320px;
}
.dm-ctx-bar .dm-ctx-sep {
    color: rgba(129, 140, 248, 0.35);
}
.dm-ctx-bar .dm-ctx-detail {
    color: #64748b;
    font-size: 0.78rem;
}
.dm-ctx-bar .dm-ctx-dot {
    color: #22c55e;
    font-size: 0.55rem;
}

/* ── Demo mode banner ──────────────────────────────────── */
.dm-demo-banner {
    background: rgba(129, 140, 248, 0.08);
    border: 1px solid rgba(129, 140, 248, 0.25);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    color: #a5b4fc;
    font-size: 0.88rem;
    line-height: 1.5;
}
.dm-demo-banner strong {
    color: #818cf8;
}

/* ── Welcome card ──────────────────────────────────────── */
.dm-welcome {
    text-align: center;
    padding: 3.5rem 1.5rem 3rem;
    max-width: 480px;
    margin: 2rem auto 0;
    background: linear-gradient(160deg, rgba(129,140,248,0.06) 0%, rgba(129,140,248,0.02) 50%, transparent 100%);
    border: 1px solid rgba(129, 140, 248, 0.12);
    border-radius: 16px;
}
.dm-welcome-logo {
    font-size: 3.2rem;
    font-weight: 800;
    color: #818cf8;
    font-family: ui-monospace, SFMono-Regular, monospace;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 30px rgba(129,140,248,0.25);
}
.dm-welcome h2 {
    color: #e2e8f0;
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0 0 0.6rem 0;
}
.dm-welcome p {
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.6;
    margin: 0 0 1.8rem 0;
}
.dm-welcome-steps {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.dm-welcome-step {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    line-height: 1.5;
}
.dm-welcome-step .step-icon {
    font-size: 1.3rem;
    margin-bottom: 0.25rem;
}
.dm-welcome-step strong {
    color: #94a3b8;
}

/* ── Sidebar ───────────────────────────────────────────── */
.dm-sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.3rem 0 1rem 0;
    border-bottom: 1px solid rgba(129,140,248,0.18);
    margin-bottom: 1rem;
}
.dm-sidebar-logo .logo-mark {
    font-size: 1.7rem;
    font-weight: 800;
    color: #818cf8;
    font-family: ui-monospace, SFMono-Regular, monospace;
    line-height: 1;
}
.dm-sidebar-logo .logo-text {
    line-height: 1.2;
}
.dm-sidebar-logo .logo-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}
.dm-sidebar-logo .logo-sub {
    font-size: 0.62rem;
    color: #64748b;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
}
.dm-sidebar-label {
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    margin: 0 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.dm-sidebar-status {
    margin-top: 0.75rem;
    padding: 0.5rem 0.7rem;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
}
.dm-sidebar-status.active {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.18);
    color: #94a3b8;
}
.dm-sidebar-status.active .status-dot {
    color: #22c55e;
    font-size: 0.5rem;
}
.dm-sidebar-status.inactive {
    background: rgba(100,116,139,0.08);
    border: 1px solid rgba(100,116,139,0.15);
    color: #64748b;
}
.dm-sidebar-status.inactive .status-dot {
    color: #64748b;
    font-size: 0.5rem;
}
.dm-sidebar-footer {
    margin-top: 1.5rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(100,116,139,0.15);
    text-align: center;
    color: #475569;
    font-size: 0.68rem;
    letter-spacing: 0.03em;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preview_text" not in st.session_state:
    st.session_state.preview_text = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "backend_pinged" not in st.session_state:
    st.session_state.backend_pinged = False

# Pre-warm the Render backend on first page load (non-blocking best-effort ping).
if not st.session_state.backend_pinged:
    import threading
    threading.Thread(target=_ping_backend, daemon=True).start()
    st.session_state.backend_pinged = True

# ── Brand header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-header">
    <h1><span>Doc</span>Mind</h1>
    <p>Precise document Q&amp;A — private, grounded, and source-cited.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="dm-sidebar-logo">
        <span class="logo-mark">D⟩</span>
        <div class="logo-text">
            <div class="logo-name">DocMind</div>
            <div class="logo-sub">Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Demo shortcut ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="dm-sidebar-label">🚀 Quick start</p>', unsafe_allow_html=True)
        if st.button(
            "Try the demo",
            help="Chat with DocMind's own SLA & Privacy Policy — no upload required.",
            use_container_width=True,
        ):
            st.session_state.session_id   = "demo"
            st.session_state.demo_mode    = True
            st.session_state.filename     = "DocMind SLA & Privacy Policy"
            st.session_state.chunks_count = 5
            st.session_state.preview_text = None
            st.session_state.messages     = []
            st.success("Demo loaded!")

    # ── File upload ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown('<p class="dm-sidebar-label">📄 Your document</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "PDF, TXT, or DOCX",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
            # A new file was uploaded — trigger ingestion
            with st.spinner("Processing & embedding…"):
                try:
                    files    = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        timeout=BACKEND_TIMEOUT,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.session_id   = data["session_id"]
                        st.session_state.filename     = data["filename"]
                        st.session_state.preview_text = data["preview"]
                        st.session_state.chunks_count = data["chunks_processed"]
                        st.session_state.demo_mode    = False
                        st.session_state.messages     = []
                        st.success(f"✓ {data['chunks_processed']} chunks processed.")
                    else:
                        detail = _safe_error_detail(response, "Upload failed")
                        st.error(f"Upload failed (HTTP {response.status_code}): {detail}")
                        st.session_state.session_id = None

                except requests.exceptions.Timeout:
                    st.warning(
                        "⏳ The server is waking up from idle — this can take up to a minute "
                        "on the free tier. Please wait a moment and try again."
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "🔌 Could not reach the backend. "
                        "Check that the API service is deployed and API_BASE_URL is correct."
                    )
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    # ── Status badge ───────────────────────────────────────────────────────
    if st.session_state.session_id:
        disp_name = st.session_state.filename or "Document"
        # Truncate long names for sidebar display
        if len(disp_name) > 28:
            disp_name = disp_name[:25] + "…"
        st.markdown(
            f'<div class="dm-sidebar-status active">'
            f'<span class="status-dot">●</span> {disp_name}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="dm-sidebar-status inactive">'
            '<span class="status-dot">●</span> No document loaded'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="dm-sidebar-footer">DocMind · RAG Pipeline v1.0</div>',
        unsafe_allow_html=True,
    )


# ── Main area: context header + chat ──────────────────────────────────────────
if st.session_state.session_id:

    # Document context bar — persistent workspace-style header
    ctx_name = st.session_state.filename or "Document"
    ctx_chunks = st.session_state.chunks_count
    ctx_mode = "Demo" if st.session_state.demo_mode else "Active"

    st.markdown(f"""
    <div class="dm-ctx-bar">
        <span>📄</span>
        <span class="dm-ctx-file">{ctx_name}</span>
        <span class="dm-ctx-sep">│</span>
        <span class="dm-ctx-detail">{ctx_chunks} chunks</span>
        <span class="dm-ctx-sep">│</span>
        <span class="dm-ctx-detail"><span class="dm-ctx-dot">●</span> {ctx_mode}</span>
    </div>
    """, unsafe_allow_html=True)

    # Demo mode banner
    if st.session_state.demo_mode:
        st.markdown("""
<div class="dm-demo-banner">
    <strong>Demo mode</strong> — You're chatting with DocMind's own SLA and Privacy Policy.<br>
    Try: <em>"What uptime does DocMind guarantee?"</em> or <em>"Who are DocMind's data sub-processors?"</em><br>
    <small style="color:#64748b;">Upload your own document in the sidebar to start a private session.</small>
</div>""", unsafe_allow_html=True)
    else:
        # Document preview pane (real uploads only)
        with st.expander(f"Preview extracted text", icon="📋"):
            st.text_area(
                "Extracted text (first few chunks)",
                st.session_state.preview_text,
                height=180,
                disabled=True,
            )

    st.divider()

    # ── Chat history ───────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_answer_metadata(
                    sources    = msg.get("sources", []),
                    confidence = msg.get("confidence", 0.0),
                    warning    = msg.get("warning"),
                )

    # ── Chat input ─────────────────────────────────────────────────────────
    chat_placeholder = (
        "Ask about DocMind's SLA or Privacy Policy…"
        if st.session_state.demo_mode
        else "Ask a question about your document…"
    )
    if prompt := st.chat_input(chat_placeholder):
        # 1. Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # 2. Call backend for answer
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking…"):
                try:
                    payload  = {"session_id": st.session_state.session_id, "query": prompt}
                    response = requests.post(
                        f"{API_URL}/chat",
                        json=payload,
                        timeout=BACKEND_TIMEOUT,
                    )

                    if response.status_code == 200:
                        data       = response.json()
                        answer     = data["answer"]
                        sources    = data["sources_cited"]
                        confidence = data["confidence"]
                        warning    = data["warning"]

                        st.markdown(answer)
                        _render_answer_metadata(sources, confidence, warning)

                        st.session_state.messages.append({
                            "role":       "assistant",
                            "content":    answer,
                            "sources":    sources,
                            "confidence": confidence,
                            "warning":    warning,
                        })

                    elif response.status_code == 404:
                        st.error("Session expired or invalid. Please re-upload your document.")
                        st.session_state.session_id = None
                    else:
                        detail = _safe_error_detail(response, "Generation failed")
                        st.error(f"Generation failed (HTTP {response.status_code}): {detail}")

                except requests.exceptions.Timeout:
                    st.warning(
                        "⏳ The server is waking up from idle — this can take up to a minute "
                        "on the free tier. Please wait a moment and try again."
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "🔌 Could not reach the backend. "
                        "Check that the API service is deployed and API_BASE_URL is correct."
                    )
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

else:
    # ── Welcome state ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="dm-welcome">
        <div class="dm-welcome-logo">D⟩</div>
        <h2>Welcome to DocMind</h2>
        <p>
            Upload a document in the sidebar to begin asking questions,
            or try the built-in demo to explore DocMind's capabilities.
        </p>
        <div class="dm-welcome-steps">
            <div class="dm-welcome-step">
                <div class="step-icon">📄</div>
                <strong>Upload</strong><br>PDF, TXT, DOCX
            </div>
            <div class="dm-welcome-step">
                <div class="step-icon">⚡</div>
                <strong>Process</strong><br>Chunk & embed
            </div>
            <div class="dm-welcome-step">
                <div class="step-icon">💬</div>
                <strong>Ask</strong><br>Get cited answers
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
