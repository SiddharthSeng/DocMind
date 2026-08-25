"""
app.py — Streamlit Frontend for DocMind
========================================
Provides a clean, intuitive UI for uploading documents and chatting with the RAG pipeline.
Communicates with the FastAPI backend (api.py).
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

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

st.title("🧠 DocMind")
st.markdown("Your private, local Document AI. Upload a file to start.")

# --- Sidebar: File Upload & Controls ---
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt", "docx"])
    
    if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
        # A new file was uploaded, trigger ingestion
        with st.spinner("Processing & Embedding Document..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.filename = data["filename"]
                    st.session_state.preview_text = data["preview"]
                    st.session_state.messages = []  # Clear chat for new document
                    st.success(f"Successfully processed {data['chunks_processed']} chunks!")
                else:
                    st.error(f"Upload Failed: {response.json().get('detail', 'Unknown Error')}")
                    st.session_state.session_id = None
                    
            except requests.exceptions.ConnectionError:
                st.error("Backend not reachable. Ensure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# --- Main Area: Preview & Chat ---
if st.session_state.session_id:
    # Optional Preview Pane
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
                    
    # Chat Input
    if prompt := st.chat_input("Ask a question about your document..."):
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
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data["sources_cited"]
                        confidence = data["confidence"]
                        warning = data["warning"]
                        
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
                        st.error(f"Generation Failed: {response.json().get('detail', 'Unknown Error')}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Backend not reachable. Ensure the FastAPI server is running on port 8000.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    
else:
    st.info("Upload a document in the sidebar to begin.")
