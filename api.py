"""
api.py — FastAPI backend for DocMind UI
=========================================
Exposes the DocMind RAG pipeline to a frontend UI via HTTP endpoints.
Includes session-scoped vector storage and automatic cleanup of old sessions.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_ingestion import process_documents
from vector_store import VectorStore
from generation import generate_answer

# Define absolute paths for storage to avoid working-directory issues
BASE_DIR = Path(__file__).parent.absolute()
SESSIONS_DIR = BASE_DIR / "sessions"

app = FastAPI(title="DocMind API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS_TO_KEEP = 5

def cleanup_old_sessions():
    """
    Keeps only the most recent N session folders and deletes the rest.
    Runs on startup to prevent disk bloat.
    """
    if not SESSIONS_DIR.exists():
        return

    # List all session directories
    session_dirs = [d for d in SESSIONS_DIR.iterdir() if d.is_dir()]
    
    # Sort by modification time, newest first
    session_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    # Delete anything beyond the max allowed
    for old_dir in session_dirs[MAX_SESSIONS_TO_KEEP:]:
        print(f"Cleaning up old session: {old_dir.name}")
        shutil.rmtree(old_dir, ignore_errors=True)


@app.on_event("startup")
async def startup_event():
    print("Starting DocMind API...")
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    cleanup_old_sessions()


class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources_cited: list[str]
    confidence: float
    warning: Optional[str] = None


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document, creates a new session, processes chunks, and embeds them.
    Returns the session_id to be used for subsequent chat requests.
    """
    ext = Path(file.filename).suffix.lower()
    
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {ext}. Only PDF and TXT are supported."
        )
        
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    session_path = SESSIONS_DIR / session_id
    os.makedirs(session_path, exist_ok=True)
    
    # Setup absolute paths for this session
    docs_folder = session_path / "docs"
    os.makedirs(docs_folder, exist_ok=True)
    
    file_path = docs_folder / file.filename
    chunks_json = session_path / "processed_chunks.json"
    chroma_dir = session_path / "chroma_db"
    
    # Save the uploaded file to the session's docs folder
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    try:
        # 1. Ingestion (Chunking) - explicitly using absolute paths
        process_documents(
            input_dir=str(docs_folder),
            output_file=str(chunks_json)
        )
        
        # 2. Embedding
        store = VectorStore(
            persist_directory=str(chroma_dir),
            chunks_json=str(chunks_json)
        )
        store.load_and_embed_chunks(str(chunks_json))
        
        # Get extracted text for preview
        preview_text = ""
        import json
        with open(chunks_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            # Combine the first few chunks for preview
            preview_text = "\n\n".join([c["text"] for c in chunks[:5]])
            if len(chunks) > 5:
                preview_text += "\n\n... (preview truncated) ..."
                
        if not preview_text.strip():
             raise HTTPException(
                status_code=422,
                detail="File processed, but no extractable text found (e.g., scanned PDF image)."
            )

        return {
            "session_id": session_id,
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "preview": preview_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Retrieves relevant chunks from the session's vector store and generates an answer.
    """
    session_path = SESSIONS_DIR / request.session_id
    
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found or expired.")
        
    chroma_dir = session_path / "chroma_db"
    chunks_json = session_path / "processed_chunks.json"
    
    try:
        # Load the session-specific VectorStore
        store = VectorStore(
            persist_directory=str(chroma_dir),
            chunks_json=str(chunks_json)
        )
        
        # Retrieve chunks
        chunks = store.retrieve(request.query, top_k=5)
        
        # Generate answer
        result = generate_answer(chunks, request.query)
        
        # UI Requirement: Filter out the "dropped chunks" warning if it's a correct abstention
        if result.get("warning") and "I don't have enough information" in result["answer"]:
             if "chunk(s) were dropped" in result["warning"]:
                 # Just clear the warning to avoid confusing the user on a correct refusal
                 result["warning"] = None
                 
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Run server on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
