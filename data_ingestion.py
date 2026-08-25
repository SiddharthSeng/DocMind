import os
import json
import uuid
import pdfplumber
import tiktoken
from pathlib import Path
from typing import List, Dict, Any

def extract_text_from_txt(file_path: str) -> str:
    """Reads raw text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from all pages of a .pdf file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                # Some pages might be scanned images with no text
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of roughly `chunk_size` tokens,
    with an overlap of `overlap` tokens between consecutive chunks.
    """
    # We use 'tiktoken' to count actual tokens rather than words.
    # 'cl100k_base' is the standard tokenizer used by modern OpenAI models (like GPT-4),
    # which is a great baseline for general RAG systems.
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    
    # If the text is empty, return an empty list
    if len(tokens) == 0:
        return []
    
    # How the overlap logic works:
    # We iterate over the list of tokens. Instead of jumping forward by the full 
    # `chunk_size` (which would give zero overlap), we jump forward by `chunk_size - overlap`.
    # This means the last `overlap` number of tokens from the current chunk will be included 
    # at the start of the next chunk. 
    # Overlap is crucial in RAG so we don't accidentally split a sentence or concept in half!
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        # Slice the tokens list from the current index `i` to `i + chunk_size`
        chunk_tokens = tokens[i:i + chunk_size]
        
        # Decode the tokens back into a readable string
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
    return chunks

def process_documents(input_dir: str, output_file: str, chunk_size: int = 500, overlap: int = 50):
    """
    Loads documents from a directory, extracts text, chunks it, and saves to JSON.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Directory not found: {input_dir}")
        return

    all_chunks: List[Dict[str, Any]] = []
    
    # Iterate through all files in the given directory
    for file_name in os.listdir(input_dir):
        file_path = input_path / file_name
        
        # Skip if it's a directory
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix.lower()
        
        # Extract text based on file type
        if ext == '.txt':
            print(f"Processing TXT: {file_name}")
            text = extract_text_from_txt(str(file_path))
        elif ext == '.pdf':
            print(f"Processing PDF: {file_name}")
            text = extract_text_from_pdf(str(file_path))
        else:
            # Skip unsupported formats (like .docx, images, etc. for now)
            continue
            
        # Ignore empty documents
        if not text.strip():
            print(f"Warning: No text found in {file_name}")
            continue
            
        # Split text into tokenized chunks
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        # Store metadata and chunk info
        for index, chunk in enumerate(text_chunks):
            chunk_data = {
                "chunk_id": f"{file_name}_chunk_{index}",  # Unique deterministic ID for each chunk
                "source_file": file_name,       # Keep track of where it came from
                "text": chunk,                  # The actual raw text of the chunk
                "chunk_index": index            # Its position in the source document
            }
            all_chunks.append(chunk_data)
            
    # Save the results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False handles special characters properly
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully processed {len(all_chunks)} chunks from {input_dir}.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    # Settings for our RAG ingestion
    DOCS_FOLDER = "docs"
    OUTPUT_JSON = "processed_chunks.json"
    CHUNK_SIZE = 150
    OVERLAP_SIZE = 30
    
    # Create the docs folder if it doesn't exist so you can easily drop files in it
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    
    print(f"Looking for documents in '{DOCS_FOLDER}'...")
    process_documents(
        input_dir=DOCS_FOLDER, 
        output_file=OUTPUT_JSON, 
        chunk_size=CHUNK_SIZE, 
        overlap=OVERLAP_SIZE
    )
