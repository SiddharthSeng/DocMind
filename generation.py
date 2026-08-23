"""
generation.py — RAG Generation Layer for DocMind
=================================================
Connects the retrieval output from VectorStore.retrieve() to a local
Ollama LLM and returns a structured response: {answer, sources_cited, confidence}.

Hallucination-resistance strategy
----------------------------------
This module does NOT merely instruct the LLM to "be honest" — a prompt-level
plea that a model can silently violate.  Instead it uses four structural checks:

  1. Similarity gate  – chunks below a cosine-similarity threshold are dropped
     before the prompt is built, so low-quality retrievals never reach the LLM.
  2. Context-only system role  – the system prompt frames the model as a "lookup
     tool", not a conversational agent, which shifts its prior toward extraction.
  3. Post-generation attribution check  – after the LLM responds we verify that
     every source_file it cites actually appeared in the retrieved chunks.
     Phantom citations are stripped and flagged.
  4. Confidence derivation  – confidence is computed from retrieval similarity
     scores, NOT from the LLM's self-reported certainty, which is unreliable.

Usage
-----
    from vector_store import VectorStore
    from generation import generate_answer

    store  = VectorStore()
    chunks = store.retrieve("What is the refund policy?", top_k=5)
    result = generate_answer(chunks, "What is the refund policy?")
    print(result)
    # {
    #   "answer": "...",
    #   "sources_cited": ["policy.pdf"],
    #   "confidence": 0.87,
    #   "warning": None          # or a string if something was flagged
    # }
"""

import json
import re
import statistics
from typing import Any

import requests  # pip install requests  (already lightweight, no extra SDK needed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"  # default Ollama endpoint
DEFAULT_MODEL   = "llama3.1"   # change to any model you have pulled

# Chunks whose cosine similarity is below this threshold are dropped before
# they reach the prompt.  Tune this based on your embedding model's score
# distribution (print similarity scores during dev to calibrate).
MIN_SIMILARITY_THRESHOLD = 0.2

# If ALL chunks are below this threshold we return early without calling the
# LLM at all — there's genuinely nothing useful to ground an answer on.
ABSTAIN_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# 1. Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a document lookup assistant. Your ONLY job is to answer questions \
using the text passages provided below — nothing else.

Rules you must follow without exception:
- Base your answer EXCLUSIVELY on the passages marked [CHUNK n | source: FILE].
- After your answer, list every source file you drew on under the heading \
"Sources:" — one file name per line, no duplicates.
- If the provided passages do not contain enough information to answer the \
question, respond with exactly: "I don't have enough information in the \
provided documents to answer this question." Do NOT speculate or use outside \
knowledge.
- Do not make up file names, page numbers, or details that are not in the \
passages.
- Keep your answer concise and factual.\
"""

def _build_user_prompt(chunks: list[dict], question: str) -> str:
    """
    Builds the user turn of the prompt, injecting retrieved chunks and the
    question in a clearly delimited format.

    Each chunk is labeled [CHUNK n | source: filename] so the model can refer
    to it precisely and we can later verify its citations against real data.
    """
    if not chunks:
        # No chunks survived the similarity gate — signal that immediately.
        context_block = "[No relevant passages were retrieved for this query.]"
    else:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source_file", "unknown")
            text   = chunk.get("text", "").strip()
            lines.append(f"[CHUNK {i} | source: {source}]\n{text}")
        context_block = "\n\n---\n\n".join(lines)

    return (
        f"PASSAGES:\n\n{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the passages above. List sources at the end."
    )


# ---------------------------------------------------------------------------
# 2. LLM call (Ollama)
# ---------------------------------------------------------------------------

def _call_ollama(
    system_prompt: str,
    user_prompt:   str,
    model:         str = DEFAULT_MODEL,
    temperature:   float = 0.0,   # 0.0 = deterministic, reduces hallucination variance
    timeout:       int   = 120,
) -> str:
    """
    Sends a chat request to the local Ollama server and returns the raw text
    response.  Uses temperature=0 by default so outputs are reproducible and
    the model is less likely to invent details to fill silence.

    Raises:
        ConnectionError  -- if Ollama is not running.
        RuntimeError     -- if the API returns a non-200 status.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 1024,   # max tokens in reply
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            "Could not reach Ollama. Make sure it is running "
            f"(`ollama serve`) at {OLLAMA_BASE_URL}."
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    return data["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 3. Post-generation attribution check
# ---------------------------------------------------------------------------

def _extract_cited_sources(llm_response: str) -> list[str]:
    """
    Parses the 'Sources:' section from the LLM response.

    Expected format (as instructed in the system prompt):
        Sources:
        policy.pdf
        faq.txt

    Returns a deduplicated list of cited filenames (may be empty).
    """
    # Split on the "Sources:" heading (case-insensitive)
    parts = re.split(r"(?i)\bsources\s*:\s*", llm_response, maxsplit=1)
    if len(parts) < 2:
        return []  # Model didn't follow the citation instruction -- caller will flag this

    sources_block = parts[1].strip()
    cited = []
    for line in sources_block.splitlines():
        line = line.strip().lstrip("-*bullet ")
        if line:  # skip blank lines
            cited.append(line)
    return list(dict.fromkeys(cited))  # deduplicate while preserving order


def _verify_citations(
    cited_sources:    list[str],
    retrieved_chunks: list[dict],
) -> tuple[list[str], list[str]]:
    """
    Structural hallucination check: verifies that every source the LLM cited
    actually exists in the retrieved chunks.  Any citation that refers to a
    file that was NOT in the context is a hallucination -- the model invented it.

    Returns:
        verified_sources  -- citations that are legitimate (subset of retrieved)
        phantom_sources   -- citations that have no grounding in retrieved chunks
    """
    real_sources = {chunk["source_file"] for chunk in retrieved_chunks}

    verified = [s for s in cited_sources if s in real_sources]
    phantoms  = [s for s in cited_sources if s not in real_sources]
    return verified, phantoms


def _strip_sources_section(llm_response: str) -> str:
    """Returns only the answer portion, removing the trailing Sources: block."""
    parts = re.split(r"(?i)\bsources\s*:\s*", llm_response, maxsplit=1)
    return parts[0].strip()


# ---------------------------------------------------------------------------
# 4. Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(chunks: list[dict]) -> float:
    """
    Derives a confidence score (0.0-1.0) from retrieval similarity scores,
    NOT from the LLM's self-assessment.

    Why this matters: LLMs are poorly calibrated -- a model that is making
    something up often expresses just as much confidence as one that isn't.
    Similarity scores from the embedding model are a far more reliable signal
    of whether relevant context was actually found.

    Formula: weighted average of top-chunk similarities, with the top chunk
    contributing more weight than the rest. Capped at 1.0.
    """
    if not chunks:
        return 0.0

    scores = [c.get("similarity_score", 0.0) for c in chunks]

    if len(scores) == 1:
        return round(min(scores[0], 1.0), 4)

    # Give the top chunk 50% of the weight, distribute the rest evenly
    top_weight   = 0.5
    rest_weight  = 0.5 / (len(scores) - 1)
    weighted_sum = scores[0] * top_weight + sum(s * rest_weight for s in scores[1:])
    return round(min(weighted_sum, 1.0), 4)


# ---------------------------------------------------------------------------
# 5. Main public function
# ---------------------------------------------------------------------------

def generate_answer(
    retrieved_chunks: list[dict],
    question:         str,
    model:            str   = DEFAULT_MODEL,
    temperature:      float = 0.0,
) -> dict[str, Any]:
    """
    Full generation step for DocMind's RAG pipeline.

    Args:
        retrieved_chunks: Output of VectorStore.retrieve() -- list of dicts with
                          keys: text, source_file, chunk_index, similarity_score.
        question:         The user's original question string.
        model:            Ollama model tag (e.g. "llama3.1:8b-instruct").
        temperature:      Sampling temperature. 0.0 = deterministic (recommended
                          for RAG to minimise hallucination variance).

    Returns:
        dict with keys:
            answer         (str)         -- cleaned answer text
            sources_cited  (list[str])   -- verified source filenames
            confidence     (float)       -- 0.0-1.0 derived from similarity scores
            warning        (str | None)  -- populated if any anomalies were detected
    """
    warnings: list[str] = []

    # --- Similarity gate: filter out low-quality chunks ---
    # We do this BEFORE building the prompt so junk context never reaches the LLM.
    good_chunks = [
        c for c in retrieved_chunks
        if c.get("similarity_score", 0.0) >= MIN_SIMILARITY_THRESHOLD
    ]

    # If everything is below the abstain threshold, don't call the LLM at all.
    if not good_chunks or max(
        c.get("similarity_score", 0.0) for c in retrieved_chunks
    ) < ABSTAIN_THRESHOLD:
        return {
            "answer": "I don't have enough information in the provided documents to answer this question.",
            "sources_cited": [],
            "confidence": 0.0,
            "warning": (
                "All retrieved chunks had similarity scores below the abstain "
                f"threshold ({ABSTAIN_THRESHOLD}). The LLM was not called."
            ),
        }

    if len(good_chunks) < len(retrieved_chunks):
        dropped = len(retrieved_chunks) - len(good_chunks)
        warnings.append(
            f"{dropped} chunk(s) were dropped (similarity < {MIN_SIMILARITY_THRESHOLD})."
        )

    # --- Build prompt ---
    user_prompt = _build_user_prompt(good_chunks, question)

    # --- Call LLM ---
    raw_response = _call_ollama(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
    )

    # --- Extract and verify citations ---
    cited_sources              = _extract_cited_sources(raw_response)
    verified_sources, phantoms = _verify_citations(cited_sources, good_chunks)

    if not cited_sources:
        # Model answered but gave no sources at all -- common failure mode
        warnings.append(
            "Model did not include a 'Sources:' section. "
            "It may have answered from training knowledge rather than the provided chunks."
        )

    if phantoms:
        # Model hallucinated file names that weren't in the retrieved context
        warnings.append(
            f"Phantom citations detected and removed: {phantoms}. "
            "These files were not in the retrieved chunks."
        )

    # --- Check for abstention phrasing in the answer body ---
    answer_text = _strip_sources_section(raw_response)
    abstain_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "not enough information",
        "cannot answer",
        "the provided documents do not",
    ]
    model_abstained = any(p in answer_text.lower() for p in abstain_phrases)

    if model_abstained:
        # Model correctly flagged insufficient context
        verified_sources = []

    # --- Derive confidence from retrieval scores, not the LLM ---
    confidence = _compute_confidence(good_chunks) if not model_abstained else 0.0

    return {
        "answer":        answer_text,
        "sources_cited": verified_sources,
        "confidence":    confidence,
        "warning":       "; ".join(warnings) if warnings else None,
    }


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from vector_store import VectorStore

    print("DocMind -- Generation Layer Smoke Test")
    print("=" * 50)

    store = VectorStore()
    if store.collection.count() == 0:
        print("Vector store is empty. Run data_ingestion.py and vector_store.py first.")
        sys.exit(1)

    test_question = input("Enter a test question: ").strip()
    chunks = store.retrieve(test_question, top_k=5)

    print(f"\nRetrieved {len(chunks)} chunks.")
    for c in chunks:
        print(f"  [{c['similarity_score']:.3f}] {c['source_file']} chunk {c['chunk_index']}")

    print("\nCalling LLM...\n")
    result = generate_answer(chunks, test_question)

    print(f"ANSWER:\n{result['answer']}\n")
    print(f"SOURCES CITED : {result['sources_cited']}")
    print(f"CONFIDENCE    : {result['confidence']}")
    if result["warning"]:
        print(f"WARNING       : {result['warning']}")
