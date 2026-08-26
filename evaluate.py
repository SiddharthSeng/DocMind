"""
evaluate.py — Formal Evaluation Harness for DocMind RAG Pipeline
=================================================================
Purpose
-------
This script produces clean, final evaluation metrics suitable for a resume or
interview defence. It is DISTINCT from the calibration/diagnostic scripts
(diag_similarity.py, run_calibration.py), which were used to tune thresholds
and explore score distributions. This harness assumes the pipeline is already
tuned and is solely concerned with measuring its performance against a fixed
test set.

Test set
--------
  - 11 queries loaded from test_cases.json
  - Types: exact-fact (2), paraphrased (6), unrelated/abstention (3)
  - Each retrieval query carries an expected_chunk_id where applicable

Metrics computed
----------------
  1. RETRIEVAL (queries with expected_chunk_id):
       - Precision@1, Precision@3, Precision@5
       - Mean Reciprocal Rank (MRR)
       - All broken down by type (exact-fact vs. paraphrased)

  2. GENERATION (all queries):
       - Correct abstention rate for unrelated queries
       - LLM-as-judge faithfulness score (1-5) for answered queries
         (see FAITHFULNESS NOTE below for assumptions/limitations)

Output
------
  - Printed summary to console
  - eval_report.json  -- latest run's full results
  - eval_history.json -- append-only log, one entry per run with timestamp
                         so improvement across pipeline iterations is trackable

Small-sample caveat
-------------------
11 test cases is an appropriate demonstration size for a portfolio project.
The metrics are internally consistent and meaningful at this scale, but they
are NOT statistically robust -- confidence intervals would be very wide, and
a single query flipping could move a metric by ~9 percentage points. Both the
console output and the saved report explicitly note this.

Usage
-----
    python evaluate.py
    # or, to skip the slower LLM faithfulness scoring:
    python evaluate.py --skip-faithfulness

FAITHFULNESS NOTE (LLM-as-judge approach)
------------------------------------------
To measure generation faithfulness without a large hand-labelled dataset, we
use an "LLM-as-judge" approach: the same local Ollama LLM is prompted to rate
how faithfully the generated answer matches the reference content on a 1-5 scale.

ASSUMPTIONS AND LIMITATIONS (be honest about these in an interview):
  1. Self-referential bias: The judge is the same model as the generator. A model
     that tends to generate a particular style of hallucination may also fail to
     detect that style when judging. An independent judge model would be stronger.
  2. Reference answers are chunk text, not gold-standard human answers: We use the
     expected chunk's raw text as the "reference". This is a reasonable proxy for
     small corpora where the chunk IS the authoritative content, but it penalises
     answers that correctly synthesise across chunks or rephrase faithfully.
  3. Score granularity: A 1-5 integer scale is coarse. The aggregate mean is used
     as a point estimate -- treat it as directional, not precise.
  4. Temperature=0 is used for the judge call to maximise reproducibility, but
     LLM scoring of subjective quality still has variance across runs.
  5. This is appropriate as a lightweight internal metric for a portfolio demo.
     Production-grade faithfulness would use RAGAS, TruLens, or a held-out human
     annotation study with inter-annotator agreement metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Local imports (same package)
# ---------------------------------------------------------------------------
from generation import _call_ollama, generate_answer
from vector_store import VectorStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_CASES_PATH = Path("test_cases.json")
REPORT_PATH     = Path("eval_report.json")
HISTORY_PATH    = Path("eval_history.json")

TOP_K_VALUES    = [1, 3, 5]    # Precision@k values to compute
RETRIEVE_TOP_K  = 5            # How many chunks to fetch per query
ABSTAIN_PHRASE  = "i don't have enough information"  # canonical abstention marker

FAITHFULNESS_SYSTEM_PROMPT = """\
You are an impartial evaluator assessing the faithfulness of an AI-generated answer
against a reference passage. Faithfulness means: does the answer contain only
information that is supported by (or is a fair paraphrase of) the reference passage?

Score the answer on a scale of 1 to 5:
  5 = Fully faithful -- every claim is directly supported by the reference.
  4 = Mostly faithful -- minor wording differences, no unsupported claims.
  3 = Partially faithful -- some claims are supported, but gaps or mild drift exist.
  2 = Mostly unfaithful -- significant unsupported claims or factual drift.
  1 = Completely unfaithful -- answer contradicts or ignores the reference.

Respond with ONLY a single integer (1, 2, 3, 4, or 5). No explanation.\
"""


# ---------------------------------------------------------------------------
# Load test cases
# ---------------------------------------------------------------------------

def load_test_cases(path: Path) -> list[dict]:
    """Loads and validates test cases from JSON."""
    if not path.exists():
        sys.exit(f"[ERROR] Test cases file not found: {path}")
    with path.open(encoding="utf-8") as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} test cases from {path}")
    return cases


# ---------------------------------------------------------------------------
# Retrieval metrics helpers
# ---------------------------------------------------------------------------

def _rank_of_expected(chunks: list[dict], expected_chunk_id: str) -> int | None:
    """
    Returns the 1-based rank of the expected chunk in the retrieved list,
    or None if it was not found in the top-k results.
    """
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.get("chunk_id") == expected_chunk_id:
            return rank
    return None


def precision_at_k(rank: int | None, k: int) -> float:
    """Binary Precision@k: 1.0 if the expected chunk appeared in the top-k, else 0.0."""
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """Reciprocal Rank: 1/rank if found, 0.0 otherwise."""
    if rank is None:
        return 0.0
    return 1.0 / rank


def _compute_retrieval_metrics(results: list[dict]) -> dict[str, Any]:
    """
    Given a list of per-query retrieval result dicts, compute aggregate
    Precision@k and MRR, plus a breakdown by query type.

    Each result dict must have keys: type, rank (int|None).
    """
    def _aggregate(subset: list[dict]) -> dict[str, Any]:
        if not subset:
            return {}
        n = len(subset)
        metrics: dict[str, Any] = {}
        for k in TOP_K_VALUES:
            metrics[f"precision_at_{k}"] = round(
                sum(precision_at_k(r["rank"], k) for r in subset) / n, 4
            )
        metrics["mrr"]   = round(sum(reciprocal_rank(r["rank"]) for r in subset) / n, 4)
        metrics["count"] = n
        return metrics

    overall    = _aggregate(results)
    exact_fact = _aggregate([r for r in results if r["type"] == "exact-fact"])
    paraphrase = _aggregate([r for r in results if r["type"] == "paraphrased"])

    return {
        "overall":  overall,
        "by_type": {
            "exact-fact":  exact_fact,
            "paraphrased": paraphrase,
        },
    }


# ---------------------------------------------------------------------------
# Faithfulness scoring (LLM-as-judge)
# ---------------------------------------------------------------------------

def _score_faithfulness(answer: str, reference_chunk_text: str) -> int | None:
    """
    Asks the local LLM to rate the faithfulness of `answer` against
    `reference_chunk_text` on a 1-5 scale.

    Returns the integer score, or None if the model's response could not
    be parsed as an integer (which we treat as a scoring failure).

    See module-level FAITHFULNESS NOTE for known limitations.
    """
    user_prompt = (
        f"REFERENCE PASSAGE:\n{reference_chunk_text.strip()}\n\n"
        f"GENERATED ANSWER:\n{answer.strip()}\n\n"
        "Faithfulness score (1-5):"
    )
    try:
        raw = _call_ollama(
            system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        # Extract the first integer found in the response (model might add a period or space)
        match = re.search(r"\b[1-5]\b", raw)
        if match:
            return int(match.group())
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] Faithfulness scoring failed: {exc}")
        return None


def _get_chunk_text_by_id(chunk_id: str, store: VectorStore) -> str | None:
    """
    Retrieves the raw text of a chunk by its ID from ChromaDB.
    Used to fetch the reference text for faithfulness scoring.
    """
    try:
        result = store.collection.get(ids=[chunk_id], include=["documents"])
        docs = result.get("documents", [])
        if docs and docs[0]:
            return docs[0]
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] Could not fetch chunk {chunk_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    skip_faithfulness: bool = False,
    retrieval_only:    bool = False,
) -> dict[str, Any]:
    """
    Runs the full evaluation harness and returns a results dict.

    Args:
        skip_faithfulness: If True, skip the LLM faithfulness scoring step.
                           Useful for faster re-runs focused on retrieval metrics,
                           or when the LLM is unavailable.
        retrieval_only:    If True, skip ALL generation steps (no Ollama required).
                           Only Precision@k, MRR, and per-query retrieval rank are
                           reported. Abstention and faithfulness are reported as N/A.
                           Implies skip_faithfulness=True.
    """
    if retrieval_only:
        skip_faithfulness = True
    cases = load_test_cases(TEST_CASES_PATH)

    print("\nInitialising VectorStore...")
    store = VectorStore()
    chunk_count = store.collection.count()
    if chunk_count == 0:
        sys.exit(
            "[ERROR] Vector store is empty. Run data_ingestion.py and vector_store.py first."
        )
    print(f"Vector store ready -- {chunk_count} chunks indexed.\n")

    # ------------------------------------------------------------------
    # Per-query evaluation
    # ------------------------------------------------------------------
    retrieval_results:  list[dict] = []  # for queries with expected_chunk_id
    generation_results: list[dict] = []  # for all queries

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        q_id     = case["id"]
        query    = case["query"]
        q_type   = case["type"]
        expected = case.get("expected_chunk_id")  # None for unrelated queries

        print(f"[{idx:2d}/{total}] {q_id} ({q_type})")
        print(f"       Query: {query[:80]}{'...' if len(query) > 80 else ''}")

        # ---- Retrieval -----------------------------------------------
        chunks = store.retrieve(query, top_k=RETRIEVE_TOP_K)
        retrieved_ids = [c["chunk_id"] for c in chunks]

        if expected is not None:
            rank = _rank_of_expected(chunks, expected)
            retrieval_results.append({
                "id":               q_id,
                "type":             q_type,
                "query":            query,
                "rank":             rank,
                "expected_chunk_id": expected,
                "retrieved_ids":    retrieved_ids,
            })
            rank_str = str(rank) if rank is not None else "NOT FOUND"
            print(f"       Retrieval rank of expected chunk: {rank_str}")

        # ---- Generation (skipped when --retrieval-only) -----------------
        if retrieval_only:
            # No LLM call: record a placeholder so aggregate loops don't break.
            gen_entry: dict[str, Any] = {
                "id":            q_id,
                "type":          q_type,
                "query":         query,
                "answer":        None,
                "is_abstention": None,
                "confidence":    None,
                "warning":       None,
                "faithfulness_score": None,
            }
            generation_results.append(gen_entry)
            print()
            continue

        gen = generate_answer(chunks, query)
        answer = gen["answer"]
        is_abstention = ABSTAIN_PHRASE in answer.lower()

        print(f"       Answer preview: {answer[:100].strip()}{'...' if len(answer) > 100 else ''}")

        gen_entry = {
            "id":            q_id,
            "type":          q_type,
            "query":         query,
            "answer":        answer,
            "is_abstention": is_abstention,
            "confidence":    gen["confidence"],
            "warning":       gen.get("warning"),
        }

        # ---- Faithfulness (only for queries that should have an answer) ----
        faithfulness_score: int | None = None
        if not skip_faithfulness and expected is not None and not is_abstention:
            ref_text = _get_chunk_text_by_id(expected, store)
            if ref_text:
                print("       Scoring faithfulness (LLM-as-judge)...")
                faithfulness_score = _score_faithfulness(answer, ref_text)
                print(f"       Faithfulness score: {faithfulness_score}/5")
            else:
                print("       [WARN] Could not fetch reference chunk text for faithfulness scoring.")
        elif is_abstention and expected is not None:
            # Model abstained on a question it should have answered -- faithfulness N/A,
            # but this is a retrieval/generation failure we surface in the summary.
            print("       [NOTE] Model abstained on a query that has an expected answer.")

        gen_entry["faithfulness_score"] = faithfulness_score
        generation_results.append(gen_entry)
        print()

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    # --- Retrieval metrics ---
    retrieval_metrics = _compute_retrieval_metrics(retrieval_results)

    # --- Abstention metrics ---
    unrelated_cases = [r for r in generation_results if r["type"] == "unrelated"]
    n_unrelated = len(unrelated_cases)
    if retrieval_only:
        n_correct_abstentions = None
        abstention_rate = None
    else:
        n_correct_abstentions = sum(1 for r in unrelated_cases if r["is_abstention"])
        abstention_rate = (
            round(n_correct_abstentions / n_unrelated, 4) if n_unrelated > 0 else None
        )

    # Count false abstentions (model abstained when it should have answered)
    should_answer = [r for r in generation_results if r["type"] != "unrelated"]
    n_false_abstentions = (
        None if retrieval_only
        else sum(1 for r in should_answer if r["is_abstention"])
    )

    # --- Faithfulness metrics ---
    faith_scores = [
        r["faithfulness_score"]
        for r in generation_results
        if r["faithfulness_score"] is not None
    ]
    mean_faithfulness = (
        round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None
    )

    # ------------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------------
    timestamp = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {
        "run_timestamp": timestamp,
        "test_set_size": total,
        "pipeline_notes": {
            "retriever": "Hybrid RRF (BAAI/bge-small-en-v1.5 + BM25Okapi, K=10)",
            "llm":       "Ollama (llama3.1 local)" if not retrieval_only else "N/A (retrieval-only run)",
            "top_k":     RETRIEVE_TOP_K,
            "retrieval_only": retrieval_only,
            "small_sample_caveat": (
                f"All metrics are computed on {total} test cases. "
                "This is an appropriate demo/portfolio size but is NOT statistically robust -- "
                "a single query change moves each metric by ~9 percentage points. "
                "Treat these numbers as directional indicators, not production benchmarks."
            ),
        },
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": {
            "abstention": {
                "n_unrelated_queries":        n_unrelated,
                "n_correct_abstentions":      n_correct_abstentions,
                "correct_abstention_rate_pct": (
                    round(abstention_rate * 100, 1) if abstention_rate is not None else None
                ),
                "n_false_abstentions": n_false_abstentions,
                "note": (
                    "A 'correct abstention' is when the model responds with the "
                    "canonical refusal phrase for a query that has no grounding document."
                ),
            },
            "faithfulness": {
                "method":      "LLM-as-judge (same Ollama model, temperature=0)",
                "scale":       "1 (unfaithful) to 5 (fully faithful)",
                "n_scored":    len(faith_scores),
                "mean_score":  mean_faithfulness,
                "all_scores":  faith_scores,
                "skipped":     skip_faithfulness,
                "limitations": (
                    "Self-referential judge (same model generates and evaluates). "
                    "Reference is raw chunk text, not a gold-standard answer. "
                    "Score is directional -- treat as internal quality signal only. "
                    "See module docstring for full list of assumptions."
                ),
            },
        },
        "per_query_details": generation_results,
        "retrieval_detail":  retrieval_results,
    }

    return report


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def print_summary(report: dict[str, Any]) -> None:
    """Prints a clean, quote-ready summary of the evaluation report."""
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        
    rm  = report["retrieval_metrics"]
    gm  = report["generation_metrics"]
    ov  = rm["overall"]
    abt = gm["abstention"]
    fth = gm["faithfulness"]
    ts  = report["run_timestamp"]
    n   = report["test_set_size"]

    sep = "=" * 68

    print(f"\n{sep}")
    print("  DocMind RAG Pipeline -- Formal Evaluation Report")
    print(f"  Run: {ts}")
    print(sep)

    print(
        f"\n  Test set: {n} queries  |  "
        f"Retriever: {report['pipeline_notes']['retriever']}"
    )
    print(
        f"  LLM: {report['pipeline_notes']['llm']}  |  "
        f"top_k={report['pipeline_notes']['top_k']}"
    )

    # ---- Retrieval metrics ----
    print(f"\n{'─' * 68}")
    print("  RETRIEVAL METRICS  (queries with a labelled expected chunk)")
    print(f"{'─' * 68}")
    print(f"  {'Metric':<28} {'Overall':>10} {'Exact-fact':>12} {'Paraphrased':>13}")
    print(f"  {'─'*28} {'─'*10} {'─'*12} {'─'*13}")

    ef  = rm["by_type"].get("exact-fact",  {})
    par = rm["by_type"].get("paraphrased", {})

    for k in TOP_K_VALUES:
        key     = f"precision_at_{k}"
        label   = f"Precision@{k}"
        ov_val  = f"{ov.get(key, 0) * 100:.0f}%"
        ef_val  = f"{ef.get(key, 0) * 100:.0f}%" if ef  else "N/A"
        par_val = f"{par.get(key, 0) * 100:.0f}%" if par else "N/A"
        print(f"  {label:<28} {ov_val:>10} {ef_val:>12} {par_val:>13}")

    mrr_ov  = f"{ov.get('mrr', 0):.3f}"
    mrr_ef  = f"{ef.get('mrr', 0):.3f}"  if ef  else "N/A"
    mrr_par = f"{par.get('mrr', 0):.3f}" if par else "N/A"
    print(f"  {'MRR':<28} {mrr_ov:>10} {mrr_ef:>12} {mrr_par:>13}")
    print(
        f"  {'n (queries)':<28} "
        f"{ov.get('count', 0):>10} "
        f"{ef.get('count', 0):>12} "
        f"{par.get('count', 0):>13}"
    )

    # ---- Generation metrics ----
    print(f"\n{'─' * 68}")
    print("  GENERATION METRICS")
    print(f"{'─' * 68}")

    abt_rate = abt.get("correct_abstention_rate_pct")
    abt_str  = f"{abt_rate:.0f}%" if abt_rate is not None else "N/A"
    n_correct = abt["n_correct_abstentions"]
    n_correct_str = str(n_correct) if n_correct is not None else "N/A"
    print(
        f"  Correct abstention rate:    {abt_str}  "
        f"({n_correct_str}/{abt['n_unrelated_queries']} unrelated queries)"
    )
    n_false = abt["n_false_abstentions"]
    if n_false is not None and n_false > 0:
        print(
            f"  WARNING  False abstentions: {n_false}  "
            "(model refused queries it should have answered)"
        )

    if fth["skipped"]:
        print("  Faithfulness scoring:       SKIPPED (--skip-faithfulness)")
    elif fth["mean_score"] is not None:
        stars = "★" * round(fth["mean_score"]) + "☆" * (5 - round(fth["mean_score"]))
        print(
            f"  Mean faithfulness score:    {fth['mean_score']:.2f}/5  {stars}  "
            f"(n={fth['n_scored']}, LLM-as-judge, 1-5 scale)"
        )
    else:
        print("  Faithfulness scoring:       No scores recorded.")

    # ---- Resume-ready one-liner ----
    print(f"\n{'─' * 68}")
    print("  RESUME-READY SUMMARY  (pair with small-sample caveat below)")
    print(f"{'─' * 68}")
    p5_pct  = round(ov.get("precision_at_5", 0) * 100)
    mrr_val = ov.get("mrr", 0)
    faith_line = (
        f", mean faithfulness {fth['mean_score']:.1f}/5 (LLM-as-judge)"
        if fth["mean_score"] is not None
        else ""
    )
    print(
        f'\n  "Achieved {p5_pct}% Precision@5 and MRR of {mrr_val:.2f} on retrieval,\n'
        f"  {abt_str} correct abstention rate on out-of-scope queries{faith_line},\n"
        f'  evaluated on a {n}-query labelled test set (exact-fact + paraphrase + abstention)."\n'
    )

    # ---- Caveats ----
    print(f"{'─' * 68}")
    print("  SMALL-SAMPLE CAVEATS  (include this context when citing metrics)")
    print(f"{'─' * 68}")
    print(f"  {report['pipeline_notes']['small_sample_caveat']}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

def _append_to_history(report: dict[str, Any]) -> None:
    """
    Appends a compact summary of this run to eval_history.json.
    Each entry is a summary row -- not the full per-query detail -- so the
    history file stays readable and compact over many iterations.
    """
    rm  = report["retrieval_metrics"]["overall"]
    abt = report["generation_metrics"]["abstention"]
    fth = report["generation_metrics"]["faithfulness"]

    entry = {
        "timestamp":               report["run_timestamp"],
        "test_set_size":           report["test_set_size"],
        "precision_at_1":          rm.get("precision_at_1"),
        "precision_at_3":          rm.get("precision_at_3"),
        "precision_at_5":          rm.get("precision_at_5"),
        "mrr":                     rm.get("mrr"),
        "correct_abstention_rate": abt.get("correct_abstention_rate_pct"),
        "mean_faithfulness":       fth.get("mean_score"),
        "faithfulness_n_scored":   fth.get("n_scored"),
        "faithfulness_skipped":    fth.get("skipped"),
    }

    history: list[dict] = []
    if HISTORY_PATH.exists():
        with HISTORY_PATH.open(encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Could not parse {HISTORY_PATH}; starting fresh history.")

    history.append(entry)

    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Run appended to history log: {HISTORY_PATH}  ({len(history)} total runs)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocMind RAG Pipeline -- Formal Evaluation Harness"
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        default=False,
        help=(
            "Skip the LLM-as-judge faithfulness scoring step. "
            "Useful for faster re-runs focused on retrieval metrics only."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        default=False,
        help=(
            "Skip ALL generation steps — no Ollama required. "
            "Reports Precision@k and MRR only. "
            "Abstention and faithfulness metrics are reported as N/A."
        ),
    )
    args = parser.parse_args()

    print("\nDocMind RAG Pipeline -- Formal Evaluation Harness")
    print("=" * 68)
    if args.skip_faithfulness:
        print("[INFO] Faithfulness scoring disabled (--skip-faithfulness).\n")
    if args.retrieval_only:
        print("[INFO] Retrieval-only mode — generation and faithfulness skipped (no Ollama required).\n")

    # Run evaluation
    report = run_evaluation(
        skip_faithfulness=args.skip_faithfulness,
        retrieval_only=args.retrieval_only,
    )

    # Print clean summary
    print_summary(report)

    # Save full report
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {REPORT_PATH}")

    # Append compact entry to history log
    _append_to_history(report)


if __name__ == "__main__":
    main()
