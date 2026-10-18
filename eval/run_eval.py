"""
run_eval.py — Gold-set evaluation harness for DocMind
======================================================
Measures what the 80% target actually refers to: ANSWER ACCURACY on real
documents, plus retrieval quality and abstention behaviour.  Replaces the
retrieval-only evaluate.py, whose 5-chunk corpus made Precision@5 trivially 100%.

Each document in the gold set is indexed on its own (mirroring one /upload
session), then every question for that document is asked against it.

Usage
-----
    # 1. Check every gold evidence snippet exists in the extracted text
    python eval/run_eval.py --validate

    # 2. Retrieval metrics only (no LLM call)
    python eval/run_eval.py --pipeline baseline --retrieval-only

    # 3. Full run: retrieval + generation + answer accuracy
    python eval/run_eval.py --pipeline baseline

Options: --docs <substring> to restrict to matching documents, --limit N per doc.

Outputs (eval/results/, gitignored because answers quote private documents):
    <timestamp>_<pipeline>.json   full per-question results
    <timestamp>_<pipeline>.csv    per-question table for manual spot-checking
    history.jsonl                 one summary line per run

Metric definitions
------------------
  accuracy                 correct / all questions (answerable + unanswerable)
  answerable_accuracy      correct / answerable questions
  abstention_recall        unanswerable questions the system declined to answer
  false_abstention_rate    answerable questions that got the canonical refusal
  evidence_hit@k           answerable questions whose evidence snippets all
                           appear within the top-k retrieved chunks
  evidence_mrr             mean of 1/rank of that covering position
Correctness rules are documented in eval/scoring.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parent
GOLD_PATH = EVAL_DIR / "gold" / "qa_gold.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
DOCS_DIR = REPO_DIR / "docs"
HIT_KS = (1, 3, 5, 10)

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(EVAL_DIR))


def resolve_doc(doc: str) -> Path:
    """Gold `doc` is a filename in docs/ (private docs) or a repo-relative path (eval/corpus/...)."""
    in_docs = DOCS_DIR / doc
    return in_docs if in_docs.exists() else REPO_DIR / doc


def load_gold(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_doc(items: list[dict], doc_filter: str | None, limit: int | None) -> dict[str, list[dict]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if doc_filter and doc_filter.lower() not in item["doc"].lower():
            continue
        by_doc[item["doc"]].append(item)
    if limit:
        by_doc = {d: qs[:limit] for d, qs in by_doc.items()}
    return dict(by_doc)


# ---------------------------------------------------------------------------
# --validate
# ---------------------------------------------------------------------------

def _pdfium_text(path: Path) -> str:
    """Reference PDF text from PDFium, which keeps reading order and word spacing on multi-column layouts."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    text = "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    # PDFium marks line-break hyphenation with U+FFFE; rejoin the split word.
    return text.replace("\r\n", "\n").replace("￾\n", "").replace("￾", "")


def validate(items: list[dict]) -> int:
    """
    Checks every evidence snippet exists in its document.  A snippet passes if
    either the baseline extractor (data_ingestion) or PDFium finds it; snippets
    only PDFium finds are reported per document as text the baseline loses.
    """
    from data_ingestion import extract_text_from_docx, extract_text_from_pdf, extract_text_from_txt
    from scoring import evidence_in_text

    extractors = {".pdf": extract_text_from_pdf, ".docx": extract_text_from_docx, ".txt": extract_text_from_txt}
    texts: dict[str, tuple[str, str]] = {}
    lost_by_baseline: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    missing = 0
    ids = set()
    for item in items:
        if item["id"] in ids:
            print(f"  DUPLICATE id {item['id']}")
            missing += 1
        ids.add(item["id"])
        doc = item["doc"]
        if doc not in texts:
            path = resolve_doc(doc)
            if not path.exists():
                print(f"  MISSING DOCUMENT {path}")
                texts[doc] = ("", "")
            else:
                base = extractors[path.suffix.lower()](str(path))
                ref = _pdfium_text(path) if path.suffix.lower() == ".pdf" else base
                texts[doc] = (base, ref)
        base, ref = texts[doc]
        for ev in item.get("evidence", []):
            in_base, in_ref = evidence_in_text(ev, base), evidence_in_text(ev, ref)
            lost_by_baseline[doc][1] += 1
            if not in_base:
                lost_by_baseline[doc][0] += 1
            if not (in_base or in_ref):
                print(f"  {item['id']}: evidence not found -> {ev!r}")
                missing += 1
    n_ev = sum(len(i.get("evidence", [])) for i in items)
    print(f"Validated {len(items)} questions / {n_ev} evidence snippets: {missing} problem(s).")
    print("\n  Evidence the baseline extractor cannot see (found only by PDFium):")
    for doc, (lost, total) in lost_by_baseline.items():
        print(f"    {lost:3d}/{total:<3d} {doc}")
    return missing


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_opts(pairs: list[str]) -> dict:
    """--opt key=value pairs -> pipeline constructor kwargs (ints, floats and booleans are cast)."""
    opts = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        opts[key] = value
    return opts


def run(pipeline_name: str, items_by_doc: dict[str, list[dict]], retrieval_only: bool,
        opts: dict | None = None) -> dict:
    from pipelines import PIPELINES
    from scoring import evidence_rank, score_answer

    opts = opts or {}
    pipeline = PIPELINES[pipeline_name](**opts)
    rows: list[dict] = []
    workroot = Path(tempfile.mkdtemp(prefix="docmind_eval_"))

    try:
        for doc_i, (doc, items) in enumerate(items_by_doc.items()):
            print(f"\n=== {doc}  ({len(items)} questions) ===")
            t0 = time.time()
            index = pipeline.build_index(resolve_doc(doc), workroot / f"doc_{doc_i}")
            print(f"    indexed in {time.time() - t0:.1f}s")

            for item in items:
                chunks = pipeline.retrieve(index, item["question"])
                chunk_texts = [c["text"] for c in chunks]
                row = {
                    "id": item["id"], "doc": doc, "type": item["type"],
                    "answerable": item["answerable"], "question": item["question"],
                    "evidence_rank": evidence_rank(item.get("evidence", []), chunk_texts)
                                     if item["answerable"] else None,
                    "has_evidence": bool(item.get("evidence")),
                    "retrieved": [{k: c.get(k) for k in ("chunk_id", "embed_score", "similarity_score")}
                                  for c in chunks],
                }
                if not retrieval_only:
                    t1 = time.time()
                    gen = pipeline.answer(index, item["question"], chunks)
                    row["latency_s"] = round(time.time() - t1, 2)
                    row["answer"] = gen["answer"]
                    row["warning"] = gen.get("warning")
                    row.update(score_answer(item, gen["answer"]))
                    mark = "OK " if row["correct"] else "XX "
                    print(f"    {mark}{item['id']:8s} {gen['answer'][:90]!r}")
                rows.append(row)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)  # Chroma may hold file locks on Windows

    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline_name,
        "opts": opts,
        "retrieval_only": retrieval_only,
        "llm_backend": None if retrieval_only else os.getenv("LLM_BACKEND", "ollama"),
        "llm_model": None if retrieval_only else (
            os.getenv("GROQ_MODEL") if os.getenv("LLM_BACKEND", "ollama").lower() == "groq"
            else os.getenv("OLLAMA_MODEL", "llama3.1")
        ),
        "embedding_backend": os.getenv("EMBEDDING_BACKEND"),
        "metrics": aggregate(rows, retrieval_only),
        "rows": rows,
    }


def _retrieval_metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["answerable"] and r["has_evidence"]]
    if not scored:
        return {}
    out = {f"evidence_hit@{k}": round(sum(1 for r in scored if r["evidence_rank"] and r["evidence_rank"] <= k) / len(scored), 4)
           for k in HIT_KS}
    out["evidence_mrr"] = round(sum(1 / r["evidence_rank"] for r in scored if r["evidence_rank"]) / len(scored), 4)
    out["n"] = len(scored)
    return out


def _answer_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    return {
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "answerable_accuracy": round(sum(r["correct"] for r in ans) / len(ans), 4) if ans else None,
        "abstention_recall": round(sum(r["correct"] for r in unans) / len(unans), 4) if unans else None,
        "false_abstention_rate": round(sum(r["strict_refusal"] for r in ans) / len(ans), 4) if ans else None,
        "n": len(rows),
    }


def aggregate(rows: list[dict], retrieval_only: bool) -> dict:
    metrics = {"retrieval": _retrieval_metrics(rows)}
    by_doc = defaultdict(list)
    by_type = defaultdict(list)
    for r in rows:
        by_doc[r["doc"]].append(r)
        by_type[r["type"]].append(r)
    metrics["retrieval_by_doc"] = {d: _retrieval_metrics(rs) for d, rs in by_doc.items()}
    metrics["retrieval_by_type"] = {t: _retrieval_metrics(rs) for t, rs in by_type.items()}
    if not retrieval_only:
        metrics["answers"] = _answer_metrics(rows)
        metrics["answers_by_doc"] = {d: _answer_metrics(rs) for d, rs in by_doc.items()}
        metrics["answers_by_type"] = {t: _answer_metrics(rs) for t, rs in by_type.items()}
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "  -  " if x is None else f"{x * 100:5.1f}%"


def print_summary(report: dict) -> None:
    m = report["metrics"]
    print("\n" + "=" * 78)
    opts = " ".join(f"{k}={v}" for k, v in report.get("opts", {}).items())
    print(f"  DocMind gold-set evaluation  |  pipeline={report['pipeline']} {opts}"
          + ("  |  retrieval-only" if report["retrieval_only"] else
             f"  |  llm={report['llm_backend']}:{report['llm_model']}"))
    print("=" * 78)

    r = m["retrieval"]
    print(f"\n  RETRIEVAL (answerable questions with evidence, n={r.get('n', 0)})")
    print(f"    hit@1 {_pct(r.get('evidence_hit@1'))}   hit@3 {_pct(r.get('evidence_hit@3'))}   "
          f"hit@5 {_pct(r.get('evidence_hit@5'))}   hit@10 {_pct(r.get('evidence_hit@10'))}   MRR {r.get('evidence_mrr', 0):.3f}")
    print(f"\n    {'by document':<52} {'hit@1':>7} {'hit@5':>7}")
    for d, rm in m["retrieval_by_doc"].items():
        if rm:
            print(f"    {d[:52]:<52} {_pct(rm['evidence_hit@1']):>7} {_pct(rm['evidence_hit@5']):>7}")

    if "answers" in m:
        a = m["answers"]
        print(f"\n  ANSWERS (n={a['n']})")
        print(f"    ACCURACY {_pct(a['accuracy'])}   answerable {_pct(a['answerable_accuracy'])}   "
              f"abstention recall {_pct(a['abstention_recall'])}   "
              f"false abstentions {_pct(a['false_abstention_rate'])}")
        print(f"\n    {'by type':<24} {'n':>4} {'accuracy':>9}")
        for t, am in sorted(m["answers_by_type"].items()):
            print(f"    {t:<24} {am['n']:>4} {_pct(am['accuracy']):>9}")
        print(f"\n    {'by document':<52} {'n':>4} {'accuracy':>9}")
        for d, am in m["answers_by_doc"].items():
            print(f"    {d[:52]:<52} {am['n']:>4} {_pct(am['accuracy']):>9}")
    print()


def save(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_retrieval" if report["retrieval_only"] else ""
    base = RESULTS_DIR / f"{stamp}_{report['pipeline']}{suffix}"

    with base.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    cols = ["id", "doc", "type", "answerable", "evidence_rank", "correct", "partial",
            "strict_refusal", "abstained", "latency_s", "question", "answer", "warning"]
    with base.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(report["rows"])

    summary = {k: report[k] for k in ("run_timestamp", "pipeline", "opts", "retrieval_only", "llm_backend", "llm_model")}
    summary["retrieval"] = report["metrics"]["retrieval"]
    summary["answers"] = report["metrics"].get("answers")
    with (RESULTS_DIR / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return base


def rescore(paths: list[str]) -> None:
    """
    Re-applies scoring.score_answer to the answers stored in saved runs, using
    the current gold set.  Gold fixes (e.g. a missing acceptable phrasing) then
    apply to every run equally, without regenerating any answers.
    """
    from scoring import score_answer

    gold = {item["id"]: item for item in load_gold(GOLD_PATH)}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report["retrieval_only"]:
            print(f"{Path(path).name}: retrieval-only run, nothing to re-score")
            continue
        changed = 0
        for row in report["rows"]:
            new = score_answer(gold[row["id"]], row["answer"])
            changed += new["correct"] != row["correct"]
            row.update(new)
        report["metrics"] = aggregate(report["rows"], retrieval_only=False)
        print(f"\n### {Path(path).name}  ({changed} verdict(s) changed by the current gold set)")
        print_summary(report)


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="DocMind gold-set evaluation")
    parser.add_argument("--pipeline", default="baseline")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--validate", action="store_true", help="only check gold evidence against extracted text")
    parser.add_argument("--docs", help="only evaluate documents whose filename contains this substring")
    parser.add_argument("--limit", type=int, help="max questions per document (smoke tests)")
    parser.add_argument("--embedding-backend", default="bge_fastembed",
                        help="EMBEDDING_BACKEND for this run (default matches the deployed image)")
    parser.add_argument("--llm-backend", choices=["ollama", "groq"],
                        help="override LLM_BACKEND from .env for this run")
    parser.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                        help="pipeline constructor option, repeatable (e.g. --opt top_k=8)")
    parser.add_argument("--rescore", nargs="+", metavar="RESULTS_JSON",
                        help="re-score saved runs against the current gold set (after a gold fix), no LLM calls")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore)
        return

    # Must be set before vector_store / generation are imported (both read env
    # at import time); load_dotenv() never overrides values already set.
    os.environ["EMBEDDING_BACKEND"] = args.embedding_backend
    if args.llm_backend:
        os.environ["LLM_BACKEND"] = args.llm_backend
    from dotenv import load_dotenv
    load_dotenv(REPO_DIR / ".env")

    items = load_gold(GOLD_PATH)
    if args.validate:
        sys.exit(1 if validate(items) else 0)

    report = run(args.pipeline, group_by_doc(items, args.docs, args.limit), args.retrieval_only,
                 parse_opts(args.opt))
    print_summary(report)
    base = save(report)
    print(f"Saved {base.with_suffix('.json').name} and .csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
