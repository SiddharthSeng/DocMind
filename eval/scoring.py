"""
scoring.py — Answer-correctness and retrieval scoring for the DocMind gold set
==============================================================================
The gold set (eval/gold/qa_gold.jsonl) labels each question in one of four
ways, and each is scored differently:

  answers     – list of acceptable answers; correct if ANY one is matched.
  all_of      – list of groups (each a list of alternatives); correct only if
                EVERY group is matched.  Used for "name the X and Y" questions.
  key_points  – list of groups; correct if >= KEY_POINT_THRESHOLD of the groups
                are matched.  Used for summaries and long lists.
  (none)      – answerable=false: correct only if the system abstains.

Matching is token RECALL, not exact match: a gold answer is matched when at
least ANSWER_RECALL_THRESHOLD of its tokens appear in the prediction.  This is
deliberate — generative answers are verbose ("The ensemble reached 97.7% test
accuracy"), so exact-match / F1 would punish correct answers for being polite.
The cost is that very short gold answers ("4", "Normal") can be matched by an
incidental token.  That bias is constant across pipelines, so comparisons
between pipelines stay fair; spot-check the per-question CSV for absolute
numbers.

Retrieval is scored against `evidence` strings (verbatim snippets from the
source document), NOT chunk ids, so the gold set stays valid when the chunker
changes.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

ANSWER_RECALL_THRESHOLD = 0.8
KEY_POINT_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_DASHES = re.compile(r"[‐-―−]")
_DIMENSION = re.compile(r"(\d)\s*[x×]\s*(\d)")
_THOUSANDS = re.compile(r"(\d),(\d{3})(?!\d)")
_DECIMAL = re.compile(r"\d+\.\d+")
_NON_TOKEN = re.compile(r"[^a-z0-9.]+")
_STRAY_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")


def normalize(text: str) -> str:
    """
    Canonicalises text so that formatting differences don't count as errors:
      "10⁻⁴" == "10^-4" == "10-4"      (NFKC + unicode dashes)
      "224×224" == "224 x 224"
      "2,768" == "2768"
      "0.9800" == "0.98", "95.0%" == "95"
      "C#" -> "csharp", "C++" -> "cplusplus" (otherwise both collapse to "c")
    Everything else that isn't [a-z0-9] or a decimal point becomes a space.
    """
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("’", "'").replace("c#", "csharp").replace("c++", "cplusplus")
    t = _DASHES.sub("-", t)
    t = _DIMENSION.sub(r"\1 x \2", t)
    while _THOUSANDS.search(t):
        t = _THOUSANDS.sub(r"\1\2", t)
    t = _DECIMAL.sub(lambda m: m.group().rstrip("0").rstrip("."), t)
    t = _NON_TOKEN.sub(" ", t)
    t = _STRAY_DOT.sub(" ", t)
    return " ".join(t.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# ---------------------------------------------------------------------------
# Abstention detection
# ---------------------------------------------------------------------------

# STRICT: the canonical refusal the system prompt asks for.  An answerable
# question that gets one of these is a false abstention, full stop.
_STRICT_REFUSAL = re.compile(
    r"(don't|do not|does not) have enough information"
    r"|not enough information"
    r"|(cannot|can't|unable to) answer",
    re.I,
)

# LENIENT: also accepts natural "the document doesn't say" phrasings.  Used to
# score unanswerable questions, where any clear "not in the document" counts.
_SOFT_REFUSAL = re.compile(
    r"(does|do|did) not (mention|specify|provide|state|report|include|contain|say|discuss|give)"
    r"|(doesn't|don't|didn't) (mention|specify|provide|state|report|include|contain|say|discuss|give)"
    r"|not (mentioned|specified|stated|provided|reported|included|given|available|discussed)"
    r"|no (information|mention|details)",
    re.I,
)


def is_strict_refusal(answer: str) -> bool:
    return bool(_STRICT_REFUSAL.search(answer.replace("’", "'")))


def is_abstention(answer: str) -> bool:
    a = answer.replace("’", "'")
    return bool(_STRICT_REFUSAL.search(a) or _SOFT_REFUSAL.search(a))


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def token_recall(gold: str, pred_counts: Counter) -> float:
    gold_counts = Counter(tokens(gold))
    total = sum(gold_counts.values())
    if total == 0:
        return 0.0
    return sum((gold_counts & pred_counts).values()) / total


def _group_matched(alternatives: list[str], pred_counts: Counter) -> bool:
    return any(token_recall(a, pred_counts) >= ANSWER_RECALL_THRESHOLD for a in alternatives)


def score_answer(item: dict, answer: str) -> dict:
    """
    Scores one prediction against one gold item.

    Returns:
        correct          (bool)  – the headline per-question verdict
        partial          (float) – fraction of answer groups matched (1.0/0.0
                                   for single-answer and unanswerable items)
        strict_refusal   (bool)  – canonical "not enough information" reply
        abstained        (bool)  – strict OR soft "document doesn't say" reply
    """
    strict = is_strict_refusal(answer)
    abstained = is_abstention(answer)

    if not item["answerable"]:
        return {"correct": abstained, "partial": float(abstained),
                "strict_refusal": strict, "abstained": abstained}

    pred_counts = Counter(tokens(answer))

    if "answers" in item:
        hits = [_group_matched(item["answers"], pred_counts)]
        needed = 1.0
    elif "all_of" in item:
        hits = [_group_matched(g, pred_counts) for g in item["all_of"]]
        needed = 1.0
    elif "key_points" in item:
        hits = [_group_matched(g, pred_counts) for g in item["key_points"]]
        needed = KEY_POINT_THRESHOLD
    else:
        raise ValueError(f"Gold item {item['id']} has no answers/all_of/key_points")

    partial = sum(hits) / len(hits)
    # A canonical refusal is never a correct answer, even if a gold token
    # happens to appear in the refusal text.
    correct = (not strict) and partial >= needed
    return {"correct": correct, "partial": round(partial, 4),
            "strict_refusal": strict, "abstained": abstained}


# ---------------------------------------------------------------------------
# Retrieval (evidence) matching
# ---------------------------------------------------------------------------

def evidence_rank(evidence: list[str], chunk_texts: list[str]) -> int | None:
    """
    Returns the smallest r such that the top-r chunks TOGETHER contain every
    evidence snippet (multi-evidence questions need all their snippets), or
    None if the retrieved list never covers them all.
    """
    if not evidence:
        return None
    needles = [f" {normalize(e)} " for e in evidence]
    remaining = set(range(len(needles)))
    for rank, text in enumerate(chunk_texts, start=1):
        hay = f" {normalize(text)} "
        remaining -= {i for i in remaining if needles[i] in hay}
        if not remaining:
            return rank
    return None


def evidence_in_text(evidence: str, full_text: str) -> bool:
    """Used by --validate: is this evidence snippet present in the extracted document?"""
    return f" {normalize(evidence)} " in f" {normalize(full_text)} "
