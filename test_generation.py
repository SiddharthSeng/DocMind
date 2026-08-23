"""
test_generation.py — Failure-mode test suite for generation.py
================================================================
Run with:  python test_generation.py

These tests do NOT require Ollama to be running.  They mock _call_ollama()
to inject controlled LLM responses and verify that generate_answer() handles
each failure mode correctly.
"""

import unittest
from unittest.mock import patch

# We import the module-level helpers directly so we can test them in isolation.
from generation import (
    _build_user_prompt,
    _compute_confidence,
    _extract_cited_sources,
    _verify_citations,
    _strip_sources_section,
    generate_answer,
    ABSTAIN_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(source: str, text: str, score: float, index: int = 0) -> dict:
    return {
        "source_file":    source,
        "text":           text,
        "chunk_index":    index,
        "similarity_score": score,
    }

GOOD_CHUNK   = _make_chunk("policy.pdf",   "Refunds are processed within 14 days.", 0.85)
MEDIUM_CHUNK = _make_chunk("faq.txt",      "Contact support for billing issues.",    0.55)
WEAK_CHUNK   = _make_chunk("notes.txt",    "Miscellaneous internal notes.",          0.15)
UNRELATED_CHUNK = _make_chunk("report.pdf", "Q3 sales grew by 12%.",                0.40)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestPromptBuilder(unittest.TestCase):
    def test_chunks_are_labeled_with_source(self):
        prompt = _build_user_prompt([GOOD_CHUNK], "What is the refund window?")
        self.assertIn("[CHUNK 1 | source: policy.pdf]", prompt)
        self.assertIn("Refunds are processed within 14 days.", prompt)

    def test_empty_chunks_produce_no_passages_message(self):
        prompt = _build_user_prompt([], "What is the refund window?")
        self.assertIn("No relevant passages were retrieved", prompt)

    def test_question_is_always_in_prompt(self):
        prompt = _build_user_prompt([GOOD_CHUNK], "My specific question?")
        self.assertIn("My specific question?", prompt)


class TestCitationExtraction(unittest.TestCase):
    def test_parses_simple_sources_block(self):
        response = "The refund window is 14 days.\n\nSources:\npolicy.pdf"
        cited = _extract_cited_sources(response)
        self.assertEqual(cited, ["policy.pdf"])

    def test_parses_multiple_sources(self):
        response = "Answer.\n\nSources:\npolicy.pdf\nfaq.txt"
        cited = _extract_cited_sources(response)
        self.assertEqual(cited, ["policy.pdf", "faq.txt"])

    def test_empty_when_no_sources_section(self):
        response = "The refund window is 14 days."
        cited = _extract_cited_sources(response)
        self.assertEqual(cited, [])

    def test_deduplicates_sources(self):
        response = "Answer.\n\nSources:\npolicy.pdf\npolicy.pdf"
        cited = _extract_cited_sources(response)
        self.assertEqual(cited, ["policy.pdf"])


class TestCitationVerification(unittest.TestCase):
    def test_real_citation_passes(self):
        verified, phantoms = _verify_citations(["policy.pdf"], [GOOD_CHUNK])
        self.assertEqual(verified, ["policy.pdf"])
        self.assertEqual(phantoms, [])

    def test_phantom_citation_is_caught(self):
        # LLM cites a file that was never in the retrieved chunks
        verified, phantoms = _verify_citations(["invented.pdf"], [GOOD_CHUNK])
        self.assertEqual(verified, [])
        self.assertEqual(phantoms, ["invented.pdf"])

    def test_mixed_citations(self):
        chunks = [GOOD_CHUNK, MEDIUM_CHUNK]
        verified, phantoms = _verify_citations(
            ["policy.pdf", "phantom.pdf", "faq.txt"], chunks
        )
        self.assertIn("policy.pdf", verified)
        self.assertIn("faq.txt", verified)
        self.assertIn("phantom.pdf", phantoms)


class TestConfidenceScoring(unittest.TestCase):
    def test_high_similarity_gives_high_confidence(self):
        chunks = [_make_chunk("a.pdf", "", 0.90), _make_chunk("b.pdf", "", 0.80)]
        conf = _compute_confidence(chunks)
        self.assertGreater(conf, 0.80)

    def test_low_similarity_gives_low_confidence(self):
        chunks = [_make_chunk("a.pdf", "", 0.25), _make_chunk("b.pdf", "", 0.22)]
        conf = _compute_confidence(chunks)
        self.assertLess(conf, 0.35)

    def test_single_chunk(self):
        conf = _compute_confidence([_make_chunk("a.pdf", "", 0.75)])
        self.assertAlmostEqual(conf, 0.75, places=2)

    def test_empty_chunks_returns_zero(self):
        self.assertEqual(_compute_confidence([]), 0.0)

    def test_confidence_never_exceeds_one(self):
        chunks = [_make_chunk("a.pdf", "", 1.0), _make_chunk("b.pdf", "", 1.0)]
        self.assertLessEqual(_compute_confidence(chunks), 1.0)


# ---------------------------------------------------------------------------
# Integration tests for generate_answer() — LLM is mocked
# ---------------------------------------------------------------------------

class TestGenerateAnswer(unittest.TestCase):

    # -----------------------------------------------------------------------
    # FM-1: Model ignores "cite sources" instruction
    # -----------------------------------------------------------------------
    def test_fm1_missing_sources_section_triggers_warning(self):
        """
        Failure mode 1: LLM answers correctly but omits the Sources: section.
        generate_answer() must detect this and include a warning.
        """
        mock_response = "Refunds take 14 days."   # No "Sources:" block at all

        with patch("generation._call_ollama", return_value=mock_response):
            result = generate_answer([GOOD_CHUNK], "What is the refund window?")

        self.assertIsNotNone(result["warning"])
        self.assertIn("Sources", result["warning"])
        self.assertEqual(result["sources_cited"], [])

    # -----------------------------------------------------------------------
    # FM-2: Model cites a file it never saw (phantom hallucination)
    # -----------------------------------------------------------------------
    def test_fm2_phantom_citation_is_stripped(self):
        """
        Failure mode 2: LLM invents a source file that wasn't in the context.
        generate_answer() must strip the phantom and flag it.
        """
        mock_response = (
            "The refund policy is 14 days.\n\n"
            "Sources:\npolicy.pdf\ninvented_file.pdf"
        )

        with patch("generation._call_ollama", return_value=mock_response):
            result = generate_answer([GOOD_CHUNK], "What is the refund window?")

        self.assertIn("policy.pdf",        result["sources_cited"])
        self.assertNotIn("invented_file.pdf", result["sources_cited"])
        self.assertIn("phantom",           result["warning"].lower())

    # -----------------------------------------------------------------------
    # FM-3: Model answers from training knowledge (no abstention)
    # -----------------------------------------------------------------------
    def test_fm3_similarity_gate_blocks_low_quality_retrieval(self):
        """
        Failure mode 3: Retrieved chunks are too weak to be useful.
        The similarity gate should fire BEFORE the LLM is called.
        """
        very_weak_chunks = [
            _make_chunk("random.pdf", "Some unrelated text.", score=0.10)
        ]

        with patch("generation._call_ollama") as mock_llm:
            result = generate_answer(very_weak_chunks, "What is the refund window?")
            mock_llm.assert_not_called()  # LLM should never have been called

        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("abstain", result["warning"].lower())

    # -----------------------------------------------------------------------
    # FM-4: Model answers correctly and abstains when it should
    # -----------------------------------------------------------------------
    def test_fm4_model_abstains_correctly(self):
        """
        Failure mode 4 (success case): Model correctly says it doesn't know.
        Confidence should be 0.0 and sources_cited should be empty.
        """
        mock_response = (
            "I don't have enough information in the provided documents "
            "to answer this question."
        )

        with patch("generation._call_ollama", return_value=mock_response):
            result = generate_answer(
                [UNRELATED_CHUNK], "What is the nuclear launch code?"
            )

        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["sources_cited"], [])

    # -----------------------------------------------------------------------
    # FM-5: Model leaks context from unrelated chunks
    # -----------------------------------------------------------------------
    def test_fm5_only_cited_real_sources_are_returned(self):
        """
        Failure mode 5: If only one chunk was actually used but the model
        doesn't cite the other, verified_sources should only list what was cited.
        """
        # Two chunks retrieved; model correctly cites only the relevant one
        mock_response = (
            "Refunds are processed within 14 days.\n\n"
            "Sources:\npolicy.pdf"
        )
        chunks = [GOOD_CHUNK, MEDIUM_CHUNK]

        with patch("generation._call_ollama", return_value=mock_response):
            result = generate_answer(chunks, "What is the refund window?")

        self.assertEqual(result["sources_cited"], ["policy.pdf"])
        self.assertNotIn("faq.txt", result["sources_cited"])

    # -----------------------------------------------------------------------
    # FM-6: Correct full-pipeline happy path
    # -----------------------------------------------------------------------
    def test_fm6_happy_path_returns_full_structure(self):
        """
        Happy path: good chunk, model answers and cites correctly.
        Result should have answer, sources, confidence > 0, no warning.
        """
        mock_response = (
            "The refund window is 14 days per the policy document.\n\n"
            "Sources:\npolicy.pdf"
        )

        with patch("generation._call_ollama", return_value=mock_response):
            result = generate_answer([GOOD_CHUNK], "What is the refund window?")

        self.assertIn("14 days",     result["answer"])
        self.assertEqual(["policy.pdf"], result["sources_cited"])
        self.assertGreater(result["confidence"], 0.0)
        self.assertIsNone(result["warning"])

    # -----------------------------------------------------------------------
    # FM-7: Low-quality chunks are filtered but high-quality ones survive
    # -----------------------------------------------------------------------
    def test_fm7_mixed_similarity_only_good_chunks_reach_llm(self):
        """
        Failure mode 7: When weak and strong chunks are mixed, only good ones
        should be passed to the LLM, and the warning should mention dropped chunks.
        """
        mock_response = "Refunds take 14 days.\n\nSources:\npolicy.pdf"
        chunks = [GOOD_CHUNK, WEAK_CHUNK]  # 0.85 passes, 0.15 is dropped

        with patch("generation._call_ollama", return_value=mock_response) as mock_llm:
            result = generate_answer(chunks, "What is the refund window?")
            # LLM SHOULD be called (at least one good chunk survived)
            mock_llm.assert_called_once()

        # Warning about dropped chunks expected
        self.assertIsNotNone(result["warning"])
        self.assertIn("dropped", result["warning"].lower())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
