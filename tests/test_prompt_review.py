"""Tests for the rule-based prompt efficiency reviewer.

The reviewer is deterministic (no model call), so every assertion here pins an
exact, reproducible behavior: a wasteful pattern is detected, the example rewrite
is shorter, savings are non-negative, and the on-by-default / CLI-off toggle works
end to end through the pipeline.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.config import Config                        # noqa: E402
from tollgate.ir import IRNode, Workflow                  # noqa: E402
from tollgate.pipeline import analyze_workflow            # noqa: E402
from tollgate.prompt_review import review_text, review_workflow  # noqa: E402


def _wf(prompt, **node_kw):
    node = IRNode(node_id="n1", kind="llm_call", prompt_template=prompt, **node_kw)
    return Workflow(workflow_id="wf", source_kind="prompt", nodes=[node],
                    edges=[], entry="n1", source_path="p.txt")


class TestDetection(unittest.TestCase):
    def test_filler_and_politeness_detected_and_removed(self):
        rev = review_text("Please kindly summarize this text. Answer in 50 words.")
        self.assertIsNotNone(rev)
        codes = {i.code for i in rev.issues}
        self.assertIn("politeness", codes)
        self.assertNotIn("please", rev.rewritten.lower())
        self.assertNotIn("kindly", rev.rewritten.lower())

    def test_wordy_phrase_shortened(self):
        rev = review_text("Use the tool in order to fetch data. Reply in one sentence.")
        self.assertIsNotNone(rev)
        self.assertIn("in_order_to", {i.code for i in rev.issues})
        self.assertIn(" to ", " " + rev.rewritten + " ")
        self.assertNotIn("in order to", rev.rewritten.lower())

    def test_ai_self_reference_removed(self):
        rev = review_text("As an AI language model, classify the sentiment. One word.")
        self.assertIsNotNone(rev)
        self.assertNotIn("as an ai", rev.rewritten.lower())

    def test_duplicate_lines_collapsed(self):
        text = "Summarize the document.\nBe concise.\nSummarize the document.\n"
        rev = review_text(text)
        self.assertIsNotNone(rev)
        self.assertIn("duplicate_lines", {i.code for i in rev.issues})
        self.assertEqual(rev.rewritten.lower().count("summarize the document"), 1)

    def test_missing_output_cap_is_advisory(self):
        rev = review_text("Write an essay about the ocean.")
        self.assertIsNotNone(rev)
        issue = next(i for i in rev.issues if i.code == "no_output_cap")
        self.assertEqual(issue.kind, "advisory")

    def test_output_cap_present_not_flagged(self):
        rev = review_text("Summarize in no more than 100 words.")
        # No length-cap nag; this prompt is already bounded and otherwise clean.
        if rev is not None:
            self.assertNotIn("no_output_cap", {i.code for i in rev.issues})

    def test_explicit_max_output_tokens_suppresses_cap_advice(self):
        rev = review_text("Please write a poem.", max_output_tokens=200)
        self.assertIsNotNone(rev)  # still flags 'please'
        self.assertNotIn("no_output_cap", {i.code for i in rev.issues})


class TestRewriteProperties(unittest.TestCase):
    def test_rewrite_never_longer_in_tokens(self):
        rev = review_text("Please, as an AI model, in order to help, kindly summarize. "
                          "Answer in 20 words.")
        self.assertIsNotNone(rev)
        self.assertLessEqual(rev.rewritten_tokens, rev.original_tokens)
        self.assertGreaterEqual(rev.tokens_saved, 0)
        self.assertGreaterEqual(rev.savings_pct, 0.0)

    def test_deterministic(self):
        p = "Please make sure that you, as an AI model, summarize this in 30 words."
        a = review_text(p)
        b = review_text(p)
        self.assertEqual(a.rewritten, b.rewritten)
        self.assertEqual([i.code for i in a.issues], [i.code for i in b.issues])

    def test_clean_prompt_returns_none(self):
        # Direct imperative, bounded output, no filler -> nothing to recommend.
        self.assertIsNone(review_text("Classify sentiment as positive or negative. "
                                      "Reply with one word."))

    def test_empty_returns_none(self):
        self.assertIsNone(review_text(""))
        self.assertIsNone(review_text("   \n  "))

    def test_recommendation_is_deduped_string(self):
        rev = review_text("Please please summarize. 10 words.")
        # 'please' appears twice but the recommendation text is listed once.
        self.assertEqual(rev.recommendation.count("politeness fillers"), 1)


class TestWorkflowAndPipeline(unittest.TestCase):
    def test_review_workflow_skips_nodes_without_prompt(self):
        wf = Workflow(workflow_id="wf", source_kind="dsl",
                      nodes=[IRNode(node_id="a", kind="llm_call")],
                      edges=[], entry="a")
        self.assertEqual(review_workflow(wf), [])

    def test_on_by_default(self):
        cfg = Config()
        self.assertTrue(cfg.prompt_review)
        res = analyze_workflow(_wf("Please kindly summarize. 10 words."), cfg=cfg)
        self.assertTrue(res.prompt_reviews)
        self.assertIn("prompt_reviews", res.to_dict())

    def test_disabled_via_config(self):
        cfg = Config()
        cfg.prompt_review = False
        res = analyze_workflow(_wf("Please kindly summarize. 10 words."), cfg=cfg)
        self.assertEqual(res.prompt_reviews, [])
        self.assertEqual(res.to_dict()["prompt_reviews"], [])


if __name__ == "__main__":
    unittest.main()
