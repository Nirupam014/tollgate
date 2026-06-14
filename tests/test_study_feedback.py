"""Tests for the field-study report generator and the self-feedback loop.

Run with: python -m unittest discover -s tests  (no third-party deps).

These pin the two correctness-sensitive guarantees of study/feedback.py:
  * recompute-from-raw course-correction converges and overrides a stale cache;
  * review flags fire (never auto-fix) for the things that would distort a
    published headline, and detector self-tuning is a hard policy guard.
And that study/report.py emits a self-contained page with every token
substituted and the shared dashboard theme inlined.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "study"))

# The field-study harness (study/) is internal tooling and is not shipped with
# the published package. These tests exercise it, so they only run where study/
# is present (the maintainer's full checkout); in the public repo they skip
# cleanly instead of failing to import.
try:
    import feedback as fb        # noqa: E402
    import report as rpt         # noqa: E402
    from tollgate.html_report import BASE_CSS  # noqa: E402
    _STUDY_AVAILABLE = True
except Exception:
    _STUDY_AVAILABLE = False


@unittest.skipUnless(_STUDY_AVAILABLE, "study/ harness not present (internal tooling)")
class _RequiresStudy(unittest.TestCase):
    """Base for tests that need the internal study/ harness."""


def _row(repo, status="ok", applicable=True, wf=1, gate="warn", findings=None):
    r = {"repo": repo, "url": f"https://github.com/{repo}.git", "status": status,
         "sha": "deadbeef", "applicable_markers": ["langgraph"] if applicable else [],
         "applicable": applicable, "workflow_count": wf, "gate": gate,
         "max_score": 50, "findings": findings or []}
    if status != "ok":
        for k in ("sha", "applicable_markers", "applicable", "workflow_count",
                  "gate", "max_score", "findings"):
            r.pop(k, None)
    return r


def _finding(cat="recursive_loop", sev="high", node="n0"):
    return {"file": "a/flow.py", "workflow_id": "wf0", "source_kind": "langgraph",
            "category": cat, "severity": sev, "node_id": node,
            "message": f"{cat} on {node}"}


def _clean_rows():
    # 3 in-scope+discovered, 1 in-scope declined, 1 out-of-scope no wf — no flags.
    return [
        _row("o/a", gate="block", findings=[_finding("recursive_loop", "critical")]),
        _row("o/b", gate="warn", findings=[_finding("prompt_bloat", "medium")]),
        _row("o/c", gate="pass", findings=[]),
        _row("o/d", wf=0, gate="none"),                       # honestly declined
        _row("o/e", applicable=False, wf=0, gate="none"),     # out of scope, no wf
    ]


class TestFeedbackLoop(_RequiresStudy):
    def test_clean_verdict_no_corrections_no_flags(self):
        out = fb.run_feedback(_clean_rows())
        self.assertEqual(out["verdict"], "clean")
        self.assertEqual(out["corrections"], [])
        self.assertEqual(out["review_required"], [])
        self.assertTrue(all(c["ok"] for c in out["checks"]))

    def test_invariants_all_pass_on_recomputed_summary(self):
        out = fb.run_feedback(_clean_rows())
        # The summary is recomputed from raw, so every invariant must hold.
        names = {c["name"] for c in out["checks"]}
        self.assertIn("coverage_partition", names)
        self.assertIn("findings_histograms_agree", names)
        self.assertTrue(all(c["ok"] for c in out["checks"]))

    def test_course_correction_overrides_stale_cache_and_converges(self):
        rows = _clean_rows()
        good = fb.summarize(rows)
        stale = dict(good)
        stale["findings_total"] = 999          # deliberately wrong
        stale["honestly_declined"] = good["honestly_declined"] + 7
        out = fb.run_feedback(rows, cached_summary=stale)
        self.assertEqual(out["verdict"], "auto_corrected")
        self.assertGreaterEqual(len(out["corrections"]), 2)
        # Published summary is the recomputed (correct) one, not the stale cache.
        self.assertEqual(out["summary"]["findings_total"], good["findings_total"])
        # Re-auditing the corrected summary yields no further drift (converged).
        again = fb.run_feedback(rows, cached_summary=out["summary"])
        self.assertEqual(again["corrections"], [])
        self.assertLessEqual(out["iterations"], fb._MAX_ITERS)

    def test_out_of_scope_hit_is_flagged_not_fixed(self):
        rows = _clean_rows()
        rows.append(_row("o/f", applicable=False, wf=1, gate="warn",
                         findings=[_finding()]))  # out-of-scope but produced a wf
        out = fb.run_feedback(rows)
        self.assertEqual(out["verdict"], "needs_review")
        self.assertTrue(any("out-of-scope" in f for f in out["review_required"]))
        self.assertEqual(out["corrections"], [])  # flagged, never auto-fixed

    def test_tiny_precision_sample_is_flagged(self):
        precision = {"overall": {"tp": 8, "fp": 1, "n": 9, "precision": 8 / 9,
                                 "ci95": [0.5, 0.99], "unsure": 0, "unlabeled": 0}}
        out = fb.run_feedback(_clean_rows(), precision=precision)
        self.assertEqual(out["verdict"], "needs_review")
        self.assertTrue(any("n=9" in f for f in out["review_required"]))

    def test_healthy_precision_passes(self):
        precision = {"overall": {"tp": 51, "fp": 6, "n": 57, "precision": 51 / 57,
                                 "ci95": [0.79, 0.95], "unsure": 0, "unlabeled": 0}}
        out = fb.run_feedback(_clean_rows(), precision=precision)
        self.assertEqual(out["verdict"], "clean")

    def test_clone_failure_bias_flagged(self):
        rows = _clean_rows()
        rows += [_row(f"o/x{i}", status="clone_failed") for i in range(4)]
        out = fb.run_feedback(rows)
        self.assertTrue(any("failed to clone" in f for f in out["review_required"]))

    def test_detector_self_tuning_is_a_policy_guard(self):
        out = fb.run_feedback(_clean_rows())
        joined = " ".join(out["policy_guards"]).lower()
        self.assertIn("not auto-tuned", joined)
        self.assertIn("precision sample", joined)

    def test_audit_catches_inconsistent_summary(self):
        # If aggregate ever emitted an inconsistent summary, the invariant fails.
        bad = {"repos_total": 10, "status_breakdown": {"ok": 3},  # 3 != 10
               "analyzed_ok": 3, "in_scope": 2, "discovered_workflows": 1,
               "honestly_declined": 1, "discovery_rate_in_scope": 0.5,
               "out_of_scope_but_found": 0, "gate_distribution": {"pass": 1},
               "findings_total": 0, "findings_by_category": {}, "findings_by_severity": {}}
        checks = fb.audit_summary(bad, {})
        failed = [c["name"] for c in checks if not c["ok"]]
        self.assertIn("status_totals_to_repos", failed)


class TestStudyReport(_RequiresStudy):
    def test_html_self_contained_and_themed(self):
        data = rpt.build_report_data(_clean_rows())
        html = rpt.to_html(data)
        for tok in ("__BASE_CSS__", "__STUDY_DATA__", "__FEEDBACK_BLOCK__"):
            self.assertNotIn(tok, html)
        self.assertIn("--accent:#2563eb", html)   # shared dashboard theme inlined
        self.assertIn("<canvas", html)

    def test_examples_anonymized_by_default(self):
        data = rpt.build_report_data(_clean_rows())
        self.assertTrue(data["anonymized"])
        for e in data["examples"]:
            self.assertTrue(e["repo"].startswith("repo-"))
            self.assertNotIn("/", e["repo"])  # owner/name never leaks

    def test_name_repos_unanonymizes(self):
        data = rpt.build_report_data(_clean_rows(), name_repos=True)
        self.assertFalse(data["anonymized"])
        self.assertTrue(any("/" in e["repo"] for e in data["examples"]))

    def test_representative_selection_is_deterministic(self):
        a = rpt.representative_examples(_clean_rows(), name_repos=False)
        b = rpt.representative_examples(_clean_rows(), name_repos=False)
        self.assertEqual(a, b)

    def test_feedback_panel_embedded_when_supplied(self):
        rows = _clean_rows()
        data = rpt.build_report_data(rows)
        out = fb.run_feedback(rows)
        html = rpt.to_html(data, feedback=out)
        self.assertIn("Self-review &amp; course-correction", html)
        # And absent (empty) when no feedback is passed.
        self.assertNotIn("Self-review &amp; course-correction", rpt.to_html(data))

    def test_precision_section_renders_data(self):
        precision = {"overall": {"tp": 51, "fp": 6, "n": 57, "precision": 51 / 57,
                                 "ci95": [0.79, 0.95], "unsure": 0, "unlabeled": 0},
                     "by_category": {}}
        data = rpt.build_report_data(_clean_rows(), precision=precision)
        html = rpt.to_html(data)
        m = re.search(r"const D = (\{.*?\});\nconst S", html, re.S)
        self.assertIsNotNone(m)
        D = json.loads(m.group(1))
        self.assertEqual(D["precision"]["overall"]["n"], 57)


class TestBlockedDetail(_RequiresStudy):
    def _rows(self):
        return [
            _row("o/a", gate="block", findings=[_finding("recursive_loop", "critical"),
                                                _finding("recursive_loop", "high")]),
            _row("o/b", gate="block", findings=[_finding("context_explosion", "high")]),
            _row("o/c", gate="warn", findings=[_finding("prompt_bloat", "medium")]),
            _row("o/d", gate="pass", findings=[]),
        ]

    def test_only_blocked_repos_included_sorted_by_score(self):
        rows = self._rows()
        rows[0]["max_score"] = 90
        rows[1]["max_score"] = 80
        items, _ = rpt.blocked_detail(rows, name_repos=False)
        self.assertEqual(len(items), 2)                 # warn/pass excluded
        self.assertEqual([i["score"] for i in items], [90, 80])  # worst first

    def test_reason_split_counts_sum_to_blocked_total(self):
        items, split = rpt.blocked_detail(self._rows(), name_repos=False)
        self.assertEqual(sum(s["count"] for s in split), len(items))

    def test_critical_reason_attribution(self):
        items, _ = rpt.blocked_detail(self._rows(), name_repos=False)
        a = next(i for i in items if i["n_findings"] == 2)
        self.assertEqual(a["reason"], "recursive_loop (critical)")
        b = next(i for i in items if i["n_findings"] == 1)
        self.assertEqual(b["reason"], "context_explosion (elevated score)")

    def test_blocked_anonymized_by_default(self):
        items, _ = rpt.blocked_detail(self._rows(), name_repos=False)
        self.assertTrue(all(i["repo"].startswith("repo-") for i in items))
        named, _ = rpt.blocked_detail(self._rows(), name_repos=True)
        self.assertTrue(any("/" in i["repo"] for i in named))

    def test_report_renders_blocked_section_and_donut_hook(self):
        data = rpt.build_report_data(self._rows())
        html = rpt.to_html(data)
        self.assertIn("Blocked repositories", html)
        self.assertIn("By reason of blocking", html)   # donut afterBody hook
        self.assertGreaterEqual(len(data["blocked"]), 1)


class TestDrilldownData(_RequiresStudy):
    def test_repo_list_covers_every_repo_and_orders_ok_first(self):
        rows = _clean_rows() + [_row("o/z", status="clone_failed")]
        rl = rpt.repo_list(rows, name_repos=False)
        self.assertEqual(len(rl), len(rows))                 # every repo present
        self.assertTrue(all(r["repo"].startswith("repo-") for r in rl))  # anonymized
        statuses = [r["status"] for r in rl]
        self.assertEqual(statuses[-1], "clone_failed")       # failures sink to bottom
        self.assertTrue(all(s == "ok" for s in statuses[:-1]))

    def test_repo_list_name_repos_discloses(self):
        rl = rpt.repo_list(_clean_rows(), name_repos=True)
        self.assertTrue(any("/" in r["repo"] for r in rl))

    def test_sev_detail_caps_at_15_and_is_deterministic(self):
        rows = [_row(f"o/r{i}", findings=[_finding("recursive_loop", "high", node=f"n{i}")])
                for i in range(20)]
        a = rpt.findings_by_severity_detail(rows, name_repos=False)
        b = rpt.findings_by_severity_detail(rows, name_repos=False)
        self.assertEqual(a, b)                               # deterministic
        self.assertLessEqual(len(a["high"]), 15)             # capped
        for it in a["high"]:
            self.assertEqual(set(it), {"repo", "category", "message"})
            self.assertTrue(it["repo"].startswith("repo-"))

    def test_sev_detail_groups_by_severity(self):
        rows = [_row("o/a", findings=[_finding("recursive_loop", "critical"),
                                      _finding("prompt_bloat", "medium")])]
        d = rpt.findings_by_severity_detail(rows, name_repos=False)
        self.assertEqual(set(d), {"critical", "medium"})

    def test_report_data_carries_repos_and_sev_detail(self):
        data = rpt.build_report_data(_clean_rows())
        self.assertEqual(len(data["repos"]), len(_clean_rows()))
        self.assertIn("sev_detail", data)

    def test_humanize_shortens_uuids(self):
        u = "ed14af2c-2251-4cef-8a70-2337251c64c0"
        out = rpt._humanize(f"Cycle {u} has no guard.")
        self.assertNotIn(u, out)
        self.assertIn("ed14af2c", out)          # stub kept
        self.assertNotIn("-2251-", out)         # rest dropped

    def test_humanize_collapses_long_cycle_chain(self):
        ids = ["ed14af2c-2251-4cef-8a70-2337251c64c0",
               "81a851af-8465-4154-83f5-36aad0ad2ba5",
               "0963256f-3d16-47b2-808c-91b6b2f88aee",
               "74ae8dfb-4ec3-442d-a965-9fcc95fc8a48",
               "2105e9c1-0698-4a01-9d02-02d791364498"]
        chain = " -> ".join(ids + [ids[0]])
        out = rpt._humanize(f"Cycle {chain} has no bounded termination guard.")
        self.assertIn("→ … →", out)             # middle collapsed
        self.assertIn("(5 nodes)", out)         # node count surfaced
        self.assertLess(len(out), 120)          # readable length

    def test_humanize_is_noop_on_clean_text(self):
        msg = "Node 'planner' appends history inside a loop without truncation."
        self.assertEqual(rpt._humanize(msg), msg)

    def test_examples_messages_are_humanized(self):
        u = "ed14af2c-2251-4cef-8a70-2337251c64c0"
        rows = [_row("o/u", findings=[_finding("recursive_loop", "critical")])]
        rows[0]["findings"][0]["message"] = f"Cycle {u} -> {u} has no guard."
        ex = rpt.representative_examples(rows, name_repos=False)
        self.assertTrue(all(u not in e["message"] for e in ex))


class TestThemeSharing(_RequiresStudy):
    def test_base_css_is_the_single_source_of_truth(self):
        # Both surfaces inline the exact same theme constant.
        self.assertIn(":root{", BASE_CSS)
        self.assertIn("--accent:#2563eb", BASE_CSS)
        data = rpt.build_report_data(_clean_rows())
        self.assertIn(BASE_CSS, rpt.to_html(data))


if __name__ == "__main__":
    unittest.main()
