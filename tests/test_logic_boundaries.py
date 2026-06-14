"""Decision-boundary unit tests for the scorer and detectors.

These were written to close the gaps that ``validation/mutation.py`` exposed: the
existing corpus exercises the *loop* path heavily but barely pinned the numeric
thresholds and boolean connectives in scoring + the secondary detectors. Each
test below asserts an exact value or an on/off boundary, so a flipped comparison,
swapped and/or, or an off-by-one constant changes a result a test checks — i.e.
the mutant dies.

Fast by construction: scorer tests touch no Monte Carlo, detector tests build a
tiny IR and call one detector directly.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.catalog import ModelCatalog                 # noqa: E402
from tollgate.detectors import DetectorEngine             # noqa: E402
from tollgate.findings import Finding                     # noqa: E402
from tollgate.graphutil import expected_executions, find_cycles  # noqa: E402
from tollgate.ir import Guard, IREdge, IRNode, Retry, Workflow   # noqa: E402
from tollgate.prediction import PredictionEngine          # noqa: E402
from tollgate.scoring import RiskScorer, _band            # noqa: E402

CAT = ModelCatalog.load()


def _f(category: str, severity: str) -> Finding:
    return Finding(finding_id=f"{category}_{severity}", category=category,
                   severity=severity, node_id="n", source_path="x", message="m", evidence={})


# ---------------------------------------------------------------- scoring ----
class TestScoringBoundaries(unittest.TestCase):
    def test_band_exact_boundaries(self):
        # Pins BANDS thresholds (25/50/75) AND the '>=' in _band: at the cutoff
        # the higher band must win; one below, the lower band.
        self.assertEqual(_band(0), "low")
        self.assertEqual(_band(24), "low")
        self.assertEqual(_band(25), "medium")
        self.assertEqual(_band(49), "medium")
        self.assertEqual(_band(50), "high")
        self.assertEqual(_band(74), "high")
        self.assertEqual(_band(75), "critical")
        self.assertEqual(_band(100), "critical")

    def test_severity_weights_exact(self):
        # One finding per category so saturation (0.5**i) never engages: the
        # score equals the raw SEVERITY_WEIGHT, pinning 40/24/10/3.
        s = RiskScorer()
        self.assertEqual(s.score([_f("a", "high")]).score, 24)
        self.assertEqual(s.score([_f("a", "medium")]).score, 10)
        self.assertEqual(s.score([_f("a", "low")]).score, 3)
        # A lone critical contributes 40 (pins the weight) even though the gate
        # will block on criticality regardless.
        self.assertEqual(s.score([_f("a", "critical")]).score, 40)

    def test_saturation_halves_each_repeat(self):
        # Two 'high' in the SAME category: 24 + 24*0.5 = 36. Pins the 0.5 factor
        # and that diminishing returns apply within a category.
        s = RiskScorer()
        self.assertEqual(s.score([_f("a", "high"), _f("a", "high")]).score, 36)

    def test_gate_thresholds(self):
        s = RiskScorer(block_at_score=75, warn_at_score=50)
        # 24 -> pass (below warn).
        self.assertEqual(s.score([_f("a", "high")]).gate_decision, "pass")
        # Three categories of 'high' = 72 -> warn (>=50, <75), no critical.
        three = [_f("a", "high"), _f("b", "high"), _f("c", "high")]
        r = s.score(three)
        self.assertEqual(r.score, 72)
        self.assertEqual(r.gate_decision, "warn")
        # Four categories of 'high' = 96 -> block via score path (still no critical).
        four = three + [_f("d", "high")]
        r4 = s.score(four)
        self.assertEqual(r4.score, 96)
        self.assertEqual(r4.gate_decision, "block")

    def test_gate_block_at_exact_threshold(self):
        # Score EXACTLY at block_at_score (75) with no critical/policy must block.
        # Pins the '>=' in the gate's score clause: a '>' mutant would warn here.
        # 24*3 (three 'high' categories) + 3 (one 'low') = 75.
        s = RiskScorer(block_at_score=75, warn_at_score=50)
        fs = [_f("a", "high"), _f("b", "high"), _f("c", "high"), _f("d", "low")]
        r = s.score(fs)
        self.assertEqual(r.score, 75)
        self.assertEqual(r.gate_decision, "block")

    def test_gate_warn_at_exact_threshold(self):
        # Score EXACTLY at warn_at_score (50), below block, no critical -> warn.
        # Pins the '>=' in the warn clause: a '>' mutant would pass here.
        # 10*5 (five 'medium' categories) = 50.
        s = RiskScorer(block_at_score=75, warn_at_score=50)
        fs = [_f(c, "medium") for c in ("a", "b", "c", "d", "e")]
        r = s.score(fs)
        self.assertEqual(r.score, 50)
        self.assertEqual(r.gate_decision, "warn")

    def test_critical_forces_block_below_threshold(self):
        # Score 40 (< 75) but a critical present -> block. Pins the has_critical
        # OR branch in the gate decision.
        s = RiskScorer(block_at_score=75, warn_at_score=50)
        r = s.score([_f("a", "critical")])
        self.assertEqual(r.score, 40)
        self.assertEqual(r.gate_decision, "block")

    def test_score_clamped_to_100(self):
        s = RiskScorer()
        r = s.score([_f("a", "critical")], policy_violations=[_f("p", "critical")])
        self.assertEqual(r.score, 100)  # 40 + 100 -> clamp


# -------------------------------------------------------------- detectors ----
def _eng() -> DetectorEngine:
    return DetectorEngine(CAT, default_model="gpt-4o")


def _wf(node: IRNode) -> Workflow:
    return Workflow("w", "dsl", [node], [], entry=node.node_id)


def _empty_pred(wf: Workflow) -> "object":
    return PredictionEngine(CAT).predict(wf)


class TestRetryStorm(unittest.TestCase):
    def _retry_findings(self, max_attempts, backoff):
        n = IRNode("n", "llm_call", "gpt-4o", retry=Retry(max_attempts=max_attempts, backoff=backoff))
        wf = _wf(n)
        return [f for f in _eng().run(wf, _empty_pred(wf)) if f.category == "retry_storm"]

    def test_cap_of_5_with_backoff_is_clean(self):
        # max_attempts == 5 is the boundary: NOT > 5, so bounded; with backoff,
        # no retry-storm finding. Pins the '> 5' (a '>=' mutant would fire here).
        self.assertEqual(self._retry_findings(5, "exponential"), [])

    def test_cap_above_5_is_unbounded_high(self):
        fs = self._retry_findings(6, "exponential")
        self.assertEqual(len(fs), 1)
        # unbounded (cap>5) but has backoff -> medium (pins the and/or in severity).
        self.assertEqual(fs[0].severity, "medium")

    def test_no_backoff_bounded_is_medium(self):
        fs = self._retry_findings(3, None)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "medium")

    def test_unbounded_and_no_backoff_is_high(self):
        fs = self._retry_findings(None, None)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "high")


class TestModelMismatch(unittest.TestCase):
    def _mismatch(self, model, task_class):
        n = IRNode("n", "llm_call", model, task_class=task_class)
        wf = _wf(n)
        return [f for f in _eng().run(wf, _empty_pred(wf)) if f.category == "model_mismatch"]

    def test_tier4_on_cheap_task_flags(self):
        # gpt-4o is tier 4 == min_tier; '>=' must fire on a 'classification' task.
        self.assertEqual(len(self._mismatch("gpt-4o", "classification")), 1)

    def test_tier2_on_cheap_task_clean(self):
        # gpt-4o-mini is tier 2 (< 4): no finding. Pins the threshold direction.
        self.assertEqual(self._mismatch("gpt-4o-mini", "classification"), [])

    def test_tier4_on_reasoning_task_clean(self):
        # Expensive task class is not in the cheap set: no mismatch.
        self.assertEqual(self._mismatch("gpt-4o", "reasoning"), [])


class TestFanout(unittest.TestCase):
    def _fanout(self, factor):
        n = IRNode("n", kind="map", fanout_factor=factor)
        wf = _wf(n)
        return [f for f in _eng().run(wf, _empty_pred(wf)) if f.category == "fanout"]

    def test_uncapped_is_high(self):
        fs = self._fanout(None)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "high")

    def test_at_warn_factor_is_medium(self):
        # factor == 25 == warn factor: '>=' must fire (a '>' mutant would not).
        fs = self._fanout(25)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "medium")

    def test_below_warn_factor_clean(self):
        self.assertEqual(self._fanout(24), [])


class TestPromptBloat(unittest.TestCase):
    def _bloat(self, static_tokens):
        n = IRNode("n", "llm_call", "gpt-4o", static_input_tokens=static_tokens)
        wf = _wf(n)
        return [f for f in _eng().run(wf, _empty_pred(wf)) if f.category == "prompt_bloat"]

    def test_at_exact_threshold_flags(self):
        # static == 8000 == prompt_bloat_tokens: the '>=' must fire (a '>' mutant
        # would not). Pins the threshold direction.
        fs = self._bloat(8000)
        self.assertEqual(len(fs), 1)

    def test_below_threshold_clean(self):
        # Nonzero but under the threshold: no finding. Pins the 'and' connective
        # (`static_input_tokens and static >= T`): an 'or' mutant fires here on the
        # truthy left operand alone.
        self.assertEqual(self._bloat(100), [])


class TestContextExplosion(unittest.TestCase):
    def _ctx(self, in_loop: bool, static=5000):
        # A history-appending node; in_loop adds a self-loop so projection over
        # the horizon engages.
        n = IRNode("n", "llm_call", "gpt-4o", appends_history=True, static_input_tokens=static)
        edges = [IREdge("n", "n", "loop")] if in_loop else []
        wf = Workflow("w", "dsl", [n], edges, entry="n")
        return [f for f in _eng().run(wf, _empty_pred(wf)) if f.category == "context_explosion"]

    def test_history_in_loop_flags(self):
        self.assertTrue(self._ctx(in_loop=True))

    def test_history_without_loop_is_clean(self):
        # Single pass: projection multiplier is 1, stays well under the fraction.
        self.assertEqual(self._ctx(in_loop=False), [])

    def test_high_fraction_without_loop_is_clean(self):
        # Even when a SINGLE-pass projection already exceeds the fraction-of-limit
        # (80k of a 128k context = 62%), context-explosion must NOT fire outside a
        # loop: the growth pattern requires accumulation. Pins the 'in_loop and ...'
        # connective — an 'or' mutant would flag this on the fraction alone.
        self.assertEqual(self._ctx(in_loop=False, static=80000), [])

    def test_fraction_exactly_at_limit_is_critical(self):
        # per_iter=6400 over 20 iters = 128000 tokens = exactly the gpt-4o limit,
        # so frac == 1.0. The boundary is inclusive: critical, not high. Pins the
        # `frac >= 1.0` severity cutoff — a `> 1.0` mutant downgrades this to high
        # and would silently turn a context-overflow into a non-blocking finding.
        fs = self._ctx(in_loop=True, static=6400)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].evidence["fraction_of_limit"], 1.0)
        self.assertEqual(fs[0].severity, "critical")

    def test_fraction_just_below_limit_is_high(self):
        # per_iter=6000 over 20 = 120000 of 128000 => frac 0.9375 (< 1.0): high,
        # not critical. Brackets the boundary from below so the cutoff is pinned
        # on both sides.
        fs = self._ctx(in_loop=True, static=6000)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "high")


class TestRecursiveLoopTelemetry(unittest.TestCase):
    def _loop(self, observed):
        # Bounded self-loop (guard max_depth=3); telemetry supplies an observed
        # production depth for the node.
        n = IRNode("n", "llm_call", "gpt-4o")
        e = IREdge("n", "n", "loop", guard=Guard(max_depth=3))
        wf = Workflow("w", "dsl", [n], [e], entry="n")
        tel = {"n": observed} if observed is not None else {}
        return [f for f in _eng().run(wf, _empty_pred(wf), telemetry_depths=tel)
                if f.category == "recursive_loop"]

    def test_observed_within_guard_is_medium(self):
        # observed (2) does NOT exceed the guard depth (3): the loop is guarded
        # and behaving -> medium. Pins the 'observed and any(depth and observed>depth)'
        # escalation: flipping either 'and' to 'or' wrongly escalates to 'high'.
        fs = self._loop(observed=2)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "medium")

    def test_observed_exceeding_guard_is_high(self):
        # Production depth (5) exceeds the declared guard (3): escalate to high.
        fs = self._loop(observed=5)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "high")

    def test_observed_equal_guard_is_medium(self):
        # observed == guard depth (3): the loop hit its cap but did NOT exceed it,
        # so it is behaving as declared -> medium. Pins the '>' in 'observed >
        # max_depth' (a '>=' mutant would wrongly escalate to high at equality).
        fs = self._loop(observed=3)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "medium")

    def test_no_telemetry_is_medium(self):
        # Guarded loop, no telemetry: medium (verify-the-bound advisory).
        fs = self._loop(observed=None)
        self.assertEqual(len(fs), 1)
        self.assertEqual(fs[0].severity, "medium")


class TestTarjanStructure(unittest.TestCase):
    """Pins Tarjan's SCC bookkeeping (structural cycle detection)."""

    def test_finalized_node_marked_offstack(self):
        # Fan-in: A->B and C->B, no cycle. After B's SCC is finalized and popped,
        # B must be marked OFF the stack. If it is left ON (the on_stack reset
        # mutated), a later root (C) with a cross-edge to the finalized B has its
        # lowlink pulled below its own index, so C is never emitted as an SCC root
        # and drops out of the component set -> expected_executions can't cover it.
        # Asserting full node coverage kills that mutant.
        a = IRNode("A", "llm_call", "gpt-4o")
        b = IRNode("B", "llm_call", "gpt-4o")
        c = IRNode("C", "llm_call", "gpt-4o")
        wf = Workflow("w", "dsl", [a, b, c],
                      [IREdge("A", "B"), IREdge("C", "B")], entry="A")
        ex = expected_executions(wf)
        self.assertEqual(set(ex), {"A", "B", "C"})
        # And there is genuinely no cycle here.
        self.assertEqual(find_cycles(wf), [])


if __name__ == "__main__":
    unittest.main()
