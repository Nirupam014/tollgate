"""Metamorphic tests for the analyzer.

Metamorphic testing checks *relations between outputs* that must hold even when
you don't know the absolute ground truth. They catch regressions that
example-based tests miss, because the relation constrains behavior across an
entire family of inputs.

Relations asserted here:

  R1  Adding a ``break`` to an unbounded ``while True`` must turn a CRITICAL loop
      into a non-critical one (it becomes bounded).
  R2  A semantically-inert edit (adding comments / blank lines / renaming a local)
      must NOT change the gate decision or the set of finding categories.
  R3  Tightening a loop guard must never *increase* risk score.
  R4  Analyzing the same input twice must produce identical output (determinism).
  R5  Duplicating a straight-line LLM call must not invent a recursive loop.

Run: python -m unittest validation.metamorphic   (or via run_all.sh)
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.config import Config                  # noqa: E402
from tollgate.ir import Guard, IREdge, IRNode, Workflow  # noqa: E402
from tollgate.parsers import parse_file             # noqa: E402
from tollgate.pipeline import analyze_workflow      # noqa: E402

CFG = Config(trials=400)


def _write(body: str, suffix: str = ".py") -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "case" + suffix)
    with open(p, "w") as fh:
        fh.write(textwrap.dedent(body))
    return p


def _analyze_py(body: str):
    return analyze_workflow(parse_file(_write(body)), cfg=CFG)


def _worst_loop(res) -> str:
    rank = {"medium": 1, "high": 2, "critical": 3}
    worst, name = 0, "none"
    for f in res.findings:
        if f.category == "recursive_loop" and rank.get(f.severity, 0) > worst:
            worst, name = rank[f.severity], f.severity
    return name


UNBOUNDED = """
import openai
def llm(p):
    return openai.chat.completions.create(model='gpt-4o',
        messages=[{'role':'user','content':p}])
def main():
    q = ['x']
    while True:
        t = q.pop(0)
        r = llm(t)
        q.append(str(r))
"""


class TestMetamorphic(unittest.TestCase):
    def test_R1_break_removes_criticality(self):
        crit = _analyze_py(UNBOUNDED)
        self.assertEqual(_worst_loop(crit), "critical")
        self.assertEqual(crit.risk.gate_decision, "block")

        bounded = _analyze_py(UNBOUNDED.replace(
            "        q.append(str(r))",
            "        q.append(str(r))\n        if len(q) > 10:\n            break"))
        self.assertNotEqual(_worst_loop(bounded), "critical")
        self.assertNotEqual(bounded.risk.gate_decision, "block")

    def test_R2_inert_edit_is_stable(self):
        base = _analyze_py(UNBOUNDED)
        # Comments, blank lines, and a renamed *local* must not change findings.
        edited_src = (
            "# a leading comment\n\n" + UNBOUNDED
            .replace("    q = ['x']", "    queue = ['x']  # renamed local")
            .replace("        t = q.pop(0)", "        t = queue.pop(0)")
            .replace("        q.append(str(r))", "\n        queue.append(str(r))")
        )
        edited = _analyze_py(edited_src)
        self.assertEqual(base.risk.gate_decision, edited.risk.gate_decision)
        self.assertEqual({f.category for f in base.findings},
                         {f.category for f in edited.findings})

    def test_R3_tightening_guard_does_not_raise_risk(self):
        def wf(max_depth):
            nodes = [IRNode("a", "llm_call", "gpt-4o", appends_history=True),
                     IRNode("b", "llm_call", "gpt-4o")]
            edges = [IREdge("a", "b", "sequence"),
                     IREdge("b", "a", "loop", guard=Guard(max_depth=max_depth))]
            return Workflow("g", "dsl", nodes, edges, entry="a")

        loose = analyze_workflow(wf(20), cfg=CFG).risk.score
        tight = analyze_workflow(wf(3), cfg=CFG).risk.score
        self.assertLessEqual(tight, loose)

    def test_R4_determinism(self):
        # Same file analyzed twice must give byte-identical output (the Monte
        # Carlo is seeded). Use one path so source_path can't differ.
        path = _write(UNBOUNDED)
        a = analyze_workflow(parse_file(path), cfg=CFG).to_dict()
        b = analyze_workflow(parse_file(path), cfg=CFG).to_dict()
        self.assertEqual(a, b)

    def test_R5_duplicating_call_makes_no_loop(self):
        one = _analyze_py("""
        import openai
        def ask(p):
            return openai.chat.completions.create(model='gpt-4o-mini',
                messages=[{'role':'user','content':p}])
        def main():
            a = ask('one')
        """)
        two = _analyze_py("""
        import openai
        def ask(p):
            return openai.chat.completions.create(model='gpt-4o-mini',
                messages=[{'role':'user','content':p}])
        def main():
            a = ask('one')
            b = ask('two')
        """)
        self.assertEqual(_worst_loop(one), "none")
        self.assertEqual(_worst_loop(two), "none")
        self.assertEqual(one.risk.gate_decision, two.risk.gate_decision)


if __name__ == "__main__":
    unittest.main()
