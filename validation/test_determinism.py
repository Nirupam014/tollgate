"""Determinism guarantee.

Tollgate gates deploys. A gate that flips verdicts between two runs of the same
code is worse than useless — it erodes trust and makes failures unreproducible.
The whole pipeline is built to be deterministic: the Monte Carlo simulation is
seeded (``Config.seed``), and every other stage is pure. This test is the
executable proof of that property, and it runs in CI on every push.

Two levels:

  D1  Per-file: parse + analyze the SAME file twice -> byte-identical ``to_dict``.
  D2  Whole-corpus scan: ``analyze_path`` over the benchmark corpus twice ->
      identical serialized RunResult. This also exercises discovery ordering,
      which must be stable.

If either fails, some stage has leaked nondeterminism (unseeded RNG, set
iteration order in output, dict ordering, wall-clock, hash randomization, ...).
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tollgate.config import Config                 # noqa: E402
from tollgate.parsers import discover, parse_file  # noqa: E402
from tollgate.pipeline import analyze_path, analyze_workflow  # noqa: E402

CORPUS = os.path.join(ROOT, "validation", "corpus", "cases")
CFG = Config(trials=400)


def _canon(d) -> str:
    # sort_keys so dict ordering can't mask a real difference (and can't cause a
    # spurious one either): we compare *content*, deterministically serialized.
    return json.dumps(d, sort_keys=True)


class TestDeterminism(unittest.TestCase):
    def test_D1_per_file_identical(self):
        files = sorted(discover([CORPUS]))
        self.assertTrue(files, "corpus discovery returned nothing")
        for path in files:
            a = analyze_workflow(parse_file(path), cfg=CFG).to_dict()
            b = analyze_workflow(parse_file(path), cfg=CFG).to_dict()
            self.assertEqual(_canon(a), _canon(b), f"nondeterministic output for {path}")

    def test_D2_directory_scan_identical(self):
        a = analyze_path([CORPUS], cfg=CFG).to_dict()
        b = analyze_path([CORPUS], cfg=CFG).to_dict()
        self.assertEqual(_canon(a), _canon(b), "directory scan is nondeterministic")

    def test_D3_scan_order_stable(self):
        # Discovery order itself must be stable run to run.
        self.assertEqual(discover([CORPUS]), discover([CORPUS]))


if __name__ == "__main__":
    unittest.main()
