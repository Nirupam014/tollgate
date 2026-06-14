"""Trivial baselines for the unbounded-loop detection task.

A detector is only worth publishing if it beats the dumb things you could do in
one line. These baselines give Tollgate something to clear:

  * always_block  - call everything critical (perfect recall, awful precision)
  * always_pass   - call nothing critical (perfect precision on negatives, zero recall)
  * grep_while_true - flag any file whose text contains ``while True`` (the naive
                      heuristic a reviewer reaches for first)

Each baseline maps a corpus case to a boolean "predicts an unbounded loop".
"""
from __future__ import annotations

import os
from typing import Callable, Dict

Baseline = Callable[[str], bool]


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def always_block(path: str) -> bool:
    return True


def always_pass(path: str) -> bool:
    return False


def grep_while_true(path: str) -> bool:
    text = _read(path)
    return "while True" in text or "while 1" in text


BASELINES: Dict[str, Baseline] = {
    "always_block": always_block,
    "always_pass": always_pass,
    "grep_while_true": grep_while_true,
}
