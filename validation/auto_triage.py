#!/usr/bin/env python3
"""Differential auto-triage of a precision sample — cut human labeling, honestly.

Hand-labeling every finding to get a precision number does not scale to a
500-repo study. This pre-labels the kwarg-presence finding categories
(`uncapped_output`, `missing_iteration_cap`) with an INDEPENDENT re-check against
the source, so a human only has to adjudicate what the oracle can't confirm.

Why "differential": the oracle does not call Tollgate's pipeline. It re-parses the
offending file and checks the one fact the finding hinges on — is the cap kwarg
actually present on the named call/constructor? If a finding says "no max_tokens"
but the source clearly sets max_tokens, that's a likely false positive; if the
cap is genuinely absent, the finding is likely a true positive. Because the
methods differ, agreement is meaningful evidence and disagreement is exactly what
a human should look at.

HONESTY LIMITS (read before trusting the output):
  * This is an *agreement* signal between two fallible static checks, NOT a
    validated precision number. Oracle-labeled rows are marked `labeled_by:
    oracle`; precision.py still computes precision from the `verdict` field, so
    you decide whether to accept the oracle calls or confirm them by hand.
  * It only triages categories whose truth is a kwarg's presence/absence. Every
    other category (recursive_loop, prompt_bloat, fanout, ...) is left ABSTAIN
    for a human — the oracle never guesses there.
  * It needs the source. Run the study/scan with --keep-clones and point
    --local-root at the clone workdir; rows whose file can't be found are abstained.

Workflow:
  python validation/sample.py --results study/results.jsonl --n 200 --out validation/sample.todo.jsonl
  python validation/auto_triage.py --sample validation/sample.todo.jsonl \
      --local-root /tmp/tollgate_study --out validation/sample.triaged.jsonl
  # hand-fill the remaining blank verdicts (the abstains/disagreements), then:
  python validation/precision.py --labeled validation/sample.triaged.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re

_OUTPUT_CAP_KWARGS = {
    "max_tokens", "max_output_tokens", "max_completion_tokens", "max_new_tokens",
    "max_tokens_to_sample", "maxOutputTokens",
}
_CALL_RE = re.compile(r"`([A-Za-z_][\w.]*)\s*\(")          # `ChatOpenAI(` -> ChatOpenAI
_KW_RE = re.compile(r"\(([a-zA-Z_][\w]*(?:\s*/\s*[a-zA-Z_][\w]*)*)\)")  # "(max_iter)" / "(a / b)"

# Categories the oracle is competent to triage (truth = a kwarg's presence).
_TRIAGEABLE = {"uncapped_output", "missing_iteration_cap"}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def resolve_source(local_root, repo, rel_file):
    """Best-effort locate the offending file among likely clone layouts."""
    if not rel_file:
        return None
    cands = [
        os.path.join(local_root, repo.replace("/", "__"), rel_file),
        os.path.join(local_root, repo.split("/")[-1], rel_file),
        os.path.join(local_root, repo, rel_file),
        os.path.join(local_root, rel_file),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _calls_named(tree, tail):
    """Yield ast.Call nodes whose dotted func ends with `tail`."""
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        parts = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr); f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
        dotted = ".".join(reversed(parts))
        if dotted == tail or dotted.endswith("." + tail) or (parts and parts[0] == tail):
            yield n


def _kwnames(call):
    return {kw.arg for kw in call.keywords if kw.arg}


def triage_row(row, local_root):
    """Return (oracle_verdict, note): tp | fp | abstain."""
    cat = row.get("category")
    if cat not in _TRIAGEABLE:
        return "abstain", f"category {cat} not kwarg-decidable"
    src = resolve_source(local_root, row.get("repo", ""), row.get("file"))
    if not src:
        return "abstain", "source file not found under --local-root"
    msg = row.get("message", "")
    m = _CALL_RE.search(msg)
    if not m:
        return "abstain", "could not parse call/ctor name from message"
    tail = m.group(1).split(".")[-1]

    if cat == "uncapped_output":
        caps = _OUTPUT_CAP_KWARGS
    else:  # missing_iteration_cap — the expected kwarg(s) are named in the message
        km = _KW_RE.search(msg)
        caps = set(re.split(r"\s*/\s*", km.group(1))) if km else set()
        if not caps:
            return "abstain", "could not parse expected cap kwarg from message"

    try:
        with open(src, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        return "abstain", "source unparseable"

    found = list(_calls_named(tree, tail))
    if not found:
        return "abstain", f"no `{tail}(` call found in source"
    # If EVERY such call sets a cap, the finding is a likely false positive.
    # If AT LEAST ONE lacks the cap, the finding is a likely true positive.
    any_uncapped = any(not (_kwnames(c) & caps) for c in found)
    if any_uncapped:
        return "tp", f"`{tail}(` present without {sorted(caps)} — finding corroborated"
    return "fp", f"every `{tail}(` sets one of {sorted(caps)} — likely false positive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="sheet from sample.py")
    ap.add_argument("--local-root", required=True,
                    help="dir holding the (kept) clones the study analyzed")
    ap.add_argument("--out", required=True, help="triaged sheet to write")
    ap.add_argument("--prefill-confident", action="store_true",
                    help="copy confident oracle verdicts into `verdict` "
                    "(marked labeled_by=oracle). Default: leave verdict blank.")
    args = ap.parse_args(argv)

    rows = load_jsonl(args.sample)
    meta = next((r for r in rows if r.get("_meta")), None)
    data = [r for r in rows if not r.get("_meta")]

    counts = {"tp": 0, "fp": 0, "abstain": 0}
    for r in data:
        ov, note = triage_row(r, args.local_root)
        r["oracle_verdict"] = ov
        r["oracle_note"] = note
        counts[ov] += 1
        if "verdict" not in r:
            r["verdict"] = ""
        if args.prefill_confident and ov in ("tp", "fp") and not r["verdict"]:
            r["verdict"] = ov
            r["labeled_by"] = "oracle"

    with open(args.out, "w", encoding="utf-8") as fh:
        if meta:
            meta["auto_triage"] = {"prefilled": bool(args.prefill_confident), **counts}
            fh.write(json.dumps(meta) + "\n")
        for r in data:
            fh.write(json.dumps(r) + "\n")

    total = len(data)
    confident = counts["tp"] + counts["fp"]
    print(f"triaged {total} findings: oracle confident on {confident} "
          f"(tp={counts['tp']} fp={counts['fp']}), abstained on {counts['abstain']}.")
    if args.prefill_confident:
        print(f"prefilled {confident} verdicts (labeled_by=oracle). A human still "
              f"needs to label the {counts['abstain']} abstains — and should spot-check "
              f"the oracle calls, since this is agreement, not validated precision.")
    else:
        print(f"verdict left blank everywhere; oracle_verdict shows the suggestion. "
              f"A human labels all {total} (or pass --prefill-confident to accept the "
              f"{confident} confident ones and only hand-label the {counts['abstain']}).")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
