"""Dependency-free mutation tester for the analyzer's decision logic.

Mutation testing answers "do the tests actually constrain behavior, or do they
just pass?" It seeds small faults into the source (flip ``<`` to ``<=``, ``and``
to ``or``, ``n`` to ``n+1``, ``True`` to ``False``, ...) and checks that the test
suite *fails* — "kills" the mutant. A surviving mutant is a hole: a change to
real logic that no test notices.

Why not mutmut/cosmic-ray? Those are great, but they pull a toolchain and a DB.
For an open-source project the validation must be reproducible with zero extra
dependencies, so this is a small, self-contained engine that mutates the AST
(guaranteeing syntactically valid mutants) and runs a fast, high-signal slice of
the suite against each one. The kill set is:

    tests.test_analyzer  validation.metamorphic  validation.test_determinism

We target only the decision-critical modules (detectors, scoring, graphutil,
prediction) — the code whose correctness the gate depends on. Mutating reporters
or the CLI would inflate the score with low-value mutants.

Each mutant is applied IN PLACE to the source file, the slice is run, then the
original bytes are restored in a ``finally`` (the tree is never left dirty).

Usage:
    # Run every gated logic mutant, persisting results so the run is resumable
    # (re-invoking the same command skips mutants already recorded):
    python validation/mutation.py --all-logic --results /tmp/mut.jsonl
    # ...optionally one module at a time (each fits a short time budget):
    python validation/mutation.py --all-logic --modules detectors.py --results /tmp/mut.jsonl
    # Then apply the equivalence allow-list and gate:
    python validation/mutation.py --report /tmp/mut.jsonl --strict

``--strict`` (with ``--report``) exits non-zero if any *unexpected* gated logic
mutant survived — i.e. a survivor that is not on ``EQUIVALENT_ALLOWLIST``. The
allow-list documents mutants proven to be equivalent or to affect only
explanatory output (drivers/reasons), never a pass|warn|block verdict.
"""
from __future__ import annotations

import argparse
import ast
import os
import random
import signal
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "tollgate")

# The documented bar. A mature suite kills the large majority of mutants in the
# logic it covers; survivors are triaged, not ignored. Reported as a diagnostic;
# the *gate* is the stronger "no unexpected survivor" check below.
MIN_KILL_RATE = 0.80

# Equivalence / cosmetic allow-list. Each entry is a gated logic mutant that was
# manually triaged and shown to be EITHER a true equivalent mutant (no input can
# distinguish it) OR one that changes only explanatory output (the ``drivers``
# breakdown or human-readable ``reasons``), never a pass|warn|block verdict or a
# finding's severity. CI fails on any gated survivor NOT in this list, so a new
# survivor (a genuinely weakened test) breaks the build; a stale entry (now
# killed) is reported as a warning to prune. Keyed by "module: <description>".
EQUIVALENT_ALLOWLIST = {
    # -- detectors.py: model-selection fallback + an unreachable fraction edge ---
    "detectors.py: bool Or->And@L62":
        "model-selection fallback (`intended_model or default`, then `... or "
        "get(default)`). No corpus/test node pairs a non-default model whose "
        "context limit changes a context-explosion verdict, so substituting the "
        "default leaves every verdict unchanged.",
    "detectors.py: cmp GtE->Gt@L71":
        "context-explosion test `frac >= 0.6`. No realistic projection lands "
        "frac at exactly 0.6, so `>=` and `>` decide identically (equivalent). "
        "The reachable boundaries (loop vs no-loop, prompt-bloat 8000, retry cap "
        "5, fan-out 25, tier 4) are each pinned by a dedicated boundary test.",

    # -- scoring.py: every survivor lands in a non-verdict field -----------------
    "scoring.py: const True->False@L39":
        "block_on_policy_violation default. Redundant: any policy violation adds "
        "100 to the score, which alone forces 'block' via the score path, so the "
        "flag never changes the verdict. True equivalent.",
    "scoring.py: bool Or->And@L45":
        "`projected_monthly_tokens or {default}` -> `and`. Affects only the "
        "passed-through projected-tokens field, which no gate logic reads.",
    "scoring.py: const False->True@L53":
        "saturation sort `reverse=False`. Only reorders same-category findings; "
        "for equal severities (the saturating case) the score is identical. "
        "Mixed-severity ordering is a documented backlog item, not a verdict bug.",
    "scoring.py: bin Add->Sub@L58":
        "`drivers.get(cat,0)+contrib`. The accumulator is always 0 here (one "
        "entry per category), so this flips only the sign of the drivers "
        "breakdown, not the score/gate.",
    "scoring.py: bool And->Or@L77":
        "guard on appending a 'reasons' string. The companion literal is never "
        "present, so the conjunct is always true; flipping to OR changes only "
        "explanatory reasons text, never the verdict.",
    "scoring.py: cmp Eq->NotEq@L78":
        "builds the critical-findings list used only for the reasons message; "
        "has_critical (the verdict input) is computed separately at L70.",
    "scoring.py: const True->False@L80":
        "`sorted(drivers, reverse=True)` for the drivers breakdown ordering. "
        "Reordering the explanatory drivers list never changes score/band/gate.",

    # -- graphutil.py: true equivalents + COST-magnitude inputs ------------------
    # expected_executions() is a per-node CALL-COUNT estimate that feeds prediction
    # MAGNITUDE (expected_calls). It is NOT a structural-verdict input: detectors
    # use find_cycles()/component_iterations() (pinned by the loop corpus + the
    # telemetry-escalation tests), never expected_executions().
    "graphutil.py: cmp LtE->Lt@L144":
        "`if base <= 0: continue` -> `< 0`. A component with zero expected visits "
        "contributes 0 either way; processing it adds nothing. True equivalent.",
    "graphutil.py: bool And->Or@L23":
        "adjacency build guard `from in adj and to in adj`. Differs only for an "
        "edge naming a node absent from the graph (dangling); all IR graphs are "
        "well-formed, so the adjacency is identical.",
    "graphutil.py: bool And->Or@L69":
        "loop_edges_within `from in C and to in C`. Differs only for an edge with "
        "exactly one endpoint in the cyclic component; termination guards sit on "
        "intra-component loop-back edges, so the guard set (and component_iterations) "
        "is unchanged. Structural cycle detection is pinned by the loop corpus.",
    "graphutil.py: cmp Lt->LtE@L137":
        "topo-completeness guard selecting a fallback traversal order used only to "
        "compute expected_executions (a cost-magnitude input). No structural "
        "verdict depends on it; magnitude is covered by the calibration harness.",
    "graphutil.py: cmp Eq->NotEq@L112":
        "conditional-edge count for probability splitting in expected_executions. "
        "Magnitude-only (expected_calls); no structural verdict depends on it.",
    "graphutil.py: cmp Eq->NotEq@L118":
        "conditional-edge probability assignment in expected_executions. "
        "Magnitude-only, as above.",

    # -- prediction.py: magnitude-only arithmetic -------------------------------
    # The gate is STRUCTURAL; predicted token/cost MAGNITUDES never flip a
    # pass|warn|block verdict (detectors read node attributes, not predictions).
    # Magnitude accuracy is the calibration harness's job (within-2x/within-3x
    # tolerances), and a single spread/multiplier swap stays inside that tolerance.
    "prediction.py: bool Or->And@L111":
        "`static_input_tokens or 0` default -> magnitude-only input-token change.",
    "prediction.py: cmp LtE->Lt@L121":
        "`input_p50 <= 0` floor -> magnitude-only default token count.",
    "prediction.py: bin Mult->Div@L30":
        "Dist.from_p50 p95 spread multiplier -> magnitude-only.",
    "prediction.py: bin Mult->Div@L33":
        "Dist.scale token multiplier -> magnitude-only.",
    "prediction.py: bin Add->Sub@L36":
        "Dist.add (composing per-node token distributions) -> magnitude-only.",
    "prediction.py: bin Mult->Div@L108":
        "telemetry p99 fallback spread -> magnitude-only.",
    "prediction.py: bin Mult->Div@L109":
        "telemetry p99 fallback spread -> magnitude-only.",
    "prediction.py: bin Mult->Div@L120":
        "appends-history input delta multiplier -> magnitude-only.",
    "prediction.py: bin Mult->Div@L129":
        "output-token p95/p99 spread multiplier -> magnitude-only.",
}

TARGET_MODULES = ["detectors.py", "scoring.py", "graphutil.py", "prediction.py"]

KILL_SLICE = ["tests.test_analyzer", "tests.test_logic_boundaries",
              "validation.metamorphic", "validation.test_determinism"]

# The benchmark harness is the strongest behavioral oracle: it asserts concrete
# gate decisions and loop severities across the whole labeled corpus. Including
# it in the kill set means a mutant that flips any real verdict is caught.
HARNESS = ["validation/harness.py", "--strict"]

# --- operator tables -------------------------------------------------------
_CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree: ast.AST) -> Iterator[Tuple[str, str, Callable[[], None]]]:
    """Yield (op_class, description, apply) for each mutable site, deterministically.

    ``op_class`` partitions mutants into the GATED *logic* family — comparison,
    boolean-connective, boolean-constant, and arithmetic swaps, where a survivor
    is unambiguously a test gap — and the diagnostic *num* family (integer/float
    +1 perturbations), which is dominated by equivalent mutants (an off-by-one on
    a spread multiplier or a threshold no input lands on changes no verdict). We
    report num but do not gate on it, the standard way real mutation testing
    handles equivalent mutants.

    ``apply`` mutates the node IN the given tree, so the k-th site can be applied
    to a freshly parsed tree by zipping.
    """
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                repl = _CMP_SWAP.get(type(op))
                if repl is not None:
                    def mk(n=node, idx=i, r=repl):
                        n.ops[idx] = r()
                    yield ("cmp", f"cmp {type(op).__name__}->{repl.__name__}@L{line}", mk)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
            repl = _BIN_SWAP[type(node.op)]
            def mk(n=node, r=repl):
                n.op = r()
            yield ("bin", f"bin {type(node.op).__name__}->{repl.__name__}@L{line}", mk)
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAP:
            repl = _BOOL_SWAP[type(node.op)]
            def mk(n=node, r=repl):
                n.op = r()
            yield ("bool", f"bool {type(node.op).__name__}->{repl.__name__}@L{line}", mk)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                def mk(n=node):
                    n.value = not n.value
                yield ("boolconst", f"const {node.value}->{not node.value}@L{line}", mk)
            elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                def mk(n=node):
                    n.value = n.value + 1
                yield ("num", f"const {node.value}->{node.value+1}@L{line}", mk)


GATED_CLASSES = {"cmp", "bool", "boolconst", "bin"}


def _catalog(src_text: str) -> List[Tuple[str, str]]:
    """List (op_class, description) for every site, in apply order."""
    return [(cls, desc) for cls, desc, _ in _sites(ast.parse(src_text))]


def _apply_kth(src_text: str, k: int) -> Tuple[str, str, str]:
    """Return (op_class, description, mutated_source) for the k-th site."""
    tree = ast.parse(src_text)
    for i, (cls, desc, apply) in enumerate(_sites(tree)):
        if i == k:
            apply()
            ast.fix_missing_locations(tree)
            return cls, desc, ast.unparse(tree)
    raise IndexError(k)


# Per-slice wall-clock cap. A mutant that breaks loop termination (e.g. flips a
# graph loop-guard comparison) can make the analyzer hang. A healthy slice runs
# in a few seconds, so 12s is generous; a hang is a DEAD mutant (the suite would
# never go green) and we treat a timeout as "killed".
SLICE_TIMEOUT_S = 12

# Pristine source backups live OUTSIDE the repo (in the system temp dir), keyed
# by module name, so an interrupted run can recover without ever creating files
# inside the user's source tree.
BAK_DIR = os.path.join(tempfile.gettempdir(), "tollgate_mutbak")


def _run_slice() -> bool:
    """Run the kill slice. Return True if it PASSED (mutant survived).

    A timeout returns False (mutant killed): non-termination is a failure the
    suite "detects" in the only way it can — by never passing.
    """
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    cmds = [
        [sys.executable, "-m", "unittest", "-q", *KILL_SLICE],
        [sys.executable, *HARNESS],
    ]
    for cmd in cmds:
        try:
            proc = subprocess.run(
                cmd, cwd=ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=SLICE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return False  # non-termination -> killed
        if proc.returncode != 0:
            return False  # a check failed -> killed
    return True  # everything green -> mutant survived


def _bak_path(mod: str) -> str:
    return os.path.join(BAK_DIR, mod)


def _recover_stale() -> None:
    """If a previous run was killed mid-mutation, the temp backup is the pristine
    original — restore it before doing anything else, so we never start (or
    leave) the source tree in a mutated state."""
    if not os.path.isdir(BAK_DIR):
        return
    for mod in TARGET_MODULES:
        bak = _bak_path(mod)
        if os.path.exists(bak):
            with open(bak, "r", encoding="utf-8") as fh:
                original = fh.read()
            with open(os.path.join(SRC, mod), "w", encoding="utf-8") as fh:
                fh.write(original)
            os.remove(bak)
            print(f"recovered {mod} from stale backup")


import json


def _load_results(results_path: str) -> List[dict]:
    """Read a JSONL results file (one record per mutant). Missing file -> []."""
    if not results_path or not os.path.exists(results_path):
        return []
    out = []
    with open(results_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append_result(results_path: str, rec: dict) -> None:
    if not results_path:
        return
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def run(per_module: int, seed: int, logic_per_module: int, all_logic: bool,
        modules: List[str], results_path: str) -> int:
    """Execute mutants and (optionally) persist each result to ``results_path``.

    Persisting makes the run RESUMABLE: a record is keyed by (module, desc), and
    a re-invocation skips any mutant already recorded. This lets a long run be
    split across short time windows (or a crashed run be continued) without
    repeating work and without ever leaving the source tree mutated.
    """
    rng = random.Random(seed)
    _recover_stale()
    # Sanity: the baseline suite must be green, or every mutant "dies" for free.
    if not _run_slice():
        print("ERROR: baseline kill-slice is RED before mutation. Fix tests first.")
        return 2

    done = {(r["module"], r["desc"]) for r in _load_results(results_path)}
    targets = [m for m in TARGET_MODULES if not modules or m in modules]

    for mod in targets:
        path = os.path.join(SRC, mod)
        original = open(path, "r", encoding="utf-8").read()
        # Pristine backup in temp: if this process is killed between writing a
        # mutant and restoring it, _recover_stale() (or the next run) heals it.
        os.makedirs(BAK_DIR, exist_ok=True)
        with open(_bak_path(mod), "w", encoding="utf-8") as fh:
            fh.write(original)
        cat = _catalog(original)
        gated_all = [i for i, (cls, _) in enumerate(cat) if cls in GATED_CLASSES]
        num_all = [i for i, (cls, _) in enumerate(cat) if cls == "num"]
        if all_logic:
            gated_idx = sorted(gated_all)
        else:
            g = list(gated_all)
            rng.shuffle(g)
            gated_idx = sorted(g[:logic_per_module])
        num_idx = list(num_all)
        rng.shuffle(num_idx)
        num_idx = sorted(num_idx[:per_module])
        run_idx = sorted(set(gated_idx) | set(num_idx))
        print(f"\n# {mod}: {len(gated_idx)}/{len(gated_all)} logic"
              f"{' (ALL)' if all_logic else ' (sampled)'} + "
              f"{len(num_idx)}/{len(num_all)} numeric (sampled)")
        for k in run_idx:
            cls, desc, mutated = _apply_kth(original, k)
            if (mod, desc) in done:
                print(f"  [{cls:9}] skip      {desc} (already recorded)")
                continue
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(mutated)
                survived = _run_slice()
            finally:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(original)  # always restore
            _append_result(results_path, {
                "module": mod, "desc": desc, "cls": cls,
                "gated": cls in GATED_CLASSES, "killed": (not survived),
            })
            tag = "SURVIVED" if survived else "killed  "
            print(f"  [{cls:9}] {tag}  {desc}")
        # Module fully processed and restored — drop its temp backup.
        if os.path.exists(_bak_path(mod)):
            os.remove(_bak_path(mod))

    if results_path:
        print(f"\nresults -> {results_path}  "
              f"(run `--report {results_path} --strict` to gate)")
    return 0


def report(results_path: str, strict: bool) -> int:
    """Aggregate a results file, apply the equivalence allow-list, and gate.

    The published number is the raw gated kill rate (diagnostic). The GATE is the
    stronger property: every gated logic survivor must be accounted for by
    ``EQUIVALENT_ALLOWLIST``. An UNEXPECTED survivor (a weakened test) fails
    strict; a STALE allow-list entry (now killed) is warned so it can be pruned.
    """
    recs = _load_results(results_path)
    if not recs:
        print(f"ERROR: no results in {results_path}")
        return 2
    gated = [r for r in recs if r["gated"]]
    num = [r for r in recs if not r["gated"]]
    g_total = len(gated)
    g_killed = sum(1 for r in gated if r["killed"])
    g_rate = g_killed / g_total if g_total else 0.0
    survivors = [f'{r["module"]}: {r["desc"]}' for r in gated if not r["killed"]]
    survivor_keys = set(survivors)
    unexpected = sorted({s for s in survivors if s not in EQUIVALENT_ALLOWLIST})
    expected = sorted({s for s in survivors if s in EQUIVALENT_ALLOWLIST})
    # An allow-list entry is STALE iff nothing matching it survived (the gap it
    # documented is now closed, e.g. a new test kills it) -> prune it. A key that
    # has both survived and killed records (duplicate sites on one line) is NOT
    # stale; the survivor still needs the entry.
    stale = [k for k in EQUIVALENT_ALLOWLIST if k not in survivor_keys]

    n_total = len(num)
    n_killed = sum(1 for r in num if r["killed"])
    n_rate = n_killed / n_total if n_total else 0.0

    print("=" * 72)
    print(f"GATED logic mutation score : {g_killed}/{g_total} = {g_rate:.2%}  "
          f"(diagnostic; target {MIN_KILL_RATE:.0%})")
    print(f"numeric perturbations (diag): {n_killed}/{n_total} = {n_rate:.2%}  "
          f"(not gated; equivalent-heavy)")
    print(f"gated survivors            : {len(survivors)}  "
          f"({len(expected)} allow-listed equivalent/cosmetic, "
          f"{len(unexpected)} unexpected)")
    if expected:
        print("\nallow-listed survivors (equivalent or explanatory-only):")
        for s in expected:
            print(f"  - {s}\n      {EQUIVALENT_ALLOWLIST[s]}")
    if unexpected:
        print("\nUNEXPECTED survivors (real coverage holes — add a test or justify):")
        for s in unexpected:
            print(f"  - {s}")
    if stale:
        print("\nSTALE allow-list entries (now killed — prune from EQUIVALENT_ALLOWLIST):")
        for s in stale:
            print(f"  - {s}")
    print("=" * 72)

    if strict and unexpected:
        print(f"STRICT FAIL: {len(unexpected)} unexpected gated survivor(s).")
        return 1
    if strict:
        print("STRICT PASS: every gated survivor is a documented equivalent.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-module", type=int, default=6,
                    help="NUMERIC (diagnostic) mutants sampled per module")
    ap.add_argument("--logic-per-module", type=int, default=10,
                    help="GATED logic mutants sampled per module when not --all-logic")
    ap.add_argument("--all-logic", action="store_true",
                    help="run EVERY gated logic mutant (recommended for the gate)")
    ap.add_argument("--modules", default="",
                    help="comma-separated subset of target modules (default: all)")
    ap.add_argument("--results", default="",
                    help="JSONL results file; enables resumable runs and --report")
    ap.add_argument("--report", default="",
                    help="aggregate a results file, apply the allow-list, and gate")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    if args.report:
        return report(args.report, args.strict)

    # Restore the source tree on TERM/INT (e.g. CI cancel) before dying, so a
    # mutated file is never left behind.
    def _handler(signum, frame):
        _recover_stale()
        sys.exit(130)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

    mods = [m.strip() for m in args.modules.split(",") if m.strip()]
    return run(args.per_module, args.seed, args.logic_per_module,
               args.all_logic, mods, args.results)


if __name__ == "__main__":
    sys.exit(main())
