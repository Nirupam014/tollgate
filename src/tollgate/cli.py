"""Command-line interface.

    tollgate analyze <paths...> [options]   # the CI entry point
    tollgate init                            # write a starter .tollgate.yml
    tollgate models [--models FILE]          # list catalog + pricing

Exit codes (so CI can gate on them):
    0  pass
    0  warn  (unless --fail-on warn)
    1  block (or warn when --fail-on warn)
    2  usage/parse error
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .catalog import ModelCatalog
from .config import Config
from .pipeline import analyze_path, apply_baseline, RunResult
from . import report as report_mod
from .version import __version__

_FORMATS = {"terminal", "markdown", "json", "sarif", "gitlab", "html"}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tollgate",
        description="Tollgate — prevention-first token-risk analysis for AI agents, for CI/CD.",
    )
    parser.add_argument("--version", action="version", version=f"tollgate {__version__}")
    sub = parser.add_subparsers(dest="command")

    pa = sub.add_parser("analyze", help="Analyze workflows/prompts and gate on risk.")
    pa.add_argument("paths", nargs="+", help="Files or directories to analyze.")
    pa.add_argument("-c", "--config", help="Path to .tollgate.yml (auto-discovered otherwise).")
    pa.add_argument("--models", help="Path to a model catalog YAML to override seed pricing.")
    pa.add_argument("-f", "--format", action="append", choices=sorted(_FORMATS),
                    help="Output format(s). Repeatable. Default: terminal.")
    pa.add_argument("-o", "--output", action="append", default=[],
                    help="format=path to write a report to a file, e.g. markdown=report.md")
    pa.add_argument("--fail-on", choices=["block", "warn", "never"],
                    help="Severity that causes a non-zero exit. Overrides config.")
    pa.add_argument("--baseline", metavar="REPORT.json",
                    help="PR-delta mode: gate on the difference vs this baseline "
                    "report.json (only NEW or WORSENED findings can fail the build; "
                    "pre-existing issues are reported but never block). Produce the "
                    "baseline on your default branch with `-o json=baseline.json`.")
    pa.add_argument("--default-model", help="Fallback model id when a node declares none.")
    pa.add_argument("--no-prompt-review", dest="prompt_review", action="store_false",
                    default=None, help="Disable the prompt efficiency reviewer "
                    "(on by default).")
    pa.add_argument("--no-prompt-scan", dest="prompt_scan", action="store_false",
                    default=None, help="Disable mining embedded prompts from "
                    "source/config (on by default).")
    traffic = pa.add_mutually_exclusive_group()
    traffic.add_argument("--traffic-per-week", type=float, metavar="N",
                         help="Estimated requests/week (default 10,000). Sets a single "
                         "steady-state scenario and overrides config scenarios.")
    traffic.add_argument("--traffic-per-day", type=float, metavar="N",
                         help="Estimated requests/day. Sets a single steady-state "
                         "scenario and overrides config scenarios.")
    pa.add_argument("--horizon-days", type=int, default=30,
                    help="Projection horizon in days for the traffic estimate (default 30).")

    pi = sub.add_parser("init", help="Write a starter .tollgate.yml in the current directory.")
    pi.add_argument("--force", action="store_true")

    pm = sub.add_parser("models", help="List models and pricing from the catalog.")
    pm.add_argument("--models", help="Path to a model catalog YAML.")

    pv = sub.add_parser("verify", help="Re-derive the gate and check a report wasn't "
                        "edited or drifted (self-healing CI check).")
    pv.add_argument("report", help="A prior report.json to verify.")
    pv.add_argument("paths", nargs="+", help="The same paths the report was produced from.")
    pv.add_argument("-c", "--config", help="Path to .tollgate.yml (auto-discovered otherwise).")
    pv.add_argument("--models", help="Path to a model catalog YAML.")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "models":
        return _cmd_models(args)
    if args.command == "verify":
        return _cmd_verify(args)
    parser.print_help()
    return 2


# --- verify (self-healing tamper/drift check) ---------------------------------
def _cmd_verify(args) -> int:
    """Recompute the gate from the same inputs and compare fingerprints.

    A mismatch means the report was hand-edited, the inputs changed, or the tool
    version differs — i.e. the published verdict no longer reflects the code. In CI
    you re-run `analyze` to overwrite the stale report (self-healing); `verify` is
    the cheap gate that fails the build when someone overwrites the gate output.
    """
    import json as _json
    try:
        with open(args.report, encoding="utf-8") as fh:
            prior = _json.load(fh)
    except (OSError, ValueError) as e:
        print(f"error: could not read report {args.report}: {e}", file=sys.stderr)
        return 2
    claimed = prior.get("fingerprint")
    if not claimed:
        print("error: report has no fingerprint (produced by an older version?)",
              file=sys.stderr)
        return 2

    cfg = Config.load(path=args.config, start_dir=args.paths[0])
    if args.models:
        cfg.models_file = args.models
    try:
        catalog = ModelCatalog.load(cfg.models_file)
    except Exception as e:
        print(f"error: could not load model catalog: {e}", file=sys.stderr)
        return 2

    run = analyze_path(args.paths, cfg=cfg, catalog=catalog)
    actual = run.fingerprint
    if actual == claimed:
        print(f"OK: report matches a fresh re-derivation (fingerprint {actual[:12]}…). "
              f"Gate: {run.gate_decision.upper()}.")
        return 0
    print("MISMATCH: the report does not match a fresh analysis of these inputs.",
          file=sys.stderr)
    print(f"  report fingerprint:   {claimed[:16]}…", file=sys.stderr)
    print(f"  recomputed fingerprint:{actual[:16]}…", file=sys.stderr)
    print(f"  report gate={prior.get('gate_decision')!r}  recomputed gate="
          f"{run.gate_decision!r}", file=sys.stderr)
    print("  -> the report was edited, the inputs changed, or the tool version "
          "differs. Re-run `tollgate analyze` to regenerate it.", file=sys.stderr)
    return 1


# --- analyze -------------------------------------------------------------------
def _cmd_analyze(args) -> int:
    cfg = Config.load(path=args.config, start_dir=args.paths[0])
    if args.models:
        cfg.models_file = args.models
    if args.default_model:
        cfg.default_model = args.default_model
    if args.fail_on:
        cfg.fail_on = args.fail_on
    if args.prompt_review is False:          # --no-prompt-review overrides config
        cfg.prompt_review = False
    if getattr(args, "prompt_scan", None) is False:
        cfg.prompt_scan = False
    # Traffic estimate from the command line overrides config scenarios with a
    # single steady-state scenario built from the given per-day/per-week volume.
    if args.traffic_per_week is not None or args.traffic_per_day is not None:
        per = "day" if args.traffic_per_day is not None else "week"
        volume = args.traffic_per_day if per == "day" else args.traffic_per_week
        cfg.scenarios = [{
            "name": f"{int(volume):,}/{per}",
            "rps": (volume / 86400.0) if per == "day" else (volume / (7 * 86400.0)),
            "horizon_days": args.horizon_days,
        }]

    try:
        catalog = ModelCatalog.load(cfg.models_file)
    except Exception as e:
        print(f"error: could not load model catalog: {e}", file=sys.stderr)
        return 2

    run = analyze_path(args.paths, cfg=cfg, catalog=catalog)

    if args.baseline:
        from .baseline import load_baseline
        try:
            baseline = load_baseline(args.baseline)
        except (OSError, ValueError) as e:
            print(f"error: could not read baseline {args.baseline}: {e}", file=sys.stderr)
            return 2
        apply_baseline(run, baseline, cfg)

    formats = args.format or ["terminal"]
    # Always print the primary (first) format to stdout.
    primary = formats[0]
    sys.stdout.write(_render(run, primary) + "\n")

    # Additional --output format=path targets.
    outputs = {}
    for spec in args.output:
        if "=" not in spec:
            print(f"error: --output expects format=path, got {spec!r}", file=sys.stderr)
            return 2
        fmt, path = spec.split("=", 1)
        if fmt not in _FORMATS:
            print(f"error: unknown output format {fmt!r}", file=sys.stderr)
            return 2
        outputs[fmt] = path
    for fmt, path in outputs.items():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_render(run, fmt))

    # Write GitHub job summary if running in Actions.
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(report_mod.to_markdown(run) + "\n")
        except OSError:
            pass

    return _exit_code(run, cfg.fail_on)


def _render(run: RunResult, fmt: str) -> str:
    if fmt == "json":
        return report_mod.to_json(run)
    if fmt == "markdown":
        return report_mod.to_markdown(run)
    if fmt == "sarif":
        return report_mod.to_sarif(run)
    if fmt == "gitlab":
        return report_mod.to_gitlab_codequality(run)
    if fmt == "html":
        return report_mod.to_html(run)
    return report_mod.to_terminal(run)


def _exit_code(run: RunResult, fail_on: str) -> int:
    # In PR-delta mode the effective gate is the delta gate (new/worsened only),
    # so a PR is never failed by pre-existing risk it didn't introduce.
    gate = run.effective_gate
    if fail_on == "never":
        return 0
    if fail_on == "warn":
        return 1 if gate in ("warn", "block") else 0
    # default: block
    return 1 if gate == "block" else 0


# --- init ----------------------------------------------------------------------
_STARTER = """# Tollgate configuration
default_model: gpt-4o
fail_on: block            # block | warn | never

prompt_review: true       # prompt efficiency reviewer (also: --no-prompt-review)

# Base traffic assumption: 10,000 requests/week.
# Override per run with --traffic-per-week / --traffic-per-day.
scenarios:
  - { name: steady_state, requests_per_week: 10000, horizon_days: 30 }
  # also accepted: requests_per_day: 1500  —  or the raw rps: 0.0165

substitution:
  min_capability: 0.75
  min_savings_pct: 20

policies:
  - name: loops_must_terminate
    type: loop_guard
    enforcement: block
    rule: { require_termination_guard: true, max_depth: 10 }
  # Token-based ceilings replace dollar budgets. Set caps in TOKENS.
  - name: prod_token_ceiling
    type: token_ceiling
    enforcement: block
    rule: { max_monthly_tokens: 2000000000, metric: projected_p95 }
  - name: per_request_token_ceiling
    type: token_ceiling
    enforcement: warn
    rule: { max_tokens_per_request: 50000 }
"""


def _cmd_init(args) -> int:
    path = ".tollgate.yml"
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists (use --force to overwrite).", file=sys.stderr)
        return 2
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_STARTER)
    print(f"wrote {path}")
    return 0


# --- models --------------------------------------------------------------------
def _cmd_models(args) -> int:
    catalog = ModelCatalog.load(args.models)
    print(f"{'model':<22}{'provider':<11}{'tier':<6}{'ctx':>10}{'in/Mtok':>10}{'out/Mtok':>10}")
    for m in sorted(catalog.all(), key=lambda x: (x.provider, -x.quality_tier)):
        print(f"{m.id:<22}{m.provider:<11}{m.quality_tier:<6}{m.context_limit:>10,}"
              f"{m.input_per_mtok:>10.3f}{m.output_per_mtok:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
