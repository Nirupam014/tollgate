# Contributing to Tollgate

Thanks for your interest! Tollgate is a deterministic, dependency-light static
analyzer. Contributions that keep it that way are easiest to merge.

## Development setup

```bash
git clone https://github.com/nirupam014/tollgate
cd tollgate
pip install -e ".[dev]"        # editable install + pytest, build, tiktoken
```

The core needs only the standard library + PyYAML. `tiktoken` is optional (exact
tokenization); everything works without it via a deterministic heuristic.

## Before you open a PR

Both of these must pass:

```bash
python -m unittest discover -s tests        # unit suite (stdlib unittest)
python validation/harness.py --strict       # correctness benchmark / gate
```

For larger changes, also run the full validation suite:

```bash
bash validation/run_all.sh                  # metamorphic, fuzz, mutation, determinism
```

## Ground rules

- **Stay deterministic.** No LLM calls, no network, no nondeterminism in analysis
  output. The tool must never execute the code it scans.
- **Separate facts from estimates.** Structural findings (loops, caps, fan-out) are
  verifiable; cost/token figures are estimates. Don't blur them.
- **Honest failure over silent wrong answers.** If a file can't be analyzed, drop it
  — never emit a confident PASS for something you couldn't read.
- **New detection behavior needs a fixture.** Add a labeled case under
  `validation/corpus/` (and a `labels.yaml` entry) so the benchmark covers it.
- **Keep dependencies minimal.** Anything beyond stdlib + PyYAML needs a good reason.

## Reporting bugs / requesting features

Open an issue using the templates under `.github/ISSUE_TEMPLATE/`. For a bug,
include the input file (or a minimal repro) and the command you ran — the analyzer
is deterministic, so a repro reproduces.

## License

By contributing, you agree your contributions are licensed under Apache-2.0.
