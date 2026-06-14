#!/usr/bin/env bash
# scan-repo.sh — read-only Tollgate scan of a public Git repo.
#
# What it does:
#   1. Shallow-clones the repo into a throwaway temp directory.
#   2. Runs the analyzer (read-only) and writes a Markdown report OUTSIDE the clone.
#   3. Always deletes the cloned repo afterward (even on error/Ctrl-C).
#
# It NEVER modifies the scanned repo. The report is a set of recommendations for
# human review — no changes are applied to the repo by this tool.
#
# Usage:
#   scripts/scan-repo.sh <git-url> [subpath] [-o report.md] [--models catalog.yml]
#
#   <git-url>            e.g. https://github.com/org/repo
#   [subpath]            optional path inside the repo to scan (default: whole repo)
#   -o <report.md>       report output path (default: ./tollgate-report.md)
#                        a visual dashboard is also written alongside it as <report>.html
#   --models <file>      your own model-pricing catalog (recommended; see note below)
#
set -euo pipefail

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac

URL="$1"; shift
SUBPATH="."
REPORT="./tollgate-report.md"
MODELS_ARG=()

need_val() { [ $# -ge 2 ] || { echo "ERROR: option '$1' requires a value" >&2; usage; }; }

while [ $# -gt 0 ]; do
  case "$1" in
    -o)        need_val "$@"; REPORT="$2"; shift 2 ;;
    --models)  need_val "$@"; MODELS_ARG=(--models "$2"); shift 2 ;;
    -*)        echo "unknown option: $1" >&2; usage ;;
    *)         SUBPATH="$1"; shift ;;
  esac
done

command -v git >/dev/null 2>&1        || { echo "ERROR: git not found" >&2; exit 1; }
command -v tollgate >/dev/null 2>&1 || { echo "ERROR: tollgate not installed (pip install ./tollgate)" >&2; exit 1; }

# Throwaway clone dir; removed on any exit.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/tollgate-scan.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT INT TERM

echo ">> Cloning (shallow, read-only): $URL"
if ! git clone --depth 1 "$URL" "$WORKDIR/repo" >/dev/null 2>&1; then
  echo "ERROR: could not clone $URL (private repo, bad URL, or no network)" >&2
  exit 1
fi

TARGET="$WORKDIR/repo/$SUBPATH"
[ -e "$TARGET" ] || { echo "ERROR: subpath '$SUBPATH' not found in repo" >&2; exit 1; }

# Resolve report path to an absolute location OUTSIDE the temp clone so it survives cleanup.
REPORT_ABS="$(cd "$(dirname "$REPORT")" && pwd)/$(basename "$REPORT")"
case "$REPORT_ABS" in
  "$WORKDIR"/*) echo "ERROR: report path is inside the temp clone; choose another -o path" >&2; exit 1 ;;
esac

# Visual dashboard goes next to the report: strip the report's extension, add .html.
HTML_ABS="${REPORT_ABS%.*}.html"
[ "$HTML_ABS" = "$REPORT_ABS" ] && HTML_ABS="$REPORT_ABS.html"

echo ">> Scanning (read-only — the repo is NOT modified)"
echo

# Terminal summary to stdout + Markdown report + visual HTML dashboard to files.
# --fail-on never so a 'block' gate does not abort the script (we always want the
# reports + cleanup). A non-zero exit here is a real analyzer error (e.g. a bad
# --models path), so surface it clearly rather than claiming success below.
if ! tollgate analyze "$TARGET" "${MODELS_ARG[@]}" \
     -f terminal -o "markdown=$REPORT_ABS" -o "html=$HTML_ABS" --fail-on never; then
  echo >&2
  echo ">> Scan did NOT complete — no reports were written (see the error above)." >&2
  exit 1
fi

# Prepend a reviewer banner so dollar figures are never mistaken for quotes and
# it is explicit that this is advisory, not an auto-fix.
BANNER="$WORKDIR/banner.md"
cat > "$BANNER" <<'EOF'
> **How to read this report.** These are **recommendations for human review**, not
> changes — this tool is read-only and modified nothing in the scanned repo.
> Dollar figures use an **illustrative** pricing catalog and should be treated as
> *relative* signal, not quotes; pass `--models <your-catalog.yml>` for real costs.
> The structural findings (unbounded loops, context explosion, model mismatch) and
> the suggested changes below are what to act on — apply them yourself, in your repo.

EOF
cat "$BANNER" "$REPORT_ABS" > "$REPORT_ABS.tmp" && mv "$REPORT_ABS.tmp" "$REPORT_ABS"

echo
echo ">> Report written to:    $REPORT_ABS"
echo ">> Dashboard written to: $HTML_ABS  (open in a browser)"
echo ">> Cloned repo deleted."
# WORKDIR removed by trap.
