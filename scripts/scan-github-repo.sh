#!/usr/bin/env bash
# scan-github-repo.sh — read-only Tollgate scan of a GitHub repo (public or private).
#
# What it does:
#   1. Shallow-clones the repo into a throwaway temp directory.
#   2. Runs the analyzer (read-only) and writes reports OUTSIDE the clone.
#   3. ALWAYS deletes the cloned repo afterward (even on error / Ctrl-C).
#
# It NEVER modifies the scanned repo. Nothing is committed, pushed, or written
# back into the clone — the reports are advisory recommendations for human review.
#
# Usage:
#   scripts/scan-github-repo.sh <github-url> [options]
#
# Arguments:
#   <github-url>              https://github.com/<owner>/<repo>[.git] or git@github.com:<owner>/<repo>.git
#
# Options:
#   --token <PAT>             auth token for private repos (or set GITHUB_TOKEN).
#                             Works with classic & fine-grained PATs and app tokens.
#   --branch <name>           branch/tag to clone (default: the repo's default branch).
#   --paths "<a> <b> ...""    space-separated subpaths inside the repo to scan (default: whole repo).
#   --output-dir <dir>        where reports are written (default: ./tollgate-reports/<repo>-<ts>).
#   --config <file>           path to a .tollgate.yml (otherwise auto-discovered in the clone).
#   --models <file>           your own model catalog YAML (overrides the seed catalog).
#   --default-model <id>      fallback model id when a node declares none.
#   --traffic-per-week <N>    estimated requests/week (default 10,000). Overrides config scenarios.
#   --traffic-per-day <N>     estimated requests/day (mutually exclusive with --traffic-per-week).
#   --horizon-days <N>        projection horizon in days for the traffic estimate (default 30).
#   --fail-on <block|warn|never>  exit code policy (default: never — a scan never "fails").
#   -h, --help                show this help.
#
# Examples:
#   scripts/scan-github-repo.sh https://github.com/org/agents
#   scripts/scan-github-repo.sh https://github.com/org/private --token "$GITHUB_TOKEN"
#   scripts/scan-github-repo.sh https://github.com/org/repo --paths "agents prompts" --fail-on block
#   scripts/scan-github-repo.sh https://github.com/org/repo --traffic-per-week 50000
#   scripts/scan-github-repo.sh https://github.com/org/repo --traffic-per-day 8000 --horizon-days 14
#
set -euo pipefail

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[ $# -ge 1 ] || usage
case "$1" in -h|--help) usage ;; esac

URL="$1"; shift

TOKEN="${GITHUB_TOKEN:-}"
BRANCH=""
PATHS=""
OUTPUT_DIR=""
CONFIG_ARG=()
MODELS_ARG=()
DEFMODEL_ARG=()
TRAFFIC_ARG=()
HORIZON_ARG=()
FAIL_ON="never"

# Require a value for options that take one, with a clear error instead of a
# cryptic "unbound variable" under set -u when the value is missing.
need_val() { [ $# -ge 2 ] || { echo "ERROR: option '$1' requires a value" >&2; usage; }; }

while [ $# -gt 0 ]; do
  case "$1" in
    --token)         need_val "$@"; TOKEN="$2"; shift 2 ;;
    --branch)        need_val "$@"; BRANCH="$2"; shift 2 ;;
    --paths)         need_val "$@"; PATHS="$2"; shift 2 ;;
    --output-dir)    need_val "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --config)        need_val "$@"; CONFIG_ARG=(--config "$2"); shift 2 ;;
    --models)        need_val "$@"; MODELS_ARG=(--models "$2"); shift 2 ;;
    --default-model) need_val "$@"; DEFMODEL_ARG=(--default-model "$2"); shift 2 ;;
    --traffic-per-week)
      need_val "$@"
      [ ${#TRAFFIC_ARG[@]} -eq 0 ] || { echo "ERROR: --traffic-per-week and --traffic-per-day are mutually exclusive" >&2; usage; }
      TRAFFIC_ARG=(--traffic-per-week "$2"); shift 2 ;;
    --traffic-per-day)
      need_val "$@"
      [ ${#TRAFFIC_ARG[@]} -eq 0 ] || { echo "ERROR: --traffic-per-week and --traffic-per-day are mutually exclusive" >&2; usage; }
      TRAFFIC_ARG=(--traffic-per-day "$2"); shift 2 ;;
    --horizon-days)  need_val "$@"; HORIZON_ARG=(--horizon-days "$2"); shift 2 ;;
    --fail-on)       need_val "$@"; FAIL_ON="$2"; shift 2 ;;
    -h|--help)       usage ;;
    *)               echo "unknown option: $1" >&2; usage ;;
  esac
done

command -v git >/dev/null 2>&1      || { echo "ERROR: git not found" >&2; exit 1; }
command -v tollgate >/dev/null 2>&1 || { echo "ERROR: tollgate not installed (pip install ./tollgate)" >&2; exit 1; }

# Derive a slug for the output folder name.
SLUG="$(basename "${URL%.git}")"
[ -n "$SLUG" ] || SLUG="repo"
TS="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-./tollgate-reports/${SLUG}-${TS}}"

# Throwaway clone dir; removed on any exit.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/tollgate-scan.XXXXXX")"
ASKPASS="$WORKDIR/askpass.sh"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT INT TERM

# --- Auth (private repos) -----------------------------------------------------
# We never put the token in the clone URL (which would persist in .git/config and
# show up in `ps`). Instead we hand it to git via GIT_ASKPASS, passing the secret
# through an env var that only this process tree can read.
CLONE_URL="$URL"
GIT_ENV=()
if [ -n "$TOKEN" ]; then
  case "$URL" in
    https://*@*) ;;  # caller already embedded credentials; leave as-is
    https://*)
      # Inject a username so git asks askpass only for the password (the token).
      CLONE_URL="https://x-access-token@${URL#https://}"
      ;;
    *)
      echo "WARNING: --token is only used for https:// URLs; ignoring for $URL" >&2
      ;;
  esac
  cat > "$ASKPASS" <<'EOF'
#!/usr/bin/env bash
printf '%s' "$TOLLGATE_GIT_TOKEN"
EOF
  chmod 700 "$ASKPASS"
  GIT_ENV=(env "TOLLGATE_GIT_TOKEN=$TOKEN" "GIT_ASKPASS=$ASKPASS" "GIT_TERMINAL_PROMPT=0")
fi

# --- Clone (shallow, read-only) ----------------------------------------------
echo ">> Cloning (shallow, read-only): ${URL}${BRANCH:+ @ $BRANCH}"
CLONE_ARGS=(clone --depth 1 --no-tags --single-branch)
[ -n "$BRANCH" ] && CLONE_ARGS+=(--branch "$BRANCH")
if ! ${GIT_ENV[@]+"${GIT_ENV[@]}"} git "${CLONE_ARGS[@]}" "$CLONE_URL" "$WORKDIR/repo" >/dev/null 2>&1; then
  echo "ERROR: could not clone $URL" >&2
  echo "       (check the URL/branch, network access, or pass --token for a private repo)" >&2
  exit 1
fi

# --- Resolve scan targets -----------------------------------------------------
TARGETS=()
if [ -n "$PATHS" ]; then
  for p in $PATHS; do
    t="$WORKDIR/repo/$p"
    [ -e "$t" ] || { echo "ERROR: path '$p' not found in repo" >&2; exit 1; }
    TARGETS+=("$t")
  done
else
  TARGETS=("$WORKDIR/repo")
fi

# --- Output dir (must live OUTSIDE the temp clone so it survives cleanup) ------
mkdir -p "$OUTPUT_DIR"
OUTPUT_ABS="$(cd "$OUTPUT_DIR" && pwd)"
case "$OUTPUT_ABS/" in
  "$WORKDIR"/*) echo "ERROR: --output-dir is inside the temp clone; choose another path" >&2; exit 1 ;;
esac

echo ">> Scanning (read-only — the repo is NOT modified)"
echo

# Terminal summary to stdout + machine/human reports to files. We capture the
# exit code so cleanup + reports always happen, then propagate it at the end.
set +e
tollgate analyze "${TARGETS[@]}" \
  ${CONFIG_ARG[@]+"${CONFIG_ARG[@]}"} ${MODELS_ARG[@]+"${MODELS_ARG[@]}"} ${DEFMODEL_ARG[@]+"${DEFMODEL_ARG[@]}"} \
  ${TRAFFIC_ARG[@]+"${TRAFFIC_ARG[@]}"} ${HORIZON_ARG[@]+"${HORIZON_ARG[@]}"} \
  -f terminal \
  -o "markdown=$OUTPUT_ABS/report.md" \
  -o "json=$OUTPUT_ABS/report.json" \
  -o "sarif=$OUTPUT_ABS/report.sarif" \
  -o "html=$OUTPUT_ABS/dashboard.html" \
  --fail-on "$FAIL_ON"
CODE=$?
set -e

# Prepend a reviewer banner to the Markdown report (token-based; advisory only).
if [ -f "$OUTPUT_ABS/report.md" ]; then
  BANNER="$WORKDIR/banner.md"
  cat > "$BANNER" <<'EOF'
> **How to read this report.** These are **recommendations for human review**, not
> changes — this tool is read-only and modified nothing in the scanned repo. All
> figures are **token** projections (consumption, not dollars) over the configured
> traffic scenarios. Act on the structural findings (unbounded loops, context
> explosion, fan-out, prompt bloat) and the token-reducing remediation below by
> applying the suggested changes yourself, in your own repo.

EOF
  cat "$BANNER" "$OUTPUT_ABS/report.md" > "$OUTPUT_ABS/report.md.tmp"
  mv "$OUTPUT_ABS/report.md.tmp" "$OUTPUT_ABS/report.md"
fi

echo
# Exit code 2 is an analyzer error (bad catalog, usage, parse) — NOT a gate
# decision (0=pass/never, 1=block). Don't claim success if nothing was produced.
if [ "$CODE" -eq 2 ] || [ ! -f "$OUTPUT_ABS/report.json" ]; then
  echo ">> Scan did NOT complete — no reports were written (see the error above)." >&2
  echo ">> Cloned repo deleted."
  exit "${CODE:-1}"
fi

echo ">> Reports written to: $OUTPUT_ABS"
for f in "report.md:(human-readable, token report)" \
         "report.json:(full machine-readable result)" \
         "report.sarif:(GitHub code-scanning format)" \
         "dashboard.html:(visual dashboard — open in a browser)"; do
  name="${f%%:*}"; desc="${f#*:}"
  [ -f "$OUTPUT_ABS/$name" ] && printf '     - %-15s %s\n' "$name" "$desc"
done
echo ">> Cloned repo deleted."
# WORKDIR removed by trap. Propagate the gate exit code for automation.
exit "$CODE"
