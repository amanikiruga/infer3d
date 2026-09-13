#!/usr/bin/env bash
# Assemble Tables 1 & 2 from the per-object eval CSVs (no GPU needed).
# Uses the CSVs bundled in assets/eval_csvs/ unless RESULTS_ROOT is set.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
PYTHONPATH=. ${PYTHON:-python3} tools/make_tables.py
