#!/usr/bin/env bash
# Recompute every paper and rebuttal table from the per-object artifacts bundled in this
# repository. No GPU, no datasets, no model weights, no install: standard library only.
#   usage: scripts/verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
echo "### Paper Tables 1 & 2 (tools/make_tables.py)"
PYTHONPATH=. $PY tools/make_tables.py
echo
echo "### Rebuttal, all tables (tools/rebuttal/report.py)"
PYTHONPATH=. $PY tools/rebuttal/report.py
echo
echo "Both completed. See RESULTS.md for the written-up numbers and REPRODUCTION.md"
echo "for the audit, including the cells that do NOT reproduce."
