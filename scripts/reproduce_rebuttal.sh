#!/usr/bin/env bash
# Recompute every rebuttal table from the per-object artifacts shipped in
# tools/rebuttal/*/results/. No GPU, no datasets, no weights.
#   usage: scripts/reproduce_rebuttal.sh
# To RE-RUN the experiments themselves (GPUs + baseline checkpoints required) see
# tools/rebuttal/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
PYTHONPATH=. ${PYTHON:-python3} tools/rebuttal/report.py
