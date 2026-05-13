#!/usr/bin/env bash
set -uo pipefail
FAILED=0
trap 'FAILED=$((FAILED+1))' ERR

# Pre-compute per-sample difficulty tier for all benchmarks (writes
# data/sample_difficulty.csv). Local-only; no GPU required. Run once
# before scripts/tasks/mqi_score.sh.
uv run python scripts/build_difficulty.py

echo "[summary] failed=$FAILED"
exit $FAILED
