#!/usr/bin/env bash
set -euo pipefail

if [[ -f .venv/bin/activate ]]; then
    # Local agent/developer workflow: use the project virtualenv when present.
    source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

python -m pytest tests/ -n auto --maxprocesses=8 --dist=loadfile -v \
    --cov=dragon3_pipelines --cov-report=xml --cov-report=term
