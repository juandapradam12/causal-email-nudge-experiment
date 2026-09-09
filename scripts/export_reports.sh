#!/usr/bin/env bash
# Execute every analysis notebook and export a rendered HTML report to
# docs/reports/. Reproducible: run from the repo root with the project venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -x ".venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

OUT_DIR="docs/reports"
mkdir -p "$OUT_DIR"

for nb in notebooks/*.ipynb; do
  echo "Rendering ${nb} ..."
  jupyter nbconvert \
    --to html \
    --execute "$nb" \
    --output-dir "$OUT_DIR" \
    --ExecutePreprocessor.timeout=1800
done

echo "Reports written to ${OUT_DIR}/"
