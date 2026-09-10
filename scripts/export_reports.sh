#!/usr/bin/env bash
# Execute every analysis notebook and export a rendered Markdown report (with
# figures) to docs/reports/, so the reports render directly on GitHub.
# Reproducible: run from the repo root with the project venv.
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
    --to markdown \
    --execute "$nb" \
    --output-dir "$OUT_DIR" \
    --ExecutePreprocessor.timeout=1800
done

# Chain the reports with sequential Previous/Next navigation footers so readers
# can move through them in order. The reading path starts at the project README
# and the conceptual docs (report 01's Previous links back to CAUSAL_ML.md); the
# last report ends the chain (no Next).
python - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
reports = sorted(out.glob("[0-9][0-9]_*.md"))
causal_ml = out.parent / "CAUSAL_ML.md"  # the doc preceding report 01


def title(md: Path) -> str:
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return md.stem


for i, md in enumerate(reports):
    parts = []
    if i == 0:
        parts.append(f"← Previous: [{title(causal_ml)}](../{causal_ml.name})")
    else:
        prev = reports[i - 1]
        parts.append(f"← Previous: [{title(prev)}]({prev.name})")
    if i + 1 < len(reports):
        nxt = reports[i + 1]
        parts.append(f"Next: [{title(nxt)}]({nxt.name}) →")
    footer = "\n\n---\n\n" + "  ·  ".join(parts) + "\n"
    md.write_text(md.read_text(encoding="utf-8").rstrip() + footer, encoding="utf-8")
print(f"Added Previous/Next navigation to {len(reports)} reports")
PY

echo "Reports written to ${OUT_DIR}/ (Markdown + figure images)"
