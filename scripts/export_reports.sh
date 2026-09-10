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

# Chain the reports with a sequential "Next" navigation footer so readers can
# move through them in order (01 → 02 → ... → 05 → project README).
python - "$OUT_DIR" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
reports = sorted(out.glob("[0-9][0-9]_*.md"))


def title(md: Path) -> str:
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return md.stem


for i, md in enumerate(reports):
    nxt = reports[i + 1] if i + 1 < len(reports) else None
    if nxt is not None:
        link = f"[{title(nxt)}]({nxt.name})"
    else:
        link = "[Project overview (README)](../../README.md)"
    body = md.read_text(encoding="utf-8").rstrip() + f"\n\n---\n\n### Next\n\n→ {link}\n"
    md.write_text(body, encoding="utf-8")
print(f"Added Next navigation to {len(reports)} reports")
PY

echo "Reports written to ${OUT_DIR}/ (Markdown + figure images)"
