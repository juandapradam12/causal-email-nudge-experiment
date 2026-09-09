#!/usr/bin/env bash
# Idempotent setup for the causal-email-nudge-experiment Cloud Agent environment.
# Installs the system packages needed to build a Python virtualenv and the
# project's Python dependencies into a repo-local .venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# System packages: python venv support plus a compiler toolchain in case a
# science dependency has no prebuilt wheel for the running interpreter.
if ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    python3-venv python3-dev build-essential
fi

# Create the virtualenv once; reuse it on subsequent (idempotent) runs.
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

echo "Environment ready: $(python --version) in ${REPO_ROOT}/.venv"
