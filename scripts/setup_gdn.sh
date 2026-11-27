#!/usr/bin/env bash
# Set up the GDN runtime and run the import-and-use smoke test.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-gdn.txt

if [ -d third_party/aidge_export_c ]; then
  python -m pip install -e third_party/aidge_export_c
fi

python examples/use_gdn.py
