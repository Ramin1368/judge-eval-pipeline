#!/usr/bin/env bash
set -euo pipefail
apt-get update -y
apt-get install -y python3 python3-pip python3-venv git
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m eval_pipeline.cli \
  --preferences data/preferences.csv \
  --policies data/policy_outputs.csv \
  --policy-a policy_a --policy-b policy_b \
  --judge heuristic --compare-judges heuristic \
  --out report.md --html report.html
echo "Setup complete. See report.md and report.html."
