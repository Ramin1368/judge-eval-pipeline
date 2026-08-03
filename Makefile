.PHONY: setup data test run html service
setup:
	pip install -r requirements.txt
data:
	python scripts/generate_synthetic_data.py
test:
	PYTHONPATH=src pytest -q
run:
	PYTHONPATH=src python -m eval_pipeline.cli --preferences data/preferences.csv --policies data/policy_outputs.csv --policy-a policy_a --policy-b policy_b --judge heuristic --compare-judges heuristic --out report.md --html report.html
service:
	PYTHONPATH=src uvicorn service.app:app --host 0.0.0.0 --port 8080
