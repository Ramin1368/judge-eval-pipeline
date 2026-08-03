# Offline Evaluation and Judge Calibration Pipeline

A pipeline that ingests pairwise human preference data, proves an automated
judge agrees with humans before trusting it, and then uses that judge to compare
two candidate policies with confidence intervals and a significance test rather
than a bare point estimate.

The guiding idea is that the hard part is not building a judge, it is knowing
whether you are allowed to believe it, so calibration sits between the judge and
every decision it informs.

## Pipeline

The flow is ingestion, then validation, then judge calibration behind a trust
gate, then policy comparison, then report generation. See
`docs/architecture.mmd` for the diagram and `docs/METHODOLOGY.md` for the
reasoning behind every design and statistical choice.

## Setup

The core pipeline uses only the Python standard library, so it runs anywhere.

```
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
```

On a fresh Ubuntu machine you can run `bash scripts/setup_droplet.sh`, which
installs everything, runs the tests, and produces a first report.

## Run

```
make run
```

or explicitly:

```
PYTHONPATH=src python -m eval_pipeline.cli \
  --preferences data/preferences.csv \
  --policies data/policy_outputs.csv \
  --policy-a policy_a --policy-b policy_b \
  --config config/config.yaml \
  --compare-judges heuristic \
  --out report.md --html report.html --out-json report.json
```

This writes a Markdown report and an HTML report with a win rate chart and a
judge calibration curve. To use DigitalOcean inference as the judge:

```
export DO_INFERENCE_BASE_URL=https://inference.do-ai.run/v1
export DO_INFERENCE_API_KEY=your_model_access_key
export DO_INFERENCE_MODEL=llama3.3-70b-instruct
PYTHONPATH=src python -m eval_pipeline.cli ... --live-llm --compare-judges heuristic llm
```

The LLM judge falls back to the heuristic on any error or missing key, and the
report discloses how often that happened.

## Tests

```
make test
```

The suite covers the parts that must be correct or the report is not
trustworthy: the statistics, the noisy label handling, position bias detection,
inter annotator agreement, confidence calibration, and end to end recovery of
the better policy.

## Reading the report

The report opens with a plain language verdict. Read the headline first: it
names the better policy with a win rate, a confidence interval, and a p value.
If the interval includes fifty percent the result is called inconclusive on
purpose. Read the trust line next: whether the judge was rated trustworthy based
on how well it agreed with real humans. Then read the gaming check: the length
controlled win rate, which tells you whether the win was partly about verbosity.
Everything below the headline is the supporting evidence.

## Service and deployment

```
make service
```

serves the pipeline on port 8080 with a compare endpoint. Deploy to DigitalOcean
App Platform with `service/.do/app.yaml` or containerize with
`service/Dockerfile`.

## Reward hacking and mitigation

A policy can inflate its win rate by padding responses with on topic but low
value text, exploiting a judge's mild length sensitivity rather than being
better. The pipeline mitigates this in three layers, all backed by tests. The
rubric instructs the LLM judge not to reward length, and the heuristic length
prior is kept weak. Policy comparison reports a length controlled win rate
computed only on prompts where the two responses are of comparable length, and
flags a large gap between the raw and length controlled figures as possible
gaming. Finally, `src/eval_pipeline/adversarial.py` ships two named attack
policies (`verbosity_padding_policy`, `sycophancy_policy`) and
`tests/test_adversarial.py` verifies that the mitigations catch them; when the
heuristic *is* exploitable by an attack, that fact is documented as a passing
test, which is the motivation for gating verdicts behind the trust check.

## Layout

```
src/eval_pipeline/
  ingestion.py        flexible CSV and JSONL loading with error collection
  validation.py       dedupe, order swap, majority vote, quarantine
  agreement.py        Fleiss inter annotator agreement (variable rater count aware)
  judges/             order robust base, heuristic, cached DigitalOcean LLM with retries+fallback
  calibration.py      accuracy, Cohen kappa with bootstrap CI, position bias, two-state trust gate
  reliability.py      confidence calibration curve and expected calibration error
  stats.py            BCa bootstrap (primary), percentile bootstrap (cross-check), Wilson CI, exact sign test, MDE, kappa bootstrap CI
  policy_compare.py   both order verdicts, win rate, length control, BCa CI
  bradley_terry.py    BT strengths with tie handling and bootstrap CI
  adversarial.py      verbosity padding and sycophancy attack policies for reward-hacking tests
  cache.py            LLM verdict cache: Protocol, JSON file backend, Valkey adapter
  config.py           YAML config loader wired through the CLI
  compare_judges.py   side by side judge calibration table
  report.py           Markdown and HTML reports with charts, kappa CI, MDE, BT CI, cache stats
  cli.py              end to end entry point with --config, --live-llm, --benchmark, --out-json
tests/                unit tests across stats, validation, calibration, agreement
scripts/              synthetic data generator, droplet setup
service/              FastAPI app, Dockerfile, App Platform spec
.github/workflows/    continuous integration
docs/                 architecture diagram and methodology write up
```
