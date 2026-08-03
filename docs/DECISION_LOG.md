# Decision Log

This file records the non-obvious design decisions in this repo, why they
were made, and which items were deliberately deferred with the reasoning
behind the deferral. It is intended to answer the "why not X?" questions a
reviewer will ask before they ask them.

## Statistical rigor

**Primary CI is BCa bootstrap, not percentile bootstrap.** The plain
percentile bootstrap under-covers for skewed proportions and small samples,
which is exactly the regime this pipeline runs in (n = 80 held-out prompts,
0.5 boundary). BCa corrects for bias in the resampling distribution and for
skew via jackknife acceleration. The percentile CI is kept as a cross-check
in the report because disagreement between the two is diagnostic.

**Cohen's kappa is reported with a bootstrap 95% CI, and the trust flag has
two states.** A single point estimate on 10-100 labeled items is nearly
uninformative; the 0.4 trust gate is only defensible if we know how noisy
the estimate is. The pipeline reports `trustworthy` when the point estimate
clears the gate and `trustworthy_with_reserve` when the point estimate
clears the gate but the CI lower bound does not. The report calls this out
explicitly to a non-research stakeholder.

**Bradley-Terry has a bootstrap CI on strengths.** For two policies BT is
algebraically equivalent to the raw win rate, which is why the point
estimate alone adds no information. The bootstrap CI on strengths turns BT
into a real ranking-stability estimate and scaffolds the multi-policy case
(3+ policies where transitivity matters) without changing the API.

**Fleiss kappa handles variable rater counts per item.** The standard form
assumes constant n across items; this repo uses the generalized form so
items with 2, 3, or 5 annotators all contribute correctly. The prior
implementation would silently truncate to `min(n_i)`, discarding real
agreement signal.

**Sign test, not a t-test, for policy significance.** Wins are Bernoulli
per prompt, not continuous outcomes, so the exact two-sided binomial sign
test against 0.5 is the correct significance test. Wilson interval is
provided in parallel for the decisive-only view.

**Minimum detectable effect (MDE) is surfaced in the report.** With n = 80
prompts at alpha = 0.05, power = 0.80, MDE ~ 8.8 percentage points around
0.5. Reviewers who do not know this get a false sense of resolution from
"we detected a 3-point difference"; the report tells them what n can and
cannot detect.

## Adversarial testing

**Reward-hacking mitigation is a test suite, not a paragraph.** Two named
attack policies live in `src/eval_pipeline/adversarial.py` (verbosity
padding and sycophancy prefix). The unit tests in
`tests/test_adversarial.py` verify that:

- Verbosity padding is caught by the length-controlled win rate (or drops
  from the parity sample entirely).
- A sycophancy prefix does not create spurious position bias, because
  `Judge.judge` averages both orders.

If these mitigations regress, the test suite will fail. That is the whole
point.

## Judge implementation

**LLM verdict caching is required for a defensible calibration study.** The
calibration loop, judge comparison, and policy comparison all issue the same
`(model, prompt, response_a, response_b)` requests. Without a cache the LLM
judge issues 3x-10x more calls than needed, which is wasted budget and adds
latency-driven variance. The cache is a `Protocol`; the default backend is
a JSON file so the pipeline runs on a laptop; a Valkey adapter documents
the production path on DigitalOcean.

**Both-order averaging is done inside `Judge.judge`, not by the caller.**
Position bias is a judge-level property. Doing the swap inside the abstract
class means every judge (heuristic, LLM, future ones) inherits the
protection uniformly and the calibration report's `position_bias_rate` is
comparable across judges.

**Exponential backoff with jitter and a heuristic fallback on API failure.**
The `DigitalOceanLLMJudge` retries on 429/5xx and transient network errors,
then falls back to the heuristic and reports `fallback_rate`. A silent
degradation is worse than a loud one: the report shows fallback rate so a
stakeholder knows how much of the verdict was live vs offline.

## Synthetic data

**Bad responses are lexically similar to good ones and length-matched to
them.** The prior generator produced kappa = 1.00 / accuracy = 100% because
the good and bad templates were so different that the heuristic's
token-overlap signal was perfect. The current generator uses hollow
distractors ("depends on many factors") and confidently-wrong statements
that share prompt tokens, so the heuristic must actually discriminate. A
label-flip rate of 15%, two prompts with unresolvable 1-A/1-B/1-tie splits,
and length parity between good and bad responses make the demo dataset a
real trust-gate test rather than a smoke test.

## Deferred items (logged, not skipped)

Items D1 and D2 were flagged by the reviewer as important and are logged
here as deferred with explicit rationale. They should be executed as a
separate pass before the final interview presentation.

### D1: Run the live DigitalOcean LLM judge and commit head-to-head calibration

**Status:** Deferred to a dedicated follow-up session.

**What's missing:** the head-to-head calibration of
`DigitalOceanLLMJudge` (Llama 3.3 70B Instruct) against the heuristic on
the same labeled slice, committed as a numbered report artifact.

**Why deferred:** the run requires a live API key on a droplet with
outbound access to `https://inference.do-ai.run/v1`, which is set up but
outside the scope of a code-only pass. All the machinery is here (cache,
retries, fallback, both-order averaging), so the run is a matter of
setting `DO_INFERENCE_API_KEY` and calling
`python -m eval_pipeline.cli --live-llm --compare-judges heuristic llm --out reports/live.md`.

**Definition of done:** a `reports/live_llm_vs_heuristic.md` artifact in
the repo with the LLM judge's kappa, position bias, ECE, fallback rate,
and cache stats, plus a paragraph on where the LLM outperformed and where
it did not.

### D2: Public preference benchmark slice

**Status:** Deferred, run separately per the user's instruction.

**What's missing:** the pipeline calibrated on a small, redistributable
slice of a public preference dataset so reviewers can trust the numbers
away from the synthetic demo.

**Sequence:** D2a first (HH-RLHF Anthropic helpful-base slice, ~200 pairs),
then D2b (LMSYS Chatbot Arena slice, ~200 pairs) as a separate run. Both
runs use the same CLI (`--benchmark`) and produce distinct report
artifacts committed to the repo.

**Why deferred:** dataset downloads and license attribution are a separate
concern from the code changes in this pass. The `--benchmark` CLI hook and
`data/benchmark/` directory are in place so the downloader script can drop
files in and the pipeline runs unchanged.

**Definition of done:** two artifacts in `reports/`: one for HH-RLHF-style
and one for Arena-style, each with dataset provenance, license note, and
the same headline metrics as the demo report.
