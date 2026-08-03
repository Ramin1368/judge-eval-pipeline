# Evaluation Methodology

This document states the assumptions behind the pipeline, how the judge was
designed and why, the failure modes that shaped it, and the statistical choices
used to compare policies.

## Problem framing

The task is a paired comparison with ties. For each prompt two responses exist
and a preference is expressed (A, B, or tie). The quantity we estimate for
policy comparison is the probability that B is preferred over A, with ties
scored as one half.

Three assumptions are made explicit. The unit of statistical independence is
the prompt, not the individual comparison, because several annotations of the
same prompt are correlated. Human labels are a noisy signal rather than ground
truth, so the human self consistency rate is treated as an approximate ceiling
on how well any judge can agree with humans. Finally, the A or B position is
arbitrary and must carry no information, so any judge behavior that depends on
ordering is treated as a defect to be measured and removed.

## Judge design

Two judges sit behind one interface. The heuristic judge scores each response
on interpretable signals: overlap with the prompt for relevance, a saturating
count of unique tokens for informativeness, a repetition penalty for non
degeneracy, and a deliberately weak length prior. It needs no network or key,
so the pipeline is fully reproducible, and it is the control that the LLM judge
must beat to justify its cost. The DigitalOcean LLM judge calls DigitalOcean's
OpenAI compatible inference endpoint with a rubric prompt that forces a strict
JSON verdict. It is the platform native choice for this exercise and swaps to
any other endpoint through environment variables.

## Failure modes and how each is handled

Position bias is the most common failure of pairwise judges. Every verdict is
computed in both orders and reconciled. If the verdict flips when the responses
are swapped, the item is scored as a tie and flagged, and the calibration
report surfaces the overall flip rate, so an order dependent verdict is never
trusted.

Length bias is a reward hacking axis. The heuristic length prior is kept weak
by design, and policy comparison additionally reports a length controlled win
rate computed only on prompts where the two responses are of comparable length.
A material gap between the raw and length controlled figures is flagged as
possible length gaming.

A judge that ties everything is caught by comparing human and judge tie rates.
Endpoint failures or a missing key degrade gracefully to the heuristic judge,
and every fallback is counted so the report can disclose how often the LLM was
actually used rather than silently substituting.

## Trust gate

A judge is not used for policy ranking until it clears a gate: Cohen's kappa of
at least 0.40, which is moderate chance corrected agreement, together with a
position bias rate at or below 0.15. Kappa is the headline rather than raw
accuracy, because accuracy is inflated when one label dominates. If the gate
fails the report still runs but labels the verdict low confidence.

The pipeline also measures how well the judge's stated confidence matches its
actual accuracy, reported as a reliability curve and an expected calibration
error, and it reports inter annotator agreement using Fleiss kappa over the
repeated human labels. The Fleiss implementation supports variable rater counts
per item, so items with 2, 3, or 5 annotators all contribute correctly rather
than being silently truncated to the minimum.

## Statistical choices

The primary interval is the BCa (bias-corrected and accelerated) bootstrap over
prompts with ten thousand resamples. It makes no distributional assumption,
handles the one half tie scoring naturally, respects the prompt as the unit of
independence, and corrects for both bias and skew that the plain percentile
bootstrap does not. The percentile bootstrap is retained as a cross-check in
the report because disagreement between the two is itself diagnostic. A Wilson
score interval on decisive items is reported as a closed form cross check,
chosen over the naive normal interval because it behaves well for proportions
near zero or one and at small sample sizes. Significance is a two sided exact
binomial sign test that the decisive win rate differs from one half, exact
rather than normal approximate so it is valid at the small sample sizes typical
of these datasets. A minimum detectable effect is reported so a reader knows the
smallest true difference the test could have caught at the given sample size.

A result is called only when the entire interval sits on one side of one half.
Otherwise the verdict is inconclusive, which is treated as a first class and
honest outcome rather than a failure.

Cohen's kappa is reported with a 95% bootstrap CI. At the sample sizes typical
of judge calibration studies (10-100 items), a bare point estimate is nearly
uninformative, and the 0.4 trust gate is only defensible when the reader sees
how noisy the estimate is. Two flags are surfaced: `trustworthy` when the point
estimate clears the gate, and `trustworthy_with_reserve` when the point clears
the gate but the CI lower bound does not.

Alongside the win rate the pipeline fits a Bradley Terry model over the pairwise
outcomes with a bootstrap CI on the strengths. Bradley Terry is the standard
model for paired comparison data and is the same objective a reward model is
trained on, so it is the natural ranking lens for this problem. For two policies
the BT strength ratio is algebraically equivalent to the raw win rate, so the
point estimate alone adds no information; the bootstrap CI on strengths turns it
into a real ranking-stability estimate and scaffolds the multi-policy case (3+
policies where transitivity matters) without changing the API. Ties are entered
as half wins, which is a simple and common treatment; Davidson's extension
models ties explicitly and is the next step.

## Reward hacking as a test suite

Reward hacking is treated as a test category, not a paragraph. Two attack
policies (`VerbosityPaddingPolicy`, `SycophancyPolicy`) live in
`src/eval_pipeline/adversarial.py`, and `tests/test_adversarial.py` verifies
that the length control catches verbosity padding and that both-order averaging
suppresses spurious position bias from a sycophantic prefix. The current
heuristic judge is documented (via a passing test) as exploitable by sycophancy
on prompt-overlap, which is the exact motivation for gating any policy verdict
behind the calibration trust check and preferring the length-controlled figure.

## LLM verdict caching

Calibration, judge comparison, and policy comparison all issue the same
`(model, prompt, response_a, response_b)` requests. A `Cache` protocol with a
JSON file backend (default) and a Valkey adapter (production) is passed into
the LLM judge so each unique triple is judged once, then reused. The report
surfaces hits, misses, writes, and hit rate so an operator can see how much of
the evaluation cost was avoided.

## Known limitations

The heuristic judge is a stand in, and on real data a calibrated LLM judge will
usually agree with humans better, which is why the code is built to slot it in.
The bootstrap over prompts assumes prompts are exchangeable, so strong topic
clustering would call for a stratified bootstrap. Ties are scored at one half,
which is conventional but a modeling choice, and a tie aware ordinal model such
as Bradley Terry with ties is the natural extension.
