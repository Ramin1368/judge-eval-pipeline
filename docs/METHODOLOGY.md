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
repeated human labels.

## Statistical choices

The primary interval is a percentile bootstrap over prompts with ten thousand
resamples. It makes no distributional assumption, handles the one half tie
scoring naturally, and respects the prompt as the unit of independence. A Wilson
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

## Known limitations

The heuristic judge is a stand in, and on real data a calibrated LLM judge will
usually agree with humans better, which is why the code is built to slot it in.
The bootstrap over prompts assumes prompts are exchangeable, so strong topic
clustering would call for a stratified bootstrap. Ties are scored at one half,
which is conventional but a modeling choice, and a tie aware ordinal model such
as Bradley Terry with ties is the natural extension.
