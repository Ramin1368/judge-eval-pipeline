# Policy Evaluation Report

## For a non-research stakeholder

**Result: 'policy_b' is the better policy.** On the held out set, 'policy_b' wins 65.6% of head to head comparisons against 'policy_a' (95 percent CI [56.2%, 74.4%], p = 0.0026). Because the whole interval sits on one side of 50 percent, the result is statistically meaningful, not noise.

How much should you trust this? The automated judge was checked against real human preferences before we used it, and it rated **TRUSTWORTHY WITH RESERVE** (chance corrected agreement, Cohen's kappa, of 0.43 (95% CI [0.20, 0.68])). Treat the verdict above as directional; the kappa lower bound sits under the trust gate, so the same judge on a different sample might not pass calibration.

Gaming check: when we restrict to prompts where both answers were similar in length, the win rate is 65.6% (vs 65.6% overall), so the result is not merely a length effect.

## Judge calibration

Judge: heuristic_v1
Labeled items evaluated: 22
Raw agreement with humans (accuracy): 63.6%
Cohen's kappa: 0.430 (95% CI [0.201, 0.676], trust gate at least 0.40; 'with reserve' means point clears the gate but CI does not)
Position bias rate (verdict flips on A/B swap): 0.0%
Expected calibration error (confidence vs correctness): 0.098
Tie rate, human vs judge: 0.0% vs 31.8%
Per slice accuracy:
    human_decisive: 63.6% (n=22)
Note: kappa point estimate 0.430 clears the 0.4 gate, but its 95 percent lower bound is 0.201, so agreement may fall below the gate on a different sample; the trust flag is 'trustworthy with reserve'

## Judge comparison

| judge | n | accuracy | kappa | position bias | ECE | trustworthy |
|---|---|---|---|---|---|---|
| heuristic_v1 | 22 | 63.6% | 0.430 | 0.0% | 0.098 | True |

## Policy comparison

Policies: A = policy_a, B = policy_b
Held out prompts: 80
Wins: A=20, B=45, ties or unstable=15
Win rate for B (ties count as 0.5): 65.6%
95 percent CI: [56.2%, 74.4%] (BCa bootstrap over prompts, 10k resamples, 95 percent)
Percentile bootstrap cross-check: [56.2%, 75.0%]
Significance: p = 0.0026 (two sided exact binomial sign test vs 0.5)
Minimum detectable effect (alpha=0.05, power=0.80): 15.7% absolute shift from 0.5; results smaller than this cannot be reliably detected at this n
Length controlled win rate for B: 65.6%
Bradley-Terry strengths: policy_a=0.727, policy_b=1.376, implied P(B preferred)=65.4% (95% CI [56.2%, 74.1%])
  Note: for two policies BT reduces to the win rate; the model earns its keep at 3+ policies and by providing a bootstrap CI on strengths for ranking stability.

## Data quality

Input rows: 53, unique comparisons kept: 22, quarantined: 3
Duplicate comparison groups: 17
Order swapped groups: 9
Contradictory groups resolved by majority: 4
Human self consistency: 86.7%
Inter annotator agreement (Fleiss kappa, variable rater count aware): 0.575
    contradictory labels for prompt='Explain what a load balancer does.': {A:4, B:1} resolved by majority
    contradictory labels for prompt='What is a reward model in RLHF? (ambiguo': {B:1, A:1, tie:1} resolved by majority
    unresolvable split {B:1, A:1, tie:1} for prompt='What is a reward model in RLHF? (ambiguo' quarantined from calibration
    contradictory labels for prompt='Describe how object storage differs from': {A:1, B:1} resolved by majority
    unresolvable split {A:1, B:1} for prompt='Describe how object storage differs from' quarantined from calibration
    contradictory labels for prompt='Explain preference optimization in one p': {B:1, A:1, tie:1} resolved by majority
    unresolvable split {B:1, A:1, tie:1} for prompt='Explain preference optimization in one p' quarantined from calibration
