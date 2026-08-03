# Policy Evaluation Report

## For a non-research stakeholder

**Result: 'policy_b' is the better policy.** On the held out set, 'policy_b' wins 66.2% of head to head comparisons against 'policy_a' (95 percent CI [55.0%, 76.2%], p = 0.0049). Because the whole interval sits on one side of 50 percent, the result is statistically meaningful, not noise.

How much should you trust this? The automated judge was checked against real human preferences before we used it, and it rated **TRUSTWORTHY** (chance corrected agreement, Cohen's kappa, of 1.00). Treat the verdict above as reliable.

Gaming check: when we restrict to prompts where both answers were similar in length, the win rate is 66.2% (vs 66.2% overall), so the result is not merely a length effect.

## Judge calibration

Judge: heuristic_v1
Labeled items evaluated: 10
Raw agreement with humans (accuracy): 100.0%
Cohen's kappa (chance corrected): 1.000 (trust gate at least 0.40)
Position bias rate (verdict flips on A/B swap): 0.0%
Expected calibration error (confidence vs correctness): 0.330
Tie rate, human vs judge: 0.0% vs 0.0%
Per slice accuracy:
    human_decisive: 100.0% (n=10)

## Judge comparison

| judge | n | accuracy | kappa | position bias | ECE | trustworthy |
|---|---|---|---|---|---|---|
| heuristic_v1 | 10 | 100.0% | 1.000 | 0.0% | 0.330 | True |

## Policy comparison

Policies: A = policy_a, B = policy_b
Held out prompts: 80
Wins: A=27, B=53, ties or unstable=0
Win rate for B (ties count as 0.5): 66.2%
95 percent CI: [55.0%, 76.2%] (percentile bootstrap over prompts, 10k resamples, 95 percent)
Significance: p = 0.0049 (two sided exact binomial sign test vs 0.5)
Length controlled win rate for B: 66.2%
Bradley-Terry strengths: policy_a=0.717, policy_b=1.395, implied P(B preferred)=66.0%

## Data quality

Input rows: 62, unique comparisons kept: 10, quarantined: 0
Duplicate comparison groups: 10
Order swapped groups: 10
Contradictory groups resolved by majority: 7
Human self consistency: 85.5%
Inter annotator agreement (Fleiss kappa): 0.493
    contradictory labels for prompt='Explain what a load balancer does.': {A:6, B:2} resolved by majority
    contradictory labels for prompt='How do I create a Droplet on DigitalOcea': {B:5, A:1} resolved by majority
    contradictory labels for prompt='What is a reward model in RLHF?': {A:5, B:1} resolved by majority
    contradictory labels for prompt='Summarize the benefits of managed databa': {A:1, B:5} resolved by majority
    contradictory labels for prompt='What is Kubernetes used for?': {A:2, B:4} resolved by majority
    contradictory labels for prompt='How does autoscaling work?': {A:5, B:1} resolved by majority
    contradictory labels for prompt='What is a vector database?': {A:5, B:1} resolved by majority
