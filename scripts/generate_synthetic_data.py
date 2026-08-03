from __future__ import annotations
import csv
import random
from pathlib import Path
SEED = 7
DATA = Path(__file__).resolve().parents[1] / 'data'
PROMPTS = [('Explain what a load balancer does.', 'load balancer', 'traffic'), ('How do I create a Droplet on DigitalOcean?', 'Droplet', 'control panel'), ('What is a reward model in RLHF?', 'reward model', 'preferences'), ('Summarize the benefits of managed databases.', 'managed databases', 'backups'), ('Describe how object storage differs from block storage.', 'object storage', 'block storage'), ('What is Kubernetes used for?', 'Kubernetes', 'containers'), ('How does autoscaling work?', 'autoscaling', 'load'), ('Explain preference optimization in one paragraph.', 'preference optimization', 'policy'), ('What is a vector database?', 'vector database', 'embeddings'), ('How do I set up CI/CD for a small service?', 'CI/CD', 'pipeline'), ('What is a webhook and when do I use one?', 'webhook', 'events'), ('How does horizontal scaling differ from vertical scaling?', 'horizontal scaling', 'vertical scaling'), ('Explain when to use a message queue.', 'message queue', 'producers'), ('What is DNS and how does resolution work?', 'DNS', 'records'), ('Describe idempotency in HTTP APIs.', 'idempotency', 'requests')]

def _good(prompt: str, kw1: str, kw2: str, rng: random.Random) -> str:
    return f'A {kw1} routes incoming {kw2} across multiple backends and keeps a health check on each one, so failed hosts are removed from rotation and users see steady latency even under uneven load.'

def _bad_distractor(prompt: str, kw1: str, kw2: str, rng: random.Random) -> str:
    return f'When people ask about {kw1} and {kw2}, the answer depends on many factors like scale, cost, and team preference, and there is no single right choice without more context about the specific use case here.'

def _bad_wrong(prompt: str, kw1: str, kw2: str, rng: random.Random) -> str:
    return f'A {kw1} is a special kind of {kw2} database that stores its state on a single machine and never talks to other systems, which is why it is used in place of caching and message-passing infrastructure.'

def _bad_variants(prompt: str, kw1: str, kw2: str, rng: random.Random) -> str:
    return _bad_distractor(prompt, kw1, kw2, rng) if rng.random() < 0.6 else _bad_wrong(prompt, kw1, kw2, rng)

def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    ex = 0
    ambig_prompt_idx = {2, 7}
    for pi, (p, kw1, kw2) in enumerate(PROMPTS):
        for _ in range(3):
            good_is_a = rng.random() < 0.5
            good_text = _good(p, kw1, kw2, rng)
            bad_text = _bad_variants(p, kw1, kw2, rng)
            a = good_text if good_is_a else bad_text
            b = bad_text if good_is_a else good_text
            true_pref = 'A' if good_is_a else 'B'
            noisy = true_pref if rng.random() < 0.9 else 'B' if true_pref == 'A' else 'A'
            rows.append({'example_id': f'ex_{ex}', 'prompt': p, 'response_a': a, 'response_b': b, 'preferred': noisy, 'annotator_id': f'ann_{rng.randint(1, 4)}'})
            ex += 1
        if pi in ambig_prompt_idx:
            good_text = _good(p, kw1, kw2, rng)
            bad_text = _bad_variants(p, kw1, kw2, rng)
            for label in ('A', 'B', 'tie'):
                rows.append({'example_id': f'ex_{ex}', 'prompt': p + ' (ambiguous)', 'response_a': good_text, 'response_b': bad_text, 'preferred': label, 'annotator_id': f'ann_amb_{label}'})
                ex += 1
    first = rows[0]
    rows.append({'example_id': f'ex_{ex}', 'prompt': first['prompt'], 'response_a': first['response_b'], 'response_b': first['response_a'], 'preferred': first['preferred'], 'annotator_id': 'ann_x'})
    ex += 1
    rows.append(dict(first, example_id=f'ex_{ex}'))
    ex += 1
    with (DATA / 'preferences.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pol_rows = []
    for i in range(80):
        base, kw1, kw2 = PROMPTS[i % len(PROMPTS)]
        p_text = f'{base} (case {i})'
        b_better = rng.random() < 0.65
        pol_rows.append({'prompt': p_text, 'policy_a': _bad_variants(base, kw1, kw2, rng) if b_better else _good(base, kw1, kw2, rng), 'policy_b': _good(base, kw1, kw2, rng) if b_better else _bad_variants(base, kw1, kw2, rng)})
    with (DATA / 'policy_outputs.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['prompt', 'policy_a', 'policy_b'])
        w.writeheader()
        w.writerows(pol_rows)
    print(f'wrote {len(rows)} preference rows and {len(pol_rows)} policy rows to {DATA}')
if __name__ == '__main__':
    main()
