from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 7
DATA = Path(__file__).resolve().parents[1] / "data"

PROMPTS = [
    "Explain what a load balancer does.",
    "How do I create a Droplet on DigitalOcean?",
    "What is a reward model in RLHF?",
    "Summarize the benefits of managed databases.",
    "Describe how object storage differs from block storage.",
    "What is Kubernetes used for?",
    "How does autoscaling work?",
    "Explain preference optimization in one paragraph.",
    "What is a vector database?",
    "How do I set up CI/CD for a small service?",
]


def _good(prompt: str, rng: random.Random) -> str:
    topic = prompt.rstrip("?.").split()[-1]
    return (
        f"A clear answer to '{prompt}' It directly addresses the request, gives a "
        f"concrete example involving {topic}, and stays relevant without padding."
    )


def _bad(prompt: str, rng: random.Random) -> str:
    return (
        "In general it depends on many factors and it is hard to say without "
        "more context, so consider the situation carefully before deciding."
    )


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    rows = []
    ex = 0
    for pi, p in enumerate(PROMPTS):
        good_prefix, bad_prefix = ("1. ", "2. ") if pi % 2 == 0 else ("2. ", "1. ")
        for _ in range(6):
            good_is_a = rng.random() < 0.5
            good_text = good_prefix + _good(p, rng)
            bad_text = bad_prefix + _bad(p, rng)
            a = good_text if good_is_a else bad_text
            b = bad_text if good_is_a else good_text
            true_pref = "A" if good_is_a else "B"
            noisy = true_pref if rng.random() < 0.85 else ("B" if true_pref == "A" else "A")
            rows.append({
                "example_id": f"ex_{ex}",
                "prompt": p,
                "response_a": a,
                "response_b": b,
                "preferred": noisy,
                "annotator_id": f"ann_{rng.randint(1, 4)}",
            })
            ex += 1

    first = rows[0]
    rows.append({
        "example_id": f"ex_{ex}", "prompt": first["prompt"],
        "response_a": first["response_b"], "response_b": first["response_a"],
        "preferred": first["preferred"], "annotator_id": "ann_x",
    })
    ex += 1
    rows.append(dict(first, example_id=f"ex_{ex}"))
    ex += 1

    with (DATA / "preferences.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pol_rows = []
    for i in range(80):
        base = PROMPTS[i % len(PROMPTS)]
        p = f"{base} (case {i})"
        b_better = rng.random() < 0.75
        pol_rows.append({
            "prompt": p,
            "policy_a": _bad(base, rng) if b_better else _good(base, rng),
            "policy_b": _good(base, rng) if b_better else _bad(base, rng),
        })
    with (DATA / "policy_outputs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt", "policy_a", "policy_b"])
        w.writeheader()
        w.writerows(pol_rows)

    print(f"wrote {len(rows)} preference rows and {len(pol_rows)} policy rows to {DATA}")


if __name__ == "__main__":
    main()
