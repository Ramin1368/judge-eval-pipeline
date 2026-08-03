from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Iterable
from .schemas import Preference, PreferenceExample
_ALIASES = {'prompt': {'prompt', 'question', 'input', 'instruction'}, 'response_a': {'response_a', 'response a', 'a', 'answer_a', 'completion_a', 'output_a'}, 'response_b': {'response_b', 'response b', 'b', 'answer_b', 'completion_b', 'output_b'}, 'preferred': {'preferred', 'label', 'winner', 'human_label', 'choice', 'preference'}, 'annotator_id': {'annotator_id', 'annotator', 'rater_id', 'worker_id'}, 'example_id': {'example_id', 'id', 'row_id', 'uid'}}

def _canonical(colname: str) -> str | None:
    c = colname.strip().lower()
    for canon, aliases in _ALIASES.items():
        if c in aliases:
            return canon
    return None

def _row_to_example(row: dict, idx: int) -> PreferenceExample:
    mapped: dict[str, str] = {}
    for raw_col, value in row.items():
        if raw_col is None:
            continue
        canon = _canonical(raw_col)
        if canon:
            mapped[canon] = value
    missing = [f for f in ('prompt', 'response_a', 'response_b', 'preferred') if f not in mapped or mapped[f] is None]
    if missing:
        raise ValueError(f'row {idx}: missing required field(s) {missing}')
    return PreferenceExample(prompt=str(mapped['prompt']).strip(), response_a=str(mapped['response_a']).strip(), response_b=str(mapped['response_b']).strip(), preferred=Preference.parse(mapped['preferred']), example_id=str(mapped.get('example_id') or f'ex_{idx}'), annotator_id=str(mapped['annotator_id']) if mapped.get('annotator_id') else None)

def load_preferences(path: str | Path) -> tuple[list[PreferenceExample], list[str]]:
    path = Path(path)
    rows: Iterable[dict]
    if path.suffix.lower() in {'.jsonl', '.ndjson'}:
        rows = _read_jsonl(path)
    elif path.suffix.lower() == '.json':
        rows = json.loads(path.read_text())
    else:
        rows = _read_csv(path)
    examples: list[PreferenceExample] = []
    errors: list[str] = []
    for idx, row in enumerate(rows):
        try:
            examples.append(_row_to_example(row, idx))
        except Exception as exc:
            errors.append(str(exc))
    return (examples, errors)

def load_policy_outputs(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() in {'.jsonl', '.ndjson'}:
        return list(_read_jsonl(path))
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text())
    return list(_read_csv(path))

def _read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
