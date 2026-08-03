import csv
from pathlib import Path

from eval_pipeline.ingestion import load_preferences
from eval_pipeline.schemas import Preference


def _write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_loads_with_alias_columns(tmp_path):
    p = tmp_path / "prefs.csv"
    _write_csv(p, [
        {"question": "Q1", "answer_a": "a", "answer_b": "b", "winner": "response_a"},
    ], ["question", "answer_a", "answer_b", "winner"])
    exs, errors = load_preferences(p)
    assert errors == []
    assert len(exs) == 1
    assert exs[0].prompt == "Q1"
    assert exs[0].preferred is Preference.A


def test_bad_rows_are_collected_not_raised(tmp_path):
    p = tmp_path / "prefs.csv"
    _write_csv(p, [
        {"prompt": "Q1", "response_a": "a", "response_b": "b", "preferred": "A"},
        {"prompt": "Q2", "response_a": "a", "response_b": "b", "preferred": "banana"},
    ], ["prompt", "response_a", "response_b", "preferred"])
    exs, errors = load_preferences(p)
    assert len(exs) == 1
    assert len(errors) == 1


def test_jsonl_ingestion(tmp_path):
    p = tmp_path / "prefs.jsonl"
    p.write_text(
        '{"prompt":"Q","response_a":"a","response_b":"b","preferred":"B"}\n',
        encoding="utf-8",
    )
    exs, errors = load_preferences(p)
    assert errors == []
    assert exs[0].preferred is Preference.B
