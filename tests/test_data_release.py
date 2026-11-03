import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import common
from data.s02_make_clips import load_selection


def test_release_relative_video_path_uses_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path)
    assert common.resolve_video_path({"video_path": "videos/000/a.mp4"}) == (
        tmp_path / "videos/000/a.mp4"
    )


def test_absolute_video_path_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", tmp_path / "unused")
    source = tmp_path / "source.mp4"
    assert common.resolve_video_path({"video_path": str(source)}) == source


def test_jsonl_selection_is_clip_level(tmp_path):
    path = tmp_path / "selection.jsonl"
    rows = [
        {"id": "source_a", "clip_id": "source_a_000"},
        {"id": "source_a", "clip_id": "source_a_026"},
        {"id": "source_b", "clip_id": "source_b_000"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_selection(path) == {
        "source_a": {"source_a_000", "source_a_026"},
        "source_b": {"source_b_000"},
    }


def test_weight_json_selection_derives_source_id(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"source_a_000": 1.0, "source_b_026": 0.8}))
    assert load_selection(path) == {
        "source_a": {"source_a_000"},
        "source_b": {"source_b_026"},
    }
