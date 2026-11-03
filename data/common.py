"""Paths and JSONL helpers shared by the public data pipeline."""
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VIDDUB_DATA_ROOT", PROJECT / "data"))
MANIFEST = DATA / "manifest"
WEIGHTS = Path(os.environ.get("VIDDUB_WEIGHT_ROOT", PROJECT / "weights"))


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def resolve_video_path(row):
    """Resolve artifact-relative media paths against VIDDUB_DATA_ROOT."""
    path = Path(row["video_path"]).expanduser()
    return path if path.is_absolute() else DATA / path


def shard(rows, shard_id, num_shards):
    return rows[shard_id::num_shards]
