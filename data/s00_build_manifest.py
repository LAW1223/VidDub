"""Normalize a user-owned source manifest into VidDub's s00 schema.

Input JSONL requires ``id`` and ``video_path``. Optional fields such as
``prompt``, ``speech_text``, ``language``, ``license`` and ``source_url`` are
preserved. Relative video paths are resolved against the input manifest.
"""
import argparse
from pathlib import Path

from .common import MANIFEST, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(MANIFEST / "s00_manifest.jsonl"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    rows = []
    seen = set()
    for row in read_jsonl(source):
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Each row needs a non-empty string id")
        if row["id"] in seen:
            raise ValueError(f"Duplicate id: {row['id']}")
        seen.add(row["id"])
        video = Path(row["video_path"])
        if not video.is_absolute():
            video = (source.parent / video).resolve()
        if not args.allow_missing and not video.is_file():
            raise FileNotFoundError(video)
        normalized = dict(row)
        normalized["video_path"] = str(video)
        normalized.setdefault("prompt", "")
        normalized.setdefault("speech_text", "")
        rows.append(normalized)
    write_jsonl(args.output, sorted(rows, key=lambda item: item["id"]))
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
