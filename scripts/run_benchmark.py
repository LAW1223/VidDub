"""Run VidDub once over the frozen same/cross VidDub-Bench protocol.

The model is loaded once and reused for all selected items. Results are written
to ``same/<key>.mp4`` and ``cross/<key>.mp4`` with a resumable JSONL receipt.
"""
import argparse
import json
import os
from pathlib import Path

from infer import infer_file, load_inference_runtime
from validate_benchmark import read_jsonl, validate


PROJECT = Path(__file__).resolve().parents[1]


def append_receipt(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Downloaded VidDub-Bench directory")
    parser.add_argument("--model", default="AIPeanutman/VidDub",
                        help="Local model folder or Hugging Face repo ID")
    parser.add_argument("--revision")
    parser.add_argument(
        "--weight-root", type=Path,
        default=Path(os.environ.get("VIDDUB_WEIGHT_ROOT", PROJECT / "weights")),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("same", "cross", "both"),
                        default="both")
    parser.add_argument("--language", choices=("all", "en", "zh"), default="all")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start within the selected language rows")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip package hashes after a prior successful validation")
    args = parser.parse_args()

    benchmark = args.benchmark.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not args.skip_validation:
        validate(benchmark, checksums=True, media=False)
    items = read_jsonl(benchmark / "manifest/items.jsonl")
    if args.language != "all":
        items = [row for row in items if row["language"] == args.language]
    stop = None if args.limit is None else args.start_index + args.limit
    items = items[args.start_index:stop]
    if not items:
        raise ValueError("No benchmark rows selected")

    conditions = ("same", "cross") if args.condition == "both" else (args.condition,)
    receipt_path = output / "results.jsonl"
    completed = set()
    if receipt_path.is_file() and not args.overwrite:
        completed = {
            (row["key"], row["condition"])
            for row in read_jsonl(receipt_path)
            if row.get("status") == "complete"
        }

    runtime = load_inference_runtime(
        args.model,
        args.weight_root,
        device=args.device,
        revision=args.revision,
    )
    total = len(items) * len(conditions)
    position = 0
    for row in items:
        for condition in conditions:
            position += 1
            identity = (row["key"], condition)
            destination = output / condition / f"{row['key']}.mp4"
            if (
                not args.overwrite
                and destination.is_file()
                and identity in completed
            ):
                print(f"[{position}/{total}] skip {condition} {row['key']}")
                continue
            audio_field = "audio_same" if condition == "same" else "audio_cross"
            print(f"[{position}/{total}] run {condition} {row['key']}", flush=True)
            infer_file(
                runtime,
                benchmark / row["video"],
                benchmark / row[audio_field],
                destination,
                seed=int(row["seed"]),
                mode="clip",
                overlap=17,
            )
            append_receipt(receipt_path, {
                "key": row["key"],
                "language": row["language"],
                "condition": condition,
                "cross_key": row["cross_key"] if condition == "cross" else None,
                "seed": int(row["seed"]),
                "source_video": row["video"],
                "target_audio": row[audio_field],
                "output": destination.relative_to(output).as_posix(),
                "model": args.model,
                "revision": args.revision,
                "status": "complete",
            })

    print(f"VidDub-Bench run complete: {output}")


if __name__ == "__main__":
    main()
