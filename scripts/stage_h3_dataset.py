"""Stage the paper-selected H3 training set from the research workspace.

The release stores original-resolution source videos plus the exact clip-level
selection used by the paper. It intentionally does not copy the derived
512x512 training clips or model-specific latent caches.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


EXPECTED_CLIPS = 85_435
EXPECTED_SOURCES = 45_221
SCHEMA_VERSION = 1


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_shards(directory, suffix=".jsonl"):
    for path in sorted(Path(directory).glob(f"shard_*{suffix}")):
        yield from read_jsonl(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_release(legacy_root):
    legacy_root = Path(legacy_root).resolve()
    manifest = legacy_root / "data/manifest"
    weights_path = manifest / "quality/weights_v1.json"
    weights = json.loads(weights_path.read_text())

    selected = []
    all_by_source = Counter()
    for row in read_shards(manifest / "s02"):
        if "error" in row:
            continue
        all_by_source[row["id"]] += 1
        if row["clip_id"] in weights:
            selected.append(dict(row, sample_weight=float(weights[row["clip_id"]])))
    selected.sort(key=lambda row: row["clip_id"])
    selected_ids = {row["clip_id"] for row in selected}
    source_ids = {row["id"] for row in selected}

    h3_weight_ids = {clip_id for clip_id in weights if clip_id.startswith("qwen35_talking_")}
    if selected_ids != h3_weight_ids:
        raise ValueError(
            "H3 s02 rows do not match weights_v1: "
            f"missing={len(h3_weight_ids - selected_ids)}, "
            f"extra={len(selected_ids - h3_weight_ids)}"
        )
    if len(selected) != EXPECTED_CLIPS or len(source_ids) != EXPECTED_SOURCES:
        raise ValueError(
            f"Unexpected paper selection: {len(selected)} clips from "
            f"{len(source_ids)} sources; expected {EXPECTED_CLIPS}/{EXPECTED_SOURCES}"
        )

    source_rows = {
        row["id"]: row
        for row in read_jsonl(manifest / "s00_manifest.jsonl")
        if row["id"] in source_ids
    }
    analysis_rows = {
        row["id"]: row for row in read_shards(manifest / "s01") if row["id"] in source_ids
    }
    if source_ids != set(source_rows) or source_ids != set(analysis_rows):
        raise ValueError(
            f"Missing metadata: source={len(source_ids - set(source_rows))}, "
            f"analysis={len(source_ids - set(analysis_rows))}"
        )

    resolutions = Counter(
        (int(analysis_rows[source_id]["width"]), int(analysis_rows[source_id]["height"]))
        for source_id in source_ids
    )
    derived_resolution = (512, 512)
    if derived_resolution in resolutions:
        raise ValueError(
            "Release inputs contain 512x512 media. Stage original videos, not processed clips."
        )

    quality = {}
    for path in sorted((manifest / "quality").glob("rank_*.jsonl")):
        for row in read_jsonl(path):
            if row.get("clip_id") in selected_ids:
                quality[row["clip_id"]] = row
    if len(quality) != len(selected):
        raise ValueError(
            f"Missing quality rows: have {len(quality)}, expected {len(selected)}"
        )

    selected_per_source = Counter(row["id"] for row in selected)
    partial_sources = sum(
        selected_per_source[source_id] != all_by_source[source_id]
        for source_id in source_ids
    )
    return {
        "legacy_root": legacy_root,
        "manifest_root": manifest,
        "selected": selected,
        "source_rows": source_rows,
        "analysis_rows": analysis_rows,
        "quality": quality,
        "resolutions": resolutions,
        "partial_sources": partial_sources,
    }


def media_destination(output, source_id):
    return Path(output) / "videos" / source_id[-6:-3] / f"{source_id}.mp4"


def materialize(source, destination, mode, source_size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.stat().st_size != source_size:
            raise ValueError(f"Existing destination has wrong size: {destination}")
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(mode)


def dataset_readme(release):
    expected = {
        "selected": EXPECTED_CLIPS,
        "sources": EXPECTED_SOURCES,
        "resolutions": {(1344, 768)},
    }
    actual = {
        "selected": len(release["selected"]),
        "sources": len(release["source_rows"]),
        "resolutions": set(release["resolutions"]),
    }
    if actual != expected:
        raise ValueError(f"Dataset-card contract mismatch: {actual}")
    template = (
        Path(__file__).resolve().parents[1] / "data/TRAINING_DATA_CARD.md"
    )
    return template.read_text()


def stage(release, output, media_mode, checksums, workers):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = []
    checksum_rows = []
    total_bytes = 0

    def stage_source(source_id):
        row = release["source_rows"][source_id]
        analysis = release["analysis_rows"][source_id]
        source = Path(row["video_path"])
        try:
            source_stat = source.stat()
        except FileNotFoundError:
            raise FileNotFoundError(source)
        destination = media_destination(output, source_id)
        if media_mode != "manifest-only":
            materialize(source, destination, media_mode, source_stat.st_size)
        size = source_stat.st_size
        relative_video = destination.relative_to(output).as_posix()
        normalized = {
            key: value for key, value in row.items() if key != "video_path"
        }
        normalized.update(
            {
                "video_path": relative_video,
                "language": row.get("speech_language", row.get("language", "")),
                "original_width": int(analysis["width"]),
                "original_height": int(analysis["height"]),
                "original_fps": float(analysis["fps"]),
                "original_num_frames": int(analysis["n_frames"]),
                "audio_duration": float(analysis["dur_audio"]),
            }
        )
        if checksums:
            digest = sha256(source)
            normalized["sha256"] = digest
            normalized["bytes"] = size
            checksum_row = f"{digest}  {relative_video}"
        else:
            checksum_row = None
        return normalized, checksum_row, size

    source_ids = sorted(release["source_rows"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(stage_source, source_ids)
        for index, (normalized, checksum_row, size) in enumerate(results):
            source_manifest.append(normalized)
            total_bytes += size
            if checksum_row is not None:
                checksum_rows.append(checksum_row)
            if (index + 1) % 1000 == 0:
                print(f"staged {index + 1}/{len(source_ids)}", flush=True)

    selection_rows = list(release["selected"])
    quality_rows = [
        release["quality"][row["clip_id"]] for row in release["selected"]
    ]
    analysis_rows = [
        release["analysis_rows"][source_id] for source_id in sorted(release["source_rows"])
    ]
    width, height = next(iter(release["resolutions"]))
    contract = {
        "schema": SCHEMA_VERSION,
        "name": "VidDub-Training-Data",
        "license": "CC-BY-NC-4.0",
        "repository": (
            "https://huggingface.co/datasets/AIPeanutman/"
            "VidDub-Training-Data"
        ),
        "paper": "https://arxiv.org/abs/2611.03404",
        "paper_selection": "weights_v1",
        "source_videos": len(source_rows),
        "selected_clips": len(selection_rows),
        "languages": {
            "source_videos": dict(Counter(
                row["language"] for row in source_manifest)),
            "selected_clips": dict(Counter(
                row["lang"] for row in selection_rows)),
        },
        "logical_source_bytes": total_bytes,
        "source_resolution": f"{width}x{height}",
        "derived_clip": {
            "frames": 81,
            "fps": 24,
            "width": 512,
            "height": 512,
            "audio_rate": 16000,
        },
        "selection_filter": {
            "iqa_min": 40,
            "sync_offset_max": 4,
            "drop_fraction": 0.2,
        },
        "synthetic_identities": True,
        "source_disjoint_from_benchmark": True,
        "uploaded": False,
    }

    write_jsonl(output / "manifest/s00_manifest.jsonl", source_manifest)
    write_jsonl(output / "manifest/release_selection.jsonl", selection_rows)
    write_jsonl(output / "manifest/s01_reference.jsonl", analysis_rows)
    write_jsonl(output / "manifest/quality_selected.jsonl", quality_rows)
    write_json(output / "manifest/release.json", contract)
    if checksums:
        checksum_path = output / "manifest/checksums.sha256"
        checksum_path.write_text("\n".join(checksum_rows) + "\n")
    (output / "README.md").write_text(dataset_readme(release))
    shutil.copy2(
        Path(__file__).resolve().parents[1]
        / "dependencies/licenses/CC-BY-NC-4.0.txt",
        output / "LICENSE",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--media-mode",
        choices=("manifest-only", "copy", "hardlink", "symlink"),
        default="manifest-only",
    )
    parser.add_argument("--checksums", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    release = collect_release(args.legacy_root)
    summary = {
        "selected_clips": len(release["selected"]),
        "source_videos": len(release["source_rows"]),
        "partial_sources": release["partial_sources"],
        "resolutions": {
            f"{width}x{height}": count
            for (width, height), count in sorted(release["resolutions"].items())
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.validate_only:
        return
    if not args.output:
        parser.error("--output is required unless --validate-only is used")
    stage(release, args.output, args.media_mode, args.checksums, args.workers)


if __name__ == "__main__":
    main()
