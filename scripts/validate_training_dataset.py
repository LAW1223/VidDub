"""Validate a staged VidDub-Training-Data package before publication."""
import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED = {
    "sources": 45_221,
    "clips": 85_435,
    "source_languages": {"en": 27_630, "zh": 17_591},
    "clip_languages": {"en": 52_658, "zh": 32_777},
    "logical_source_bytes": 226_805_933_478,
}


def read_jsonl(path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe artifact path: {value}")
    return path


def validate(root, manifests_only=False, public=False):
    required = (
        "README.md",
        "LICENSE",
        "manifest/release.json",
        "manifest/s00_manifest.jsonl",
        "manifest/release_selection.jsonl",
        "manifest/s01_reference.jsonl",
        "manifest/quality_selected.jsonl",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)

    release = json.loads((root / "manifest/release.json").read_text())
    if release.get("schema") != 1 or release.get("name") != "VidDub-Training-Data":
        raise ValueError("unsupported training-data release")
    if release.get("license") != "CC-BY-NC-4.0":
        raise ValueError("training-data license must be CC-BY-NC-4.0")
    if release.get("paper_selection") != "weights_v1":
        raise ValueError("release is not the frozen paper selection")
    if release.get("synthetic_identities") is not True:
        raise ValueError("release must identify its media as synthetic")
    if release.get("source_disjoint_from_benchmark") is not True:
        raise ValueError("training sources must remain disjoint from VidDub-Bench")
    if public and release.get("uploaded") is not True:
        raise ValueError("public release.json must set uploaded=true")

    source_ids = set()
    source_languages = Counter()
    source_paths = {}
    source_bytes = 0
    for row in read_jsonl(root / "manifest/s00_manifest.jsonl"):
        source_id = row.get("id")
        if not source_id or source_id in source_ids:
            raise ValueError(f"invalid or duplicate source id: {source_id}")
        source_ids.add(source_id)
        language = row.get("language")
        if language not in ("en", "zh"):
            raise ValueError(f"{source_id}: unsupported language: {language}")
        source_languages[language] += 1
        relative = safe_relative(row.get("video_path", ""))
        expected = PurePosixPath(
            "videos", source_id[-6:-3], f"{source_id}.mp4")
        if relative != expected:
            raise ValueError(f"{source_id}: unexpected video path: {relative}")
        source_paths[source_id] = relative
        if (
            row.get("original_width") != 1344
            or row.get("original_height") != 768
            or float(row.get("original_fps", 0)) != 24.0
        ):
            raise ValueError(f"{source_id}: unexpected source media contract")
        if not manifests_only:
            media = root.joinpath(*relative.parts)
            if not media.is_file():
                raise FileNotFoundError(media)
            source_bytes += media.stat().st_size

    selected_ids = set()
    selected_sources = Counter()
    clip_languages = Counter()
    for row in read_jsonl(root / "manifest/release_selection.jsonl"):
        clip_id = row.get("clip_id")
        source_id = row.get("id")
        if not clip_id or clip_id in selected_ids:
            raise ValueError(f"invalid or duplicate clip id: {clip_id}")
        if source_id not in source_ids:
            raise ValueError(f"{clip_id}: missing source: {source_id}")
        selected_ids.add(clip_id)
        selected_sources[source_id] += 1
        language = row.get("lang")
        if language not in ("en", "zh"):
            raise ValueError(f"{clip_id}: unsupported language: {language}")
        clip_languages[language] += 1
        expected_dir = f"{source_id[-6:-3]}/{clip_id}"
        if row.get("dir") != expected_dir:
            raise ValueError(f"{clip_id}: unexpected derived clip path")

    reference_ids = {
        row.get("id")
        for row in read_jsonl(root / "manifest/s01_reference.jsonl")
    }
    quality_ids = {
        row.get("clip_id")
        for row in read_jsonl(root / "manifest/quality_selected.jsonl")
    }
    if reference_ids != source_ids:
        raise ValueError("s01 reference ids do not match source ids")
    if quality_ids != selected_ids:
        raise ValueError("quality ids do not match selected clip ids")
    if set(selected_sources) != source_ids:
        raise ValueError("every released source must contribute a selected clip")
    source_multiplicity = Counter(selected_sources.values())
    if source_multiplicity != Counter({2: 40_214, 1: 5_007}):
        raise ValueError(f"unexpected selected windows per source: {source_multiplicity}")

    if len(source_ids) != EXPECTED["sources"]:
        raise ValueError(f"unexpected source count: {len(source_ids)}")
    if len(selected_ids) != EXPECTED["clips"]:
        raise ValueError(f"unexpected selected clip count: {len(selected_ids)}")
    if dict(source_languages) != EXPECTED["source_languages"]:
        raise ValueError(f"unexpected source languages: {source_languages}")
    if dict(clip_languages) != EXPECTED["clip_languages"]:
        raise ValueError(f"unexpected clip languages: {clip_languages}")
    if not manifests_only and source_bytes != EXPECTED["logical_source_bytes"]:
        raise ValueError(f"unexpected logical source bytes: {source_bytes}")

    contract_counts = (
        release.get("source_videos"),
        release.get("selected_clips"),
        release.get("logical_source_bytes"),
    )
    expected_counts = (
        EXPECTED["sources"],
        EXPECTED["clips"],
        EXPECTED["logical_source_bytes"],
    )
    if contract_counts != expected_counts:
        raise ValueError("release.json counts do not match the frozen contract")
    return len(source_ids), len(selected_ids), source_languages, clip_languages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifests-only", action="store_true",
        help="Skip 45,221 media stat calls; use for card/manifest staging")
    parser.add_argument(
        "--public", action="store_true",
        help="Require release.json uploaded=true for the final public package")
    args = parser.parse_args()
    result = validate(
        args.root.expanduser().resolve(),
        manifests_only=args.manifests_only,
        public=args.public,
    )
    sources, clips, source_languages, clip_languages = result
    print(
        "VidDub training-data validation passed: "
        f"{sources} sources {dict(source_languages)}, "
        f"{clips} clips {dict(clip_languages)}")


if __name__ == "__main__":
    main()
