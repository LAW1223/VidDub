"""Validate a downloaded VidDub-Bench package.

The structural and checksum checks use only the Python standard library.
Pass ``--media`` to additionally inspect every audio/video file with ffprobe.
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath


REQUIRED_ITEM_FIELDS = {
    "key",
    "language",
    "video",
    "audio_same",
    "audio_cross",
    "mask",
    "reference",
    "cross_key",
    "seed",
    "fps",
    "frames",
    "seconds",
}
PATH_FIELDS = ("video", "audio_same", "audio_cross", "mask", "reference")


def read_jsonl(path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def safe_path(root, value):
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {value}")
    return root.joinpath(*relative.parts)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checksums(root):
    receipt = root / "manifest/checksums.sha256"
    seen = set()
    for line_number, line in enumerate(receipt.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(None, 1)
        except ValueError as exc:
            raise ValueError(f"{receipt}:{line_number}: malformed checksum") from exc
        relative = relative.lstrip("*")
        path = safe_path(root, relative)
        if relative in seen:
            raise ValueError(f"duplicate checksum entry: {relative}")
        seen.add(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    return seen


def probe_media(root, items):
    video_specs = Counter()
    audio_specs = Counter()
    for row in items:
        video = safe_path(root, row["video"])
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of", "json", str(video),
            ],
            check=True, capture_output=True, text=True,
        )
        stream = json.loads(result.stdout)["streams"][0]
        video_specs[(
            int(stream["width"]),
            int(stream["height"]),
            stream["avg_frame_rate"],
            int(stream["nb_read_frames"]),
        )] += 1

        audio = safe_path(root, row["audio_same"])
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-show_entries", "format=duration", "-of", "json", str(audio),
            ],
            check=True, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        audio_specs[(
            int(stream["sample_rate"]),
            int(stream["channels"]),
            round(float(payload["format"]["duration"]), 3),
        )] += 1

    if video_specs != Counter({(512, 512, "24/1", 81): len(items)}):
        raise ValueError(f"unexpected video specifications: {video_specs}")
    if any(rate != 16000 or channels != 1 for rate, channels, _ in audio_specs):
        raise ValueError(f"unexpected audio specifications: {audio_specs}")


def validate(root, checksums=True, media=False):
    required = (
        "README.md",
        "LICENSE",
        "manifest/items.jsonl",
        "manifest/rights.jsonl",
        "manifest/release.json",
        "manifest/checksums.sha256",
    )
    for relative in required:
        if not (root / relative).is_file():
            raise FileNotFoundError(root / relative)

    release = json.loads((root / "manifest/release.json").read_text())
    items = read_jsonl(root / "manifest/items.jsonl")
    rights = read_jsonl(root / "manifest/rights.jsonl")
    if release.get("name") != "VidDub-Bench" or release.get("schema") != 1:
        raise ValueError("unsupported benchmark release")
    if release.get("license") != "CC-BY-NC-4.0" or release.get("uploaded") is not True:
        raise ValueError("benchmark release metadata is not public-release ready")
    if release.get("source_disjoint_from_training") is not True:
        raise ValueError("benchmark must remain source-disjoint from training")
    manifest_digest = sha256(root / "manifest/items.jsonl")
    if release.get("manifest_sha256") != manifest_digest:
        raise ValueError("release manifest_sha256 does not match items.jsonl")
    if len(items) != release.get("items") or len(rights) != len(items):
        raise ValueError("manifest item counts disagree")

    keys = [row.get("key") for row in items]
    if len(keys) != len(set(keys)):
        raise ValueError("item keys must be unique")
    rights_keys = [row.get("key") for row in rights]
    if len(rights_keys) != len(set(rights_keys)):
        raise ValueError("rights keys must be unique")
    by_key = {row["key"]: row for row in items}
    rights_by_key = {row.get("key"): row for row in rights}
    if set(rights_by_key) != set(keys):
        raise ValueError("rights manifest keys do not match item keys")
    if any(row.get("release_status") != "owner-approved" for row in rights):
        raise ValueError("not every item is approved for release")
    if any(row.get("identity") != "synthetic" for row in rights):
        raise ValueError("not every item is marked as synthetic")
    if any(row.get("license") != "CC-BY-NC-4.0" for row in rights):
        raise ValueError("not every item carries the benchmark license")

    languages = Counter()
    referenced = set()
    cross_keys = []
    for row in items:
        missing = REQUIRED_ITEM_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{row.get('key')}: missing fields: {sorted(missing)}")
        if row["fps"] != 24 or row["frames"] != 81 or row["seconds"] != 3.375:
            raise ValueError(f"{row['key']}: unexpected clip protocol")
        if not isinstance(row["seed"], int):
            raise ValueError(f"{row['key']}: seed must be an integer")
        if row["language"] not in ("en", "zh"):
            raise ValueError(f"{row['key']}: unsupported language")
        languages[row["language"]] += 1
        cross_keys.append(row["cross_key"])
        if row["cross_key"] == row["key"]:
            raise ValueError(f"{row['key']}: self replacement is not allowed")
        for field in PATH_FIELDS:
            path = safe_path(root, row[field])
            if not path.is_file():
                raise FileNotFoundError(path)
            referenced.add(path.relative_to(root).as_posix())

    if dict(languages) != release.get("languages"):
        raise ValueError(
            f"language counts disagree: {dict(languages)} != {release.get('languages')}")
    if set(cross_keys) != set(keys) or len(cross_keys) != len(set(cross_keys)):
        raise ValueError("cross_key assignments must be a permutation of all keys")
    for row in items:
        target = by_key[row["cross_key"]]
        if row["audio_cross"] != target["audio_same"]:
            raise ValueError(f"{row['key']}: audio_cross does not match cross_key")
        if row["language"] != target["language"]:
            raise ValueError(f"{row['key']}: cross_key changes language")

    listed = validate_checksums(root) if checksums else set()
    if checksums:
        expected = referenced | {
            "README.md",
            "LICENSE",
            "manifest/items.jsonl",
            "manifest/rights.jsonl",
            "manifest/release.json",
        }
        if listed != expected:
            missing = sorted(expected - listed)
            extra = sorted(listed - expected)
            raise ValueError(f"checksum coverage differs; missing={missing}, extra={extra}")
    if media:
        probe_media(root, items)
    return len(items), languages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Downloaded VidDub-Bench directory")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--media", action="store_true",
                        help="Inspect every clip with ffprobe (slower)")
    args = parser.parse_args()
    count, languages = validate(
        args.root.expanduser().resolve(),
        checksums=not args.skip_checksums,
        media=args.media,
    )
    print(
        f"VidDub-Bench validation passed: {count} items "
        f"({languages['en']} en, {languages['zh']} zh)")


if __name__ == "__main__":
    main()
