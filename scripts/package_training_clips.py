"""Package the frozen VidDub training clips into deterministic ZIP shards."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


MEDIA_FILES = ("video.mp4", "audio.wav", "mask.png", "ref.png")
EXPECTED_CLIPS = 85_435
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
ROOT = Path(__file__).resolve().parents[1]


def read_selection(path):
    rows = [
        json.loads(line)
        for line in Path(path).open()
        if line.strip()
    ]
    rows.sort(key=lambda row: row["clip_id"])
    clip_ids = {row["clip_id"] for row in rows}
    if len(clip_ids) != len(rows):
        raise ValueError("selection contains duplicate clip IDs")
    for row in rows:
        relative = PurePosixPath(row["dir"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe clip directory: {row['dir']}")
        if relative.name != row["clip_id"]:
            raise ValueError(
                f"clip directory does not match ID: {row['dir']} / {row['clip_id']}"
            )
    return rows


def zip_info(name, size):
    info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = size
    info.external_attr = 0o100644 << 16
    return info


def add_file(archive, source, member):
    size = source.stat().st_size
    with source.open("rb") as input_file:
        with archive.open(zip_info(member, size), "w", force_zip64=True) as output_file:
            shutil.copyfileobj(input_file, output_file, length=16 * 1024 * 1024)
    return size


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def package_shard(index, rows, clips_root, output):
    archive_name = f"clips-{index:05d}.zip"
    archive_path = output / "zips" / archive_name
    checksum_path = archive_path.with_suffix(".zip.sha256")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.is_file() and checksum_path.is_file():
        expected = checksum_path.read_text().split()[0]
        actual = sha256(archive_path)
        if actual == expected:
            return {
                "index": index,
                "archive": f"zips/{archive_name}",
                "clips": len(rows),
                "first_clip": rows[0]["clip_id"],
                "last_clip": rows[-1]["clip_id"],
                "media_bytes": None,
                "archive_bytes": archive_path.stat().st_size,
                "sha256": actual,
                "resumed": True,
            }

    temporary = archive_path.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    media_bytes = 0
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            manifest = "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ).encode()
            archive.writestr(
                zip_info(f"manifest/shard-items-{index:05d}.jsonl", len(manifest)),
                manifest,
            )
            for row in rows:
                source_root = clips_root.joinpath(*PurePosixPath(row["dir"]).parts)
                for filename in MEDIA_FILES:
                    source = source_root / filename
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    member = f"clips/{row['dir']}/{filename}"
                    media_bytes += add_file(archive, source, member)
        temporary.replace(archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    digest = sha256(archive_path)
    write_text(checksum_path, f"{digest}  {archive_name}\n")
    return {
        "index": index,
        "archive": f"zips/{archive_name}",
        "clips": len(rows),
        "first_clip": rows[0]["clip_id"],
        "last_clip": rows[-1]["clip_id"],
        "media_bytes": media_bytes,
        "archive_bytes": archive_path.stat().st_size,
        "sha256": digest,
        "resumed": False,
    }


def package(selection, clips_root, output, shard_size, workers, expected_clips):
    rows = read_selection(selection)
    if expected_clips and len(rows) != expected_clips:
        raise ValueError(
            f"selection has {len(rows)} clips; expected {expected_clips}"
        )
    output.mkdir(parents=True, exist_ok=True)
    shards = [
        rows[start:start + shard_size]
        for start in range(0, len(rows), shard_size)
    ]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                package_shard, index, shard, clips_root, output
            ): index
            for index, shard in enumerate(shards)
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"packaged {record['archive']} "
                f"({record['clips']} clips, {record['archive_bytes']} bytes)",
                flush=True,
            )
    records.sort(key=lambda record: record["index"])

    manifest = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    shard_manifest = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    checksums = "".join(
        f"{record['sha256']}  {record['archive']}\n" for record in records
    )
    write_text(output / "manifest/items.jsonl", manifest)
    write_text(output / "manifest/shards.jsonl", shard_manifest)
    write_text(output / "manifest/checksums.sha256", checksums)
    release = {
        "schema": 1,
        "name": "VidDub-Training-Clips",
        "license": "CC-BY-NC-4.0",
        "source_dataset": (
            "https://huggingface.co/datasets/AIPeanutman/"
            "VidDub-Training-Data"
        ),
        "clips": len(rows),
        "zip_shards": len(records),
        "clips_per_full_shard": shard_size,
        "files_per_clip": list(MEDIA_FILES),
        "archive_bytes": sum(record["archive_bytes"] for record in records),
        "uploaded": False,
    }
    write_text(
        output / "manifest/release.json",
        json.dumps(release, ensure_ascii=False, indent=2) + "\n",
    )
    shutil.copy2(ROOT / "data/TRAINING_CLIPS_CARD.md", output / "README.md")
    shutil.copy2(
        ROOT / "dependencies/licenses/CC-BY-NC-4.0.txt",
        output / "LICENSE",
    )
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)
    args = parser.parse_args()
    if args.shard_size <= 0 or args.workers <= 0:
        parser.error("--shard-size and --workers must be positive")
    release = package(
        args.selection.resolve(),
        args.clips_root.resolve(),
        args.output.resolve(),
        args.shard_size,
        args.workers,
        args.expected_clips,
    )
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
