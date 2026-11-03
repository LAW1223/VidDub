"""Copy one atomic checkpoint into a portable, checksummed release artifact."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sanitize_metadata(value, key=None):
    """Remove workstation paths while preserving reproducibility fields."""
    if isinstance(value, dict):
        return {name: sanitize_metadata(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        if key == "rows_file":
            return "data/manifest/train.json"
        if key == "clips_root":
            return "data/clips"
        if key == "path":
            return Path(value).name
        return Path(value).name
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.output_root).resolve() / args.name
    target.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": 1, "name": args.name, "stage": args.stage,
               "step": args.step, "files": {}}
    for component in ("dit", "vace", "adapter"):
        src = source / f"{component}_step{args.step:06d}.safetensors"
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = target / src.name
        if dst.exists() or dst.is_symlink():
            if not args.force and (dst.is_symlink() or digest(dst) != digest(src)):
                raise FileExistsError(dst)
            dst.unlink()
        shutil.copy2(src, dst)
        receipt["files"][component] = {
            "filename": src.name, "bytes": src.stat().st_size, "sha256": digest(src)}
    for metadata_name in (f"causal_step{args.step:06d}.json",):
        src = source / metadata_name
        if src.is_file():
            dst = target / metadata_name
            if dst.exists() and not args.force:
                raise FileExistsError(dst)
            metadata = sanitize_metadata(json.loads(src.read_text()))
            dst.write_text(json.dumps(metadata, indent=2) + "\n")
            receipt["metadata"] = metadata_name
    if args.config:
        receipt["config"] = args.config
    (target / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
