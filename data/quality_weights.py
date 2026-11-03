"""Build paper-compatible per-clip quality weights from quality-score records.

This operates on the full pre-release train pool. Do not run it again on the
already filtered VidDub-Training-Data release package; that package's membership is
frozen by ``manifest/release_selection.jsonl``.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import MANIFEST


SOURCES = ("h3", "hallo3", "ls_real", "astri")
SOURCE_MANIFESTS = {
    "h3": MANIFEST,
    "hallo3": MANIFEST / "hallo3",
    "ls_real": MANIFEST / "ls_real",
    "astri": MANIFEST / "astri",
}


def training_rows():
    exclude_path = MANIFEST / "bench_exclude.json"
    excluded = (
        {group for values in json.loads(exclude_path.read_text()).values() for group in values}
        if exclude_path.is_file()
        else set()
    )
    rows = []
    for source in SOURCES:
        source_root = SOURCE_MANIFESTS[source]
        groups = (
            json.loads((source_root / "groups.json").read_text())
            if source != "h3"
            else None
        )
        for path in sorted((source_root / "s02").glob("shard_*.jsonl")):
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "error" in row:
                        continue
                    row["source"] = source
                    row["domain"], row["group"] = (
                        groups[row["id"]] if groups else ("H3", row["id"])
                    )
                    is_val = (
                        int(hashlib.md5(row["group"].encode()).hexdigest(), 16) % 100
                        == 0
                    )
                    if row["group"] not in excluded and not is_val:
                        rows.append(row)
    return sorted(rows, key=lambda row: row["clip_id"])


def cached_clip_ids():
    ids = set()
    for source_root in SOURCE_MANIFESTS.values():
        for path in (source_root / "s03").glob("shard_*.txt"):
            ids.update(path.read_text().split())
    return ids


def percentile(values):
    return np.argsort(np.argsort(values)) / max(1, len(values) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="v1")
    parser.add_argument("--iqa-min", type=float, default=40)
    parser.add_argument("--off-max", type=int, default=4)
    parser.add_argument("--drop-frac", type=float, default=0.2)
    args = parser.parse_args()

    quality = {}
    for path in sorted((MANIFEST / "quality").glob("rank_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    quality[row["clip_id"]] = row

    cached = cached_clip_ids()
    by_domain = {}
    for row in training_rows():
        item = quality.get(row["clip_id"])
        if row["clip_id"] in cached and item and item.get("ok"):
            by_domain.setdefault(row["domain"], []).append(row)

    weights = {}
    stats = {}
    for domain, rows in sorted(by_domain.items()):
        iqa = np.array([quality[row["clip_id"]]["iqa"] for row in rows])
        mouth_sharpness = np.nan_to_num(
            np.array([quality[row["clip_id"]]["msharp"] for row in rows])
        )
        sync_confidence = np.array([row["sync_conf"] for row in rows])
        offsets = np.abs([row["sync_offset"] for row in rows])
        score = (
            percentile(sync_confidence)
            + percentile(iqa)
            + percentile(mouth_sharpness)
        ) / 3
        hard = (iqa >= args.iqa_min) & (offsets < args.off_max)
        if not hard.any():
            raise ValueError(f"No clips pass hard filters for domain {domain}")
        keep = hard & (score >= np.quantile(score[hard], args.drop_frac))
        kept_rows = [row for row, selected in zip(rows, keep) if selected]
        domain_weights = 0.5 + percentile(score[keep])
        for row, weight in zip(kept_rows, domain_weights):
            weights[row["clip_id"]] = round(float(weight), 3)
        stats[domain] = {
            "clips": len(rows),
            "iqa<min": int((iqa < args.iqa_min).sum()),
            "off>=max": int((offsets >= args.off_max).sum()),
            "kept": int(keep.sum()),
            "kept_frac": round(float(keep.mean()), 3),
            "kept_iqa_p50": round(float(np.median(iqa[keep])), 1),
            "kept_sync_p50": round(float(np.median(sync_confidence[keep])), 2),
        }
        print(domain, stats[domain])

    output = Path(MANIFEST) / f"quality/weights_{args.name}.json"
    output.write_text(json.dumps(weights))
    output.with_suffix(".stats.json").write_text(
        json.dumps({"args": vars(args), "stats": stats}, indent=1) + "\n"
    )
    print(f"{len(weights)} clips -> {output}")


if __name__ == "__main__":
    main()
