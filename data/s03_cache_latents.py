"""Precompute VAE latents (gt / masked / ref) + T5 context per clip -> data/clips/<dir>/latents.pt (resumable)."""
import argparse
import time

import torch

from .common import MANIFEST, shard

from viddub.full_context.runtime import CLIPS, compute_latents, load_clip, load_models, load_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    rows = shard(load_rows("train") + load_rows("val"), args.shard, args.num_shards)
    done_path = MANIFEST / "s03" / f"shard_{args.shard:04d}.txt"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    pipe, _ = load_models("cuda")
    t0 = time.time()
    for i, row in enumerate(rows):
        out = CLIPS / row["dir"] / "latents.pt"
        if row["clip_id"] in done:
            continue
        try:
            lat = compute_latents(pipe, load_clip(row))
        except Exception as e:
            print(f"ERROR {row['clip_id']}: {e!r}", flush=True)
            continue
        tmp = out.with_suffix(".tmp")
        torch.save({k: v.contiguous() for k, v in lat.items()}, tmp)
        tmp.rename(out)
        with open(done_path, "a") as f:
            f.write(row["clip_id"] + "\n")
        if i % 20 == 0:
            print(f"[shard {args.shard}] {i}/{len(rows)} {(time.time() - t0) / (i + 1):.2f}s/it", flush=True)


if __name__ == "__main__":
    main()
