"""Visual quality of every cached clip (all sources, train + val) from its ref.png (512 face crop):
  iqa    HyperIQA (KonIQ, 0-100) on the face at 256 px, centre 224 crop (= LatentSync's filter_visual_quality, which keeps >= 40)
  sharp  variance of the Laplacian of the grey face at 256 px
  msharp the same inside the mouth mask's bounding box
Usage (one process per GPU): quality_score.py --rank R --world N  -> data/manifest/quality/rank_RRR.jsonl (resumable)."""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

PROJECT = Path(__file__).resolve().parents[1]
LS_ROOT = Path(os.environ.get("VIDDUB_LATENTSYNC_ROOT", PROJECT / "weights/LatentSync"))
sys.path.insert(0, str(LS_ROOT))
from viddub.full_context.runtime import CLIPS, MANIFEST, cached_clip_ids, load_rows  # noqa: E402
import eval.hyper_iqa as hyper_iqa  # noqa: E402

SOURCES = ("h3", "hallo3", "ls_real", "astri")
OUT = MANIFEST / "quality"


class Refs(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            img = cv2.cvtColor(cv2.imread(str(CLIPS / r["dir"] / "ref.png")), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(CLIPS / r["dir"] / "mask.png"), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) > 127
            lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)
            ys, xs = np.nonzero(mask)
            msharp = float(lap[ys.min():ys.max() + 1, xs.min():xs.max() + 1].var()) if len(ys) else float("nan")
            return r["clip_id"], torch.from_numpy(img).permute(2, 0, 1).float() / 255, float(lap.var()), msharp, True
        except Exception:
            return r["clip_id"], torch.zeros(3, 256, 256), float("nan"), float("nan"), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"rank_{args.rank:03d}.jsonl"
    done = {json.loads(l)["clip_id"] for l in open(out)} if out.exists() else set()

    cached = cached_clip_ids(SOURCES)
    rows = [r for s in ("train", "val") for r in load_rows(s, SOURCES) if r["clip_id"] in cached]
    rows = sorted(rows, key=lambda r: r["clip_id"])[args.rank::args.world]
    rows = [r for r in rows if r["clip_id"] not in done]
    print(f"rank {args.rank}: {len(rows)} clips to score ({len(done)} done)", flush=True)
    if not rows:
        return

    hyper_iqa.model_zoo.load_url = lambda *a, **k: {}  # ImageNet init is overwritten by the KonIQ checkpoint
    dev = "cuda"
    net = hyper_iqa.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(dev).eval()
    net.load_state_dict(torch.load(LS_ROOT / "checkpoints/auxiliary/koniq_pretrained.pkl", map_location=dev, weights_only=True))
    tf = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                         torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    loader = torch.utils.data.DataLoader(Refs(rows), batch_size=args.batch, num_workers=8)
    with open(out, "a") as f, torch.no_grad():
        for k, (ids, imgs, sharp, msharp, ok) in enumerate(loader):
            paras = net(tf(imgs.to(dev)))
            iqa = hyper_iqa.TargetNet(paras).to(dev)(paras["target_in_vec"]).view(-1).tolist()
            for i in range(len(ids)):
                f.write(json.dumps({"clip_id": ids[i], "iqa": iqa[i] if ok[i] else None, "sharp": float(sharp[i]),
                                    "msharp": float(msharp[i]), "ok": bool(ok[i])}) + "\n")
            if k % 50 == 0:
                f.flush()
                print(f"rank {args.rank}: {(k + 1) * args.batch}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
