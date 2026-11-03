"""Gate for S3: does StableSyncNet on our (non-aligned) crops separate aligned audio from shifted / other-clip audio?

python check_syncloss.py --n 48
"""
import argparse

import numpy as np
import torch

from .runtime import FPS, load_clip, load_rows
from .sync_loss import N_FRAMES, SyncLoss, melspectrogram, mel_window, mouth_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    dev = torch.device("cuda")
    loss = SyncLoss().to(dev)
    clips = [load_clip(r) for r in load_rows("val")[:args.n]]
    mels = [melspectrogram(c["audio"]) for c in clips]
    variants = [(0.0, 0.0), (0.1, 0.0), (0.25, 0.0), (0.1, 0.15), (0.25, 0.3)]
    for pad_x, pad_top in variants:
        pos, shift, other = [], [], []
        for i, c in enumerate(clips):
            box = mouth_box(c["mask"], pad_x, pad_top)
            video = torch.from_numpy(c["frames"]).to(dev).permute(3, 0, 1, 2).float() / 127.5 - 1
            for s in (1, 33, 64):
                v = video[:, s:s + N_FRAMES]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pos.append(loss.cos(v, box, mel_window(mels[i], s / FPS)).item())
                    s2 = s + 12 if s + 12 + N_FRAMES <= 81 else s - 12
                    shift.append(loss.cos(v, box, mel_window(mels[i], s2 / FPS)).item())
                    other.append(loss.cos(v, box, mel_window(mels[(i + 1) % len(clips)], s / FPS)).item())
        pos, shift, other = map(np.array, (pos, shift, other))
        print(f"pad_x {pad_x:.2f} pad_top {pad_top:.2f} | cos pos {pos.mean():.3f} shift {shift.mean():.3f} other {other.mean():.3f} | "
              f"acc vs shift {(pos > shift).mean():.2f} vs other {(pos > other).mean():.2f} | bce pos {-np.log(np.clip(pos, 1e-6, 1)).mean():.3f}",
              flush=True)


if __name__ == "__main__":
    main()
