"""LatentSync-1.6 StableSyncNet as a differentiable lip-sync loss on our 512 face crops.

StableSyncNet sees the lower half of a 256x256 affine-aligned face (128x256), 16 frames @25fps, and 52 mel columns (80 Hz).
Our crops are not affine-aligned; the lower half is approximated by the clip's mouth mask box (nose bottom -> below chin,
cheek to cheek), widened by `pad_x` on each side. 16 frames @24fps span 0.667s, so the matching 53.3 mel columns are
resampled to 52.
"""
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import signal

from .runtime import FPS, SR, WEIGHTS

LS_ROOT = Path(os.environ.get("VIDDUB_LATENTSYNC_ROOT", WEIGHTS / "LatentSync"))
N_FRAMES, MEL_HZ, MEL_WIN = 16, 80, 52
_MEL_BASIS = librosa.filters.mel(sr=SR, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def melspectrogram(wav):
    """Same as latentsync/utils/audio.py with configs/audio.yaml: [80, T] at 80 Hz, values in [-4, 4]."""
    d = librosa.stft(y=signal.lfilter([1, -0.97], [1], wav), n_fft=800, hop_length=200, win_length=800)
    s = 20 * np.log10(np.maximum(1e-5, _MEL_BASIS @ np.abs(d))) - 20
    return np.clip(8 * (s + 100) / 100 - 4, -4, 4).astype(np.float32)


def mel_window(mel, t0):
    """Mel columns for the 16 video frames starting at time t0 (s), resampled to MEL_WIN: [1, 1, 80, 52]."""
    a = int(round(t0 * MEL_HZ))
    b = a + int(round(N_FRAMES / FPS * MEL_HZ))
    m = torch.from_numpy(mel[:, a:b])
    if m.shape[1] < b - a:
        m = F.pad(m, (0, b - a - m.shape[1]), value=-4.0)
    return F.interpolate(m[None, None], size=(80, MEL_WIN), mode="bilinear", align_corners=False)


def mouth_box(mask, pad_x=0.1, pad_top=0.0):
    """Pixel box (t, b, l, r) of the mouth mask, widened; mask: bool [H, W]."""
    ys, xs = np.nonzero(mask)
    t, b, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    w, h = r - l, b - t
    H, W = mask.shape
    return max(0, int(t - pad_top * h)), b, max(0, int(l - pad_x * w)), min(W, int(r + pad_x * w))


class SyncLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(LS_ROOT))
        from latentsync.models.stable_syncnet import StableSyncNet
        cfg = yaml.safe_load(open(LS_ROOT / "configs/syncnet/syncnet_16_pixel_attn.yaml"))["model"]
        self.net = StableSyncNet(cfg, gradient_checkpointing=True)
        self.net.load_state_dict(torch.load(LS_ROOT / "checkpoints/stable_syncnet.pt", map_location="cpu", weights_only=True)["state_dict"])
        self.net.requires_grad_(False).eval()

    def crop(self, video, box):
        """video: [3, 16, H, W] in [-1, 1] -> [1, 48, 128, 256] (frame-major, like LatentSync)."""
        t, b, l, r = box
        x = F.interpolate(video[:, :, t:b, l:r].transpose(0, 1).float(), size=(128, 256), mode="bilinear", align_corners=False)
        return x.reshape(1, -1, 128, 256)

    def cos(self, video, box, mel):
        v, a = self.net(self.crop(video, box), mel.to(video.device))
        return F.cosine_similarity(v.float(), a.float())

    def forward(self, video, box, mel):
        """BCE(cos, 1) as in LatentSync's train_unet.py."""
        return -torch.log(self.cos(video, box, mel).clamp(1e-6, 1.0)).mean()
