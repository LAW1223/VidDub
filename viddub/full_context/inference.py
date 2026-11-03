"""Legacy SFT inference utility for component checkpoints.

This module retains the multi-step teacher/SFT path used during development.
Public two-step VidDub inference must use ``python scripts/infer.py`` so the
serialized release config, DMD sampler, native timing, and overlap contract are
validated together.
"""
import argparse
from pathlib import Path

import cv2
import decord
import librosa
import numpy as np
import torch
from safetensors.torch import load_file

from .audio import DubAudioAdapter
from .color_harmonization import harmonize_lower_face
from .runtime import FPS, PROJECT, SR, ZeroLatents, load_models, sample, write_mp4

from data.s01_analyze import detect_landmarks  # noqa: E402
from data.s02_make_clips import CROP_SCALE, LM_LIP_LOW, LM_LIP_UP, NUM_FRAMES, SIZE, crop_box, lower_face_mask  # noqa: E402


PAPER_OVERLAP = 17


def windows(n_frames, overlap=PAPER_OVERLAP, require_stride_multiple=False):
    """Return bounded source-reanchored windows, shifting the last to cover the tail."""
    stride = NUM_FRAMES - overlap
    if n_frames < NUM_FRAMES:
        raise ValueError(f"video has {n_frames} frames, need at least {NUM_FRAMES}")
    if stride <= 0 or (require_stride_multiple and stride % 4):
        raise ValueError(
            f"window stride must be positive"
            f"{' and a multiple of four' if require_stride_multiple else ''}; "
            f"got {stride}")
    starts = list(range(0, n_frames - NUM_FRAMES + 1, stride))
    if starts[-1] + NUM_FRAMES < n_frames:
        starts.append(n_frames - NUM_FRAMES)
    return starts


def paste_back(frame, gen, box, alpha):
    """Blend the generated crop back into the full frame inside the feathered mouth mask."""
    x0, y0, side = box
    g = cv2.resize(gen, (side, side), interpolation=cv2.INTER_CUBIC)
    a = cv2.resize(alpha, (side, side), interpolation=cv2.INTER_LINEAR)[..., None]
    out = frame.copy()
    region = out[y0:y0 + side, x0:x0 + side].astype(np.float32)
    out[y0:y0 + side, x0:x0 + side] = (region * (1 - a) + g.astype(np.float32) * a).round().astype(np.uint8)
    return out


def load_dub_models(adapter_path, dit_path=None, vace_path=None, device=torch.device("cuda")):
    pipe, whisper = load_models(device)
    if dit_path:
        pipe.dit.load_state_dict(load_file(dit_path))
    if vace_path:
        pipe.vace.load_state_dict(load_file(vace_path))
    adapter = DubAudioAdapter().to(device)
    adapter.load_state_dict(load_file(adapter_path))
    adapter.eval()
    return pipe, whisper, adapter, ZeroLatents(pipe)


def dub_video(
    models,
    frames,
    audio,
    prompt="",
    steps=30,
    audio_cfg=1.0,
    feather=15,
    seed=0,
    window_box=False,
    fps=FPS,
    stretch=False,
    overlap=PAPER_OVERLAP,
    mode="auto",
    harmonize=True,
    log=print,
    sample_fn=None,
):
    """frames: uint8 [T,H,W,3] at 24 fps; audio: 16 kHz mono. Returns dubbed frames, same shape.
    window_box: crop box and mask from each window's own landmarks (as in training) instead of the whole video.
    fps: native frame rate of `frames`. The audio tokens are taken at the native frame times, or (stretch) the audio is
    time-stretched (pitch kept) to the training frame rate.
    mode: `clip` makes one 81-frame call; `windowed` uses bounded source-reanchored windows; `auto` selects by length.
    overlap: frames shared by consecutive windows, linearly cross-faded in crop space (static crop box only).
    harmonize: apply the paper's lower-face color correction before window blending."""
    pipe, whisper, adapter, zero_latents = models
    n, H, W = len(frames), *frames.shape[1:3]
    if mode not in {"auto", "clip", "windowed"}:
        raise ValueError(f"unknown inference mode: {mode}")
    mode = "clip" if mode == "auto" and n == NUM_FRAMES else (
        "windowed" if mode == "auto" else mode)
    if mode == "clip" and n != NUM_FRAMES:
        raise ValueError(
            f"clip mode requires exactly {NUM_FRAMES} frames; got {n}")
    if window_box and mode == "windowed" and overlap:
        raise ValueError("overlap blending requires one stable full-video crop")
    starts = [0] if mode == "clip" else windows(
        n, overlap, require_stride_multiple=sample_fn is not None)
    audio = np.pad(audio, (0, max(0, int(n / fps * SR) - len(audio))))
    if stretch and fps != FPS:
        audio = librosa.effects.time_stretch(audio, rate=FPS / fps)
    t_fps = FPS if stretch else fps

    lm, _ = detect_landmarks(frames)
    if not np.isfinite(lm).any():
        # MediaPipe's default detector can miss clear but relatively small
        # faces. Retry only complete misses so the standard path and benchmark
        # behavior remain unchanged.
        lm, _ = detect_landmarks(
            frames,
            min_detection_confidence=0.05,
            min_tracking_confidence=0.1,
        )
        if not np.isfinite(lm).any():
            raise RuntimeError("no face landmarks detected in any frame")
        log(
            "default face detector missed all frames; "
            "using low-confidence FaceMesh fallback"
        )
    # Tight shots (face*CROP_SCALE > frame): edge-pad so the face/crop ratio matches training.
    face = np.nanmedian(np.maximum(np.ptp(lm[:, :, 0], 1), np.ptp(lm[:, :, 1], 1)))
    if not np.isfinite(face):
        raise RuntimeError("face landmark extent is not finite")
    pad = max(0, int(np.ceil((face * CROP_SCALE - min(H, W)) / 2)) + 1)
    if pad > 1:
        frames_p = np.pad(frames, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
        lm = lm + pad
    else:
        pad, frames_p = 0, frames
    ref_idx = int(np.nanargmax(lm[:, LM_LIP_LOW, 1] - lm[:, LM_LIP_UP, 1]))

    def region(lm_w):
        box = crop_box(lm_w, H + 2 * pad, W + 2 * pad)
        mask = lower_face_mask(lm_w, box)
        return box, mask, cv2.GaussianBlur(mask, (0, 0), feather).astype(np.float32) / 255

    def crop(f, box):
        x0, y0, side = box
        return cv2.resize(f[y0:y0 + side, x0:x0 + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    # Default: one crop box and one mask for the whole video, keeping the mouth region stable across windows.
    box, mask, alpha = region(lm)
    out = frames_p.copy()
    crop_output = None
    written = 0
    if not window_box:
        source_crops = np.stack([crop(f, box) for f in frames_p])
        crop_output = np.empty_like(source_crops)
    for window_index, start in enumerate(starts):
        if window_box:
            box, mask, alpha = region(lm[start:start + NUM_FRAMES])
            crops = np.stack([
                crop(f, box) for f in frames_p[start:start + NUM_FRAMES]
            ])
        else:
            crops = source_crops[start:start + NUM_FRAMES]
        clip = {"frames": crops, "mask": mask > 127, "ref": crop(frames_p[ref_idx], box), "prompt": prompt}
        sample_start = round(start / t_fps * SR)
        sample_count = round(NUM_FRAMES / t_fps * SR)
        a = audio[sample_start:sample_start + sample_count]
        a = np.pad(a, (0, max(0, sample_count - len(a))))
        sampler = sample if sample_fn is None else sample_fn
        window_seed = seed + window_index
        gen = sampler(
            pipe, whisper, adapter, zero_latents, clip, a,
            steps=steps, audio_cfg=audio_cfg, seed=window_seed, fps=t_fps)
        color_report = None
        if harmonize:
            gen, color_report = harmonize_lower_face(gen, crops, mask > 127)
        if window_box:
            paste_alpha = (mask > 127).astype(np.float32) if harmonize else alpha
            for i, g in enumerate(gen):
                out[start + i] = paste_back(
                    frames_p[start + i], g, box, paste_alpha)
            written = max(written, start + len(gen))
        else:
            end = start + len(gen)
            ov = max(0, written - start)
            if ov:
                weight = (
                    np.arange(1, ov + 1, dtype=np.float32) / (ov + 1)
                )[:, None, None, None]
                crop_output[start:start + ov] = np.rint(
                    crop_output[start:start + ov].astype(np.float32) * (1 - weight)
                    + gen[:ov].astype(np.float32) * weight
                ).astype(np.uint8)
            crop_output[start + ov:end] = gen[ov:]
            written = max(written, end)
        detail = ""
        if color_report is not None:
            detail = (
                f", chroma MAE {color_report['interior_chroma_mae_before']:.3f}"
                f"->{color_report['interior_chroma_mae_after']:.3f}"
            )
        log(
            f"window {window_index}: {start}-{start + NUM_FRAMES}, "
            f"seed {window_seed}{detail}")
    if written != n:
        raise RuntimeError(f"window merge wrote {written} frames, expected {n}")
    if not window_box:
        paste_alpha = (mask > 127).astype(np.float32) if harmonize else alpha
        for index, generated in enumerate(crop_output):
            out[index] = paste_back(
                frames_p[index], generated, box, paste_alpha)
    return out[:, pad:pad + H, pad:pad + W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dit", default=None)
    ap.add_argument("--vace", default=None)
    ap.add_argument("--prompt", default="", help="ignored (kept for old command lines): the text context is always empty")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--audio_cfg", type=float, default=1.0)
    ap.add_argument("--feather", type=int, default=15, help="mask feather radius in crop pixels")
    ap.add_argument("--mode", choices=("auto", "clip", "windowed"), default="auto")
    ap.add_argument("--overlap", type=int, default=PAPER_OVERLAP,
                    help="shared frames (paper protocol: 17, giving stride 64)")
    ap.add_argument("--no_harmonize", action="store_true",
                    help="disable paper lower-face color harmonization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window_box", action="store_true", help="per-window crop box and mask")
    args = ap.parse_args()

    models = load_dub_models(args.adapter, args.dit, args.vace)
    frames = decord.VideoReader(args.video, num_threads=2)[:].asnumpy()
    audio = librosa.load(args.audio, sr=SR, mono=True)[0]
    out = dub_video(models, frames, audio, args.prompt, args.steps, args.audio_cfg,
                    args.feather, args.seed, args.window_box, overlap=args.overlap,
                    mode=args.mode, harmonize=not args.no_harmonize)
    write_mp4(args.out, out, audio[:int(len(out) / FPS * SR)])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
