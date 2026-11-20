"""Legacy SFT inference utility for component checkpoints.

This module retains the multi-step teacher/SFT path used during development.
Public two-step VidDub inference must use ``python scripts/infer.py`` so the
serialized release config, DMD sampler, native timing, and overlap contract are
validated together.
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from safetensors.torch import load_file

from data.s01_analyze import (  # noqa: E402
    _is_valid_face_landmarks,
    detect_landmarks,
    detect_landmarks_iter,
)
from data.s02_make_clips import (  # noqa: E402
    CROP_SCALE,
    LM_CHEEK_L,
    LM_CHEEK_R,
    LM_CHIN,
    LM_FOREHEAD,
    LM_LIP_LOW,
    LM_LIP_UP,
    LM_NOSE_BOTTOM,
    NUM_FRAMES,
    SIZE,
    crop_box,
    lower_face_mask,
)
# Import decord after torch and mediapipe so decord's bundled VC++ runtime on
# Windows does not shadow the newer runtime expected by mediapipe.
import decord  # noqa: E402

from .audio import DubAudioAdapter
from .color_harmonization import HarmonizationConfig, harmonize_lower_face, skin_anchor_mask
from .runtime import FPS, PROJECT, SR, ZeroLatents, load_models, sample, write_mp4


PAPER_OVERLAP = 17
_NO_FACE_ERROR = (
    "no face landmarks detected in any frame "
    "(the face may be too small, too far, occluded, or off-axis; "
    "MediaPipe typically requires a mostly front-facing face ~60+ px tall)"
)


def _face_paste_mask(lm, max_gap=2):
    """True on frames that should receive mouth paste-back; bridges tiny 1-2 frame drops."""
    valid = np.isfinite(lm).all(axis=(1, 2)).copy()
    idx = np.flatnonzero(valid)
    if len(idx) >= 2:
        for prev, curr in zip(idx[:-1], idx[1:]):
            if 1 < curr - prev <= max_gap + 1:
                valid[prev + 1:curr] = True
    return valid


def clean_landmark_track(lm, height, width):
    """Reject geometrically invalid frames and abrupt spatial/scale landmark outliers."""
    cleaned = np.asarray(lm, dtype=np.float32).copy()
    n = len(cleaned)
    rejected = 0

    for t in range(n):
        if np.isfinite(cleaned[t]).any() and not _is_valid_face_landmarks(
            cleaned[t], width, height
        ):
            cleaned[t] = np.nan
            rejected += 1

    valid_idx = np.flatnonzero(np.isfinite(cleaned).all(axis=(1, 2)))
    if len(valid_idx) < 3:
        return cleaned, rejected

    valid_lm = cleaned[valid_idx]
    cx = (valid_lm[:, :, 0].min(axis=1) + valid_lm[:, :, 0].max(axis=1)) / 2.0
    cy = (valid_lm[:, :, 1].min(axis=1) + valid_lm[:, :, 1].max(axis=1)) / 2.0
    sz = np.maximum(np.ptp(valid_lm[:, :, 0], axis=1), np.ptp(valid_lm[:, :, 1], axis=1))
    med_sz = float(np.median(sz))

    keep = (sz >= 0.35 * med_sz) & (sz <= 2.8 * med_sz)
    if keep.sum() >= 3:
        kept_pos = np.flatnonzero(keep)
        half_win = min(7, len(kept_pos) // 2)
        for rank, pos in enumerate(kept_pos):
            lo = max(0, rank - half_win)
            hi = min(len(kept_pos), rank + half_win + 1)
            nbrs = kept_pos[lo:hi]
            loc_cx = float(np.median(cx[nbrs]))
            loc_cy = float(np.median(cy[nbrs]))
            loc_sz = float(np.median(sz[nbrs]))
            dist = float(np.hypot(cx[pos] - loc_cx, cy[pos] - loc_cy))
            if dist > 1.0 * loc_sz or sz[pos] < 0.5 * loc_sz or sz[pos] > 2.0 * loc_sz:
                keep[pos] = False

    if keep.any():
        bad = valid_idx[~keep]
        if len(bad):
            cleaned[bad] = np.nan
            rejected += len(bad)
    return cleaned, rejected


def _smooth_1d(values, sigma=1.5):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1 or sigma <= 0.0:
        return arr
    radius = max(1, int(np.ceil(3.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _needs_stabilized_tracking(lm, box):
    """Return True when a single static crop box would clip the face or bloat the edit mask."""
    valid = np.isfinite(lm).all(axis=(1, 2))
    if valid.sum() < 2:
        return False
    lm_v = lm[valid]
    x0, y0, side = box
    if side <= 0:
        return True

    face_h = lm_v[:, LM_CHIN, 1] - lm_v[:, LM_FOREHEAD, 1]
    med_face_h = float(np.median(face_h))
    min_nose_y = float(lm_v[:, LM_NOSE_BOTTOM, 1].min())
    max_bottom_y = float(lm_v[:, LM_CHIN, 1].max()) + 0.1 * med_face_h
    min_left_x = float(lm_v[:, LM_CHEEK_R, 0].min())
    max_right_x = float(lm_v[:, LM_CHEEK_L, 0].max())

    # 1. Static crop clips upper face/nose, chin, or cheeks.
    if (
        min_nose_y < y0 + 0.15 * side
        or max_bottom_y > y0 + 0.98 * side
        or min_left_x < x0 + 0.05 * side
        or max_right_x > x0 + 0.95 * side
    ):
        return True

    # 2. Union lower-face mask inflates significantly beyond a single frame's lower face
    #    (which would expose neck/chest/background pixels inside the edit mask).
    per_lh = (lm_v[:, LM_CHIN, 1] - lm_v[:, LM_NOSE_BOTTOM, 1]) + 0.1 * face_h
    per_fw = lm_v[:, LM_CHEEK_L, 0] - lm_v[:, LM_CHEEK_R, 0]
    med_lh = max(float(np.median(per_lh)), 1.0)
    med_fw = max(float(np.median(per_fw)), 1.0)
    if (max_bottom_y - min_nose_y) > 1.35 * med_lh or (max_right_x - min_left_x) > 1.40 * med_fw:
        return True

    # 3. Significant face scale change (e.g. leaning toward/away from camera).
    sz = np.maximum(np.ptp(lm_v[:, :, 0], axis=1), np.ptp(lm_v[:, :, 1], axis=1))
    if float(sz.max()) / max(float(sz.min()), 1.0) > 1.35:
        return True

    return False


def _stabilized_edge_pad(lm, height, width):
    """Extra border padding so edge-framed speakers stay centered in stabilized 512x512 crops."""
    valid = np.isfinite(lm).all(axis=(1, 2))
    if not valid.any():
        return 0
    lm_v = lm[valid]
    cx = (lm_v[:, :, 0].min(axis=1) + lm_v[:, :, 0].max(axis=1)) / 2.0
    cy = (lm_v[:, :, 1].min(axis=1) + lm_v[:, :, 1].max(axis=1)) / 2.0
    sz = np.maximum(np.ptp(lm_v[:, :, 0], axis=1), np.ptp(lm_v[:, :, 1], axis=1))
    half = sz * CROP_SCALE / 2.0
    overshoot = float(
        max(
            0.0,
            np.max(half - cx),
            np.max(cx + half - width),
            np.max(half - cy),
            np.max(cy + half - height),
            np.max((sz * CROP_SCALE - min(height, width)) / 2.0),
        )
    )
    pad = int(np.ceil(overshoot)) + 1 if overshoot > 0.0 else 0
    return pad if pad > 1 else 0


def _stabilized_crop_boxes(lm, height, width, smooth_sigma=1.5):
    """Build smooth per-frame face-centered crop boxes for videos with significant head motion."""
    n = len(lm)
    valid = np.isfinite(lm).all(axis=(1, 2))
    v_idx = np.flatnonzero(valid).astype(np.float64)
    idx = np.arange(n, dtype=np.float64)
    lm_v = lm[valid]
    cx_v = (lm_v[:, :, 0].min(axis=1) + lm_v[:, :, 0].max(axis=1)) / 2.0
    cy_v = (lm_v[:, :, 1].min(axis=1) + lm_v[:, :, 1].max(axis=1)) / 2.0
    sz_v = np.maximum(np.ptp(lm_v[:, :, 0], axis=1), np.ptp(lm_v[:, :, 1], axis=1))

    cx = _smooth_1d(np.interp(idx, v_idx, cx_v), sigma=smooth_sigma)
    cy = _smooth_1d(np.interp(idx, v_idx, cy_v), sigma=smooth_sigma)
    sz = _smooth_1d(np.interp(idx, v_idx, sz_v), sigma=smooth_sigma)

    max_side = min(height, width)
    boxes = []
    for t in range(n):
        side = int(round(min(max(sz[t] * CROP_SCALE, 32.0), max_side)))
        x0 = int(np.clip(round(cx[t] - side / 2.0), 0, width - side))
        y0 = int(np.clip(round(cy[t] - side / 2.0), 0, height - side))
        boxes.append((x0, y0, side))
    return boxes


def _stabilized_lower_face_mask(lm, boxes, feather):
    """Compute a tight lower-face mask in stabilized 512x512 crop coordinates."""
    valid_idx = np.flatnonzero(np.isfinite(lm).all(axis=(1, 2)))
    key_indices = (LM_FOREHEAD, LM_NOSE_BOTTOM, LM_CHIN, LM_CHEEK_R, LM_CHEEK_L)
    tops, bottoms, lefts, rights, face_hs = [], [], [], [], []
    for t in valid_idx:
        x0, y0, side = boxes[t]
        s = SIZE / side
        pts = (lm[t, key_indices] - np.array([x0, y0], dtype=np.float32)) * s
        forehead_y, nose_y, chin_y, cheek_r_x, cheek_l_x = (
            float(pts[0, 1]),
            float(pts[1, 1]),
            float(pts[2, 1]),
            float(pts[3, 0]),
            float(pts[4, 0]),
        )
        fh = max(chin_y - forehead_y, 1.0)
        face_hs.append(fh)
        tops.append(nose_y)
        bottoms.append(chin_y + 0.1 * fh)
        lefts.append(cheek_r_x)
        rights.append(cheek_l_x)

    l = int(round(min(lefts)))
    t_top = int(round(min(tops)))
    r = int(round(max(rights)))
    b = int(round(max(bottoms)))
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[max(t_top, 0):min(b, SIZE), max(l, 0):min(r, SIZE)] = 255
    alpha = cv2.GaussianBlur(mask, (0, 0), feather).astype(np.float32) / 255
    return mask, alpha


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
    pipe, whisper = load_models(device, include_text_encoder=False)
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
    harmonization_config=None,
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
    lm, rejected = clean_landmark_track(lm, H, W)
    if not np.isfinite(lm).any():
        # MediaPipe's default detector can miss clear but relatively small
        # faces. Retry only complete misses so the standard path and benchmark
        # behavior remain unchanged.
        lm, _ = detect_landmarks(
            frames,
            min_detection_confidence=0.05,
            min_tracking_confidence=0.1,
        )
        lm, rejected = clean_landmark_track(lm, H, W)
        if not np.isfinite(lm).any():
            raise RuntimeError(_NO_FACE_ERROR)
        log(
            "default face detector missed all frames; "
            "using low-confidence FaceMesh fallback"
        )
    if rejected:
        log(f"filtered {rejected} outlier face-landmark frame(s)")
    paste_frame = _face_paste_mask(lm)
    if not paste_frame.all():
        log(
            f"face detected on {int(paste_frame.sum())}/{n} frames "
            "(remaining frames will be passed through unchanged)"
        )
    # Tight shots (face*CROP_SCALE > frame): edge-pad so the face/crop ratio matches training.
    face_sizes = np.maximum(np.ptp(lm[:, :, 0], 1), np.ptp(lm[:, :, 1], 1))
    face = np.nanmedian(face_sizes)
    if not np.isfinite(face):
        raise RuntimeError("face landmark extent is not finite")
    pad = max(0, int(np.ceil((face * CROP_SCALE - min(H, W)) / 2)) + 1)
    if pad <= 1:
        pad = 0
    use_stabilized = (not window_box) and _needs_stabilized_tracking(
        lm + pad, crop_box(lm + pad, H + 2 * pad, W + 2 * pad)
    )
    if use_stabilized:
        pad = max(pad, _stabilized_edge_pad(lm, H, W))
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
    # When head motion would clip the crop or bloat the union mask onto neck/chest, track per frame.
    frame_boxes = None
    if use_stabilized:
        frame_boxes = _stabilized_crop_boxes(lm, H + 2 * pad, W + 2 * pad)
        box = frame_boxes[0]
        mask, alpha = _stabilized_lower_face_mask(lm, frame_boxes, feather)
        log("significant head motion detected; enabling stabilized per-frame face tracking")
    else:
        box, mask, alpha = region(lm)
    box_for = (lambda idx: frame_boxes[idx]) if frame_boxes is not None else (lambda idx: box)
    effective_harmonize = harmonize
    if not window_box and effective_harmonize:
        try:
            skin_anchor_mask(mask > 127, harmonization_config or HarmonizationConfig())
        except ValueError as exc:
            log(f"color harmonization skipped: {exc} (falling back to feathered blending)")
            effective_harmonize = False
    out = frames_p.copy()
    crop_output = None
    written = 0
    if not window_box:
        source_crops = np.stack([crop(f, box_for(i)) for i, f in enumerate(frames_p)])
        crop_output = np.empty_like(source_crops)
    for window_index, start in enumerate(starts):
        window_has_face = bool(paste_frame[start:start + NUM_FRAMES].any())
        sampler = sample if sample_fn is None else sample_fn
        window_seed = seed + window_index
        color_report = None
        window_harmonized = False
        if window_box and not window_has_face:
            written = max(written, start + NUM_FRAMES)
        else:
            if window_box:
                box, mask, alpha = region(lm[start:start + NUM_FRAMES])
                crops = np.stack([
                    crop(f, box) for f in frames_p[start:start + NUM_FRAMES]
                ])
                ref_box = box
            else:
                crops = source_crops[start:start + NUM_FRAMES]
                ref_box = box_for(ref_idx)
            if window_has_face:
                clip = {"frames": crops, "mask": mask > 127, "ref": crop(frames_p[ref_idx], ref_box), "prompt": prompt}
                sample_start = round(start / t_fps * SR)
                sample_count = round(NUM_FRAMES / t_fps * SR)
                a = audio[sample_start:sample_start + sample_count]
                a = np.pad(a, (0, max(0, sample_count - len(a))))
                gen = sampler(
                    pipe, whisper, adapter, zero_latents, clip, a,
                    steps=steps, audio_cfg=audio_cfg, seed=window_seed, fps=t_fps)
                if effective_harmonize:
                    try:
                        if harmonization_config is None:
                            gen, color_report = harmonize_lower_face(gen, crops, mask > 127)
                        else:
                            gen, color_report = harmonize_lower_face(
                                gen, crops, mask > 127, config=harmonization_config
                            )
                        window_harmonized = True
                    except ValueError as exc:
                        log(
                            f"window {window_index}: color harmonization skipped: {exc} "
                            "(falling back to feathered blending)"
                        )
                        if not window_box:
                            effective_harmonize = False
            else:
                gen = crops
            if window_box:
                paste_alpha = (mask > 127).astype(np.float32) if window_harmonized else alpha
                for i, g in enumerate(gen):
                    if paste_frame[start + i]:
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
        detail = "" if window_has_face else ", skipped (no face)"
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
        paste_alpha = (mask > 127).astype(np.float32) if effective_harmonize else alpha
        for index, generated in enumerate(crop_output):
            if paste_frame[index]:
                out[index] = paste_back(
                    frames_p[index], generated, box_for(index), paste_alpha)
    return out[:, pad:pad + H, pad:pad + W]


class DecordFrameSource:
    """Sequential chunked reader over decord.VideoReader for bounded-RAM dubbing."""

    def __init__(self, reader, chunk_size=64):
        self.reader = reader
        self.chunk_size = int(chunk_size)
        self.num_frames = len(reader)
        if self.num_frames <= 0:
            raise ValueError("video contains zero frames")
        first = reader[0].asnumpy()
        self.height, self.width = first.shape[:2]

    def iter_chunks(self):
        for start in range(0, self.num_frames, self.chunk_size):
            end = min(self.num_frames, start + self.chunk_size)
            yield self.reader.get_batch(list(range(start, end))).asnumpy()

    def get_frame(self, index):
        return self.reader[int(index)].asnumpy()


def resolve_padding_frames(lead_in_seconds=0.0, lead_out_seconds=0.0, fps=FPS):
    """Validate lead-in / lead-out durations (seconds) and convert to frame counts."""
    lead_in = float(lead_in_seconds)
    lead_out = float(lead_out_seconds)
    fps_value = float(fps)
    if not np.isfinite(lead_in) or lead_in < 0.0:
        raise ValueError(
            f"lead_in_seconds must be a finite non-negative number; got {lead_in_seconds}"
        )
    if not np.isfinite(lead_out) or lead_out < 0.0:
        raise ValueError(
            f"lead_out_seconds must be a finite non-negative number; got {lead_out_seconds}"
        )
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError(f"fps must be positive; got {fps}")
    return int(round(lead_in * fps_value)), int(round(lead_out * fps_value))


class PaddedFrameSource:
    """Hold-edge frame-source wrapper for lead-in / lead-out silence padding."""

    def __init__(self, source, lead_in_frames=0, lead_out_frames=0):
        self.source = source
        self.lead_in_frames = int(lead_in_frames)
        self.lead_out_frames = int(lead_out_frames)
        if self.lead_in_frames < 0 or self.lead_out_frames < 0:
            raise ValueError("lead_in_frames and lead_out_frames must be non-negative")
        self.chunk_size = int(getattr(source, "chunk_size", 64))
        self.num_frames = source.num_frames + self.lead_in_frames + self.lead_out_frames
        self.height = source.height
        self.width = source.width

    def _yield_repeated(self, frame, count):
        remaining = count
        while remaining > 0:
            batch = min(remaining, self.chunk_size)
            yield np.repeat(frame[None], batch, axis=0)
            remaining -= batch

    def iter_chunks(self):
        if self.lead_in_frames > 0:
            first = self.source.get_frame(0)
            yield from self._yield_repeated(first, self.lead_in_frames)
        last = None
        for chunk in self.source.iter_chunks():
            if len(chunk) > 0:
                last = chunk[-1]
            yield chunk
        if self.lead_out_frames > 0:
            if last is None:
                last = self.source.get_frame(self.source.num_frames - 1)
            yield from self._yield_repeated(last, self.lead_out_frames)

    def get_frame(self, index):
        index = int(index)
        if index < 0 or index >= self.num_frames:
            raise IndexError(
                f"frame index {index} out of range for {self.num_frames} frames"
            )
        src_index = min(max(index - self.lead_in_frames, 0), self.source.num_frames - 1)
        return self.source.get_frame(src_index)


class TrimmingFrameWriter:
    """Drop lead-in / lead-out padded frames before forwarding to an underlying writer."""

    def __init__(self, writer, skip_start=0, keep_frames=None):
        self.writer = writer
        self.skip_start = int(skip_start)
        self.keep_frames = None if keep_frames is None else int(keep_frames)
        if self.skip_start < 0:
            raise ValueError("skip_start must be non-negative")
        if self.keep_frames is not None and self.keep_frames < 0:
            raise ValueError("keep_frames must be non-negative")
        self.seen = 0
        self.written = 0

    def write(self, frames):
        count = len(frames)
        if count == 0:
            return
        chunk_start = self.seen
        chunk_end = chunk_start + count
        self.seen = chunk_end

        keep_start = self.skip_start
        keep_end = (
            chunk_end
            if self.keep_frames is None
            else self.skip_start + self.keep_frames
        )
        lo = max(chunk_start, keep_start)
        hi = min(chunk_end, keep_end)
        if hi > lo:
            sliced = frames[lo - chunk_start:hi - chunk_start]
            self.writer.write(sliced)
            self.written += len(sliced)


def dub_video_stream(
    models,
    frame_source,
    audio,
    writer,
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
    harmonization_config=None,
):
    """Bounded-RAM streaming equivalent of ``dub_video`` for long videos."""
    pipe, whisper, adapter, zero_latents = models
    n, H, W = frame_source.num_frames, frame_source.height, frame_source.width
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

    lm, _ = detect_landmarks_iter(frame_source.iter_chunks(), n, H, W)
    lm, rejected = clean_landmark_track(lm, H, W)
    if not np.isfinite(lm).any():
        lm, _ = detect_landmarks_iter(
            frame_source.iter_chunks(),
            n,
            H,
            W,
            min_detection_confidence=0.05,
            min_tracking_confidence=0.1,
        )
        lm, rejected = clean_landmark_track(lm, H, W)
        if not np.isfinite(lm).any():
            raise RuntimeError(_NO_FACE_ERROR)
        log(
            "default face detector missed all frames; "
            "using low-confidence FaceMesh fallback"
        )
    if rejected:
        log(f"filtered {rejected} outlier face-landmark frame(s)")
    paste_frame = _face_paste_mask(lm)
    if not paste_frame.all():
        log(
            f"face detected on {int(paste_frame.sum())}/{n} frames "
            "(remaining frames will be passed through unchanged)"
        )
    face_sizes = np.maximum(np.ptp(lm[:, :, 0], 1), np.ptp(lm[:, :, 1], 1))
    face = np.nanmedian(face_sizes)
    if not np.isfinite(face):
        raise RuntimeError("face landmark extent is not finite")
    pad = max(0, int(np.ceil((face * CROP_SCALE - min(H, W)) / 2)) + 1)
    if pad <= 1:
        pad = 0
    use_stabilized = (not window_box) and _needs_stabilized_tracking(
        lm + pad, crop_box(lm + pad, H + 2 * pad, W + 2 * pad)
    )
    if use_stabilized:
        pad = max(pad, _stabilized_edge_pad(lm, H, W))
    if pad > 1:
        lm = lm + pad
    else:
        pad = 0
    ref_idx = int(np.nanargmax(lm[:, LM_LIP_LOW, 1] - lm[:, LM_LIP_UP, 1]))
    ref_raw = frame_source.get_frame(ref_idx)
    ref_padded = (
        np.pad(ref_raw, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        if pad > 1 else ref_raw
    )

    def region(lm_w):
        box = crop_box(lm_w, H + 2 * pad, W + 2 * pad)
        mask = lower_face_mask(lm_w, box)
        return box, mask, cv2.GaussianBlur(mask, (0, 0), feather).astype(np.float32) / 255

    def crop(f, box):
        x0, y0, side = box
        return cv2.resize(f[y0:y0 + side, x0:x0 + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    effective_harmonize = harmonize
    frame_boxes = None
    if not window_box:
        if use_stabilized:
            frame_boxes = _stabilized_crop_boxes(lm, H + 2 * pad, W + 2 * pad)
            box = frame_boxes[0]
            mask, alpha = _stabilized_lower_face_mask(lm, frame_boxes, feather)
            log("significant head motion detected; enabling stabilized per-frame face tracking")
        else:
            box, mask, alpha = region(lm)
        box_for = (lambda idx: frame_boxes[idx]) if frame_boxes is not None else (lambda idx: box)
        ref_crop = crop(ref_padded, box_for(ref_idx))
        if effective_harmonize:
            try:
                skin_anchor_mask(mask > 127, harmonization_config or HarmonizationConfig())
            except ValueError as exc:
                log(f"color harmonization skipped: {exc} (falling back to feathered blending)")
                effective_harmonize = False
        paste_alpha = (mask > 127).astype(np.float32) if effective_harmonize else alpha

    chunk_iter = iter(frame_source.iter_chunks())
    src_buf = deque()
    crop_buf = []
    pasted_buf = []
    buf_start = 0
    src_end = 0
    written = 0

    def ensure_src(target_end):
        nonlocal src_end
        while src_end < target_end:
            chunk = next(chunk_iter)
            if pad > 1:
                chunk = np.pad(
                    chunk, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge"
                )
            for frame in chunk:
                src_buf.append(frame)
            src_end += len(chunk)

    for window_index, start in enumerate(starts):
        end = start + NUM_FRAMES
        ensure_src(end)
        window_frames = [src_buf[idx - buf_start] for idx in range(start, end)]
        window_has_face = bool(paste_frame[start:end].any())
        sampler = sample if sample_fn is None else sample_fn
        window_seed = seed + window_index
        color_report = None
        window_harmonized = False
        if window_box and not window_has_face:
            for offset in range(NUM_FRAMES):
                buf_offset = start - buf_start + offset
                if buf_offset >= len(pasted_buf):
                    pasted_buf.append(src_buf[buf_offset])
            written = max(written, end)
        else:
            if window_box:
                box, mask, alpha = region(lm[start:end])
                ref_crop = crop(ref_padded, box)
                crops = np.stack([crop(f, box) for f in window_frames])
            else:
                crops = np.stack([
                    crop(f, box_for(start + offset))
                    for offset, f in enumerate(window_frames)
                ])
            if window_has_face:
                clip = {
                    "frames": crops,
                    "mask": mask > 127,
                    "ref": ref_crop,
                    "prompt": prompt,
                }
                sample_start = round(start / t_fps * SR)
                sample_count = round(NUM_FRAMES / t_fps * SR)
                a = audio[sample_start:sample_start + sample_count]
                a = np.pad(a, (0, max(0, sample_count - len(a))))
                gen = sampler(
                    pipe, whisper, adapter, zero_latents, clip, a,
                    steps=steps, audio_cfg=audio_cfg, seed=window_seed, fps=t_fps)
                if effective_harmonize:
                    try:
                        if harmonization_config is None:
                            gen, color_report = harmonize_lower_face(gen, crops, mask > 127)
                        else:
                            gen, color_report = harmonize_lower_face(
                                gen, crops, mask > 127, config=harmonization_config
                            )
                        window_harmonized = True
                    except ValueError as exc:
                        log(
                            f"window {window_index}: color harmonization skipped: {exc} "
                            "(falling back to feathered blending)"
                        )
                        if not window_box:
                            effective_harmonize = False
                            paste_alpha = alpha
            else:
                gen = crops
            if window_box:
                paste_alpha = (mask > 127).astype(np.float32) if window_harmonized else alpha
                for offset, g in enumerate(gen):
                    buf_offset = start - buf_start + offset
                    pasted = (
                        paste_back(src_buf[buf_offset], g, box, paste_alpha)
                        if paste_frame[start + offset]
                        else src_buf[buf_offset]
                    )
                    if buf_offset < len(pasted_buf):
                        pasted_buf[buf_offset] = pasted
                    else:
                        pasted_buf.append(pasted)
                written = max(written, end)
            else:
                ov = max(0, written - start)
                if ov:
                    weight = (
                        np.arange(1, ov + 1, dtype=np.float32) / (ov + 1)
                    )[:, None, None, None]
                    tail_start = start - buf_start
                    existing = np.stack(crop_buf[tail_start:tail_start + ov])
                    blended = np.rint(
                        existing.astype(np.float32) * (1 - weight)
                        + gen[:ov].astype(np.float32) * weight
                    ).astype(np.uint8)
                    for offset, blended_frame in enumerate(blended):
                        crop_buf[tail_start + offset] = blended_frame
                crop_buf.extend(gen[ov:])
                written = max(written, end)
        detail = "" if window_has_face else ", skipped (no face)"
        if color_report is not None:
            detail = (
                f", chroma MAE {color_report['interior_chroma_mae_before']:.3f}"
                f"->{color_report['interior_chroma_mae_after']:.3f}"
            )
        log(
            f"window {window_index}: {start}-{start + NUM_FRAMES}, "
            f"seed {window_seed}{detail}")

        finalize_until = (
            starts[window_index + 1] if window_index + 1 < len(starts) else n
        )
        count = finalize_until - buf_start
        if count > 0:
            out_chunk = np.empty((count, H, W, 3), dtype=np.uint8)
            for offset in range(count):
                if window_box:
                    pasted = pasted_buf[offset]
                elif paste_frame[buf_start + offset]:
                    pasted = paste_back(
                        src_buf[offset], crop_buf[offset], box_for(buf_start + offset), paste_alpha
                    )
                else:
                    pasted = src_buf[offset]
                out_chunk[offset] = pasted[pad:pad + H, pad:pad + W]
            writer.write(out_chunk)
            for _ in range(count):
                src_buf.popleft()
            if window_box:
                del pasted_buf[:count]
            else:
                del crop_buf[:count]
            buf_start = finalize_until

    if written != n:
        raise RuntimeError(f"window merge wrote {written} frames, expected {n}")
    return written


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
