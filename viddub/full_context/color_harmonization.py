"""Color harmonization for source-video lower-face editing."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HarmonizationConfig:
    feather_px: float = 18.0
    anchor_side_fraction: float = 0.22
    anchor_top_fraction: float = 0.24
    temporal_window: int = 9
    max_lab_delta: tuple[float, float, float] = (18.0, 20.0, 20.0)
    spatial_sigma_px: float = 24.0
    luminance_field_strength: float = 0.35
    chroma_field_strength: float = 0.90
    max_spatial_delta: tuple[float, float, float] = (18.0, 30.0, 30.0)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("editable mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def skin_anchor_mask(mask: np.ndarray, config: HarmonizationConfig) -> np.ndarray:
    """Select cheek and upper-mask anchors while excluding the central mouth."""
    mask = np.asarray(mask, dtype=bool)
    left, top, right, bottom = _bbox(mask)
    width, height = right - left, bottom - top
    side = max(4, round(width * config.anchor_side_fraction))
    upper = max(4, round(height * config.anchor_top_fraction))
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    geometry = (xx < left + side) | (xx >= right - side) | (yy < top + upper)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    anchors = mask & geometry & interior
    if anchors.sum() < 64:
        raise ValueError("editable mask has too few skin anchors")
    return anchors


def feather_alpha(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """Return an alpha that is zero outside and ramps inward from every edge."""
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    padded = np.pad(binary, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    alpha = np.clip(distance / max(float(feather_px), 1.0), 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _smooth_deltas(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window) | 1)
    radius = window // 2
    if radius == 0:
        return values
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.median(padded[index:index + window], axis=0)
         for index in range(len(values))]
    )


def estimate_lab_deltas(
    generated: np.ndarray,
    source: np.ndarray,
    anchors: np.ndarray,
    config: HarmonizationConfig,
) -> np.ndarray:
    """Estimate temporally stable source-minus-generated Lab offsets."""
    if generated.shape != source.shape or generated.ndim != 4 or generated.shape[-1] != 3:
        raise ValueError("generated and source must be matching [T,H,W,3] RGB arrays")
    raw = []
    for generated_frame, source_frame in zip(generated, source):
        generated_lab = cv2.cvtColor(
            generated_frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_lab = cv2.cvtColor(
            source_frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        valid = anchors & (source_lab[..., 0] > 35) & (source_lab[..., 0] < 235)
        if valid.sum() < 64:
            valid = anchors
        raw.append(np.median(source_lab[valid] - generated_lab[valid], axis=0))
    deltas = _smooth_deltas(
        np.asarray(raw, dtype=np.float32), config.temporal_window)
    limits = np.asarray(config.max_lab_delta, dtype=np.float32)
    return np.clip(deltas, -limits, limits)


def harmonize_lower_face(
    generated: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    config: HarmonizationConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Correct masked-region color and composite it into exact source context."""
    config = config or HarmonizationConfig()
    generated = np.asarray(generated, dtype=np.uint8)
    source = np.asarray(source, dtype=np.uint8)
    anchors = skin_anchor_mask(mask, config)
    alpha = feather_alpha(mask, config.feather_px)[..., None].astype(np.float32)
    deltas = estimate_lab_deltas(generated, source, anchors, config)

    output = np.empty_like(generated)
    before = []
    after = []
    interior = np.asarray(mask, dtype=bool) & (alpha[..., 0] > 0.95)
    for index, (generated_frame, source_frame) in enumerate(
        zip(generated, source)
    ):
        generated_lab = cv2.cvtColor(
            generated_frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_lab = cv2.cvtColor(
            source_frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        base_lab = generated_lab + deltas[index]
        field = cv2.GaussianBlur(
            source_lab - base_lab,
            (0, 0),
            config.spatial_sigma_px,
            borderType=cv2.BORDER_REFLECT,
        )
        field[..., 0] *= config.luminance_field_strength
        field[..., 1:] *= config.chroma_field_strength
        limits = np.asarray(config.max_spatial_delta, dtype=np.float32)
        field = np.clip(field, -limits, limits)
        corrected_lab = np.clip(base_lab + field, 0, 255).astype(np.uint8)
        corrected = cv2.cvtColor(
            corrected_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        composite = (
            source_frame.astype(np.float32) * (1.0 - alpha) + corrected * alpha
        )
        output[index] = np.clip(np.rint(composite), 0, 255).astype(np.uint8)
        before.append(np.mean(np.abs(
            generated_lab[interior][:, 1:] - source_lab[interior][:, 1:]
        )))
        output_lab = cv2.cvtColor(
            output[index], cv2.COLOR_RGB2LAB).astype(np.float32)
        after.append(np.mean(np.abs(
            output_lab[interior][:, 1:] - source_lab[interior][:, 1:]
        )))

    report = {
        "method": "low_frequency_lab_field_source_feather",
        "feather_px": config.feather_px,
        "spatial_sigma_px": config.spatial_sigma_px,
        "luminance_field_strength": config.luminance_field_strength,
        "chroma_field_strength": config.chroma_field_strength,
        "anchor_pixels": int(anchors.sum()),
        "median_lab_delta": np.median(deltas, axis=0).round(4).tolist(),
        "interior_chroma_mae_before": float(np.mean(before)),
        "interior_chroma_mae_after": float(np.mean(after)),
    }
    return output, report
