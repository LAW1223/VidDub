"""Filter s01 results and cut training clips.

Per clip (data/clips/<sub>/<clip_id>/):
  video.mp4   512x512, 81 frames, 24fps (face-centred crop, fixed box per clip)
  mask.png    512x512 uint8, 255 = region to generate (lower face, union over the clip)
  ref.png     512x512 face frame from outside the clip window (same crop box)
  audio.wav   16kHz mono, aligned to the clip (SyncNet offset corrected)
Rows -> data/manifest/s02/shard_XXXX.jsonl
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import decord
import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile

from .common import DATA, MANIFEST, read_jsonl, resolve_video_path, shard

NUM_FRAMES, SIZE, CROP_SCALE = 81, 512, 1.8
LM_NOSE_BOTTOM, LM_CHEEK_R, LM_CHEEK_L, LM_CHIN, LM_LIP_UP, LM_LIP_LOW, LM_FOREHEAD = 2, 234, 454, 152, 13, 14, 10
FILTERS = {  # decided from the s01 distribution, see STATE.md
    "face_missing_max": 0.02, "multi_face_max": 0.02, "face_h_min": 150, "yaw_ok_min": 0.7,
    "sync_conf_min": 6.0, "sync_offset_range": (-4, 2), "mouth_std_min": 0.01,
}


def keep(r):
    f = FILTERS
    return ("error" not in r and "sync_conf" in r and r["face_missing"] <= f["face_missing_max"]
            and r["multi_face"] <= f["multi_face_max"] and r["face_h"] >= f["face_h_min"] and r["yaw_ok"] >= f["yaw_ok_min"]
            and r["sync_conf"] >= f["sync_conf_min"] and f["sync_offset_range"][0] <= r["sync_offset"] <= f["sync_offset_range"][1]
            and r["mouth_std"] >= f["mouth_std_min"] and r["n_frames"] >= NUM_FRAMES)


def clean_prompt(prompt, speech_text):
    """Drop sentences that carry the spoken line (it would conflict with new audio at dubbing time)."""
    prompt = prompt.replace(speech_text, "")
    parts = re.split(r"(?<=[。！？.!?])\s*", prompt)
    bad = re.compile(r"台词|口型|说道|说：|lip|speak|says|saying|dialogue|words", re.I)
    return " ".join(p for p in parts if p.strip() and not bad.search(p)).strip()


def crop_box(lm, H, W):
    cx = (np.nanmin(lm[:, :, 0]) + np.nanmax(lm[:, :, 0])) / 2
    cy = (np.nanmin(lm[:, :, 1]) + np.nanmax(lm[:, :, 1])) / 2
    face = np.nanmedian(np.maximum(np.ptp(lm[:, :, 0], 1), np.ptp(lm[:, :, 1], 1)))
    side = int(min(face * CROP_SCALE, H, W))
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))
    return x0, y0, side


def lower_face_mask(lm, box):
    x0, y0, side = box
    s = SIZE / side
    face_h = np.nanmedian(lm[:, LM_CHIN, 1] - lm[:, LM_FOREHEAD, 1])
    top = np.nanmin(lm[:, LM_NOSE_BOTTOM, 1])
    bottom = np.nanmax(lm[:, LM_CHIN, 1]) + 0.1 * face_h
    left, right = np.nanmin(lm[:, LM_CHEEK_R, 0]), np.nanmax(lm[:, LM_CHEEK_L, 0])
    mask = np.zeros((SIZE, SIZE), np.uint8)
    pts = np.array([[left, top], [right, bottom]])
    (l, t), (r, b) = ((pts - [x0, y0]) * s).round().astype(int)
    mask[max(t, 0):min(b, SIZE), max(l, 0):min(r, SIZE)] = 255
    return mask


def write_video(path, frames):
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
                             "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def load_selection(path):
    """Return source_id -> selected clip_ids from JSON, JSONL, or text."""
    path = Path(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text())
        values = value.keys() if isinstance(value, dict) else value
    else:
        values = []
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                values.append(json.loads(line) if line.lstrip().startswith("{") else line.strip())
    selected = {}
    for value in values:
        clip_id = value["clip_id"] if isinstance(value, dict) else str(value)
        source_id = value.get("id") if isinstance(value, dict) else None
        source_id = source_id or clip_id.rsplit("_", 1)[0]
        selected.setdefault(source_id, set()).add(clip_id)
    if not selected:
        raise ValueError(f"Empty selection: {path}")
    return selected


def make_clips(row, s01, selected=None, done=None):
    vid = row["id"]
    sub = vid[-6:-3]
    face = np.load(DATA / "face" / sub / f"{vid}.npz")
    lm = face["lm"].astype(np.float32)
    T = len(lm)
    vr = decord.VideoReader(str(resolve_video_path(row)), num_threads=2)
    frames = vr.get_batch(list(range(T))).asnumpy()
    H, W = frames.shape[1:3]
    _, audio = wavfile.read(DATA / "audio" / sub / f"{vid}.wav")
    mouth = (lm[:, LM_LIP_LOW, 1] - lm[:, LM_LIP_UP, 1]) / (lm[:, LM_CHIN, 1] - lm[:, LM_FOREHEAD, 1])
    mouth = np.nan_to_num(mouth, nan=-1)
    # best-matching audio for video time t is at t - offset/25 s (offset < 0: audio is late)
    audio_shift = -s01["sync_offset"] / 25.0
    starts = [0] if T - NUM_FRAMES < 16 else [0, T - NUM_FRAMES]
    out = []
    for start in starts:
        clip_id = f"{vid}_{start:03d}"
        if selected is not None and clip_id not in selected:
            continue
        if done is not None and clip_id in done:
            continue
        win = slice(start, start + NUM_FRAMES)
        box = crop_box(lm[win], H, W)
        x0, y0, side = box
        outside = np.r_[0:start, start + NUM_FRAMES:T]
        ref_idx = int(outside[np.argmax(mouth[outside])]) if len(outside) else (T - 1 if start == 0 else 0)
        crop = lambda f: cv2.resize(f[y0:y0 + side, x0:x0 + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)

        d = DATA / "clips" / sub / clip_id
        d.mkdir(parents=True, exist_ok=True)
        write_video(d / "video.mp4", np.stack([crop(f) for f in frames[win]]))
        cv2.imwrite(str(d / "mask.png"), lower_face_mask(lm[win], box))
        cv2.imwrite(str(d / "ref.png"), cv2.cvtColor(crop(frames[ref_idx]), cv2.COLOR_RGB2BGR))
        a0 = int(round((start / 24.0 + audio_shift) * 16000))
        n = int(round(NUM_FRAMES / 24.0 * 16000))
        seg = np.zeros(n, np.int16)
        src = audio[max(a0, 0):max(a0 + n, 0)]
        seg[max(-a0, 0):max(-a0, 0) + len(src)] = src[:n - max(-a0, 0)]
        wavfile.write(d / "audio.wav", 16000, seg)
        out.append({"clip_id": clip_id, "id": vid, "dir": str(d.relative_to(DATA / "clips")), "start": start, "ref_idx": ref_idx,
                    "crop_box": box, "sync_conf": s01["sync_conf"], "sync_offset": s01["sync_offset"],
                    "lang": row.get("speech_language", row.get("language", "")),
                    "prompt": clean_prompt(row.get("prompt", ""), row.get("speech_text", ""))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--selection",
        help="Optional JSON/JSONL/text clip selection; unlisted windows are not materialized",
    )
    args = ap.parse_args()

    selection = load_selection(args.selection) if args.selection else None
    s01 = {}
    for p in sorted((MANIFEST / "s01").glob("shard_*.jsonl")):
        s01.update({r["id"]: r for r in read_jsonl(p)})
    rows = [r for r in read_jsonl(MANIFEST / "s00_manifest.jsonl") if r["id"] in s01 and keep(s01[r["id"]])]
    if selection is not None:
        rows = [r for r in rows if r["id"] in selection]
    if args.limit:
        rows = rows[:args.limit]
    rows = shard(rows, args.shard, args.num_shards)
    out_path = MANIFEST / "s02" / f"shard_{args.shard:04d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {
        r["clip_id"] for r in read_jsonl(out_path) if "clip_id" in r
    } if out_path.exists() else set()
    with open(out_path, "a") as f:
        for i, row in enumerate(rows):
            try:
                clips = make_clips(
                    row,
                    s01[row["id"]],
                    selected=selection.get(row["id"]) if selection is not None else None,
                    done=done,
                )
            except Exception as e:
                clips = [{"id": row["id"], "error": repr(e)[:300]}]
            for c in clips:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                if "clip_id" in c:
                    done.add(c["clip_id"])
            f.flush()
            if i % 50 == 0:
                print(f"[shard {args.shard}] {i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
