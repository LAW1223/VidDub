"""Per video: face landmarks (mediapipe), 16k mono wav, SyncNet offset/confidence.

Outputs:
  data/face/<sub>/<id>.npz            lm [T,468,2] float16 (nan = no face), nfaces [T]
  data/audio/<sub>/<id>.wav           16kHz mono
  data/manifest/s01/shard_XXXX.jsonl  per-video stats (append; resumable)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import decord
import mediapipe as mp
import numpy as np
import python_speech_features
import torch
from scipy import signal
from scipy.io import wavfile

from .common import DATA, MANIFEST, PROJECT, WEIGHTS, read_jsonl, resolve_video_path, shard

LATENTSYNC_ROOT = Path(os.environ.get("VIDDUB_LATENTSYNC_ROOT", WEIGHTS / "LatentSync"))

SYNCNET_PATH = Path(os.environ.get(
    "VIDDUB_SYNCNET_PATH", WEIGHTS / "LatentSync-1.5/auxiliary/syncnet_v2.model"))
LM_NOSE, LM_CHEEK_R, LM_CHEEK_L, LM_LIP_UP, LM_LIP_LOW, LM_FOREHEAD, LM_CHIN = 1, 234, 454, 13, 14, 10, 152


def sub_dir(vid):
    return vid[-6:-3]


def load_syncnet(device):
    syncnet_root = LATENTSYNC_ROOT / "eval/syncnet"
    if str(syncnet_root) not in sys.path:
        sys.path.insert(0, str(syncnet_root))
    from syncnet import S
    net = S(num_layers_in_fc_layers=1024).to(device).eval()
    net.load_state_dict(torch.load(SYNCNET_PATH, map_location="cpu", weights_only=True))
    return net


def detect_landmarks(
    frames,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    T, H, W = frames.shape[:3]
    lm = np.full((T, 468, 2), np.nan, dtype=np.float32)
    nfaces = np.zeros(T, dtype=np.int8)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        for t in range(T):
            res = face_mesh.process(frames[t])
            if not res.multi_face_landmarks:
                continue
            nfaces[t] = len(res.multi_face_landmarks)
            faces = [np.array([(p.x * W, p.y * H) for p in f.landmark[:468]]) for f in res.multi_face_landmarks]
            lm[t] = max(faces, key=lambda a: np.ptp(a[:, 1]))  # largest face
    return lm, nfaces


def syncnet_crops(frames, lm, crop_scale=0.4):
    """Replicates syncnet_python crop (S3FD box -> landmark bbox), median-filtered track."""
    valid = ~np.isnan(lm[:, 0, 0])
    idx = np.arange(len(lm))
    xs = np.interp(idx, idx[valid], (np.nanmin(lm[valid, :, 0], 1) + np.nanmax(lm[valid, :, 0], 1)) / 2)
    ys = np.interp(idx, idx[valid], (np.nanmin(lm[valid, :, 1], 1) + np.nanmax(lm[valid, :, 1], 1)) / 2)
    ss = np.interp(idx, idx[valid], np.maximum(np.ptp(lm[valid, :, 0], 1), np.ptp(lm[valid, :, 1], 1)) / 2)
    xs, ys, ss = (signal.medfilt(v, kernel_size=13) for v in (xs, ys, ss))
    crops = []
    for t in range(len(frames)):
        bs = ss[t]
        bsi = int(bs * (1 + 2 * crop_scale))
        img = np.pad(frames[t], ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my, mx = ys[t] + bsi, xs[t] + bsi
        face = img[int(my - bs):int(my + bs * (1 + 2 * crop_scale)), int(mx - bs * (1 + crop_scale)):int(mx + bs * (1 + crop_scale))]
        crops.append(cv2.resize(face, (224, 224)))
    return np.stack(crops)


@torch.no_grad()
def syncnet_score(net, crops_rgb_25fps, audio_int16, device, vshift=15, batch=64):
    im = crops_rgb_25fps[..., ::-1].astype(np.float32)  # syncnet_python feeds cv2 BGR frames
    im = torch.from_numpy(np.ascontiguousarray(im)).permute(3, 0, 1, 2)[None]  # 1,3,T,224,224
    mfcc = np.stack(python_speech_features.mfcc(audio_int16, 16000), 1)  # 13, T_a
    cc = torch.from_numpy(mfcc.astype(np.float32))[None, None]
    n = min(im.shape[2], len(audio_int16) // 640) - 5
    im_feat, cc_feat = [], []
    for i in range(0, n, batch):
        r = range(i, min(n, i + batch))
        im_feat.append(net.forward_lip(torch.cat([im[:, :, v:v + 5] for v in r]).to(device)).cpu())
        cc_feat.append(net.forward_aud(torch.cat([cc[:, :, :, v * 4:v * 4 + 20] for v in r]).to(device)).cpu())
    im_feat, cc_feat = torch.cat(im_feat), torch.cat(cc_feat)
    win = 2 * vshift + 1
    ccp = torch.nn.functional.pad(cc_feat, (0, 0, vshift, vshift))
    dists = torch.stack([torch.nn.functional.pairwise_distance(im_feat[[i]].repeat(win, 1), ccp[i:i + win]) for i in range(n)], 1)
    mean_d = dists.mean(1)
    min_d, minidx = mean_d.min(0)
    return int(vshift - minidx), float(min_d), float(mean_d.median() - min_d)


def analyze(row, net, device):
    vid = row["id"]
    video_path = resolve_video_path(row)
    vr = decord.VideoReader(str(video_path), num_threads=2)
    fps = float(vr.get_avg_fps())
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # T,H,W,3 RGB
    T, H, W = frames.shape[:3]

    wav_path = DATA / "audio" / sub_dir(vid) / f"{vid}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le", str(wav_path)], check=True)
    _, audio = wavfile.read(wav_path)

    lm, nfaces = detect_landmarks(frames)
    face_path = DATA / "face" / sub_dir(vid) / f"{vid}.npz"
    face_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(face_path, lm=lm.astype(np.float16), nfaces=nfaces)

    out = {"id": vid, "n_frames": T, "fps": fps, "height": H, "width": W, "dur_audio": len(audio) / 16000,
           "face_missing": float((nfaces == 0).mean()), "multi_face": float((nfaces > 1).mean())}
    valid = nfaces > 0
    if valid.sum() < 0.5 * T:
        return out
    v = lm[valid]
    face_h = v[:, LM_CHIN, 1] - v[:, LM_FOREHEAD, 1]
    yaw = (v[:, LM_NOSE, 0] - v[:, LM_CHEEK_R, 0]) / (v[:, LM_CHEEK_L, 0] - v[:, LM_CHEEK_R, 0] + 1e-6)
    mouth = (v[:, LM_LIP_LOW, 1] - v[:, LM_LIP_UP, 1]) / face_h
    out.update({"face_h": float(np.median(face_h)), "yaw_ok": float(((yaw > 0.3) & (yaw < 0.7)).mean()),
                "mouth_std": float(mouth.std())})

    crops = syncnet_crops(frames, lm)
    idx25 = (np.arange(int(T * 25 / fps)) * fps / 25).astype(int)
    offset, min_dist, conf = syncnet_score(net, crops[idx25], audio, device)
    out.update({"sync_offset": offset, "sync_dist": min_dist, "sync_conf": conf})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = read_jsonl(MANIFEST / "s00_manifest.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    rows = shard(rows, args.shard, args.num_shards)
    out_path = MANIFEST / "s01" / f"shard_{args.shard:04d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = [r for r in rows if r["id"] not in done]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_syncnet(device)
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, row in enumerate(rows):
            try:
                res = analyze(row, net, device)
            except Exception as e:  # corrupt video etc.: record and continue
                res = {"id": row["id"], "error": repr(e)[:300]}
            f.write(json.dumps(res) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[shard {args.shard}] {i}/{len(rows)} {(time.time() - t0) / (i + 1):.2f}s/it", flush=True)


if __name__ == "__main__":
    main()
