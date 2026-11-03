"""One-off: cached-latent path (compute_latents + build_inputs) must match DiffSynth's own units."""
import numpy as np
import torch
from PIL import Image

from .runtime import ZeroLatents, build_inputs, compute_latents, load_clip, load_models, load_rows

pipe, _ = load_models("cuda")
pipe.scheduler.set_timesteps(1000, training=True)
clip = load_clip(load_rows("train")[0], 49)

masked = clip["frames"].copy()
masked[:, clip["mask"]] = 128
mask_img = Image.fromarray(clip["mask"].astype(np.uint8) * 255).convert("RGB")
shared = {"input_video": [Image.fromarray(f) for f in clip["frames"]], "vace_video": [Image.fromarray(f) for f in masked],
          "vace_video_mask": [mask_img] * 49, "vace_reference_image": Image.fromarray(clip["ref"]), "vace_scale": 1,
          "height": 512, "width": 512, "num_frames": 49, "seed": 0, "rand_device": "cpu", "cfg_scale": 1, "cfg_merge": False, "tiled": False}
posi, nega = {"prompt": clip["prompt"], "positive": True}, {}
with torch.no_grad():
    for unit in pipe.units:
        shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
    lat = compute_latents(pipe, clip)
    x0, vace, ctx = build_inputs(lat, clip["mask"], 49, ZeroLatents(pipe), context=lat["context"])

for name, a, b in (("x0", shared["input_latents"], x0), ("vace_context", shared["vace_context"], vace), ("context", posi["context"], ctx)):
    a, b = a.float(), b.float()
    print(f"{name}: shapes {tuple(a.shape)} {tuple(b.shape)} max|diff| {(a - b).abs().max().item():.4f} mean|a| {a.abs().mean().item():.4f}")
a, b = shared["vace_context"].float(), vace.float()
for part, sl in (("inactive", slice(0, 16)), ("reactive", slice(16, 32)), ("mask", slice(32, 96))):
    d = (a[:, sl] - b[:, sl]).abs()
    print(f"  {part}: max {d.max().item():.4f} mean {d.mean().item():.5f} ref-frame max {d[:, :, :1].max().item():.4f}")
