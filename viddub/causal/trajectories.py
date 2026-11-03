"""Validated on-disk causal teacher trajectories (one clip per .pt shard)."""
import json
from pathlib import Path

import torch

from .sampling import schedule


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, directory, cfg):
        source = Path(directory).resolve()
        self.manifest_path = source if source.is_file() else source / 'manifest.json'
        self.directory = self.manifest_path.parent
        self.cfg = cfg
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != 1 or not self.manifest.get("complete"):
            raise ValueError("Trajectory generation has not completed")
        for key in ("chunk_size", "num_ref", "fps", "num_frames", "shift"):
            if self.manifest["config"][key] != cfg[key]:
                raise ValueError(f"Trajectory/config mismatch for {key}")
        if self.manifest["student_steps"] != cfg["student_steps"]:
            raise ValueError("Trajectory student step count mismatch")
        if self.manifest["teacher_cfg"] != cfg["teacher_cfg"]:
            raise ValueError("Trajectory guidance mismatch")
        self.records = self.manifest["records"]
        if not self.records:
            raise ValueError("No trajectories")
        self.indices = self.manifest["boundary_indices"][:-1]
        self.sigmas = self.manifest["teacher_sigmas"]
        expected = schedule(cfg["student_steps"], cfg["shift"])
        actual = torch.tensor([self.sigmas[i] for i in self.manifest["boundary_indices"]])
        if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise ValueError("Teacher trajectory boundaries do not match student schedule")
        for record in self.records:
            path = (self.directory / record["file"]).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise ValueError(f"Missing/unsafe trajectory path: {path}")
            if "bytes" in record and path.stat().st_size != record["bytes"]:
                raise ValueError(f"Trajectory size mismatch: {path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = torch.load(self.directory / record["file"], map_location="cpu", weights_only=True)
        if any(self.cfg.get(k, 0) for k in ('ode_gt_flow_weight','pixel_weight','lpips_weight',
                                            'sync_weight','keep_weight')):
            if data.get('gt_latents') is None:
                raise ValueError('This ODE variant requires trajectories with gt_latents')
            if data.get('conditions', {}).get('gt_frames') is None:
                raise ValueError('This ODE variant requires trajectories with pixel GT')
        chunks = data["chunks"]
        chunk = chunks[int(torch.randint(len(chunks), ()))]
        step = int(torch.randint(len(self.indices), ()))
        i = self.indices[step]
        a, b = chunk["start"], chunk["end"]
        conditions = dict(data["conditions"])
        conditions["cross"] = data.get("mode", record.get("mode", "same")) == "cross"
        return {"shape_reference": data["shape_reference"], "conditions": conditions,
                "record": {"start": a, "end": b, "sigma": self.sigmas[i],
                           "noisy": chunk["states"][i], "target": chunk["states"][-1],
                           "prefix": data["generated"][:, :, :a],
                           "gt_x0": data.get("gt_latents")},
                "clip_id": data["clip_id"]}
