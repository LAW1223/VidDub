# Training guide

Run all commands from the VidDub repository root. The examples use one
machine for readability; use standard Accelerate configuration for DDP.
Training media and model caches are not bundled with the source repository.
Prepare authorized media with the public data pipeline and set
`VIDDUB_DATA_ROOT` and `VIDDUB_WEIGHT_ROOT` before launching training.

## 1. Prepare data

```bash
python -m data.s00_build_manifest --input data/source_manifest.jsonl
python -m data.s01_analyze --shard 0 --num_shards 1
python -m data.s02_make_clips --shard 0 --num_shards 1
python -m data.s03_cache_latents --shard 0 --num_shards 1
```

For distributed preprocessing, assign a unique `--shard` in
`[0, --num_shards)` to each process. The scripts are resumable through their
per-shard manifests.

Create `data/manifest/train.json` for distillation. It is a JSON array of
processed s02 rows. Each row minimally needs `clip_id` and `dir`; fixed
counterfactual pairing may add `audio_cross_id`.

## 2. Full-context teacher

Stage S1 trains only the 287M audio adapter. Stage S2 unfreezes VACE and DiT.

```bash
accelerate launch -m viddub.full_context.train \
  --output outputs/s1 --stage s1 --num_frames 49 \
  --lr 1e-4 --warmup 200 --max_steps 3000

accelerate launch -m viddub.full_context.train \
  --output outputs/s2 --stage s2 --num_frames 81 \
  --init_adapter outputs/s1/adapter_step003000.safetensors \
  --lr 5e-5 --lr_base 1e-5 --warmup 500 --max_steps 6000
```

The published full-context parent uses the S3 recipe below. It starts from the
selected S2 checkpoint, trains for 300 updates, and exports the EMA state at
step 200. Paths and data weights are artifact-relative placeholders; the
optimization flags match the published teacher run.

```bash
accelerate launch -m viddub.full_context.train \
  --output outputs/s3 --stage s2 --num_frames 81 \
  --init_adapter weights/s2-selected/adapter_step002500.safetensors \
  --init_dit weights/s2-selected/dit_step002500.safetensors \
  --init_vace weights/s2-selected/vace_step002500.safetensors \
  --data h3,hallo3,ls_real,astri --sampling mix \
  --domain_weights H3=30,TalkVid=20,SpeakerVid=15,HDTF=12,hallo3=10,VoxCeleb2=8 \
  --clip_weights data/manifest/quality/weights_v1.json \
  --mask_lambda 2 --p_drop_text 1 --p_drop_audio 0.1 --p_drop_ref 0.1 \
  --p_cross 0.3 --sync_weight 0.02 --sync_sigma_max 0.8 \
  --pix_weight 0 --lpips_weight 0.1 \
  --repa art --repa_weight 0.1 --repa_layer 8 --repa_lr 1e-4 --repa_warmup 50 \
  --dit_lora_rank 64 --freeze_dit --ema 0.99 \
  --lr 1e-5 --lr_base 2e-6 --lora_lr 5e-5 \
  --grad_accum 3 --warmup 50 --lr_schedule const \
  --workers 3 --skip_val0 --save_every 50 --max_steps 300
```

The full S1/S2 history used resumed runs; their selected cumulative update counts
are 17,700 and 5,500, respectively. For distillation, start from the published
`sft-r2a-ema200` atomic checkpoint rather than approximating that history.

## 3. Full-context teacher trajectories

Generate matched paired- and replacement-audio trajectories directly from the
replacement-aware full-context teacher:

```bash
python -m viddub.causal.make_trajectories \
  --config configs/full_context/ode2.json \
  --teacher-dir weights/sft-r2a-ema200 --teacher-step 200 \
  --teacher-steps 48 --audio-mode both \
  --output outputs/full2-trajectories
```

The trajectory manifest records `causal: false`; the trainer rejects a
trajectory/student architecture mismatch.

## 4. Two-step trajectory regression

Validate inputs before allocating GPUs:

```bash
python -m viddub.causal.train_ode \
  --config configs/full_context/ode2.json \
  --init-dir weights/sft-r2a-ema200 --init-step 200 \
  --trajectories outputs/full2-trajectories \
  --output outputs/full2-ode --check-only
```

Train the full-context two-step initializer:

```bash
accelerate launch -m viddub.causal.train_ode \
  --config configs/full_context/ode2.json \
  --init-dir weights/sft-r2a-ema200 --init-step 200 \
  --trajectories outputs/full2-trajectories \
  --output outputs/full2-ode
```

The ODE schedule used for inference must match the trained `student_steps` and
`shift` fields.

## 5. Full-context distribution matching

Pretrain a shared fake-score model from the selected ODE student:

```bash
accelerate launch -m viddub.causal.pretrain_fake \
  --config configs/full_context/fake2.json \
  --student-dir outputs/full2-ode --student-step 400 \
  --teacher-dir weights/sft-r2a-ema200 --teacher-step 200 \
  --output outputs/full2-fake
```

Run DMD:

```bash
accelerate launch -m viddub.causal.train_dmd \
  --config configs/full_context/dmd2.json \
  --init-dir outputs/full2-ode --init-step 400 \
  --teacher-dir weights/sft-r2a-ema200 --teacher-step 200 \
  --fake-dir outputs/full2-fake --fake-step 200 \
  --output outputs/full2-dmd
```

The selected recipe uses two denoising steps, teacher CFG 2, a 200-update
fake-score initializer, and 100 DMD generator updates. The exact lineage is:

| Stage | Published checkpoint |
|---|---|
| Full-context parent | `sft-r2a-ema200` |
| Full-context two-step initializer | `FULL2_ODE_R3@400` |
| Full-context fake-score initializer | `FULL2_FAKE_R3@200` |
| Final model | `FULL2_DMD_R4@100` |

## 6. Optional causal ablation

The `configs/causal/` recipes and `viddub.causal.cf` package reproduce the
full-context-versus-causal ablation. They are research branches, not the
selected VidDub model.

```bash
python -m viddub.causal.cf.train \
  --config /path/to/cd-config.json \
  --init-dir outputs/ar --init-step 400 \
  --output outputs/cd
```

See `viddub/causal/cf/NOTICE` and source comments for differences from upstream.

## 7. Verification and resume

Use `--check-only` before GPU jobs. Checkpoint metadata records and validates the
full-context/causal architecture flag, sampling schedule, and component hashes.

Training state directories are written atomically as `state_stepNNNNNN`. Resume
only into the same output directory and with identical configuration, data
provenance, checkpoint fingerprints, and world size.
