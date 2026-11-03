# Release validation

This note records the equivalence checks used for the initial VidDub release.
The release implementation was compared against the research implementation on
2026-10-16 using eight nodes with eight NVIDIA H20 GPUs per node.

## Scope

The checks cover:

- the selected full-context parent and causal checkpoint lineage;
- published training recipes, dataset rows, and ODE trajectories;
- one optimizer update for AR adaptation, ODE regression, fake-score training,
  and DMD refinement; and
- deterministic inference from the paper-selected DMD checkpoint.

The full-context SFT recipe was checked field by field against its recorded run.
The four causal stages were additionally executed as paired reference/release
training probes.

## Training contract

All five recorded recipes (SFT, AR, ODE, fake score, and DMD) match the
corresponding research runs. The causal dataset contains 16,384 canonical rows
with SHA-256
`1476f7554e2bbab0130c1a9f24bae29b4275f39cdd0faa24100f3fbdd48150c6`.
The ODE trajectory manifest contains 4,096 complete records and matches the
recorded byte count and SHA-256.

The audit identified and corrected two stale release settings before the
equivalence runs:

- ODE `rollout_quality_weight` was restored to
  `0.3664845873554578`.
- Fake-score `dmd_split_quality_backward` was restored to `false`.

## Paired one-update probes

Each pair used the same data, seed, initialization, configuration, and
eight-GPU world size. Differences below are measured after one optimizer
update.

| Stage | Loss comparison | Maximum tensor absolute error | Maximum relative L2 error |
| --- | --- | ---: | ---: |
| AR | exact | `1.946e-7` | `5.802e-8` |
| ODE | exact | `1.192e-7` | `5.061e-8` |
| Fake score | exact | `3.912e-8` | `1.477e-8` |
| DMD | relative difference `2.48e-4` | `5.960e-8` | `8.118e-8` |

The small DMD scalar-loss difference is within the `1e-3` distributed-metric
tolerance. DMD alternates online fake-score updates and therefore inherits
cross-node NCCL reduction-order noise. The resulting model update remains
numerically equivalent under the stricter tensor criteria of `1e-6` maximum
absolute error and `1e-7` relative L2 error.

## Final-checkpoint inference

Reference and release inference were run on eight fixed 81-frame validation
clips at 512 by 512 resolution. Both implementations used the same paper
checkpoint, input tensors, seed, two denoising steps, and single audio branch.

- Passed clips: 8 of 8.
- Maximum pixel error: 0.
- Mean pixel error: 0.
- Exact pixel fraction: 1.0 for every clip.
- Median paired denoising-time ratio (release/reference): 1.003.

Thus repository reorganization does not change final-checkpoint model output
for the tested inputs. Timing was collected only as a regression guard and is
not an independent throughput benchmark.

## Integrity and release checks

All 15 component files across the five staged training artifacts match their
recorded byte counts and SHA-256 digests.
The source package and exported model package pass the release audit. The test
suite checks the consolidated package layout, selected paper lineage,
two-step/single-branch inference contract, model-package checksums, command-line
entry points, and VidDub Individual Commercial and Research License
metadata.

## Interpretation

These checks establish configuration parity, one-update numerical equivalence
for every causal training stage, and exact inference equivalence for the
released paper checkpoint. They do not replace a fresh end-to-end rerun of all
training stages or a new evaluation of every paper benchmark. Consequently,
the evidence supports preserving the performance of the released checkpoint;
it should not be described as an independent reproduction of the complete
training run.
