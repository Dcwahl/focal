# Making a decisive exponent safe to ship

**Date:** 2026-09-12
**Question:** the previous round found that raising the focus-weighting exponent roughly
doubles recovered detail but tripled background colour drift on the user's own stacks, so
p4 could not ship. The plan was to gate the exponent on focus confidence. Does that work?

**Answer:** no - the premise was wrong, and measuring it is what showed that. The
regression is not caused by weak focus evidence. It is caused by the frames disagreeing
about *brightness*, and it lives entirely in the coarsest pyramid level. Exempting that
one level fixes it at zero measured cost.

Shipped as `FocusStacker(focus_power=...)`, default `1.0` (bit-identical to previous
output). `--focus-power` on the CLI. Not exposed in the UI.

---

## 1. Confidence gating, built and falsified

`GatedStacker` in `laplacian_weight_experiment.py` sets a per-pixel exponent
`p_eff = 1 + (p-1) * m_max/(m_max + tau)`, where `m_max` is the peak focus measure across
frames and `tau` a floor taken from a low percentile of `m_max`. Pixels with no focus
evidence fall back to p=1; pixels with evidence get the full exponent. A `gate_blur`
option pools `m_max` first, since "no frame is in focus here" is a regional property.

On stack 1's background - the region that blocked p4 - it barely moved:

| method | background lab_drift |
|---|---|
| p1 (shipped) | 9.96 |
| p4 uniform | 22.51 |
| gate p4, pct10 | 20.52 |
| gate p4, pct10 + blur 4 | 22.29 |
| gate p4, pct25 + blur 4 | 21.59 |

A design flaw was visible before the scores: `gated_fraction` came back exactly equal to
`floor_pct` on every scene and every level. A percentile floor gates the bottom N% of
pixels by rank, always, whether or not the scene contains a defocused region at all.

### Why no confidence statistic would have worked

Three candidate statistics, measured on stack 1's frozen diagnostic regions (10 frames,
6000x4000, level 0). `background` is the region that fails; `wire` and `glasses` are
genuinely focused regions that must keep the exponent.

| region | median m_max | its global percentile | peak/mean | argmax local std |
|---|---|---|---|---|
| clip_front | 27.5 | 16.0% | 2.18 | 1.68 |
| wire | 134.2 | 71.7% | 2.52 | 2.04 |
| glasses | 139.5 | 73.6% | 2.71 | 1.60 |
| **background** | **76.2** | **43.2%** | **2.57** | **1.56** |

The background sits mid-distribution, so no floor reaches it. Its peak-to-mean ratio is
indistinguishable from the focused regions. Its winning frame is *more* spatially
coherent than theirs, not less. By every measure of focus confidence, that background
looks like a region where the ranking should be trusted.

## 2. What is actually happening

Cross-frame statistics on the same regions - per-pixel std across the 10 aligned frames,
and the same after low-passing each frame at sigma 12 to strip focus detail:

| region | cross-frame luma std | coarse-only (sigma 12) | frame mean luma range |
|---|---|---|---|
| clip_front | 4.25 | 3.40 | 10.6 .. 12.3 |
| wire | 27.97 | 26.06 | 76.3 .. 129.8 |
| glasses | 31.01 | 29.24 | 96.5 .. 161.5 |
| **background** | **31.13** | **31.04** | **29.8 .. 101.7** |

The background's mean brightness spans a factor of 3.4 across the stack, and essentially
*all* of its cross-frame variation survives a sigma-12 low-pass (31.04 of 31.13). The
frames disagree about exposure there, not about focus. A decisive exponent picks one
frame per pixel and stamps that frame's brightness down in patches. Visually it is not
subtle: p4's background renders pale tan where p1 renders deep red, with a ghost of the
arm that moves through the scene.

The ranking is not noise. It is a confident, coherent answer to the wrong question.

## 3. The fix follows from the pyramid

A Laplacian pyramid already separates these bands. Its coarsest level is the base image,
carrying overall brightness; the finer levels carry detail. So the exponent can be
applied per band: decisive where detail lives, proportional where brightness does.
`coarse_levels` counts exempted levels from the coarsest end.

Stack 1, sobel metric (the non-circular one) and the background probe:

| method | sobel recovery | halo | background lab_drift |
|---|---|---|---|
| p1 (shipped) | 0.416 | 0.000 | 9.96 |
| p4 uniform | 0.790 | 0.001 | 22.51 |
| **p4, base exempt** | **0.788** | **0.001** | **10.02** |
| p4, 2 coarse exempt | 0.784 | 0.001 | 9.97 |
| p4, 3 coarse exempt | 0.779 | 0.000 | 9.96 |
| p8, base exempt | 0.886 | 0.002 | 10.03 |

Exempting the single coarsest level recovers p1's background behaviour and keeps p4's
detail. Exempting more adds nothing, which is itself the confirmation: the entire
regression lived in the base level.

## 4. No cost on the dataset

Frozen 6-scene subset, 30 frames, half resolution, alignment disabled. `p4_c1` is the
level-gated variant:

| method | recovery | halo | flat_excess |
|---|---|---|---|
| laplacian (p1, shipped) | 0.346 | 0.000 | 0.315 |
| laplacian_p2 | 0.592 | 0.000 | 0.395 |
| laplacian_p4 | 0.802 | 0.002 | 0.607 |
| **laplacian_p4_c1** | **0.802** | **0.002** | **0.607** |
| laplacian_p8_c1 | 0.925 | 0.006 | 0.944 |
| helicon_focus (reference) | 0.884 | 0.004 | 1.223 |
| complex_wavelet | 1.026 | 0.234 | 1.873 |

`p4_c1` matches `p4` to three decimals on all six scenes individually, not just in the
mean. Level gating is free here because these scenes have no exposure disagreement - the
dataset is tripod-shot under constant light, which is exactly why it could not have
surfaced the bug on its own.

## 5. A second failure mode, real but separate

`confidence_gate_gt.py` extends the synthetic ground-truth control with zones the earlier
one lacked: depth beyond the focus sweep (never sharp in any frame) and a constant-colour
patch (no texture at any blur). MAE against the known sharp original, 30 frames:

| method (noise=3) | overall | covered | uncovered | flat |
|---|---|---|---|---|
| p1 | 4.38 | 3.29 | 6.90 | 1.04 |
| p4 | 3.24 | 1.90 | 5.88 | 1.20 |
| p8 | 3.23 | 1.89 | 5.76 | **1.64** |
| p4, base exempt | 3.24 | 1.90 | 5.89 | 1.20 |
| gate p4 pct10 | 3.40 | 2.12 | 6.01 | **1.09** |

Here the flat zone genuinely *is* noise-driven ranking, and level gating does not help it
(1.20, unchanged) while confidence gating does (1.20 -> 1.09). So both mechanisms are
real; they simply appear in different places. Level gating addresses the one that blocked
shipping. Confidence gating remains available in `laplacian_weight_experiment.py` for the
noise mechanism, unshipped, at a measured cost of about 0.02 recovery.

This is also the argument for keeping the default exponent at 4 rather than 8: p8 wins on
detail everywhere measured, but pays for it in exactly this zone (1.64 vs 1.20).

## 6. What shipped

- `FocusStacker(focus_power=...)`, default `1.0`. Verified bit-identical to the previous
  output at 1.0, and bit-identical to the measured `lvl4_c1` variant at 4.0.
- The coarsest pyramid level is always weighted proportionally, at any exponent.
- `focus_power` joins `fusion_fingerprint`, so substack caches invalidate on change.
- `FocusStacker.copy()` replaces the field-by-field rebuild in `MainWindow._run_stack`.
  That rebuild would have silently dropped `focus_power` from the stack worker - the same
  bug class as the substack cache key fixed in the previous round. The test compares
  fingerprints rather than listing fields, so a future setting cannot be missed.
- `--focus-power` on the CLI. Deliberately not in the UI: no evidence yet about what a
  photographer should do with the control, and it is a fusion-quality default rather than
  a per-image choice.

## 7. Reproducing

```bash
.venv/bin/python docs/investigations/confidence_gate_gt.py
.venv/bin/python docs/investigations/gate_real_stacks.py
.venv/bin/python docs/investigations/original_stacks_p4.py
.venv/bin/python docs/investigations/benchmark_dataset.py --algorithms laplacian_p4_c1 laplacian_p8_c1
.venv/bin/python docs/investigations/benchmark_metrics.py --width 2592
```

The venv is pinned to cv2 4.11.0.86 / numpy 2.2.6 for comparability with these numbers.

## 8. Open

- **The default stays 1.0.** Nothing here decides that a photographer wants 4; it decides
  that 4 is now safe to choose. Changing the default needs a call on output that users
  have already accepted.
- **Exposure normalisation across frames** would attack the root cause rather than
  confining it to one band, and would likely help the wavelet path too, which shows the
  same 22.20 lab_drift.
- **Combining both gates** (level for brightness, confidence for noise) is untested.
- The stack-2 `left_edge` probe separates nothing for any method (54.14 -> 54.23) and is
  not evidence either way; it was never a uniform region.
