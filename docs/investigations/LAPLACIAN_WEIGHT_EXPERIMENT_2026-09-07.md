# Laplacian focus-weight experiment — September 7, 2026

## Question and result

Does making focus selection more decisive recover source detail that the current
Laplacian blend loses? **Partly, with a measurable noise tradeoff.** Squaring the
focus measures is a promising conservative candidate. Application defaults and
production algorithm code were not changed in this experiment.

The baseline uses each frame's local focus measure directly as its blending weight,
normalized over frames. Candidates raise those measures to power 2 or 4, then
normalize. A 2:1 focus-score ratio therefore becomes 4:1 or 16:1. A fourth variant
smooths the power-4 masks with a Gaussian sigma of one pixel at each pyramid level.
All variants retain equal weights when every measure is zero.

## Comparisons

- Six additional full 24MP runs: powers 2, 4, and smoothed 4 on both real stacks,
  using exactly the previously saved shared alignment. Baseline outputs were reused.
- All four variants on Lytro pairs 01, 05, 10, without alignment.
- Controlled three-frame clean/noisy focus sequences, a 12-frame noisy sequence,
  a one-pixel residual-misalignment case, and constant/duplicate-frame checks.
  Synthetic inputs use RNG seed 45 and simplified bandwise Gaussian blur.

## Findings

The figurine's native mouth/face crop recovers visibly more fine texture and edge
definition with stronger weighting. Power 2 obtains much of the benefit without
the additional grain seen at power 4. The cuff/arm crop also improves some texture,
but the remaining softness is not eliminated. Sharper weighting does not fix
misregistration or guarantee that every local focus choice is correct.

The changing background of the clip stack changes color/brightness as the frame
weights change. Power 4 makes these variations more pronounced. These candidates
are smoother than the earlier wavelet output in the inspected background crop,
but none resolves inconsistent scene content.

Lytro contact-sheet inspection showed no gross failure in the tested pairs. That
is a limited visual check, not proof of general boundary or halo robustness.

Controlled image MAE, on a 0–255 scale (lower is better):

| Case | Baseline | Power 2 | Power 4 | Power 4 + smoothing |
|---|---:|---:|---:|---:|
| Clean, 3 frames | 0.677 | 0.599 | 0.578 | 0.599 |
| Noise, 3 frames | 3.439 | 2.560 | 2.545 | 2.542 |
| Noise, 12 frames | 2.955 | 1.619 | 1.581 | 1.561 |
| Residual 1px shift | 8.475 | 8.472 | 8.465 | 8.471 |

Those whole-image scores conceal a regression in the flat patch. Its MAE in the
12-frame noisy case increases from **1.124 → 1.193 → 1.440**, for baseline, power 2,
and power 4. Mask smoothing reduces the power-4 value to 1.334 but does not remove
the tradeoff. More decisive selection can favor noise where no meaningful sharpness
difference exists. Constant and duplicate inputs remain preserved within one level.

The saved native crop metrics compare high-pass detail to a preselected source.
They are diagnostics only: the selected source is not ground truth, and extra noise
can resemble detail. Visual inspection and the controlled flat-region check are
needed alongside those numbers.

Native candidate fusion times ranged from 7.6–14.7 seconds, with roughly 9.1–11.3 GiB
peak RSS. These are single observations during other activity, including a dataset
download; they do not establish speed differences. This experiment does not improve
the current memory footprint.

## Artifacts

In `test_outputs/real_stacks/focus_stack_test-2/weight_experiment/`, start with
`crop_face.png` and `crop_left_edge.png`. Columns are baseline, power 2, power 4,
and power 4 with smoothing; each crop preserves native pixels.

Both stack directories contain `overview.jpg`, all previously fixed diagnostic crops,
and full outputs `p2_s0.png`, `p4_s0.png`, and `p4_s1.png`. The Lytro comparison is
`test_outputs/weight_experiment/lytro_comparison.jpg`. Numerical observations are
preserved in `laplacian_weight_metrics_2026-09-07.json`.

Reproduce a full candidate or the synthetic controls with the project installed:

```bash
python docs/investigations/laplacian_weight_experiment.py test_outputs/real_stacks/focus_stack_test-2/native_aligned test_outputs/weight-rerun --power 2
python docs/investigations/weight_experiment_controls.py test_outputs/weight-controls-rerun
```

## Decision

Keep power 2 as a candidate for the independent longer-stack evaluation. Before
promoting it to a default, check whether its flat-region penalty remains acceptable
on those sequences and whether the sharper edges develop halos. A confidence-aware
fallback for weak focus evidence is a more useful next experiment than increasing
the exponent further. This is a targeted modification of the existing Laplacian
pipeline, not a new stacking algorithm.
