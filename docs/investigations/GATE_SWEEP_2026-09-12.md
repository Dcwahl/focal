# Closing the grain gap: what worked, and three things that did not

**Date:** 2026-09-12
**Question:** `focus_power=4` recovers roughly double the detail of the shipped default but
visibly adds grain in smooth regions. Is that grain fundamental to a decisive exponent, or
just untuned?

**Answer:** not fundamental. A confidence-gated exponent at `focus_power=8` beats the
committed `focus_power=4` on detail *and* grain simultaneously on both real stacks. But
the two simpler explanations that the synthetic control endorsed both fail on real images,
and the reason is a limitation of that control worth recording.

---

## 1. The result that holds

`ConfidenceGatedStacker(focus_power=8, floor_pct=25, measure_blur=2)`, which is
production's level gate plus a per-pixel exponent pulled toward 1 where the peak focus
measure sits near an image-derived floor.

Stack 1 (10 frames, 6000x4000), sobel metric:

| method | recovery | halo | grain (flat_excess) | background drift |
|---|---|---|---|---|
| p1 (shipped default) | 0.416 | 0.000 | 0.518 | 9.96 |
| focus_power=4 (committed) | 0.788 | 0.001 | 0.952 | 10.02 |
| focus_power=8, no gate | 0.886 | 0.002 | 1.133 | 10.03 |
| **gated fp8, pct25, mb2** | **0.810** | 0.001 | **0.793** | 10.02 |

Stack 2 replicates the ordering and the margins: recovery 0.711 vs fp4's 0.686, grain
0.760 vs 0.877.

The gate is doing real work, not just the pooling inside it. At the same pooling sigma:

| stack 1 | recovery | grain |
|---|---|---|
| pooling only, fp8 mb2 | 0.853 | 1.073 |
| pooling + gate, fp8 pct25 mb2 | 0.810 | **0.793** |

It buys 0.28 of grain for 0.04 of detail. That is the trade this round was looking for.

## 2. Three things that did not work

**Winner-vs-runner-up margin as the confidence statistic.** Scoring confidence by how far
the sharpest frame beats the second sharpest suppresses the exponent almost everywhere:
GT covered-zone MAE 3.08 against p1's 3.29, i.e. barely better than not raising the
exponent at all. Consistent with the earlier finding that peak/mean does not separate
focused regions from defocused ones.

**Pooling the focus measure, without a gate.** On the 512px synthetic this looked like the
whole answer - it improved both zones monotonically and reached p1's flat-zone error. On
the 24MP stacks it collapses detail:

| stack 1 | recovery | grain | drift |
|---|---|---|---|
| fp8, no pooling | 0.886 | 1.133 | 10.03 |
| fp8, pool sigma 4 | 0.728 | 0.939 | 4.94 |
| fp8, pool sigma 6 | 0.641 | 0.869 | 2.55 |

Drift falling to 2.55, *below* p1's 9.96, is the tell: it is not selecting better, it is
smearing the selection across genuinely different depths.

**Widening `kernel_size` instead of adding a parameter.** The focus measure already has a
box-filter window, and on the synthetic widening it reproduced pooling exactly
(kernel_size=25: overall 2.62 / covered 1.28 / flat 1.06, against pooling's 2.64 / 1.28 /
1.06). That was briefly written up as "no new knob needed". It is wrong at native
resolution:

| stack 1 | recovery | grain |
|---|---|---|
| fp8, kernel_size=5 | 0.886 | 1.133 |
| fp8, kernel_size=17 | 0.629 | 0.868 |
| fp8, kernel_size=25 | 0.566 | 0.821 |
| fp8, kernel_size=35 | 0.527 | 0.790 |

Recovery falls below the committed fp4 (0.788) at every widened setting.

## 3. The methodological finding

**`confidence_gate_gt.py` is 512x512. Any parameter with a spatial scale that is tuned on
it will not transfer to 24MP images.** Sigma 6 spans 1.2% of the synthetic frame and 0.1%
of a 6000px frame, but the *content* scale differs far more than that ratio suggests, and
both pooling and kernel width landed on the wrong side of it.

What the control is still good for, and what it got right this round:
- identifying *which mechanism* is at play (it correctly separated the flat-zone noise
  mechanism from the coarse-brightness mechanism, which no real-stack metric did);
- ranking methods that have no spatial-scale parameter;
- providing true MAE where the real stacks only have proxy metrics.

Rule going forward: use the synthetic to choose a mechanism, then tune every spatial
parameter on native-resolution real stacks, and confirm on the dataset subset.

## 4. Status

Not shipped. `focus_power` and the level gate are in production and unchanged; the gate
lives in `laplacian_weight_experiment.py` as `ConfidenceGatedStacker`. Promoting it needs
a decision about API surface - it adds two parameters, both spatial, both tuned on two
sequences, which is exactly the overfitting risk this document is about.

## 5. Reproducing

```bash
.venv/bin/python docs/investigations/gate_sweep.py        # GT sweep of floor x pooling
.venv/bin/python docs/investigations/gate_dataset.py      # gated fusions, 6-scene subset
.venv/bin/python docs/investigations/benchmark_metrics.py --width 2592
```
