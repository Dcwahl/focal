# Research dataset benchmark — first results

Date: September 12, 2026. Branch `fix/stacking-correctness`, base commit `42d87ed`.
No production code changed. This is measurement only.

## What the dataset is

`test_stacks/dataset/focus_stack_dataset/focus_stack_dataset/dataset`, 107 GiB,
9,566 files, extracted from a 96 GiB archive. 94 scenes split train (84) / test (10),
plus 10 iPhone bursts. **The train/test split is irrelevant to us** — we train nothing,
and every scene carries the same assets:

- `jpg/` — 30 source frames, 5184×3888 (20 MP)
- `raw/` — the same 30 frames as Panasonic `.rw2` (needs rawpy/LibRaw; unused so far)
- `aligned/` — the 30 frames pre-registered by the dataset authors
- `aligned copie/` — present in `chess_piece3` **only** (not all scenes), a verified
  byte-identical duplicate of `aligned/`; a glob hazard, so scripts here name `aligned/`
  explicitly
- six `helicon_focus*.jpg` renders — Helicon Focus output, the tool CLAUDE.md names
  as the competitor
- `transforms.pth` — the authors' alignment transforms (PyTorch, unused so far)

Two properties make it more useful than the existing `test_stacks/focus_stack_test-{1,2}`:

1. **`aligned/` decouples alignment from fusion.** Feeding both algorithms the same
   pre-registered frames with `skip_alignment=True` removes the confound that made the
   old `docs/PHASE6_TESTING.md` "winner" claims untrustworthy.
2. **30-frame stacks**, against the 8–10 frames tested so far. Deeper stacks are where
   blending defects accumulate, and the focus steps are finer.

Helicon renders were verified to be **pixel-registered to `aligned/`** (phase
correlation shift < 0.1 px on needle2 and leaf2), so per-pixel comparison is valid.

## Frozen regression subset

Six scenes, chosen to span fine-detail macro and depth-layered scenes with smooth
backgrounds: `needle2`, `chess_piece3`, `flower3` (olympus macro) and `leaf2`,
`chairs3`, `eastern_red_cedar_tree2` (lumix). Defined in `benchmark_dataset.py`.

## Method

There is **no all-in-focus ground truth** in this dataset, so nothing is scored against
a "correct" image. Instead each scene gets a per-pixel **focus envelope**: the maximum
local sharpness any of the 30 aligned frames achieves at that pixel. That is the best
detail the inputs actually contain, and it bounds what any fusion could legitimately
recover.

- `recovery` — median(fused sharpness / envelope) over textured pixels (top 40% of
  envelope). 1.0 means the sharpest available detail was kept. Below 1.0 is mush.
  **Above 1.0 is not a win**: it means contrast was invented that no source had.
- `halo_fraction` — share of textured pixels exceeding the envelope by >25%.
- `flat_excess` — fused high-frequency energy where *every* source is smooth (bottom
  20% of envelope), over the mean source energy there. 1.0 is faithful; above 1.0 is
  blending speckle, below 1.0 is smoothing.

**Helicon is measured on the same metrics purely as a reference point and never enters
another method's score.** Scoring against Helicon would make "reproduce Helicon,
artifacts included" the objective.

Run at half resolution (2592×1944), all 30 frames, both algorithms, 12/12 succeeded.
Half res was chosen over frame subsampling deliberately: dropping to every 3rd frame
would have widened the focus steps and discarded the thing that makes this dataset new.

## Results (mean over the six scenes)

| method | recovery | halo_fraction | flat_excess | peak RSS | fusion time |
|---|---|---|---|---|---|
| laplacian | **0.346** | 0.000 | **0.315** | 7.06 GiB | 5.2 s |
| complex_wavelet | **1.026** | **0.234** | **1.873** | 2.79 GiB | 13.3 s |
| helicon_focus | 0.884 | 0.004 | 1.223 | — | — |

Per-scene numbers in `test_outputs/dataset_benchmark/metrics_w2592.json`.

### Laplacian is averaging, not selecting

Recovery 0.22–0.58 (mean 0.35) means it discards roughly two thirds of the available
detail. `flat_excess` of 0.31 is the tell: in flat regions it retains about a third of
the source high-frequency energy, close to what averaging ~30 noisy frames would give.
It behaves like a weighted average almost everywhere rather than a focus selector.
Visually confirmed — `needle2/crop_3_texture.png` shows the needle's granular texture
smeared away entirely while both wavelet and Helicon resolve it.

This independently corroborates the existing `laplacian_weight_experiment.py` finding
that baseline weighting is too soft, and strengthens the case for the power-2 candidate.
Note the halo count is 0.000 for every scene — it never overshoots, it only blurs.

### Complex wavelet trades mush for invented contrast

Recovery 1.026 looks like a win against Helicon's 0.884 and is not one. On the three
macro scenes it exceeds 1.0 (needle2 1.289) with halo fractions of 0.25–0.55, meaning
a quarter to a half of textured pixels carry more contrast than any source had. Paired
with `flat_excess` 1.87 — nearly double the noise Helicon leaves in smooth regions —
this is the speckling/blotchiness already recorded in HANDOFF.md, now quantified.

### Helicon's profile is the actual target

0.884 recovery, 0.004 halo, 1.223 flat_excess: it keeps ~88% of available detail,
essentially never invents contrast, and adds modest noise. Neither Focal algorithm is
close to that balance — Laplacian misses on detail, wavelet misses on restraint.

### Tonal behaviour

Global mean-luma drift against the sources is small for all three: laplacian −0.71,
wavelet +1.92, Helicon −1.18. The visible darkening in `chairs3/crop_0_disagree.png` is
therefore **local** drift, not a global shift, consistent with the blotchy colour
boundaries noted previously. Recorded in `tonal_w2592.json`.

## Memory

At half res, Laplacian peaks at 7.06 GiB against wavelet's 2.79 GiB. Scaling by pixel
count, **native 20 MP × 30 frames projects to ~28 GiB for Laplacian against 25 GiB
available on this machine** — it will OOM or swap. Wavelet projects to ~11 GiB and
fits. Native-resolution 30-frame runs are wavelet-only until Laplacian memory is
addressed; this is a sharper version of the memory problem already in HANDOFF.md.

## Reproducing

```bash
# venv now lives in the repo (.venv is gitignored) and survives /tmp clears
cd /home/diegowahl/focal
export NUMBA_CACHE_DIR=/home/diegowahl/.cache/focal-numba
.venv/bin/python docs/investigations/benchmark_dataset.py --width 2592
.venv/bin/python docs/investigations/benchmark_metrics.py --width 2592
.venv/bin/python docs/investigations/benchmark_crops.py  --width 2592
```

Artifacts land in `test_outputs/dataset_benchmark/<scene>/`: `laplacian.png`,
`complex_wavelet.png`, `helicon_reference.png`, per-run timing/RSS JSON, `overview.jpg`,
and four 400×400 crops (three at highest algorithm disagreement, one most textured).

## Suggested next steps

1. Re-run the power-2 Laplacian weighting candidate through this harness. Recovery
   should rise from 0.35; watch `flat_excess` for the flat-region regression the earlier
   controlled test found. This subset is a much better test of it than two user stacks.
2. Treat `halo_fraction` and `flat_excess` as wavelet's acceptance criteria — its
   recovery is already adequate, its restraint is not.
3. Laplacian native-resolution memory, before any 30-frame native work is possible.
4. Unused leads: `raw/` via rawpy for a 16-bit path, and `transforms.pth` as an
   independent check on `align.py`.

## Housekeeping not done (user's call)

- `test_stacks/dataset/focus_stack_dataset.zip` — 96 GiB, fully extracted, deletable.
- `aligned copie/` in `chess_piece3` only — 109 MB, verified byte-identical, deletable.
- 195 GB free at time of writing, so neither is urgent.

---

# Verification pass, and corrections to the claims above

Added September 12, 2026, after the power-sweep. Several claims in this document were
tested against independent methods. Two did not survive; they are struck through below
rather than quietly edited, and the scripts that falsified them are listed so the same
checks can be rerun.

## Independent-metric check (`recovery` is not circular, but it is harsh)

`sharpness()` above is Laplacian-based, and `FocusStacker._compute_focus_measure` is
*also* Laplacian variance — so the metric risked grading the algorithm on exactly what
it optimises. Re-scored under two operators that share no basis with it: Sobel gradient
magnitude, and local intensity standard deviation.

| method | Laplacian metric | Sobel | local-std |
|---|---|---|---|
| laplacian (p1) | 0.346 | 0.537 | 0.561 |
| laplacian_p2 | 0.592 | 0.748 | 0.758 |
| laplacian_p4 | 0.802 | 0.866 | 0.871 |
| complex_wavelet | 1.026 | 0.943 | 0.937 |
| helicon_focus | 0.884 | 0.914 | 0.916 |

**The ordering is stable under all three.** p1 < p2 < p4 < helicon holds regardless of
operator, so the central finding is not a metric artifact. But absolute values shift a
lot, and the Laplacian metric is the harshest on the shipped algorithm. Treat `recovery`
as ordinal, not as a percentage of anything real.

## ~~"Complex wavelet invents contrast on 23% of textured pixels"~~ — OVERSTATED

Halo fraction is metric-dependent: 0.234 under the Laplacian metric, **0.074 under
Sobel**. Wavelet's recovery also stays below 1.0 under both independent operators, so
the specific claim that it exceeds what any source contained is not robust. What *does*
survive: wavelet's flat-region noise is ~1.7-1.9x the source level against Helicon's
1.2x under every metric tried, and its halo is still an order of magnitude above
Helicon's 0.002. The noise/speckle problem is real; the "invented detail" framing was
too strong.

## ~~"The failure gets worse the deeper the stack"~~ — NOT ESTABLISHED

`depth_gt_control.py` renders a synthetic depth-ramp scene, sharp original known, frame
*i* focused at depth i/(N-1), and scores fusions by MAE against ground truth — no focus
envelope involved. MAE, lower is better:

| noise | N | best single frame | p1 | p2 | p4 | p8 |
|---|---|---|---|---|---|---|
| 0 | 5 | 5.14 | 1.90 | 1.36 | 1.32 | 1.32 |
| 0 | 10 | 5.15 | 1.70 | 0.94 | **0.78** | 0.93 |
| 0 | 20 | 5.14 | 1.67 | 0.93 | **0.77** | 0.90 |
| 0 | 30 | 5.14 | 1.67 | 0.93 | **0.77** | 0.90 |
| 3 | 10 | 6.05 | 4.22 | 3.40 | 2.76 | 2.68 |
| 3 | 30 | 6.04 | 4.08 | 3.19 | 2.35 | **2.19** |

p1's error is **flat** from N=10 to N=30 in the clean case (1.70 → 1.67). The optimal
exponent does not migrate upward with depth; p4 is optimal at N=10, 20 and 30 alike.
What is true, and weaker: the *benefit* of a high exponent grows with depth and with
noise, and under noise p8 edges past p4 at large N.

The real-data depth sweep that suggested otherwise (needle2 p1 recovery 0.386 at 8
frames vs 0.219 at 30) is confounded: adding frames raises the focus envelope itself, so
the bar moves with N. `benchmark_metrics.py --step` matches the envelope to the frames
fused, which fixes cross-run comparison but not comparison *across different N*.

## ~~"Shipped Laplacian barely beats not stacking at all"~~ — WRONG

Under the envelope metric the sharpest single source frame scores 0.470 against p1's
0.346, which looked damning. Against true ground truth it is not: best single frame MAE
5.14, p1 MAE 1.67. p1 is far closer to correct overall — it gets every depth region
roughly right, where a single frame is wrong everywhere outside its own focal band. The
envelope comparison was apples-to-oranges and should not be quoted.

## What survived, and how it was checked

| claim | status | checked by |
|---|---|---|
| Proportional weighting dilutes the sharp frame; best frame gets 8.4% of weight on needle2 | **solid** | direct read of `_compute_weights` output, no metric involved |
| p4 > p2 > p1 for detail retention | **solid** | 3 independent metrics + ground-truth MAE |
| p4 adds no halo and no flat-region grain | **solid** | halo 0.000-0.002 under both metrics; flat_excess 0.607 < 1.0 |
| Wavelet leaves ~1.7x source noise in flat areas | **solid** | holds under every metric tried |
| p4 should be the new default | **unverified** | see below |

## Still unverified — do not ship p4 on this evidence

- Not retested on `test_stacks/focus_stack_test-{1,2}`, where the earlier experiment
  found higher powers made background colour/brightness drift worse. Those stacks have
  scene motion and lighting change; all six dataset scenes are static. Both findings can
  be true at once, and that regression is the one that would actually bite users.
- Not run at native resolution (Laplacian OOMs at 30x20MP; see memory section).
- p8 was not swept on real data, only in the GT control.
- No check on whether p4 interacts with the substack/retouch workflow.

---

# p4 on the user's own two stacks — the shipping decision

Run against `test_outputs/real_stacks/*/native_aligned` at native 6000x4000 (10 and 8
frames), reusing the September 7 p2/p4 renders, which were produced under the same
pinned versions. Script: `docs/investigations/original_stacks_p4.py`.

## Detail gains reproduce

| stack | metric | p1 | p2 | p4 | wavelet |
|---|---|---|---|---|---|
| test-1 | laplacian | 0.435 | 0.660 | **0.896** | 1.198 |
| test-1 | sobel | 0.416 | 0.603 | **0.790** | 0.949 |
| test-2 | laplacian | 0.441 | 0.574 | **0.784** | 1.210 |
| test-2 | sobel | 0.417 | 0.529 | **0.689** | 0.963 |

Same ordering as the research dataset, on different hardware, different scenes and at
native resolution. The detail finding is now confirmed on every set tried.
`test-2/p4_comparison/crop_face.png` shows p4 resolving mouth and surface texture that
p1 smears, without the colour fringing the wavelet puts on the blue/yellow boundary.

## The regression is real, and p4 makes it clearly worse

The earlier experiment's warning holds. In stack 1's defocused background — the region
where an arm enters the scene and lighting changes — large-scale colour drift rises
monotonically with the exponent:

| method | lab_drift | luma_std |
|---|---|---|
| p1 (shipped) | 9.96 | 10.34 |
| p2 | 16.99 | 17.39 |
| **p4** | **22.51** | **22.51** |
| complex_wavelet | 22.20 | 28.33 |

**p4 more than doubles background blotching.** `test-1/p4_comparison/crop_background.png`
shows it plainly: p1 renders a smooth gradient, p2 develops patches, p4 shows pronounced
mottling, and the wavelet disintegrates into speckle.

`flat_excess` corroborates: on these stacks p4 reaches 1.099 and 1.058, i.e. *above*
source energy, where on the static dataset scenes it stayed at 0.607.

Stack 2's `left_edge` region shows no separation (54.14 to 54.57), but that region is not
uniform for any method and is a poor uniformity probe; it should not be read as a pass.

## Mechanism

In sharply defocused regions every frame scores low and the ranking between them is
mostly noise plus whatever moved. Proportional weighting (p1) averages across all of
them, which smooths the disagreement away. A decisive exponent instead commits to
whichever frame happens to win each patch, so a moving arm and changing light get
selected in patches — blotches. The exponent that recovers detail in textured regions is
the same exponent that amplifies noise-driven selection in untextured ones.

## Conclusion: do not ship p4 as an unconditional default

It is the right behaviour where there is real focus evidence and the wrong behaviour
where there is not. The fix already proposed in HANDOFF.md — a confidence-aware fallback
in weak-focus regions — is now supported by direct evidence rather than intuition:
gate the exponent on focus confidence, so textured regions get decisive selection while
low-evidence regions fall back toward averaging. That is the next experiment.

A fixed p4 would be defensible only for tripod-stable, static-subject stacks, which
describes the research dataset and not stack 1.
