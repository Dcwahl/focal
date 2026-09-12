# Focal baseline review — September 7, 2026

Focal is a functioning prototype with worthwhile application structure and several
serious correctness gaps. Both fusion algorithms can combine focused regions, but
the current implementation does not justify the README's general claim that
wavelets provide better quality. Repairing and measuring the existing pipeline is
a sensible next step before choosing a replacement algorithm.

Reviewed commit: `42d87ed` (January 22, 2026). No application code was changed.
Added only this assessment, numerical results, and a reproducible diagnostic script.

## Evidence and limits

- Existing suite: **61 passed** (`QT_QPA_PLATFORM=offscreen python -m pytest -q`).
- Additional diagnostics: `baseline_probe.py`, fixed seed 20260907; results in
  `baseline_metrics_2026-09-07.json`.
- Three public Lytro pairs: 01, 05, 10, both algorithms with alignment disabled.
  Inputs and full-resolution outputs are in `/tmp/focal-review`; the comparison
  sheet is `/tmp/focal-review/lytro_contact_sheet.png`.
- Fresh installation from the declared dependency ranges, **not the historical
  lockfile**: NumPy 2.5.3, OpenCV 5.0.0.93, Numba 0.67.0, PySide6 6.11.2,
  Python 3.12.3. Numerical results describe this environment. Recheck the locked
  versions when implementing regression fixes.
- GUI code was reviewed and its brush routine exercised directly without opening
  a window. This was not a full interactive usability, packaging, performance, or
  commercial-product comparison.
- Synthetic blur and random textures isolate faults; they do not simulate realistic
  macro optics. Three small Lytro pairs cannot establish broad photographic quality.

## Findings, in repair order

### 1. Laplacian fusion destroys flat image regions

`src/focal/core/stacker.py:361`: weights are focus / (sum of focus + epsilon).
When every image has zero focus energy, all weights are zero rather than summing
to one. The same rule is applied to the coarsest Gaussian image, which carries the
background brightness, so this can destroy the baseline illumination too.

**Reproduced:** stacking two identical 256×256 images filled with 128 produces
an entirely black image. Mean absolute error (MAE) is 128/255. The wavelet path
preserves the same input exactly. Identical textured inputs survive both paths.

Repair with a defined low-confidence fallback and deliberate coarse-level fusion.
Add tests for constants, smooth backgrounds, gradients, and duplicate frames.

### 2. Wavelet consistency filtering is disconnected

`src/focal/core/stacker.py:270`: the production pipeline calls
`merge_wavelet_incremental` with a zero threshold. It never reads `self.consistency`
after construction and never invokes the consistency routines in `merge_wavelets`.

**Reproduced:** consistency=0 and consistency=2 produce identical outputs on the
split-focus diagnostic. The existing tests exercise the alternate merge function,
not the full pipeline with this setting.

The incremental merge does save coefficient storage, but it bypasses the advertised
selection regularization. Reintroducing selection consistency needs a deliberate
memory-aware design; simply calling the old routine restores all decompositions
in memory and does not establish that its filtering is good enough.

The earlier `wavelet_bottlebrush_artifacts.md` already reports incoherent coefficient
selection and unsuccessful attempts to fix dense-detail artifacts. Those historical
observations are useful leads, not newly reproduced measurements here.

### 3. Alignment loses its coarse estimate and conceals failures

`src/focal/core/align.py:105`: rough alignment is computed, then overwritten by a
fine solve initialized independently to identity. This is not coarse-to-fine
refinement. At line 71, OpenCV errors are swallowed. Because ECC can modify the
matrix before raising, the return may also contain a partial failed estimate,
despite the comment promising identity on failure.

**Reproduced on a 512×512 synthetic textured scene:**

| Source translation (x, y) | Interior MAE before | Interior MAE after |
|---|---:|---:|
| (3, 1.5) pixels | 14.20 | 0.51 |
| (20, 10) pixels | 16.63 | 16.68 |
| (60, 30) pixels | 16.67 | 16.81 |

These are examples of success and failure, not universal displacement limits.
The small shift also verifies that the current transform direction is correct.

Carry the coarse estimate through coordinate scales, record convergence and
alignment quality, and use valid-pixel masks. Reflected borders currently participate
in fusion as if they were genuine observations. Global affine warps also cannot
resolve arbitrary subject motion or depth-dependent parallax.

### 4. Retouching does not apply the full affine warp

`src/focal/ui/main_window.py:526`: only the brush center is inverse-transformed and
rounded. The surrounding source square is copied without rotation, scaling,
shearing, or subpixel resampling. Thus a correctly aligned stack can be damaged
by supposedly alignment-aware retouching.

**Reproduced:** start with the correctly warped checkerboard, then paint from that
same source using its known 10-degree rotation / 1.05 scale transform. A central
36×36 patch acquires MAE **112.02/255** instead of staying unchanged. This deliberately
demanding pattern makes the geometric error visible, not representative in magnitude.

Warp the source region into result coordinates before feathering. Flash compare
also displays the unwarped source (`main_window.py:447`), so its geometry differs
from the stacked result.

### 5. PCA grayscale conversion can divide by zero

`src/focal/core/grayscale.py:43`: a principal component is normalized by its channel
sum. Opposing channel variation can produce a direction whose sum is zero or very
small. A valid PCA direction does not guarantee usable nonnegative grayscale weights.

**Reproduced:** B=x, G=255−x, R=0 yields nonfinite weights, runtime warnings,
and an all-zero grayscale image. This loses the focus information for that input.
Near-zero denominators and negative weights also warrant testing for clipping.

Use a guarded fallback or a better-defined focus luminance conversion; preserve
floating-point precision until the image-writing stage where possible.

### 6. Full-resolution memory and photographic I/O remain limited

Code inspection, not a measured large-stack benchmark:

- Both algorithms load the whole stack. Laplacian additionally retains float color,
  float grayscale, and per-frame pyramids.
- Wavelet merging is incremental, but color reassignment builds contiguous stacks
  and then another per-pixel layout. Existing color/grayscale arrays plus those two
  representations alone require approximately **12 × frames × pixels bytes** at
  that stage. A 50-frame, 24MP stack implies roughly **14.4 GB decimal**, before
  other arrays, padding, UI images, and temporary allocations.
- The GUI's 4GB source cache does not bound stacking allocations or undo history.
- Default `cv2.imread` and uint8 outputs discard 16-bit TIFF precision. There is no
  explicit ICC-managed image pipeline or raw-development pipeline.

Streaming/tiled processing, incremental color selection, and bounded undo storage
would matter before claiming high-resolution production readiness.

### 7. GUI state and substack lifetime need a separate regression pass

Additional code-inspection risks:

- Algorithm/alignment controls can replace `self.stacker` while a worker runs;
  completion reads transforms from `self.stacker`, not necessarily the worker's
  stacker (`main_window.py:399`). Retouching can then use missing/wrong transforms.
- Substacks use the same mutable stacker, execute synchronously, and pump UI events.
  This permits reentrant actions and potentially competing work.
- Substack cache keys contain only frame indices, not algorithm/alignment settings;
  cache eviction and recomputation after settings changes may change the paint source.
- The existing substack tests cover the data object, not the stacking/painting lifecycle.

## What works

The core/UI separation makes diagnosis and headless evaluation straightforward.
The application includes a useful combination of source selection, substacks,
feathered retouching, undo/redo, and background main-stack execution. The wavelet
transform roundtrip test passes; there is no evidence here that the transform itself
needs replacing.

On the deliberately simple complementary-focus target, the best individual source
has MAE **10.66**, versus **0.668** for Laplacian and **0.477** for wavelet. Both
therefore recover useful focused content under favorable conditions.

Visual inspection of Lytro 01 (golfer), 05 (fence), and 10 (trees) shows that both
combine foreground/background detail. Fine occlusion boundaries remain worth
inspecting at native resolution, especially the fence. These unscored examples do
not establish an overall winner or parity with commercial software.

## Dataset and comparison plan

1. **Lytro:** quick small-image fusion checks. The author's repository provides 20
   color pairs and four three-frame series. Three pairs were downloaded and evaluated
   for this review. Retain its attribution file and cite the authors when publishing.
   [Author's repository](https://github.com/mnnejati/LytroDataset).
2. **MFFW:** a useful harder test for defocus spread and occlusion boundaries.
   Located, not downloaded/run here.
   [Dataset paper](https://arxiv.org/abs/2002.04780).
3. **LSFD:** 94 real high-resolution bursts, 30 frames each, with Helicon-derived
   pseudo ground truth. More representative of long stacks; the reference images
   are commercial outputs, not error-free physical ground truth. Located, not run.
   [Author's code and download link](https://github.com/araujoalexandre/FocusStackingDataset),
   [paper](https://arxiv.org/html/2311.17846v1).
4. **Historical macro examples:** Helicon still lists Bottlebrush, Godetia, Crystal,
   Ladybird, and other sources used in the old notes. Its stated usage terms are
   specifically for testing Helicon/demonstration/marketing, with contact requested
   for sharing results; don't assume a general redistributable benchmark license.
   No Helicon files were downloaded in this review.
   [Demo stacks and terms](https://www.heliconsoft.com/helicon-focus-gallery/).

Compare fusion methods using **the same aligned inputs**, then evaluate alignment
separately. The old `PHASE6_TESTING.md` often compares unaligned Laplacian with
aligned wavelet output, confounding those two questions.

Save original inputs, transforms, validity masks, fusion outputs, and fixed native
resolution crops. Score sharp-detail recovery, halo/ghosting artifacts, smooth-region
stability, color drift, wall time, and peak memory. Use reference-image metrics where
the reference is meaningful; a global sharpness score can reward noise and ringing.

Recommended next implementation sequence: fix image-preservation and PCA failures;
repair alignment and brush geometry; restore/test consistent wavelet selection;
then compare against stronger fusion baselines on a fixed mix of macro subjects.
Keep the useful GUI and demand measurable quality improvements from algorithm changes.
