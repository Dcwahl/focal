# First correctness fixes — September 7, 2026

Branch: `fix/stacking-correctness`.

This pass repairs reproducible image-preservation and registration failures from
the baseline review. Wavelet coefficient selection is unchanged; restoring its
consistency filtering remains a separate task requiring quality and memory tests.

## Changes

- Laplacian weights fall back to equal contributions when there is no focus
  evidence, preserving flat backgrounds and the coarsest pyramid's brightness.
- PCA weights fall back to standard BGR luminance for degenerate or mixed-sign
  principal directions, avoiding division by zero and clipping caused by negative
  weights. Valid nonnegative PCA weights retain their existing behavior.
- Coarse alignment uses phase correlation to initialize translation; fine ECC
  refinement receives that estimate, with correct scaling between resolutions.
  ECC errors retain the initial estimate rather than a partially mutated failed
  solve, and emit a log warning. Invalid nonfinite or nonpositive-determinant
  results also fall back.
- Brush painting resamples the entire affected source rectangle through the
  affine transform, preserving rotation, scale, shear, and subpixel translations.
  Uncovered pixels are left unchanged. Flash compare uses the same geometry.
- Main-stack workers own their stacker configuration and input list. Completion
  reads the transforms belonging to that worker, even if UI settings have changed.

## Validation

**79 tests pass**, including 18 new regression cases. Coverage includes constants,
smooth backgrounds around detail, PCA degeneracy, known translations, failed ECC
mutation, transform scaling, affine brush sampling, repeated undo/redo, uncovered
pixels, and transform ownership after a UI configuration change.

The final suite and before/after probes used the original lockfile's numerical
versions: NumPy 2.2.6, OpenCV 4.11.0.86, Numba 0.63.1. PySide6 remained 6.11.2 and
Python 3.12.3; this is not a full historical environment reconstruction. An earlier
78-test pass also succeeded with the freshly resolved numerical dependencies.

Baseline source was independently extracted from HEAD into `/tmp/focal-baseline-src`
and rerun with the same numerical environment as the fixed source. Recorded results
are in `correctness_metrics_2026-09-07.json`.

| Diagnostic | Before MAE (0–255) | After MAE |
|---|---:|---:|
| Two identical gray images | 128 | 0 |
| Rotated/scaled brush, center patch | 112.003 | 0.0077 |
| Alignment, translation (20, 10) | 16.676 | 0 |
| Alignment, translation (60, 30) | 16.807 | 0.0030 |

The tiny remaining brush difference is an isolated interpolation-rounding difference
between cropped and full-frame warps in OpenCV 4.11. The opponent-color grayscale
diagnostic now produces finite weights and retains contrast without warnings.

These fixes do not imply that every quality metric improves: the guarded PCA fallback
changes wavelet output on the synthetic split-focus texture, whose MAE goes from
0.477 to 0.623. Both are much better than the best source (10.663). The fallback
trades that particular score for defined luminance behavior on problematic colors.

Three Lytro pairs (01, 05, 10) were rerun with alignment disabled for both algorithms.
Visual inspection at contact-sheet scale did not reveal a gross regression. Native
outputs and the contact sheet are in `/tmp/focal-review-fixed`; source images remain
in `/tmp/focal-review/lytro`. Neither full interactive GUI behavior nor long macro
stacks have been validated yet.

To rerun in the temporary environment:

```bash
QT_QPA_PLATFORM=offscreen NUMBA_CACHE_DIR=/tmp/focal-numba-locked /tmp/focal-review-venv/bin/python -m pytest -q
NUMBA_CACHE_DIR=/tmp/focal-numba-probe-locked /tmp/focal-review-venv/bin/python docs/investigations/baseline_probe.py /tmp/focal-review-fixed
```

The next quality pass should restore and evaluate wavelet selection consistency
without restoring all per-frame decompositions in memory. Stacking validity masks,
large-stack memory, 16-bit/color-managed I/O, and the broader substack lifecycle
also remain open findings from the baseline review.
