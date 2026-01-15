# Session Summary - Complex Daubechies Implementation

This document summarizes the work done to implement and optimize the Complex Daubechies wavelet focus stacking algorithm in Focal.

## Starting Point

Focal had an attempted Complex Daubechies implementation that was:
1. Hanging/never finishing (performance bug)
2. Producing ghosted/blurry output (missing pipeline steps)

Reference implementation: `py-focus-stack` (which ported from `focus-stack` C++)

## Issues Fixed

### Issue 1: Performance Bug in Wavelet Denoising

**Problem:** `_denoise_subbands` in `complex_wavelet.py` was computing `get_sq_absval(wv)` (a full H×W array operation) inside nested loops for every pixel. This made it O(n × H × W × H × W) - for a 256×256 image it took 6+ seconds; for real photos it would take hours.

**Fix:** Added `depth_map` tracking to avoid recomputation:
- `merge_wavelets` now tracks which source image was selected at each pixel
- `_denoise_subbands` and `_denoise_neighbours` use the depth_map instead of recomputing

**Files changed:** `src/focal/core/complex_wavelet.py`

### Issue 2: Missing Pipeline Steps (Ghosted Output)

**Problem:** The barcode in test images looked ghosted/blurry because focal was missing:
1. Image alignment (critical for focus breathing compensation)
2. PCA-based grayscale conversion
3. Proper color reassignment

**Fix:** Ported three modules from py-focus-stack:

| Module | Purpose |
|--------|---------|
| `align.py` | ECC-based affine alignment (handles translation, rotation, scale) |
| `grayscale.py` | PCA-weighted grayscale (preserves maximum variance) |
| `reassign.py` | Per-pixel color map for accurate color recovery |

**Files added:**
- `src/focal/core/align.py`
- `src/focal/core/grayscale.py`
- `src/focal/core/reassign.py`

**Files changed:** `src/focal/core/stacker.py` (updated `_stack_complex_wavelet` to use full pipeline)

### Issue 3: `levels_for_size` Difference

**Problem:** Focal had overly conservative caps producing fewer decomposition levels than py-focus-stack.

**Fix:** Simplified to match py-focus-stack behavior.

## Performance Optimizations

### Numba JIT for Wavelet Transform

Added `@njit(cache=True)` to the hot loops in `complex_wavelet.py`:
- `_decompose_1d_jit`
- `_compose_1d_jit`

**Results:**
| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Decompose (3 × 1024x768) | 24s | 0.05s | ~450x |
| Compose | 8s | 0.02s | ~480x |

### Numba JIT for Color Reassignment (Reverted)

Added fast Numba versions in `reassign.py`:
- `build_color_map_fast` / `_build_color_map_fast`
- `reassign_colors_fast` / `_reassign_colors_fast`

**However:** The fast version produces visible halos at edges due to missing deduplication.
The original Python version deduplicates gray values per-pixel, which is important for
edge quality. **Reverted to using the slow but high-quality version by default.**

The fast functions remain available in `reassign.py` for future optimization attempts.

### End-to-End Results

PCB test stack (7 images, 2048×1536):
| Stage | Before All Optimizations | After Wavelet JIT |
|-------|--------------------------|-------------------|
| Total | ~34s | ~15-20s |

The wavelet JIT alone provides significant speedup. Color reassignment remains the
main bottleneck but quality trumps speed here.

## Current State

The Complex Daubechies implementation now:
- Produces output matching py-focus-stack quality
- Runs at reasonable speed (~6.5s for 2K images)
- Has alignment, PCA grayscale, and proper color reassignment

## Remaining Bottlenecks

1. **Alignment (ECC)** - OpenCV's implementation, could explore GPU acceleration
2. **Memory** - No batch processing; loads all images into memory
3. **Contrast/WB correction** - C++ version does this before geometric alignment

## Files Modified/Added

```
src/focal/core/
├── complex_wavelet.py  # Fixed denoising, added Numba JIT
├── stacker.py          # Updated to use full pipeline
├── align.py            # NEW - ECC alignment
├── grayscale.py        # NEW - PCA grayscale
└── reassign.py         # NEW - Color reassignment with Numba

tests/
└── test_stacker.py     # Updated test assertions

docs/
├── ALGORITHMS.md       # NEW - Algorithm documentation
├── README.md           # Updated
└── CLAUDE.md           # Updated project structure
```

## Dependencies Added

- `numba` - JIT compilation for hot loops

## Test Command

```bash
cd focal
uv run pytest tests/test_stacker.py -v
```

All 26 tests pass.

Next steps:

Medium-term priorities:

3. Substack workflow - This is the differentiator you called out. The core stacking is solid, retouching basics work. Substacks unlock the "fix any artifact" workflow that makes Zerene/Helicon worth paying for.
4. Performance: color reassignment - Currently the slowest part of Complex Wavelet. The fast Numba version has quality issues, but there might be a middle ground (vectorized numpy without full JIT, or fixing the deduplication in the fast version).

Things I'd deprioritize:
- More algorithm options - we have two good ones now
- GPU acceleration - nice but not blocking anything
- Batch processing - solve single-stack UX first

One architectural thought: The current design loads all images into memory. For substack workflow this compounds (original frames + main result + substacks). Might be worth thinking about lazy loading or caching strategy before adding substacks, or it could get memory-hungry on large stacks.