# Phase 3 - Complex Daubechies Wavelet Algorithm

A second stacking algorithm with better edge handling and alignment support.

## Why This Matters

Laplacian pyramid stacking is fast but has limitations:
- No image alignment (requires tripod shots)
- Standard grayscale conversion loses information
- Can produce halos at high-contrast edges

Complex Daubechies wavelets address these with a more sophisticated pipeline that handles handheld shots and produces cleaner results on fine detail.

## Features Implemented

### 1. Complex Daubechies Wavelet Transform

6-tap filters with real and imaginary components:
- Better shift invariance than real wavelets
- Phase information aids edge detection
- Magnitude provides reliable focus measure

Files: `core/complex_wavelet.py`

### 2. Image Alignment (ECC)

Two-pass alignment using OpenCV's Enhanced Correlation Coefficient:
- Rough pass at 256px resolution
- Fine pass at full resolution
- Affine transform handles translation, rotation, scale
- Compensates for focus breathing and camera shake

Files: `core/align.py`

### 3. PCA Grayscale Conversion

Optimal channel weights computed via Principal Component Analysis:
- Samples reference image to find first principal component
- Weights maximize variance preservation
- Better than fixed BGR2GRAY coefficients for varied subjects

Files: `core/grayscale.py`

### 4. Color Reassignment

Since wavelets operate on grayscale, colors must be recovered:
- Build per-pixel map of grayscale values across the stack
- For each output pixel, find source with closest grayscale match
- Preserves original colors without blending artifacts

Files: `core/reassign.py`

### 5. Consistency Filtering (Denoising)

Two levels of artifact reduction:
- **Level 1 (Subband voting)**: 2-out-of-3 voting across H/V/D subbands
- **Level 2 (Neighbor filtering)**: Remove outlier pixels where all 4 neighbors disagree

Configurable via `consistency` parameter (0=none, 1=subband, 2=full).

### 6. UI Algorithm Selector

Dropdown in top bar to choose between algorithms:
- "Laplacian" - fast, good for tripod shots
- "Complex Wavelet" - better quality, handles handheld

Files: `ui/main_window.py`

## Pipeline Comparison

| Step | Laplacian | Complex Wavelet |
|------|-----------|-----------------|
| Load | Direct | Pad to 2^n size |
| Grayscale | BGR2GRAY | PCA-weighted |
| Align | None | ECC affine |
| Transform | Gaussian/Laplacian pyramid | Complex Daubechies |
| Merge | Focus-weighted blend | Max magnitude selection |
| Denoise | None | Subband + neighbor voting |
| Reconstruct | Pyramid collapse | Inverse wavelet |
| Color | Works on color directly | Reassign from grayscale |

## Performance Optimizations

### Numba JIT for Wavelet Transform

The wavelet 1D convolution loops were the original bottleneck. Added `@njit(cache=True)` decorators:
- `_decompose_1d_jit`
- `_compose_1d_jit`

Results on 1024x768 images (3 image stack):
| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Decompose | 24s | 0.05s | ~450x |
| Compose | 8s | 0.02s | ~480x |

### Color Reassignment (Not Optimized)

Numba-accelerated versions exist (`build_color_map_fast`, `reassign_colors_fast`) but produce visible halos at edges due to missing per-pixel deduplication. The slower Python version is used by default for quality.

## Remaining Bottlenecks

1. **Alignment** - OpenCV ECC is CPU-bound; could explore GPU acceleration
2. **Color reassignment** - Pure Python loops; quality/speed tradeoff
3. **Memory** - All images loaded at once; no batch processing

## Dependencies Added

- `numba>=0.63.1` - JIT compilation for hot loops

## Technical Notes

### Module Organization
```
core/
├── stacker.py          # Algorithm selection, orchestration
├── complex_wavelet.py  # Transform, merge, denoise
├── align.py            # ECC alignment
├── grayscale.py        # PCA conversion
└── reassign.py         # Color recovery
```

### Usage
```python
from focal.core.stacker import FocusStacker, StackAlgorithm

# Complex Wavelet (quality)
stacker = FocusStacker(
    algorithm=StackAlgorithm.COMPLEX_WAVELET,
    consistency=2  # 0=none, 1=subband, 2=full denoising
)
result = stacker.stack(image_paths)
```

## Definition of Done

- [x] Complex Daubechies wavelet transform implemented
- [x] ECC-based image alignment working
- [x] PCA grayscale conversion
- [x] Color reassignment from merged grayscale
- [x] Consistency filtering (2 levels)
- [x] UI dropdown for algorithm selection
- [x] Numba JIT for wavelet performance
- [x] All tests passing (26 tests)

## What's Not Included

Features from the C++ reference that were skipped:
- Contrast correction before alignment
- White balance matching before alignment
- OpenCL GPU acceleration
- Batch processing for memory management
- Depth map visualization/export
