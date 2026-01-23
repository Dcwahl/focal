# Stacking Algorithms

Focal currently supports two focus stacking algorithms with different trade-offs.

## Algorithm Comparison

| Feature | Laplacian Pyramid | Complex Daubechies Wavelet |
|---------|-------------------|---------------------------|
| Speed | Fast | Slower (more processing steps) |
| Quality | Good | Better (especially fine detail) |
| Alignment | Optional (default: off) | Optional (default: on) |
| Grayscale | Simple BGR2GRAY | PCA-weighted (preserves variance) |
| Color output | Direct (works on color) | Reassigned from grayscale |
| Best for | Tripod shots, quick preview | Handheld, final output |

## Laplacian Pyramid

A fast, simple algorithm that works directly on color images.

### Pipeline
1. Load images
2. Build Gaussian pyramid for each image
3. Build Laplacian pyramid from Gaussian
4. Compute focus measure (Laplacian variance) at each level
5. Blend pyramids using focus-weighted masks
6. Reconstruct from blended Laplacian pyramid

### Characteristics
- **Optional alignment**: ECC alignment available but off by default for speed
- **Works on color**: No grayscale conversion, preserves original colors
- **Fast**: Minimal preprocessing, efficient pyramid operations

### When to Use
- Quick preview of stack quality
- Tripod-mounted shots (alignment off for speed)
- Handheld shots (enable alignment via checkbox or `--align`)
- When speed matters more than ultimate quality

---

## Complex Daubechies Wavelet

A sophisticated algorithm based on Forster et al. (2004) that produces higher quality results through a full preprocessing pipeline.

### Pipeline
1. **Load & Expand**: Load images, pad to size divisible by 2^levels
2. **PCA Grayscale**: Compute optimal channel weights, convert to grayscale
3. **Align**: ECC-based affine alignment to reference frame (middle of stack)
4. **Decompose**: Multi-level Complex Daubechies wavelet transform
5. **Merge**: Select coefficients with highest magnitude at each position
6. **Denoise**: Consistency filtering (subband voting + neighbor smoothing)
7. **Compose**: Inverse wavelet transform to grayscale
8. **Reassign Colors**: Map grayscale back to original colors via per-pixel lookup

### Key Components

#### Complex Daubechies Wavelets
6-tap filters with both real and imaginary components. The complex representation provides:
- Better shift invariance than real wavelets
- Phase information for edge detection
- Magnitude as reliable focus measure

Reference: "Image Processing with Complex Daubechies Wavelets" (J.M. Lina, 1997)

#### Image Alignment (`align.py`)
Uses OpenCV's Enhanced Correlation Coefficient (ECC) algorithm:
- Two-pass: rough alignment at 256px, then fine at full resolution
- Affine transform handles translation, rotation, scale
- Compensates for focus breathing and camera shake

#### PCA Grayscale (`grayscale.py`)
Instead of standard BGR weights (0.114, 0.587, 0.299), computes optimal weights via PCA:
- Samples pixels from reference image
- Finds first principal component direction
- Weights that maximize variance preserve the most information

#### Consistency Filtering
Two levels of denoising to reduce artifacts:
- **Level 1 (Subband voting)**: 2-out-of-3 voting across H/V/D subbands
- **Level 2 (Neighbor filtering)**: Remove outlier pixels where all 4 neighbors disagree

#### Color Reassignment (`reassign.py`)
Since wavelets operate on grayscale, colors must be recovered:
- Build per-pixel map of all grayscale values seen across stack
- For each output pixel, find source with closest grayscale match
- Preserves original colors without blending artifacts

### When to Use
- Final output requiring best quality
- Handheld shots with slight movement between frames
- Macro photography with focus breathing
- Images with fine detail (hair, fabric, circuit boards)

Example implementation: focus-stack (https://github.com/PetteriAimonen/focus-stack Petteri Aimonen C++)

---

## Areas with room for improvment
  What's Working Well

  The core algorithm port is solid - the wavelet transform, merge logic, and color reassignment are faithful to the C++ implementation. The fact that results are comparable to focus-stack validates that.

  Areas for Improvement

  1. Performance

  ~~The wavelet transform was the bottleneck~~ - **FIXED with Numba JIT!**

  The `_decompose_1d` and `_compose_1d` functions now use `@njit` decorators, achieving ~450x speedup:
  - Decompose (1024x768, 3 images): 24s → 0.05s
  - Compose: 8s → 0.02s

  Remaining bottlenecks:
  - Alignment (ECC algorithm) - could explore GPU acceleration
  - Color reassignment (`reassign.py`) - still pure Python loops (see note below)
  - The C++ version has OpenCL acceleration we skipped entirely

  **Note on color reassignment:** A Numba-accelerated version exists (`build_color_map_fast`,
  `reassign_colors_fast`) but produces visible halos at edges due to missing deduplication logic.
  The original Python version deduplicates gray values per-pixel, which improves edge quality.
  We use the slower but higher-quality version by default.

  2. We simplified the alignment

  The C++ version does contrast correction AND white balance matching before geometric alignment. We just do the geometric part. For well-lit, consistent exposures this is fine. For auto-exposure stacks, results might suffer.

  3. Memory

  We load everything into memory. The C++ version's batch processing (--batchsize) was specifically designed to limit memory usage. For large stacks or high-res images, this could hurt.

  Experimentation Opportunities

  This is where Python actually shines over C++:

  1. Different wavelets - We ported Complex Daubechies, but you could experiment with other families. PyWavelets has tons of options - might find something that works better for certain subjects.
  2. The merge decision - We pick max magnitude, but what about weighted blending? Or ML-based focus detection? Python makes this easy to prototype.
  3. Depth map work - We generate it but barely use it. The C++ has smoothing, inpainting, 3D preview... could be fun to explore or go a completely different direction.
  4. Handling ghosting - Moving subjects between frames cause artifacts. This is a known limitation of the algorithm - could experiment with detection/mitigation.


---

## Implementation Notes

### Module Organization
```
core/
├── stacker.py          # FocusStacker class, algorithm selection
├── complex_wavelet.py  # Wavelet transform, merge, denoise
├── align.py            # ECC alignment
├── grayscale.py        # PCA grayscale conversion
└── reassign.py         # Color reassignment
```

### Usage
```python
from focal.core.stacker import FocusStacker, StackAlgorithm

# Laplacian (fast)
stacker = FocusStacker(algorithm=StackAlgorithm.LAPLACIAN)
result = stacker.stack(image_paths)

# Complex Wavelet (quality)
stacker = FocusStacker(
    algorithm=StackAlgorithm.COMPLEX_WAVELET,
    consistency=2  # 0=none, 1=subband, 2=full denoising
)
result = stacker.stack(image_paths)
```

### References
- Forster, Van De Ville, Berent, Sage, Unser (2004): "Complex Wavelets for Extended Depth-of-Field: A New Method for the Fusion of Multichannel Microscopy Images"
- J.M. Lina (1997): "Image Processing with Complex Daubechies Wavelets"
- Original C++ implementation: [focus-stack](https://github.com/PetteriAimworthy/focus-stack)
