# Investigation: Wavelet Pipeline Artifacts on Bottlebrush Stack

**Date:** 2025-01-22
**Status:** Partial improvements found, fundamental issues remain

## Problem Statement

The wavelet (Complex Daubechies) algorithm produces severe artifacts on the Bottlebrush test stack compared to laplacian:
- Ghosting/doubling on stamens (especially dense bottom-right region)
- Noise in uniform black background
- Edge artifacts on leaves

This is a tripod shot where frames are already well-aligned.

## Investigation Process

### 1. Isolated the Problem Stage

Added debug output at various pipeline stages:

| Stage | Artifacts Present? |
|-------|-------------------|
| Source frames | No - clean inputs |
| Aligned frames | No - alignment not adding artifacts |
| Merged grayscale (before color reassignment) | **Yes** |
| Final output | Yes |

**Conclusion:** Problem is in wavelet fusion (`merge_wavelets`), not alignment or color reassignment.

### 2. Analyzed Depth Map (Frame Selection)

Visualized which frame contributes each wavelet coefficient:

- **Background regions:** Extremely noisy/speckled selection pattern
- **Dense stamen regions:** Chaotic frame mixing at pixel level

The max-magnitude selection becomes essentially random when:
- Coefficients are low magnitude (uniform regions like black background)
- Multiple frames have similar magnitude (dense overlapping detail)

### 3. Tested Skip-Alignment

Hypothesis: ECC alignment might be over-correcting already-aligned tripod frames.

Result: Skipping alignment helped slightly but artifacts persisted. **Alignment was not the root cause.**

### 4. Tested Magnitude Threshold

Added `magnitude_threshold` parameter to `merge_wavelets`. Only switch frames when coefficient magnitude exceeds threshold.

```python
mask = (absval > max_absval) & (absval > magnitude_threshold)
```

**Result with threshold=100:**
- Background noise: **Fixed** - now uses reference frame consistently
- Dense stamen artifacts: **Not fixed** - still present

### 5. Tested Median Filter on Depth Map

Applied 5x5 median filter to depth map after selection, before reconstruction. Forces spatial consistency in frame selection.

**Result:**
- Dense stamen artifacts: **Improved** - noticeably cleaner
- Still not as sharp as laplacian

### 6. Tested on Other Stacks

| Stack | Threshold Helps? | Issue Type |
|-------|-----------------|------------|
| Bottlebrush | Yes (background) | Low-magnitude noise |
| HTTin | No | OOF bokeh artifacts |
| Godetia | No | Smooth gradient textures |

The magnitude threshold specifically helps uniform dark backgrounds, not other artifact types.

## Root Cause

The wavelet fusion selects coefficients **per-pixel** based on maximum magnitude. This creates problems:

1. **Low-magnitude regions:** Near-zero coefficients mean sensor noise determines selection -> random frame switching in uniform areas

2. **Dense detail regions:** Many overlapping features at different focus planes have similar magnitudes -> creates patchwork of coefficients from different frames -> reconstructs as ghosting/smearing

The existing denoising (`consistency` parameter) is too weak:
- 2-out-of-3 subband voting doesn't help when all subbands are noisy
- Neighbor outlier detection requires ALL 4 neighbors to agree

## Potential Improvements

### Implemented (not yet integrated)

1. **Magnitude threshold** - Prevents noisy selection in low-contrast regions
2. **Median filter on depth map** - Enforces spatial consistency

### Not Implemented

3. **Block-based selection** - Select entire NxN regions from same frame
4. **Weighted selection** - Consider local variance, not just magnitude
5. **Edge-aware smoothing** - Preserve edges while smoothing depth map

## Code Changes

### Changes to `src/focal/core/complex_wavelet.py`

```diff
 def merge_wavelets(
     wavelets: list[np.ndarray],
     consistency: int = 2,
+    magnitude_threshold: float = 0.0,
 ) -> np.ndarray:
     """
     Merge multiple wavelet images by selecting highest magnitude coefficients.

     Args:
         wavelets: List of wavelet coefficient arrays (H, W, 2)
         consistency: Denoising level 0-2
+        magnitude_threshold: Minimum squared magnitude to allow frame switching.
+            Below this threshold, reference frame (first) is used. Prevents
+            noisy selection in low-contrast regions.

     Returns:
         Merged wavelet coefficients
     """
     h, w = wavelets[0].shape[:2]

     result = wavelets[0].copy()
     depth_map = np.zeros((h, w), dtype=np.uint16)
     max_absval = get_sq_absval(result)

     # Select maximum magnitude wavelet at each position
     for i, wavelet in enumerate(wavelets[1:], 1):
         absval = get_sq_absval(wavelet)
-        mask = absval > max_absval
+        # Only switch frames if magnitude is above threshold AND greater than current max
+        mask = (absval > max_absval) & (absval > magnitude_threshold)

         result[mask] = wavelet[mask]
         depth_map[mask] = i
         max_absval = np.maximum(max_absval, absval)
```

### Changes to `src/focal/core/stacker.py`

```diff
 class FocusStacker:
     """Focus stacking with selectable algorithm."""

     def __init__(
         self,
         algorithm: StackAlgorithm = StackAlgorithm.LAPLACIAN,
         num_levels: int | None = None,
         kernel_size: int = 5,
         consistency: int = 2,
+        skip_alignment: bool = False,
     ):
         self.algorithm = algorithm
         self.num_levels = num_levels
         self.kernel_size = kernel_size
         self.consistency = consistency
+        self.skip_alignment = skip_alignment
         self.last_transforms: dict[int, np.ndarray] = {}
```

And in `_stack_complex_wavelet`, the alignment section:

```diff
-        for i, (gray, color) in enumerate(zip(grayscales, expanded_images)):
-            if i == ref_idx:
-                aligned_colors.append(color)
-                aligned_grays.append(gray)
-                self.last_transforms[i] = np.eye(2, 3, dtype=np.float32)
-            else:
-                aligned_color, transform = align_image(...)
-                ...
+        if self.skip_alignment:
+            # Skip alignment - use images as-is with identity transforms
+            aligned_colors = expanded_images
+            aligned_grays = grayscales
+            for i in range(len(grayscales)):
+                self.last_transforms[i] = np.eye(2, 3, dtype=np.float32)
+        else:
+            for i, (gray, color) in enumerate(zip(grayscales, expanded_images)):
+                # ... existing alignment code ...
```

### Changes to `src/focal/cli.py`

```diff
     stack_parser.add_argument(
         "--compare",
         action="store_true",
         help="Run both algorithms and save with suffixed names",
     )
+    stack_parser.add_argument(
+        "--skip-alignment",
+        action="store_true",
+        help="Skip ECC alignment in wavelet pipeline (for testing)",
+    )
```

### Median Filter (Not Integrated - Experimental)

This was tested but not integrated into the codebase:

```python
def merge_with_median_smooth(wavelets, consistency=2, magnitude_threshold=100.0):
    '''Merge with median filter on depth map for spatial consistency.'''
    h, w = wavelets[0].shape[:2]

    result = wavelets[0].copy()
    depth_map = np.zeros((h, w), dtype=np.uint8)
    max_absval = complex_wavelet.get_sq_absval(result)

    for i, wavelet in enumerate(wavelets[1:], 1):
        absval = complex_wavelet.get_sq_absval(wavelet)
        mask = (absval > max_absval) & (absval > magnitude_threshold)
        result[mask] = wavelet[mask]
        depth_map[mask] = i
        max_absval = np.maximum(max_absval, absval)

    # Apply median filter to depth map for spatial consistency
    depth_map_smooth = cv2.medianBlur(depth_map, 5)
    depth_map_smooth = np.clip(depth_map_smooth, 0, len(wavelets) - 1)

    # Rebuild result using smoothed depth map
    for y in range(h):
        for x in range(w):
            result[y, x] = wavelets[depth_map_smooth[y, x]][y, x]

    return result
```

## Recommendations

1. **Keep magnitude_threshold** as a parameter (default ~100 for typical images)
2. **Consider median filter** for dense-detail scenarios, but needs more testing
3. **For tripod shots**, laplacian remains the better choice
4. **Wavelet's value** is primarily for handheld stacks requiring alignment

## Test Commands

```bash
# Run wavelet with threshold (after integrating changes)
uv run python -m focal.cli stack ../test-stacks/051110_D2x_Bottlebrush_Stack04_1200w \
    -o ../test-outputs/bottlebrush_test.jpg --algorithm wavelet --skip-alignment

# Compare algorithms
uv run python -m focal.cli stack ../test-stacks/051110_D2x_Bottlebrush_Stack04_1200w \
    -o ../test-outputs/bottlebrush.jpg --compare
```

## Files Modified

- `src/focal/core/complex_wavelet.py` - Added magnitude_threshold parameter
- `src/focal/core/stacker.py` - Added skip_alignment parameter
- `src/focal/cli.py` - Added --skip-alignment flag

tl;dr (for me);
  We did learn some useful things even without fix:                                                                                                 
                                                                                                                                                      
  1. The problem is definitively in wavelet fusion - not alignment, not color reassignment                                                            
  2. Per-pixel coefficient selection is the fundamental issue - laplacian's spatial blending avoids this                                              
  3. The existing consistency denoising is nearly useless - too weak for real noise patterns                                                          
  4. Threshold helps background, median helps dense detail - both partial solutions if you ever revisit           
