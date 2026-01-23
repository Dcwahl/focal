# Phase 8: Performance Optimization

The algorithms and retouching workflow are solid. What's holding back a seamless experience is performance: OOM on large stacks, slow stacking times, and choppy brush response.

## Completed

### Memory Optimization (Jan 2026)

**Problem:** OOM when stacking 10-15 24MP images on 16GB Mac.

**Solution implemented:**

1. **Incremental wavelet merge** (`complex_wavelet.py`)
   - Added `merge_wavelet_incremental()` function
   - Merges wavelets one at a time instead of storing all N decompositions
   - Memory: O(1) instead of O(N) for wavelet phase

2. **Aggressive cleanup** (`stacker.py`)
   - Added `gc.collect()` calls between pipeline phases
   - Delete intermediate arrays (`images`, `expanded_images`, `grayscales`) after alignment
   - Delete `aligned_grays`, `aligned_colors` after color map is built

3. **Fast Numba color reassignment** (`reassign.py`)
   - Switched from slow Python `build_color_map`/`reassign_colors` to JIT-compiled versions
   - Fixed unsigned integer overflow bug: `int()` in Numba doesn't behave like Python's `int()` - must use `np.int64()` to avoid wrap-around when subtracting uint8 values

**Results (8 × 24MP Canon R10 images):**
- Peak memory: **2.67GB** (down from estimated 10-15GB)
- Time: **~17s** (fast reassign contributes to speed improvement)
- Quality: parity with original implementation

---

## Remaining Problem Areas

### 1. Stacking Speed — Medium Priority

**Symptoms:** Wavelet took 121s for Enrico (6 high-res frames). Now faster with optimizations but still noticeable on large stacks.

**Likely bottlenecks (need profiling to confirm):**
1. **ECC alignment** — Iterative optimization, runs N-1 times, slow on large images
2. **Wavelet decomposition** — Numba-jitted but still O(pixels × levels) per image
3. **Image I/O** — Loading large JPEGs/TIFFs

**Potential solutions:**

1. **Downscaled alignment** — ECC on 1/4 res is ~16× faster AND uses less memory.

2. **Parallel decomposition** — Wavelet decompose multiple images concurrently (multiprocessing). Currently sequential.

3. **Caching alignment transforms** — If user re-stacks same images with different algorithm, reuse computed transforms.

**Relevant files:**
- `src/focal/core/align.py` — ECC alignment
- `src/focal/core/complex_wavelet.py` — wavelet transform
- `src/focal/core/stacker.py` — orchestration

### 2. Brush Responsiveness — Lower Priority

**Symptoms:** Brush painting is choppy. Fast mouse movement results in discrete circles instead of smooth strokes (gaps between paint dabs).

**Likely causes:**
1. **Paint loop can't keep up** — Each brush dab may trigger expensive operations (array copies, display updates)
2. **Missing interpolation** — No interpolation between mouse sample points
3. **Synchronous updates** — Display refresh blocks next paint operation

**Potential solutions:**

1. **Interpolate between mouse points** — When mouse moves fast, interpolate intermediate positions and paint along the line. Standard brush engine technique.

2. **Batch paint operations** — Accumulate dabs, apply in batch, single display refresh.

3. **Async display updates** — Paint to buffer immediately, update display on next frame (decouple paint from render).

4. **Reduce per-dab overhead** — Profile what happens on each `brush_paint` signal. Optimize the hot path.

**Relevant files:**
- `src/focal/ui/image_viewer.py` — brush cursor, mouse events, `brush_paint` signal
- `src/focal/ui/main_window.py` — handles `brush_paint`, does actual pixel copying

## Success Criteria

- [x] 15-image 24MP stack completes without OOM on 16GB machine
- [ ] Identify and document top 3 speed bottlenecks
- [ ] Implement at least one speed optimization (likely downscaled alignment)
- [ ] Brush interpolation for smooth strokes (if time permits)
