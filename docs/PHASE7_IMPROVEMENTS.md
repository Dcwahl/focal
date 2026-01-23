# Phase 7: Algorithm Improvements

Based on findings from Phase 6 testing across 7 diverse test stacks.

## Context

Testing revealed a clear split: **Laplacian** excels on tripod shots (fast, clean), **Wavelet** is essential for handheld (ECC alignment). However, wavelet introduces artifacts on some subjects that appear to stem from color reassignment rather than the wavelet fusion itself.

## Improvement Areas

### 1. ~~Investigate Color Reassignment Artifacts~~ (Investigated - Dead End)

**Problem:** Wavelet results show streaky/painted textures on smooth gradients (Godetia petals) and blotchy artifacts in OOF bokeh areas (HTTin background).

**Investigation Results:**

✅ **Root cause confirmed:** Color reassignment IS the culprit for Godetia and HTTin (not wavelet fusion). Debug output showed merged grayscale was clean; artifacts appeared only after `reassign_colors()`.

Note: Bottlebrush artifacts are different - they appear in the grayscale before reassignment, likely from alignment issues on tripod shots.

**What we tried:**
- Changed from nearest-neighbor color lookup to inverse-distance weighted blending
- Stored ALL gray/color pairs (not just unique grays) to enable proper interpolation
- Tested on Godetia, HTTin, and Bottlebrush

**Result:** No visible improvement despite 2-3x performance hit (24s→66s on Godetia). The weighted blending approach didn't produce perceptible quality gains.

**Why it's a dead end:** The fundamental problem is that grayscale→color mapping loses information. When source frames have genuinely different colors at the same pixel (due to focus/lighting variations), no blending math can recover the "correct" color - you're mapping one grayscale value to multiple possible colors.

**Possible future directions (not pursued):**
- Direct color wavelet fusion (skip grayscale entirely, do wavelet on RGB/LAB channels) - bigger architectural change
- Accept artifacts and rely on retouching tools

**Relevant files:**
- `src/focal/core/reassign.py` - color reassignment logic
- `src/focal/core/grayscale.py` - PCA grayscale conversion
- `src/focal/core/stacker.py` lines 230-237 - where reassignment is called

### 2. Make Alignment Optional for Both Algorithms (Medium Priority)

**Problem:**
- Laplacian has no alignment → fails on handheld stacks (Peter_Lee, Peter_Lee_2 unusable)
- Wavelet always aligns → *degrades* tripod shots (Bottlebrush has severe stamen ghosting, background noise)

**Proposal:** Make alignment a separate, user-controllable option for both algorithms.

**Benefits:**
- Laplacian + align: best quality on handheld
- Wavelet - align: cleaner output on tripod shots
- User controls the tradeoff based on their shooting conditions

**Implementation:**
- Extract alignment logic from `_stack_complex_wavelet()` into reusable function
- Add `align: bool` parameter to `FocusStacker` (or per-stack call)
- Laplacian: default `align=False`
- Wavelet: default `align=True` (current behavior), but allow `align=False`
- CLI flag: `--align` / `--no-align`
- GUI: checkbox in stacking options

**Relevant files:**
- `src/focal/core/align.py` - existing ECC alignment
- `src/focal/core/stacker.py` - both algorithm implementations

### 3. Profile Wavelet Performance (Low Priority)

**Problem:** Wavelet took 121 seconds for Enrico_Curschellas (6 high-res frames). Users with 50MP cameras will have poor experience.

**Investigation:**
- Profile to identify bottlenecks (alignment? wavelet decomposition? color map building?)
- Consider lazy/incremental approaches
- Evaluate if any steps can be parallelized

**Relevant files:**
- `src/focal/core/complex_wavelet.py` - wavelet transform
- `src/focal/core/align.py` - ECC alignment (likely slow on large images)

## For Subagents

1. **Test images are in** `../test-stacks/` and **results in** `../test-outputs/`
2. **Use `Read` tool** to view images directly for visual evaluation
3. **Run CLI with** `uv run python -m focal.cli stack ...`
4. **See Phase 6 doc** (`docs/PHASE6_TESTING.md`) for detailed per-stack evaluation results

## Success Criteria

- [x] Root cause of wavelet color artifacts identified (color reassignment for Godetia/HTTin, alignment for Bottlebrush)
- [ ] ~~Color artifacts fixed or significantly reduced~~ (dead end - fundamental limitation of grayscale→color approach)
- [x] Alignment made optional for both algorithms
- [ ] Re-run test stacks to verify improvements
