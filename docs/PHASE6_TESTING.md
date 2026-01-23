# Phase 6: Automated Testing Infrastructure

## Overview

This phase establishes infrastructure for running focus stacking algorithms against test image sets and evaluating results. The goal is to enable systematic testing as we iterate on algorithms and identify issues across diverse image types.

## Key Discovery

**Claude can visually inspect images** using the `Read` tool. This enables direct evaluation of stacking results - spotting halos, ghosting, mush, alignment issues, etc. This is the primary evaluation method; quantitative metrics supplement but don't replace visual review.

## Test Stacks Available

Located in `../test-stacks/` (i.e., `temp2/test-stacks/`):

| Folder | Images | Notes |
|--------|--------|-------|
| `051110_D2x_Bottlebrush_Stack04_1200w` | 10 | Macro flower, red stamens, black bg |
| `godetia_stack04_1200w` | 22 | Flower macro |
| `Enrico_Curschellas` | ? | TBD |
| `HTTin` | 13 | TBD |
| `Peter_Lee` | 9 | TBD |
| `Peter_Lee_2` | 13 | TBD |
| `Rifle` | 24 | TBD |

## Implementation Plan

### 1. CLI Module (`src/focal/cli.py`)

Create a command-line interface that mirrors the GUI's stacking logic exactly.

```bash
# Basic usage
uv run python -m focal.cli stack /path/to/images -o result.jpg

# Specify algorithm
uv run python -m focal.cli stack /path/to/images -o result.jpg --algorithm wavelet

# Run both algorithms for comparison
uv run python -m focal.cli stack /path/to/images -o result.jpg --compare
```

**Requirements:**
- Use `FocusStacker` from `focal.core.stacker` directly (same code path as GUI)
- Support both algorithms: `laplacian` and `wavelet` (complex wavelet)
- Output formats: JPEG, TIFF, PNG
- `--compare` flag generates both algorithm outputs with suffixed names
- Print timing information to stdout
- Exit codes: 0 success, 1 error

**Implementation notes:**
- Use `argparse` or `click` for argument parsing
- Keep it minimal - this is a testing tool, not a user-facing CLI
- No progress bars needed (print simple status messages)

### 2. Evaluation Workflow

When evaluating a test stack:

1. **Run the stack:**
   ```bash
   cd /Users/diegowahl/temp2/focal
   uv run python -m focal.cli stack ../test-stacks/STACK_NAME -o ../test-outputs/STACK_NAME_result.jpg --compare
   ```

2. **View results:**
   - Use `Read` tool on output image(s)
   - Compare against source frames if needed (also via `Read`)

3. **Document issues:**
   - Note specific artifacts (halos, ghosting, mush, color shifts)
   - Reference frame numbers if issue comes from specific source frames
   - Update `docs/TODO.md` with actionable bugs

### 3. Output Directory Structure

```
temp2/test-outputs/
├── bottlebrush_laplacian.jpg
├── bottlebrush_wavelet.jpg
├── godetia_laplacian.jpg
├── godetia_wavelet.jpg
└── ...
```

### 4. Optional: Metrics Module

Quantitative metrics for regression tracking (lower priority than visual eval):

- **Sharpness:** Laplacian variance of result
- **Contrast:** Standard deviation of luminance
- **Edge density:** Canny edge pixel percentage

These help detect regressions but won't catch visual artifacts like halos.

## Common Artifacts to Look For

| Artifact | Description | Typical Cause |
|----------|-------------|---------------|
| **Halos** | Bright/dark outlines at high-contrast edges | Misaligned frames, pyramid blending |
| **Ghosting** | Semi-transparent duplicates | Subject movement, alignment failure |
| **Mush** | Loss of fine detail, soft areas | Over-blending, wrong focus selection |
| **Color shifts** | Unnatural color in blended regions | Color reassignment issues |
| **Seams** | Visible boundaries between regions | Abrupt weight transitions |

## For Subagents

If you're implementing or running tests:

1. **The stacking code lives in** `src/focal/core/stacker.py` - the `FocusStacker` class
2. **You can view images** by using the `Read` tool on image files (JPEG, PNG, TIFF)
3. **Test stacks are in** `../test-stacks/` relative to the focal repo
4. **Always use `uv run`** to execute Python in this project (dependencies managed by uv)
5. **Save outputs to** `../test-outputs/` to keep them separate from source images

## Success Criteria

- [x] CLI module implemented and working
- [x] Can run both algorithms on any test stack from command line
- [x] Can visually evaluate results via Read tool
- [ ] Initial evaluation of all 7 test stacks documented
- [ ] Any discovered issues added to TODO.md

---

## Evaluation Results

### Bottlebrush (9 frames, tripod, black background)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Stamens** | Very sharp, crisp needles | ❌ Severe ghosting/doubling artifacts, especially bottom-right |
| **Background** | Clean pure black | Visible noise/grain contamination |
| **Leaves** | Good detail | Right leaf has doubling/edge artifacts |
| **Overall** | ⭐ Excellent | Poor — significant degradation |

**Winner: Laplacian** — This is a tripod shot with perfectly aligned frames. Wavelet's ECC alignment appears to be *adding* error rather than correcting it, introducing:
- Ghosting/doubling on fine stamens (most visible bottom-right where densely packed)
- Noise in uniform black background (interpolation artifacts?)
- Edge artifacts on leaves

**Key insight:** For already-aligned tripod shots, the alignment step may be counterproductive. Consider making alignment optional for wavelet, or auto-detecting when frames are already well-aligned.

**Investigation needed:** Output intermediate merged grayscale (before color reassignment) to isolate whether degradation comes from alignment, wavelet fusion, or color reassignment.

### Godetia (21 frames, flower macro)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Petals** | Smooth, natural texture | Streaky/painted appearance |
| **Stamens** | Sharp, good detail | Slightly mushy |
| **Colors** | Accurate | Some color bleeding (upper left) |
| **Artifacts** | None obvious | Dark shadow artifact on left, unnatural texture |

**Winner: Laplacian** — Wavelet has strange painted/waxy texture on petals and color bleeding artifacts. Likely a color reassignment issue or over-alignment introducing warping.

### Peter_Lee (9 frames, mineral/crystal macro)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Alignment** | None — massive ghosting | ECC alignment worked well |
| **Crystal detail** | Completely smeared | Sharp facets visible |
| **Small crystals** | Unrecognizable blur | Good detail on white cubes |
| **Usability** | ❌ Unusable | ✓ Usable output |

**Winner: Wavelet** — Source frames had significant shift (handheld). Without alignment, laplacian completely failed. Wavelet's ECC alignment saved this stack.

### Summary So Far

| Stack | Frames | Winner | Key Factor |
|-------|--------|--------|------------|
| Bottlebrush | 9 | Laplacian | Tripod, high-contrast fine detail |
| Godetia | 21 | Laplacian | Wavelet artifacts on smooth gradients |
| Peter_Lee | 9 | Wavelet | Handheld — alignment required |

### HTTin (13 frames, stink bug macro)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Bug detail** | Sharp, good texture | Similar sharpness |
| **Antennae** | Clean | Clean |
| **Background bokeh** | Natural, smooth | Watery/blotchy artifacts |
| **OOF areas** | Natural appearance | Processed/artificial look |

**Winner: Laplacian** — In-focus subject similar, but wavelet introduces visible artifacts in out-of-focus regions giving an over-processed look.

### Rifle (24 frames, still life)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Barrel detail** | Sharp | Comparable |
| **Background** | Good | Good |
| **Overall** | Clean | Clean |

**Tie** — Both algorithms produce similar results. Likely a stable tripod shot with minimal frame movement.

### Enrico_Curschellas (6 frames, model train, studio shot)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Train detail** | Sharp | Sharp |
| **Background** | Clean blue | Clean blue |
| **Text/logos** | Crisp | Crisp |

**Winner: Wavelet** (marginal) — Studio shot with uniform background. Wavelet produces slightly better detail/sharpness to trained eye despite similar appearance at first glance. (Note: wavelet took 121s due to high-resolution source images.)

### Peter_Lee_2 (13 frames, ladybug on mineral, microscopy)

| Aspect | Laplacian | Wavelet |
|--------|-----------|---------|
| **Overall** | ❌ Complete disaster | ✓ Sharp, detailed |
| **Subject** | Unrecognizable blur | Ladybug clearly visible |
| **Crystals** | Ghosted mess | Sharp white cubes |

**Winner: Wavelet** — Another handheld microscopy shot from same photographer as Peter_Lee. Without ECC alignment, laplacian is completely unusable.

---

## Summary

| Stack | Frames | Winner | Key Factor |
|-------|--------|--------|------------|
| Bottlebrush | 9 | Laplacian | Tripod, high-contrast fine detail |
| Godetia | 21 | Laplacian | Wavelet artifacts on smooth gradients |
| Peter_Lee | 9 | **Wavelet** | Handheld — alignment required |
| HTTin | 13 | Laplacian* | Wavelet artifacts in OOF regions (*but better focus selection) |
| Rifle | 24 | Tie | Stable tripod shot |
| Enrico | 6 | Wavelet | Marginally sharper detail |
| Peter_Lee_2 | 13 | **Wavelet** | Handheld — alignment required |

**Score: Laplacian 3, Wavelet 3, Tie 1**

## Key Observations

1. **Laplacian** excels on tripod shots with minimal frame shift — faster and cleaner results
2. **Laplacian** fails catastrophically on handheld stacks (no alignment) — completely unusable
3. **Wavelet** handles misaligned frames via ECC alignment — essential for handheld/microscopy
4. **Wavelet** introduces artifacts on some subjects:
   - Streaky/painted textures on smooth gradients (godetia petals)
   - Artifacts in OOF bokeh areas (HTTin background)
   - Likely a **color reassignment issue** rather than wavelet fusion itself
5. **Wavelet** may have better focus selection even when artifacts present (HTTin trichomes)

## Potential Improvements

- [ ] Investigate color reassignment artifacts in wavelet pipeline
- [ ] Consider adding optional alignment to laplacian algorithm
- [ ] Profile wavelet performance on high-res images (Enrico took 121s for 6 frames)
