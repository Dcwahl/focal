# Real-stack baseline and GUI exercise — September 7, 2026

Branch: `fix/stacking-correctness`, with the first correctness fixes applied.
No further production algorithm changes were made during this evaluation.

## Inputs and method

- `test_stacks/focus_stack_test-1`: 10 Canon EOS R10 JPEGs, 6000×4000,
  binder clip, glasses, wire, tabletop, changing illumination/background.
- `test_stacks/focus_stack_test-2`: 8 JPEGs at the same resolution, figurine
  with painted detail, shiny surfaces, rounded edges, and a thin staff.
- All frames were included. Original JPEGs were left unchanged.
- Both algorithms were run at 1500×1000 with and without shared alignment.
- Both algorithms were also run at native 6000×4000 using the same aligned images.
  These were full-image, full-sequence runs, not independent crop stacks.
- Alignment used ordinary BGR luminance and the middle frame (6 and 5 respectively).
  The aligned images were written losslessly as PNG, then both algorithms ran with
  their own alignment disabled. This isolates fusion quality; it is not a test of
  wavelet's separate PCA-based alignment default.
- No exposure/color normalization, manual retouching, or source-frame removal was
  applied to the benchmark outputs. The separately saved GUI output includes one
  test brush click.
- PNG masks record common coverage. Fusion currently still includes reflected
  borders; the masks were recorded for diagnosis, not integrated into fusion.

Inputs, transforms, source hashes, crop coordinates, timings, library versions,
and GUI checks are recorded in `real_stack_metrics_2026-09-07.json`.

## Visual findings

### Binder clip / glasses

The source sequence visibly changes in background appearance and brightness.
Mean preview grayscale ranges from 90.2 to 100.5, but this global measurement is
descriptive only: it does not isolate exposure from changing focus or local motion.
Registration estimates show substantial magnification change over the sequence;
alignment helps geometric consistency but cannot reconcile all lighting changes.

Laplacian produces a much smoother overall image than wavelet. At 1:1, however,
it softens local detail relative to a sharp source. The wavelet output has severe
blotchy/speckled background transitions that remain after shared alignment. Thus
alignment alone does not solve this case. Both should be treated as imperfect
baselines, not successful final renders.

Useful files under `test_outputs/real_stacks/focus_stack_test-1/`:

- `sources.jpg`: all 10 source previews.
- `preview_comparison.jpg`: alignment off/on × both fusion methods.
- `native_comparison.jpg`: downsampled overview of native-resolution fusion.
- `crop_background.png`: especially clear example of wavelet speckling and color
  patchwork. The single source is not a color ground truth for changing illumination.
- `crop_clip_front.png`, `crop_wire.png`, `crop_glasses.png`: native detail checks.
- `native_results/laplacian.png` and `native_results/complex_wavelet.png`: full outputs.

### Figurine

The mean source grayscale changes less (85.0–87.6). Shared alignment visibly reduces
doubled contours and improves the overall fusion. Native crops reveal a tradeoff
that is much less apparent in the overview: Laplacian softens the mouth and collar
detail relative to the sharp source; wavelet preserves more texture but introduces
speckling and abrupt color changes, particularly near the dark collar/cuff.

The corresponding files under `test_outputs/real_stacks/focus_stack_test-2/` include
`crop_face.png`, `crop_buttons.png`, `crop_left_edge.png`, and `crop_hand_staff.png`.
The face crop is a useful first comparison for further fusion changes.

Crop sheets show one aligned source and both outputs at native pixel size.
The source was chosen using maximum local Laplacian variance over that crop.
This is a reproducible diagnostic selection, **not ground truth**: noise, brightness,
and mixed-depth regions can affect that choice.

## Measured performance

Each fusion ran in a fresh process with OpenCV limited to four threads. Numbers
are individual observations on this machine, not repeated controlled benchmarks.
Fusion time includes input decoding but excludes output PNG writing and shared
alignment. Peak RSS is measured after output writing and includes process overhead.

| Stack | Algorithm | Fusion seconds | Peak RSS GiB |
|---|---|---:|---:|
| 10 frames, 24MP | Laplacian | 8.96 | 11.16 |
| 10 frames, 24MP | Wavelet | 20.21 | 3.88 |
| 8 frames, 24MP | Laplacian | 7.28 | 9.00 |
| 8 frames, 24MP | Wavelet | 16.25 | 3.18 |

Shared alignment plus writing aligned PNGs took 11.70 and 8.75 seconds respectively.
Common geometric coverage was 93.19% and 96.00%. Coverage is not an accuracy score.
The large Laplacian memory footprint confirms that longer native-resolution stacks
need memory work rather than merely a larger GUI cache.

## Actual Qt GUI exercise

`gui_smoke.py` launches `MainWindow` with Qt's offscreen platform. It drives mouse
and keyboard events through QTest, with native file-dialog responses substituted.
On all eight figurine frames resized to 1500×1000, these checks passed:

1. Open source files through the Open button.
2. Enable alignment and click Stack; wait for the real background worker.
3. Select a different source through the list widget.
4. Enable Brush and paint through a viewport mouse event; verify pixels changed.
5. Undo and redo with keyboard shortcuts; verify exact pixel restoration.
6. Hold and release S; verify flash state and preservation of the edited result.
7. Save via the Save button; decode the PNG and verify it matches the edited array.

Screenshots and saved output are in `test_outputs/real_stacks/gui_smoke/`.
This demonstrates actual application-event testing without a general desktop-control
tool. It does not test the OS file chooser, desktop window management, real mouse
input, display color management, or long interactive sessions. Substack creation
and brush dragging across many points were not exercised in this smoke test.

## Reproduction

With the project installed, prepare once and feed the identical alignment to both
algorithms. Use `--width 1500` on `prepare` for a smaller run, or omit for native size:

```bash
python docs/investigations/evaluate_real_stacks.py prepare test_stacks/focus_stack_test-1 test_outputs/rerun/aligned
python docs/investigations/evaluate_real_stacks.py fuse test_outputs/rerun/aligned test_outputs/rerun/results --algorithm laplacian
python docs/investigations/evaluate_real_stacks.py fuse test_outputs/rerun/aligned test_outputs/rerun/results --algorithm complex_wavelet
QT_QPA_PLATFORM=offscreen python docs/investigations/gui_smoke.py test_outputs/real_stacks/focus_stack_test-2/preview_inputs test_outputs/gui-rerun
```

## Next experimental questions

These two stacks are useful development cases, but should not become the entire
acceptance set. Keep the Lytro cases and synthetic invariants, and add independent
longer stacks when available.

The immediate questions are why Laplacian blends away available sharp detail, and
how to make wavelet frame selection/color reassignment stable in smooth regions
without losing fine detail. Separate illumination handling from focus selection:
global exposure scaling cannot generally repair local moving shadows/backgrounds.
Compare any selection-consistency or weighting change on both native crop sets and
the existing clean cases before making it the default.

## Subsequent substack GUI regression pass

Added `tests/test_substack_gui.py`: six offscreen Qt regression cases using small,
deterministic generated focus sequences. Four combinations cover Laplacian/wavelet
with main-stack alignment on/off. These exercise actual checkbox selection,
substack creation, preview pixels, a drag with overlapping paint dabs, alignment of
the painted pixels, switching sources before undo/redo, cache eviction/reselection,
flash-preview pixels, and PNG export. Two additional cases cover deletion of the
selected substack and re-registration after a new main result.

Fixed three related GUI issues: the source panel displayed the first input instead
of the fused substack; deleting the selected substack left it active for painting;
and cached substacks lacked refreshed alignment after a new main stack. The full
suite now passes **85 tests**. Existing Qt mouse-position deprecation warnings remain.

Also ran `gui_smoke.py --substack` on the eight figurine frames resized to 1500×1000,
creating a substack from frames 7–8 and painting from it. The source-switch,
undo/redo, flash, and save-roundtrip checks passed. Screenshots and the result record
are in `test_outputs/real_stacks/substack_gui_smoke/`. Native file-dialog responses
are still substituted. This pass does not establish stability across changing
substack algorithm settings, input removal/reindexing, or reentrant UI actions.
