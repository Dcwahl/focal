# Focal session handoff

Updated: September 7, 2026. Read this first when resuming in a new session.

## User's intent and working preferences

Revive this focus-stacking application. Improve photographic quality using actual
evidence; retain the useful source-frame/substack retouch workflow. GUI appearance
is lower priority. Eventually support GUI, CLI, and a public Python library through
one shared core (CLI and importable FocusStacker already exist).

The user authorizes autonomous investigation, fixes, and testing and trusts routine
implementation choices. Keep updates concise; they know their own application's
workflow and do not need tutorials. Do not add unsolicited GUI walkthroughs.
Do not spawn agents unless explicitly authorized (current instructions prohibit it).

**The dataset is downloaded and fully extracted** (September 12, 2026) to
`test_stacks/dataset/focus_stack_dataset/focus_stack_dataset/dataset` — 107 GiB,
9,566 files, 94 scenes of 30 frames at 5184x3888, each with pre-registered
`aligned/` frames and Helicon Focus renders. ~195 GB free afterward. The 96 GiB
`test_stacks/dataset/focus_stack_dataset.zip` is still present and deletable.
A first benchmark has been run: see **docs/investigations/DATASET_BENCHMARK_2026-09-12.md**,
which supersedes the "select a subset" task. Read it before further quality work.

## Workspace and Git

- Workspace: `/home/diegowahl/focal`
- Branch: `fix/stacking-correctness`
- Base commit: `42d87ed` (January 22, 2026; memory/performance improvements).
- **Committed September 12, 2026** in two commits on this branch:
  `24f02ec` (correctness fixes + substack cache identity, 88 tests passing) and
  `fa1390b` (measurement harness and findings, no behaviour change).
- **Not pushed. No PR, merge, or deployment was performed.** The branch exists
  only locally; `git push -u origin fix/stacking-correctness` when wanted.
- `.git` is read-only inside the ordinary sandbox; branch creation required an
  approved escalation. Workspace files and `/tmp` are writable.
- `test_outputs/` was added to `.gitignore`. Original image extensions were already
  ignored. User-provided `test_stacks/` and generated outputs exist locally but
  must not be assumed to travel with a Git clone.
- No applicable AGENTS.md was found in the project. CLAUDE.md describes the project.

## Environment and running checks

**`/tmp/focal-review-venv` is GONE — /tmp was cleared.** The venv was rebuilt at
`/home/diegowahl/focal/.venv` (Python 3.12.3), which is gitignored and survives /tmp
clears. Use that path; the commands below have been updated. Fresh installs resolve
to newer numerics (cv2 5.0.0, numpy 2.5.3, numba 0.67.0), so the four versions below
were explicitly pinned back for comparability with all recorded metrics.
Most recent numerical dependencies match the original lockfile:
NumPy 2.2.6, OpenCV 4.11.0.86, Numba 0.63.1, llvmlite 0.46.0.
PySide6 remains 6.11.2; this is not a complete historical environment reconstruction.

```bash
cd /home/diegowahl/focal
# Launch the actual GUI on this machine:
.venv/bin/focal
# Full test suite:
QT_QPA_PLATFORM=offscreen NUMBA_CACHE_DIR=/home/diegowahl/.cache/focal-numba .venv/bin/python -m pytest -q
# Only the new substack GUI regression cases:
NUMBA_CACHE_DIR=/home/diegowahl/.cache/focal-numba .venv/bin/python -m pytest tests/test_substack_gui.py -q
```

**Last full test result: 85 passed, 16 existing Qt mouse-position deprecation
warnings** — re-confirmed September 12, 2026 in the rebuilt `.venv`, matching exactly. No production changes followed that run; the later weight experiments
are standalone subclasses/scripts only. `git diff --check` passed afterward.

If /tmp has been cleared, rebuild with `uv venv` and `uv pip install -e '.[dev]'`
in an allowed environment, then pin the numerical versions above for comparability.
Use `UV_CACHE_DIR=/tmp/focal-uv-cache` if the default cache is read-only. Downloads
needed approved network escalation. Plain `uv sync` may pull the macOS-specific
pyobjc development dependency; the pip optional-extra route avoids that.

## Completed production fixes

1. `src/focal/core/stacker.py`: Laplacian zero-focus weights fall back to equal
   contributions. Two identical middle-gray images previously became solid black.
2. `src/focal/core/grayscale.py`: PCA normalization falls back to standard BGR
   luminance for degenerate or mixed-sign weights, preventing divide-by-zero and
   clipping-prone negative weights.
3. `src/focal/core/align.py`: phase-correlation translation initialization for coarse
   ECC; carry coarse affine estimate into fine ECC with correct resolution scaling;
   keep the initial estimate when ECC fails instead of returning a mutated failed
   solve; log failures and reject nonfinite/nonpositive-determinant results.
4. Added `sample_aligned_region`: sample a result-space rectangle from a source with
   its entire affine warp and a source-coverage mask. Brush painting now respects
   rotation, scale, shear, and fractional translation. Flash uses the same geometry.
5. Main stack workers own their configuration/input list; completion reads that
   worker's transforms rather than potentially changed UI settings.
6. Substack GUI fixes: preview the **fused substack**, not its first input; deleting
   the selected substack stops painting from it; lazily restore missing substack
   registration after a new main result, including when its pixels remain cached.

New tests:
- `tests/test_quality_regressions.py`: 18 numerical/brush/worker regressions.
- `tests/test_substack_gui.py`: six real offscreen Qt cases. Four combinations of
  algorithm and alignment cover creating a substack, preview pixels, real drag
  painting, source switching, exact repeated undo/redo, cache eviction/reselection,
  flash pixels, and saving. Two cover selected-substack deletion and restacking.
  Tiny generated sequences mean these tests require no external datasets.

## Baseline findings still relevant

The program is a useful prototype, but the fusion-quality problems are not solved.

- Production wavelet merge uses `merge_wavelet_incremental` and **ignores
  `self.consistency`**. Threshold is zero. The older merge/denoising code exists but
  is bypassed. Restoring it naïvely would retain every decomposition in memory.
- Laplacian blending loses source detail; wavelet often preserves more texture
  but introduces speckling/blotchy color boundaries in smooth or inconsistent areas.
- Global affine alignment does not solve moving subjects/backgrounds or local
  lighting changes. Fusion still treats reflected borders as real content; common
  coverage masks have only been saved for evaluation, not integrated into fusion.
- Memory is high: native 10×24MP Laplacian peaked at **11.16 GiB**, wavelet at
  **3.88 GiB**. Eight-frame case: **9.00/3.18 GiB**. Both load all frames; wavelet's
  incremental coefficient merge does not make the entire pipeline constant-memory.
- Input/output is uint8 OpenCV; 16-bit/color-managed/RAW workflows are unfinished.
- Substack lifecycle still needs work around algorithm settings changing before
  eviction/recomputation, source removal/reindexing, and reentrant UI actions.
- Old `docs/PHASE6_TESTING.md` often compared unaligned Laplacian against aligned
  wavelet, so those historical "winner" claims confound alignment and fusion.
- Old `docs/investigations/wavelet_bottlebrush_artifacts.md` contains useful leads.

## Local datasets and existing evaluation artifacts

Original user inputs, untouched:
- `test_stacks/focus_stack_test-1/`: 10 Canon EOS R10 JPEGs, 6000×4000, binder clip,
  glasses, wire/tabletop. Lighting/background changes; user noticed an arm enters
  the background. Do not make "perfectly reconstruct this moving scene" a target.
- `test_stacks/focus_stack_test-2/`: 8 JPEGs, 6000×4000, miniature figurine.
  Alignment helps; native mouth/collar/cuff crops expose detail loss and artifacts.

Artifacts under `test_outputs/real_stacks/focus_stack_test-{1,2}/`:
- `preview_inputs/`: 1500×1000 source copies.
- `preview_aligned/`, `native_aligned/`: shared BGR-luminance affine alignment,
  middle reference frame (6 or 5), lossless PNGs. Both fusion algorithms were fed
  these exact same images with internal alignment disabled.
- `preview_unaligned_results/`, `preview_aligned_results/`, `native_results/`:
  both algorithms' PNGs and per-process timing/peak-RSS JSON.
- `sources.jpg`, `preview_comparison.jpg`, `native_comparison.jpg`.
- `crop_*.png`, `crops.json`: fixed native 400×400 diagnostic regions, with a source
  chosen by maximum local Laplacian variance. That source is **not ground truth**.
- `*_alignment.json` and `*_overlap.png`: registration and coverage records.

`/tmp/focal-review/lytro/` holds Lytro pairs 01, 05, 10 and the attribution readme.
These are small real-image pair checks, not a long-stack benchmark. Older synthetic
and before/after outputs also live in `/tmp/focal-review*`; those may be ephemeral.

Evaluation utilities:
- `docs/investigations/baseline_probe.py`: known-answer diagnostics, optional Lytro.
- `docs/investigations/evaluate_real_stacks.py`: prepare shared alignment and run a
  fusion method in a separately measured process. CLI docs in its module docstring.
- `docs/investigations/gui_smoke.py`: actual Qt event-driven smoke workflow, with
  native file chooser responses substituted. Optional `--substack` creates/paints
  the last two frames and switches source before undo/redo.
- Real GUI smoke artifacts: `test_outputs/real_stacks/gui_smoke/` and
  `test_outputs/real_stacks/substack_gui_smoke/`. Both succeeded on the eight
  figurine previews. Substack screenshot: `substack_selected.png`.

No general desktop computer-use tool was connected. Browser tools were present.
Qt offscreen rendering + QTest mouse/key events successfully exercised the native
application; this does not test OS dialogs, window management, or display color.
The inherited environment has `QT_QPA_PLATFORM=wayland;xcb`; explicitly select
`offscreen` for GUI automation or QApplication can abort in the sandbox.

## Latest work: experimental Laplacian focus weighting

The last user asked for something useful to do while the dataset downloads. We ran
a bounded experiment on recovering detail by strengthening frame selection.

**This has NOT landed in GUI/CLI defaults or production FocusStacker.** It lives in
`docs/investigations/laplacian_weight_experiment.py`, subclass `WeightedStacker`.

Candidates:
- Baseline power 1, no smoothing (existing production algorithm).
- `p2_s0`: square focus scores before normalizing (2:1 becomes 4:1).
- `p4_s0`: fourth power (2:1 becomes 16:1).
- `p4_s1`: fourth power plus sigma-1 Gaussian smoothing of weights at every level.

Ran six extra **full 24MP** fusions on the same aligned user stacks, all four methods
on the three Lytro pairs, and controlled clean/noisy/residually misaligned sequences.

Findings:
- More decisive weighting visibly recovers some figurine mouth/face texture.
- Power 2 gets much of the benefit; power 4 adds more grain and makes changing
  background color/brightness more pronounced. Neither fixes scene motion.
- Controlled 12-frame noisy case: whole-image MAE baseline 2.955, power 2 1.619,
  power 4 1.581, smoothed power 4 1.561. **But flat-region MAE worsens** from
  1.124 to 1.193, 1.440, 1.334 respectively. Do not hide this behind the overall score.
- Constant/duplicate preservation passed within one uint8 level. Stronger weighting
  does not materially improve the one-pixel residual-misalignment diagnostic.
- Lytro contact-sheet inspection showed no gross failure, not a comprehensive pass.
- Keep **power 2 as a candidate**, not an established new default. A confidence-aware
  fallback in weak-focus regions is a better next experiment than higher exponents.

Key comparison for user/new session:
`test_outputs/real_stacks/focus_stack_test-2/weight_experiment/crop_face.png`
Columns: baseline, power 2, power 4, power 4+smoothing. Also inspect `crop_left_edge.png`
and stack 1's `weight_experiment/crop_background.png`.

Each `weight_experiment/` contains native candidate PNGs, `overview.jpg`, crops and
timing JSON. `test_outputs/weight_experiment/` contains controlled checks and
`lytro_comparison.jpg`. `weight_experiment_controls.py` reproduces the synthetic
checks exactly; its saved-script rerun matched the original metrics exactly.

```bash
.venv/bin/python docs/investigations/laplacian_weight_experiment.py test_outputs/real_stacks/focus_stack_test-2/native_aligned test_outputs/weight-rerun --power 2
.venv/bin/python docs/investigations/weight_experiment_controls.py test_outputs/weight-controls-rerun
```

## Reports and durable numerical records

All in `docs/investigations/` (currently untracked, do not lose them):
- `BASELINE_REVIEW_2026-09-07.md`, `baseline_metrics_2026-09-07.json`
- `CORRECTNESS_FIXES_2026-09-07.md`, `correctness_metrics_2026-09-07.json`
- `REAL_STACK_REVIEW_2026-09-07.md`, `real_stack_metrics_2026-09-07.json`
- `LAPLACIAN_WEIGHT_EXPERIMENT_2026-09-07.md`,
  `laplacian_weight_metrics_2026-09-07.json`

The first baseline used newer numerical packages. The correctness report includes
a rerun of original source and fixed source under the same locked numerical versions.

## External dataset sources already located

- Download in progress (user-managed): https://github.com/araujoalexandre/FocusStackingDataset
  LSFD: 94 real high-resolution bursts, 30 frames each, Helicon-derived **pseudo**
  ground truth. Inspect format and archive layout before choosing/preparing inputs.
  References are not perfect physical ground truth. Paper: https://arxiv.org/abs/2311.17846
- Lytro author repository: https://github.com/mnnejati/LytroDataset
- MFFW, harder defocus spread cases, located but not downloaded:
  https://arxiv.org/abs/2002.04780
- Historical Helicon demo stacks: https://www.heliconsoft.com/helicon-focus-gallery/
  These include the old Bottlebrush/Godetia/etc. Their stated usage terms specifically
  cover testing Helicon/demonstration/marketing and request contact before sharing
  results; do not assume a generic redistributable benchmark license. Not downloaded.

## Suggested next steps

1. Preserve/review the current uncommitted work; this file and tests are enough to
   resume without relying on conversation context. Do not reset the working tree.
2. If download is ready: inspect ZIP member sizes and available disk, identify a
   small evaluation subset with references, and extract only what is useful initially.
3. Test moderate Laplacian weighting on independent stacks before promoting it.
   Explore a low-confidence/noise-aware fallback, with the existing flat-region
   diagnostic as a guardrail. Avoid overfitting to two imperfect user sequences.
4. Investigate memory-aware wavelet consistency and color reassignment separately.
   Keep alignment identical when comparing fusion methods.
5. Preserve substack retouching regressions. GUI styling and broader CLI/library
   product work remain lower priority than photographic quality and correctness.

Do not claim the image-quality problems are solved. Current production work fixes
specific correctness issues; the latest sharpness change is an evaluated experiment.

## Dataset benchmark headline numbers (September 12, 2026)

Six-scene frozen subset, 30 frames each, half resolution, both algorithms fed the
dataset's own `aligned/` frames with alignment disabled so this measures fusion only.
No ground truth exists, so results are scored against a per-pixel **focus envelope**
(max sharpness available across the 30 sources). Full method and caveats in
`docs/investigations/DATASET_BENCHMARK_2026-09-12.md`.

| method | recovery | halo | flat_excess | peak RSS | time |
|---|---|---|---|---|---|
| laplacian | 0.346 | 0.000 | 0.315 | 7.06 GiB | 5.2 s |
| complex_wavelet | 1.026 | 0.234 | 1.873 | 2.79 GiB | 13.3 s |
| helicon_focus (reference) | 0.884 | 0.004 | 1.223 | — | — |

- **Laplacian averages instead of selecting**: proportional weighting means the sharpest
  frame gets only 8.4% of the weight on needle2, the rest coming from 29 blurrier frames.
  This is a direct read of `_compute_weights`, not a metric, and is the most solid finding.
- **A power sweep overturns the earlier power-2 recommendation.** Mean recovery: p1 0.346,
  p2 0.592, p4 **0.802**, against Helicon 0.884, with p4's halo still 0.002 and no added
  flat-region grain. Confirmed under three independent sharpness metrics AND against true
  ground truth in `depth_gt_control.py`, and re-confirmed at native 6000x4000 on the
  user's own two stacks. **p4 must NOT ship as an unconditional default**: on stack 1's
  moving background it more than doubles colour blotching (lab_drift 9.96 -> 22.51),
  reproducing the earlier warning. Detail gain and background damage come from the same
  mechanism. **Resolved 2026-09-12, but not by confidence gating** - see below.
- **Wavelet's flat-region noise (~1.7x source) is real**; the stronger "invents contrast
  on 23% of pixels" claim was metric-inflated and was retracted (0.074 under Sobel).
- **Helicon is the balance to target**: high recovery with almost no invented contrast.
- Helicon renders are pixel-registered to `aligned/` (<0.1 px), so per-pixel comparison
  is valid. They are a reference only and never enter another method's score.
- **Native 20 MP x 30 frames projects to ~28 GiB for Laplacian vs 25 GiB available** —
  it will OOM. Wavelet projects to ~11 GiB. Native 30-frame runs are wavelet-only
  until Laplacian memory is addressed.

Two claims in the first draft were falsified by the verification pass and are struck
through in the report: "the failure worsens with stack depth" and "shipped Laplacian
barely beats not stacking". Read the verification section before quoting any number.

## Decisive exponent now ships (September 12, 2026)

`FocusStacker(focus_power=...)`, default `1.0`, `--focus-power` on the CLI, not in the
UI. Full method in `docs/investigations/CONFIDENCE_GATE_2026-09-12.md`.

**Confidence gating was built and falsified, not skipped.** On stack 1's background the
gated exponent moved lab_drift only 22.51 -> 20.52 against p1's 9.96. The reason is
measurable: that background sits at the **43rd percentile** of peak focus measure, its
peak/mean ratio (2.57) is indistinguishable from genuinely focused regions (2.52, 2.71),
and its winning frame is *more* spatially coherent than theirs. No focus-confidence
statistic separates it, because weak focus evidence was never the cause.

**The cause is exposure disagreement between frames.** That region's mean luma spans
29.8..101.7 across the 10 frames and essentially all of its cross-frame variation
survives a sigma-12 low-pass (31.04 of 31.13). A decisive exponent there chooses between
frames that differ in brightness, not focus.

**The fix is one pyramid level.** Brightness lives in the coarsest Laplacian level (the
base image); detail lives in the finer bands. Weighting the base proportionally at any
exponent gives, on stack 1: sobel recovery 0.788 (p4 uniform: 0.790, p1: 0.416) with
lab_drift 10.02 (p4 uniform: 22.51, p1: 9.96). Exempting two or three coarse levels adds
nothing, which confirms the whole regression lived in the base.

Cost on the 6-scene dataset is **zero**: `laplacian_p4_c1` matches `laplacian_p4` to
three decimals on every scene (mean 0.802 / 0.002 / 0.607). It is free there because
those scenes are tripod-shot under constant light and have no exposure disagreement -
which is why the dataset alone could never have surfaced this.

| method | recovery | halo | flat_excess |
|---|---|---|---|
| laplacian (p1, default) | 0.346 | 0.000 | 0.315 |
| focus_power=4 | 0.802 | 0.002 | 0.607 |
| focus_power=8 | 0.925 | 0.006 | 0.944 |
| helicon_focus (reference) | 0.884 | 0.004 | 1.223 |

Caveats worth carrying:
- **Default stays 1.0.** This work makes 4 safe to choose; it does not establish that a
  photographer wants it. Changing the default is a product call on output users accept.
- **A second, separate failure mode is real and unfixed.** In textureless regions the
  ranking genuinely is noise, and level gating does not help (GT MAE 1.20 at p4,
  unchanged); confidence gating does (1.09). That is the argument for defaulting any
  recommendation to 4 rather than 8, which costs 1.64 there.
- `FocusStacker.copy()` now builds the stack worker's stacker. The old field-by-field
  rebuild in `_run_stack` would have silently dropped `focus_power` - same bug class as
  the substack cache key. Its test compares `fusion_fingerprint`, not a field list.
- Laplacian memory is untouched and still the binding constraint for native 30-frame
  runs (~28 GiB projected vs 25 GiB available).

## Confidence gate: validated, NOT shipped (September 12, 2026)

`ConfidenceGatedStacker(focus_power=8, floor_pct=25, measure_blur=2)` in
`laplacian_weight_experiment.py` beats the committed `focus_power=4` on detail AND grain
at once, on all three measurement beds. Full method in
`docs/investigations/GATE_SWEEP_2026-09-12.md`.

| 6-scene dataset | recovery | halo | flat_excess |
|---|---|---|---|
| laplacian (shipped default) | 0.346 | 0.000 | 0.315 |
| focus_power=4 (committed) | 0.802 | 0.002 | 0.607 |
| **gated fp8** | **0.859** | 0.002 | **0.498** |
| helicon_focus (reference) | 0.884 | 0.004 | 1.223 |

Real stacks agree: stack 1 recovery 0.810 vs fp4's 0.788 with grain 0.793 vs 0.952;
stack 2 0.711 vs 0.686 and 0.760 vs 0.877. Promoting it needs two new parameters, both
spatial — that is the only reason it did not ship this round.

**Three approaches were tried and failed; do not re-run them.**
- Winner-vs-runner-up *margin* as the confidence statistic suppresses the exponent nearly
  everywhere (GT covered 3.08 vs p1's 3.29).
- *Pooling the focus measure without a gate* collapses detail at native resolution
  (recovery 0.886 -> 0.641 at sigma 6) even though it looked perfect on the synthetic.
- *Widening `kernel_size`* instead of adding a parameter does the same (0.886 -> 0.527 at
  ks=35), also despite looking perfect on the synthetic.

**Methodological rule that came out of this, and the most reusable thing here:**
`confidence_gate_gt.py` is 512x512. It is reliable for identifying *which mechanism* is at
play and for ranking methods with no spatial parameter. It CANNOT pick a spatial scale for
24MP images — sigma 6 spans 1.2% of its frame and 0.1% of a 6000px frame. Tune every
spatial parameter on native-resolution real stacks, then confirm on the dataset subset.

Also fixed: `benchmark_metrics.py` accepted only stems matching `laplacian_p*` and dropped
every other fusion with a bare `continue`, so a variant could go unscored while looking
like it had been scored and found unremarkable. It now prints what it ignores.

Nothing in production changed. New files: `docs/investigations/benchmark_dataset.py`,
`benchmark_metrics.py`, `benchmark_crops.py`, `depth_gt_control.py`, `blind_compare.py`,
and the report above.
