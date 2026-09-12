"""Score dataset benchmark fusions without ground truth.

The dataset has no all-in-focus reference, so nothing here compares a fusion to a
"correct" image. Instead each scene gets a per-pixel *focus envelope*: the maximum
local sharpness any aligned source frame achieves at that pixel. The envelope is the
best detail the inputs actually contain, which bounds what any fusion could recover.

  recovery     median(fused sharpness / envelope) over textured pixels.
               1.0 means the fusion kept the sharpest available detail; below 1.0 is
               the mush/detail-loss failure. Above 1.0 is not a win - it means the
               fusion invented contrast the sources never had (halos, ringing).
  flat_excess  fused high-frequency energy in regions where EVERY source is smooth,
               divided by the mean source energy there. 1.0 is faithful; above 1.0 is
               blending speckle or grain in what should be clean background.
  halo         share of textured pixels whose fused sharpness exceeds the envelope by
               more than 25%, a direct count of invented edge contrast.

Helicon Focus renders are measured on the SAME metrics purely as a competitive
reference point. They are not treated as truth and never enter another run's score.

    python docs/investigations/benchmark_metrics.py --width 2592
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DATASET = Path('test_stacks/dataset/focus_stack_dataset/focus_stack_dataset/dataset')
TEXTURE_PCT = 60      # pixels above this envelope percentile count as "textured"
FLAT_PCT = 20         # pixels below this envelope percentile count as "flat"


def sharpness(bgr):
    """Local high-frequency energy, smoothed so it is stable under 1px shifts."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    return cv2.boxFilter(lap, -1, (7, 7))


def load(path, width):
    im = cv2.imread(str(path))
    if im is None:
        raise SystemExit(f'Cannot read {path}')
    if width and im.shape[1] != width:
        im = cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])),
                        interpolation=cv2.INTER_AREA)
    return im


def score(fused_sharp, envelope, src_mean_flat, textured, flat):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(envelope > 1e-6, fused_sharp / envelope, np.nan)
    rec = float(np.nanmedian(ratio[textured]))
    halo = float(np.mean(ratio[textured] > 1.25))
    fe = float(np.mean(fused_sharp[flat]))
    excess = fe / src_mean_flat if src_mean_flat > 1e-6 else float('nan')
    return dict(recovery=rec, halo_fraction=halo, flat_excess=excess,
                flat_energy=fe, mean_sharpness=float(np.mean(fused_sharp)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('test_outputs/dataset_benchmark'))
    ap.add_argument('--dataset', type=Path, default=DATASET)
    ap.add_argument('--width', type=int, default=2592)
    ap.add_argument('--step', type=int, default=1,
                    help='envelope uses every Nth frame; must match the run being scored')
    args = ap.parse_args()

    results = []
    for dest in sorted(p for p in args.out.iterdir() if p.is_dir()):
        scene = dest.name.replace('__', '/')
        aligned = args.dataset / scene / 'aligned'
        if not aligned.is_dir():
            print(f'skip {dest.name}: no aligned/'); continue
        frames = sorted(aligned.glob('*.jpg'), key=lambda p: int(p.stem.split('_')[1]))
        frames = frames[::args.step]      # envelope must come from the frames actually fused
        if not frames:
            continue

        envelope = None
        flat_accum = None
        for f in frames:                       # streamed: one frame resident at a time
            s = sharpness(load(f, args.width))
            envelope = s if envelope is None else np.maximum(envelope, s)
            flat_accum = s.astype(np.float64) if flat_accum is None else flat_accum + s
        src_mean = flat_accum / len(frames)

        t_thr = np.percentile(envelope, TEXTURE_PCT)
        f_thr = np.percentile(envelope, FLAT_PCT)
        textured = envelope >= t_thr
        flat = envelope <= f_thr
        src_mean_flat = float(np.mean(src_mean[flat]))

        for png in sorted(dest.glob('*.png')):
            if png.name == 'helicon_reference.png':
                label = 'helicon_focus'
            elif png.stem in ('laplacian', 'complex_wavelet') or png.stem.startswith('laplacian_p'):
                label = png.stem
            else:
                continue
            im = load(png, args.width)
            if im.shape[:2] != envelope.shape:
                print(f'  {scene} {label}: shape mismatch {im.shape[:2]} vs {envelope.shape}, skipped')
                continue
            rec = score(sharpness(im), envelope, src_mean_flat, textured, flat)
            rec.update(scene=scene, method=label, frames=len(frames), width=args.width)
            results.append(rec)
            print(f'{scene:46s} {label:16s} recovery={rec["recovery"]:.3f} '
                  f'halo={rec["halo_fraction"]:.3f} flat_excess={rec["flat_excess"]:.3f}')

    out = args.out / f'metrics_w{args.width}_s{args.step}.json'
    out.write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out}')

    print('\n=== mean across scenes ===')
    for m in sorted({r['method'] for r in results}):
        rows = [r for r in results if r['method'] == m]
        if rows:
            print(f'{m:16s} recovery={np.mean([r["recovery"] for r in rows]):.3f}  '
                  f'halo={np.mean([r["halo_fraction"] for r in rows]):.3f}  '
                  f'flat_excess={np.mean([r["flat_excess"] for r in rows]):.3f}   (n={len(rows)})')


if __name__ == '__main__':
    main()
