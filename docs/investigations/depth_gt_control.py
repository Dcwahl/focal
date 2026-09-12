"""Ground-truth check: does the exponent needed by Laplacian weighting grow with stack depth?

Unlike the dataset benchmark, this has a real reference image. A synthetic scene with a
left-to-right depth ramp is rendered N times; frame i is sharp at depth i/(N-1) and
progressively blurred away from it, which is what a real focus stack looks like. Fusions
are scored by mean absolute error against the known sharp original, so no focus-envelope
proxy is involved and no metric shares the stacker's Laplacian basis.

    python docs/investigations/depth_gt_control.py OUTPUT
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from laplacian_weight_experiment import WeightedStacker

SIZE = 512
MAX_SIGMA = 6.0
LEVELS = np.linspace(0, MAX_SIGMA, 25)
COUNTS = [5, 10, 20, 30]
POWERS = [1, 2, 4, 8]


def make_sharp(rng):
    base = rng.integers(40, 215, (SIZE, SIZE, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(base, (3, 3), 0)          # fine texture everywhere
    for _ in range(18):                               # plus structure at varied scales
        x, y = rng.integers(0, SIZE, 2)
        r = int(rng.integers(12, 60))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(x), int(y)), r, c, -1)
        else:
            cv2.rectangle(img, (int(x), int(y)), (int(x)+r, int(y)+r), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def render(sharp, depth, focus, noise, rng):
    """Blur each pixel by sigma proportional to its distance from the focus plane."""
    sigma = np.abs(depth - focus) * MAX_SIGMA
    stack = [cv2.GaussianBlur(sharp, (0, 0), s) if s > 0.05 else sharp.copy() for s in LEVELS]
    idx = np.clip(np.searchsorted(LEVELS, sigma) , 1, len(LEVELS)-1)
    lo, hi = LEVELS[idx-1], LEVELS[idx]
    t = np.where(hi > lo, (sigma - lo) / np.maximum(hi - lo, 1e-6), 0.0)[..., None]
    arr = np.stack(stack)                              # (L,H,W,3)
    gi = np.ogrid[:SIZE, :SIZE]
    out = arr[idx-1, gi[0], gi[1]] * (1-t) + arr[idx, gi[0], gi[1]] * t
    if noise:
        out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'test_outputs/depth_gt_control')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sharp = make_sharp(rng)
    cv2.imwrite(str(out / 'ground_truth.png'), sharp)
    depth = np.tile(np.linspace(0, 1, SIZE), (SIZE, 1))

    results = []
    for noise in (0, 3):
        for n in COUNTS:
            folder = out / f'n{n}_noise{noise}'
            folder.mkdir(exist_ok=True)
            paths = []
            for i in range(n):
                im = render(sharp, depth, i/(n-1), noise, rng)
                p = folder / f'{i:02d}.png'
                assert cv2.imwrite(str(p), im)
                paths.append(p)
            # floor: the best any single input frame achieves
            best = min(float(np.abs(cv2.imread(str(p)).astype(float)-sharp).mean()) for p in paths)
            row = dict(frames=n, noise=noise, best_single_frame=best)
            for power in POWERS:
                res = WeightedStacker(power, 0).stack(paths)
                cv2.imwrite(str(folder / f'p{power}.png'), res)
                row[f'p{power}'] = float(np.abs(res.astype(float) - sharp).mean())
            results.append(row)
            cells = '  '.join(f'p{p}={row[f"p{p}"]:6.2f}' for p in POWERS)
            print(f'noise={noise}  n={n:2d}  best_single={best:6.2f}   {cells}', flush=True)

    (out / 'depth_gt_metrics.json').write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out}/depth_gt_metrics.json  (MAE vs ground truth, lower is better)')


if __name__ == '__main__':
    main()
