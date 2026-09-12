"""Ground-truth control built to contain the failure a fixed exponent has, not just the case it wins.

depth_gt_control.py sweeps focus across the whole depth range, so every pixel is sharp in
some frame and a high exponent is always right. The real stacks are not like that: part of
the frame is defocused in every shot, and there the sharpest-frame ranking is noise. This
scene has three zones scored separately against a known sharp original:

  covered    depth inside the focus sweep, textured    - the case a high exponent should win
  uncovered  depth past the sweep, never sharp         - ranking degrades toward noise
  flat       constant-colour patch inside covered depth - ranking is pure noise

    python docs/investigations/confidence_gate_gt.py [OUTPUT]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from laplacian_weight_experiment import WeightedStacker, GatedStacker, LevelGatedStacker

SIZE = 512
MAX_SIGMA = 6.0
FOCUS_MAX = 0.65          # focus sweeps only this far; columns beyond are never sharp
LEVELS = np.linspace(0, MAX_SIGMA, 25)
FLAT_BOX = (200, 340, 40, 180)   # y0, y1, x0, x1 - inside covered depth
FRAMES = 30


def make_sharp(rng):
    base = rng.integers(40, 215, (SIZE, SIZE, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(base, (3, 3), 0)
    for _ in range(18):
        x, y = rng.integers(0, SIZE, 2)
        r = int(rng.integers(12, 60))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(x), int(y)), r, c, -1)
        else:
            cv2.rectangle(img, (int(x), int(y)), (int(x)+r, int(y)+r), c, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    y0, y1, x0, x1 = FLAT_BOX
    img[y0:y1, x0:x1] = (128, 128, 128)   # no texture at any blur: ranking here is noise
    return img


def render(sharp, depth, focus, noise, rng):
    sigma = np.abs(depth - focus) * MAX_SIGMA
    stack = [cv2.GaussianBlur(sharp, (0, 0), s) if s > 0.05 else sharp.copy() for s in LEVELS]
    idx = np.clip(np.searchsorted(LEVELS, sigma), 1, len(LEVELS)-1)
    lo, hi = LEVELS[idx-1], LEVELS[idx]
    t = np.where(hi > lo, (sigma - lo) / np.maximum(hi - lo, 1e-6), 0.0)[..., None]
    arr = np.stack(stack)
    gi = np.ogrid[:SIZE, :SIZE]
    out = arr[idx-1, gi[0], gi[1]] * (1-t) + arr[idx, gi[0], gi[1]] * t
    if noise:
        out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def methods():
    yield 'p1', lambda: WeightedStacker(1, 0)
    yield 'p2', lambda: WeightedStacker(2, 0)
    yield 'p4', lambda: WeightedStacker(4, 0)
    yield 'p8', lambda: WeightedStacker(8, 0)
    for pct in (5, 10, 25):
        yield f'gate4_pct{pct}', (lambda p=pct: GatedStacker(4, p, 1.0))
    for blur in (2, 4, 8):
        yield f'gate4_pct10_b{blur}', (lambda b=blur: GatedStacker(4, 10, 1.0, b))
    for scale in (0.5, 2.0):
        yield f'gate4_pct10_s{scale:g}_b4', (lambda sc=scale: GatedStacker(4, 10, sc, 4))
    yield 'gate8_pct10_b4', lambda: GatedStacker(8, 10, 1.0, 4)
    for pw in (4, 8, 16):
        yield f'lvl{pw}_c1', (lambda w=pw: LevelGatedStacker(w, 1))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'test_outputs/confidence_gate_gt')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sharp = make_sharp(rng)
    cv2.imwrite(str(out / 'ground_truth.png'), sharp)

    depth = np.tile(np.linspace(0, 1, SIZE), (SIZE, 1))
    y0, y1, x0, x1 = FLAT_BOX
    flat = np.zeros((SIZE, SIZE), bool)
    flat[y0:y1, x0:x1] = True
    covered = (depth <= FOCUS_MAX) & ~flat
    uncovered = (depth > FOCUS_MAX) & ~flat
    zones = dict(covered=covered, uncovered=uncovered, flat=flat)

    results = []
    for noise in (0, 3):
        folder = out / f'noise{noise}'
        folder.mkdir(exist_ok=True)
        paths = []
        for i in range(FRAMES):
            p = folder / f'{i:02d}.png'
            assert cv2.imwrite(str(p), render(sharp, depth, i/(FRAMES-1)*FOCUS_MAX, noise, rng))
            paths.append(p)
        print(f'\n=== noise={noise}, {FRAMES} frames, focus sweep 0..{FOCUS_MAX} ===')
        print(f'{"method":14s} {"overall":>8s} ' + ' '.join(f'{z:>10s}' for z in zones))
        for name, build in methods():
            res = build().stack(paths)
            cv2.imwrite(str(folder / f'{name}.png'), res)
            err = np.abs(res.astype(np.float64) - sharp).mean(axis=2)
            row = dict(noise=noise, method=name, overall=float(err.mean()))
            row.update({z: float(err[m].mean()) for z, m in zones.items()})
            results.append(row)
            print(f'{name:14s} {row["overall"]:8.2f} ' +
                  ' '.join(f'{row[z]:10.2f}' for z in zones), flush=True)

    (out / 'confidence_gate_gt.json').write_text(json.dumps(results, indent=2))
    print(f'\nWrote {out}/confidence_gate_gt.json  (MAE vs ground truth, lower is better)')


if __name__ == '__main__':
    main()
