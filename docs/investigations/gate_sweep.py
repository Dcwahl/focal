"""Sweep the confidence gate's floor and pooling against ground truth.

pct25 + pool was the most aggressive setting in the first pass and was still improving
at the edge of the sweep, so this asks whether the grain that survives it is fundamental
or just untuned. Scored against the known sharp original from confidence_gate_gt.py, in
the three zones that file defines - `flat` is the grain case.

    python docs/investigations/gate_sweep.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from confidence_gate_gt import FLAT_BOX, FOCUS_MAX, FRAMES, SIZE, make_sharp, render
from laplacian_weight_experiment import ConfidenceGatedStacker, WeightedStacker

NOISE = 3
FLOORS = [10, 25, 40, 55, 70]
BLURS = [0, 2, 4]
POWERS = [4.0, 8.0]


def main():
    out = Path('test_outputs/gate_sweep')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    sharp = make_sharp(rng)
    depth = np.tile(np.linspace(0, 1, SIZE), (SIZE, 1))
    y0, y1, x0, x1 = FLAT_BOX
    flat = np.zeros((SIZE, SIZE), bool)
    flat[y0:y1, x0:x1] = True
    zones = dict(covered=(depth <= FOCUS_MAX) & ~flat,
                 uncovered=(depth > FOCUS_MAX) & ~flat, flat=flat)

    folder = out / 'frames'
    folder.mkdir(exist_ok=True)
    paths = []
    for i in range(FRAMES):
        p = folder / f'{i:02d}.png'
        assert cv2.imwrite(str(p), render(sharp, depth, i/(FRAMES-1)*FOCUS_MAX, NOISE, rng))
        paths.append(p)

    def score(name, res):
        err = np.abs(res.astype(np.float64) - sharp).mean(axis=2)
        row = dict(method=name, overall=float(err.mean()))
        row.update({z: float(err[m].mean()) for z, m in zones.items()})
        print(f'{name:22s} {row["overall"]:8.2f} ' +
              ' '.join(f'{row[z]:10.2f}' for z in zones), flush=True)
        return row

    print(f'{"method":22s} {"overall":>8s} ' + ' '.join(f'{z:>10s}' for z in zones))
    results = [score('p1', WeightedStacker(1, 0).stack(paths)),
               score('p4 uniform', WeightedStacker(4, 0).stack(paths))]
    for power in POWERS:
        for pct in FLOORS:
            for blur in BLURS:
                name = f'fp{power:g}_pct{pct}_mb{blur}'
                results.append(score(name, ConfidenceGatedStacker(
                    power, floor_pct=pct, measure_blur=blur).stack(paths)))

    (out / 'gate_sweep.json').write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda r: r['overall'])
    print(f"\nlowest overall MAE: {best['method']} ({best['overall']:.2f})")
    print(f"Wrote {out}/gate_sweep.json")


if __name__ == '__main__':
    main()
