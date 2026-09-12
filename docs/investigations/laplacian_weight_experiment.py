"""Experimental focus weighting; does not change application defaults.

python docs/investigations/laplacian_weight_experiment.py INPUT OUTPUT --power 4
INPUT must contain only one sequence, already aligned when needed.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from focal.core.stacker import FocusStacker


class WeightedStacker(FocusStacker):
    def __init__(self, power=1, smoothing=0):
        super().__init__(skip_alignment=True)
        self.power = power
        self.smoothing = smoothing

    def _compute_weights(self, focus_measures):
        if self.power == 1 and self.smoothing == 0:
            return super()._compute_weights(focus_measures)
        measures = np.stack(focus_measures)
        maximum = measures.max(axis=0, keepdims=True)
        np.divide(measures, maximum, out=measures, where=maximum > 0)
        np.power(measures, self.power, out=measures)
        total = measures.sum(axis=0, keepdims=True)
        weights = np.full_like(measures, 1 / len(measures))
        np.divide(measures, total, out=weights, where=total > 0)
        if self.smoothing:
            for i in range(len(weights)):
                weights[i] = cv2.GaussianBlur(weights[i], (0, 0), self.smoothing)
            weights /= weights.sum(axis=0, keepdims=True)
        return list(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--power', type=float, default=1)
    parser.add_argument('--smoothing', type=float, default=0)
    args = parser.parse_args()
    if args.power < 1 or args.smoothing < 0:
        parser.error('power must be >= 1 and smoothing >= 0')
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    start = time.perf_counter()
    result = WeightedStacker(args.power, args.smoothing).stack(paths)
    elapsed = time.perf_counter() - start
    name = f'p{args.power:g}_s{args.smoothing:g}'
    assert cv2.imwrite(str(args.output / f'{name}.png'), result)
    metrics = dict(power=args.power, smoothing=args.smoothing, input=str(args.input.resolve()),
                   frames=len(paths), shape=list(result.shape), seconds=elapsed,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (args.output / f'{name}.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
