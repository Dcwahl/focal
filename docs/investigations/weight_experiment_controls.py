"""Reproduce the controlled checks for laplacian_weight_experiment.py.

python docs/investigations/weight_experiment_controls.py OUTPUT
Synthetic blur is deliberately simplified, not a model of macro optics.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from laplacian_weight_experiment import WeightedStacker

VARIANTS = [(1, 0), (2, 0), (4, 0), (4, 1)]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(45)
    h, w = 256, 384
    sharp = cv2.GaussianBlur(rng.integers(30, 225, (h, w, 3), dtype=np.uint8), (3, 3), 0)
    sharp[160:240, 20:110] = (80, 140, 180)
    blur = cv2.GaussianBlur(sharp, (0, 0), 3)
    metrics = {}
    cases = [('clean', 3, 0, 0), ('noise', 3, 3, 0),
             ('noise_longer', 12, 3, 0), ('residual_shift', 3, 0, 1)]
    for name, count, noise, shift in cases:
        folder = out / name
        folder.mkdir(exist_ok=True)
        paths = []
        for i in range(count):
            image = blur.copy()
            region = i % 3
            image[:, region*128:(region+1)*128] = sharp[:, region*128:(region+1)*128]
            if noise:
                image = np.clip(image.astype(float) + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
            if shift:
                image = cv2.warpAffine(image, np.float32([[1, 0, (i-1)*shift], [0, 1, 0]]),
                                       (w, h), borderMode=cv2.BORDER_REFLECT)
            path = folder / f'{i:02d}.png'
            assert cv2.imwrite(str(path), image)
            paths.append(path)
        metrics[name] = {}
        for power, smoothing in VARIANTS:
            result = WeightedStacker(power, smoothing).stack(paths)
            error = np.abs(result.astype(float) - sharp)
            label = f'p{power}_s{smoothing}'
            metrics[name][label] = dict(mae=float(error.mean()),
                                       flat_region_mae=float(error[175:225, 35:95].mean()))
            assert cv2.imwrite(str(out / f'{name}_{label}.png'), result)
    for name, image in [('flat', np.full((256, 256, 3), 128, np.uint8)), ('duplicate', sharp)]:
        folder = out / name
        folder.mkdir(exist_ok=True)
        paths = []
        for i in range(3):
            path = folder / f'{i}.png'
            assert cv2.imwrite(str(path), image)
            paths.append(path)
        metrics[name] = {}
        for power, smoothing in VARIANTS:
            result = WeightedStacker(power, smoothing).stack(paths)
            maximum = int(np.abs(result.astype(int) - image).max())
            assert maximum <= 1
            metrics[name][f'p{power}_s{smoothing}'] = dict(max_error=maximum)
    (out / 'controlled_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
