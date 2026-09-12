"""Run the confidence-gated exponent over the frozen 6-scene subset.

The gate's two spatial parameters were tuned on the user's 24MP stacks after the 512px
synthetic control picked values that did not transfer. This is the independent check that
they do not overfit those two sequences either. Writes into the same scene directories
the benchmark uses, so benchmark_metrics.py picks the result up by glob.

    python docs/investigations/gate_dataset.py
"""
import shutil
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_dataset import DATASET, DEFAULT_SCENES, stage
from laplacian_weight_experiment import ConfidenceGatedStacker

WIDTH = 2592
OUT = Path('test_outputs/dataset_benchmark')
VARIANTS = {'laplacian_gate8': dict(focus_power=8.0, floor_pct=25, measure_blur=2),
            'laplacian_gate4': dict(focus_power=4.0, floor_pct=25, measure_blur=2)}


def main():
    cv2.setNumThreads(4)
    for scene in DEFAULT_SCENES:
        tag = scene.replace('/', '__')
        dest = OUT / tag
        work = dest / 'inputs'
        src_dir, n = stage(DATASET / scene, work, WIDTH, 1)
        for name, kw in VARIANTS.items():
            t = time.perf_counter()
            res = ConfidenceGatedStacker(**kw).stack(sorted(src_dir.glob('*.png')))
            assert cv2.imwrite(str(dest / f'{name}.png'), res)
            print(f'[{tag}] {name}: {n} frames, {time.perf_counter()-t:.1f}s', flush=True)
        if work.exists():
            shutil.rmtree(work)


if __name__ == '__main__':
    main()
