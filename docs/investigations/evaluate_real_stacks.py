"""Prepare shared alignment and run each fusion in a separate measured process.

python docs/investigations/evaluate_real_stacks.py prepare INPUT OUTPUT --width 1500
python docs/investigations/evaluate_real_stacks.py fuse INPUT OUTPUT --algorithm laplacian
width=0 preserves native resolution. Outputs are local evaluation artifacts.
"""
import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from focal.core.align import align_image
from focal.core.stacker import FocusStacker, StackAlgorithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fuse'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--algorithm', choices=[a.value for a in StackAlgorithm], default='laplacian')
    args = parser.parse_args()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if not paths:
        parser.error('No source images')
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    started = time.perf_counter()
    if args.command == 'fuse':
        result = FocusStacker(algorithm=StackAlgorithm(args.algorithm), skip_alignment=True).stack(paths)
        elapsed = time.perf_counter() - started
        assert cv2.imwrite(str(args.output / f'{args.algorithm}.png'), result)
        stats = dict(algorithm=args.algorithm, frames=len(paths), shape=list(result.shape),
                     fusion_seconds=elapsed, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     alignment_during_fusion=False, input=str(args.input.resolve()))
        (args.output / f'{args.algorithm}.json').write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)
        return

    def read(path):
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(f'Cannot read {path}')
        if args.width and im.shape[1] > args.width:
            im = cv2.resize(im, (args.width, round(im.shape[0]*args.width/im.shape[1])), interpolation=cv2.INTER_AREA)
        return im

    reference_index = len(paths) // 2
    ref = read(paths[reference_index])
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    h, w = ref.shape[:2]
    overlap = np.ones((h, w), dtype=np.uint8)
    transforms = []
    for i, path in enumerate(paths):
        src = read(path)
        if i == reference_index:
            aligned, transform = src, np.eye(2, 3, dtype=np.float32)
        else:
            aligned, transform = align_image(ref_gray, ref, cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), src,
                                               return_transform=True)
        valid = cv2.warpAffine(np.ones(src.shape[:2], dtype=np.uint8), transform, (w,h),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        overlap &= valid
        assert cv2.imwrite(str(args.output / f'{i:02d}.png'), aligned)
        transforms.append(dict(frame=i+1, path=str(path.resolve()), transform=transform.tolist(),
                               determinant=float(np.linalg.det(transform[:, :2]))))
        print(f'aligned {i+1}/{len(paths)}', flush=True)
    # Keep mask outside the input folder so it cannot accidentally join a stack.
    cv2.imwrite(str(args.output.parent / f'{args.output.name}_overlap.png'), overlap*255)
    stats = dict(reference_frame=reference_index+1, shape=[h,w], frames=transforms,
                 seconds=time.perf_counter()-started, common_coverage_fraction=float(overlap.mean()),
                 peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                 note='Coverage only; this does not establish registration correctness. No illumination normalization.')
    (args.output.parent / f'{args.output.name}_alignment.json').write_text(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
