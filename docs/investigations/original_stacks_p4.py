"""Score and display p1/p2/p4 on the user's own two stacks.

These are the stacks the earlier experiment flagged: higher powers were said to make
background colour/brightness changes more pronounced. Unlike the research dataset, these
scenes have real motion (an arm enters stack 1's background) and lighting change, so
this is the regression check that decides whether p4 can ship.

    python docs/investigations/original_stacks_p4.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('test_outputs/real_stacks')
STACKS = ['focus_stack_test-1', 'focus_stack_test-2']
SOURCES = {
    'laplacian':   'native_results/laplacian.png',
    'laplacian_p2': 'weight_experiment/p2_s0.png',
    'laplacian_p4': 'weight_experiment/p4_s0.png',
    'complex_wavelet': 'native_results/complex_wavelet.png',
}
LABELS = {'laplacian': 'p1 (shipped)', 'laplacian_p2': 'p2',
          'laplacian_p4': 'p4', 'complex_wavelet': 'Complex wavelet'}


def sharp_lap(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), -1, (7, 7))


def sharp_sobel(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.boxFilter(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3),
                                       cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)), -1, (7, 7))


def annotate(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    report = []
    for stack in STACKS:
        base = ROOT / stack
        aligned = sorted((base / 'native_aligned').glob('*.png'))
        imgs = {k: cv2.imread(str(base / v)) for k, v in SOURCES.items() if (base / v).exists()}
        keys = [k for k in SOURCES if k in imgs]
        print(f'\n=== {stack} ({len(aligned)} frames, {imgs[keys[0]].shape[1]}x{imgs[keys[0]].shape[0]}) ===')

        for fn, mname in ((sharp_lap, 'laplacian_metric'), (sharp_sobel, 'sobel_metric')):
            env = None
            acc = None
            for f in aligned:
                s = fn(cv2.imread(str(f)))
                env = s if env is None else np.maximum(env, s)
                acc = s.astype(np.float64) if acc is None else acc + s
            srcmean = acc / len(aligned)
            tex = env >= np.percentile(env, 60)
            flat = env <= np.percentile(env, 20)
            smf = float(srcmean[flat].mean())
            print(f'  -- {mname} --')
            for k in keys:
                s = fn(imgs[k])
                with np.errstate(invalid='ignore', divide='ignore'):
                    r = np.where(env > 1e-6, s / env, np.nan)
                rec = float(np.nanmedian(r[tex])); halo = float(np.mean(r[tex] > 1.25))
                fe = float(s[flat].mean() / smf)
                print(f'     {LABELS[k]:18s} recovery={rec:.3f} halo={halo:.3f} flat_excess={fe:.3f}')
                report.append(dict(stack=stack, metric=mname, method=k,
                                   recovery=rec, halo=halo, flat_excess=fe))

        # crops at the frozen diagnostic regions, background region included
        regions = json.loads((base / 'crops.json').read_text())
        outdir = base / 'p4_comparison'
        outdir.mkdir(exist_ok=True)
        for reg in regions:
            y0, y1, x0, x1 = reg['bounds']
            tiles = [annotate(imgs[k][y0:y1, x0:x1], LABELS[k]) for k in keys]
            sep = np.full((tiles[0].shape[0], 6, 3), 40, np.uint8)
            row = []
            for i, t in enumerate(tiles):
                if i:
                    row.append(sep)
                row.append(t)
            cv2.imwrite(str(outdir / f"crop_{reg['name']}.png"), np.hstack(row))

            # background drift: how uniform is a region that should be smooth?
            if reg['name'] in ('background', 'left_edge'):
                print(f"  -- {reg['name']} region uniformity (lower = less blotchy) --")
                for k in keys:
                    patch = imgs[k][y0:y1, x0:x1].astype(np.float32)
                    lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
                    coarse = cv2.GaussianBlur(lab, (0, 0), 12)   # large-scale colour drift only
                    drift = float(np.sqrt(((coarse - coarse.mean(axis=(0, 1)))**2).sum(axis=2)).mean())
                    print(f'     {LABELS[k]:18s} lab_drift={drift:6.2f}  luma_std={patch.mean(axis=2).std():6.2f}')
                    report.append(dict(stack=stack, metric=f"{reg['name']}_uniformity",
                                       method=k, lab_drift=drift))
        print(f'  crops -> {outdir}')

    Path(ROOT / 'p4_original_stacks.json').write_text(json.dumps(report, indent=2))
    print(f'\nWrote {ROOT}/p4_original_stacks.json')


if __name__ == '__main__':
    main()
