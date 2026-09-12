"""Build side-by-side contact sheets and detail crops for benchmark fusions.

Numbers say where to look; these say what is actually wrong. For each scene a full
overview strip and several 400x400 native-scale crops are written, columns ordered
laplacian | complex_wavelet | helicon_focus.

Crop locations are chosen automatically as the regions of highest *disagreement*
between the two Focal algorithms, which is where a blending defect in one of them
tends to live, plus the single most textured region. Helicon is shown for reference
only; it is not truth.

    python docs/investigations/benchmark_crops.py --width 2592
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLUMNS = ['laplacian', 'laplacian_p2', 'laplacian_p4', 'complex_wavelet', 'helicon_reference']
LABELS = {'laplacian': 'Laplacian (p1, shipped)', 'laplacian_p2': 'Laplacian p2',
          'laplacian_p4': 'Laplacian p4', 'complex_wavelet': 'Complex wavelet',
          'helicon_reference': 'Helicon (reference)'}
CROP = 400


def annotate(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def row(images, labels):
    h = min(i.shape[0] for i in images)
    tiles = []
    for im, lb in zip(images, labels):
        if im.shape[0] != h:
            im = cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h))
        tiles.append(annotate(im, lb))
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
             for t in tiles]
    sep = np.full((h, 6, 3), 40, np.uint8)
    out = []
    for i, t in enumerate(tiles):
        if i:
            out.append(sep)
        out.append(t)
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('test_outputs/dataset_benchmark'))
    ap.add_argument('--width', type=int, default=2592)
    ap.add_argument('--crops', type=int, default=3)
    args = ap.parse_args()

    index = {}
    for dest in sorted(p for p in args.out.iterdir() if p.is_dir()):
        present = {c: dest / f'{c}.png' for c in COLUMNS if (dest / f'{c}.png').exists()}
        if len(present) < 2:
            continue
        imgs = {k: cv2.imread(str(v)) for k, v in present.items()}
        shape = next(iter(imgs.values())).shape[:2]
        imgs = {k: (v if v.shape[:2] == shape else cv2.resize(v, (shape[1], shape[0])))
                for k, v in imgs.items()}
        keys = [c for c in COLUMNS if c in imgs]

        overview = row([cv2.resize(imgs[k], (620, round(shape[0] * 620 / shape[1]))) for k in keys],
                       [LABELS[k] for k in keys])
        cv2.imwrite(str(dest / 'overview.jpg'), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Disagreement between the two Focal algorithms, coarse-binned to find regions.
        picks = []
        if 'laplacian' in imgs and 'complex_wavelet' in imgs:
            d = cv2.absdiff(cv2.cvtColor(imgs['laplacian'], cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(imgs['complex_wavelet'], cv2.COLOR_BGR2GRAY)).astype(np.float32)
            score = cv2.boxFilter(d, -1, (CROP, CROP))
            work = score.copy()
            m = CROP // 2
            work[:m, :] = work[-m:, :] = work[:, :m] = work[:, -m:] = -1
            for _ in range(args.crops):
                _, mx, _, loc = cv2.minMaxLoc(work)
                if mx <= 0:
                    break
                picks.append(('disagree', loc))
                cv2.circle(work, loc, CROP, -1, -1)

        g = cv2.cvtColor(imgs[keys[0]], cv2.COLOR_BGR2GRAY).astype(np.float32)
        tex = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (CROP, CROP))
        m = CROP // 2
        tex[:m, :] = tex[-m:, :] = tex[:, :m] = tex[:, -m:] = -1
        _, _, _, loc = cv2.minMaxLoc(tex)
        picks.append(('texture', loc))

        recorded = []
        for i, (kind, (cx, cy)) in enumerate(picks):
            x0 = int(np.clip(cx - CROP // 2, 0, shape[1] - CROP))
            y0 = int(np.clip(cy - CROP // 2, 0, shape[0] - CROP))
            tiles = [imgs[k][y0:y0 + CROP, x0:x0 + CROP] for k in keys]
            sheet = row(tiles, [LABELS[k] for k in keys])
            name = f'crop_{i}_{kind}.png'
            cv2.imwrite(str(dest / name), sheet)
            recorded.append(dict(name=name, kind=kind, x=x0, y=y0, size=CROP))
        (dest / 'crops.json').write_text(json.dumps(recorded, indent=2))
        index[dest.name] = recorded
        print(f'{dest.name}: overview.jpg + {len(recorded)} crops')

    (args.out / 'crops_index.json').write_text(json.dumps(index, indent=2))
    print(f'\nWrote {args.out}/crops_index.json')


if __name__ == '__main__':
    main()
