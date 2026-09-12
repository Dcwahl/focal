"""Emit unlabelled crops in randomised column order, plus a sealed answer key.

Every other check here is a number I chose how to compute. This one removes the
labels so your eye decides first and the key is read afterwards.

    python docs/investigations/blind_compare.py            # make sheets
    python docs/investigations/blind_compare.py --reveal   # print the key
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

METHODS = ['laplacian', 'laplacian_p2', 'laplacian_p4', 'complex_wavelet', 'helicon_reference']
CROP = 400


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench', type=Path, default=Path('test_outputs/dataset_benchmark'))
    ap.add_argument('--out', type=Path, default=Path('test_outputs/blind_compare'))
    ap.add_argument('--seed', type=int, default=20260912)
    ap.add_argument('--reveal', action='store_true')
    args = ap.parse_args()
    key_path = args.out / 'ANSWER_KEY.json'

    if args.reveal:
        if not key_path.exists():
            raise SystemExit('No key yet; run without --reveal first.')
        for sheet, cols in json.loads(key_path.read_text()).items():
            print(f'{sheet}:')
            for i, m in enumerate(cols):
                print(f'   {chr(65+i)} = {m}')
        return

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    key = {}
    for dest in sorted(p for p in args.bench.iterdir() if p.is_dir()):
        have = [m for m in METHODS if (dest / f'{m}.png').exists()]
        if len(have) < 3:
            continue
        imgs = {m: cv2.imread(str(dest / f'{m}.png')) for m in have}
        shape = next(iter(imgs.values())).shape[:2]
        # crop where the candidates disagree most, i.e. where the choice actually matters
        a, b = imgs[have[0]], imgs[have[-2] if len(have) > 2 else have[-1]]
        d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
        sc = cv2.boxFilter(d.astype(np.float32), -1, (CROP, CROP))
        m = CROP // 2
        sc[:m, :] = sc[-m:, :] = sc[:, :m] = sc[:, -m:] = -1
        _, _, _, (cx, cy) = cv2.minMaxLoc(sc)
        x0 = int(np.clip(cx - CROP//2, 0, shape[1]-CROP)); y0 = int(np.clip(cy - CROP//2, 0, shape[0]-CROP))

        order = have[:]; rng.shuffle(order)
        tiles = []
        for i, mth in enumerate(order):
            t = imgs[mth][y0:y0+CROP, x0:x0+CROP].copy()
            cv2.rectangle(t, (0, 0), (CROP, 26), (0, 0, 0), -1)
            cv2.putText(t, chr(65+i), (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            tiles.append(t)
        sep = np.full((CROP, 6, 3), 40, np.uint8)
        row = []
        for i, t in enumerate(tiles):
            if i: row.append(sep)
            row.append(t)
        name = f'{dest.name}.png'
        cv2.imwrite(str(args.out / name), np.hstack(row))
        key[name] = order
        print(f'{name}: {len(order)} unlabelled columns')

    key_path.write_text(json.dumps(key, indent=2))
    print(f'\nSheets in {args.out}/  — key written to {key_path}')
    print('Pick your preferred column per sheet, then: --reveal')


if __name__ == '__main__':
    main()
