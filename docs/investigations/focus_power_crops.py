"""Side-by-side crops of the shipped exponent at the frozen diagnostic regions.

    python docs/investigations/focus_power_crops.py

Writes one PNG per region to test_outputs/focus_power_crops/.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('test_outputs/real_stacks')
OUT = Path('test_outputs/focus_power_crops')
COLS = [('focus_power=1 (default)', 'native_results/laplacian.png'),
        ('focus_power=4', 'weight_experiment/prod_p4.png'),
        ('focus_power=8', 'weight_experiment/prod_p8.png')]
TILE = 620


def label(img, text, warn=False):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    colour = (90, 160, 255) if warn else (255, 255, 255)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for stack in ['focus_stack_test-1', 'focus_stack_test-2']:
        base = ROOT / stack
        imgs = {lbl: cv2.imread(str(base / rel)) for lbl, rel in COLS}
        for reg in json.loads((base / 'crops.json').read_text()):
            y0, y1, x0, x1 = reg['bounds']
            tiles = []
            for lbl, _ in COLS:
                t = imgs[lbl][y0:y1, x0:x1]
                t = cv2.resize(t, (TILE, round(t.shape[0] * TILE / t.shape[1])),
                               interpolation=cv2.INTER_CUBIC)
                tiles.append(label(t, f"{reg['name']} - {lbl}"))
            sep = np.full((tiles[0].shape[0], 4, 3), 40, np.uint8)
            row = [tiles[0]]
            for t in tiles[1:]:
                row += [sep, t]
            path = OUT / f"{stack.replace('focus_stack_', '')}_{reg['name']}.png"
            assert cv2.imwrite(str(path), np.hstack(row))
            written.append(path)
    for p in written:
        print(p)


if __name__ == '__main__':
    main()
