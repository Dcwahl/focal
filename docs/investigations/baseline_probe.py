"""Reproduce baseline quality checks without modifying Focal's algorithms.

Run with the project installed:
    python docs/investigations/baseline_probe.py /tmp/focal-review

Optionally put Lytro pairs 01, 05, 10 in OUTPUT/lytro first.
Synthetic images are diagnostic inputs, not a realistic optics benchmark.
"""

import json
from pathlib import Path
import sys
import warnings
from importlib.metadata import version
from types import SimpleNamespace

import cv2
import numpy as np

from focal.core.align import align_image
from focal.core.grayscale import compute_pca_weights, to_grayscale
from focal.core.stacker import FocusStacker, StackAlgorithm


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/tmp/focal-review')
    root.mkdir(parents=True, exist_ok=True)
    metrics = {'versions': {name: version(name) for name in
                           ('numpy', 'opencv-python', 'numba', 'PySide6')}}

    def save(name, img):
        path = root / (name + '.png')
        assert cv2.imwrite(str(path), img)
        return path

    def mae(a, b):
        return float(np.abs(a.astype(float) - b.astype(float)).mean())

    def run(name, imgs, target=None):
        paths = [save(f'{name}_source_{i}', img) for i, img in enumerate(imgs)]
        outputs = {}
        for algo in StackAlgorithm:
            result = FocusStacker(algorithm=algo, skip_alignment=True).stack(paths)
            save(name + '_' + algo.value, result)
            outputs[algo.value] = result
            metrics[name + '_' + algo.value] = {
                'mean': float(result.mean()), 'min': int(result.min()),
                'max': int(result.max()),
            }
            if target is not None:
                metrics[name + '_' + algo.value]['mae_to_target'] = mae(result, target)
        return paths, outputs

    flat = np.full((256, 256, 3), 128, np.uint8)
    run('flat', [flat, flat], flat)

    rng = np.random.default_rng(20260907)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), (3, 3), 0)
    run('identical_texture', [texture, texture], texture)
    # Complementary sharp halves with a known target, deliberately simplified blur.
    blurred = cv2.GaussianBlur(texture, (0, 0), 4)
    a, b = texture.copy(), texture.copy()
    a[:, 128:] = blurred[:, 128:]
    b[:, :128] = blurred[:, :128]
    paths, _ = run('split_focus', [a, b], texture)
    metrics['split_focus_best_source_mae'] = min(mae(a, texture), mae(b, texture))
    c0 = FocusStacker(algorithm=StackAlgorithm.COMPLEX_WAVELET, consistency=0, skip_alignment=True).stack(paths)
    c2 = FocusStacker(algorithm=StackAlgorithm.COMPLEX_WAVELET, consistency=2, skip_alignment=True).stack(paths)
    metrics['consistency_0_vs_2_max_difference'] = int(np.abs(c0.astype(int)-c2.astype(int)).max())

    # Equal-and-opposite channel variation makes the PCA direction sum to zero.
    colored = np.zeros((256, 256, 3), dtype=np.uint8)
    colored[:, :, 0] = np.arange(256, dtype=np.uint8)
    colored[:, :, 1] = 255 - colored[:, :, 0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        weights = compute_pca_weights(colored)
        gray = to_grayscale(colored, weights)
    metrics['pca_opponent_colors'] = {
        'weights_finite': bool(np.isfinite(weights).all()),
        'gray_unique_values': np.unique(gray).tolist(),
        'warnings': [str(w.message) for w in caught],
    }

    # Exercise the GUI brush implementation without opening a window.
    from focal.ui.main_window import MainWindow
    yy, xx = np.indices((256, 256))
    checker = np.repeat((((xx // 5 + yy // 5) % 2) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    rotation = cv2.getRotationMatrix2D((128, 128), 10, 1.05).astype(np.float32)
    expected = cv2.warpAffine(checker, rotation, (256, 256), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
    brush = SimpleNamespace(edited_result=expected.copy(), brush_size=100,
                            _get_paint_source_array=lambda: checker,
                            _get_paint_source_transform=lambda: rotation)
    MainWindow._apply_brush_stroke(brush, 128, 128, record_undo=False)
    metrics['brush_rotation_scale_interior_mae'] = mae(
        brush.edited_result[110:146, 110:146], expected[110:146, 110:146])
    save('brush_expected', expected)
    save('brush_actual', brush.edited_result)

    # Evaluate the actual applied warp against a known geometric transform.
    ref = cv2.GaussianBlur(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), (7, 7), 0)
    for shift in (3, 20, 60):
        matrix = np.float32([[1, 0, shift], [0, 1, shift / 2]])
        src = cv2.warpAffine(ref, matrix, (512, 512), borderMode=cv2.BORDER_REFLECT)
        aligned, estimated = align_image(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), ref,
                                        cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), src, return_transform=True)
        metrics[f'alignment_shift_{shift}'] = {
            'expected_transform': cv2.invertAffineTransform(matrix).tolist(),
            'estimated_transform': estimated.tolist(),
            'interior_mae_before': mae(src[80:-80, 80:-80], ref[80:-80, 80:-80]),
            'interior_mae_after': mae(aligned[80:-80, 80:-80], ref[80:-80, 80:-80]),
        }

    rows = []
    for n in (1, 5, 10):
        paths = [root / 'lytro' / f'lytro-{n:02d}-{s}.jpg' for s in 'AB']
        if not all(p.exists() for p in paths):
            continue
        imgs = [cv2.imread(str(p)) for p in paths]
        tiles = imgs.copy()
        for algo in StackAlgorithm:
            result = FocusStacker(algorithm=algo, skip_alignment=True).stack(paths)
            save(f'lytro_{n:02d}_{algo.value}', result)
            tiles.append(result)
        labeled = []
        for title, tile in zip(('Source A', 'Source B', 'Laplacian', 'Wavelet'), tiles):
            tile = cv2.copyMakeBorder(tile, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
            cv2.putText(tile, f'{n:02d} {title}', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 1)
            labeled.append(tile)
        rows.append(np.hstack(labeled))
    if rows:
        save('lytro_contact_sheet', np.vstack(rows))
    (root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
