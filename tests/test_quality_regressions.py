"""Image-level regressions for the failures found in the baseline review."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from focal.core.align import align_image, compute_transform, sample_aligned_region
from focal.core.grayscale import compute_pca_weights, to_grayscale
from focal.core.stacker import FocusStacker
from focal.ui.main_window import BrushStroke, MainWindow


def stack_arrays(tmp_path, images):
    paths = []
    for i, image in enumerate(images):
        path = tmp_path / f'{i}.png'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return FocusStacker(skip_alignment=True).stack(paths)


@pytest.mark.parametrize('value', [0, 37, 128, 255])
def test_laplacian_preserves_constant_frames(tmp_path, value):
    image = np.full((256, 256, 3), value, dtype=np.uint8)
    result = stack_arrays(tmp_path, [image] * 3)
    np.testing.assert_allclose(result, image, atol=1)


def test_laplacian_preserves_smooth_background_next_to_detail(tmp_path):
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    image[80:176, 80:176] = np.random.default_rng(10).integers(
        0, 256, (96, 96, 3), dtype=np.uint8)
    result = stack_arrays(tmp_path, [image, image])
    np.testing.assert_allclose(result, image, atol=1)


def test_pca_opponent_colors_remain_finite_and_keep_contrast():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(256, dtype=np.uint8)
    image[:, :, 1] = 255 - image[:, :, 0]
    with np.errstate(all='raise'):
        weights = compute_pca_weights(image)
        gray = to_grayscale(image, weights)
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1)
    assert np.ptp(gray) > 100


@pytest.mark.parametrize('color', [(0, 0, 0), (128, 128, 128), (30, 140, 220)])
def test_pca_constant_image_is_defined(color):
    image = np.full((64, 64, 3), color, dtype=np.uint8)
    with np.errstate(all='raise'):
        weights = compute_pca_weights(image)
        gray = to_grayscale(image, weights)
    assert np.isfinite(weights).all()
    assert np.ptp(gray) == 0


def texture(size=512):
    image = np.random.default_rng(31).integers(0, 256, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (7, 7), 0)


@pytest.mark.parametrize('shift', [3, 20, 60])
def test_alignment_recovers_translation(shift):
    ref = texture()
    transform = np.float32([[1, 0, shift], [0, 1, shift / 2]])
    src = cv2.warpAffine(ref, transform, (512, 512), borderMode=cv2.BORDER_REFLECT)
    aligned, estimated = align_image(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), ref,
        cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), src, return_transform=True)
    np.testing.assert_allclose(estimated, cv2.invertAffineTransform(transform), atol=0.2)
    assert np.abs(aligned[80:-80, 80:-80].astype(float) - ref[80:-80, 80:-80]).mean() < 1


def test_failed_ecc_cannot_leak_mutated_transform(monkeypatch, caplog):
    def fail(*args):
        args[2][:] = 123
        raise cv2.error('deliberate convergence failure')
    monkeypatch.setattr(cv2, 'findTransformECC', fail)
    image = np.zeros((333, 517), dtype=np.uint8)
    initial = np.float32([[1.02, 0.03, 15], [-0.04, 0.99, -6]])
    result = compute_transform(image, image, max_resolution=128, initial_transform=initial)
    np.testing.assert_allclose(result, initial, atol=1e-5)
    assert 'keeping the initial estimate' in caplog.text


def test_fine_alignment_uses_full_resolution_seed(monkeypatch):
    initial = np.float32([[1.02, 0.03, 15], [-0.04, 0.99, -6]])
    image = np.zeros((333, 517), dtype=np.uint8)
    def unchanged(src, ref, warp, *args):
        resize = np.diag([ref.shape[1] / 517, ref.shape[0] / 333, 1])
        full = np.eye(3)
        full[:2] = initial
        np.testing.assert_allclose(warp, (resize @ full @ np.linalg.inv(resize))[:2], atol=1e-5)
        return 1.0, warp
    monkeypatch.setattr(cv2, 'findTransformECC', unchanged)
    result = compute_transform(image, image, max_resolution=128, initial_transform=initial)
    np.testing.assert_allclose(result, initial, atol=1e-5)


def brush_state(source, transform, result, size=100):
    return SimpleNamespace(
        edited_result=result.copy(), brush_size=size,
        _get_paint_source_array=lambda: source,
        _get_paint_source_transform=lambda: transform,
        _current_stroke=BrushStroke([], 0, size),
        _undo_stack=[], _redo_stack=[],
        result_viewer=SimpleNamespace(load_array=lambda *a, **kw: None),
    )


def test_aligned_brush_preserves_correctly_warped_detail_and_undo():
    yy, xx = np.indices((256, 256))
    source = np.repeat((((xx // 5 + yy // 5) % 2) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    transform = cv2.getRotationMatrix2D((128, 128), 10, 1.05).astype(np.float32)
    expected = cv2.warpAffine(source, transform, (256, 256), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
    state = brush_state(source, transform, np.zeros_like(source))
    MainWindow._apply_brush_stroke(state, 128, 128)
    # OpenCV 4.11 rounds an isolated interpolation coordinate differently for
    # a cropped warp versus a full-frame warp. Check geometric agreement by
    # mean error rather than requiring every checkerboard edge pixel to match.
    error = np.abs(state.edited_result[110:146, 110:146].astype(float)
                   - expected[110:146, 110:146])
    assert error.mean() < 0.1
    MainWindow._apply_brush_stroke(state, 138, 128)
    MainWindow._on_stroke_finished(state)
    painted = state.edited_result.copy()
    for _ in range(2):
        MainWindow._undo(state)
        assert not state.edited_result.any()
        MainWindow._redo(state)
        np.testing.assert_array_equal(state.edited_result, painted)


def test_brush_leaves_uncovered_pixels_unchanged():
    source = np.full((64, 64, 3), 220, dtype=np.uint8)
    initial = np.full_like(source, 17)
    transform = np.float32([[1, 0, 20], [0, 1, 0]])
    state = brush_state(source, transform, initial, size=50)
    MainWindow._apply_brush_stroke(state, 20, 32)
    np.testing.assert_array_equal(state.edited_result[:, :20], initial[:, :20])
    assert state.edited_result[32, 22, 0] == 220


def test_sample_region_matches_full_warp_for_subpixel_translation():
    source = texture(128)
    transform = np.float32([[1, 0, 3.5], [0, 1, -2.25]])
    full = cv2.warpAffine(source, transform, (128, 128), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)
    region, valid = sample_aligned_region(source, transform, (25, 70, 30, 80))
    np.testing.assert_array_equal(region, full[25:70, 30:80])
    assert valid.all()


def test_stack_completion_uses_worker_transforms_after_settings_change():
    transform = np.float32([[1, 0, -20], [0, 1, 10]])
    worker_stacker = FocusStacker()
    worker_stacker.last_transforms = {0: transform}
    control = SimpleNamespace(setVisible=lambda value: None, setEnabled=lambda value: None)
    state = SimpleNamespace(
        worker=SimpleNamespace(stacker=worker_stacker),
        stacker=FocusStacker(),  # UI settings were changed during the job.
        progress=control, stack_btn=control, open_btn=control,
        save_btn=control, brush_btn=control,
        _substack_transforms={}, _undo_stack=[], _redo_stack=[],
        result_viewer=SimpleNamespace(load_array=lambda *args, **kwargs: None),
    )
    MainWindow._on_stack_finished(state, np.zeros((8, 8, 3), dtype=np.uint8))
    np.testing.assert_array_equal(state._frame_transforms[0], transform)


def _two_frame_split_focus(rng):
    """Two frames of one textured scene, each sharp on the half the other blurs."""
    sharp = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    left, right = sharp.copy(), sharp.copy()
    left[:, 128:] = cv2.GaussianBlur(sharp[:, 128:], (0, 0), 4)
    right[:, :128] = cv2.GaussianBlur(sharp[:, :128], (0, 0), 4)
    return sharp, [left, right]


def _detail(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())


def stack_power(tmp_path, images, power):
    paths = []
    for i, image in enumerate(images):
        path = tmp_path / f'{i}.png'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return FocusStacker(skip_alignment=True, focus_power=power).stack(paths)


def test_focus_power_default_is_unweighted(tmp_path):
    """focus_power=1.0 must stay bit-identical to the proportional blend."""
    _, frames = _two_frame_split_focus(np.random.default_rng(3))
    np.testing.assert_array_equal(stack_power(tmp_path, frames, 1.0),
                                  stack_arrays(tmp_path, frames))


def test_focus_power_recovers_more_detail(tmp_path):
    """A higher exponent must let the sharp frame win, not average it away."""
    sharp, frames = _two_frame_split_focus(np.random.default_rng(4))
    p1 = _detail(stack_power(tmp_path, frames, 1.0))
    p4 = _detail(stack_power(tmp_path, frames, 4.0))
    assert p1 < p4 <= _detail(sharp) * 1.05


def test_focus_power_keeps_base_brightness_when_frames_disagree(tmp_path):
    """The exponent must not pick between frames that differ only in exposure.

    This is the regression a uniform exponent caused on a real stack: where every frame
    is defocused the sharpest-frame ranking is arbitrary, so committing to a winner
    stamped that frame's brightness onto the result in patches. Exempting the coarsest
    pyramid level - the base image - keeps brightness averaged at any exponent.
    """
    rng = np.random.default_rng(5)
    base = np.full((256, 256, 3), 60, np.uint8)
    base[:, :128] = rng.integers(0, 256, (256, 128, 3), dtype=np.uint8)  # detail on one side
    frames = []
    for offset in (-40, 0, 40):
        frames.append(np.clip(base.astype(np.int16) + offset, 0, 255).astype(np.uint8))
    flat = (slice(None), slice(160, 256))
    means = [stack_power(tmp_path, frames, p)[flat].mean() for p in (1.0, 4.0, 8.0)]
    assert max(means) - min(means) < 2.0, f'exponent shifted flat-region brightness: {means}'


def test_stacker_copy_carries_every_pixel_affecting_setting():
    """A copy must be indistinguishable from its source to the cache.

    The stack worker takes a copy so UI changes mid-run cannot alter its configuration.
    When that copy was built by re-listing fields, a newly added setting silently failed
    to reach it; comparing fingerprints ties the copy to the same definition of identity
    the cache uses, so a new setting cannot be missed by one and not the other.
    """
    stacker = FocusStacker(num_levels=4, kernel_size=7, consistency=1,
                           skip_alignment=True, focus_power=4.0)
    clone = stacker.copy()
    assert clone is not stacker
    assert clone.fusion_fingerprint == stacker.fusion_fingerprint
