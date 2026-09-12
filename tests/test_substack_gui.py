"""Exercise substack retouching through the real Qt window and input events.

Uses small generated focus sequences, not downloaded datasets. File chooser
responses are substituted; fusion, alignment, cache, controls and painting are real.
"""
import os
import sys
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from focal.core.stacker import FocusStacker, StackAlgorithm
from focal.ui.main_window import MainWindow
from focal.ui.substack_list import Substack


@pytest.fixture(scope='module')
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def window(app, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, 'excepthook', lambda *error: errors.append(error))
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args))
    widget = MainWindow()
    widget.resize(1200, 750)
    widget.show()
    widget.activateWindow()
    QTest.qWait(20)
    yield widget
    if widget.worker is not None:
        widget.worker.wait(10000)
    app.removeEventFilter(widget)
    widget.close()
    widget.deleteLater()
    app.processEvents()
    assert not errors, errors


@pytest.fixture
def frames(tmp_path):
    rng = np.random.default_rng(84)
    sharp = cv2.GaussianBlur(rng.integers(20, 235, (192, 256, 3), dtype=np.uint8), (3, 3), 0)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    paths = []
    for i, (left, right) in enumerate(((0, 86), (86, 172), (172, 256))):
        image = blurred.copy()
        image[:, left:right] = sharp[:, left:right]
        # Different reference frames require nonidentity substack registration.
        image = cv2.warpAffine(image, np.float32([[1, 0, i*2], [0, 1, i]]), (256, 192),
                               borderMode=cv2.BORDER_REFLECT)
        path = tmp_path / f'frame_{i}.png'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def open_frames(window, frames, monkeypatch):
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames', lambda *a, **kw: ([str(p) for p in frames], ''))
    QTest.mouseClick(window.open_btn, Qt.LeftButton)
    assert window.images == frames


def stack(window):
    QTest.mouseClick(window.stack_btn, Qt.LeftButton)
    worker = window.worker  # Retain until the actual QThread exits.
    assert worker is not None
    deadline = time.monotonic() + 15
    while window.worker is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert worker.wait(1000), 'Stack worker timed out'
    assert window.worker is None
    assert window.edited_result is not None


def create_substack(window):
    assert not window.create_substack_btn.isEnabled()
    # Choose frames 2 and 3; their reference differs from the main reference.
    for row in (1, 2):
        item = window.image_list.itemWidget(window.image_list.item(row))
        QTest.mouseClick(item.checkbox, Qt.LeftButton)
        assert item.checkbox.isChecked()
        assert window.create_substack_btn.isEnabled() == (row == 2)
    QTest.mouseClick(window.create_substack_btn, Qt.LeftButton)
    substack = window.substack_list.get_selected_substack()
    assert isinstance(substack, Substack)
    assert substack.frame_indices == (1, 2)
    assert window.current_paint_source is substack
    assert not window.image_list.get_selected_indices()
    assert not window.create_substack_btn.isEnabled()
    assert not window.substack_progress.isVisible()
    return substack


def viewer_pixels(viewer):
    image = viewer._current_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    rgb = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine())
    return rgb[:, :image.width()*3].reshape(image.height(), image.width(), 3)[:, :, ::-1].copy()


def select_frame(window, row):
    rect = window.image_list.visualItemRect(window.image_list.item(row))
    QTest.mouseClick(window.image_list.viewport(), Qt.LeftButton, pos=rect.center())
    assert window.current_paint_source == row
    assert window.substack_list.get_selected_substack() is None


def select_substack(window, substack):
    widget = window.substack_list.list_widget.itemWidget(window.substack_list.list_widget.item(0))
    QTest.mouseClick(widget, Qt.LeftButton)
    assert window.current_paint_source is substack


@pytest.mark.parametrize('algorithm', list(StackAlgorithm))
@pytest.mark.parametrize('alignment', [False, True])
def test_substack_preview_paint_switch_undo_save(window, frames, monkeypatch, tmp_path, algorithm, alignment):
    open_frames(window, frames, monkeypatch)
    window.algo_combo.setCurrentIndex(window.algo_combo.findData(algorithm))
    if window.align_checkbox.isChecked() != alignment:
        QTest.mouseClick(window.align_checkbox, Qt.LeftButton)
    stack(window)
    before = window.edited_result.copy()
    main_transforms = {i: t.copy() for i, t in window._frame_transforms.items()}
    substack = create_substack(window)
    fused = window._get_substack_array(substack).copy()
    assert np.any(fused != cv2.imread(str(frames[1])))
    np.testing.assert_array_equal(viewer_pixels(window.source_viewer), fused)
    aligned_source = cv2.warpAffine(fused, window._get_paint_source_transform(), (256, 192),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    for i, transform in main_transforms.items():
        np.testing.assert_array_equal(window._frame_transforms[i], transform)

    QTest.mouseClick(window.brush_btn, Qt.LeftButton)
    # A real drag with overlapping dabs, not just a direct handler call.
    points = [window.result_viewer.mapFromScene(QPointF(x, 96)) for x in (125, 129, 133)]
    viewport = window.result_viewer.viewport()
    QTest.mousePress(viewport, Qt.LeftButton, pos=points[0])
    for point in points[1:]:
        QTest.mouseMove(viewport, point, delay=5)
    QTest.mouseRelease(viewport, Qt.LeftButton, pos=points[-1])
    painted = window.edited_result.copy()
    assert np.any(painted != before)
    assert len(window._undo_stack) == 1
    assert len(window._undo_stack[0].points) >= 2
    # Central pixels lie in the fully opaque interior of all overlapping dabs.
    np.testing.assert_allclose(painted[94:98, 127:131], aligned_source[94:98, 127:131], atol=1)

    select_frame(window, 0)
    np.testing.assert_array_equal(viewer_pixels(window.source_viewer), cv2.imread(str(frames[0])))
    # Undo/redo must restore snapshots, independent of the newly selected source.
    for _ in range(2):
        QTest.keySequence(window, QKeySequence(QKeySequence.Undo))
        np.testing.assert_array_equal(window.edited_result, before)
        QTest.keySequence(window, QKeySequence(QKeySequence.Redo))
        np.testing.assert_array_equal(window.edited_result, painted)

    window._cache.invalidate(window._substack_key(substack))
    select_substack(window, substack)
    np.testing.assert_array_equal(viewer_pixels(window.source_viewer), fused)
    np.testing.assert_array_equal(window._get_paint_source_array(), fused)
    QTest.keyPress(window, Qt.Key_S)
    assert window._flash_active
    np.testing.assert_array_equal(viewer_pixels(window.result_viewer)[16:-16, 16:-16],
                                  aligned_source[16:-16, 16:-16])
    QTest.keyRelease(window, Qt.Key_S)
    assert not window._flash_active
    np.testing.assert_array_equal(window.edited_result, painted)
    saved = tmp_path / 'retouched.png'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(saved), 'PNG (*.png)'))
    QTest.mouseClick(window.save_btn, Qt.LeftButton)
    np.testing.assert_array_equal(cv2.imread(str(saved)), painted)


def test_deleting_selected_substack_stops_using_it(window, frames, monkeypatch):
    open_frames(window, frames, monkeypatch)
    stack(window)
    create_substack(window)
    item = window.substack_list.list_widget.itemWidget(window.substack_list.list_widget.item(0))
    QTest.mouseClick(item.delete_btn, Qt.LeftButton)
    assert not window.substack_list.substacks
    assert not isinstance(window.current_paint_source, Substack)


def test_restack_preserves_alignment_of_existing_substack(window, frames, monkeypatch):
    open_frames(window, frames, monkeypatch)
    stack(window)
    substack = create_substack(window)
    assert window._get_paint_source_transform() is not None
    QTest.mouseClick(window.align_checkbox, Qt.LeftButton)
    stack(window)
    select_substack(window, substack)
    assert window._get_paint_source_transform() is not None


def test_algorithm_change_does_not_serve_stale_substack(window, frames, monkeypatch):
    """Changing the algorithm must refuse the substack pixels fused by the old one.

    The substack cache was previously keyed on frame indices alone, so switching the
    algorithm combo kept serving the previous algorithm's pixels for painting.
    """
    open_frames(window, frames, monkeypatch)
    window.algo_combo.setCurrentIndex(window.algo_combo.findData(StackAlgorithm.LAPLACIAN))
    stack(window)
    substack = create_substack(window)
    QTest.qWait(20)
    laplacian_pixels = window._get_substack_array(substack)
    assert laplacian_pixels is not None
    laplacian_pixels = laplacian_pixels.copy()

    window.algo_combo.setCurrentIndex(window.algo_combo.findData(StackAlgorithm.COMPLEX_WAVELET))
    QTest.qWait(20)
    served = window._get_substack_array(substack)
    assert served is not None

    paths = [frames[i] for i in substack.frame_indices]
    expected = FocusStacker(algorithm=StackAlgorithm.COMPLEX_WAVELET,
                            skip_alignment=not window.align_checkbox.isChecked()).stack(paths)
    np.testing.assert_allclose(served.astype(float), expected.astype(float), atol=2)
    assert not np.array_equal(served, laplacian_pixels), 'served the old algorithm\'s pixels'

    # Switching back must return the original pixels, not recompute something new.
    window.algo_combo.setCurrentIndex(window.algo_combo.findData(StackAlgorithm.LAPLACIAN))
    QTest.qWait(20)
    np.testing.assert_array_equal(window._get_substack_array(substack), laplacian_pixels)


def test_alignment_change_does_not_serve_stale_substack(window, frames, monkeypatch):
    """The alignment checkbox changes fused pixels too, so it must split the cache."""
    open_frames(window, frames, monkeypatch)
    window.algo_combo.setCurrentIndex(window.algo_combo.findData(StackAlgorithm.COMPLEX_WAVELET))
    window.align_checkbox.setChecked(True)
    stack(window)
    substack = create_substack(window)
    QTest.qWait(20)
    aligned_pixels = window._get_substack_array(substack).copy()

    window.align_checkbox.setChecked(False)
    QTest.qWait(20)
    unaligned = window._get_substack_array(substack)
    paths = [frames[i] for i in substack.frame_indices]
    expected = FocusStacker(algorithm=StackAlgorithm.COMPLEX_WAVELET,
                            skip_alignment=True).stack(paths)
    np.testing.assert_allclose(unaligned.astype(float), expected.astype(float), atol=2)
    assert not np.array_equal(unaligned, aligned_pixels)
