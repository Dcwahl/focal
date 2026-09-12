"""Drive the real Qt window offscreen and capture local screenshots.

QT_QPA_PLATFORM=offscreen python docs/investigations/gui_smoke.py INPUT OUTPUT
Add --substack to create and paint from the final two frames as a substack.
Only native file chooser responses are substituted; buttons, keys, painting,
worker execution, and saving use the application's real event handlers.
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from focal.ui.main_window import MainWindow


def main():
    inputs, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(inputs.glob('*.png'))
    assert paths
    cv2.setNumThreads(4)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1400, 850)
    window.show()
    window.activateWindow()
    QTest.qWait(100)
    with patch.object(QFileDialog, 'getOpenFileNames', return_value=([str(p.resolve()) for p in paths], '')):
        QTest.mouseClick(window.open_btn, Qt.LeftButton)
    assert len(window.images) == len(paths)
    QTest.mouseClick(window.align_checkbox, Qt.LeftButton)
    QTest.mouseClick(window.stack_btn, Qt.LeftButton)
    worker = window.worker  # Keep the QThread alive through its actual termination.
    errors = []
    worker.error.connect(errors.append)
    deadline = time.monotonic() + 120
    while window.worker is not None and time.monotonic() < deadline:
        QTest.qWait(50)
    assert worker.wait(1000)
    assert not errors, errors
    assert window.edited_result is not None
    QTest.qWait(100)
    assert window.grab().save(str(out / 'stacked.png'))

    # Select a source through the list widget and paint through viewport events.
    row = len(paths) - 1
    rect = window.image_list.visualItemRect(window.image_list.item(row))
    QTest.mouseClick(window.image_list.viewport(), Qt.LeftButton, pos=rect.center())
    assert window.current_paint_source == row
    substack = None
    if '--substack' in sys.argv[3:]:
        for index in (len(paths)-2, len(paths)-1):
            item = window.image_list.itemWidget(window.image_list.item(index))
            QTest.mouseClick(item.checkbox, Qt.LeftButton)
        assert window.create_substack_btn.isEnabled()
        QTest.mouseClick(window.create_substack_btn, Qt.LeftButton)
        substack = window.substack_list.get_selected_substack()
        assert substack is not None
        assert window.current_paint_source is substack
        assert window.grab().save(str(out / 'substack_selected.png'))
    before = window.edited_result.copy()
    QTest.mouseClick(window.brush_btn, Qt.LeftButton)
    h,w = before.shape[:2]
    point = window.result_viewer.mapFromScene(QPointF(w*.50,h*.52))
    QTest.mouseClick(window.result_viewer.viewport(), Qt.LeftButton, pos=point)
    painted = window.edited_result.copy()
    assert np.any(painted != before), 'Painting changed no pixels'
    assert len(window._undo_stack) == 1
    if substack is not None:
        # Source changes must not affect undo/redo snapshots.
        rect = window.image_list.visualItemRect(window.image_list.item(0))
        QTest.mouseClick(window.image_list.viewport(), Qt.LeftButton, pos=rect.center())
        assert window.current_paint_source == 0
    QTest.keySequence(window, QKeySequence(QKeySequence.Undo))
    np.testing.assert_array_equal(window.edited_result, before)
    QTest.keySequence(window, QKeySequence(QKeySequence.Redo))
    np.testing.assert_array_equal(window.edited_result, painted)
    if substack is not None:
        item = window.substack_list.list_widget.itemWidget(window.substack_list.list_widget.item(0))
        QTest.mouseClick(item, Qt.LeftButton)
        assert window.current_paint_source is substack
    QTest.keyPress(window, Qt.Key_S)
    assert window._flash_active
    assert window.grab().save(str(out / 'flash.png'))
    QTest.keyRelease(window, Qt.Key_S)
    assert not window._flash_active
    np.testing.assert_array_equal(window.edited_result, painted)
    with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(out / 'saved.png'), 'PNG (*.png)')):
        QTest.mouseClick(window.save_btn, Qt.LeftButton)
    np.testing.assert_array_equal(cv2.imread(str(out / 'saved.png')), painted)
    result = dict(frames=len(paths), input=str(inputs.resolve()), image_shape=list(before.shape),
                  checks=['Open', 'Align checkbox', 'Stack worker', 'Source selection',
                          'Brush mouse event', 'Undo shortcut', 'Redo shortcut', 'Flash key', 'Save roundtrip'],
                  mode='Qt offscreen / QTest events; native file dialogs substituted',
                  desktop_input_control=False)
    if substack is not None:
        result['substack_frames'] = [i+1 for i in substack.frame_indices]
        result['checks'].extend(['Substack checkboxes', 'Create substack',
                                 'Paint from substack', 'Switch source before undo', 'Reselect substack'])
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    window.close()
    app.processEvents()


if __name__ == '__main__':
    main()
