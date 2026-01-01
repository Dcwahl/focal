"""Main application window."""

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QFileDialog, QSplitter, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import numpy as np

from focal.ui.image_list import ImageList
from focal.ui.image_viewer import ImageViewer
from focal.core.stacker import FocusStacker


class StackWorker(QThread):
    """Worker thread for focus stacking."""
    progress = Signal(int)
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, stacker: FocusStacker, image_paths: list[Path]):
        super().__init__()
        self.stacker = stacker
        self.image_paths = image_paths

    def run(self):
        try:
            result = self.stacker.stack(
                self.image_paths,
                progress_callback=self.progress.emit
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Focal")
        self.setMinimumSize(900, 600)

        self.images: list[Path] = []
        self.result_image: np.ndarray | None = None
        self.stacker = FocusStacker()
        self.worker: StackWorker | None = None

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top bar
        top_bar = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._open_folder)
        top_bar.addWidget(self.open_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Main content: viewer + image list
        splitter = QSplitter(Qt.Horizontal)

        self.viewer = ImageViewer()
        splitter.addWidget(self.viewer)

        self.image_list = ImageList()
        self.image_list.image_selected.connect(self._on_image_selected)
        splitter.addWidget(self.image_list)

        splitter.setSizes([700, 200])
        layout.addWidget(splitter, stretch=1)

        # Bottom bar
        bottom_bar = QHBoxLayout()

        self.stack_btn = QPushButton("Stack")
        self.stack_btn.clicked.connect(self._run_stack)
        self.stack_btn.setEnabled(False)
        bottom_bar.addWidget(self.stack_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        bottom_bar.addWidget(self.progress, stretch=1)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save_result)
        self.save_btn.setEnabled(False)
        bottom_bar.addWidget(self.save_btn)

        layout.addLayout(bottom_bar)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self._load_images(Path(folder))

    def _load_images(self, folder: Path):
        extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        self.images = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in extensions
        ])

        self.image_list.set_images(self.images)
        self.stack_btn.setEnabled(len(self.images) > 1)
        self.result_image = None
        self.save_btn.setEnabled(False)

        if self.images:
            self.viewer.load_image(self.images[0])

    def _on_image_selected(self, path: Path):
        self.viewer.load_image(path)

    def _run_stack(self):
        if not self.images:
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.stack_btn.setEnabled(False)
        self.open_btn.setEnabled(False)

        self.worker = StackWorker(self.stacker, self.images)
        self.worker.progress.connect(self._on_stack_progress)
        self.worker.finished.connect(self._on_stack_finished)
        self.worker.error.connect(self._on_stack_error)
        self.worker.start()

    def _on_stack_progress(self, value: int):
        self.progress.setValue(value)

    def _on_stack_finished(self, result: np.ndarray):
        self.result_image = result
        self.progress.setVisible(False)
        self.stack_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        # Display result in viewer
        self.viewer.load_array(result)
        self.worker = None

    def _on_stack_error(self, error_msg: str):
        self.progress.setVisible(False)
        self.stack_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        QMessageBox.critical(self, "Stacking Error", error_msg)
        self.worker = None

    def _save_result(self):
        if self.result_image is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Result",
            filter="TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png)"
        )
        if path:
            # Add extension if not present
            if selected_filter == "TIFF (*.tif)" and not path.endswith('.tif'):
                path += '.tif'
            elif selected_filter == "JPEG (*.jpg)" and not path.endswith('.jpg'):
                path += '.jpg'
            elif selected_filter == "PNG (*.png)" and not path.endswith('.png'):
                path += '.png'

            cv2.imwrite(path, self.result_image)
