"""Main application window."""

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QFileDialog, QSplitter,
)
from PySide6.QtCore import Qt

from focal.ui.image_list import ImageList
from focal.ui.image_viewer import ImageViewer
from focal.core.stacker import FocusStacker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Focal")
        self.setMinimumSize(900, 600)

        self.images: list[Path] = []
        self.result_image = None
        self.stacker = FocusStacker()

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
        # TODO: Implement actual stacking
        # For now, just a placeholder that shows progress
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.stack_btn.setEnabled(False)

        # Placeholder: just show we'd do something
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.stack_btn.setEnabled(True)

    def _save_result(self):
        if self.result_image is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result",
            filter="TIFF (*.tif);;JPEG (*.jpg)"
        )
        if path:
            # TODO: Save self.result_image to path
            pass
