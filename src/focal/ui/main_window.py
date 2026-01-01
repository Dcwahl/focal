"""Main application window."""

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QFileDialog, QSplitter, QMessageBox,
    QLabel, QFrame, QComboBox, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCursor
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
        self.edited_result: np.ndarray | None = None  # Result with brush edits applied
        self.source_arrays: dict[int, np.ndarray] = {}  # Cached source images
        self.stacker = FocusStacker()
        self.worker: StackWorker | None = None
        self.current_source_index: int = 0
        self._flash_active: bool = False

        # Brush state
        self.brush_mode: bool = False
        self.brush_size: int = 30
        self._painting: bool = False

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

        # Source frame selector
        source_selector_label = QLabel("Source frame:")
        top_bar.addWidget(source_selector_label)
        self.source_selector = QComboBox()
        self.source_selector.setMinimumWidth(150)
        self.source_selector.currentIndexChanged.connect(self._on_source_selector_changed)
        top_bar.addWidget(self.source_selector)

        # Flash compare hint
        flash_hint = QLabel("(Hold S to flash compare)")
        flash_hint.setStyleSheet("color: gray; font-size: 11px;")
        top_bar.addWidget(flash_hint)

        top_bar.addStretch()

        # Brush controls
        self.brush_btn = QPushButton("Brush: Off")
        self.brush_btn.setCheckable(True)
        self.brush_btn.clicked.connect(self._toggle_brush_mode)
        self.brush_btn.setEnabled(False)
        top_bar.addWidget(self.brush_btn)

        brush_size_label = QLabel("Size:")
        top_bar.addWidget(brush_size_label)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(5)
        self.brush_slider.setMaximum(100)
        self.brush_slider.setValue(30)
        self.brush_slider.setFixedWidth(100)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        top_bar.addWidget(self.brush_slider)
        self.brush_size_label = QLabel("30")
        self.brush_size_label.setFixedWidth(25)
        top_bar.addWidget(self.brush_size_label)

        layout.addLayout(top_bar)

        # Main content: source viewer + result viewer + image list
        splitter = QSplitter(Qt.Horizontal)

        # Source viewer with label
        source_container = QWidget()
        source_layout = QVBoxLayout(source_container)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_label = QLabel("Source")
        source_label.setAlignment(Qt.AlignCenter)
        source_label.setStyleSheet("font-weight: bold; padding: 4px;")
        source_layout.addWidget(source_label)
        self.source_viewer = ImageViewer()
        source_layout.addWidget(self.source_viewer, stretch=1)
        splitter.addWidget(source_container)

        # Result viewer with label
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_label = QLabel("Result")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("font-weight: bold; padding: 4px;")
        result_layout.addWidget(result_label)
        self.result_viewer = ImageViewer()
        self.result_viewer.brush_paint.connect(self.on_brush_paint)
        result_layout.addWidget(self.result_viewer, stretch=1)
        splitter.addWidget(result_container)

        self.image_list = ImageList()
        self.image_list.image_selected.connect(self._on_image_selected)
        splitter.addWidget(self.image_list)

        splitter.setSizes([400, 400, 200])
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

        # Validate dimensions match
        if len(self.images) > 1:
            dimensions = []
            for img_path in self.images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    dimensions.append((img_path.name, img.shape[:2]))

            if dimensions:
                first_dim = dimensions[0][1]
                mismatched = [
                    (name, dim) for name, dim in dimensions
                    if dim != first_dim
                ]
                if mismatched:
                    msg = f"Image dimension mismatch detected.\n\n"
                    msg += f"Expected: {first_dim[1]}x{first_dim[0]} (from {dimensions[0][0]})\n\n"
                    msg += "Mismatched images:\n"
                    for name, dim in mismatched:
                        msg += f"  • {name}: {dim[1]}x{dim[0]}\n"
                    msg += "\nStacking requires all images to have the same dimensions."
                    QMessageBox.warning(self, "Dimension Mismatch", msg)
                    self.images = []
                    self.image_list.set_images([])
                    self.stack_btn.setEnabled(False)
                    return

        self.image_list.set_images(self.images)
        self.stack_btn.setEnabled(len(self.images) > 1)
        self.result_image = None
        self.save_btn.setEnabled(False)

        # Update source selector dropdown
        self.source_selector.blockSignals(True)
        self.source_selector.clear()
        for img in self.images:
            self.source_selector.addItem(img.name)
        self.source_selector.blockSignals(False)

        if self.images:
            self.current_source_index = 0
            self.source_selector.setCurrentIndex(0)
            self.source_viewer.load_image(self.images[0])
            self.result_viewer.clear()

    def _on_image_selected(self, path: Path):
        # Find index and sync dropdown
        try:
            idx = self.images.index(path)
            self.current_source_index = idx
            self.source_selector.blockSignals(True)
            self.source_selector.setCurrentIndex(idx)
            self.source_selector.blockSignals(False)
        except ValueError:
            pass
        self.source_viewer.load_image(path)

    def _on_source_selector_changed(self, index: int):
        if 0 <= index < len(self.images):
            self.current_source_index = index
            self.source_viewer.load_image(self.images[index])
            # Sync sidebar selection
            self.image_list.blockSignals(True)
            self.image_list.setCurrentRow(index)
            self.image_list.blockSignals(False)

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
        self.edited_result = result.copy()  # Start with unedited result
        self.progress.setVisible(False)
        self.stack_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.brush_btn.setEnabled(True)

        # Display result in result viewer
        self.result_viewer.load_array(self.edited_result)
        self.worker = None

    def _on_stack_error(self, error_msg: str):
        self.progress.setVisible(False)
        self.stack_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        QMessageBox.critical(self, "Stacking Error", error_msg)
        self.worker = None

    def _save_result(self):
        if self.edited_result is None:
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

            cv2.imwrite(path, self.edited_result)

    def keyPressEvent(self, event):
        """Handle key press - S for flash compare."""
        if event.key() == Qt.Key_S and not event.isAutoRepeat():
            if self.result_image is not None and self.images:
                self._flash_active = True
                # Show current source in the result panel
                self.result_viewer.load_image(self.images[self.current_source_index])
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release - restore result view."""
        if event.key() == Qt.Key_S and not event.isAutoRepeat():
            if self._flash_active and self.edited_result is not None:
                self._flash_active = False
                # Restore result view
                self.result_viewer.load_array(self.edited_result)
        super().keyReleaseEvent(event)

    def _toggle_brush_mode(self, checked: bool):
        """Toggle brush painting mode."""
        self.brush_mode = checked
        self.brush_btn.setText("Brush: On" if checked else "Brush: Off")
        if checked:
            self.result_viewer.setCursor(Qt.CrossCursor)
            self.result_viewer.set_brush_mode(True, self.brush_size)
        else:
            self.result_viewer.setCursor(Qt.ArrowCursor)
            self.result_viewer.set_brush_mode(False, 0)

    def _on_brush_size_changed(self, value: int):
        """Update brush size."""
        self.brush_size = value
        self.brush_size_label.setText(str(value))
        if self.brush_mode:
            self.result_viewer.set_brush_mode(True, value)

    def _get_source_array(self, index: int) -> np.ndarray | None:
        """Get source image as numpy array, caching for performance."""
        if index not in self.source_arrays:
            if 0 <= index < len(self.images):
                img = cv2.imread(str(self.images[index]))
                if img is not None:
                    self.source_arrays[index] = img
        return self.source_arrays.get(index)

    def _apply_brush_stroke(self, img_x: int, img_y: int):
        """Apply brush stroke: copy pixels from source to result."""
        if self.edited_result is None:
            return

        source = self._get_source_array(self.current_source_index)
        if source is None:
            return

        h, w = self.edited_result.shape[:2]
        radius = self.brush_size // 2

        # Create circular mask
        y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x_coords**2 + y_coords**2 <= radius**2

        # Calculate bounds in image coordinates
        y_start = max(0, img_y - radius)
        y_end = min(h, img_y + radius + 1)
        x_start = max(0, img_x - radius)
        x_end = min(w, img_x + radius + 1)

        # Calculate corresponding bounds in mask
        mask_y_start = max(0, radius - img_y)
        mask_y_end = mask.shape[0] - max(0, img_y + radius + 1 - h)
        mask_x_start = max(0, radius - img_x)
        mask_x_end = mask.shape[1] - max(0, img_x + radius + 1 - w)

        # Apply masked copy from source to result
        mask_slice = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
        for c in range(3):
            result_region = self.edited_result[y_start:y_end, x_start:x_end, c]
            source_region = source[y_start:y_end, x_start:x_end, c]
            result_region[mask_slice] = source_region[mask_slice]

    def on_brush_paint(self, img_x: int, img_y: int):
        """Called from result_viewer when painting occurs."""
        self._apply_brush_stroke(img_x, img_y)
        self.result_viewer.load_array(self.edited_result)
