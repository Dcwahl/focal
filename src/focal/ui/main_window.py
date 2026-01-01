"""Main application window."""

from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QFileDialog, QSplitter, QMessageBox,
    QLabel, QFrame, QSlider, QShortcut,
)
from PySide6.QtGui import QKeySequence
from PySide6.QtCore import Qt, QThread, Signal
import cv2
import numpy as np

from focal.ui.image_list import ImageList
from focal.ui.image_viewer import ImageViewer
from focal.core.stacker import FocusStacker


class BrushStroke:
    """Represents a single brush stroke for undo/redo."""
    def __init__(self, points: list[tuple[int, int]], source_index: int, brush_size: int):
        self.points = points  # List of (x, y) image coordinates
        self.source_index = source_index
        self.brush_size = brush_size
        # Store the original pixels before this stroke was applied
        self.original_pixels: dict[tuple[int, int], np.ndarray] = {}


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

        # Undo/redo stacks
        self._undo_stack: list[BrushStroke] = []
        self._redo_stack: list[BrushStroke] = []
        self._current_stroke: BrushStroke | None = None

        self._setup_ui()
        self._setup_shortcuts()

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

        top_bar.addStretch()

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        top_bar.addWidget(zoom_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)   # 10% zoom
        self.zoom_slider.setMaximum(500)  # 500% zoom
        self.zoom_slider.setValue(100)    # 100% = fit
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        top_bar.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(40)
        top_bar.addWidget(self.zoom_label)

        # Reset zoom button
        self.reset_zoom_btn = QPushButton("Fit")
        self.reset_zoom_btn.setToolTip("Reset zoom to fit image (Ctrl+0, scroll to zoom, drag to pan)")
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        top_bar.addWidget(self.reset_zoom_btn)

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
        self.source_viewer.zoom_changed.connect(self._on_zoom_changed)
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
        self.result_viewer.brush_stroke_started.connect(self._on_stroke_started)
        self.result_viewer.brush_stroke_finished.connect(self._on_stroke_finished)
        self.result_viewer.zoom_changed.connect(self._on_zoom_changed)
        result_layout.addWidget(self.result_viewer, stretch=1)
        splitter.addWidget(result_container)

        self.image_list = ImageList()
        self.image_list.image_selected.connect(self._on_image_selected)
        self.image_list.image_remove_requested.connect(self._remove_image)
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
        """Open images via folder selection or file selection."""
        # Use getOpenFileNames for individual file selection
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images",
            filter="Images (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)"
        )
        if files:
            self._load_image_files([Path(f) for f in files])

    def _load_image_files(self, files: list[Path]):
        """Load a list of image files."""
        self.images = sorted(files)

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
        self.edited_result = None
        self.save_btn.setEnabled(False)
        self.brush_btn.setEnabled(False)
        self.source_arrays.clear()

        # Clear undo/redo stacks
        self._undo_stack.clear()
        self._redo_stack.clear()

        if self.images:
            self.current_source_index = 0
            self.source_viewer.load_image(self.images[0])
            self.result_viewer.clear()

    def _on_image_selected(self, path: Path):
        """Handle image selection from sidebar."""
        try:
            idx = self.images.index(path)
            self.current_source_index = idx
        except ValueError:
            pass
        # Preserve zoom when switching source frames
        self.source_viewer.load_image(path, preserve_zoom=True)

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

        # Clear undo/redo stacks for new stack result
        self._undo_stack.clear()
        self._redo_stack.clear()

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
        """Handle key press - S for flash compare, Ctrl+/- for zoom."""
        if event.key() == Qt.Key_S and not event.isAutoRepeat():
            if self.result_image is not None and self.images:
                self._flash_active = True
                # Show current source in the result panel, preserving zoom
                self.result_viewer.load_image(self.images[self.current_source_index], preserve_zoom=True)
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
                # Ctrl++ or Ctrl+= (= is on same key as +)
                self.source_viewer.zoom_in()
                self.result_viewer.zoom_in()
            elif event.key() == Qt.Key_Minus:
                # Ctrl+-
                self.source_viewer.zoom_out()
                self.result_viewer.zoom_out()
            elif event.key() == Qt.Key_0:
                # Ctrl+0 - reset zoom
                self._reset_zoom()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release - restore result view."""
        if event.key() == Qt.Key_S and not event.isAutoRepeat():
            if self._flash_active and self.edited_result is not None:
                self._flash_active = False
                # Restore result view, preserving zoom
                self.result_viewer.load_array(self.edited_result, preserve_zoom=True)
        super().keyReleaseEvent(event)

    def _toggle_brush_mode(self, checked: bool):
        """Toggle brush painting mode."""
        self.brush_mode = checked
        self.brush_btn.setText("Brush: On" if checked else "Brush: Off")
        # Cursor handling is now inside ImageViewer
        self.result_viewer.set_brush_mode(checked, self.brush_size)

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

    def _apply_brush_stroke(self, img_x: int, img_y: int, record_undo: bool = True):
        """Apply brush stroke: copy pixels from source to result with feathered edges."""
        if self.edited_result is None:
            return

        source = self._get_source_array(self.current_source_index)
        if source is None:
            return

        h, w = self.edited_result.shape[:2]
        radius = self.brush_size // 2

        # Calculate bounds in image coordinates
        y_start = max(0, img_y - radius)
        y_end = min(h, img_y + radius + 1)
        x_start = max(0, img_x - radius)
        x_end = min(w, img_x + radius + 1)

        if y_start >= y_end or x_start >= x_end:
            return

        # Create feathered circular mask with alpha falloff
        y_coords, x_coords = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist = np.sqrt(x_coords**2 + y_coords**2)
        # Feather: full opacity in center, fading to 0 at edge
        # Use inner 60% as full opacity, outer 40% as falloff
        inner_radius = radius * 0.6
        alpha = np.clip((radius - dist) / (radius - inner_radius), 0, 1)

        # Calculate corresponding bounds in mask
        mask_y_start = max(0, radius - img_y)
        mask_y_end = alpha.shape[0] - max(0, img_y + radius + 1 - h)
        mask_x_start = max(0, radius - img_x)
        mask_x_end = alpha.shape[1] - max(0, img_x + radius + 1 - w)

        alpha_slice = alpha[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

        # Store original pixels for undo (only pixels we're actually changing)
        if record_undo and self._current_stroke is not None:
            key = (img_x, img_y)
            if key not in self._current_stroke.original_pixels:
                # Store original region before modification
                self._current_stroke.original_pixels[key] = self.edited_result[
                    y_start:y_end, x_start:x_end
                ].copy()
            self._current_stroke.points.append(key)

        # Apply alpha-blended copy from source to result
        for c in range(3):
            result_region = self.edited_result[y_start:y_end, x_start:x_end, c].astype(np.float32)
            source_region = source[y_start:y_end, x_start:x_end, c].astype(np.float32)
            blended = result_region * (1 - alpha_slice) + source_region * alpha_slice
            self.edited_result[y_start:y_end, x_start:x_end, c] = blended.astype(np.uint8)

    def on_brush_paint(self, img_x: int, img_y: int):
        """Called from result_viewer when painting occurs."""
        self._apply_brush_stroke(img_x, img_y)
        self.result_viewer.load_array(self.edited_result, preserve_zoom=True)

    def _on_stroke_started(self):
        """Called when a new brush stroke begins."""
        self._current_stroke = BrushStroke(
            points=[],
            source_index=self.current_source_index,
            brush_size=self.brush_size
        )

    def _on_stroke_finished(self):
        """Called when a brush stroke ends."""
        if self._current_stroke is not None and self._current_stroke.points:
            self._undo_stack.append(self._current_stroke)
            self._redo_stack.clear()  # Clear redo stack on new action
        self._current_stroke = None

    def _undo(self):
        """Undo the last brush stroke."""
        if not self._undo_stack or self.edited_result is None:
            return

        stroke = self._undo_stack.pop()

        # Store current state for redo
        redo_pixels = {}
        for (x, y), orig in stroke.original_pixels.items():
            h, w = self.edited_result.shape[:2]
            radius = stroke.brush_size // 2
            y_start = max(0, y - radius)
            y_end = min(h, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(w, x + radius + 1)
            redo_pixels[(x, y)] = self.edited_result[y_start:y_end, x_start:x_end].copy()

        # Restore original pixels
        for (x, y), orig in stroke.original_pixels.items():
            h, w = self.edited_result.shape[:2]
            radius = stroke.brush_size // 2
            y_start = max(0, y - radius)
            y_end = min(h, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(w, x + radius + 1)
            self.edited_result[y_start:y_end, x_start:x_end] = orig

        # Save for redo with swapped pixels
        stroke.original_pixels = redo_pixels
        self._redo_stack.append(stroke)

        self.result_viewer.load_array(self.edited_result, preserve_zoom=True)

    def _redo(self):
        """Redo the last undone brush stroke."""
        if not self._redo_stack or self.edited_result is None:
            return

        stroke = self._redo_stack.pop()

        # Store current state for undo
        undo_pixels = {}
        for (x, y), redo_state in stroke.original_pixels.items():
            h, w = self.edited_result.shape[:2]
            radius = stroke.brush_size // 2
            y_start = max(0, y - radius)
            y_end = min(h, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(w, x + radius + 1)
            undo_pixels[(x, y)] = self.edited_result[y_start:y_end, x_start:x_end].copy()

        # Apply redo (restore the painted state)
        for (x, y), redo_state in stroke.original_pixels.items():
            h, w = self.edited_result.shape[:2]
            radius = stroke.brush_size // 2
            y_start = max(0, y - radius)
            y_end = min(h, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(w, x + radius + 1)
            self.edited_result[y_start:y_end, x_start:x_end] = redo_state

        # Swap pixels for next undo
        stroke.original_pixels = undo_pixels
        self._undo_stack.append(stroke)

        self.result_viewer.load_array(self.edited_result, preserve_zoom=True)

    def _reset_zoom(self):
        """Reset zoom on both viewers."""
        self.source_viewer.reset_zoom()
        self.result_viewer.reset_zoom()

    def _on_zoom_changed(self, percent: int):
        """Update zoom slider and label when zoom changes (from either viewer)."""
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(percent)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{percent}%")

    def _on_zoom_slider_changed(self, value: int):
        """Apply zoom slider value to both viewers."""
        self.source_viewer.set_zoom_percent(value)
        self.result_viewer.set_zoom_percent(value)
        self.zoom_label.setText(f"{value}%")

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Undo: Ctrl+Z
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._undo)

        # Redo: Ctrl+Shift+Z or Ctrl+Y
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self._redo)

    def _remove_image(self, index: int):
        """Remove an image from the stack."""
        if 0 <= index < len(self.images):
            removed_path = self.images.pop(index)
            # Clear cache if present
            if index in self.source_arrays:
                del self.source_arrays[index]
            # Rebuild cache keys (shift indices)
            new_cache = {}
            for k, v in self.source_arrays.items():
                if k > index:
                    new_cache[k - 1] = v
                else:
                    new_cache[k] = v
            self.source_arrays = new_cache

            # Update UI
            self.image_list.set_images(self.images)
            self.stack_btn.setEnabled(len(self.images) > 1)

            # Update selection
            if self.images:
                new_index = min(index, len(self.images) - 1)
                self.current_source_index = new_index
                self.image_list.setCurrentRow(new_index)
                self.source_viewer.load_image(self.images[new_index], preserve_zoom=True)
