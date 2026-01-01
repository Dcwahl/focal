"""Image preview widget."""

from pathlib import Path
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget, QVBoxLayout
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal
import numpy as np


class ImageViewer(QScrollArea):
    # Signal emitted when brush painting occurs: (img_x, img_y)
    brush_paint = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMouseTracking(True)
        layout.addWidget(self.label)

        self.setWidget(container)
        self.current_pixmap = None

        # Brush state
        self._brush_mode = False
        self._brush_size = 0
        self._painting = False

        # For coordinate mapping
        self._image_size = None  # (width, height) of original image
        self._display_rect = None  # QRect of where image is displayed in label

    def load_image(self, path: Path):
        pixmap = QPixmap(str(path))
        self.current_pixmap = pixmap
        self._image_size = (pixmap.width(), pixmap.height())
        self._update_display()

    def set_pixmap(self, pixmap: QPixmap):
        """Set pixmap directly (for showing stacking result)."""
        self.current_pixmap = pixmap
        self._update_display()

    def load_array(self, array: np.ndarray):
        """Load a numpy array (BGR, uint8) as the display image."""
        # Convert BGR to RGB for Qt
        rgb = array[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        self._image_size = (w, h)
        self._update_display()

    def clear(self):
        """Clear the displayed image."""
        self.current_pixmap = None
        self.label.clear()

    def _update_display(self):
        if self.current_pixmap is None:
            return

        # Scale to fit while maintaining aspect ratio
        scaled = self.current_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)

        # Store display rect for coordinate mapping
        # The scaled image is centered in the label
        label_w, label_h = self.label.width(), self.label.height()
        scaled_w, scaled_h = scaled.width(), scaled.height()
        x_offset = (label_w - scaled_w) // 2
        y_offset = (label_h - scaled_h) // 2
        self._display_rect = (x_offset, y_offset, scaled_w, scaled_h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def set_brush_mode(self, enabled: bool, size: int):
        """Enable or disable brush mode."""
        self._brush_mode = enabled
        self._brush_size = size

    def _screen_to_image_coords(self, pos):
        """Convert screen position to image coordinates."""
        if self._display_rect is None or self._image_size is None:
            return None

        x_offset, y_offset, disp_w, disp_h = self._display_rect
        img_w, img_h = self._image_size

        # Get position relative to label
        label_pos = self.label.mapFrom(self, pos)
        x, y = label_pos.x(), label_pos.y()

        # Check if within displayed image bounds
        if x < x_offset or x >= x_offset + disp_w:
            return None
        if y < y_offset or y >= y_offset + disp_h:
            return None

        # Map to image coordinates
        rel_x = (x - x_offset) / disp_w
        rel_y = (y - y_offset) / disp_h
        img_x = int(rel_x * img_w)
        img_y = int(rel_y * img_h)

        return (img_x, img_y)

    def mousePressEvent(self, event):
        if self._brush_mode and event.button() == Qt.LeftButton:
            coords = self._screen_to_image_coords(event.pos())
            if coords:
                self._painting = True
                self.brush_paint.emit(coords[0], coords[1])
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._brush_mode and self._painting:
            coords = self._screen_to_image_coords(event.pos())
            if coords:
                self.brush_paint.emit(coords[0], coords[1])
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._painting = False
        super().mouseReleaseEvent(event)
