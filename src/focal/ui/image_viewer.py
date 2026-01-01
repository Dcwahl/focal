"""Image preview widget with zoom/pan support."""

from pathlib import Path
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QFrame,
)
from PySide6.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPainter
from PySide6.QtCore import Qt, Signal, QRectF
import numpy as np


class ImageViewer(QGraphicsView):
    """Image viewer with zoom, pan, and brush support."""

    # Signal emitted when brush painting occurs: (img_x, img_y)
    brush_paint = Signal(int, int)

    def __init__(self):
        super().__init__()

        # Scene setup
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Image item
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._current_pixmap: QPixmap | None = None

        # For coordinate mapping to original image
        self._image_size: tuple[int, int] | None = None  # (width, height)

        # View settings
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))

        # Zoom state
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0

        # Pan state
        self._panning = False
        self._pan_start = None

        # Brush state
        self._brush_mode = False
        self._brush_size = 30
        self._painting = False
        self._brush_cursor: QGraphicsEllipseItem | None = None

        # Track mouse for brush cursor
        self.setMouseTracking(True)

    def load_image(self, path: Path):
        """Load image from file path."""
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self._set_pixmap(pixmap)
            self._image_size = (pixmap.width(), pixmap.height())

    def load_array(self, array: np.ndarray):
        """Load a numpy array (BGR, uint8) as the display image."""
        # Convert BGR to RGB for Qt
        rgb = array[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._set_pixmap(pixmap)
        self._image_size = (w, h)

    def _set_pixmap(self, pixmap: QPixmap):
        """Set the displayed pixmap."""
        self._current_pixmap = pixmap

        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)

        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._fit_in_view()

    def _fit_in_view(self):
        """Fit image to view while maintaining aspect ratio."""
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        # Store the current zoom as our baseline
        self._zoom_factor = self.transform().m11()

    def clear(self):
        """Clear the displayed image."""
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._current_pixmap = None
        self._image_size = None
        self._remove_brush_cursor()

    def set_brush_mode(self, enabled: bool, size: int):
        """Enable or disable brush mode."""
        self._brush_mode = enabled
        self._brush_size = size

        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.BlankCursor)  # Hide system cursor, we'll draw our own
            self._update_brush_cursor_size()
        else:
            self.setCursor(Qt.ArrowCursor)
            self._remove_brush_cursor()

    def _create_brush_cursor(self):
        """Create the brush cursor circle."""
        if self._brush_cursor is None:
            pen = QPen(QColor(255, 255, 255, 200))
            pen.setWidth(2)
            pen.setCosmetic(True)  # Width doesn't scale with zoom
            self._brush_cursor = self._scene.addEllipse(
                0, 0, self._brush_size, self._brush_size,
                pen, QBrush(Qt.NoBrush)
            )
            self._brush_cursor.setZValue(1000)  # Always on top

    def _remove_brush_cursor(self):
        """Remove the brush cursor from scene."""
        if self._brush_cursor is not None:
            self._scene.removeItem(self._brush_cursor)
            self._brush_cursor = None

    def _update_brush_cursor_size(self):
        """Update brush cursor size."""
        if self._brush_cursor is not None:
            self._brush_cursor.setRect(
                0, 0, self._brush_size, self._brush_size
            )

    def _update_brush_cursor_pos(self, scene_pos):
        """Update brush cursor position in scene coordinates."""
        if self._brush_mode:
            if self._brush_cursor is None:
                self._create_brush_cursor()
            # Center the cursor on the mouse position
            half_size = self._brush_size / 2
            self._brush_cursor.setPos(
                scene_pos.x() - half_size,
                scene_pos.y() - half_size
            )

    def _scene_to_image_coords(self, scene_pos) -> tuple[int, int] | None:
        """Convert scene position to image coordinates."""
        if self._pixmap_item is None or self._image_size is None:
            return None

        # Scene coords map directly to image coords since we set scene rect to pixmap rect
        x, y = int(scene_pos.x()), int(scene_pos.y())
        w, h = self._image_size

        # Bounds check
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None

    # ----- Event handlers -----

    def wheelEvent(self, event):
        """Handle zoom with mouse wheel."""
        if event.modifiers() == Qt.ControlModifier or not self._brush_mode:
            # Zoom
            factor = 1.15
            if event.angleDelta().y() < 0:
                factor = 1 / factor

            new_zoom = self._zoom_factor * factor
            if self._min_zoom <= new_zoom <= self._max_zoom:
                self.scale(factor, factor)
                self._zoom_factor = new_zoom
        else:
            # Could use scroll to change brush size here
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press for panning or painting."""
        if event.button() == Qt.MiddleButton:
            # Start panning
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            if self._brush_mode:
                # Start painting
                self._painting = True
                scene_pos = self.mapToScene(event.pos())
                coords = self._scene_to_image_coords(scene_pos)
                if coords:
                    self.brush_paint.emit(coords[0], coords[1])
            else:
                # Start panning with left button when not in brush mode
                self._panning = True
                self._pan_start = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for panning, painting, or cursor update."""
        scene_pos = self.mapToScene(event.pos())

        # Update brush cursor position
        if self._brush_mode:
            self._update_brush_cursor_pos(scene_pos)

        if self._panning and self._pan_start is not None:
            # Pan the view
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        elif self._painting and self._brush_mode:
            # Continue painting
            coords = self._scene_to_image_coords(scene_pos)
            if coords:
                self.brush_paint.emit(coords[0], coords[1])
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._pan_start = None
            if self._brush_mode:
                self.setCursor(Qt.BlankCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton:
            if self._painting:
                self._painting = False
            else:
                self._panning = False
                self._pan_start = None
                if self._brush_mode:
                    self.setCursor(Qt.BlankCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """Handle resize - refit image if at default zoom."""
        super().resizeEvent(event)
        # Optionally refit on resize - for now we preserve zoom level
        # Uncomment to auto-fit on resize:
        # if self._pixmap_item is not None:
        #     self._fit_in_view()

    def enterEvent(self, event):
        """Show brush cursor when mouse enters."""
        if self._brush_mode:
            self.setCursor(Qt.BlankCursor)
            if self._brush_cursor is None:
                self._create_brush_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide brush cursor when mouse leaves."""
        if self._brush_mode:
            self._remove_brush_cursor()
        super().leaveEvent(event)

    def reset_zoom(self):
        """Reset to fit-in-view zoom level."""
        self.resetTransform()
        self._fit_in_view()
