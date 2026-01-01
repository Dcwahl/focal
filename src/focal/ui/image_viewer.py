"""Image preview widget."""

from pathlib import Path
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget, QVBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class ImageViewer(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setWidget(container)
        self.current_pixmap = None

    def load_image(self, path: Path):
        pixmap = QPixmap(str(path))
        self.current_pixmap = pixmap
        self._update_display()

    def set_pixmap(self, pixmap: QPixmap):
        """Set pixmap directly (for showing stacking result)."""
        self.current_pixmap = pixmap
        self._update_display()

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
