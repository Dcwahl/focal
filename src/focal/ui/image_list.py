"""Image list sidebar widget."""

from pathlib import Path
from PySide6.QtWidgets import QListWidget, QListWidgetItem
from PySide6.QtCore import Signal


class ImageList(QListWidget):
    image_selected = Signal(Path)

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.itemClicked.connect(self._on_item_clicked)

    def set_images(self, images: list[Path]):
        self.images = images
        self.clear()
        for img in images:
            item = QListWidgetItem(img.name)
            self.addItem(item)

        if images:
            self.setCurrentRow(0)

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = self.row(item)
        if 0 <= idx < len(self.images):
            self.image_selected.emit(self.images[idx])
