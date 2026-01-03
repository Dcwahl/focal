"""Image list sidebar widget."""

from pathlib import Path
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QMenu
from PySide6.QtCore import Signal, Qt


class ImageList(QListWidget):
    image_selected = Signal(Path)
    image_remove_requested = Signal(int)  # Signal with index to remove

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.itemClicked.connect(self._on_item_clicked)
        self.currentRowChanged.connect(self._on_row_changed)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_images(self, images: list[Path]):
        self.images = list(images)  # Copy to avoid mutation issues
        self.clear()
        for img in self.images:
            item = QListWidgetItem(img.name)
            self.addItem(item)

        if self.images:
            self.setCurrentRow(0)

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = self.row(item)
        if 0 <= idx < len(self.images):
            self.image_selected.emit(self.images[idx])

    def _on_row_changed(self, row: int):
        """Handle arrow key navigation."""
        if 0 <= row < len(self.images):
            self.image_selected.emit(self.images[row])

    def _show_context_menu(self, pos):
        """Show context menu for image removal."""
        item = self.itemAt(pos)
        if item is None:
            return

        idx = self.row(item)
        menu = QMenu(self)
        remove_action = menu.addAction(f"Remove '{item.text()}'")

        action = menu.exec_(self.mapToGlobal(pos))
        if action == remove_action:
            self.image_remove_requested.emit(idx)
