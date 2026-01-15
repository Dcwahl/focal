"""Image list sidebar widget with checkbox selection for substacks."""

from pathlib import Path
from PySide6.QtWidgets import (
    QListWidget, QListWidgetItem, QMenu, QWidget, QHBoxLayout, QCheckBox, QLabel,
)
from PySide6.QtCore import Signal, Qt


class ImageListItem(QWidget):
    """Custom list item with checkbox and filename."""

    checkbox_changed = Signal(int, bool)  # index, checked

    def __init__(self, index: int, filename: str, parent=None):
        super().__init__(parent)
        self.index = index

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        layout.addWidget(self.checkbox)

        self.label = QLabel(filename)
        layout.addWidget(self.label, stretch=1)

    def _on_checkbox_changed(self, state: int):
        # Use isChecked() instead of state == Qt.Checked; Qt6 enum comparison is unreliable
        self.checkbox_changed.emit(self.index, self.checkbox.isChecked())

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool):
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(checked)
        self.checkbox.blockSignals(False)


class ImageList(QListWidget):
    image_selected = Signal(Path)
    image_remove_requested = Signal(int)  # Signal with index to remove
    selection_changed = Signal(set)  # Emits set of selected indices

    def __init__(self):
        super().__init__()
        self.images: list[Path] = []
        self.selected_indices: set[int] = set()
        self._last_clicked_index: int | None = None  # For shift-click range selection

        self.itemClicked.connect(self._on_item_clicked)
        self.currentRowChanged.connect(self._on_row_changed)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_images(self, images: list[Path]):
        self.images = list(images)  # Copy to avoid mutation issues
        self.selected_indices.clear()
        self._last_clicked_index = None
        self.clear()

        for i, img in enumerate(self.images):
            item = QListWidgetItem()
            self.addItem(item)
            widget = self._create_item_widget(i, img.name)
            item.setSizeHint(widget.sizeHint())
            self.setItemWidget(item, widget)

        if self.images:
            self.setCurrentRow(0)

        self.selection_changed.emit(self.selected_indices.copy())

    def _create_item_widget(self, index: int, filename: str) -> ImageListItem:
        widget = ImageListItem(index, filename)
        widget.checkbox_changed.connect(self._on_checkbox_toggled)
        return widget

    def _on_checkbox_toggled(self, index: int, checked: bool):
        """Handle checkbox toggle."""
        if checked:
            self.selected_indices.add(index)
        else:
            self.selected_indices.discard(index)
        self._last_clicked_index = index
        self.selection_changed.emit(self.selected_indices.copy())

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = self.row(item)
        if 0 <= idx < len(self.images):
            self.image_selected.emit(self.images[idx])

    def _on_row_changed(self, row: int):
        """Handle arrow key navigation."""
        if 0 <= row < len(self.images):
            self.image_selected.emit(self.images[row])

    def mousePressEvent(self, event):
        """Handle shift-click for range selection, ctrl-click for toggle."""
        item = self.itemAt(event.pos())
        if item is None:
            super().mousePressEvent(event)
            return

        idx = self.row(item)
        widget = self.itemWidget(item)

        if not isinstance(widget, ImageListItem):
            super().mousePressEvent(event)
            return

        modifiers = event.modifiers()

        if modifiers & Qt.ShiftModifier and self._last_clicked_index is not None:
            # Shift-click: select range
            start = min(self._last_clicked_index, idx)
            end = max(self._last_clicked_index, idx)
            for i in range(start, end + 1):
                self.selected_indices.add(i)
                row_item = self.item(i)
                if row_item:
                    row_widget = self.itemWidget(row_item)
                    if isinstance(row_widget, ImageListItem):
                        row_widget.set_checked(True)
            self.selection_changed.emit(self.selected_indices.copy())
        elif modifiers & Qt.ControlModifier:
            # Ctrl-click: toggle single checkbox (let the checkbox handle it)
            widget.checkbox.setChecked(not widget.checkbox.isChecked())
        else:
            # Normal click: just select the image for viewing, don't toggle checkbox
            pass

        self._last_clicked_index = idx
        super().mousePressEvent(event)

    def clear_selection(self):
        """Clear all checkbox selections."""
        self.selected_indices.clear()
        for i in range(self.count()):
            item = self.item(i)
            if item:
                widget = self.itemWidget(item)
                if isinstance(widget, ImageListItem):
                    widget.set_checked(False)
        self._last_clicked_index = None
        self.selection_changed.emit(self.selected_indices.copy())

    def get_selected_indices(self) -> list[int]:
        """Return sorted list of selected frame indices."""
        return sorted(self.selected_indices)

    def _show_context_menu(self, pos):
        """Show context menu for image removal."""
        item = self.itemAt(pos)
        if item is None:
            return

        idx = self.row(item)
        widget = self.itemWidget(item)
        filename = self.images[idx].name if 0 <= idx < len(self.images) else "image"

        menu = QMenu(self)
        remove_action = menu.addAction(f"Remove '{filename}'")

        action = menu.exec_(self.mapToGlobal(pos))
        if action == remove_action:
            self.image_remove_requested.emit(idx)
