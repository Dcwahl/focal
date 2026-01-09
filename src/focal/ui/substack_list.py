"""Substack list widget for displaying and selecting substacks."""

from dataclasses import dataclass
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QFrame,
)
from PySide6.QtCore import Signal, Qt

from focal.core.image_cache import CacheKey


@dataclass
class Substack:
    """A stacked subset of source frames."""

    frame_indices: tuple[int, ...]  # Immutable tuple for cache key
    display_name: str  # e.g., "Frames 3-7"

    @property
    def cache_key(self) -> CacheKey:
        return CacheKey("substack", self.frame_indices)

    @staticmethod
    def create_display_name(indices: tuple[int, ...]) -> str:
        """Generate display name from frame indices (1-based for users)."""
        if not indices:
            return "Empty"
        if len(indices) == 1:
            return f"Frame {indices[0] + 1}"

        # Check if consecutive
        sorted_indices = sorted(indices)
        is_consecutive = all(
            sorted_indices[i] + 1 == sorted_indices[i + 1]
            for i in range(len(sorted_indices) - 1)
        )

        if is_consecutive:
            return f"Frames {sorted_indices[0] + 1}-{sorted_indices[-1] + 1}"
        else:
            # Non-consecutive: show first few
            if len(indices) <= 3:
                return "Frames " + ", ".join(str(i + 1) for i in sorted_indices)
            else:
                return f"Frames {sorted_indices[0] + 1}, {sorted_indices[1] + 1}, ... ({len(indices)} total)"


class SubstackListItem(QWidget):
    """Single substack entry with name and delete button."""

    delete_requested = Signal(int)  # index in list
    clicked = Signal(int)  # index in list

    def __init__(self, index: int, substack: Substack, parent=None):
        super().__init__(parent)
        self.index = index
        self.substack = substack

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 4, 4)
        layout.setSpacing(4)

        self.label = QLabel(substack.display_name)
        self.label.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.label, stretch=1)

        self.delete_btn = QPushButton("×")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setToolTip("Delete substack")
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.index))
        layout.addWidget(self.delete_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        """Visual feedback for selection."""
        if selected:
            self.setStyleSheet("background-color: palette(highlight); border-radius: 3px;")
            self.label.setStyleSheet("color: palette(highlighted-text);")
        else:
            self.setStyleSheet("")
            self.label.setStyleSheet("")


class SubstackList(QWidget):
    """Widget showing list of substacks with selection and delete."""

    substack_selected = Signal(object)  # Emits Substack or None
    substack_deleted = Signal(int)  # Emits index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.substacks: list[Substack] = []
        self.selected_index: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Substacks")
        header.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(header)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # List container
        self.list_widget = QListWidget()
        self.list_widget.setSpacing(2)
        layout.addWidget(self.list_widget)

        # Empty state label
        self.empty_label = QLabel("No substacks yet.\nSelect frames and click\n'Create Substack'.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: gray; padding: 10px;")
        layout.addWidget(self.empty_label)

        self._update_empty_state()

    def add_substack(self, substack: Substack):
        """Add a substack to the list."""
        self.substacks.append(substack)
        self._rebuild_list()
        # Auto-select the new substack
        self._select_index(len(self.substacks) - 1)

    def remove_substack(self, index: int):
        """Remove a substack by index."""
        if 0 <= index < len(self.substacks):
            self.substacks.pop(index)
            if self.selected_index == index:
                self.selected_index = None
                self.substack_selected.emit(None)
            elif self.selected_index is not None and self.selected_index > index:
                self.selected_index -= 1
            self._rebuild_list()
            self.substack_deleted.emit(index)

    def clear(self):
        """Clear all substacks."""
        self.substacks.clear()
        self.selected_index = None
        self._rebuild_list()
        self.substack_selected.emit(None)

    def deselect(self):
        """Deselect current substack."""
        if self.selected_index is not None:
            self.selected_index = None
            self._update_selection_visuals()
            self.substack_selected.emit(None)

    def _rebuild_list(self):
        """Rebuild the list widget from substacks."""
        self.list_widget.clear()

        for i, substack in enumerate(self.substacks):
            item = QListWidgetItem()
            widget = SubstackListItem(i, substack)
            widget.clicked.connect(self._on_item_clicked)
            widget.delete_requested.connect(self._on_delete_requested)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

        self._update_empty_state()
        self._update_selection_visuals()

    def _update_empty_state(self):
        """Show/hide empty state message."""
        has_substacks = len(self.substacks) > 0
        self.list_widget.setVisible(has_substacks)
        self.empty_label.setVisible(not has_substacks)

    def _on_item_clicked(self, index: int):
        """Handle substack item click."""
        self._select_index(index)

    def _select_index(self, index: int):
        """Select a substack by index."""
        if 0 <= index < len(self.substacks):
            self.selected_index = index
            self._update_selection_visuals()
            self.substack_selected.emit(self.substacks[index])

    def _update_selection_visuals(self):
        """Update visual selection state."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if isinstance(widget, SubstackListItem):
                widget.set_selected(i == self.selected_index)

    def _on_delete_requested(self, index: int):
        """Handle delete button click."""
        self.remove_substack(index)

    def get_selected_substack(self) -> Substack | None:
        """Get currently selected substack."""
        if self.selected_index is not None and 0 <= self.selected_index < len(self.substacks):
            return self.substacks[self.selected_index]
        return None
