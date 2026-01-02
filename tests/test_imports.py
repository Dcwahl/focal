"""Import tests to catch dependency issues early."""

import pytest


def test_import_focal():
    """Test that the main focal package imports."""
    import focal


def test_import_main():
    """Test that focal.main imports without errors."""
    from focal import main


def test_import_core_stacker():
    """Test that core.stacker imports without errors."""
    from focal.core import stacker
    from focal.core.stacker import FocusStacker, StackAlgorithm


def test_import_core_complex_wavelet():
    """Test that core.complex_wavelet imports without errors."""
    from focal.core import complex_wavelet
    from focal.core.complex_wavelet import decompose, compose, merge_wavelets


def test_import_ui_main_window():
    """Test that ui.main_window imports without errors."""
    from focal.ui import main_window
    from focal.ui.main_window import MainWindow


def test_import_ui_image_viewer():
    """Test that ui.image_viewer imports without errors."""
    from focal.ui import image_viewer
    from focal.ui.image_viewer import ImageViewer


def test_import_ui_image_list():
    """Test that ui.image_list imports without errors."""
    from focal.ui import image_list
    from focal.ui.image_list import ImageListWidget
