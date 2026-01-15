# Focal - AI Development Guide

## Project Overview

Focal is a focus stacking application targeting photographers who stack macro/micro images. It aims to compete with commercial tools (Helicon Focus, Zerene Stacker) by providing robust source-frame retouching workflows.

## Project Structure

```
focal/
├── src/focal/
│   ├── main.py              # Entry point
│   ├── ui/
│   │   ├── main_window.py   # Main application window
│   │   ├── image_viewer.py  # QGraphicsView-based image display with zoom/pan/brush
│   │   └── image_list.py    # File list sidebar
│   └── core/
│       ├── stacker.py       # Focus stacking orchestrator (algorithm selection)
│       ├── complex_wavelet.py # Complex Daubechies wavelet transform
│       ├── align.py         # ECC-based image alignment
│       ├── grayscale.py     # PCA-based grayscale conversion
│       └── reassign.py      # Color reassignment from grayscale
├── docs/
│   ├── PHASE1_MVP.md        # Completed MVP scope
│   ├── PHASE2_RETOUCHING.md # Current phase - source-frame retouching
│   └── ALGORITHMS.md        # Stacking algorithm details
└── tests/
```

## Current State

- **Stacking**: Two algorithms (Laplacian pyramid, Complex Daubechies wavelet with ECC alignment)
- **UI**: Side-by-side source/result panels with synchronized zoom/pan
- **Retouching**: Feathered brush with alignment-aware coordinate mapping
- **Substacks**: Stack subsets of frames, use as aligned paint sources
- **Undo/Redo**: Full stroke-level undo/redo for brush edits
- **Flash compare**: Hold S to see current paint source overlaid on result
- **Caching**: LRU cache for source images and substacks

## Key Patterns

### Image Handling
- Images stored as numpy arrays (BGR, uint8 - OpenCV convention)
- Display uses QPixmap via QGraphicsView
- `load_array(arr, preserve_zoom=True)` to update display without resetting view

### Brush/Paint System
- `ImageViewer` emits `brush_paint(x, y)` signal on paint
- `MainWindow` handles the actual pixel copying between source and result arrays
- Brush cursor is a QGraphicsEllipseItem in scene coordinates

### Zoom/Pan
- QGraphicsView handles transforms
- Both panels zoom together via shared slider
- Ctrl+/- and Ctrl+0 for keyboard zoom

## Development Priorities

1. **Bug fixes** - Undo edge cases, checkbox selection quirks (see docs/TODO.md)
2. **UI polish** - Focus handling for S key, app icon, general refinement
3. **Release prep** - End-user README, packaging for distribution
4. **Performance** - Large image handling (50MP+), lazy alignment for substacks

## Testing

Run with sample focus stack images. Look for:
- Halos at high-contrast edges
- Mush in fine detail areas
- Ghosting from moving subjects

These are what the retouching tools need to fix.

## Dependencies

- PySide6 (Qt bindings)
- OpenCV (cv2)
- NumPy

## Running

```bash
pip install -e .
focal
```
