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
│       └── stacker.py       # Focus stacking algorithm (Laplacian pyramid)
├── docs/
│   ├── PHASE1_MVP.md        # Completed MVP scope
│   └── PHASE2_RETOUCHING.md # Current phase - source-frame retouching
└── tests/
```

## Current State

- **Stacking**: Working Laplacian pyramid stacking
- **UI**: Side-by-side source/result panels with zoom/pan
- **Retouching**: Basic pixel brush (copies from source to result)
- **Flash compare**: Hold S to see source overlaid on result

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

1. **Undo/redo** - Critical for usable retouching
2. **Substack workflow** - Stack subset of frames, use as paint source
3. **Detail brush** - Adaptive blending instead of raw pixel copy
4. **Performance** - Large image handling (50MP+)

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
