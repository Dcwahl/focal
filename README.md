# Focal

Focus stacking application for macro and micro photography.

<!-- TODO: Add hero screenshot/gif showing side-by-side retouching workflow -->

## Features

### Two Stacking Algorithms

| Algorithm | Speed | Quality | Alignment | Best For |
|-----------|-------|---------|-----------|----------|
| Laplacian Pyramid | Fast | Good | None | Tripod shots, quick previews |
| Complex Daubechies Wavelet | Slower | Better | ECC-based | Handheld, final output |

### Source-Frame Retouching

Focal lets you paint pixels from any source frame directly onto your result to fix these issues. You can also select multiple source frames and create a substack as your source.

- **Side-by-side view** — See source and result simultaneously with synchronized zoom/pan
- **Flash compare** — Hold S to overlay the current source on the result
- **Feathered brush** — Soft edges blend naturally into surrounding pixels
- **Alignment-aware** — Brush coordinates automatically compensate for frame alignment
- **Substacks** — Stack a subset of frames to use as an aligned paint source
- **Full undo/redo** — Every brush stroke can be undone

## Installation

### Download (macOS)

Download the latest `.dmg` from the [Releases](../../releases) page. Open the DMG and drag Focal to your Applications folder.

**Note:** The app is unsigned. On first launch, right-click the app and select "Open", then click "Open" in the dialog.

### Windows

Coming soon.

### From Source

Requires Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/Dcwahl/focal.git
cd focal

# Install with pip
pip install -e .

# Run
focal
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run focal
```

## Usage

1. **Open** — Click Open or drag a folder of images onto the window
2. **Stack** — Select an algorithm and click Stack
3. **Retouch** — Select a source frame from the sidebar, then paint on the result to fix artifacts
4. **Save** — Export as TIFF or JPEG

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `S` (hold) | Flash compare — show source overlaid on result |
| `Ctrl/Cmd + Z` | Undo brush stroke |
| `Ctrl/Cmd + Shift + Z` | Redo brush stroke |
| `Ctrl/Cmd + +` | Zoom in |
| `Ctrl/Cmd + -` | Zoom out |
| `Ctrl/Cmd + 0` | Reset zoom |

### Tips

- **Start with Laplacian** for a quick preview, then switch to Complex Wavelet for final output
- **Look for halos** at high-contrast edges—these are the most common artifacts to retouch
- **Use substacks** when you need aligned pixels from multiple frames blended together
- **Zoom in** to paint fine details, zoom out to check your work in context

## Algorithm Details

See [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for technical details on the stacking algorithms.

## License

MIT License. See [LICENSE](LICENSE) for details.
