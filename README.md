# Focal

Focus stacking application for macro/micro photography.

## Features

- **Two stacking algorithms**:
  - Laplacian Pyramid - fast, good for tripod shots
  - Complex Daubechies Wavelet - higher quality, with alignment
- **Source-frame retouching** - paint from source frames to fix artifacts
- **Side-by-side comparison** - view source and result simultaneously

## Installation

```bash
pip install -e .
```

## Usage

```bash
focal
```

## Algorithms

See `docs/ALGORITHMS.md` for detailed algorithm documentation.

| Algorithm | Speed | Quality | Alignment |
|-----------|-------|---------|-----------|
| Laplacian Pyramid | Fast | Good | No |
| Complex Daubechies | Slower | Better | Yes (ECC) |

## Development

- `docs/PHASE1_MVP.md` - MVP scope (completed)
- `docs/PHASE2_RETOUCHING.md` - Current phase
- `docs/ALGORITHMS.md` - Algorithm details
