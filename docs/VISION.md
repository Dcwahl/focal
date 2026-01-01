# Focal - Vision

A straightforward focus stacking tool that does one thing well.

## Long-term Goals

### Core Stacking
- Multiple algorithm options (Laplacian, complex wavelets, etc.)
- Algorithm comparison/evaluation tooling
- Dehalo/artifact reduction post-processing

### Alignment
- Feature-based alignment for hand-held stacks
- Sub-pixel registration
- Auto-crop to common area

### Retouching (The Differentiator)
- Source-frame brushing: paint pixels from any source image onto result
- Flash-compare: toggle between source and result for quick comparison
- Substack editing: stack subset of frames, use result as brush source
- Undo/redo stack for non-destructive editing

### Workflow
- Batch processing (multiple stacks in queue)
- Metadata preservation (EXIF from source images)
- Project files (save/resume stacking sessions)

### Distribution
- Cross-platform builds (Windows, macOS, Linux)
- Single executable packaging

## Non-Goals
- Raw processing (use dedicated tools, feed us TIFFs/JPEGs)
- HDR/exposure blending (different problem)
- AI-based super resolution or enhancement
