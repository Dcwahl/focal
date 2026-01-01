# Phase 1 - MVP

Minimal viable focus stacking app. Get the full loop working before adding features.

## Scope

### UI Layout
```
┌─────────────────────────────────────────────────────┐
│ [Open]                                    (top bar) │
├───────────────────────────────────┬─────────────────┤
│                                   │ image1.jpg      │
│                                   │ image2.jpg      │
│      (image preview area)         │ image3.jpg      │
│                                   │ ...             │
│                                   │ (click to view) │
├───────────────────────────────────┴─────────────────┤
│ [Stack]                    [████████░░] 80%  [Save] │
└─────────────────────────────────────────────────────┘
```

### Features
1. **Open** - Folder picker or multi-file select, loads images
2. **Image list** - Right sidebar, shows filenames, click to preview
3. **Preview area** - Shows selected source image, or result after stacking
4. **Stack button** - Runs Laplacian pyramid stacking
5. **Progress bar** - Visual feedback during stacking
6. **Save** - Export result as TIFF or JPEG

### Technical Decisions
- PySide6 for UI
- Laplacian pyramid algorithm only (proven in pyfocusstack)
- No alignment (tripod-only assumption)
- Single-threaded stacking with progress callbacks

### Out of Scope for Phase 1
- Algorithm selection
- Parameter tuning
- Alignment
- Batch processing
- Dehalo post-processing (add in phase 2)

## Definition of Done
- Can open a folder of images
- Can preview individual source images
- Can run stacking with progress feedback
- Can save the result
- Tested end-to-end with a real focus stack
