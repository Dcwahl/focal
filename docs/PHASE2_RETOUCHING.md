# Phase 2 - Source-Frame Retouching

The killer feature that separates professional stacking tools from open source.

## Why This Matters

Stacking algorithms are "good enough" across the board. The commercial products (Helicon Focus, Zerene Stacker) don't win on algorithm quality - they win on the retouching workflow that lets you fix artifacts after the stack.

Common artifacts that need manual fixing:
- Halos (bright/dark outlines at edges)
- Transparent foreground ghosting
- Stacking mush (loss of fine detail in problem areas)
- Hot pixel trails

## Features

### 1. Source Frame Selector
- Sidebar list shows all loaded images (click to select)
- Keyboard shortcuts to quickly scrub through frames (future: arrow keys)

Note: Originally planned a top-bar dropdown, but sidebar selection proved sufficient and less cluttered.

### 2. Flash Compare
- Hold a key to see currently-selected source frame
- Release to see result
- Quick way to spot differences and find best source

### 3. Pixel Brush
- Select brush size
- Paint from selected source onto result
- Pixels are directly copied (may show seams at edges)

### 4. Undo Stack
- Track all brush strokes
- Undo/redo (Ctrl+Z / Ctrl+Shift+Z)
- Non-destructive until final save

## UI Layout (Updated)

```
┌──────────────────────────────────────────────────────────────┐
│ [Open]                              [Source: ▼ image3.jpg]   │
├───────────────┬───────────────┬──────────────────────────────┤
│    Source     │    Result     │ image1.jpg                   │
│               │               │ image2.jpg                   │
│  (selected    │  (editable -  │ image3.jpg  ←                │
│   frame)      │   paint here) │ ...                          │
│               │               │                              │
│               │               │ Brush: ○ 10px                │
├───────────────┴───────────────┴──────────────────────────────┤
│ [Stack]                   [░░░░░░░░░░]  [Undo] [Redo] [Save] │
└──────────────────────────────────────────────────────────────┘
```

- Source selector in top bar (dropdown or slider)
- Result panel becomes editable canvas
- Brush size control in sidebar
- Undo/Redo buttons in bottom bar

## Technical Approach

### Brush Implementation
The simplest approach:
1. Track mouse position and button state on result canvas
2. On mouse drag with brush active:
   - Get circular region around cursor
   - Copy pixels from source image at same coordinates
   - Paste onto result
3. Store each stroke as an operation for undo

### Compositing Layer
Rather than modifying result directly:
1. Keep original stacked result as base layer
2. Maintain a separate "edits" layer (RGBA, starts transparent)
3. Each brush stroke writes to edits layer
4. Display = base layer composited with edits layer
5. Save = flatten and export

This allows non-destructive editing and simpler undo.

### Edge Blending (Future Enhancement)
For v1, just copy pixels directly. May show hard edges at brush boundary.

For v2, could add:
- Feathered brush edges (alpha falloff)
- "Detail brush" that adapts lightness/contrast rather than copying pixels
- Poisson blending for seamless compositing

## Technical Questions

Things to figure out during implementation:

1. **Performance**: How fast can we blit pixels during brush drag? May need to batch updates.

2. **Canvas interaction**: PySide6 QGraphicsScene or paint directly on QLabel? Graphics scene is more powerful but heavier.

3. **Coordinate mapping**: When zoomed/panned, need to map screen coords back to image coords correctly.

4. **Large images**: 50MP images at full resolution - need to work with scaled view but edit at full res.

## Definition of Done

- Can select any source frame from dropdown
- Can flash-compare source vs result (hold key)
- Can paint pixels from source onto result
- Can undo/redo brush strokes
- Can save edited result
- Tested with real artifacts that need fixing
