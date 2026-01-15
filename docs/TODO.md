List of documented issues that should be addressed

## Bugs
- Undo doesn't seem to work right. Hold down right click -> paint for a bit -> ctrl z -> not everything goes away
- Substack selection interactions in the sidebar still aren't exactly right (vis-a-vis the checkboxes, mainly)
    - Checkboxes don't actually work correctly (need to highlight them in order for the button to work)

## Small Improvements
- S key doesn't always work on main window (need to click on the window first before can do S)
- App icon
- Other UI polish

## Pre-release Checklist
- [ ] Fix remaining bugs above
- [ ] Test on variety of image sets (macro, portrait, landscape)
- [ ] README for end users (installation, usage)
- [ ] Package for distribution (pyinstaller or similar)

## Things to look into
- Does this work out of the box for non-macro shots? E.g. portrait shots

## Completed ✓
- [x] Alignment for brush painting (source frames and substacks now align to result coordinates)
- [x] Undo/redo for brush strokes
- [x] Substack workflow (create, select, paint from)
- [x] LRU cache for images
- [x] Complex Daubechies wavelet stacking
