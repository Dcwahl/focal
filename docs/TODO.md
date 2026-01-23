List of documented issues that should be addressed

## Bugs

## Small Improvements
- Drag-and-drop folder/files onto window to load images
- App icon - add rounded corners / transparency for proper macOS icon mask
- Other UI polish
- maybe do this if do website; https://wisp.place/
- ~~Create DMG installer with drag-to-Applications background~~ done, see Building section
- Windows build (PyInstaller should work, need to test on Windows machine, create .ico icon)
- Homebrew formula for easier Mac distribution
- Permit workflow where can load source images and also load result image (in case want to circle back and do more retouching, or import result from different tool). Might require some thinking vis-a-vis alignment

## Pre-release Checklist
- [x] Fix remaining bugs above
- [ ] Test on variety of image sets (macro, portrait, landscape)
- [x] README for end users (installation, usage)
- [x] Package for distribution (pyinstaller) - see Building section below

## Things to look into
- Does this work out of the box for non-macro shots? E.g. portrait shots
- This guy JUST released a similar thing; https://www.youtube.com/watch?v=PMPBrCNLOr0
    - he has some interesting features that might be worth taking a look at

## Completed ✓
- [x] Alignment for brush painting (source frames and substacks now align to result coordinates)
- [x] Undo/redo for brush strokes
- [x] Substack workflow (create, select, paint from)
- [x] LRU cache for images
- [x] Complex Daubechies wavelet stacking


## Things to think about after release
- Performance
TODO fill this out as think of things

---

## Building for Distribution

### Prerequisites
Dev dependencies include pyinstaller and pyobjc (for macOS menu bar fix):
```bash
uv sync --dev
```

### Build the app
```bash
uv run pyinstaller focal.spec
```

Output:
- `dist/Focal.app` - macOS app bundle (~325MB)
- `dist/Focal/` - unpacked version

### Create distributable zip
```bash
cd dist && zip -r Focal-0.1.0-macOS.zip Focal.app
```

### Create DMG (recommended)
Requires `brew install create-dmg`:
```bash
create-dmg \
  --volname "Focal" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "Focal.app" 150 185 \
  --app-drop-link 450 185 \
  "dist/Focal-0.1.0-macOS.dmg" \
  "dist/Focal.app"
```
This creates a ~121MB DMG with the standard drag-to-Applications layout.

### Create a GitHub Release
```bash
# Create release and upload dmg in one command
gh release create v0.1.0 dist/Focal-0.1.0-macOS.dmg --title "v0.1.0" --notes "Initial release"
```

Or via web UI:
1. Go to repo → Releases → "Create a new release"
2. Create a tag (e.g., `v0.1.0`)
3. Drag `.dmg` into "Attach binaries by dropping them here" area (at the bottom, NOT the description box)
4. Publish

### Notes
- The app is unsigned, so users will need to right-click → Open → "Open Anyway" on first launch
- To update the icon: replace `assets/icons/focal-icon-1.png`, then run:
  ```bash
  # Regenerate .icns from PNG (requires macOS)
  mkdir -p assets/icons/focal-icon.iconset
  cd assets/icons
  sips -z 16 16 focal-icon-1.png --out focal-icon.iconset/icon_16x16.png
  sips -z 32 32 focal-icon-1.png --out focal-icon.iconset/icon_16x16@2x.png
  sips -z 32 32 focal-icon-1.png --out focal-icon.iconset/icon_32x32.png
  sips -z 64 64 focal-icon-1.png --out focal-icon.iconset/icon_32x32@2x.png
  sips -z 128 128 focal-icon-1.png --out focal-icon.iconset/icon_128x128.png
  sips -z 256 256 focal-icon-1.png --out focal-icon.iconset/icon_128x128@2x.png
  sips -z 256 256 focal-icon-1.png --out focal-icon.iconset/icon_256x256.png
  sips -z 512 512 focal-icon-1.png --out focal-icon.iconset/icon_256x256@2x.png
  sips -z 512 512 focal-icon-1.png --out focal-icon.iconset/icon_512x512.png
  sips -z 1024 1024 focal-icon-1.png --out focal-icon.iconset/icon_512x512@2x.png
  iconutil -c icns focal-icon.iconset -o focal-icon.icns
  rm -rf focal-icon.iconset
  ```
- Rebuild after code changes by running `uv run pyinstaller focal.spec` again