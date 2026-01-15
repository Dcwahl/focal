# Phase 5: Substack Workflow

**Status: ✅ Implemented** (January 2026)

This document specifies the substack feature - the ability to stack a subset of source frames and use the result as a paint source during retouching.

## Implementation Notes

The substack workflow is fully implemented with an important enhancement beyond the original spec:

**Alignment-aware brush painting:** Both source frames and substacks are aligned to the main result coordinate space. When painting from any source:
1. The brush position in result space is transformed to source space using the inverse alignment matrix
2. Pixels are sampled from the correctly aligned location in the source
3. This eliminates misalignment artifacts when painting, especially for images with focus breathing

Key implementation details:
- `stacker.py` stores per-frame transforms in `last_transforms` during stacking
- `main_window.py` computes substack-to-result alignment after substack creation via `_compute_substack_alignment()`
- `_apply_brush_stroke()` uses `invert_transform()` to map result coordinates → source coordinates
- Laplacian algorithm doesn't do alignment, so falls back to same-coordinate sampling

## Why Substacks

The main stack algorithm makes per-pixel decisions across all frames. Sometimes this produces artifacts:
- Halos at high-contrast edges
- Ghosting from subject movement
- "Mush" where the algorithm blended poorly

The fix: stack just the 3-5 frames that are clean in the problem area, then paint from that substack onto the main result.

## Alternative Workflows (Future Consideration)

The design below focuses on **artifact repair** - the primary use case. Two other workflows may be valuable later:

### Focus Zone Isolation
Stack "foreground" frames (1-5) separately from "background" frames (6-10). Use each substack to paint its respective region. Useful for images with distinct depth planes.

**What would change:** Might want better organization (named substacks, grouping). Current design doesn't preclude this, but auto-naming like "Frames 1-5" may be less useful than "Foreground" / "Background".

### Experimentation
Try different frame combinations to find what produces the best result in a problem area. Keep multiple substacks as options.

**What would change:** Might want comparison tools (A/B toggle between substacks). Current design supports having multiple substacks; comparison UI would be additive.

---

## Design Decisions

### Frame Selection

**Approach:** Checkboxes in the source frame sidebar, with shift/ctrl-click accelerators.

- Each source frame gets a checkbox
- Shift-click selects a range
- Ctrl-click toggles individual frames
- Selection state is visual (checkbox checked)

**Rationale:** Familiar UI pattern from file managers. Flexible enough to select arbitrary frames or contiguous ranges.

### Substack Creation

**Approach:** Explicit "Create Substack" button, blocking execution.

Flow:
1. User selects frames via checkboxes
2. User clicks "Create Substack" button
3. Stacking runs synchronously with progress indicator
4. Substack appears in substacks list
5. Checkboxes clear for next selection

**Rationale:**
- Blocking is acceptable because substacks are small (3-5 frames) and stack quickly (~1-2 seconds)
- Explicit button prevents accidental substack creation
- Clearing checkboxes after creation keeps UI clean

**Button placement:** Candidates:
- Bottom of sidebar near frame list
- Toolbar alongside brush controls **[selected]**
- Right-click context menu on selection

### Substack Display

**Approach:** Vertical list in collapsible "Substacks" section below source frames.

```
┌─────────────────────┐
│ Sources             │
│ ☐ frame001.jpg      │
│ ☐ frame002.jpg      │
│ ☐ frame003.jpg      │
│ ...                 │
├─────────────────────┤
│ Substacks           │
│   Frames 1-3    [x] │
│   Frames 5-7    [x] │
└─────────────────────┘
```

- Auto-named from frame range (e.g., "Frames 3-7")
- Delete button (X) on each entry
- Clicking a substack selects it as paint source
- Current paint source (frame or substack) is highlighted

**Rationale:**
- Same sidebar keeps all paint sources in one place
- Vertical list is simple; can switch to horizontal chips later if cramped
- Auto-naming is sufficient for artifact repair workflow; user-naming adds friction

### Paint Source Switching

**Approach:** Unified selection in sidebar - clicking a source frame OR a substack sets it as the current paint source.

- Only one paint source active at a time
- Visual highlight shows current selection
- Flash compare (S key) works with substacks too

**Rationale:** Consistent with existing source frame workflow. No new UI patterns to learn.

### Substack Lifecycle

**Persistence:** Session only. Substacks are lost when application closes.

**Deletion:**
- Explicit delete via X button on substack entry
- Right-click context menu (optional, implementation detail)
- All substacks cleared when loading new project

**Rationale:** Persistence adds project file complexity. Substacks are quick to recreate. Can add persistence later if users request it.

### Cache Integration

Substacks use the existing LRU cache (`image_cache.py`):

```python
# On substack creation
key = CacheKey("substack", (3, 4, 5, 6, 7))  # tuple of frame indices
self._cache.put(key, substack_result)

# On substack access (for painting)
cached = self._cache.get(key)
if cached is None:
    # Recompute - silent, blocking, ~1-2s
    cached = self._recompute_substack(frame_indices)
    self._cache.put(key, cached)
```

**Eviction behavior:** Silent recompute. If a substack gets evicted from cache and user tries to paint from it, we restack transparently. Brief pause (~1-2s) but no error or warning.

**Rationale:** Substacks are small and fast to recompute. Visible eviction state adds UI complexity for a rare edge case.

---

## Implementation Guidance

### Data Structures

```python
@dataclass
class Substack:
    """A stacked subset of source frames."""
    frame_indices: tuple[int, ...]  # Immutable tuple for cache key
    display_name: str  # e.g., "Frames 3-7"

    @property
    def cache_key(self) -> CacheKey:
        return CacheKey("substack", self.frame_indices)
```

In `MainWindow`:
```python
self.substacks: list[Substack] = []  # Created substacks
self.current_paint_source: int | Substack | None = None  # Frame index or Substack
```

### UI Components

**Frame checkboxes:** Modify `ImageList` widget to support checkboxes.
- Add `QCheckBox` to each list item
- Track selection state: `self.selected_frames: set[int]`
- Emit signal on selection change for button enable/disable

**Substacks section:** Add a new `QListWidget` or custom widget below frame list.
- Substack entries with name and delete button
- Click to select as paint source
- Visual distinction from source frames (indent, icon, or styling)

**Create Substack button:**
- Enabled only when 2+ frames selected
- Triggers blocking stack operation
- Clears selection on completion

### Integration Points

**`_get_source_array()` modification:**

Current code handles frame indices. Extend to handle substacks:

```python
def _get_paint_source_array(self) -> np.ndarray | None:
    """Get current paint source (frame or substack)."""
    if isinstance(self.current_paint_source, Substack):
        return self._get_substack_array(self.current_paint_source)
    elif isinstance(self.current_paint_source, int):
        return self._get_source_array(self.current_paint_source)
    return None

def _get_substack_array(self, substack: Substack) -> np.ndarray | None:
    """Get substack result, recomputing if evicted from cache."""
    cached = self._cache.get(substack.cache_key)
    if cached is not None:
        return cached

    # Recompute
    result = self._compute_substack(substack.frame_indices)
    if result is not None:
        self._cache.put(substack.cache_key, result)
    return result
```

**`_compute_substack()` implementation:**

```python
def _compute_substack(self, frame_indices: tuple[int, ...]) -> np.ndarray | None:
    """Stack a subset of frames."""
    paths = [self.images[i] for i in frame_indices]
    # Reuse existing stacker, blocking call
    return self.stacker.stack(paths, progress_callback=None)
```

**Brush painting modification:**

Update `on_brush_paint()` to use `_get_paint_source_array()` instead of `_get_source_array(self.current_source_index)`.

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| 0 frames selected, click Create | Button disabled |
| 1 frame selected, click Create | Button disabled (or allow? just returns that frame) |
| Duplicate substack (same frames) | Allow - cache deduplicates automatically via key |
| Delete substack while painting | Switch to no paint source, or fall back to first source frame |
| All substacks deleted | Substacks section shows empty or collapses |

### Testing Strategy

**Unit tests:**
- `Substack` dataclass creation and cache key generation
- Frame selection state management

**Integration tests:**
- Create substack from selected frames, verify appears in list
- Paint from substack, verify correct pixels copied
- Delete substack, verify removed from list
- Cache eviction and recomputation

**Manual testing:**
- Create substack, paint from it, undo, redo
- Flash compare with substack selected
- Switch between frame and substack paint sources
- Large stack (30+ frames), create multiple substacks, verify memory reasonable

---

## Migration / Rollout

This feature is additive - no changes to existing workflows required.

1. Add checkbox support to `ImageList`
2. Add substacks list UI
3. Add "Create Substack" button and logic
4. Extend paint source to support substacks
5. Wire up delete functionality

Can be implemented incrementally and tested at each step.

---

## Open Questions

1. **Button placement** - Toolbar vs sidebar? Decide during implementation based on visual balance.
   → **Decision:** Toolbar.

2. **1-frame substack** - Allow or require 2+? Allowing is harmless (just returns that frame), but pointless. Probably disable button for < 2.
   → **Decision:** Require 2+. Disable button for < 2 frames selected.

3. **Progress indication during substack creation** - Cursor change? Mini progress bar? Brief spinner? Keep simple.
   → **Decision:** Mini progress bar.

4. **Substack in flash compare** - When substack is selected, S key shows... what? The substack result makes sense. Confirm this works naturally with existing code.
   → **Decision:** Shows the substack result. Confirm works with existing code.
