# Phase 4: Memory Management - Stage 1: LRU Cache

This document specifies the memory management strategy for Focal, designed to support the upcoming substack workflow without excessive memory consumption.

## Problem Statement

The current architecture loads all images into memory:
- Source images cached in `MainWindow.source_arrays` (`main_window.py:60`)
- Result + edited result held as full numpy arrays (`main_window.py:58-59`)
- During stacking, all frames loaded simultaneously (`stacker.py:68-75`, `stacker.py:137-145`)

For the substack workflow, this compounds:
- Original N source frames
- Main stack result + edited copy
- M substack results (each a full-resolution image)
- Each substack needs source access for retouching

**Memory math (24MP images, ~72MB each):**
| Scenario | Sources | Results | Total |
|----------|---------|---------|-------|
| 10 frames, no substacks | 720MB | 144MB | ~860MB |
| 30 frames, no substacks | 2.1GB | 144MB | ~2.3GB |
| 30 frames, 3 substacks | 2.1GB | 576MB | ~2.7GB |
| 50 frames @ 50MP, 3 substacks | 7.5GB | 900MB | ~8.4GB |

The last scenario would crash on 16GB machines when accounting for OS and application overhead.

---

## Options Considered

### Option 1: Simple LRU Cache (Selected)

Set a configurable memory budget. Keep sources and results in a unified cache with least-recently-used eviction. When cache is full, evict oldest items; recompute substacks on-demand if accessed again.

**Pros:**
- Self-balancing: small stacks stay fully cached, large stacks evict automatically
- Simple mental model for users
- Single mechanism handles sources and results uniformly

**Cons:**
- Evicted substacks require recomputation (~5-15s depending on size)
- Need to track memory usage accurately

### Option 2: Tiered Storage

Keep thumbnails/previews in RAM, full-resolution on disk. Load full-res on demand with ~50-100ms latency.

**Pros:**
- Handles arbitrarily large stacks
- Predictable latency (disk read vs recompute)

**Cons:**
- More complex implementation
- Requires temp file management
- SSD assumed for acceptable performance

### Option 3: Configurable Memory Modes

Expose settings: "Performance" (keep all in RAM), "Balanced" (LRU), "Memory Saver" (aggressive eviction).

**Pros:**
- Users choose their tradeoff
- Power users can opt into full RAM usage

**Cons:**
- UI complexity for settings
- More code paths to test
- Users may not know which mode suits them

### Decision

**Start with Option 1 (LRU Cache).** It solves the problem with minimal complexity. Option 3's configurability can be added later if users report specific needs. Option 2 is overkill for Stage 1.

---

## Stage 1 Specification: LRU Image Cache

### Core Data Structure

Create a new module `src/focal/core/image_cache.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
from collections import OrderedDict
import threading

@dataclass(frozen=True)
class CacheKey:
    """Identifies a cached image."""
    kind: str  # "source", "result", "substack"
    identifier: str | int | tuple  # path for sources, id for results

    def __hash__(self):
        return hash((self.kind, str(self.identifier)))

@dataclass
class CacheEntry:
    """A cached image with metadata."""
    key: CacheKey
    array: np.ndarray
    size_bytes: int

    @classmethod
    def from_array(cls, key: CacheKey, array: np.ndarray) -> "CacheEntry":
        return cls(key=key, array=array, size_bytes=array.nbytes)

class ImageCache:
    """LRU cache for images with memory budget."""

    def __init__(self, max_bytes: int = 4 * 1024**3):  # 4GB default
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> np.ndarray | None:
        """Get image from cache, returning None if not present."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key].array
            return None

    def put(self, key: CacheKey, array: np.ndarray) -> None:
        """Add image to cache, evicting LRU items if needed."""
        entry = CacheEntry.from_array(key, array)

        with self._lock:
            # If already cached, remove old entry first
            if key in self._cache:
                self._current_bytes -= self._cache[key].size_bytes
                del self._cache[key]

            # Evict until we have room
            while self._current_bytes + entry.size_bytes > self._max_bytes:
                if not self._cache:
                    break  # Can't evict anything, just store anyway
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._current_bytes -= oldest_entry.size_bytes

            self._cache[key] = entry
            self._current_bytes += entry.size_bytes

    def invalidate(self, key: CacheKey) -> None:
        """Remove a specific item from cache."""
        with self._lock:
            if key in self._cache:
                self._current_bytes -= self._cache[key].size_bytes
                del self._cache[key]

    def invalidate_by_kind(self, kind: str) -> None:
        """Remove all items of a specific kind (e.g., all substacks)."""
        with self._lock:
            to_remove = [k for k in self._cache if k.kind == kind]
            for key in to_remove:
                self._current_bytes -= self._cache[key].size_bytes
                del self._cache[key]

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "items": len(self._cache),
                "bytes_used": self._current_bytes,
                "bytes_max": self._max_bytes,
                "utilization": self._current_bytes / self._max_bytes if self._max_bytes else 0,
            }
```

### Cache Key Design

| Kind | Identifier | Example |
|------|------------|---------|
| `"source"` | File path as string | `CacheKey("source", "/path/to/img001.jpg")` |
| `"result"` | `"main"` or substack id | `CacheKey("result", "main")` |
| `"substack"` | Tuple of frame indices | `CacheKey("substack", (0, 2, 4))` |

Using tuples for substack identifiers allows automatic deduplication: if user creates the same substack twice, we recognize it.

### Integration Points

#### 1. Replace `MainWindow.source_arrays`

Current code (`main_window.py:393-400`):
```python
def _get_source_array(self, index: int) -> np.ndarray | None:
    """Get source image as numpy array, caching for performance."""
    if index not in self.source_arrays:
        if 0 <= index < len(self.images):
            img = cv2.imread(str(self.images[index]))
            if img is not None:
                self.source_arrays[index] = img
    return self.source_arrays.get(index)
```

Replace with:
```python
def _get_source_array(self, index: int) -> np.ndarray | None:
    """Get source image, loading from disk if not cached."""
    if not (0 <= index < len(self.images)):
        return None

    path = self.images[index]
    key = CacheKey("source", str(path))

    # Try cache first
    cached = self._cache.get(key)
    if cached is not None:
        return cached

    # Load from disk and cache
    img = cv2.imread(str(path))
    if img is not None:
        self._cache.put(key, img)
    return img
```

#### 2. Main Result Handling

The main `result_image` and `edited_result` are kept as direct references (not in the cache) since they're always needed while active. No changes needed to `_on_stack_finished()` for Stage 1.

#### 3. Cache Substack Results (Future)

When substacks are implemented:
```python
def _on_substack_finished(self, substack_id: tuple[int, ...], result: np.ndarray):
    key = CacheKey("substack", substack_id)
    self._cache.put(key, result)
    # Store reference for UI
    self.substacks[substack_id] = key
```

To retrieve (with recomputation fallback):
```python
def _get_substack(self, substack_id: tuple[int, ...]) -> np.ndarray:
    key = CacheKey("substack", substack_id)
    cached = self._cache.get(key)
    if cached is not None:
        return cached

    # Recompute - this is the cost of eviction
    result = self._compute_substack(substack_id)
    self._cache.put(key, result)
    return result
```

#### 4. Clear Cache on New Project

When loading new images (`main_window.py:253-263`):
```python
def _load_image_files(self, files: list[Path]):
    self.images = sorted(files)

    # Clear cache - new project, old data irrelevant
    self._cache.clear()

    # ... rest of validation and setup
```

### Memory Budget Selection

Default: **4GB**

Rationale:
- 16GB machine with 4GB for OS/apps leaves ~12GB
- 4GB cache + application overhead + headroom = reasonable
- Supports ~55 frames at 24MP, or ~25 frames at 50MP

Consider exposing via environment variable for power users:
```python
import os
DEFAULT_CACHE_BYTES = int(os.environ.get("FOCAL_CACHE_MB", 4096)) * 1024**2
```

### Handling Result/Edited Result

The `result_image` and `edited_result` arrays are special:
- Always needed for display and editing
- Should not be evicted while active

**Approach:** Keep these as direct references (current behavior), separate from cache. Only cache substack results, which can be recomputed.

```python
class MainWindow:
    def __init__(self):
        # Direct references - not evictable
        self.result_image: np.ndarray | None = None
        self.edited_result: np.ndarray | None = None

        # LRU cache for sources and substacks
        self._cache = ImageCache(max_bytes=DEFAULT_CACHE_BYTES)
```

### Thread Safety

The `ImageCache` uses a lock for thread safety because:
- `StackWorker` runs in a background thread (`main_window.py:29-48`)
- UI thread accesses cache for display
- Future: multiple substack workers could run concurrently

The lock is coarse-grained (entire operation). This is acceptable because:
- Cache operations are fast (dict lookup/insert)
- Actual image loading happens outside the lock
- Contention is rare (UI reads, worker writes)

### Eviction Callbacks (Optional Enhancement)

For future UI feedback ("Substack X was evicted, will recompute if needed"):
```python
class ImageCache:
    def __init__(self, ..., on_evict: Callable[[CacheKey], None] | None = None):
        self._on_evict = on_evict

    def _evict_one(self) -> bool:
        if not self._cache:
            return False
        oldest_key, oldest_entry = self._cache.popitem(last=False)
        self._current_bytes -= oldest_entry.size_bytes
        if self._on_evict:
            self._on_evict(oldest_key)
        return True
```

---

## Testing Strategy

### Unit Tests for ImageCache

```python
def test_cache_basic_operations():
    cache = ImageCache(max_bytes=1024**2)  # 1MB

    key = CacheKey("source", "test.jpg")
    arr = np.zeros((100, 100, 3), dtype=np.uint8)  # 30KB

    assert cache.get(key) is None
    cache.put(key, arr)
    assert cache.get(key) is not None
    assert np.array_equal(cache.get(key), arr)

def test_cache_eviction():
    cache = ImageCache(max_bytes=100_000)  # 100KB

    # Each ~30KB, so 4 should trigger eviction
    for i in range(5):
        key = CacheKey("source", f"test{i}.jpg")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        cache.put(key, arr)

    # First one should be evicted
    assert cache.get(CacheKey("source", "test0.jpg")) is None
    # Last ones should remain
    assert cache.get(CacheKey("source", "test4.jpg")) is not None

def test_cache_lru_behavior():
    cache = ImageCache(max_bytes=100_000)

    # Add 3 items
    for i in range(3):
        cache.put(CacheKey("source", f"test{i}.jpg"),
                  np.zeros((100, 100, 3), dtype=np.uint8))

    # Access first one (makes it recently used)
    cache.get(CacheKey("source", "test0.jpg"))

    # Add more to trigger eviction
    for i in range(3, 6):
        cache.put(CacheKey("source", f"test{i}.jpg"),
                  np.zeros((100, 100, 3), dtype=np.uint8))

    # test0 should survive (was accessed), test1 should be evicted
    assert cache.get(CacheKey("source", "test0.jpg")) is not None
    assert cache.get(CacheKey("source", "test1.jpg")) is None
```

### Integration Tests

1. Load 20 source frames, verify memory stays under budget
2. Switch between sources rapidly, verify no cache misses for recently-used
3. Create substack, evict it by filling cache, access again, verify recomputation works

---

## Migration Path

### Step 1: Add ImageCache module
Create `src/focal/core/image_cache.py` with the cache implementation.

### Step 2: Integrate into MainWindow
- Add `self._cache = ImageCache()` in `__init__`
- Replace `self.source_arrays` dict with cache calls
- Update `_get_source_array()` method
- Update `_load_image_files()` to clear cache
- Remove `self.source_arrays` entirely

### Step 3: Prepare for substack caching (Future)
- Main result kept as direct reference (not cached)
- Substack results will use cache when implemented

### Step 4: Tests
- Add unit tests for ImageCache
- Add integration tests for cache behavior in MainWindow

---

## Future Stages

### Stage 2: Disk-Backed Cache (Option 2)
If users report issues with recomputation latency, add disk tier:
- Write evicted substacks to temp files
- Load from disk instead of recomputing
- ~100ms load vs ~10s recompute

### Stage 3: Configurable Modes (Option 3)
If different user segments emerge with different needs:
- Settings panel with mode selection
- Presets: "Performance", "Balanced", "Memory Saver"
- Custom budget slider

---

## Open Questions

1. **Should we show cache stats in UI?** (Memory usage indicator, eviction warnings)
2. **Should substack recomputation be async with progress indicator?**
3. **Should we pin certain sources?** (e.g., "favorite" frames that never evict)

These can be addressed during implementation or deferred to user feedback.
