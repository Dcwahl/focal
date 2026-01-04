"""LRU image cache with configurable memory budget.

Provides memory management for source images and future substack results.
Uses least-recently-used eviction when memory budget is exceeded.
"""

from collections import OrderedDict
from dataclasses import dataclass
import os
import threading

import numpy as np


# Default 4GB, configurable via environment variable
DEFAULT_CACHE_BYTES = int(os.environ.get("FOCAL_CACHE_MB", 4096)) * 1024**2


@dataclass(frozen=True)
class CacheKey:
    """Identifies a cached image.

    Attributes:
        kind: Type of cached item ("source", "substack")
        identifier: Unique identifier (path for sources, tuple for substacks)
    """

    kind: str
    identifier: str | tuple

    def __hash__(self) -> int:
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
    """LRU cache for images with memory budget.

    Thread-safe for concurrent access from UI and worker threads.

    Args:
        max_bytes: Maximum memory budget in bytes. Defaults to FOCAL_CACHE_MB
                   environment variable or 4GB.
    """

    def __init__(self, max_bytes: int = DEFAULT_CACHE_BYTES):
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> np.ndarray | None:
        """Get image from cache, returning None if not present.

        Accessing an item marks it as recently used.
        """
        with self._lock:
            if key in self._cache:
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
                "utilization": self._current_bytes / self._max_bytes
                if self._max_bytes
                else 0,
            }
