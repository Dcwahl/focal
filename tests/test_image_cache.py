"""Unit tests for the ImageCache class."""

import numpy as np
import pytest

from focal.core.image_cache import ImageCache, CacheKey, CacheEntry


class TestCacheKey:
    """Tests for CacheKey dataclass."""

    def test_source_key(self):
        """Source keys use path as identifier."""
        key = CacheKey("source", "/path/to/img.jpg")
        assert key.kind == "source"
        assert key.identifier == "/path/to/img.jpg"

    def test_substack_key(self):
        """Substack keys use tuple of indices."""
        key = CacheKey("substack", (0, 2, 4))
        assert key.kind == "substack"
        assert key.identifier == (0, 2, 4)

    def test_hash_equality(self):
        """Equal keys should have equal hashes."""
        key1 = CacheKey("source", "/path/to/img.jpg")
        key2 = CacheKey("source", "/path/to/img.jpg")
        assert hash(key1) == hash(key2)
        assert key1 == key2

    def test_hash_inequality(self):
        """Different keys should have different hashes."""
        key1 = CacheKey("source", "/path/to/img1.jpg")
        key2 = CacheKey("source", "/path/to/img2.jpg")
        assert key1 != key2


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_from_array(self):
        """Create entry from numpy array."""
        key = CacheKey("source", "test.jpg")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        entry = CacheEntry.from_array(key, arr)

        assert entry.key == key
        assert np.array_equal(entry.array, arr)
        assert entry.size_bytes == arr.nbytes


class TestImageCacheBasic:
    """Tests for basic ImageCache operations."""

    def test_get_missing_returns_none(self):
        """Getting a missing key returns None."""
        cache = ImageCache(max_bytes=1024**2)
        key = CacheKey("source", "test.jpg")
        assert cache.get(key) is None

    def test_put_and_get(self):
        """Put then get returns the same array."""
        cache = ImageCache(max_bytes=1024**2)
        key = CacheKey("source", "test.jpg")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        cache.put(key, arr)
        result = cache.get(key)

        assert result is not None
        assert np.array_equal(result, arr)

    def test_put_replaces_existing(self):
        """Putting same key replaces the value."""
        cache = ImageCache(max_bytes=1024**2)
        key = CacheKey("source", "test.jpg")
        arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
        arr2 = np.ones((100, 100, 3), dtype=np.uint8)

        cache.put(key, arr1)
        cache.put(key, arr2)
        result = cache.get(key)

        assert np.array_equal(result, arr2)

    def test_invalidate(self):
        """Invalidate removes a specific key."""
        cache = ImageCache(max_bytes=1024**2)
        key = CacheKey("source", "test.jpg")
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        cache.put(key, arr)
        cache.invalidate(key)

        assert cache.get(key) is None

    def test_invalidate_missing_key_no_error(self):
        """Invalidating a missing key doesn't raise."""
        cache = ImageCache(max_bytes=1024**2)
        key = CacheKey("source", "test.jpg")
        cache.invalidate(key)  # Should not raise

    def test_clear(self):
        """Clear removes all entries."""
        cache = ImageCache(max_bytes=1024**2)
        for i in range(5):
            key = CacheKey("source", f"test{i}.jpg")
            arr = np.zeros((10, 10, 3), dtype=np.uint8)
            cache.put(key, arr)

        cache.clear()

        for i in range(5):
            key = CacheKey("source", f"test{i}.jpg")
            assert cache.get(key) is None


class TestImageCacheEviction:
    """Tests for LRU eviction behavior."""

    def test_evicts_oldest(self):
        """When full, evicts least recently used."""
        cache = ImageCache(max_bytes=100_000)  # ~100KB

        # Each ~30KB, so 4 should trigger eviction
        for i in range(5):
            key = CacheKey("source", f"test{i}.jpg")
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            cache.put(key, arr)

        # First one should be evicted
        assert cache.get(CacheKey("source", "test0.jpg")) is None
        # Last ones should remain
        assert cache.get(CacheKey("source", "test4.jpg")) is not None

    def test_lru_access_prevents_eviction(self):
        """Accessing an item moves it to end, preventing eviction."""
        # Each array is 30KB. Budget of 130KB fits ~4 items.
        cache = ImageCache(max_bytes=130_000)

        # Add 3 items (~90KB total, under budget)
        for i in range(3):
            cache.put(
                CacheKey("source", f"test{i}.jpg"),
                np.zeros((100, 100, 3), dtype=np.uint8),
            )

        # Access first one (makes it recently used)
        # Order now: test1, test2, test0
        cache.get(CacheKey("source", "test0.jpg"))

        # Add 2 more to trigger eviction
        # test3: 120KB total, under budget, no eviction
        # test4: 150KB > 130KB, evicts test1 (oldest)
        for i in range(3, 5):
            cache.put(
                CacheKey("source", f"test{i}.jpg"),
                np.zeros((100, 100, 3), dtype=np.uint8),
            )

        # test0 should survive (was accessed), test1 should be evicted
        assert cache.get(CacheKey("source", "test0.jpg")) is not None
        assert cache.get(CacheKey("source", "test1.jpg")) is None

    def test_large_item_evicts_multiple(self):
        """A large item can evict multiple smaller ones."""
        cache = ImageCache(max_bytes=100_000)

        # Add 3 small items (~30KB each = ~90KB)
        for i in range(3):
            cache.put(
                CacheKey("source", f"small{i}.jpg"),
                np.zeros((100, 100, 3), dtype=np.uint8),
            )

        # Add one large item (~60KB) - should evict 2 small ones
        cache.put(
            CacheKey("source", "large.jpg"),
            np.zeros((200, 100, 3), dtype=np.uint8),
        )

        # Large item should be present
        assert cache.get(CacheKey("source", "large.jpg")) is not None
        # At least the first small item should be evicted
        assert cache.get(CacheKey("source", "small0.jpg")) is None


class TestImageCacheInvalidateByKind:
    """Tests for invalidate_by_kind method."""

    def test_invalidate_by_kind(self):
        """Invalidate removes all items of a specific kind."""
        cache = ImageCache(max_bytes=1024**2)

        # Add sources and substacks
        for i in range(3):
            cache.put(
                CacheKey("source", f"test{i}.jpg"),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
            cache.put(
                CacheKey("substack", (i,)),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )

        # Invalidate all substacks
        cache.invalidate_by_kind("substack")

        # Sources should remain
        for i in range(3):
            assert cache.get(CacheKey("source", f"test{i}.jpg")) is not None
        # Substacks should be gone
        for i in range(3):
            assert cache.get(CacheKey("substack", (i,))) is None


class TestImageCacheStats:
    """Tests for cache statistics."""

    def test_stats_empty(self):
        """Empty cache has zero usage."""
        cache = ImageCache(max_bytes=1024**2)
        stats = cache.stats

        assert stats["items"] == 0
        assert stats["bytes_used"] == 0
        assert stats["bytes_max"] == 1024**2
        assert stats["utilization"] == 0.0

    def test_stats_with_items(self):
        """Stats reflect cached items."""
        cache = ImageCache(max_bytes=1024**2)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        cache.put(CacheKey("source", "test.jpg"), arr)

        stats = cache.stats

        assert stats["items"] == 1
        assert stats["bytes_used"] == arr.nbytes
        assert stats["utilization"] == arr.nbytes / (1024**2)

    def test_stats_after_eviction(self):
        """Stats decrease after eviction."""
        cache = ImageCache(max_bytes=100_000)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        for i in range(5):
            cache.put(CacheKey("source", f"test{i}.jpg"), arr)

        stats = cache.stats
        # Should have evicted some items to stay under budget
        assert stats["bytes_used"] <= stats["bytes_max"]
