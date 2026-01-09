"""Unit tests for the Substack dataclass."""

import pytest

from focal.ui.substack_list import Substack
from focal.core.image_cache import CacheKey


class TestSubstack:
    """Tests for Substack dataclass."""

    def test_create_basic(self):
        """Create a basic substack."""
        substack = Substack(
            frame_indices=(0, 1, 2),
            display_name="Frames 1-3"
        )
        assert substack.frame_indices == (0, 1, 2)
        assert substack.display_name == "Frames 1-3"

    def test_cache_key(self):
        """Cache key uses frame indices tuple."""
        substack = Substack(
            frame_indices=(3, 4, 5),
            display_name="Frames 4-6"
        )
        key = substack.cache_key
        assert isinstance(key, CacheKey)
        assert key.kind == "substack"
        assert key.identifier == (3, 4, 5)

    def test_cache_key_equality(self):
        """Same frame indices produce equal cache keys."""
        s1 = Substack(frame_indices=(0, 1), display_name="A")
        s2 = Substack(frame_indices=(0, 1), display_name="B")
        assert s1.cache_key == s2.cache_key

    def test_cache_key_inequality(self):
        """Different frame indices produce different cache keys."""
        s1 = Substack(frame_indices=(0, 1), display_name="A")
        s2 = Substack(frame_indices=(1, 2), display_name="A")
        assert s1.cache_key != s2.cache_key


class TestSubstackDisplayName:
    """Tests for Substack.create_display_name static method."""

    def test_empty_indices(self):
        """Empty indices returns 'Empty'."""
        assert Substack.create_display_name(()) == "Empty"

    def test_single_frame(self):
        """Single frame shows 1-based index."""
        assert Substack.create_display_name((0,)) == "Frame 1"
        assert Substack.create_display_name((5,)) == "Frame 6"

    def test_consecutive_range(self):
        """Consecutive frames show as range."""
        assert Substack.create_display_name((0, 1, 2)) == "Frames 1-3"
        assert Substack.create_display_name((4, 5, 6, 7)) == "Frames 5-8"

    def test_non_consecutive_few(self):
        """Non-consecutive frames (3 or fewer) show all."""
        assert Substack.create_display_name((0, 2)) == "Frames 1, 3"
        assert Substack.create_display_name((0, 2, 5)) == "Frames 1, 3, 6"

    def test_non_consecutive_many(self):
        """Non-consecutive frames (4+) show truncated."""
        result = Substack.create_display_name((0, 2, 5, 8))
        assert "1, 3" in result
        assert "4 total" in result

    def test_unsorted_indices(self):
        """Unsorted indices are handled correctly."""
        # Should sort and recognize as consecutive
        assert Substack.create_display_name((2, 0, 1)) == "Frames 1-3"
