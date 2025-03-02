"""Tests for the jorbit.utils.cache module."""
from jorbit.utils.cache import view_jorbit_cache


def test_view_jorbit_cache() -> None:
    """Test the view_jorbit_cache function."""
    _ = view_jorbit_cache()

# for now not testing the cache clearing function, want to leave CI caches in-place
# download_file_wrapper is tested elsewhere
