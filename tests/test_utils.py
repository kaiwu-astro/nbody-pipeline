"""
Tests for nbody_pipeline.utils module
"""

import importlib.util
import logging

import pytest

from nbody_pipeline.utils import save, read, get_output, can_convert_to_float, log_time


def _check_colour_available():
    """Check if colour-science package is available"""
    try:
        return importlib.util.find_spec("colour.colorimetry") is not None
    except ModuleNotFoundError:
        return False


class TestSerialization:
    """Tests for serialization functions"""

    def test_save_and_read_pkl(self, temp_dir):
        """Test saving and reading pickle files"""
        data = [1, 2, 3, "test", {"key": "value"}]
        filepath = temp_dir / "test.pkl"

        save(str(filepath), data)
        loaded = read(str(filepath))

        assert loaded == data
        assert filepath.exists()

    def test_save_and_read_with_dirname(self, temp_dir):
        """Test saving with directory and filename"""
        data = ["a", "b", "c"]

        save(str(temp_dir), data, fname="custom.pkl")
        loaded = read(str(temp_dir), fname="custom.pkl")

        assert loaded == data

    def test_save_and_read_gz(self, temp_dir):
        """Test saving and reading gzipped pickle files"""
        data = [1, 2, 3]
        filepath = temp_dir / "test.pkl.gz"

        save(str(filepath), data)
        loaded = read(str(filepath))

        assert loaded == data


class TestShell:
    """Tests for shell utilities"""

    def test_get_output_simple(self):
        """Test getting output from simple command"""
        result = get_output("echo 'hello'")
        assert result == ["hello"]

    def test_get_output_multiline(self):
        """Test getting output with multiple lines"""
        # Use a command that definitely outputs 3 lines
        result = get_output("echo line1; echo line2; echo line3")
        assert len(result) >= 3

    def test_can_convert_to_float_valid(self):
        """Test can_convert_to_float with valid inputs"""
        assert can_convert_to_float("123")
        assert can_convert_to_float("123.456")
        assert can_convert_to_float("-123.456")
        assert can_convert_to_float("1e5")

    def test_can_convert_to_float_invalid(self):
        """Test can_convert_to_float with invalid inputs"""
        assert not can_convert_to_float("abc")
        assert not can_convert_to_float("123abc")
        assert not can_convert_to_float("")
        assert not can_convert_to_float("last")


class TestLogging:
    """Tests for logging decorator"""

    def test_log_time_decorator(self):
        """Test log_time decorator"""
        logger = logging.getLogger("test")

        @log_time(logger)
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

    def test_log_time_with_args(self):
        """Test log_time decorator with function arguments"""
        logger = logging.getLogger("test")

        @log_time(logger)
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5


class TestColorConverter:
    """Tests for BlackbodyColorConverter"""

    def test_import_color_converter(self):
        """Test that BlackbodyColorConverter can be imported"""
        from nbody_pipeline.utils import BlackbodyColorConverter

        assert BlackbodyColorConverter is not None

    @pytest.mark.skipif(
        not _check_colour_available(), reason="colour-science package not available"
    )
    def test_color_converter_init(self, temp_dir):
        """Test BlackbodyColorConverter initialization"""
        from nbody_pipeline.utils import BlackbodyColorConverter

        # Use temp directory for cache
        cache_path = temp_dir / "rgb_cache.pkl"
        converter = BlackbodyColorConverter(cache_path=str(cache_path))

        assert converter.r_interp is not None
        assert converter.g_interp is not None
        assert converter.b_interp is not None
