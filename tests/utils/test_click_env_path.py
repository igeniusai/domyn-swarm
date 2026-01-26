import pathlib

from click.exceptions import BadParameter
import pytest

from domyn_swarm.utils.click_env_path import ClickEnvPath


def test_click_env_path_converts_valid_path(tmp_path):
    """Converts a valid path-like value to a pathlib.Path.

    Args:
        tmp_path: Temporary directory fixture.
    """
    param = ClickEnvPath()
    result = param.convert(str(tmp_path), None, None)
    assert isinstance(result, pathlib.Path)
    assert result == tmp_path


def test_click_env_path_invalid_value_raises():
    """Raises a Click error when the path is invalid."""
    param = ClickEnvPath()
    with pytest.raises(BadParameter):
        param.convert(object(), None, None)
