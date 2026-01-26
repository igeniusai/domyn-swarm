# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any


def _column_names_from_collect_schema(data: Any) -> list[str] | None:
    """Try to read column names via a `collect_schema()` method.

    This avoids triggering expensive schema resolution on objects like polars LazyFrame.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    collect_schema = getattr(data, "collect_schema", None)
    if not callable(collect_schema):
        return None
    try:
        schema = collect_schema()
        names = getattr(schema, "names", None)
        if callable(names):
            names = names()
        if names is None:
            return None
        if isinstance(names, list):
            return names
        if isinstance(names, tuple):
            return list(names)
        return None
    except Exception:
        return None


def _column_names_from_columns_attr(data: Any) -> list[str] | None:
    """Try to read column names from a `.columns` attribute.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    if not hasattr(data, "columns"):
        return None
    try:
        return list(data.columns)
    except TypeError:
        return None


def _column_names_from_schema(data: Any) -> list[str] | None:
    """Try to read column names from a `.schema()` method.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    if not hasattr(data, "schema"):
        return None
    try:
        schema = data.schema()
    except Exception:
        return None
    names: list[str] | None = getattr(schema, "names", None)
    if callable(names):
        names = names()
    if names is None:
        try:
            names = [field.name for field in schema]
        except TypeError:
            names = []
    return names


def _get_column_names(data: Any) -> list[str] | None:
    """Resolve column names from a dataset-like object.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    return (
        _column_names_from_collect_schema(data)
        or _column_names_from_columns_attr(data)
        or _column_names_from_schema(data)
    )


def _require_column_names(data: Any) -> list[str]:
    """Resolve column names or raise if unavailable.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names.

    Raises:
        ValueError: If column names cannot be resolved.
    """
    names = _get_column_names(data)
    if names is None:
        raise ValueError(f"Cannot validate id column on data type: {type(data)!r}")
    return names


def _validate_required_id(data: Any, id_col: str) -> None:
    """Validate that the input data contains the required id column.

    Args:
        data: Input dataset or DataFrame.
        id_col: Required column name.

    Raises:
        ValueError: If the column is missing or cannot be validated.
    """
    names = _require_column_names(data)
    if id_col not in names:
        raise ValueError(f"Input data missing required id column '{id_col}'.")
