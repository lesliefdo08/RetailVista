"""Backward-compatible imports for legacy code paths.

Use app.core.preprocess for all new logic.
"""

from app.core.preprocess import create_preprocessor, get_feature_columns, prepare_features, validate_input_schema

__all__ = [
    "create_preprocessor",
    "get_feature_columns",
    "prepare_features",
    "validate_input_schema",
]
