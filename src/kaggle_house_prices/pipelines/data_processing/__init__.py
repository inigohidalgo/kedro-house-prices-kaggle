"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

from .pipeline import create_pipeline
from .nodes import _is_true

__all__ = ["create_pipeline", "_is_true"]

__version__ = "0.0.1"
