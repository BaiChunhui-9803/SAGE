"""Config module for PredictionRTS"""

from .base_config import map_id, data_id, N, K, model_conf, model_name, params, suffix
from src import get_config

__all__ = [
    "map_id",
    "data_id",
    "N",
    "K",
    "model_conf",
    "model_name",
    "params",
    "suffix",
    "get_config",
]
