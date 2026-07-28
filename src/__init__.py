"""PredictionRTS Package"""

from pathlib import Path
import logging

__version__ = "0.1.0"

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CACHE_DIR = ROOT_DIR / "cache"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_config():
    """Load config from YAML files."""
    import yaml

    config_path = CONFIG_DIR / "config.yaml"
    paths_path = CONFIG_DIR / "paths.yaml"

    config = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(yaml.safe_load(f) or {})

    if paths_path.exists():
        with open(paths_path, "r", encoding="utf-8") as f:
            config["paths"] = yaml.safe_load(f) or {}

    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
