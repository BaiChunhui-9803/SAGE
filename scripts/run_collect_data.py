#!/usr/bin/env python
"""
Data collection script for PredictionRTS
Collects game data from StarCraft II environment
Usage:
    python scripts/run_collect_data.py map=sce-1 episodes=100
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.from_preconfig(ROOT_DIR / "configs")
    except ImportError:
        from src import get_config

        cfg = get_config()

    set_seed(cfg.get("seed", 42))

    try:
        from env import starcraft

        logger.info("Starting StarCraft II data collection...")
        starcraft.main_run()
    except ImportError as e:
        logger.error(f"Failed to import StarCraft II modules: {e}")
        logger.error("Make sure pysc2 is installed: pip install pysc2")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        sys.exit(1)

    logger.info("Data collection completed.")


if __name__ == "__main__":
    main()
