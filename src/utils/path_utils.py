"""Path utilities for PredictionRTS"""

from pathlib import Path
from typing import Dict, Any, Optional

from src import ROOT_DIR


def get_data_paths(cfg) -> Dict[str, Path]:
    """Get data paths from configuration."""
    paths_config = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    data_root = Path(
        paths_config.get(
            "data_root", "D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree"
        )
    )
    map_id = paths_config.get("map_id", "MarineMicro_MvsM_4")
    data_id = paths_config.get("data_id", "6")

    base_path = data_root / map_id / data_id

    return {
        "primary_bktree": base_path / "bktree" / "primary_bktree.json",
        "secondary_bktree_prefix": str(base_path / "bktree" / "secondary_bktree"),
        "state_node": base_path / "graph" / "state_node.txt",
        "node_log": base_path / "graph" / "node_log.txt",
        "game_result": base_path / "game_result.txt",
        "action_log": base_path / "action_log.csv",
        "action_path": str(base_path / "sub_q_table/"),
        "episode_result_path": str(base_path / "sub_episode/"),
        "distance_matrix_folder": str(base_path / "distance/"),
    }


def get_output_paths(cfg) -> Dict[str, Path]:
    """Get output paths from configuration."""
    paths_config = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    output_dir = Path(paths_config.get("output_dir", "output"))
    cache_dir = Path(paths_config.get("cache_dir", "cache"))

    return {
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
        "models": cache_dir / "model",
        "npy": cache_dir / "npy",
    }


def generate_suffix(params: Dict) -> str:
    """Generate suffix from parameters."""
    state_seq_abbr_map = {
        "mdl_spatial_prior": "sp",
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
    }

    dt_abbr_map = {
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
    }

    model_name = params.get("model_name", "decisionTransformer")

    if model_name == "decisionTransformer":
        abbr_map = dt_abbr_map
    else:
        abbr_map = state_seq_abbr_map

    suffix_parts = []

    for key, value in params.items():
        if key in abbr_map:
            abbr = abbr_map[key]
            if isinstance(value, bool):
                if value:
                    suffix_parts.append(abbr)
            else:
                suffix_parts.append(f"{key}{value}")

    return "_" + "_".join(suffix_parts) if suffix_parts else ""
