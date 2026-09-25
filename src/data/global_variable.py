"""Global variables for PredictionRTS data module"""

from src.config.base_config import map_id, data_id, params
from src import ROOT_DIR

# Data paths
DATA_ROOT = str(ROOT_DIR / "data")

primary_bktree_path = f"{DATA_ROOT}/{map_id}/{data_id}/bktree/primary_bktree.json"
secondary_bktree_prefix = f"{DATA_ROOT}/{map_id}/{data_id}/bktree/secondary_bktree"
state_node_path = f"{DATA_ROOT}/{map_id}/{data_id}/graph/state_node.txt"
node_log_path = f"{DATA_ROOT}/{map_id}/{data_id}/graph/node_log.txt"
game_result_path = f"{DATA_ROOT}/{map_id}/{data_id}/game_result.txt"
action_log_path = f"{DATA_ROOT}/{map_id}/{data_id}/action_log.csv"
action_path = f"{DATA_ROOT}/{map_id}/{data_id}/sub_q_table/"
episode_result_path = f"{DATA_ROOT}/{map_id}/{data_id}/sub_episode/"
distance_matrix_folder = f"{DATA_ROOT}/{map_id}/{data_id}/distance/"

# Output paths
FIG_PATH = ROOT_DIR / "output" / "figures"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"


# Generate suffix from params
def _generate_suffix():
    abbr_map = {
        "mdl_spatial_prior": "sp",
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
        "N": f"N{params.get('N', 10)}",
        "K": f"K{params.get('K', 5)}",
    }

    suffix_parts = []
    for key, value in params.items():
        if key in abbr_map:
            if isinstance(value, bool):
                if value:
                    suffix_parts.append(abbr_map[key])
            else:
                pass

    return "_" + "_".join(suffix_parts) if suffix_parts else ""


suffix = _generate_suffix()
