"""
Data loading module for PredictionRTS
Provides data loading utilities and lazy-loaded global variables
"""

import json
import logging
import csv
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.config.base_config import map_id, data_id, params
from src import ROOT_DIR

logger = logging.getLogger(__name__)

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

FIG_PATH = ROOT_DIR / "output" / "figures"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR = ROOT_DIR / "data"


def load_bk_tree_from_file(file_path):
    """Load BKTree from file"""
    from src.structure.BKTree import BKTreeNode, BKTree

    def deserialize_node(node_data):
        state = {"state": [node_data["state"]]}
        node = BKTreeNode(state, node_data["cluster_id"])
        for dist, child_data in node_data["children"].items():
            child_node = deserialize_node(child_data)
            node.add_child(float(dist), child_node)
        return node

    try:
        with open(file_path, "r") as file:
            tree_data = json.load(file)
        bk_tree = BKTree()
        bk_tree.root = deserialize_node(tree_data)
        logger.debug(f"Loaded BKTree from {file_path}")
        return bk_tree
    except Exception as e:
        logger.error(f"Failed to load BKTree from {file_path}: {e}")
        raise


def load_all_bktrees(primary_bk_tree):
    """Load all secondary BK-Trees"""
    from src.structure.BKTree import get_max_cluster_id

    secondary_bk_trees = {}
    if primary_bk_tree is None or primary_bk_tree.root is None:
        return secondary_bk_trees

    logger.info("--- Primary BKTree Analysis Started ---")
    logger.info(f"Root cluster ID: {primary_bk_tree.root.cluster_id}")
    logger.info(f"Root children counts: {len(primary_bk_tree.root.children)}")

    cluster_count = get_max_cluster_id(primary_bk_tree)
    logger.info(f"Detected {cluster_count} clusters to load.")

    for cluster_id in range(1, cluster_count + 1):
        path = f"{secondary_bktree_prefix}_{cluster_id}.json"
        try:
            tree = load_bk_tree_from_file(path)
            secondary_bk_trees[cluster_id] = tree
            logger.debug(f"Loaded Secondary BKTree [{cluster_id}/{cluster_count}]")
        except FileNotFoundError:
            logger.warning(f"Secondary BKTree file missing: {path}")
        except Exception as e:
            logger.error(f"Unexpected error on cluster {cluster_id}: {e}")

    logger.info(f"All trees loaded. Total: {len(secondary_bk_trees)}")
    return secondary_bk_trees


def read_state_node_file(file_path):
    """Read state node file into dictionary"""
    state_node_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            key = eval(parts[0])
            id = int(parts[1])
            score = float(parts[2])
            state_node_dict[key] = {"id": id, "score": score}

    reverse_dict = {}
    state_value = []
    for key, value in state_node_dict.items():
        id = value["id"]
        score = value["score"]
        if id not in reverse_dict:
            reverse_dict[id] = {"cluster": key, "score": score}
            state_value.append(score)

    return state_node_dict, reverse_dict, state_value


def read_node_log_file(file_path):
    """Read node log file into list of lists"""
    result = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            parts = [int(part) for part in parts]
            result.append(parts)
    return result


def read_game_result_file(file_path):
    """Read game result file"""
    result = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            outcome = parts[0]
            steps = int(float(parts[1].strip("[]")))
            score = int(parts[2])
            penalty = int(parts[3])
            result.append([outcome, steps, score, penalty])
    return result


def read_action_csv(file_path):
    """Read action log CSV file"""
    result = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                result.append(row)
    return result


def create_action_dictionary(action_path):
    """Create action dictionary from directory"""
    action_dict = {}
    if not os.path.exists(action_path):
        return action_dict

    for filename in os.listdir(action_path):
        if filename.endswith(".csv"):
            action_name = filename.replace(".csv", "")
            action_dict[action_name] = len(action_dict)
    return action_dict


def calculate_and_save_distance_matrix(
    reverse_dict, custom_distance, secondary_bk_trees, distance_matrix_folder
):
    """Calculate and save distance matrix"""
    import numpy as np
    from src.utils.calculate_utils import calculate_distance_matrix

    if not os.path.exists(distance_matrix_folder):
        os.makedirs(distance_matrix_folder, exist_ok=True)

    state_distance_matrix_path = os.path.join(
        distance_matrix_folder, "state_distance_matrix.npy"
    )

    if os.path.exists(state_distance_matrix_path):
        logger.info(f"Cache hit. Loading distance matrix: {state_distance_matrix_path}")
        return np.load(state_distance_matrix_path)

    logger.warning("No cache hit. Starting fresh calculation...")
    distance_matrix = calculate_distance_matrix(
        reverse_dict, custom_distance, secondary_bk_trees
    )
    np.save(state_distance_matrix_path, distance_matrix)
    logger.info(f"Matrix saved to {state_distance_matrix_path}")
    return distance_matrix


def calculate_and_save_dtw_distance_matrix(
    state_log, state_distance_matrix, distance_matrix_folder
):
    """Calculate and save DTW distance matrix"""
    import numpy as np
    from src.utils.calculate_utils import calculate_dtw_distance_matrix

    if not os.path.exists(distance_matrix_folder):
        os.makedirs(distance_matrix_folder, exist_ok=True)

    log_distance_matrix_path = os.path.join(
        distance_matrix_folder, "log_distance_matrix.npy"
    )

    if os.path.exists(log_distance_matrix_path):
        logger.info(
            f"Cache hit. Loading DTW distance matrix from {log_distance_matrix_path}"
        )
        return np.load(log_distance_matrix_path)

    logger.warning("No DTW cache hit. Starting calculation...")
    dtw_matrix = calculate_dtw_distance_matrix(state_log, state_distance_matrix)
    np.save(log_distance_matrix_path, dtw_matrix)
    logger.info(f"DTW matrix saved to {log_distance_matrix_path}")
    return dtw_matrix


def generate_fitness_landscape(game_results, state_log_distance_matrix):
    """Generate fitness landscape from game results and distance matrix"""
    from src.structure.generate_FL import generate_fitness_landscape as gen_fl

    return gen_fl(game_results, state_log_distance_matrix)


def get_state_landscape(state_distance_matrix, state_value):
    """Get state landscape"""
    from src.structure.generate_FL import get_state_landscape as get_sl

    return get_sl(state_distance_matrix, state_value)


def preprocess_decision_transformer_data(state_log, action_log, r_log):
    """Preprocess data for Decision Transformer"""
    all_actions = sorted(list(set([a for sublist in action_log for a in sublist])))
    action_to_id = {act: i for i, act in enumerate(all_actions)}

    processed_states = []
    processed_actions = []
    processed_rtgs = []

    for s_raw, a_raw, r_raw in zip(state_log, action_log, r_log):
        s_aligned = s_raw[:-1]
        a_aligned = a_raw[:-1]
        r_aligned = r_raw[1:]

        a_ids = [action_to_id[act] for act in a_aligned]

        rtg_aligned = []
        current_val = 0
        for r in reversed(r_aligned):
            current_val += r
            rtg_aligned.append(current_val)
        rtg_aligned.reverse()

        processed_states.append(s_aligned)
        processed_actions.append(a_ids)
        processed_rtgs.append(rtg_aligned)

    return {
        "states": processed_states,
        "actions": processed_actions,
        "rtgs": processed_rtgs,
    }, action_to_id


def get_sampling_masks(log_fitness):
    """Get sampling masks based on fitness values"""
    import numpy as np

    data = np.array(log_fitness)

    max_val = np.max(data)
    best_mask = data == max_val

    top_5_threshold = np.percentile(data, 95)
    top_5_mask = data >= top_5_threshold

    min_val = np.min(data)
    worst_mask = data == min_val

    bottom_5_threshold = np.percentile(data, 5)
    bottom_5_mask = data <= bottom_5_threshold

    median_low = np.percentile(data, 47.5)
    median_high = np.percentile(data, 52.5)
    median_mask = (data >= median_low) & (data <= median_high)

    return {
        "best": best_mask,
        "top_5pct": top_5_mask,
        "worst": worst_mask,
        "bottom_5pct": bottom_5_mask,
        "median": median_mask,
    }


# Suffix generation
def _generate_suffix():
    abbr_map = {
        "mdl_spatial_prior": "sp",
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
    }

    suffix_parts = []
    for key, value in params.items():
        if key in abbr_map:
            if isinstance(value, bool):
                if value:
                    suffix_parts.append(abbr_map[key])

    return "_" + "_".join(suffix_parts) if suffix_parts else ""


suffix = _generate_suffix()

# Lazy-loaded global variables
_primary_bk_tree = None
_secondary_bk_trees = None
_state_node_dict = None
_reverse_dict = None
_state_value = None
_state_log = None
_state_distance_matrix = None
_state_log_distance_matrix = None
_game_results = None
_log_fitness = None
_action_dict = None
_action_log = None
_r_log = None
_dt_data = None
_action_vocab = None
_sampling_masks = None


def get_primary_bk_tree():
    global _primary_bk_tree
    if _primary_bk_tree is None:
        try:
            _primary_bk_tree = load_bk_tree_from_file(primary_bktree_path)
        except Exception as e:
            logger.warning(f"Could not load primary BK tree: {e}")
    return _primary_bk_tree


def get_secondary_bk_trees():
    global _secondary_bk_trees
    if _secondary_bk_trees is None:
        _secondary_bk_trees = load_all_bktrees(get_primary_bk_tree())
    return _secondary_bk_trees


def get_state_node_dict():
    global _state_node_dict, _reverse_dict, _state_value
    if _state_node_dict is None:
        try:
            _state_node_dict, _reverse_dict, _state_value = read_state_node_file(
                state_node_path
            )
        except Exception as e:
            logger.warning(f"Could not load state node dict: {e}")
            _state_node_dict, _reverse_dict, _state_value = {}, {}, []
    return _state_node_dict


def get_reverse_dict():
    global _reverse_dict
    if _reverse_dict is None:
        get_state_node_dict()
    return _reverse_dict


def get_state_value():
    global _state_value
    if _state_value is None:
        get_state_node_dict()
    return _state_value


def get_state_log():
    global _state_log
    if _state_log is None:
        try:
            _state_log = read_node_log_file(node_log_path)
        except Exception as e:
            logger.warning(f"Could not load state log: {e}")
            _state_log = []
    return _state_log


def get_state_distance_matrix():
    global _state_distance_matrix
    if _state_distance_matrix is None:
        from src.structure.BKTree import custom_distance

        _state_distance_matrix = calculate_and_save_distance_matrix(
            get_reverse_dict(),
            custom_distance,
            get_secondary_bk_trees(),
            distance_matrix_folder,
        )
    return _state_distance_matrix


def get_state_log_distance_matrix():
    global _state_log_distance_matrix
    if _state_log_distance_matrix is None:
        _state_log_distance_matrix = calculate_and_save_dtw_distance_matrix(
            get_state_log(), get_state_distance_matrix(), distance_matrix_folder
        )
    return _state_log_distance_matrix


def get_game_results():
    global _game_results
    if _game_results is None:
        try:
            _game_results = read_game_result_file(game_result_path)
        except Exception as e:
            logger.warning(f"Could not load game results: {e}")
            _game_results = []
    return _game_results


def get_log_fitness():
    global _log_fitness
    if _log_fitness is None:
        _log_fitness, _, _, _ = generate_fitness_landscape(
            get_game_results(), get_state_log_distance_matrix()
        )
    return _log_fitness


def get_action_dict():
    global _action_dict
    if _action_dict is None:
        _action_dict = create_action_dictionary(action_path)
    return _action_dict


def get_action_log():
    global _action_log
    if _action_log is None:
        try:
            _action_log = read_action_csv(action_log_path)
        except Exception as e:
            logger.warning(f"Could not load action log: {e}")
            _action_log = []
    return _action_log


def get_r_log():
    global _r_log
    if _r_log is None:
        # TODO: implement reward processing
        _r_log = []
    return _r_log


def get_dt_data():
    global _dt_data, _action_vocab
    if _dt_data is None:
        _dt_data, _action_vocab = preprocess_decision_transformer_data(
            get_state_log(), get_action_log(), get_r_log()
        )
    return _dt_data


def get_action_vocab():
    global _action_vocab
    if _action_vocab is None:
        get_dt_data()
    return _action_vocab


def get_sampling_masks():
    global _sampling_masks
    if _sampling_masks is None:
        _sampling_masks = get_sampling_masks(get_log_fitness())
    return _sampling_masks
