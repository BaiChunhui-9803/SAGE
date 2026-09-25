import csv

import numpy as np
from src.config.base_config import *
from src.data.global_variable import *


def load_distance_matrix(file_path):
    """Load a NumPy distance matrix from file_path."""
    return np.load(file_path)


def save_distance_matrix(matrix, file_path):
    """Save the distance matrix to file_path in NumPy format."""
    np.save(file_path, matrix)


def generate_suffix(params_dict):
    # Map parameter names to filename abbreviations.
    # Additional parameters can use entries such as 'learning_rate': 'lr'.
    state_seq_ABBR_MAP = {
        "mdl_spatial_prior": "sp",
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
        "N": N,
        "K": K,
        # 'use_batch_norm': 'bn',
        # 'dropout_rate': 'dr'
    }

    dt_ABBR_MAP = {
        "mdl_init_embedding_freeze": "embedF",
        "mdl_init_embedding_train": "embedT",
    }

    if model_name == "decisionTransformer":
        ABBR_MAP = dt_ABBR_MAP
    else:
        ABBR_MAP = state_seq_ABBR_MAP

    suffix_parts = []

    for key, value in params_dict.items():
        if key in ABBR_MAP:
            abbr = ABBR_MAP[key]
            # True flags use only the abbreviation; other values include their value.
            if isinstance(value, bool):
                if value:
                    suffix_parts.append(abbr)
            else:
                suffix_parts.append(f"{key}{value}")

    # Join components with underscores and prepend an underscore.
    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def create_action_dictionary(action_path):
    """Read action column names from the first action CSV and map letters starting at a to those names."""
    import os
    import pandas as pd

    try:
        # List action CSV files.
        if os.path.exists(action_path):
            csv_files = [f for f in os.listdir(action_path) if f.endswith(".csv")]
            if csv_files:
                # Read the first CSV file.
                first_csv = csv_files[0]
                csv_path = os.path.join(action_path, first_csv)

                # Read the CSV file.
                df = pd.read_csv(csv_path)

                # Exclude unnamed index columns from action names.
                action_columns = [col for col in df.columns if col != "Unnamed: 0"]

                # Map consecutive letters to action names.
                action_dict = {}
                for i, action_name in enumerate(action_columns):
                    key = chr(ord("a") + i)
                    action_dict[key] = action_name

                print(f"Loaded {len(action_dict)} actions from CSV: {first_csv}")
                return action_dict

        # Fall back to the default action dictionary if CSV loading fails.
        print(
            "Warning: Could not load actions from CSV, using default action dictionary"
        )
        default_action_dict = {
            "a": "action_ATK_nearest",
            "b": "action_ATK_clu_nearest",
            "c": "action_ATK_nearest_weakest",
            "d": "action_ATK_clu_nearest_weakest",
            "e": "action_ATK_threatening",
            "f": "action_DEF_clu_nearest",
            "g": "action_MIX_gather",
            "h": "action_MIX_lure",
            "i": "action_MIX_sacrifice_lure",
            "j": "do_randomly",
            "k": "do_nothing",
        }
        return default_action_dict

    except Exception as e:
        print(f"Error loading action dictionary: {e}")
        print("Using default action dictionary")
        default_action_dict = {
            "a": "action_ATK_nearest",
            "b": "action_ATK_clu_nearest",
            "c": "action_ATK_nearest_weakest",
            "d": "action_ATK_clu_nearest_weakest",
            "e": "action_ATK_threatening",
            "f": "action_DEF_clu_nearest",
            "g": "action_MIX_gather",
            "h": "action_MIX_lure",
            "i": "action_MIX_sacrifice_lure",
            "j": "do_randomly",
            "k": "do_nothing",
        }
        return default_action_dict


def preprocess_decision_transformer_data(state_log, action_log, r_log):
    """Align nested state, action, and reward sequences. Return processed state/action/return-to-go records and the action-string-to-ID vocabulary."""
    # 1. Build the action vocabulary.
    # Collect unique actions across trajectories.
    all_actions = sorted(list(set([a for sublist in action_log for a in sublist])))
    action_to_id = {act: i for i, act in enumerate(all_actions)}

    processed_states = []
    processed_actions = []
    processed_rtgs = []

    for s_raw, a_raw, r_raw in zip(state_log, action_log, r_log):
        # --- Align states, actions, and rewards ---
        # s: Drop the final state.
        # a: Drop the invalid final action.
        # r: Drop the initial placeholder reward.
        s_aligned = s_raw[:-1]
        a_aligned = a_raw[:-1]
        r_aligned = r_raw[1:]

        # Ensure the three sequences have equal length.
        assert len(s_aligned) == len(a_aligned) == len(r_aligned)

        # --- Encode actions as IDs ---
        a_ids = [action_to_id[act] for act in a_aligned]

        # --- Compute return-to-go ---
        rtg_aligned = []
        current_val = 0
        for r in reversed(r_aligned):
            current_val += r
            rtg_aligned.append(current_val)
        rtg_aligned.reverse()  # Restore chronological order.

        # Append the processed trajectory.
        processed_states.append(s_aligned)
        processed_actions.append(a_ids)
        processed_rtgs.append(rtg_aligned)

    return {
        "states": processed_states,
        "actions": processed_actions,
        "rtgs": processed_rtgs,
    }, action_to_id


def get_sampling_masks(log_fitness):
    data = np.array(log_fitness)

    # 1. Mask the global maximum.
    max_val = np.max(data)
    best_mask = data == max_val

    # 2. Mask the top five percent.
    top_5_threshold = np.percentile(data, 95)
    top_5_mask = data >= top_5_threshold

    # 3. Mask the global minimum.
    min_val = np.min(data)
    worst_mask = data == min_val

    # 4. Mask the bottom five percent.
    bottom_5_threshold = np.percentile(data, 5)
    bottom_5_mask = data <= bottom_5_threshold

    # 5. Mask the central five percent around the median.
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
