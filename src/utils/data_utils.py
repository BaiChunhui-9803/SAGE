"""
Data utilities for different context window training
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


def prepare_dt_data_with_context(
    dt_data: Dict,
    r_log: List[List[float]],
    context_window: int = 20,
    prediction_steps: List[int] = [1],
    min_episode_length: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Prepare DT training data with specific context window

    Args:
        dt_data: Dict with 'states', 'actions', 'rtgs'
        r_log: Reward log for Q-value computation
        context_window: Maximum context window size
        prediction_steps: List of prediction horizons to train
        min_episode_length: Minimum episode length to include

    Returns:
        samples: List of training samples
        stats: Statistics about the prepared data
    """
    if min_episode_length is None:
        min_episode_length = context_window + max(prediction_steps)

    samples = []
    stats = {
        "total_episodes": len(dt_data["states"]),
        "valid_episodes": 0,
        "total_samples": 0,
        "samples_by_pred_step": {k: 0 for k in prediction_steps},
        "episode_lengths": [],
        "action_distribution": Counter(),
    }

    for episode_idx in range(len(dt_data["states"])):
        states = dt_data["states"][episode_idx]
        actions = dt_data["actions"][episode_idx]
        rtgs = dt_data["rtgs"][episode_idx]
        rewards = r_log[episode_idx] if episode_idx < len(r_log) else []

        stats["episode_lengths"].append(len(states))

        # Skip short episodes
        if len(states) < min_episode_length:
            continue

        stats["valid_episodes"] += 1

        # Generate samples for each time step
        for t in range(len(states)):
            # Context: states[0:t], actions[0:t]
            context_len = min(t, context_window)

            if context_len == 0:
                continue

            # For each prediction horizon
            for pred_k in prediction_steps:
                if t + pred_k > len(actions):
                    continue

                # Target: actions[t:t+pred_k]
                target_actions = actions[t : t + pred_k]

                # Compute Q-value (cumulative reward from t onwards)
                if rewards:
                    q_value = sum(rewards[t:])
                else:
                    q_value = 0

                sample = {
                    "episode_idx": episode_idx,
                    "timestep": t,
                    "context_window": context_window,
                    "actual_context_len": context_len,
                    "history_states": states[max(0, t - context_window) : t],
                    "history_actions": actions[max(0, t - context_window) : t],
                    "history_rtgs": rtgs[max(0, t - context_window) : t],
                    "target_actions": target_actions,
                    "prediction_horizon": pred_k,
                    "q_value": q_value,
                }

                samples.append(sample)
                stats["total_samples"] += 1
                stats["samples_by_pred_step"][pred_k] += 1

                # Track action distribution
                for a in target_actions:
                    stats["action_distribution"][a] += 1

    return samples, stats


def filter_episodes_by_length(
    dt_data: Dict,
    r_log: List[List[float]],
    min_length: int,
    max_length: Optional[int] = None,
) -> Tuple[Dict, List[List[float]]]:
    """
    Filter episodes by length

    Args:
        dt_data: Original data
        r_log: Reward log
        min_length: Minimum episode length
        max_length: Maximum episode length (optional)

    Returns:
        filtered_data: Filtered dt_data
        filtered_r_log: Filtered reward log
    """
    filtered_data = {
        "states": [],
        "actions": [],
        "rtgs": [],
    }
    filtered_r_log = []

    for i in range(len(dt_data["states"])):
        length = len(dt_data["states"][i])

        if length < min_length:
            continue
        if max_length is not None and length > max_length:
            continue

        filtered_data["states"].append(dt_data["states"][i])
        filtered_data["actions"].append(dt_data["actions"][i])
        filtered_data["rtgs"].append(dt_data["rtgs"][i])
        if i < len(r_log):
            filtered_r_log.append(r_log[i])

    return filtered_data, filtered_r_log


def split_by_history_length(
    dt_data: Dict, r_log: List[List[float]], context_windows: List[int] = [5, 10, 20]
) -> Dict[int, Tuple[Dict, List[List[float]]]]:
    """
    Split data by suitable context window

    Args:
        dt_data: Original data
        r_log: Reward log
        context_windows: List of context window sizes

    Returns:
        splits: Dict mapping context_window -> (filtered_data, filtered_r_log)
    """
    splits = {}

    for ctx in sorted(context_windows):
        min_length = ctx + 1  # Need at least ctx history + 1 target
        filtered_data, filtered_r_log = filter_episodes_by_length(
            dt_data, r_log, min_length=min_length
        )
        splits[ctx] = (filtered_data, filtered_r_log)

    return splits


def compute_data_statistics(dt_data: Dict, r_log: List[List[float]]) -> Dict:
    """
    Compute comprehensive data statistics

    Args:
        dt_data: DT data
        r_log: Reward log

    Returns:
        stats: Statistics dictionary
    """
    episode_lengths = [len(s) for s in dt_data["states"]]

    # Action distribution
    all_actions = [a for ep in dt_data["actions"] for a in ep]
    action_dist = Counter(all_actions)

    # RTG distribution
    all_rtgs = [sum(ep) for ep in r_log]

    # Unique states
    all_states = [s for ep in dt_data["states"] for s in ep]
    unique_states = len(set(all_states))

    return {
        "num_episodes": len(dt_data["states"]),
        "episode_length": {
            "min": min(episode_lengths),
            "max": max(episode_lengths),
            "mean": np.mean(episode_lengths),
            "std": np.std(episode_lengths),
        },
        "action_distribution": dict(action_dist),
        "action_vocab_size": len(action_dist),
        "unique_states": unique_states,
        "total_state_occurrences": len(all_states),
        "rtg": {
            "min": min(all_rtgs) if all_rtgs else 0,
            "max": max(all_rtgs) if all_rtgs else 0,
            "mean": np.mean(all_rtgs) if all_rtgs else 0,
            "std": np.std(all_rtgs) if all_rtgs else 0,
        },
        "episodes_by_length": {
            ">=5": sum(1 for l in episode_lengths if l >= 5),
            ">=10": sum(1 for l in episode_lengths if l >= 10),
            ">=15": sum(1 for l in episode_lengths if l >= 15),
            ">=20": sum(1 for l in episode_lengths if l >= 20),
            ">=25": sum(1 for l in episode_lengths if l >= 25),
        },
    }


def print_data_statistics(stats: Dict):
    """Print formatted data statistics"""
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)

    print(f"\nEpisodes: {stats['num_episodes']}")

    print(f"\nEpisode Lengths:")
    print(f"  Min: {stats['episode_length']['min']}")
    print(f"  Max: {stats['episode_length']['max']}")
    print(f"  Mean: {stats['episode_length']['mean']:.1f}")
    print(f"  Std: {stats['episode_length']['std']:.1f}")

    print(f"\nEpisodes by Length:")
    for threshold, count in stats["episodes_by_length"].items():
        pct = count / stats["num_episodes"] * 100
        print(f"  {threshold}: {count} ({pct:.1f}%)")

    print(f"\nActions:")
    print(f"  Vocab Size: {stats['action_vocab_size']}")

    print(f"\nStates:")
    print(f"  Unique: {stats['unique_states']}")
    print(f"  Total Occurrences: {stats['total_state_occurrences']}")
    print(
        f"  Uniqueness: {stats['unique_states'] / stats['total_state_occurrences'] * 100:.2f}%"
    )

    print(f"\nRTG (Return-to-Go):")
    print(f"  Min: {stats['rtg']['min']:.2f}")
    print(f"  Max: {stats['rtg']['max']:.2f}")
    print(f"  Mean: {stats['rtg']['mean']:.2f}")
    print(f"  Std: {stats['rtg']['std']:.2f}")

    print("=" * 60)


def create_multi_step_targets(
    actions: List[int], prediction_steps: List[int]
) -> Dict[int, List[int]]:
    """
    Create multi-step prediction targets

    Args:
        actions: Full action sequence
        prediction_steps: List of prediction horizons

    Returns:
        targets: Dict mapping horizon -> target actions
    """
    targets = {}

    for k in prediction_steps:
        if len(actions) >= k:
            targets[k] = actions[:k]

    return targets
