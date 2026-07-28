#!/usr/bin/env python
"""
Build Decision Knowledge Graph

This script builds the decision knowledge graph from raw episode data.
Two versions are created:
1. Simple (state-only)
2. Context-aware (state + history)

Also builds state transition networks for each KG.

Usage:
    python scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_4 --data-id 6
    python scripts/build_knowledge_graph.py --map-id MarineMicro_MvsM_4 --data-id 6 --context-windows 0 5 10
    python scripts/build_knowledge_graph.py --output-dir cache/knowledge_graph/CustomMap

Author: PredictionRTS Team
Date: 2026-03-21
"""

import sys
import os
import csv
import pickle
import argparse
from typing import List, Dict, Optional, Set, Tuple
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_config, ROOT_DIR, set_seed
from src.data.loader import DataLoader
from src.decision.knowledge_graph import DecisionKnowledgeGraph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raw_action_data(cfg) -> List[List[str]]:
    """
    Load raw action data with cluster info (format: '4d', '3a', etc.)

    Returns:
        List of action episodes, where each action is like '4d'
    """
    paths_config = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    data_root = paths_config.get(
        "data_root", "D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree"
    )
    map_id = paths_config.get("map_id", "MarineMicro_MvsM_4")
    data_id = paths_config.get("data_id", "6")

    base_path = os.path.join(data_root, map_id, data_id)
    action_log_path = os.path.join(base_path, "action_log.csv")

    action_episodes = []
    with open(action_log_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            raw = row[0]
            actions = [raw[j : j + 2] for j in range(0, len(raw), 2)]
            action_episodes.append(actions)

    logger.info(f"Loaded {len(action_episodes)} action episodes from raw data")
    return action_episodes


def build_transitions(
    state_episodes: List[List[int]],
    action_episodes: List[List[str]],
    outcome_episodes: List[str],
    output_dir: Path,
    context_window: int,
    unique_states: Optional[Set[int]] = None,
) -> Dict:
    """
    Build state transition network from episode data.

    Args:
        state_episodes: List of state sequences
        action_episodes: List of action sequences
        outcome_episodes: List of outcomes ("Win"/"Loss"/...)
        output_dir: Directory to save the transitions file
        context_window: Context window (used for filename only)

    Returns:
        Transitions dict: {state: {action: {next_states: {state: count}, wins, total, win_rate}}}
    """
    logger.info(f"\nBuilding transitions (context_window={context_window})...")

    transitions = defaultdict(dict)

    for ep_idx in range(len(state_episodes)):
        states = state_episodes[ep_idx]
        actions = action_episodes[ep_idx] if ep_idx < len(action_episodes) else []
        outcome = (
            outcome_episodes[ep_idx] if ep_idx < len(outcome_episodes) else "Unknown"
        )

        for t in range(len(states) - 1):
            state = states[t]
            action = actions[t] if t < len(actions) else "0a"
            next_state = states[t + 1]

            if action not in transitions[state]:
                transitions[state][action] = {
                    "next_states": defaultdict(int),
                    "wins": 0,
                    "total": 0,
                }

            transitions[state][action]["next_states"][next_state] += 1
            transitions[state][action]["total"] += 1
            if outcome.lower() == "win":
                transitions[state][action]["wins"] += 1

    for state in transitions:
        for action in transitions[state]:
            trans = transitions[state][action]
            trans["next_states"] = dict(trans["next_states"])
            trans["win_rate"] = (
                trans["wins"] / trans["total"] if trans["total"] > 0 else 0.0
            )

    transitions = dict(transitions)

    if unique_states is not None:
        terminal_count = 0
        for state_id in unique_states:
            if state_id not in transitions:
                transitions[state_id] = {"__terminal__": True}
                terminal_count += 1
        logger.info(
            f"  Marked {terminal_count} terminal states (in KG but no transitions)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if context_window == 0:
        filename = "kg_simple_transitions.pkl"
    else:
        filename = f"kg_context_{context_window}_transitions.pkl"

    path = output_dir / filename
    with open(path, "wb") as f:
        pickle.dump(transitions, f)

    logger.info(f"Transitions saved to {path} ({len(transitions)} states)")
    return transitions


def build_knowledge_graph(
    context_window: int,
    output_dir: Path,
    state_episodes: List[List[int]],
    action_episodes: List[List[str]],
    reward_episodes: List[List[float]],
    outcome_episodes: List[str],
    verbose: bool = True,
) -> DecisionKnowledgeGraph:
    """
    Build knowledge graph with specified context window

    Args:
        context_window: 0 for simple version, >0 for context-aware
        output_dir: Directory to save the knowledge graph
        state_episodes: List of state sequences
        action_episodes: List of action sequences
        reward_episodes: List of reward sequences
        outcome_episodes: List of outcomes
        verbose: Print progress

    Returns:
        Built knowledge graph
    """
    use_context = context_window > 0

    if verbose:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Building Knowledge Graph (context_window={context_window})")
        logger.info(f"{'=' * 60}")

    kg = DecisionKnowledgeGraph(
        use_context=use_context,
        context_window=context_window if use_context else 0,
        action_format="cluster+letter",
    )

    kg.build_from_data(
        state_episodes=state_episodes,
        action_episodes=action_episodes,
        reward_episodes=reward_episodes,
        outcome_episodes=outcome_episodes,
        verbose=verbose,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_context:
        filename = f"kg_context_{context_window}.pkl"
    else:
        filename = "kg_simple.pkl"

    output_path = output_dir / filename
    kg.save(str(output_path))

    stats = kg.get_statistics()
    if verbose:
        logger.info(f"\nKnowledge Graph Statistics:")
        logger.info(f"  Total visits: {stats['total_visits']}")
        logger.info(f"  Total trajectories: {stats['total_trajectories']}")
        logger.info(f"  Unique states: {stats['unique_states']}")
        logger.info(f"  Unique actions: {stats['unique_actions']}")
        logger.info(f"  Total keys: {stats['total_keys']}")
        logger.info(
            f"  States with multiple actions: {stats['states_with_multiple_actions']}"
        )

    return kg


def validate_knowledge_graph(kg: DecisionKnowledgeGraph, test_states: List[int] = None):
    """
    Validate knowledge graph by checking some example queries

    Args:
        kg: Knowledge graph to validate
        test_states: States to test (default: use first few states)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("Validating Knowledge Graph")
    logger.info(f"{'=' * 60}")

    if test_states is None:
        test_states = list(kg.unique_states)[:10]

    for state in test_states:
        top_actions = kg.get_top_k_actions(state, k=5, metric="quality_score")

        if not top_actions:
            continue

        logger.info(f"\nState {state}:")
        logger.info(f"  Top actions by quality_score:")

        for action, stats in top_actions[:3]:
            logger.info(
                f"    {action}: visits={stats['visits']:4d}, "
                f"future_r={stats['avg_future_reward']:6.1f}, "
                f"win_rate={stats['win_rate'] * 100:5.1f}%, "
                f"quality={stats['quality_score']:6.1f}"
            )

    states_with_data = 0
    for state in kg.unique_states:
        top_actions = kg.get_top_k_actions(state, k=1)
        if top_actions:
            states_with_data += 1

    coverage = states_with_data / len(kg.unique_states) * 100 if kg.unique_states else 0
    logger.info(
        f"\nCoverage: {states_with_data}/{len(kg.unique_states)} states have data ({coverage:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(description="Build Decision Knowledge Graph")
    parser.add_argument(
        "--map-id",
        type=str,
        default=None,
        help="Map ID, overrides paths.yaml (e.g. MarineMicro_MvsM_4_mirror)",
    )
    parser.add_argument(
        "--data-id",
        type=str,
        default=None,
        help="Data ID, overrides paths.yaml (e.g. 1, 3, 6)",
    )
    parser.add_argument(
        "--context-windows",
        type=int,
        nargs="+",
        default=[0],
        help="Context windows to build (0=simple, >0=context-aware). Default: [0]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: cache/knowledge_graph/<map_id>)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate knowledge graphs after building",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    cfg = get_config()

    if args.map_id:
        cfg.setdefault("paths", {})["map_id"] = args.map_id
    if args.data_id:
        cfg.setdefault("paths", {})["data_id"] = args.data_id

    paths_config = cfg.get("paths", {})
    map_id = paths_config.get("map_id", "MarineMicro_MvsM_4")
    data_id = paths_config.get("data_id", "6")

    if args.output_dir is None:
        output_dir = ROOT_DIR / "cache" / "knowledge_graph" / map_id
    else:
        output_dir = Path(args.output_dir)

    logger.info(f"Map: {map_id}, Data ID: {data_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Context windows: {args.context_windows}")

    logger.info("\nLoading data...")
    loader = DataLoader(cfg)

    state_episodes = loader.state_log

    action_episodes = load_raw_action_data(cfg)

    reward_episodes = loader.r_log

    game_results = loader.game_results
    outcome_episodes = [result[0] if result else "Unknown" for result in game_results]

    logger.info(f"Loaded {len(state_episodes)} episodes")
    logger.info(f"State episodes: {len(state_episodes)}")
    logger.info(f"Action episodes: {len(action_episodes)}")
    logger.info(f"Reward episodes: {len(reward_episodes)}")
    logger.info(f"Outcome episodes: {len(outcome_episodes)}")

    knowledge_graphs = {}

    for context_window in args.context_windows:
        kg = build_knowledge_graph(
            context_window=context_window,
            output_dir=output_dir,
            state_episodes=state_episodes,
            action_episodes=action_episodes,
            reward_episodes=reward_episodes,
            outcome_episodes=outcome_episodes,
            verbose=True,
        )

        knowledge_graphs[context_window] = kg

        build_transitions(
            state_episodes=state_episodes,
            action_episodes=action_episodes,
            outcome_episodes=outcome_episodes,
            output_dir=output_dir,
            context_window=context_window,
            unique_states=kg.unique_states,
        )

        if args.validate:
            validate_knowledge_graph(kg)

    logger.info(f"\n{'=' * 60}")
    logger.info("BUILD COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"\nMap: {map_id}, Data ID: {data_id}")
    logger.info(f"Built {len(knowledge_graphs)} knowledge graphs:")
    for context_window, kg in knowledge_graphs.items():
        stats = kg.get_statistics()
        logger.info(
            f"  context_window={context_window}: {stats['total_keys']} keys, "
            f"{stats['unique_actions']} actions"
        )

    logger.info(f"\nKnowledge graphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
