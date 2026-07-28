#!/usr/bin/env python
"""
Decision Trajectory Simulation Script

This script simulates decision trajectories starting from an initial state (default: state 0),
using the knowledge graph for action recommendations and predicting next states to validate
whether the trajectory can reach a winning terminal state.

Usage:
    python scripts/simulate_kg_trajectory.py --episodes 100
    python scripts/simulate_kg_trajectory.py --mode network --episodes 100
    python scripts/simulate_kg_trajectory.py --verbose --save-trajectories output/trajectories.json

Author: PredictionRTS Team
Date: 2026-03-23
"""

import sys
import argparse
import json
import logging
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_config, set_seed, ROOT_DIR
from src.data.loader import DataLoader
from src.decision.knowledge_graph import DecisionKnowledgeGraph
from src.models.StateTransitionPredictor import StateTransitionPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrajectorySimulator:
    """
    Decision Trajectory Simulator

    Simulates decision trajectories from an initial state using knowledge graph
    for action recommendations and state transition prediction.
    """

    def __init__(
        self,
        kg: DecisionKnowledgeGraph,
        transitions: Dict,
        terminal_states: Dict,
        action_vocab: Dict[str, int],
        predictor: Optional[nn.Module] = None,
        mode: str = "probability",
        strategy: str = "roulette",
        verbose: bool = False,
    ):
        self.kg = kg
        self.transitions = transitions
        self.terminal_states = terminal_states
        self.action_vocab = action_vocab
        self.reverse_action_vocab = {v: k for k, v in action_vocab.items()}
        self.predictor = predictor
        self.mode = mode
        self.strategy = strategy
        self.verbose = verbose

        if predictor is not None:
            self.predictor.eval()

    def check_terminal(self, state: int) -> Tuple[bool, Optional[str]]:
        if state not in self.terminal_states:
            return False, None

        info = self.terminal_states[state]
        if not info.get("is_terminal", False):
            return False, None

        if info["win_rate"] >= 0.5:
            return True, "win"
        else:
            return True, "loss"

    def roulette_wheel_select(
        self, actions: List[Tuple[str, Dict]], metric: str = "quality_score"
    ) -> Tuple[str, Dict]:
        if not actions:
            return None, None

        action_list = [a for a, _ in actions]
        qualities = [stats.get(metric, 0) for _, stats in actions]

        min_q = min(qualities)
        if min_q < 0:
            qualities = [q - min_q + 0.1 for q in qualities]
        else:
            qualities = [q + 0.1 for q in qualities]

        total = sum(qualities)
        probs = [q / total for q in qualities]

        r = np.random.random()
        cumsum = 0
        for action, prob in zip(action_list, probs):
            cumsum += prob
            if r <= cumsum:
                for a, stats in actions:
                    if a == action:
                        return action, stats

        return actions[-1]

    def greedy_select(
        self, actions: List[Tuple[str, Dict]], metric: str = "quality_score"
    ) -> Tuple[str, Dict]:
        if not actions:
            return None, None

        return max(actions, key=lambda x: x[1].get(metric, 0))

    def select_action(
        self, state: int, top_k: int = 5, metric: str = "quality_score"
    ) -> Tuple[Optional[str], Optional[Dict]]:
        top_actions = self.kg.get_top_k_actions(
            state=state, k=top_k, metric=metric, min_visits=1
        )

        if not top_actions:
            return None, None

        if self.strategy == "roulette":
            return self.roulette_wheel_select(top_actions, metric)
        elif self.strategy == "greedy":
            return self.greedy_select(top_actions, metric)
        else:
            return self.roulette_wheel_select(top_actions, metric)

    def predict_by_probability(self, state: int, action: str) -> int:
        if state not in self.transitions:
            return state

        if action not in self.transitions[state]:
            return state

        next_states_dict = self.transitions[state][action].get("next_states", {})
        if not next_states_dict:
            return state

        states = list(next_states_dict.keys())
        counts = list(next_states_dict.values())
        total = sum(counts)
        probs = [c / total for c in counts]

        return int(np.random.choice(states, p=probs))

    def predict_by_network(
        self, state: int, action: str, temperature: float = 1.0
    ) -> int:
        if self.predictor is None:
            logger.warning("No predictor loaded, falling back to probability mode")
            return self.predict_by_probability(state, action)

        if action not in self.action_vocab:
            return self.predict_by_probability(state, action)

        state_tensor = torch.tensor([state], dtype=torch.long)
        action_tensor = torch.tensor([self.action_vocab[action]], dtype=torch.long)

        with torch.no_grad():
            top_k_indices, top_k_probs = self.predictor.predict_top_k(
                state_tensor, action_tensor, k=5, temperature=temperature
            )

            idx = torch.multinomial(top_k_probs[0], 1)
            next_state = top_k_indices[0, idx].item()

        return int(next_state)

    def predict_next_state(self, state: int, action: str) -> int:
        if self.mode == "network":
            return self.predict_by_network(state, action)
        else:
            return self.predict_by_probability(state, action)

    def simulate_episode(
        self, start_state: int = 0, max_steps: int = 50, top_k: int = 5
    ) -> Dict[str, Any]:
        trajectory = []
        state = start_state

        for step in range(max_steps):
            is_terminal, outcome = self.check_terminal(state)
            if is_terminal:
                break

            action, stats = self.select_action(state, top_k=top_k)

            if action is None:
                if self.verbose:
                    print(f"Step {step}: No actions for state {state}, stopping")
                break

            next_state = self.predict_next_state(state, action)

            step_info = {
                "step": step,
                "state": int(state),
                "action": action,
                "next_state": int(next_state),
                "visits": stats.get("visits", 0),
                "avg_step_reward": stats.get("avg_step_reward", 0.0),
                "avg_future_reward": stats.get("avg_future_reward", 0.0),
                "quality_score": stats.get("quality_score", 0.0),
                "win_rate": stats.get("win_rate", 0.0),
            }
            trajectory.append(step_info)

            if self.verbose:
                print(
                    f"Step {step}: State {state} --{action}--> State {next_state} "
                    f"(quality={stats['quality_score']:.2f}, win_rate={stats['win_rate'] * 100:.1f}%)"
                )

            state = next_state

        final_is_terminal, final_outcome = self.check_terminal(state)

        if final_outcome is None:
            final_outcome = "max_steps"

        if trajectory:
            avg_quality = np.mean([t["quality_score"] for t in trajectory])
            avg_win_rate = np.mean([t["win_rate"] for t in trajectory])
        else:
            avg_quality = 0.0
            avg_win_rate = 0.0

        return {
            "trajectory": trajectory,
            "final_state": int(state),
            "outcome": final_outcome,
            "length": len(trajectory),
            "avg_quality": float(avg_quality),
            "avg_win_rate": float(avg_win_rate),
        }

    def run_simulation(
        self,
        n_episodes: int = 100,
        start_state: int = 0,
        max_steps: int = 50,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        episodes = []
        outcomes = {"win": 0, "loss": 0, "max_steps": 0}

        for ep_idx in range(n_episodes):
            result = self.simulate_episode(
                start_state=start_state, max_steps=max_steps, top_k=top_k
            )
            result["episode_id"] = ep_idx
            episodes.append(result)
            outcomes[result["outcome"]] += 1

            if (ep_idx + 1) % 20 == 0:
                logger.info(f"Simulated {ep_idx + 1}/{n_episodes} episodes")

        lengths = [ep["length"] for ep in episodes]
        qualities = [ep["avg_quality"] for ep in episodes]
        win_rates = [ep["avg_win_rate"] for ep in episodes]

        all_actions = []
        all_states = []
        for ep in episodes:
            for step in ep["trajectory"]:
                all_actions.append(step["action"])
                all_states.append(step["state"])

        action_dist = Counter(all_actions)
        state_dist = Counter(all_states)

        return {
            "config": {
                "mode": self.mode,
                "strategy": self.strategy,
                "n_episodes": n_episodes,
                "start_state": start_state,
                "max_steps": max_steps,
                "top_k": top_k,
            },
            "summary": {
                "outcomes": outcomes,
                "outcome_rates": {k: v / n_episodes for k, v in outcomes.items()},
                "avg_length": float(np.mean(lengths)),
                "std_length": float(np.std(lengths)),
                "avg_quality": float(np.mean(qualities)),
                "avg_win_rate": float(np.mean(win_rates)),
            },
            "analysis": {
                "action_distribution": dict(action_dist.most_common(20)),
                "state_distribution": dict(state_dist.most_common(20)),
            },
            "episodes": episodes,
        }


def build_terminal_state_map(
    state_log: List[List[int]], game_results: List[List]
) -> Dict[int, Dict]:
    terminal_states = defaultdict(
        lambda: {
            "is_terminal": False,
            "win_count": 0,
            "loss_count": 0,
            "total_count": 0,
            "win_rate": 0.0,
        }
    )

    for ep_idx, (states, result) in enumerate(zip(state_log, game_results)):
        if not states:
            continue

        last_state = states[-1]
        outcome = result[0] if result else "Unknown"

        terminal_states[last_state]["is_terminal"] = True
        terminal_states[last_state]["total_count"] += 1

        if outcome.lower() == "win":
            terminal_states[last_state]["win_count"] += 1
        else:
            terminal_states[last_state]["loss_count"] += 1

    for state_id in terminal_states:
        info = terminal_states[state_id]
        if info["total_count"] > 0:
            info["win_rate"] = info["win_count"] / info["total_count"]

    return dict(terminal_states)


def load_predictor(
    model_path: str, state_dim: int = 940, action_dim: int = 11
) -> Optional[nn.Module]:
    if not Path(model_path).exists():
        logger.warning(f"Predictor model not found: {model_path}")
        return None

    predictor = StateTransitionPredictor(state_dim=state_dim, action_dim=action_dim)

    checkpoint = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        predictor.load_state_dict(checkpoint["model_state_dict"])
    else:
        predictor.load_state_dict(checkpoint)

    predictor.eval()
    logger.info(f"Loaded predictor from {model_path}")
    return predictor


def print_results(results: Dict, verbose: bool = False, show_trajectories: int = 5):
    config = results["config"]
    summary = results["summary"]
    analysis = results["analysis"]

    print("\n" + "=" * 70)
    print("         DECISION TRAJECTORY SIMULATION RESULTS")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Prediction Mode:     {config['mode']}")
    print(f"  Action Strategy:     {config['strategy']}")
    print(f"  Episodes:            {config['n_episodes']}")
    print(f"  Start State:         {config['start_state']}")
    print(f"  Max Steps:           {config['max_steps']}")
    print(f"  Top-K Actions:       {config['top_k']}")
    print()

    print("Trajectory Outcomes:")
    print("-" * 50)
    outcomes = summary["outcomes"]
    rates = summary["outcome_rates"]
    print(f"  Win Terminals:        {outcomes['win']:4d} ({rates['win'] * 100:5.1f}%)")
    print(
        f"  Loss Terminals:       {outcomes['loss']:4d} ({rates['loss'] * 100:5.1f}%)"
    )
    print(
        f"  Max Steps Reached:    {outcomes['max_steps']:4d} ({rates['max_steps'] * 100:5.1f}%)"
    )
    print()

    print("Metrics:")
    print("-" * 50)
    print(
        f"  Avg Episode Length:   {summary['avg_length']:.2f} +/- {summary['std_length']:.2f}"
    )
    print(f"  Avg Quality Score:    {summary['avg_quality']:.2f}")
    print(f"  Avg Win Rate:         {summary['avg_win_rate'] * 100:.1f}%")
    print()

    print("Top 10 Most Visited States:")
    print("-" * 50)
    for state, count in list(analysis["state_distribution"].items())[:10]:
        print(f"  State {state:4d}: {count:4d} visits")
    print()

    print("Top 10 Most Selected Actions:")
    print("-" * 50)
    total_actions = sum(analysis["action_distribution"].values())
    for action, count in list(analysis["action_distribution"].items())[:10]:
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action}: {count:4d} times ({pct:5.1f}%)")
    print()

    if show_trajectories > 0:
        print("Sample Trajectories:")
        print("-" * 50)
        for i, ep in enumerate(results["episodes"][:show_trajectories]):
            traj_str = " -> ".join([f"{t['state']}" for t in ep["trajectory"][:8]])
            if len(ep["trajectory"]) > 8:
                traj_str += " -> ..."
            outcome_str = ep["outcome"].upper()
            print(f"  Episode {ep['episode_id']:3d}: {traj_str} [{outcome_str}]")
        print()

    if verbose and results["episodes"]:
        print("Detailed First Trajectory:")
        print("-" * 70)
        print(
            f"{'Step':<5} {'State':<6} {'Action':<6} {'Next':<6} {'Quality':<8} {'WinRate':<8}"
        )
        print("-" * 70)
        for step in results["episodes"][0]["trajectory"][:20]:
            print(
                f"{step['step']:<5} {step['state']:<6} {step['action']:<6} "
                f"{step['next_state']:<6} {step['quality_score']:<8.2f} {step['win_rate'] * 100:.1f}%"
            )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate decision trajectories using knowledge graph"
    )
    parser.add_argument(
        "--kg-type", type=str, default="simple", choices=["simple", "context"]
    )
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument(
        "--mode",
        type=str,
        default="probability",
        choices=["probability", "network"],
        help="Next state prediction mode",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="roulette",
        choices=["roulette", "greedy"],
        help="Action selection strategy",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--start-state", type=int, default=0, help="Starting state")
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Max steps per episode"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-k actions to consider"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--show-trajectories",
        type=int,
        default=15,
        help="Number of sample trajectories to show",
    )
    parser.add_argument(
        "--save-trajectories",
        type=str,
        default=None,
        help="Save trajectories to JSON file",
    )
    parser.add_argument(
        "--predictor-path",
        type=str,
        default=None,
        help="Path to predictor model (default: cache/model/state_predictor.pth)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    kg_dir = ROOT_DIR / "cache" / "knowledge_graph"
    if args.kg_type == "simple":
        kg_path = kg_dir / "kg_simple.pkl"
        transitions_path = kg_dir / "kg_simple_transitions.pkl"
    else:
        kg_path = kg_dir / f"kg_context_{args.context_window}.pkl"
        transitions_path = kg_dir / f"kg_context_{args.context_window}_transitions.pkl"

    if not kg_path.exists():
        logger.error(f"Knowledge graph not found: {kg_path}")
        logger.error("Please run: python scripts/build_knowledge_graph.py")
        return

    kg = DecisionKnowledgeGraph.load(str(kg_path))
    logger.info(f"Loaded knowledge graph from {kg_path}")
    logger.info(f"  States: {len(kg.unique_states)}, Actions: {len(kg.unique_actions)}")

    if not transitions_path.exists():
        logger.error(f"Transitions file not found: {transitions_path}")
        return

    with open(transitions_path, "rb") as f:
        transitions = pickle.load(f)
    logger.info(f"Loaded transitions from {transitions_path}")
    logger.info(f"  States with transitions: {len(transitions)}")

    cfg = get_config()
    loader = DataLoader(cfg)
    state_log = loader.state_log
    game_results = loader.game_results
    action_vocab = loader.action_vocab

    if not action_vocab:
        action_vocab = {chr(ord("a") + i): i for i in range(11)}

    logger.info("Building terminal state map...")
    terminal_states = build_terminal_state_map(state_log, game_results)
    n_terminal = sum(1 for s in terminal_states.values() if s["is_terminal"])
    logger.info(f"Found {n_terminal} terminal states")

    predictor = None
    if args.mode == "network":
        predictor_path = args.predictor_path
        if predictor_path is None:
            predictor_path = str(ROOT_DIR / "cache" / "model" / "state_predictor.pth")

        predictor = load_predictor(predictor_path)
        if predictor is None:
            logger.warning("Failed to load predictor, falling back to probability mode")
            args.mode = "probability"

    logger.info(
        f"\nInitializing simulator (mode={args.mode}, strategy={args.strategy})..."
    )
    simulator = TrajectorySimulator(
        kg=kg,
        transitions=transitions,
        terminal_states=terminal_states,
        action_vocab=action_vocab,
        predictor=predictor,
        mode=args.mode,
        strategy=args.strategy,
        verbose=args.verbose,
    )

    logger.info(f"Running {args.episodes} episodes from state {args.start_state}...")
    results = simulator.run_simulation(
        n_episodes=args.episodes,
        start_state=args.start_state,
        max_steps=args.max_steps,
        top_k=args.top_k,
    )

    print_results(
        results, verbose=args.verbose, show_trajectories=args.show_trajectories
    )

    if args.save_trajectories:
        output_path = Path(args.save_trajectories)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Trajectories saved to {output_path}")

    logger.info("Simulation complete!")


if __name__ == "__main__":
    main()
