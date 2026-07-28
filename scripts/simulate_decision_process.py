#!/usr/bin/env python
"""
Simulate Decision Process with Knowledge Graph

This script simulates decision making using the knowledge graph's state transition network.
Starting from state 0, it uses roulette wheel selection to choose actions and follows
the transitions to see if it can reach a winning state.

Usage:
    python scripts/simulate_decision_process.py --kg-type simple --episodes 100
    python scripts/simulate_decision_process.py --kg-type context --context-window 5 --verbose

Author: PredictionRTS Team
Date: 2026-03-21
"""

import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_config, set_seed, ROOT_DIR
from src.data.loader import DataLoader
from src.decision.knowledge_graph import DecisionKnowledgeGraph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StateTransitionSimulator:
    """
    Simulates decision process using knowledge graph

    Uses roulette wheel selection based on action quality scores
    """

    def __init__(self, kg: DecisionKnowledgeGraph, verbose: bool = False):
        self.kg = kg
        self.verbose = verbose

        # Build state transition network from knowledge graph
        self.transitions = self._build_transitions()

    def _build_transitions(self) -> dict:
        """
        Build state transition network from knowledge graph

        Returns:
            transitions[state][action] = {
                'next_states': {next_state: count, ...},
                'quality': float,
                'win_rate': float
            }
        """
        transitions = defaultdict(dict)

        # Load original data to get state transitions
        cfg = get_config()
        loader = DataLoader(cfg)

        state_episodes = loader.dt_data["states"]

        # Load raw action data
        import csv

        data_root = cfg.get(
            "data_root", "D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree"
        )
        map_id = cfg.get("map_id", "MarineMicro_MvsM_4")
        data_id = cfg.get("data_id", "6")

        action_log_path = f"{data_root}/{map_id}/{data_id}/action_log.csv"

        action_episodes = []
        with open(action_log_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    raw = row[0]
                    actions = [raw[j : j + 2] for j in range(0, len(raw), 2)]
                    action_episodes.append(actions)

        # Get outcomes
        game_results = loader.game_results
        outcomes = [result[0] if result else "Unknown" for result in game_results]

        # Build transition counts
        for ep_idx in range(len(state_episodes)):
            states = state_episodes[ep_idx]
            actions = action_episodes[ep_idx] if ep_idx < len(action_episodes) else []
            outcome = outcomes[ep_idx] if ep_idx < len(outcomes) else "Unknown"

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

        # Convert to regular dict and add quality metrics
        for state in transitions:
            for action in transitions[state]:
                trans = transitions[state][action]
                trans["next_states"] = dict(trans["next_states"])

                # Get quality from knowledge graph
                quality = self.kg.get_action_quality(state, action)
                if quality:
                    trans["quality_score"] = quality["quality_score"]
                    trans["win_rate"] = quality["win_rate"]
                else:
                    trans["quality_score"] = 0
                    trans["win_rate"] = (
                        trans["wins"] / trans["total"] if trans["total"] > 0 else 0
                    )

        return dict(transitions)

    def roulette_wheel_select(self, actions: dict) -> str:
        """
        Select action using roulette wheel selection based on quality scores

        Args:
            actions: dict of {action: {'quality_score': float, ...}}

        Returns:
            Selected action
        """
        if not actions:
            return None

        # Get quality scores
        action_list = list(actions.keys())
        qualities = [actions[a]["quality_score"] for a in action_list]

        # Shift to positive values
        min_q = min(qualities)
        if min_q < 0:
            qualities = [q - min_q + 0.1 for q in qualities]
        else:
            qualities = [q + 0.1 for q in qualities]

        # Normalize to probabilities
        total = sum(qualities)
        probs = [q / total for q in qualities]

        # Roulette wheel selection
        r = np.random.random()
        cumsum = 0
        for action, prob in zip(action_list, probs):
            cumsum += prob
            if r <= cumsum:
                return action

        return action_list[-1]

    def get_next_state(self, state: int, action: str) -> int:
        """
        Get next state from transition network

        Args:
            state: Current state
            action: Action taken

        Returns:
            Next state (randomly selected based on transition counts)
        """
        if state not in self.transitions or action not in self.transitions[state]:
            return state  # Stay in current state if no transition

        next_states = self.transitions[state][action]["next_states"]
        if not next_states:
            return state

        # Weighted random selection
        states = list(next_states.keys())
        counts = list(next_states.values())
        total = sum(counts)
        probs = [c / total for c in counts]

        return np.random.choice(states, p=probs)

    def simulate_episode(
        self, start_state: int = 0, max_steps: int = 30, top_k: int = 5
    ) -> dict:
        """
        Simulate one episode starting from start_state

        Args:
            start_state: Starting state
            max_steps: Maximum steps to simulate
            top_k: Number of top actions to consider for selection

        Returns:
            Episode result with trajectory and outcome
        """
        trajectory = []
        state = start_state

        for step in range(max_steps):
            # Get top-k actions from knowledge graph
            top_actions = self.kg.get_top_k_actions(
                state=state, k=top_k, metric="quality_score", min_visits=1
            )

            if not top_actions:
                if self.verbose:
                    print(f"Step {step}: No actions for state {state}, stopping")
                break

            # Convert to dict for roulette wheel
            actions_dict = {action: stats for action, stats in top_actions}

            # Select action using roulette wheel
            action = self.roulette_wheel_select(actions_dict)

            if action is None:
                break

            # Get next state
            next_state = self.get_next_state(state, action)

            # Get quality metrics
            quality = actions_dict[action]

            # Record transition
            trajectory.append(
                {
                    "step": step,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "avg_step_reward": quality["avg_step_reward"],
                    "avg_future_reward": quality["avg_future_reward"],
                    "quality_score": quality["quality_score"],
                    "win_rate": quality["win_rate"],
                }
            )

            if self.verbose:
                print(
                    f"Step {step}: State {state} --{action}--> State {next_state} "
                    f"(quality={quality['quality_score']:.2f}, win_rate={quality['win_rate'] * 100:.1f}%)"
                )

            state = next_state

        # Calculate episode metrics
        if trajectory:
            avg_quality = np.mean([t["quality_score"] for t in trajectory])
            avg_win_rate = np.mean([t["win_rate"] for t in trajectory])
        else:
            avg_quality = 0
            avg_win_rate = 0

        return {
            "trajectory": trajectory,
            "length": len(trajectory),
            "avg_quality": avg_quality,
            "avg_win_rate": avg_win_rate,
            "final_state": state,
        }

    def run_simulation(
        self,
        num_episodes: int = 100,
        start_state: int = 0,
        max_steps: int = 30,
        top_k: int = 5,
    ) -> dict:
        """
        Run multiple simulation episodes

        Returns:
            Aggregated simulation results
        """
        results = []

        for ep in range(num_episodes):
            result = self.simulate_episode(
                start_state=start_state, max_steps=max_steps, top_k=top_k
            )
            results.append(result)

            if (ep + 1) % 20 == 0:
                logger.info(f"Simulated {ep + 1}/{num_episodes} episodes")

        # Aggregate results
        lengths = [r["length"] for r in results]
        qualities = [r["avg_quality"] for r in results]
        win_rates = [r["avg_win_rate"] for r in results]

        return {
            "num_episodes": num_episodes,
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "avg_quality": np.mean(qualities),
            "avg_win_rate": np.mean(win_rates),
            "episodes": results,
        }


def main():
    parser = argparse.ArgumentParser(description="Simulate decision process")
    parser.add_argument(
        "--kg-type", type=str, default="simple", choices=["simple", "context"]
    )
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-state", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--detail-log",
        action="store_true",
        help="输出每次模拟的详细状态-动作-奖励记录",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="将详细记录输出到指定文件 (JSON格式)",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # Load knowledge graph
    kg_dir = ROOT_DIR / "cache" / "knowledge_graph"
    if args.kg_type == "simple":
        kg_path = kg_dir / "kg_simple.pkl"
    else:
        kg_path = kg_dir / f"kg_context_{args.context_window}.pkl"

    if not kg_path.exists():
        logger.error(f"Knowledge graph not found: {kg_path}")
        logger.error("Please run: python scripts/build_knowledge_graph.py")
        return

    kg = DecisionKnowledgeGraph.load(str(kg_path))
    logger.info(f"Loaded knowledge graph from {kg_path}")

    # Create simulator
    logger.info("Building state transition network...")
    simulator = StateTransitionSimulator(kg, verbose=args.verbose)

    logger.info(
        f"State transition network built with {len(simulator.transitions)} states"
    )

    # Run simulation
    logger.info(
        f"\nSimulating {args.episodes} episodes from state {args.start_state}..."
    )
    results = simulator.run_simulation(
        num_episodes=args.episodes,
        start_state=args.start_state,
        max_steps=args.max_steps,
        top_k=args.top_k,
    )

    # Print results
    print("\n" + "=" * 60)
    print("       SIMULATION RESULTS")
    print("=" * 60)
    print()
    print(f"Knowledge Graph: {args.kg_type}")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Start State: {args.start_state}")
    print()
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(
        f"{'Avg Episode Length':<30} {results['avg_length']:.2f} +/- {results['std_length']:.2f}"
    )
    print(f"{'Avg Quality Score':<30} {results['avg_quality']:.2f}")
    print(f"{'Avg Win Rate':<30} {results['avg_win_rate'] * 100:.1f}%")
    print()

    # Analyze trajectory patterns
    print("Trajectory Analysis:")
    print("-" * 50)

    # Action distribution
    all_actions = []
    for ep in results["episodes"]:
        for step in ep["trajectory"]:
            all_actions.append(step["action"])

    from collections import Counter

    action_dist = Counter(all_actions)
    print(f"\nTop 10 Actions:")
    for action, count in action_dist.most_common(10):
        pct = count / len(all_actions) * 100 if all_actions else 0
        print(f"  {action}: {count} ({pct:.1f}%)")

    # State visit distribution
    all_states = []
    for ep in results["episodes"]:
        for step in ep["trajectory"]:
            all_states.append(step["state"])

    state_dist = Counter(all_states)
    print(f"\nTop 10 Visited States:")
    for state, count in state_dist.most_common(10):
        pct = count / len(all_states) * 100 if all_states else 0
        print(f"  State {state}: {count} ({pct:.1f}%)")

    print()
    logger.info("Simulation complete!")

    if args.detail_log or args.output_file:
        import json

        detail_records = []
        for ep_idx, ep in enumerate(results["episodes"]):
            for step_info in ep["trajectory"]:
                detail_records.append(
                    {
                        "episode": ep_idx,
                        "step": int(step_info["step"]),
                        "state": int(step_info["state"]),
                        "action": step_info["action"],
                        "next_state": int(step_info["next_state"]),
                        "avg_step_reward": float(step_info["avg_step_reward"]),
                        "avg_future_reward": float(step_info["avg_future_reward"]),
                        "quality_score": float(step_info["quality_score"]),
                        "win_rate": float(step_info["win_rate"]),
                    }
                )

        if args.detail_log:
            print("\n" + "=" * 80)
            print("       DETAILED SIMULATION LOG")
            print("=" * 80)
            print(
                f"{'Ep':<4} {'Step':<5} {'State':<6} {'Action':<6} {'Next':<6} "
                f"{'StepRwd':<10} {'FutRwd':<10} {'Quality':<8} {'WinRate'}"
            )
            print("-" * 80)
            for r in detail_records[:200]:
                print(
                    f"{r['episode']:<4} {r['step']:<5} {r['state']:<6} {r['action']:<6} "
                    f"{r['next_state']:<6} {r['avg_step_reward']:<10.4f} "
                    f"{r['avg_future_reward']:<10.4f} {r['quality_score']:<8.2f} "
                    f"{r['win_rate'] * 100:.1f}%"
                )
            if len(detail_records) > 200:
                print(f"... (共 {len(detail_records)} 条记录，仅显示前200条)")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(detail_records, f, indent=2, ensure_ascii=False)
            print(f"\nDetail log saved to: {args.output_file}")


if __name__ == "__main__":
    main()
