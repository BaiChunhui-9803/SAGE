"""
KGDecisionHelper - Decision Helper with Knowledge Graph

Provides a clean interface for real-time decision making using the knowledge graph.

Supports:
1. Simple action selection (roulette wheel or top-k)
2. State transition network for multi-step prediction
3. Winning probability estimation

Usage:
    # Simulation scenario
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")
    state = 42
    action, info = helper.select_action(state, k=5)
    print(f"Selected: {action}, Quality: {info['quality_score']}")

    # Real-time decision scenario
    helper = KGDecisionHelper("cache/knowledge_graph/kg_simple.pkl")
    state = get_current_state()
    action = helper.select_action(state, k=5)
    execute_action(action)
    next_state = observe_result()
    helper.update(state, action, next_state)
"""

import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from src.decision.knowledge_graph import DecisionKnowledgeGraph

from src.structure.state_distance import custom_distance

logger = logging.getLogger(__name__)


class KGDecisionHelper:
    """
    Knowledge Graph Decision Helper

    Provides action recommendations and trajectory predictions based on the knowledge graph.
    """

    def __init__(self, kg_path: str, cfg=None):
        """
        Initialize the helper.

        Args:
            kg_path: Path to knowledge graph pickle file
            cfg: Optional config dict (with 'paths' key). If None, uses get_config().
        """
        self.kg_path = kg_path
        self.cfg = cfg
        self.kg = DecisionKnowledgeGraph.load(kg_path)
        self.transitions: Dict = None
        self._load_transitions()

        self._action_history: List[str] = []
        self._current_state: Optional[int] = None
        self._current_action: Optional[str] = None
        self._step_count: int = 0

    def _load_transitions(self):
        """Load state transition network from cache or build from data."""
        if self.transitions is not None:
            return

        transitions_path = self.kg_path.replace(".pkl", "_transitions.pkl")
        if transitions_path.endswith("_transitions.pkl"):
            transitions_path = transitions_path.replace(
                "_transitions.pkl", ".transitions.pkl"
            )

        try:
            with open(transitions_path, "rb") as f:
                self.transitions = pickle.load(f)
            logger.info(f"Transitions loaded from {transitions_path}")
            return
        except FileNotFoundError:
            pass

        logger.info("Building transitions from training data...")

        from src import get_config
        from src.data.loader import DataLoader

        if self.cfg is not None:
            cfg = self.cfg
        else:
            cfg = get_config()

        loader = DataLoader(cfg)

        state_episodes = loader.dt_data["states"]

        paths_config = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
        data_root = paths_config.get(
            "data_root", "D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree"
        )
        map_id = paths_config.get("map_id", "MarineMicro_MvsM_4")
        data_id = paths_config.get("data_id", "6")

        import csv

        action_log_path = f"{data_root}/{map_id}/{data_id}/action_log.csv"

        action_episodes = []
        with open(action_log_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    raw = row[0]
                    actions = [raw[j : j + 2] for j in range(0, len(raw), 2)]
                    action_episodes.append(actions)

        game_results = loader.game_results
        outcomes = [result[0] if result else "Unknown" for result in game_results]

        transitions = defaultdict(dict)

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

        for state in transitions:
            for action in transitions[state]:
                trans = transitions[state][action]
                trans["next_states"] = dict(trans["next_states"])
                trans["win_rate"] = (
                    trans["wins"] / trans["total"] if trans["total"] > 0 else 0.0
                )

        self.transitions = dict(transitions)
        self._save_transitions()
        logger.info(f"Transitions built: {len(self.transitions)} states")

    def _save_transitions(self):
        """Save transitions to cache file."""
        transitions_path = self.kg_path.replace(".pkl", "_transitions.pkl")
        with open(transitions_path, "wb") as f:
            pickle.dump(self.transitions, f)
        logger.info(f"Transitions saved to {transitions_path}")

    def select_action(
        self,
        state: int,
        history: List[str] = None,
        k: int = 5,
        min_visits: int = 3,
        strategy: str = "roulette",
        temperature: float = 1.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select an action using the specified strategy.

        Args:
            state: Current state ID
            history: Action history (for context-aware version)
            k: Number of candidate actions
            min_visits: Minimum visits threshold
            strategy: "roulette" (probabilistic) or "top" (greedy)
            temperature: Temperature for softmax (higher = more random)

        Returns:
            (action, info): Selected action and detailed info
        """
        top_k_actions = self.kg.get_top_k_actions(
            state=state,
            history=history,
            k=k,
            min_visits=min_visits,
            metric="quality_score",
        )

        if not top_k_actions:
            return None, {}

        action_list = []
        qualities = []

        for action, info in top_k_actions:
            action_list.append(action)
            qualities.append(info["quality_score"])

        if strategy == "roulette":
            qualities = np.array(qualities)
            min_q = qualities.min()
            if min_q < 0:
                qualities = qualities - min_q + 0.1
            else:
                qualities = qualities + 0.1

            if temperature != 1.0:
                qualities = qualities / temperature
                probs = qualities / qualities.sum()
            else:
                probs = np.ones(len(qualities)) / len(qualities)

            r = np.random.random()
            cumsum = 0.0
            for action, prob in zip(action_list, probs):
                cumsum += prob
                if r <= cumsum:
                    selected_action = action
                    selected_info = self.kg.get_action_quality(state, action, history)
                    break
            else:
                idx = np.argmax(qualities)
                selected_action = action_list[idx]
                selected_info = top_k_actions[idx][1]

        self._current_state = state
        self._current_action = selected_action

        return selected_action, selected_info

    def update(
        self,
        state: int,
        action: str,
        next_state: int,
        immediate_reward: float = None,
    ):
        """
        Update decision history (for real-time decision scenario).

        Args:
            state: Current state ID
            action: Action executed
            next_state: Resulting state
            immediate_reward: Immediate reward (optional)
        """
        self._action_history.append(action)
        self._current_state = state
        self._current_action = action
        self._step_count += 1

    def get_top_k_actions(
        self, state: int, k: int = 5, min_visits: int = 3
    ) -> List[Dict]:
        """Get top-k candidate actions (without selection)."""
        return self.kg.get_top_k_actions(
            state=state, k=k, min_visits=min_visits, metric="quality_score"
        )

    def predict_next_states(self, state: int, action: str) -> Dict[int, float]:
        """
        Predict possible next states and their probabilities.

        Args:
            state: Current state ID
            action: Action to execute

        Returns:
            Dict mapping next_state -> probability
        """
        if state not in self.transitions or action not in self.transitions[state]:
            return {}

        next_states = self.transitions[state][action]["next_states"]
        total = sum(next_states.values())
        return {s: c / total for s, c in next_states.items()}

    def get_winning_probability(self, state: int, action: str) -> float:
        """
        Get predicted winning probability for a state-action pair.

        Args:
            state: Current state ID
            action: Action to execute

        Returns:
            Predicted winning probability (0-1)
        """
        quality = self.kg.get_action_quality(state, action)
        if quality:
            return quality["win_rate"]
        return 0.0

    def get_expected_future_reward(self, state: int, action: str) -> float:
        """
        Get expected future reward for a state-action pair.

        Args:
            state: Current state ID
            action: Action to execute

        Returns:
            Expected future reward
        """
        quality = self.kg.get_action_quality(state, action)
        if quality:
            return quality["avg_future_reward"]
        return 0.0

    def predict_trajectory(
        self, state: int, action: str, steps: int = 5, top_k: int = 5
    ) -> List[Dict]:
        """
        Predict future trajectory using transition network.

        This is useful for:
        - Evaluating long-term value of actions
        - Identifying potentially dangerous state sequences
        - Planning multi-step strategies

        Args:
            state: Current state ID
            action: First action to execute
            steps: Number of steps to predict
            top_k: Number of top actions to consider at each step

        Returns:
            List of predicted trajectory steps, each containing:
            - step: Step number
            - state: State ID
            - action: Selected action
            - probability: Selection probability
            - quality: Action quality score
        """
        if steps <= 0:
            return []

        trajectory = []
        current_state = state
        current_action = action

        quality = self.kg.get_action_quality(current_state, current_action)
        if not quality:
            return []

        first_next_states = self.predict_next_states(current_state, current_action)
        if not first_next_states:
            return []

        most_likely_next = max(first_next_states, key=first_next_states.get)

        trajectory.append(
            {
                "step": 0,
                "state": current_state,
                "action": current_action,
                "probability": 1.0,
                "quality": quality["quality_score"] if quality else 0.0,
            }
        )

        for step in range(1, steps):
            next_states_probs = self.predict_next_states(current_state, current_action)
            if not next_states_probs:
                break

            next_state = np.random.choice(
                list(next_states_probs.keys()), p=list(next_states_probs.values())
            )

            top_actions = self.kg.get_top_k_actions(
                state=next_state, k=top_k, min_visits=1, metric="quality_score"
            )
            if not top_actions:
                break

            best_action = top_actions[0][0]
            best_quality = top_actions[0][1]["quality_score"]

            current_state = next_state
            current_action = best_action

            trajectory.append(
                {
                    "step": step,
                    "state": current_state,
                    "action": best_action,
                    "probability": next_states_probs[next_state],
                    "quality": best_quality,
                }
            )

        return trajectory

    def evaluate_action_sequence(
        self, state: int, actions: List[str], start_step: int = 0
    ) -> Tuple[float, List[Dict]]:
        """
        Evaluate a sequence of actions from a starting state.

        Args:
            state: Starting state ID
            actions: List of actions to evaluate
            start_step: Step to start evaluating from

        Returns:
            (final_quality: Average quality score of trajectory)
            (reached_states: List of states visited)
        """
        if not actions:
            return 0.0, []

        current_state = state
        qualities = []
        reached_states = []

        for i in range(len(actions)):
            if current_state not in self.transitions:
                break

            action = actions[i]
            if action not in self.transitions.get(current_state, {}):
                break

            next_states = (
                self.transitions[current_state].get(action, {}).get("next_states", {})
            )
            if not next_states:
                break

            next_state = list(next_states.keys())[0]
            current_state = next_state
            reached_states.append(current_state)

            quality = self.kg.get_action_quality(state, action)
            if quality:
                qualities.append(quality["quality_score"])

        final_quality = np.mean(qualities) if qualities else 0.0
        return final_quality, reached_states
