"""
Decision Knowledge Graph Module

This module implements a decision knowledge graph that stores state-action-reward statistics
for real-time decision making support.

Two versions are supported:
1. Simple: Key = state_id (fast, ignores context)
2. Context-Aware: Key = (state_id, history_hash) (more precise, requires more data)

Author: PredictionRTS Team
Date: 2026-03-21
"""

import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionStats:
    """Statistics for a state-action pair"""

    visits: int = 0
    step_rewards: List[float] = field(default_factory=list)
    future_rewards: List[float] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)  # 'win' or 'loss'
    trajectory_ids: List[int] = field(default_factory=list)

    # Computed metrics
    avg_step_reward: float = 0.0
    avg_future_reward: float = 0.0
    win_rate: float = 0.0
    quality_score: float = 0.0

    def update(
        self, step_reward: float, future_reward: float, outcome: str, trajectory_id: int
    ):
        """Add a new visit"""
        self.visits += 1
        self.step_rewards.append(step_reward)
        self.future_rewards.append(future_reward)
        self.outcomes.append(outcome)
        self.trajectory_ids.append(trajectory_id)

        # Update computed metrics
        self.avg_step_reward = np.mean(self.step_rewards)
        self.avg_future_reward = np.mean(self.future_rewards)
        self.win_rate = sum(1 for o in self.outcomes if o.lower() == "win") / len(
            self.outcomes
        )
        # Quality score: weighted combination
        self.quality_score = self.avg_future_reward * 0.7 + self.win_rate * 30.0


class DecisionKnowledgeGraph:
    """
    Decision Knowledge Graph

    Stores state-action statistics for real-time decision support.
    Supports both simple (state-only) and context-aware (state + history) versions.

    Example usage:
        kg = DecisionKnowledgeGraph(use_context=True, context_window=5)
        kg.build_from_data(states, actions, rewards, outcomes)
        top_actions = kg.get_top_k_actions(state=42, history=['4d', '3a'], k=5)
    """

    def __init__(
        self,
        use_context: bool = False,
        context_window: int = 5,
        action_format: str = "cluster+letter",  # 'cluster+letter' (4d) or 'letter' (d)
    ):
        """
        Initialize knowledge graph

        Args:
            use_context: Whether to include history context in keys
            context_window: Number of recent actions to include in context
            action_format: Format of actions ('cluster+letter' or 'letter')
        """
        self.use_context = use_context
        self.context_window = context_window
        self.action_format = action_format

        # Core storage: {key: {action: ActionStats}}
        # key = state_id (simple) or (state_id, history_hash) (context-aware)
        self.state_action_map: Dict[Any, Dict[str, ActionStats]] = defaultdict(dict)

        # Metadata
        self.total_visits = 0
        self.total_trajectories = 0
        self.unique_states = set()
        self.unique_actions = set()

        # Optional: similarity encoder (can be set later)
        self.similarity_encoder = None

    def _get_key(self, state: int, history: Optional[List[str]] = None) -> Any:
        """
        Generate key for state-action map

        Args:
            state: State ID
            history: List of recent actions (e.g., ['4d', '3a', '1b'])

        Returns:
            Key (int for simple, tuple for context-aware)
        """
        if not self.use_context:
            return state
        else:
            if history is None:
                history = []
            # Take last N actions and hash
            context = tuple(history[-self.context_window :]) if history else ()
            return (state, context)

    def add_visit(
        self,
        state: int,
        action: str,
        step_reward: float,
        future_reward: float,
        outcome: str,
        trajectory_id: int,
        history: Optional[List[str]] = None,
    ):
        """
        Add a state-action visit to the knowledge graph

        Args:
            state: State ID
            action: Action string (e.g., '4d')
            step_reward: Immediate reward
            future_reward: Cumulative future reward
            outcome: 'win' or 'loss'
            trajectory_id: Episode/trajectory ID
            history: Recent action history (for context-aware version)
        """
        key = self._get_key(state, history)

        if action not in self.state_action_map[key]:
            self.state_action_map[key][action] = ActionStats()

        self.state_action_map[key][action].update(
            step_reward, future_reward, outcome, trajectory_id
        )

        self.total_visits += 1
        self.unique_states.add(state)
        self.unique_actions.add(action)

    def build_from_data(
        self,
        state_episodes: List[List[int]],
        action_episodes: List[List[str]],
        reward_episodes: List[List[float]],
        outcome_episodes: List[str],
        verbose: bool = True,
    ):
        """
        Build knowledge graph from episode data

        Args:
            state_episodes: List of state sequences
            action_episodes: List of action sequences
            reward_episodes: List of reward sequences
            outcome_episodes: List of outcomes ('Win' or 'Loss')
            verbose: Print progress
        """
        if verbose:
            logger.info(
                f"Building knowledge graph from {len(state_episodes)} episodes..."
            )

        for ep_idx in range(len(state_episodes)):
            states = state_episodes[ep_idx]
            actions = action_episodes[ep_idx]
            rewards = reward_episodes[ep_idx]
            outcome = (
                outcome_episodes[ep_idx]
                if ep_idx < len(outcome_episodes)
                else "unknown"
            )

            if len(states) == 0:
                continue

            self.total_trajectories += 1

            # Compute cumulative rewards from the end
            cumulative_rewards = []
            future_reward = 0
            for r in reversed(rewards):
                future_reward += r
                cumulative_rewards.insert(0, future_reward)

            # Add each state-action pair
            for t in range(len(states)):
                state = states[t]
                action = actions[t] if t < len(actions) else "0a"  # Default if missing
                step_reward = rewards[t] if t < len(rewards) else 0
                future_reward = (
                    cumulative_rewards[t] if t < len(cumulative_rewards) else 0
                )

                # Get history (actions before current)
                history = actions[:t] if self.use_context else None

                self.add_visit(
                    state=state,
                    action=action,
                    step_reward=step_reward,
                    future_reward=future_reward,
                    outcome=outcome,
                    trajectory_id=ep_idx,
                    history=history,
                )

            if verbose and (ep_idx + 1) % 1000 == 0:
                logger.info(f"  Processed {ep_idx + 1}/{len(state_episodes)} episodes")

        if verbose:
            logger.info(f"Knowledge graph built:")
            logger.info(f"  Total visits: {self.total_visits}")
            logger.info(f"  Total trajectories: {self.total_trajectories}")
            logger.info(f"  Unique states: {len(self.unique_states)}")
            logger.info(f"  Unique actions: {len(self.unique_actions)}")
            logger.info(f"  Total keys: {len(self.state_action_map)}")

    def get_top_k_actions(
        self,
        state: int,
        history: Optional[List[str]] = None,
        k: int = 5,
        metric: str = "quality_score",
        min_visits: int = 3,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get top-k high-quality actions for a state

        Args:
            state: State ID
            history: Recent action history (for context-aware version)
            k: Number of actions to return
            metric: Metric to sort by ('quality_score', 'future_reward', 'win_rate', 'visits')
            min_visits: Minimum visits required for an action to be considered

        Returns:
            List of (action, stats_dict) tuples sorted by metric
        """
        key = self._get_key(state, history)

        if key not in self.state_action_map:
            return []

        # Filter by min_visits and get stats
        valid_actions = []
        for action, stats in self.state_action_map[key].items():
            if stats.visits >= min_visits:
                valid_actions.append(
                    (
                        action,
                        {
                            "visits": stats.visits,
                            "avg_step_reward": stats.avg_step_reward,
                            "avg_future_reward": stats.avg_future_reward,
                            "win_rate": stats.win_rate,
                            "quality_score": stats.quality_score,
                        },
                    )
                )

        # Sort by metric
        valid_actions.sort(key=lambda x: x[1][metric], reverse=True)

        return valid_actions[:k]

    def get_action_quality(
        self, state: int, action: str, history: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get quality metrics for a specific state-action pair

        Args:
            state: State ID
            action: Action string
            history: Recent action history

        Returns:
            Dictionary with quality metrics, or None if not found
        """
        key = self._get_key(state, history)

        if key not in self.state_action_map:
            return None

        if action not in self.state_action_map[key]:
            return None

        stats = self.state_action_map[key][action]
        return {
            "visits": stats.visits,
            "avg_step_reward": stats.avg_step_reward,
            "avg_future_reward": stats.avg_future_reward,
            "win_rate": stats.win_rate,
            "quality_score": stats.quality_score,
        }

    def get_action_confidence(
        self, state: int, action: str, history: Optional[List[str]] = None
    ) -> float:
        """
        Get confidence (frequency) for a state-action pair

        Args:
            state: State ID
            action: Action string
            history: Recent action history

        Returns:
            Confidence score (0-1)
        """
        key = self._get_key(state, history)

        if key not in self.state_action_map:
            return 0.0

        total_visits = sum(s.visits for s in self.state_action_map[key].values())
        if total_visits == 0:
            return 0.0

        if action not in self.state_action_map[key]:
            return 0.0

        return self.state_action_map[key][action].visits / total_visits

    def get_similar_states(
        self, state: int, k: int = 5, use_encoder: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get similar states using similarity encoder or distance matrix

        Args:
            state: State ID
            k: Number of similar states to return
            use_encoder: Whether to use similarity encoder (if available)

        Returns:
            List of (state_id, similarity) tuples
        """
        # If encoder is available and requested, use it
        if use_encoder and self.similarity_encoder is not None:
            # TODO: Implement encoder-based similarity
            logger.warning(
                "Similarity encoder not yet implemented, using frequency-based"
            )

        # Fallback: use frequency-based similarity (states with similar action patterns)
        if state not in self.unique_states:
            return []

        # Get actions for current state
        key = self._get_key(state)
        if key not in self.state_action_map:
            return []

        current_actions = set(self.state_action_map[key].keys())

        # Find states with similar action sets
        similarities = []
        for other_state in self.unique_states:
            if other_state == state:
                continue

            other_key = self._get_key(other_state)
            if other_key not in self.state_action_map:
                continue

            other_actions = set(self.state_action_map[other_key].keys())

            # Jaccard similarity
            intersection = len(current_actions & other_actions)
            union = len(current_actions | other_actions)
            similarity = intersection / union if union > 0 else 0

            similarities.append((other_state, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about the knowledge graph"""
        # Compute action distribution
        action_counts = defaultdict(int)
        for key, actions in self.state_action_map.items():
            for action, stats in actions.items():
                action_counts[action] += stats.visits

        # Compute state coverage
        states_with_multiple_actions = sum(
            1 for actions in self.state_action_map.values() if len(actions) > 1
        )

        return {
            "total_visits": self.total_visits,
            "total_trajectories": self.total_trajectories,
            "unique_states": len(self.unique_states),
            "unique_actions": len(self.unique_actions),
            "total_keys": len(self.state_action_map),
            "states_with_multiple_actions": states_with_multiple_actions,
            "action_distribution": dict(action_counts),
            "use_context": self.use_context,
            "context_window": self.context_window if self.use_context else None,
        }

    def save(self, path: str):
        """Save knowledge graph to file"""
        data = {
            "use_context": self.use_context,
            "context_window": self.context_window,
            "action_format": self.action_format,
            "state_action_map": dict(self.state_action_map),
            "total_visits": self.total_visits,
            "total_trajectories": self.total_trajectories,
            "unique_states": self.unique_states,
            "unique_actions": self.unique_actions,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Knowledge graph saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DecisionKnowledgeGraph":
        """Load knowledge graph from file"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        kg = cls(
            use_context=data["use_context"],
            context_window=data["context_window"],
            action_format=data["action_format"],
        )

        kg.state_action_map = defaultdict(dict, data["state_action_map"])
        kg.total_visits = data["total_visits"]
        kg.total_trajectories = data["total_trajectories"]
        kg.unique_states = data["unique_states"]
        kg.unique_actions = data["unique_actions"]

        logger.info(f"Knowledge graph loaded from {path}")
        return kg
