"""
ActionTuningModel -- Monte Carlo action tuning model.

The model is intentionally stored outside the ETG.  It learns online
state-action return estimates from exploratory episodes and can recommend
when to keep the ETG action, switch to a better-supported tuned action, or
try an under-explored action with UCB-style exploration.
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


ACTION_LETTERS = "abcdefghijk"


def is_valid_action_code(action_code: Optional[str]) -> bool:
    return (
        isinstance(action_code, str)
        and len(action_code) == 2
        and action_code[0] in "01234"
        and action_code[1] in ACTION_LETTERS
    )


def save_atomic(model: "ActionTuningModel", path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    backup = path + ".backup"
    if os.path.exists(path):
        os.replace(path, backup)
    tmp = path + ".tmp"
    model.save(tmp)
    os.replace(tmp, path)


@dataclass
class TuningActionStats:
    visits: int = 0
    wins: int = 0
    total_return: float = 0.0
    total_sq_return: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0
    confidence: float = 0.0
    last_updated: str = ""

    def update(
        self,
        value: float,
        result: str,
        target_visits: int,
        return_scale: float = 50.0,
    ) -> None:
        self.visits += 1
        if result == "Win":
            self.wins += 1
        self.total_return += value
        self.total_sq_return += value * value
        self.mean_return = self.total_return / self.visits
        variance = max(self.total_sq_return / self.visits - self.mean_return**2, 0.0)
        self.std_return = math.sqrt(variance)
        visit_conf = min(1.0, self.visits / max(target_visits, 1))
        scale = max(float(return_scale), 1e-6)
        stability_conf = 1.0 / (1.0 + self.std_return / scale)
        self.confidence = visit_conf * stability_conf
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def recompute_confidence(self, target_visits: int, return_scale: float = 50.0) -> None:
        visit_conf = min(1.0, self.visits / max(target_visits, 1))
        scale = max(float(return_scale), 1e-6)
        stability_conf = 1.0 / (1.0 + self.std_return / scale)
        self.confidence = visit_conf * stability_conf


@dataclass
class TuningDecision:
    action: Optional[str]
    source: str
    confidence: float = 0.0
    advantage: float = 0.0
    reason: str = ""
    etg_action: Optional[str] = None
    candidate_action: Optional[str] = None
    candidate_visits: int = 0


@dataclass
class ActionTuningModel:
    state_action_stats: Dict[Any, Dict[str, TuningActionStats]] = field(default_factory=dict)
    target_visits: int = 10
    exploration_c: float = 1.4
    discount_factor: float = 0.95
    outcome_bonus: float = 50.0
    confidence_return_scale: float = 50.0
    trained_episodes: int = 0
    updated_at: str = ""

    def _ensure_runtime_defaults(self) -> None:
        if not hasattr(self, "confidence_return_scale"):
            self.confidence_return_scale = 50.0
        if not hasattr(self, "state_action_stats") or self.state_action_stats is None:
            self.state_action_stats = {}

    def recalibrate_confidences(self) -> None:
        self._ensure_runtime_defaults()
        for actions in self.state_action_stats.values():
            for stats in actions.values():
                stats.recompute_confidence(
                    self.target_visits,
                    self.confidence_return_scale,
                )

    def _state_key(self, state_id: Any) -> Any:
        if isinstance(state_id, str):
            return state_id
        try:
            return int(state_id)
        except (TypeError, ValueError):
            return state_id

    def update(self, state_id: Any, action_code: str, value: float, result: str = "") -> None:
        self._ensure_runtime_defaults()
        if state_id is None or not is_valid_action_code(action_code):
            return
        actions = self.state_action_stats.setdefault(self._state_key(state_id), {})
        stats = actions.setdefault(action_code, TuningActionStats())
        stats.update(
            float(value),
            result,
            self.target_visits,
            self.confidence_return_scale,
        )
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def update_episode(
        self,
        trajectory: List[Dict],
        result: str,
        final_score: float = 0.0,
        credit_mode: str = "every_visit",
    ) -> None:
        terminal = self.outcome_bonus if result == "Win" else -self.outcome_bonus if result == "Loss" else 0.0
        running_return = float(final_score) + terminal
        seen_pairs = set()
        for item in reversed(trajectory):
            running_return = float(item.get("reward", 0.0)) + self.discount_factor * running_return
            pair = (item.get("nid"), item.get("action_code", ""))
            if credit_mode == "first_visit" and pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            self.update(pair[0], pair[1], running_return, result)
        self.trained_episodes += 1

    def total_visits(self, state_id: Any) -> int:
        return sum(s.visits for s in self.state_action_stats.get(self._state_key(state_id), {}).values())

    def get_stats(self, state_id: Any, action_code: str) -> Optional[TuningActionStats]:
        return self.state_action_stats.get(self._state_key(state_id), {}).get(action_code)

    def estimate(self, state_id: Any, action_code: str) -> float:
        stats = self.get_stats(state_id, action_code)
        return stats.mean_return if stats and stats.visits > 0 else 0.0

    def confidence(self, state_id: Any, action_code: str) -> float:
        stats = self.get_stats(state_id, action_code)
        return stats.confidence if stats else 0.0

    def _candidate_actions(
        self,
        state_id: Any,
        etg_action: Optional[str],
        ranked_actions: Optional[Iterable[str]],
        excluded_actions: Optional[Iterable[str]] = None,
    ) -> List[str]:
        excluded = {
            action
            for action in excluded_actions or []
            if is_valid_action_code(action)
        }
        candidates: List[str] = []
        for action in ranked_actions or []:
            if (
                is_valid_action_code(action)
                and action not in excluded
                and action not in candidates
            ):
                candidates.append(action)
        for action in self.state_action_stats.get(self._state_key(state_id), {}):
            if (
                is_valid_action_code(action)
                and action not in excluded
                and action not in candidates
            ):
                candidates.append(action)
        cluster = etg_action[0] if is_valid_action_code(etg_action) else "4"
        for letter in ACTION_LETTERS:
            action = cluster + letter
            if action not in excluded and action not in candidates:
                candidates.append(action)
        return candidates

    def ucb_score(self, state_id: Any, action_code: str) -> float:
        stats = self.get_stats(state_id, action_code)
        total = max(self.total_visits(state_id), 1)
        if stats is None or stats.visits == 0:
            return float("inf")
        bonus = self.exploration_c * math.sqrt(math.log(total + 1.0) / stats.visits)
        return stats.mean_return + bonus

    def choose_action(
        self,
        state_id: Any,
        etg_action: Optional[str],
        ranked_actions: Optional[Iterable[str]] = None,
        min_confidence: float = 0.35,
        min_advantage: float = 1.0,
        min_visits: int = 3,
        explore: bool = False,
        excluded_actions: Optional[Iterable[str]] = None,
    ) -> TuningDecision:
        candidates = self._candidate_actions(
            state_id,
            etg_action,
            ranked_actions,
            excluded_actions=excluded_actions,
        )
        if not candidates:
            return TuningDecision(None, "etg", reason="no_candidates", etg_action=etg_action)

        if explore:
            best_explore = max(candidates, key=lambda a: self.ucb_score(state_id, a))
            stats = self.get_stats(state_id, best_explore)
            return TuningDecision(
                best_explore,
                "mc_explore",
                confidence=self.confidence(state_id, best_explore),
                advantage=self.estimate(state_id, best_explore) - self.estimate(state_id, etg_action or ""),
                reason="ucb_exploration",
                etg_action=etg_action,
                candidate_action=best_explore,
                candidate_visits=stats.visits if stats else 0,
            )

        scored: List[Tuple[str, float, float]] = []
        for action in candidates:
            scored.append((action, self.estimate(state_id, action), self.confidence(state_id, action)))
        scored.sort(key=lambda item: item[1], reverse=True)
        best_action, best_value, best_conf = scored[0]
        best_stats = self.get_stats(state_id, best_action)
        best_visits = best_stats.visits if best_stats else 0
        etg_value = self.estimate(state_id, etg_action or "")
        advantage = best_value - etg_value
        if (
            best_action != etg_action
            and best_visits >= min_visits
            and best_conf >= min_confidence
            and advantage >= min_advantage
        ):
            return TuningDecision(
                best_action,
                "tuning",
                confidence=best_conf,
                advantage=advantage,
                reason="confident_advantage",
                etg_action=etg_action,
                candidate_action=best_action,
                candidate_visits=best_visits,
            )
        return TuningDecision(
            etg_action,
            "etg",
            confidence=best_conf,
            advantage=advantage,
            reason="etg_preferred",
            etg_action=etg_action,
            candidate_action=best_action,
            candidate_visits=best_visits,
        )

    def get_state_summary(self, state_id: Any) -> Dict:
        actions = []
        state_key = self._state_key(state_id)
        for action, stats in self.state_action_stats.get(state_key, {}).items():
            actions.append(
                {
                    "action": action,
                    "visits": stats.visits,
                    "wins": stats.wins,
                    "mean_return": round(stats.mean_return, 4),
                    "std_return": round(stats.std_return, 4),
                    "confidence": round(stats.confidence, 4),
                    "ucb_score": round(self.ucb_score(state_id, action), 4)
                    if stats.visits > 0
                    else None,
                }
            )
        actions.sort(key=lambda item: item["mean_return"], reverse=True)
        return {"state_id": state_key, "total_visits": self.total_visits(state_key), "actions": actions}

    def get_summary(self) -> Dict:
        self._ensure_runtime_defaults()
        total_states = len(self.state_action_stats)
        total_pairs = sum(len(actions) for actions in self.state_action_stats.values())
        total_visits = sum(
            stats.visits
            for actions in self.state_action_stats.values()
            for stats in actions.values()
        )
        return {
            "total_states": total_states,
            "total_pairs": total_pairs,
            "total_visits": total_visits,
            "target_visits": self.target_visits,
            "exploration_c": self.exploration_c,
            "discount_factor": self.discount_factor,
            "confidence_return_scale": self.confidence_return_scale,
            "trained_episodes": self.trained_episodes,
            "updated_at": self.updated_at,
        }

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ActionTuningModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        model._ensure_runtime_defaults()
        model.recalibrate_confidences()
        return model
