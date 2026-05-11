"""
ActionOverrideModel -- 反事实动作覆盖模型

基于反事实模拟（counterfactual simulation）发现 beam search 的次优决策，
以 (state_id, original_action) -> replacement_action 规则的形式保存，
供 agent 运行时按置信度替换 beam search 推荐动作。

数据结构:
  _overrides: Dict[(state_id, original_action), OverrideEntry]

置信度公式:
  improvement = cf_avg_score - original_avg_score
  improvement_normalized = sigmoid(improvement / scale_factor)
  confidence = improvement_normalized * (1 - 1/(1 + cf_runs))

持久化: pickle 格式
"""

from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def save_atomic(model: "ActionOverrideModel", path: str) -> None:
    backup = path + ".backup"
    if os.path.exists(path):
        os.replace(path, backup)
    tmp = path + ".tmp"
    model.save(tmp)
    os.replace(tmp, path)


@dataclass
class OverrideEntry:
    state_id: int = 0
    original_action: str = ""
    replacement_action: str = ""
    original_episodes: int = 0
    original_wins: int = 0
    original_total_score: float = 0.0
    original_avg_score: float = 0.0
    cf_runs: int = 0
    cf_wins: int = 0
    cf_total_score: float = 0.0
    cf_avg_score: float = 0.0
    improvement: float = 0.0
    confidence: float = 0.0
    created_trial: int = -1
    last_updated_trial: int = -1


@dataclass
class ActionOverrideModel:
    _overrides: Dict[Tuple[int, str], OverrideEntry] = field(default_factory=dict)
    confidence_scale: float = 10.0

    def suggest_override(
        self,
        state_id: int,
        action_code: str,
        min_confidence: float = 0.5,
    ) -> Optional[str]:
        entry = self._overrides.get((state_id, action_code))
        if entry is None:
            return None
        if entry.confidence < min_confidence:
            return None
        if entry.cf_runs < 3:
            return None
        return entry.replacement_action

    def record_original(
        self,
        state_id: int,
        action_code: str,
        result: str,
        score: float,
        trial_number: int = -1,
    ) -> None:
        key = (state_id, action_code)
        if key not in self._overrides:
            self._overrides[key] = OverrideEntry(
                state_id=state_id,
                original_action=action_code,
                created_trial=trial_number,
                last_updated_trial=trial_number,
            )
        entry = self._overrides[key]
        entry.original_episodes += 1
        entry.original_total_score += score
        if result == "Win":
            entry.original_wins += 1
        entry.original_avg_score = entry.original_total_score / entry.original_episodes
        self._recalculate(entry)

    def update_counterfactual(
        self,
        state_id: int,
        original_action: str,
        replacement_action: str,
        result: str,
        score: float,
        trial_number: int = -1,
    ) -> None:
        key = (state_id, original_action)
        if key not in self._overrides:
            self._overrides[key] = OverrideEntry(
                state_id=state_id,
                original_action=original_action,
                replacement_action=replacement_action,
                created_trial=trial_number,
                last_updated_trial=trial_number,
            )
        entry = self._overrides[key]
        if not entry.replacement_action:
            entry.replacement_action = replacement_action
        entry.cf_runs += 1
        entry.cf_total_score += score
        if result == "Win":
            entry.cf_wins += 1
        entry.cf_avg_score = entry.cf_total_score / entry.cf_runs
        entry.last_updated_trial = max(entry.last_updated_trial, trial_number)
        self._recalculate(entry)

    def _recalculate(self, entry: OverrideEntry) -> None:
        entry.improvement = entry.cf_avg_score - entry.original_avg_score
        raw = entry.improvement / max(self.confidence_scale, 0.1)
        raw = np.clip(raw, -20, 20)
        improvement_normalized = 1.0 / (1.0 + np.exp(-raw))
        runs_factor = 1.0 - 1.0 / (1.0 + entry.cf_runs) if entry.cf_runs > 0 else 0.0
        entry.confidence = float(improvement_normalized * runs_factor)

    def get_all_entries(self) -> List[OverrideEntry]:
        return sorted(
            self._overrides.values(),
            key=lambda e: -e.confidence,
        )

    def get_summary(self) -> Dict:
        entries = list(self._overrides.values())
        if not entries:
            return {
                "total_rules": 0,
                "avg_improvement": 0.0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "total_cf_runs": 0,
            }
        improvements = [e.improvement for e in entries]
        confidences = [e.confidence for e in entries]
        return {
            "total_rules": len(entries),
            "avg_improvement": round(float(np.mean(improvements)), 4),
            "avg_confidence": round(float(np.mean(confidences)), 4),
            "max_confidence": round(float(np.max(confidences)), 4),
            "total_cf_runs": sum(e.cf_runs for e in entries),
        }

    def merge(self, other: "ActionOverrideModel") -> None:
        for key, other_entry in other._overrides.items():
            if key in self._overrides:
                self_entry = self._overrides[key]
                self_entry.original_episodes += other_entry.original_episodes
                self_entry.original_wins += other_entry.original_wins
                self_entry.original_total_score += other_entry.original_total_score
                self_entry.original_avg_score = (
                    self_entry.original_total_score / self_entry.original_episodes
                    if self_entry.original_episodes > 0
                    else 0.0
                )
                self_entry.cf_runs += other_entry.cf_runs
                self_entry.cf_wins += other_entry.cf_wins
                self_entry.cf_total_score += other_entry.cf_total_score
                self_entry.cf_avg_score = (
                    self_entry.cf_total_score / self_entry.cf_runs
                    if self_entry.cf_runs > 0
                    else 0.0
                )
                if other_entry.created_trial >= 0:
                    if (
                        self_entry.created_trial < 0
                        or other_entry.created_trial < self_entry.created_trial
                    ):
                        self_entry.created_trial = other_entry.created_trial
                if other_entry.last_updated_trial >= 0:
                    self_entry.last_updated_trial = max(
                        self_entry.last_updated_trial, other_entry.last_updated_trial
                    )
                if not self_entry.replacement_action and other_entry.replacement_action:
                    self_entry.replacement_action = other_entry.replacement_action
                self._recalculate(self_entry)
            else:
                self._overrides[key] = OverrideEntry(
                    state_id=other_entry.state_id,
                    original_action=other_entry.original_action,
                    replacement_action=other_entry.replacement_action,
                    original_episodes=other_entry.original_episodes,
                    original_wins=other_entry.original_wins,
                    original_total_score=other_entry.original_total_score,
                    original_avg_score=other_entry.original_avg_score,
                    cf_runs=other_entry.cf_runs,
                    cf_wins=other_entry.cf_wins,
                    cf_total_score=other_entry.cf_total_score,
                    cf_avg_score=other_entry.cf_avg_score,
                    improvement=other_entry.improvement,
                    confidence=other_entry.confidence,
                    created_trial=other_entry.created_trial,
                    last_updated_trial=other_entry.last_updated_trial,
                )

    def save(self, path: str) -> None:
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        data = {
            "overrides": {f"{k[0]}_{k[1]}": v for k, v in self._overrides.items()},
            "confidence_scale": self.confidence_scale,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            f"ActionOverrideModel saved to {path} ({len(self._overrides)} rules)"
        )

    @classmethod
    def load(cls, path: str) -> "ActionOverrideModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        overrides = {}
        for k_str, v in data.get("overrides", {}).items():
            parts = k_str.split("_", 1)
            if len(parts) == 2:
                state_id = int(parts[0])
                action = parts[1]
                overrides[(state_id, action)] = v
        model = cls(
            _overrides=overrides,
            confidence_scale=data.get("confidence_scale", 10.0),
        )
        logger.info(f"ActionOverrideModel loaded from {path} ({len(overrides)} rules)")
        return model

    def prune(self, min_confidence: float = 0.1, min_cf_runs: int = 1) -> int:
        removed = 0
        keys_to_remove = []
        for key, entry in self._overrides.items():
            if entry.confidence < min_confidence or entry.cf_runs < min_cf_runs:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._overrides[key]
            removed += 1
        if removed > 0:
            logger.info(
                f"Pruned {removed} low-confidence rules, {len(self._overrides)} remaining"
            )
        return removed
