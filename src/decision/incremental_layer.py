"""
Incremental layer scaffolding for online exploration data.

This module intentionally keeps all writes in a delta directory and never
modifies the base ETG or base BKTree files.  It is a safe first step toward
incremental BKTree / ETG updates while preserving reproducibility.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class IncrementalLayerConfig:
    enabled: bool = False
    update_bktree: bool = False
    update_etg_delta: bool = False
    use_delta_for_planning: bool = False
    persist_interval_episodes: int = 10
    min_new_state_distance: float = 1.0
    delta_dir: str = "output/incremental_layer"

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "IncrementalLayerConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            update_bktree=bool(data.get("update_bktree", False)),
            update_etg_delta=bool(data.get("update_etg_delta", False)),
            use_delta_for_planning=bool(data.get("use_delta_for_planning", False)),
            persist_interval_episodes=int(data.get("persist_interval_episodes", 10)),
            min_new_state_distance=float(data.get("min_new_state_distance", 1.0)),
            delta_dir=str(data.get("delta_dir", "output/incremental_layer")),
        )


@dataclass
class ETGDeltaStore:
    state_action_counts: Dict[Tuple[int, str], int] = field(default_factory=dict)
    transition_counts: Dict[Tuple[int, str, int], int] = field(default_factory=dict)
    episodes: int = 0
    updated_at: str = ""

    def record_transition(
        self,
        state_id: Optional[int],
        action_code: str,
        next_state_id: Optional[int],
    ) -> None:
        if state_id is None or next_state_id is None or not action_code:
            return
        key = (int(state_id), action_code)
        self.state_action_counts[key] = self.state_action_counts.get(key, 0) + 1
        tkey = (int(state_id), action_code, int(next_state_id))
        self.transition_counts[tkey] = self.transition_counts.get(tkey, 0) + 1
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def mark_episode(self) -> None:
        self.episodes += 1
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "ETGDeltaStore":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> Dict:
        return {
            "episodes": self.episodes,
            "state_action_pairs": len(self.state_action_counts),
            "transitions": len(self.transition_counts),
            "updated_at": self.updated_at,
        }


class IncrementalLayerStore:
    def __init__(self, config: IncrementalLayerConfig):
        self.config = config
        self.delta_dir = Path(config.delta_dir)
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        self.etg_delta_path = self.delta_dir / "etg_delta.pkl"
        self.meta_path = self.delta_dir / "incremental_meta.json"
        if self.etg_delta_path.exists():
            try:
                self.etg_delta = ETGDeltaStore.load(str(self.etg_delta_path))
            except Exception:
                self.etg_delta = ETGDeltaStore()
        else:
            self.etg_delta = ETGDeltaStore()

    def record_episode_transitions(self, frames) -> None:
        if not self.config.enabled or not self.config.update_etg_delta:
            return
        for current, nxt in zip(frames, frames[1:]):
            self.etg_delta.record_transition(
                current.get("nid"),
                current.get("action_code", ""),
                nxt.get("nid"),
            )
        self.etg_delta.mark_episode()
        if self.etg_delta.episodes % max(self.config.persist_interval_episodes, 1) == 0:
            self.save()

    def save(self) -> None:
        if self.config.update_etg_delta:
            self.etg_delta.save(str(self.etg_delta_path))
        meta = {
            "config": self.config.__dict__,
            "etg_delta": self.etg_delta.summary(),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> Dict:
        return {
            "enabled": self.config.enabled,
            "update_bktree": self.config.update_bktree,
            "update_etg_delta": self.config.update_etg_delta,
            "use_delta_for_planning": self.config.use_delta_for_planning,
            "delta_dir": str(self.delta_dir),
            "etg_delta": self.etg_delta.summary(),
        }
