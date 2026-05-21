"""
FinetuneModel — 基于置信度的状态微调机

模型存储: Q-Table + k-NN 平滑
- Q-Table 天然提供 visits/置信度
- k-NN 平滑解决稀疏状态的泛化问题
- 复用项目已有的 dist_matrix (N×N 对称距离矩阵)

双指标: replacement_score = confidence * (1 - action_rank)
  <0.1    未探索 → 优先探索
  0.1~0.3 探索不足 → 继续收集
  0.3~thr 需微调 → 用模型替换 beam search 建议
  >=thr   置信足够 → 直接用 beam search
"""

from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def save_atomic(model: "FinetuneModel", path: str) -> None:
    backup = path + ".backup"
    if os.path.exists(path):
        os.replace(path, backup)
    tmp = path + ".tmp"
    model.save(tmp)
    os.replace(tmp, path)


@dataclass
class ActionEstimate:
    visits: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    confidence: float = 0.0
    action_rank: float = 0.0


@dataclass
class GlobalPrior:
    global_mean: float = 0.0
    prior_strength: float = 5.0


@dataclass
class FinetuneModel:
    q_table: Dict[int, Dict[str, ActionEstimate]] = field(default_factory=dict)
    prior: GlobalPrior = field(default_factory=GlobalPrior)
    sigma: float = 0.5
    target_visits: int = 10
    trained_episodes: int = 0
    last_trained: str = ""

    def update(self, state_id: int, action_code: str, reward: float) -> None:
        if state_id not in self.q_table:
            self.q_table[state_id] = {}
        if action_code not in self.q_table[state_id]:
            self.q_table[state_id][action_code] = ActionEstimate()

        est = self.q_table[state_id][action_code]
        est.visits += 1
        est.total_reward += reward
        est.avg_reward = est.total_reward / est.visits
        est.confidence = min(1.0, est.visits / self.target_visits)
        self._refresh_ranks(state_id)
        self.trained_episodes += 1
        self.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _refresh_ranks(self, state_id: int) -> None:
        actions = self.q_table.get(state_id)
        if not actions:
            return
        sorted_acts = sorted(actions.values(), key=lambda e: e.avg_reward, reverse=True)
        n = len(sorted_acts)
        for rank, est in enumerate(sorted_acts):
            est.action_rank = rank / max(n - 1, 1)

    def posterior_mean(self, state_id: int, action_code: str) -> float:
        est = self.q_table.get(state_id, {}).get(action_code)
        if est is None or est.visits == 0:
            return self.prior.global_mean
        return (
            est.visits * est.avg_reward
            + self.prior.prior_strength * self.prior.global_mean
        ) / (est.visits + self.prior.prior_strength)

    def smooth_q(
        self, state_id: int, action_code: str, dist_matrix: Optional[np.ndarray] = None
    ) -> float:
        est = self.q_table.get(state_id, {}).get(action_code)
        total_visits_s = sum(e.visits for e in self.q_table.get(state_id, {}).values())

        if est is not None and total_visits_s >= self.target_visits:
            return self.posterior_mean(state_id, action_code)

        if dist_matrix is None or state_id not in self.q_table:
            return self.posterior_mean(state_id, action_code)

        alpha = min(1.0, total_visits_s / self.target_visits)
        local_q = self.posterior_mean(state_id, action_code)

        neighbor_q_sum = 0.0
        neighbor_w_sum = 0.0
        for sid in self.q_table:
            if sid == state_id:
                continue
            try:
                d = float(dist_matrix[state_id, sid])
            except (IndexError, TypeError, KeyError):
                continue
            if not np.isfinite(d) or np.isnan(d):
                continue
            w = np.exp(-d / self.sigma)
            if w < 1e-8:
                continue
            neighbor_q_sum += w * self.posterior_mean(sid, action_code)
            neighbor_w_sum += w

        if neighbor_w_sum > 0:
            neighbor_avg = neighbor_q_sum / neighbor_w_sum
        else:
            neighbor_avg = self.prior.global_mean

        return alpha * local_q + (1.0 - alpha) * neighbor_avg

    def replacement_score(self, state_id: int, action_code: str) -> float:
        est = self.q_table.get(state_id, {}).get(action_code)
        if est is None:
            return 0.0
        return est.confidence * (1.0 - est.action_rank)

    def rank_actions_by_finetune(
        self,
        state_id: int,
        dist_matrix: Optional[np.ndarray] = None,
    ) -> List[str]:
        actions = self.q_table.get(state_id, {})
        if not actions:
            return []
        scored = []
        for ac in actions:
            q = self.smooth_q(state_id, ac, dist_matrix)
            scored.append((ac, q))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ac for ac, _ in scored]

    def get_state_stats(self, state_id: int) -> Dict:
        actions = self.q_table.get(state_id, {})
        total_v = sum(e.visits for e in actions.values())
        scores = []
        for ac, est in actions.items():
            scores.append(
                {
                    "action": ac,
                    "visits": est.visits,
                    "avg_reward": round(est.avg_reward, 4),
                    "confidence": round(est.confidence, 4),
                    "action_rank": round(est.action_rank, 4),
                    "replacement_score": round(self.replacement_score(state_id, ac), 4),
                }
            )
        scores.sort(key=lambda x: x["avg_reward"], reverse=True)
        return {
            "state_id": state_id,
            "total_visits": total_v,
            "num_actions": len(actions),
            "actions": scores,
        }

    def get_overall_stats(self) -> Dict:
        total_states = len(self.q_table)
        total_visits = sum(
            e.visits for acts in self.q_table.values() for e in acts.values()
        )
        explored_states = sum(
            1
            for acts in self.q_table.values()
            if any(e.visits > 0 for e in acts.values())
        )
        all_scores = []
        for sid in self.q_table:
            for ac in self.q_table[sid]:
                all_scores.append(self.replacement_score(sid, ac))
        avg_rs = float(np.mean(all_scores)) if all_scores else 0.0
        explored_ratio = explored_states / max(total_states, 1)
        return {
            "total_states": total_states,
            "total_visits": total_visits,
            "explored_states": explored_states,
            "explored_ratio": round(explored_ratio, 4),
            "avg_replacement_score": round(avg_rs, 4),
            "target_visits": self.target_visits,
            "sigma": self.sigma,
            "trained_episodes": self.trained_episodes,
            "last_trained": self.last_trained,
        }

    def update_batch(self, updates: List[Tuple[int, str, float]]) -> None:
        dirty_states = set()
        for state_id, action_code, reward in updates:
            if state_id is None:
                continue
            if state_id not in self.q_table:
                self.q_table[state_id] = {}
            if action_code not in self.q_table[state_id]:
                self.q_table[state_id][action_code] = ActionEstimate()
            est = self.q_table[state_id][action_code]
            est.visits += 1
            est.total_reward += reward
            est.avg_reward = est.total_reward / est.visits
            est.confidence = min(1.0, est.visits / self.target_visits)
            dirty_states.add(state_id)
        for sid in dirty_states:
            self._refresh_ranks(sid)
        self.trained_episodes += len(updates)
        self.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"FinetuneModel saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FinetuneModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"FinetuneModel loaded from {path}")
        return model
