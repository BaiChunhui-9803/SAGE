"""
KGGuidedAgent — 基于 Experience Transition Graph 引导的实时决策 Agent

继承 SmartAgent，保留全部 action 执行方法，重写 step() 决策逻辑：
    1. 每帧在本地执行状态聚类 + beam search 规划（无需跨进程通信）
    2. 回放模式下从本地列表消费预设动作
    3. 无有效决策时回退到用户配置的默认策略
    4. 每 N 局批量推送对局记录到 bridge_server 供 Web 显示
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.sc2env.agent import SmartAgent
from src.sc2env.bridge import GameBridge
from src.sc2env.config import get_map_config

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")

ACTION_NAME_MAP: Dict[str, str] = {
    "k_means_000": "k_means_000",
    "k_means_025": "k_means_025",
    "k_means_050": "k_means_050",
    "k_means_075": "k_means_075",
    "k_means_100": "k_means_100",
    "action_ATK_nearest": "action_ATK_nearest",
    "action_ATK_clu_nearest": "action_ATK_clu_nearest",
    "action_ATK_nearest_weakest": "action_ATK_nearest_weakest",
    "action_ATK_clu_nearest_weakest": "action_ATK_clu_nearest_weakest",
    "action_ATK_threatening": "action_ATK_threatening",
    "action_DEF_clu_nearest": "action_DEF_clu_nearest",
    "action_MIX_gather": "action_MIX_gather",
    "action_MIX_lure": "action_MIX_lure",
    "action_MIX_sacrifice_lure": "action_MIX_sacrifice_lure",
    "do_randomly": "do_randomly",
    "do_nothing": "do_nothing",
}

FALLBACK_ACTIONS = list(SmartAgent.actions)


def _states_match(
    actual: int, expected: int, dist_matrix=None, threshold: float = 0.2
) -> bool:
    if actual == expected:
        return True
    if dist_matrix is not None:
        try:
            d = float(dist_matrix[actual, expected])
            if not np.isnan(d) and d < threshold:
                return True
        except (IndexError, TypeError, KeyError):
            pass
    return False


class KGGuidedAgent(SmartAgent):
    def __init__(
        self,
        bridge: GameBridge,
        fallback_action: str = "action_ATK_nearest_weakest",
        initial_bktree_data: Optional[dict] = None,
        state_id_map: Optional[Dict[Tuple[int, int], int]] = None,
        kg=None,
        transitions=None,
        dist_matrix=None,
        mode: str = "multi_step",
        beam_params: Optional[Dict[str, Any]] = None,
        replay_actions: Optional[List[str]] = None,
        replay_runs: int = 1,
        action_strategy: str = "best_beam",
        data_dir: Optional[str] = None,
        kg_file: Optional[str] = None,
    ):
        super(KGGuidedAgent, self).__init__()
        self.bridge = bridge
        self._fallback_action = fallback_action
        self._data_dir = data_dir
        self._kg_file = kg_file
        self._prev_state_cluster: Optional[Tuple[int, int]] = None
        self._last_action_executed: str = ""
        self._action_history: List[Dict[str, Any]] = []
        self._pending_cluster: Optional[str] = None
        self._bktree_loaded = False
        self._state_id_map = state_id_map or {}
        self._nid_to_ps: Dict[int, Tuple[int, int]] = {
            v: k for k, v in self._state_id_map.items()
        }
        self._nid_norm_states: Dict[int, dict] = {}
        self._nid_norm_states_loaded: bool = False
        self._nid_fallback_cache: Dict[Tuple[int, int], int] = {}
        self._ep_history: List[Dict[str, Any]] = []
        self._prev_end_game_flag: bool = False
        self.kg = kg
        self.transitions = transitions
        self._dist_matrix = dist_matrix
        self._mode = mode
        self._beam_params = beam_params or {}
        self._action_strategy = action_strategy
        self._replay_actions: List[str] = list(replay_actions) if replay_actions else []
        self._replay_idx: int = 0
        self._replay_done: bool = False
        self._replay_per_ep: int = len(self._replay_actions)
        self._replay_frame_count: int = 0
        self._replay_runs_remaining: int = max(1, replay_runs) if replay_actions else 0

        self._action_plan: List[str] = []
        self._planned_states: List[int] = []
        self._plan_idx: int = 0
        self._last_plan_snap: Optional[Dict] = None
        self._all_beam_states: Set[int] = set()
        self._backup_continuations: Dict[int, Tuple[List[str], List[int]]] = {}

        self._ep_batch: List[Dict[str, Any]] = []
        self._ep_push_batch_size: int = 5
        self._frame_count: int = 0
        self._status_push_interval: int = 50
        self._ranked_actions: List[str] = []
        self._exploration_targets: Dict[int, str] = {}
        self._exploration_active: bool = False
        self._exploration_trace: List[dict] = []

        self._log_interval: int = 50
        self._log_counters: Dict[str, int] = {
            "nid_none": 0,
            "nid_fallback": 0,
            "fallback": 0,
            "kg_plan": 0,
            "kg_follow": 0,
            "diverge": 0,
            "ft_plan": 0,
            "kg_relaxed": 0,
            "fuzzy_plan": 0,
            "terminal_fix": 0,
            "total": 0,
        }

        self._local_result_dir: Optional[str] = None
        self._local_completed: int = 0

        self._plan_log_file = None

        self._prev_hp_my: Optional[int] = None
        self._prev_hp_enemy: Optional[int] = None
        self._shared_model_path: Optional[str] = None
        self._ep_action_log: List[Dict[str, Any]] = []
        self._kg_sam: Optional[Dict[int, Dict]] = None
        self._ep_completed_count: int = 0

        if initial_bktree_data is not None:
            self._load_bktree_from_data(initial_bktree_data)
            self._bktree_loaded = True

        print(
            f"[KGGuidedAgent] mode={self._mode}, state_id_map={len(self._state_id_map)}, "
            f"kg={'loaded' if self.kg else 'None'}, transitions={'loaded' if self.transitions else 'None'}, "
            f"dist_matrix={'loaded' if self._dist_matrix is not None else 'None'}"
        )

    def _load_bktree_from_data(self, data: dict) -> None:
        from src.structure.BKTree_sc2 import ClusterNode, BKTree, get_max_cluster_id

        def deserialize_node(node_data):
            if node_data is None:
                return None
            node = ClusterNode(node_data["state"], node_data["cluster_id"])
            for dist_key, child_data in node_data.get("children", {}).items():
                dist_val = int(dist_key) if dist_key.isdigit() else float(dist_key)
                child_node = deserialize_node(child_data)
                if child_node is not None:
                    node.children[dist_val] = child_node
            return node

        if "primary" in data and data["primary"] is not None:
            primary_root = deserialize_node(data["primary"])
            if primary_root is not None:
                self.primary_bktree = BKTree(
                    self.custom_distance_manager.multi_distance, distance_index=0
                )
                self.primary_bktree.root = primary_root
                max_id = get_max_cluster_id(self.primary_bktree)
                if max_id >= self.primary_bktree.next_cluster_id:
                    self.primary_bktree.next_cluster_id = max_id + 1

        if "secondary" in data:
            for cluster_id, sec_data in data["secondary"].items():
                sec_root = deserialize_node(sec_data)
                if sec_root is not None:
                    tree = BKTree(
                        self.custom_distance_manager.multi_distance, distance_index=1
                    )
                    tree.root = sec_root
                    max_id = get_max_cluster_id(tree)
                    if max_id >= tree.next_cluster_id:
                        tree.next_cluster_id = max_id + 1
                    self.secondary_bktree[int(cluster_id)] = tree

    def new_game(self):
        super().new_game()
        if not hasattr(self, "_mode"):
            return
        self._replay_frame_count = 0
        self._prev_end_game_flag = False
        self._action_plan = []
        self._planned_states = []
        self._plan_idx = 0
        self._all_beam_states = set()
        self._backup_continuations = {}
        self._exploration_targets = {}
        self._exploration_active = False
        self._exploration_trace = []

        if not (self._mode == "replay" and self._replay_actions):
            if not self._prev_end_game_flag and self._ep_history:
                if hasattr(self, "ctx") and self.ctx:
                    self.ctx.episode_count += 1
                ep_id = self.ctx.episode_count if self.ctx else 0
                self._ep_batch.append(
                    {
                        "episode_id": ep_id,
                        "frames": list(self._ep_history),
                        "result": self.end_game_state or "Dogfall",
                        "score": 0,
                    }
                )
                self._ep_history = []
                self._flush_ep_batch()
            elif self._prev_end_game_flag and self._ep_history:
                self._ep_history = []

        if self._mode == "replay" and self._replay_actions:
            if self._ep_history:
                ep_id = self.ctx.episode_count if self.ctx else 0
                self._ep_batch.append(
                    {
                        "episode_id": ep_id,
                        "frames": list(self._ep_history),
                        "result": self.end_game_state or "Dogfall",
                        "score": 0,
                    }
                )
                self._ep_history = []
            self._flush_ep_batch()

            self._replay_runs_remaining -= 1
            if self._replay_runs_remaining > 0:
                self._replay_idx = 0
                self._replay_done = False
            else:
                self._replay_idx = len(self._replay_actions)

        if hasattr(self, "ctx") and self.ctx:
            ep = self.ctx.episode_count
            self.bridge.update_status(
                episode=ep,
                agent_mode=self._mode,
                agent_params={
                    "mode": self._mode,
                    "beam_width": self._beam_params.get("beam_width", 3),
                    "max_steps": self._beam_params.get("lookahead_steps", 5),
                    "min_visits": self._beam_params.get("min_visits", 1),
                    "min_cum_prob": self._beam_params.get("min_cum_prob", 0.01),
                    "score_mode": self._beam_params.get("score_mode", "quality"),
                    "max_state_revisits": self._beam_params.get(
                        "max_state_revisits", 2
                    ),
                    "discount_factor": self._beam_params.get("discount_factor", 0.9),
                    "action_strategy": self._action_strategy,
                    "epsilon": self._beam_params.get("epsilon", 0.1),
                    "enable_backup": self._beam_params.get("enable_backup", False),
                    "backup_score_threshold": self._beam_params.get(
                        "backup_score_threshold", 0.3
                    ),
                    "backup_distance_threshold": self._beam_params.get(
                        "backup_distance_threshold", 0.2
                    ),
                    "replay_runs": self._replay_runs_remaining
                    + (1 if self._replay_idx < len(self._replay_actions) else 0),
                    "replay_per_ep": self._replay_per_ep,
                },
            )
            self.bridge.put_event(
                {
                    "level": "info",
                    "source": "game",
                    "message": f"Episode #{ep} 开始",
                }
            )

    def _resolve_action(self, raw_action: str) -> Optional[str]:
        self._pending_cluster = None
        if len(raw_action) == 2 and raw_action[0].isdigit() and raw_action[1].isalpha():
            cluster_idx = int(raw_action[0])
            action_idx = ord(raw_action[1]) - ord("a")
            if 0 <= cluster_idx < len(self.clusters):
                self._pending_cluster = self.clusters[cluster_idx]
            if 0 <= action_idx < len(self.actions):
                combat_action = self.actions[action_idx]
                if hasattr(self, combat_action):
                    return combat_action
            return None
        mapped = ACTION_NAME_MAP.get(raw_action, raw_action)
        if hasattr(self, mapped):
            return mapped
        return None

    def _query_readonly(self, norm_state):
        p_id, p_dist = self.primary_bktree.query_nearest(norm_state)
        if p_id is None:
            p_id, p_dist = 1, 0.0
        sec_tree = self.secondary_bktree.get(p_id)
        if sec_tree is None or sec_tree.root is None:
            return (p_id, 1)
        s_id, s_dist = sec_tree.query_nearest(norm_state)
        return (p_id, s_id if s_id is not None else 1)

    def get_state_cluster(self, norm_state):
        if self._bktree_loaded and self.primary_bktree.root is not None:
            return self._query_readonly(norm_state)
        return super(KGGuidedAgent, self).get_state_cluster(norm_state)

    def _record_action(self, state_cluster: Any, action_name: str, source: str) -> None:
        entry = {
            "game_loop": 0,
            "state_cluster": str(state_cluster),
            "action": action_name,
            "source": source,
        }
        if len(self._action_history) > 0:
            entry["game_loop"] = self._action_history[-1].get("game_loop", 0) + 1
        self._action_history.append(entry)
        if len(self._action_history) > 2000:
            self._action_history = self._action_history[-1000:]

    def _get_next_replay_action(self) -> Optional[str]:
        if self._replay_idx < len(self._replay_actions):
            action = self._replay_actions[self._replay_idx]
            self._replay_idx += 1
            return action
        return None

    def _get_finetune_model(self):
        model_path = self._beam_params.get("finetune_model_path")
        if not model_path:
            return None
        if (
            hasattr(self, "_cached_ft_model_path")
            and self._cached_ft_model_path == model_path
        ):
            return self._cached_ft_model
        from pathlib import Path as _Path

        p = _Path(model_path)
        if not p.exists():
            return None
        try:
            from src.decision.finetune_model import FinetuneModel

            self._cached_ft_model = FinetuneModel.load(str(p))
            self._cached_ft_model_path = model_path
            return self._cached_ft_model
        except Exception:
            return None

    def _get_plan_from_beam(
        self, state_id: int, lookahead_steps: int
    ) -> Tuple[Optional[str], List[str], List[int], List[Dict], List[Dict], List[str]]:
        from src.decision.kg_beam_search import plan_action

        if self.kg is None or self.transitions is None:
            _dbg = f"[PLAN-DEBUG] early return: kg={self.kg is None}, transitions={self.transitions is None}"
            self._write_plan_log(_dbg)
            print(_dbg, flush=True)
            return None, [], [], [], [], []

        finetune_model = self._get_finetune_model()
        dm = self._dist_matrix

        plan = plan_action(
            self.kg,
            self.transitions,
            state_id,
            beam_width=self._beam_params.get("beam_width", 3),
            max_steps=lookahead_steps,
            min_visits=self._beam_params.get("min_visits", 1),
            min_cum_prob=self._beam_params.get("min_cum_prob", 0.01),
            score_mode=self._beam_params.get("score_mode", "quality"),
            max_state_revisits=self._beam_params.get("max_state_revisits", 2),
            discount_factor=self._beam_params.get("discount_factor", 0.9),
            action_strategy=self._action_strategy,
            finetune_model=finetune_model,
            dist_matrix=dm,
        )

        if plan.recommended_action is None:
            self._all_beam_states = set()
            self._backup_continuations = {}
            return None, [], [], [], [], []

        beam_dicts = []
        for r in plan.beam_results:
            beam_dicts.append(
                {
                    "step": getattr(r, "step", 0),
                    "state": getattr(r, "state", 0),
                    "action": getattr(r, "action", ""),
                    "beam_id": getattr(r, "beam_id", 0),
                    "parent_idx": getattr(r, "parent_idx", None),
                    "cumulative_probability": getattr(r, "cumulative_probability", 0),
                    "quality_score": getattr(r, "quality_score", 0),
                    "win_rate": getattr(r, "win_rate", 0),
                    "avg_step_reward": getattr(r, "avg_step_reward", 0),
                    "avg_future_reward": getattr(r, "avg_future_reward", 0),
                }
            )

        all_beam_states = set()
        for path in plan.beam_paths:
            for node in path:
                all_beam_states.add(node.state)

        backup_continuations = {}
        for rank, path in enumerate(plan.beam_paths):
            if rank == plan.best_path_index:
                continue
            for i in range(len(path) - 1):
                src_state = path[i].state
                if src_state in backup_continuations:
                    continue
                remaining_actions = []
                remaining_states = [src_state]
                for j in range(i + 1, len(path)):
                    remaining_states.append(path[j].state)
                    if path[j].action:
                        remaining_actions.append(path[j].action)
                if remaining_actions:
                    backup_continuations[src_state] = (
                        remaining_actions,
                        remaining_states,
                    )

        self._all_beam_states = all_beam_states
        self._backup_continuations = backup_continuations

        beam_paths = []
        for rank, path in enumerate(plan.beam_paths):
            is_chosen = rank == plan.best_path_index
            path_steps = []
            for node in path:
                path_steps.append(
                    {
                        "state": node.state,
                        "action": node.action or "",
                        "cum_prob": node.cumulative_probability,
                        "win_rate": node.win_rate,
                    }
                )
            beam_paths.append(
                {
                    "rank": rank + 1,
                    "chosen": is_chosen,
                    "steps": path_steps,
                    "cum_prob": path[-1].cumulative_probability if path else 0,
                }
            )

        if finetune_model is not None and plan.action_plan:
            threshold = self._beam_params.get("finetune_threshold", 0.4)
            for i, (sid, ac) in enumerate(zip(plan.planned_states, plan.action_plan)):
                ft_ranked = finetune_model.rank_actions_by_finetune(sid, dm)
                if not ft_ranked:
                    continue
                best_ft = ft_ranked[0]
                if best_ft and best_ft != ac:
                    score = finetune_model.replacement_score(sid, ac)
                    if score < threshold:
                        plan.action_plan[i] = best_ft

        return (
            plan.recommended_action,
            plan.action_plan,
            plan.planned_states,
            beam_dicts,
            beam_paths,
            plan.ranked_actions,
        )

    def _local_decide(
        self, state_id: int, enemy_count: int = 0
    ) -> Tuple[Optional[str], str, Optional[Dict]]:
        if self.kg is None or self.transitions is None:
            return None, "fallback", None

        if self._mode == "single_step":
            self._action_plan = []
            self._planned_states = []
            self._plan_idx = 0

        lookahead = self._beam_params.get("lookahead_steps", 5)
        enable_backup = self._beam_params.get("enable_backup", False)
        backup_dist_threshold = self._beam_params.get("backup_distance_threshold", 0.2)
        is_diverge = False

        if self._plan_idx < len(self._action_plan):
            expected = (
                self._planned_states[self._plan_idx]
                if self._plan_idx < len(self._planned_states)
                else None
            )

            if expected is not None and state_id == expected:
                action = self._action_plan[self._plan_idx]
                self._plan_idx += 1
                return action, "kg_follow", self._last_plan_snap

            is_diverge = True

            if (
                enable_backup
                and self._all_beam_states
                and state_id in self._all_beam_states
            ):
                if state_id in self._backup_continuations:
                    actions, states = self._backup_continuations[state_id]
                    self._action_plan = list(actions)
                    self._planned_states = list(states)
                    self._plan_idx = 1
                    return actions[0], "backup_switch_exact", self._last_plan_snap

                best_backup_state = None
                best_backup_dist = float("inf")
                for bs in self._backup_continuations:
                    if self._dist_matrix is not None and state_id != bs:
                        try:
                            d = float(self._dist_matrix[state_id, bs])
                            if not np.isnan(d) and d < best_backup_dist:
                                best_backup_dist = d
                                best_backup_state = bs
                        except (IndexError, TypeError):
                            pass

                if (
                    best_backup_state is not None
                    and best_backup_dist < backup_dist_threshold
                ):
                    actions, states = self._backup_continuations[best_backup_state]
                    self._action_plan = list(actions)
                    self._planned_states = list(states)
                    self._plan_idx = 1
                    return actions[0], "backup_switch_fuzzy", self._last_plan_snap

            self._action_plan = []
            self._planned_states = []
            self._plan_idx = 0

        if self._plan_idx >= len(self._action_plan):
            trigger = "diverge" if is_diverge else "exhausted"

            try:
                first_action, actions, states, beam_dicts, beam_paths, ranked = (
                    self._get_plan_from_beam(state_id, lookahead)
                )
            except Exception as _pe:
                _msg = f"[PLAN] nid={state_id} -> beam exception: {_pe}"
                self._write_plan_log(_msg)
                first_action, actions, states = None, [], []

            if not actions:
                action_code, source = self._fallback_chain(state_id, trigger)
                return action_code, source, None

            self._action_plan = actions
            self._planned_states = states
            self._ranked_actions = ranked
            self._plan_idx = 0

            plan_snap = {
                "state_id": state_id,
                "action_plan": actions,
                "planned_states": states,
                "beam_results": beam_dicts,
                "beam_paths": beam_paths,
                "mode": "multi_step",
                "trigger": trigger,
            }
            self._last_plan_snap = plan_snap
            _top3 = ranked[:3] if ranked else []
            _ap = actions[:3] if actions else []
            _msg = f"[PLAN] nid={state_id} trigger={trigger} plan={_ap} ranked={_top3}"
            self._write_plan_log(_msg)

        action = self._action_plan[self._plan_idx]
        if self._plan_idx == 0:
            masked = self._beam_params.get("masked_actions", [])
            if masked and action in masked:
                _orig = action
                ranked = getattr(self, "_ranked_actions", [])
                for ra in ranked:
                    if ra not in masked:
                        action = ra
                        break
                else:
                    action = self._resolve_action(self._fallback_action)
                    if action:
                        _ci = (
                            str(self._prev_state_cluster[0])
                            if self._prev_state_cluster
                            else "4"
                        )
                        action_code = (
                            _ci + chr(ord("a") + self.actions.index(action))
                            if action in self.actions
                            else action
                        )
                        action = action_code
                _msg = f"  nid={state_id} original={_orig} -> replacement={action}"
                self._write_plan_log(_msg)
            if self._plan_idx == 0:
                event_type = "diverge" if is_diverge else "kg_plan"
            else:
                event_type = "kg_follow"

        if (
            self._exploration_active
            and state_id is not None
            and state_id in self._exploration_targets
        ):
            action = self._exploration_targets[state_id]
            self._exploration_trace.append({"state": state_id, "action": action})
            event_type = "exploration"
            print(
                f"[EXPLORATION] state_id={state_id} forced_action={action}", flush=True
            )
            del self._exploration_targets[state_id]
            if not self._exploration_targets:
                self._exploration_active = False

        self._plan_idx += 1
        return action, event_type, self._last_plan_snap

    def _fallback_chain(self, state_id: int, trigger: str) -> Tuple[Optional[str], str]:
        finetune_model = self._get_finetune_model()

        if finetune_model is not None:
            dm = self._dist_matrix
            ft_ranked = finetune_model.rank_actions_by_finetune(state_id, dm)
            if ft_ranked:
                action = ft_ranked[0]
                _msg = f"[PLAN] nid={state_id} trigger={trigger} -> ft_plan action={action}"
                self._write_plan_log(_msg)
                return action, "ft_plan"

        if self.kg is not None:
            relaxed = self.kg.get_top_k_actions(
                state=state_id, k=1, min_visits=1, metric="quality_score"
            )
            if relaxed:
                action = relaxed[0][0]
                _msg = f"[PLAN] nid={state_id} trigger={trigger} -> kg_relaxed action={action}"
                self._write_plan_log(_msg)
                return action, "kg_relaxed"

        if self._dist_matrix is not None and self.kg is not None:
            best_sid = None
            best_dist = float("inf")
            for sid in self.kg.unique_states:
                if sid == state_id:
                    continue
                try:
                    d = float(self._dist_matrix[state_id, sid])
                except (IndexError, TypeError, KeyError):
                    continue
                if np.isnan(d):
                    continue
                if d < best_dist:
                    best_dist = d
                    best_sid = sid
            if best_sid is not None and best_dist < 0.3:
                fuzzy_actions = self.kg.get_top_k_actions(
                    state=best_sid, k=1, min_visits=1, metric="quality_score"
                )
                if fuzzy_actions:
                    action = fuzzy_actions[0][0]
                    _msg = f"[PLAN] nid={state_id} trigger={trigger} -> fuzzy_plan matched_sid={best_sid} dist={best_dist:.3f} action={action}"
                    self._write_plan_log(_msg)
                    return action, "fuzzy_plan"

        _msg = f"[PLAN] nid={state_id} trigger={trigger} -> fallback (all chains exhausted)"
        self._write_plan_log(_msg)
        return None, "fallback"

    def _push_status(self, obs, state_cluster_str, my_units, enemy_units):
        if self._frame_count % self._status_push_interval == 0:
            self.bridge.update_status(
                frame=obs.observation.game_loop[0],
                my_count=len(my_units),
                enemy_count=len(enemy_units),
                state_cluster=state_cluster_str,
                my_total_hp=int(sum(u.health for u in my_units)),
                enemy_total_hp=int(sum(u.health for u in enemy_units)),
            )
        self._frame_count += 1

    def _flush_ep_batch(self, counters_snapshot=None):
        if not self._ep_batch:
            return
        if self._local_result_dir:
            import json as _json
            from pathlib import Path as _Path

            result_dir = _Path(self._local_result_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            ep_file = result_dir / "episodes.jsonl"
            progress_file = result_dir / "progress.json"
            with open(str(ep_file), "a", encoding="utf-8") as f:
                trial_number = getattr(self, "_flush_trial_number", None)
                for ep in self._ep_batch:
                    record = {
                        "episode_id": ep.get("episode_id", 0),
                        "result": ep.get("result", "Unknown"),
                        "score": ep.get("score", 0),
                        "frames": ep.get("frames", []),
                    }
                    if trial_number is not None:
                        record["trial_number"] = trial_number
                    f.write(_json.dumps(record, ensure_ascii=False) + "\n")
                    self._local_completed += 1
            hp_file = result_dir / "episodes_hp.jsonl"
            with open(str(hp_file), "a", encoding="utf-8") as f:
                trial_number = getattr(self, "_flush_trial_number", None)
                for ep in self._ep_batch:
                    frames = ep.get("frames", [])
                    steps = []
                    for i, fr in enumerate(frames):
                        steps.append(
                            {
                                "frame": i,
                                "nid": fr.get("nid"),
                                "state_cluster": fr.get("state_cluster"),
                                "hp_my": fr.get("hp_my", 0),
                                "hp_enemy": fr.get("hp_enemy", 0),
                                "hp_delta": fr.get("hp_delta", 0),
                                "action_code": fr.get("action_code", ""),
                                "action_source": fr.get("action_source", ""),
                                "my_count": fr.get("my_count", 0),
                                "enemy_count": fr.get("enemy_count", 0),
                            }
                        )
                    hp_record = {
                        "episode_id": ep.get("episode_id", 0),
                        "result": ep.get("result", "Unknown"),
                        "final_score": ep.get("score", 0),
                        "steps": steps,
                    }
                    if trial_number is not None:
                        hp_record["trial_number"] = trial_number
                    f.write(_json.dumps(hp_record, ensure_ascii=False) + "\n")
            target = self._beam_params.get("target_episodes", 0)
            progress = {"completed": self._local_completed}
            if target > 0:
                progress["target"] = target
            if self._local_completed > 0:
                c = counters_snapshot if counters_snapshot else self._log_counters
                total = max(c.get("total", 1), 1)
                progress["fallback_ratio"] = round(
                    (c.get("fallback", 0) + c.get("nid_none", 0)) / total, 4
                )
                progress["explore_ratio"] = round(
                    (
                        c.get("ft_plan", 0)
                        + c.get("kg_relaxed", 0)
                        + c.get("fuzzy_plan", 0)
                        + c.get("fallback", 0)
                    )
                    / total,
                    4,
                )
                for k in (
                    "ft_plan",
                    "kg_relaxed",
                    "fuzzy_plan",
                    "kg_plan",
                    "kg_follow",
                    "fallback",
                    "terminal_fix",
                    "nid_fallback",
                ):
                    if k in c:
                        progress[k] = c[k]
            with open(str(progress_file), "w", encoding="utf-8") as f:
                _json.dump(progress, f)
        for ep in self._ep_batch:
            try:
                self.bridge.put_history(ep)
            except Exception:
                pass
        self._ep_batch = []

    def _write_plan_log(self, msg: str) -> None:
        if self._plan_log_file is None:
            plan_log_path = self._beam_params.get("plan_log_path")
            if plan_log_path:
                try:
                    self._plan_log_file = open(
                        str(plan_log_path), "w", encoding="utf-8"
                    )
                except Exception:
                    pass
        if self._plan_log_file is not None:
            try:
                self._plan_log_file.write(msg + "\n")
                self._plan_log_file.flush()
            except Exception:
                pass

    def _save_shared_model(self) -> None:
        model = self._get_finetune_model()
        if model is None:
            return
        path = self._shared_model_path
        if path is None:
            return
        try:
            from src.decision.finetune_model import save_atomic

            save_atomic(model, path)
        except Exception as e:
            print(f"[WARN] failed to save shared model: {e}", flush=True)

    def _load_kg_sam(self) -> None:
        if self._kg_sam is not None:
            return
        kg_file = self._kg_file or self._beam_params.get("kg_file")
        data_dir = self._data_dir or self._beam_params.get("data_dir")
        if not kg_file or not data_dir:
            return
        try:
            import pickle as _pickle
            from src import ROOT_DIR as _ROOT

            kg_path = _ROOT / "cache" / "knowledge_graph" / kg_file
            if not kg_path.exists():
                return
            with open(str(kg_path), "rb") as f:
                raw = _pickle.load(f)
            sam = raw.get("state_action_map", {})
            if sam:
                self._kg_sam = sam
                print(f"[KG-SAM] loaded {len(sam)} states for ETG reward", flush=True)
        except Exception as e:
            print(f"[WARN] failed to load kg_sam: {e}", flush=True)

    def _load_nid_norm_states(self) -> None:
        if self._nid_norm_states_loaded:
            return
        self._nid_norm_states_loaded = True
        data_dir = self._data_dir or self._beam_params.get("data_dir")
        if not data_dir:
            return
        try:
            import json as _json
            from pathlib import Path as _Path

            bktree_dir = _Path(data_dir) / "bktree"
            if not bktree_dir.exists():
                return

            def _collect_nodes(node):
                nodes = []
                if node is not None:
                    nodes.append(node)
                    for child in node.get("children", {}).values():
                        nodes.extend(_collect_nodes(child))
                return nodes

            secondary_states: dict = {}
            for sf in bktree_dir.glob("secondary_bktree_*.json"):
                cid_str = sf.stem.replace("secondary_bktree_", "")
                try:
                    with open(str(sf), "r") as f:
                        root = _json.load(f)
                    p_id = int(cid_str)
                    for node in _collect_nodes(root):
                        s_id = node.get("cluster_id")
                        st = node.get("state")
                        if s_id is not None and st is not None:
                            secondary_states[(p_id, s_id)] = st
                except Exception:
                    pass

            primary_file = bktree_dir / "primary_bktree.json"
            if primary_file.exists():
                with open(str(primary_file), "r") as f:
                    root = _json.load(f)
                for node in _collect_nodes(root):
                    p_id = node.get("cluster_id")
                    if p_id is not None:
                        for (pk, sk), nid in self._state_id_map.items():
                            if pk == p_id and (pk, sk) in secondary_states:
                                self._nid_norm_states[nid] = secondary_states[(pk, sk)]

            print(
                f"[NID-NORM] loaded {len(self._nid_norm_states)} nid norm_states, "
                f"{len(secondary_states)} secondary states",
                flush=True,
            )
        except Exception as e:
            print(f"[WARN] failed to load nid_norm_states: {e}", flush=True)

    def _resolve_nid(
        self, p: int, s: int, state_norm=None
    ) -> Tuple[Optional[int], bool]:
        nid = self._state_id_map.get((p, s))
        if nid is not None:
            return nid, False

        cached = self._nid_fallback_cache.get((p, s))
        if cached is not None:
            return cached, True

        if state_norm is not None and self._nid_norm_states:
            from src.structure.custom_distance_sc2 import DistributionDistance

            best_nid = None
            best_dist = float("inf")
            best_hp = float("inf")
            for nid_cand, stored in self._nid_norm_states.items():
                ps = self._nid_to_ps.get(nid_cand)
                if ps is None or ps[0] != p:
                    continue
                try:
                    d, h = DistributionDistance(state_norm, stored)()
                    if d < best_dist or (d == best_dist and h < best_hp):
                        best_dist = d
                        best_hp = h
                        best_nid = nid_cand
                except Exception:
                    pass
            if best_nid is not None:
                self._nid_fallback_cache[(p, s)] = best_nid
                print(
                    f"[NID-FALLBACK] (p,s)=({p},{s}) -> nid={best_nid} "
                    f"dist={best_dist:.3f}",
                    flush=True,
                )
                return best_nid, True

        return None, False

    def _etg_recalibrate(self) -> None:
        model = self._get_finetune_model()
        if model is None or self._kg_sam is None:
            return
        gamma = self._beam_params.get("etg_gamma", 0.9)
        alpha = self._beam_params.get("etg_recalibrate_alpha", 0.3)
        corrected = 0
        for sid, actions in model.q_table.items():
            if sid not in self._kg_sam:
                continue
            for ac, est in actions.items():
                if ac not in self._kg_sam[sid]:
                    continue
                kg_stats = self._kg_sam[sid][ac]
                try:
                    etg_v = float(kg_stats.avg_step_reward) + gamma * float(
                        kg_stats.avg_future_reward
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
                correction = alpha * (etg_v - est.avg_reward)
                est.avg_reward += correction
                est.total_reward = est.avg_reward * est.visits
                corrected += 1
        if corrected > 0:
            print(
                f"[ETG-RECAL] corrected {corrected} state-action pairs",
                flush=True,
            )

    def _refresh_finetune_model(self) -> None:
        path = self._beam_params.get("finetune_model_path")
        if path and path != getattr(self, "_cached_ft_model_path", None):
            self._cached_ft_model = None
            self._cached_ft_model_path = None
            self._shared_model_path = path

    def step(self, obs, env):
        from pysc2.lib import actions as sc2_actions

        super(SmartAgent, self).step(obs, env)

        try:
            while True:
                new_params = self.bridge.param_update_queue.get_nowait()
                if "mode" in new_params:
                    self._mode = new_params["mode"]
                if "local_result_dir" in new_params:
                    self._local_result_dir = new_params["local_result_dir"]
                    self._local_completed = 0
                    self._flush_trial_number = new_params.get("trial_number")
                self._beam_params.update(new_params)
                if "trial_number" in new_params:
                    self.bridge.confirm_params(new_params["trial_number"])
                if "exploration_targets" in new_params:
                    self._exploration_targets = {
                        int(k): v for k, v in new_params["exploration_targets"].items()
                    }
                    self._exploration_active = bool(self._exploration_targets)
                    self._exploration_trace = []
        except Exception:
            pass

        if obs.last():
            result_event = {
                "type": "episode_end",
                "result": self.end_game_state,
                "frames": int(self.end_game_frames)
                if hasattr(self.end_game_frames, "__int__")
                else self.end_game_frames,
                "episode": getattr(self.ctx, "episode_count", 0) if self.ctx else 0,
            }
            self.bridge.put_event(result_event)
            level = (
                "success"
                if self.end_game_state == "Win"
                else ("error" if self.end_game_state == "Loss" else "warn")
            )
            self.bridge.put_event(
                {
                    "level": level,
                    "source": "game",
                    "message": f"Episode #{result_event['episode']} 结束: {self.end_game_state} (frame={result_event['frames']})",
                }
            )
            self.bridge.update_status(result=self.end_game_state)
            if self._ep_history:
                if self.ctx:
                    self.ctx.episode_count += 1
                ep_id = self.ctx.episode_count if self.ctx else 0
                self._ep_batch.append(
                    {
                        "episode_id": ep_id,
                        "frames": list(self._ep_history),
                        "result": self.end_game_state or "Dogfall",
                        "score": 0,
                    }
                )
                self._ep_history = []
                self._replay_frame_count = 0
            self._flush_ep_batch()
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if obs.first():
            self._termination_signaled = False
            self._prev_hp_my = None
            self._prev_hp_enemy = None
            self._ep_action_log = []
            self._refresh_finetune_model()
            reward_mode = self._beam_params.get("reward_mode", "hp_episodic")
            if reward_mode in ("etg_correct", "etg_offline"):
                self._load_kg_sam()
            if not self._initial_spawned:
                unit_list_my = self.get_my_units_by_type(obs, _MAP["unit_type"])
                unit_list_enemy = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                self._initial_units_my = [(u.x, u.y) for u in unit_list_my]
                self._initial_units_enemy = [(u.x, u.y) for u in unit_list_enemy]
                self._initial_spawned = True
            self._replay_frame_count = 0

            unit_list_my = self.get_my_units_by_type(obs, _MAP["unit_type"])
            unit_list_enemy = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
            self.score_attack_max = sum([item["health"] for item in unit_list_enemy])
            self.score_defense_max = sum([item["health"] for item in unit_list_my])
            self.score_cumulative_attack_last = sum(
                [item["health"] for item in unit_list_enemy]
            )
            self.score_cumulative_defense_last = sum(
                [item["health"] for item in unit_list_my]
            )

        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])

        if not self._termination_signaled and len(enemy_units) == 0:
            self.end_game_state = "Win"
            self.end_game_flag = True
            self._termination_signaled = True
            env.f_result = "win"
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop

        if not self._termination_signaled and len(my_units) == 0:
            self.end_game_state = "Loss"
            self.end_game_flag = True
            self._termination_signaled = True
            env.f_result = "loss"
            if self.end_game_frames > obs.observation.game_loop:
                self.end_game_frames = obs.observation.game_loop

        map_resolution = _ENV_CONFIG["_MAP_RESOLUTION"]
        my_sorted = sorted(my_units, key=lambda u: u.tag)
        enemy_sorted = sorted(enemy_units, key=lambda u: u.tag)
        state_norm = {
            "red_army": [
                (
                    u.x / (map_resolution / 2) - 1.0,
                    1.0 - u.y / (map_resolution / 2),
                    u.health / 45.0,
                )
                for u in my_sorted
            ],
            "blue_army": [
                (
                    u.x / (map_resolution / 2) - 1.0,
                    1.0 - u.y / (map_resolution / 2),
                    u.health / 45.0,
                )
                for u in enemy_sorted
            ],
        }
        state_cluster = self.get_state_cluster(state_norm)
        self._prev_state_cluster = state_cluster

        self._push_status(obs, str(state_cluster), my_units, enemy_units)

        p, s = int(state_cluster[0]), int(state_cluster[1])
        nid, nid_fb = self._resolve_nid(p, s, state_norm)
        if nid is None and not self._nid_norm_states_loaded:
            self._load_nid_norm_states()
            nid, nid_fb = self._resolve_nid(p, s, state_norm)
        action_code = "4c"
        action_to_execute = None
        action_source = "fallback"
        plan_snap = None

        hp_my = int(sum(u.health for u in my_units))
        hp_enemy = int(sum(u.health for u in enemy_units))
        hp_delta = 0
        if self._prev_hp_my is not None:
            hp_delta = (hp_my - self._prev_hp_my) - (hp_enemy - self._prev_hp_enemy)

        if self._mode == "replay":
            replay_action = self._get_next_replay_action()
            if replay_action is not None:
                resolved = self._resolve_action(replay_action)
                if resolved is not None:
                    action_to_execute = resolved
                    action_source = "replay"
                    action_code = replay_action
                else:
                    action_to_execute = self._resolve_action(self._fallback_action)
                    action_source = "replay_fallback"
            else:
                if not self._replay_done:
                    self._replay_done = True
                    self._flush_ep_batch()
                    self.bridge.put_event(
                        {
                            "level": "info",
                            "source": "game",
                            "message": f"回放完成: 共执行 {self._replay_idx}/{len(self._replay_actions)} 步",
                        }
                    )
                    self.bridge.update_status(replay_done=True)
                    try:
                        self.bridge.send_control("pause")
                    except Exception:
                        pass
                return sc2_actions.RAW_FUNCTIONS.no_op()

        elif self._mode != "replay" and nid is not None:
            action_code_raw, evt_type, plan_snap = self._local_decide(
                nid, enemy_count=len(enemy_units)
            )
            if (
                action_code_raw is not None
                and len(enemy_units) <= 1
                and action_code_raw != "4b"
            ):
                self._log_counters["terminal_fix"] = (
                    self._log_counters.get("terminal_fix", 0) + 1
                )
                action_code_raw = "4b"
            if action_code_raw is not None:
                resolved = self._resolve_action(action_code_raw)
                if resolved is not None:
                    action_to_execute = resolved
                    action_source = evt_type
                    action_code = action_code_raw
                else:
                    action_to_execute = self._resolve_action(self._fallback_action)
                    action_source = "fallback"
            else:
                action_to_execute = self._resolve_action(self._fallback_action)
                action_source = "fallback"

        if action_to_execute is None:
            if len(enemy_units) <= 1:
                action_to_execute = self._resolve_action("4b")
                action_source = "terminal_fix"
                self._log_counters["terminal_fix"] = (
                    self._log_counters.get("terminal_fix", 0) + 1
                )
            else:
                action_to_execute = self._resolve_action(self._fallback_action)
                if action_to_execute is None:
                    action_to_execute = "action_ATK_nearest_weakest"
                action_source = "fallback"

        if action_source == "fallback" and action_to_execute in self.actions:
            a_idx = self.actions.index(action_to_execute)
            _ci = str(self._prev_state_cluster[0]) if self._prev_state_cluster else "4"
            action_code = _ci + chr(ord("a") + a_idx)

        self._last_action_executed = action_to_execute
        self._record_action(state_cluster, action_to_execute, action_source)

        if self._mode != "replay":
            c = self._log_counters
            c["total"] += 1
            if nid is None:
                c["nid_none"] += 1
            if nid_fb:
                c["nid_fallback"] = c.get("nid_fallback", 0) + 1
            key = action_source if action_source in c else "fallback"
            c[key] = c.get(key, 0) + 1
            if c["total"] % self._log_interval == 0:
                exploit = c.get("kg_plan", 0) + c.get("kg_follow", 0)
                explore = (
                    c.get("ft_plan", 0)
                    + c.get("kg_relaxed", 0)
                    + c.get("fuzzy_plan", 0)
                    + c.get("fallback", 0)
                )
                print(
                    f"[SUMMARY] frames={c['total']} | nid_none={c['nid_none']} nid_fb={c.get('nid_fallback', 0)} | "
                    f"exploit={exploit} "
                    f"(kg_plan={c.get('kg_plan', 0)} kg_follow={c.get('kg_follow', 0)}) | "
                    f"explore={explore} "
                    f"(ft_plan={c.get('ft_plan', 0)} kg_relaxed={c.get('kg_relaxed', 0)} "
                    f"fuzzy_plan={c.get('fuzzy_plan', 0)} fallback={c.get('fallback', 0)})",
                    flush=True,
                )

        if (my_units or enemy_units) and not (
            self._mode == "replay"
            and self._replay_per_ep > 0
            and self._replay_frame_count >= self._replay_per_ep
        ):
            self._ep_history.append(
                {
                    "state_cluster": (p, s),
                    "nid": nid,
                    "action": action_to_execute,
                    "action_code": action_code,
                    "action_source": action_source,
                    "is_exploration": action_source
                    in ("ft_plan", "kg_relaxed", "fuzzy_plan", "fallback"),
                    "my_count": len(my_units),
                    "enemy_count": len(enemy_units),
                    "hp_my": hp_my,
                    "hp_enemy": hp_enemy,
                    "hp_delta": hp_delta,
                    "game_loop": int(obs.observation.game_loop[0]),
                    "end_game_flag": self.end_game_flag,
                    "plan": plan_snap,
                    "my_units_pos": [
                        {"x": float(u.x), "y": float(u.y), "hp": float(u.health)}
                        for u in my_units
                    ],
                    "enemy_units_pos": [
                        {"x": float(u.x), "y": float(u.y), "hp": float(u.health)}
                        for u in enemy_units
                    ],
                }
            )
            self._replay_frame_count += 1

            if self._mode != "replay" and nid is not None:
                self._ep_action_log.append({"nid": nid, "action_code": action_code})

            if (
                self._prev_hp_my is not None
                and self._mode != "replay"
                and nid is not None
            ):
                hp_delta = (hp_my - self._prev_hp_my) - (hp_enemy - self._prev_hp_enemy)
                reward_mode = self._beam_params.get("reward_mode", "hp_episodic")
                step_reward = hp_delta

                if reward_mode == "etg_correct" and self._kg_sam is not None:
                    sam = self._kg_sam
                    if nid in sam and action_code in sam[nid]:
                        try:
                            gamma = self._beam_params.get("etg_gamma", 0.9)
                            kg_stats = sam[nid][action_code]
                            etg_expected = float(
                                kg_stats.avg_step_reward
                            ) + gamma * float(kg_stats.avg_future_reward)
                            step_reward = hp_delta - etg_expected
                        except (AttributeError, TypeError, ValueError):
                            pass

                ft_model = self._get_finetune_model()
                if ft_model is not None:
                    ft_model.update(nid, action_code, float(step_reward))

        self._prev_hp_my = hp_my
        self._prev_hp_enemy = hp_enemy

        end_flag = self.end_game_flag
        if end_flag and not self._prev_end_game_flag:
            if self._ep_history:
                _ep_counters = dict(self._log_counters)
                c = self._log_counters
                if self._mode != "replay":
                    exploit_count = c.get("kg_plan", 0) + c.get("kg_follow", 0)
                    explore_count = (
                        c.get("ft_plan", 0)
                        + c.get("kg_relaxed", 0)
                        + c.get("fuzzy_plan", 0)
                        + c.get("fallback", 0)
                    )
                    print(
                        f"[EP-END] ep={self.ctx.episode_count if self.ctx else '?'} "
                        f"result={self.end_game_state} frames={c['total']} | "
                        f"exploit={exploit_count} "
                        f"(kg_plan={c.get('kg_plan', 0)} kg_follow={c.get('kg_follow', 0)}) | "
                        f"explore={explore_count} "
                        f"(ft_plan={c.get('ft_plan', 0)} kg_relaxed={c.get('kg_relaxed', 0)} "
                        f"fuzzy_plan={c.get('fuzzy_plan', 0)} fallback={c.get('fallback', 0)}) | "
                        f"terminal_fix={c.get('terminal_fix', 0)} nid_none={c['nid_none']} nid_fb={c.get('nid_fallback', 0)}",
                        flush=True,
                    )

                    if self._ep_action_log:
                        ft_model = self._get_finetune_model()
                        reward_mode = self._beam_params.get(
                            "reward_mode", "hp_episodic"
                        )
                        if ft_model is not None and reward_mode != "hp":
                            result = self.end_game_state or "Dogfall"
                            n_steps = len(self._ep_action_log)
                            if result == "Win":
                                bonus = 50.0 / n_steps
                            elif result == "Loss":
                                bonus = -50.0 / n_steps
                            else:
                                bonus = 0.0
                            if bonus != 0.0:
                                for entry in self._ep_action_log:
                                    ft_model.update(
                                        entry["nid"], entry["action_code"], bonus
                                    )

                    if self._beam_params.get("reward_mode") == "etg_offline":
                        self._ep_completed_count += 1
                        interval = self._beam_params.get("etg_recalibrate_interval", 20)
                        if self._ep_completed_count % interval == 0:
                            self._load_kg_sam()
                            self._etg_recalibrate()

                    self._save_shared_model()

                    for k in self._log_counters:
                        self._log_counters[k] = 0
                    self._ep_action_log = []
            if self.ctx:
                self.ctx.episode_count += 1
            ep_id = self.ctx.episode_count if self.ctx else 0
            self._ep_batch.append(
                {
                    "episode_id": ep_id,
                    "frames": list(self._ep_history),
                    "result": self.end_game_state,
                    "score": float(hp_my - hp_enemy),
                }
            )
            self._ep_history = []
            self._replay_frame_count = 0
            self._flush_ep_batch(counters_snapshot=_ep_counters)
        self._prev_end_game_flag = end_flag

        if self._pending_cluster and hasattr(self, self._pending_cluster):
            self.cluster_result = getattr(self, self._pending_cluster)(obs)

        if hasattr(self, action_to_execute):
            result = getattr(self, action_to_execute)(obs)
            if result is not None:
                return result

        return sc2_actions.RAW_FUNCTIONS.no_op()

    def set_fallback_action(self, action_name: str) -> bool:
        resolved = self._resolve_action(action_name)
        if resolved is not None:
            self._fallback_action = resolved
            return True
        return False

    def get_action_history(self) -> List[Dict[str, Any]]:
        return list(self._action_history)
