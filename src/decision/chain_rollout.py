"""
Chain Rollout — Beam-search-guided single-path rollout with unified tree structure.

Uses plan_action() as the unified planning entry point (from etg_beam_search).
At each step, obtains beam search results, mounts the tree into a RolloutResult,
selects an action, and advances.  The resulting tree preserves all explored
branches for full traceability.

Usage:
    from src.decision.chain_rollout import chain_rollout, RolloutResult

    result = chain_rollout(
        ETG, transitions, start_state=42,
        action_strategy="best_beam",
        score_mode="quality",
        next_state_mode="sample",
        beam_width=3,
        lookahead_steps=5,
        max_rollout_steps=50,
    )
    for nid in result.chosen_path_ids:
        node = result.nodes[nid]
        print(f"Step {node.rollout_depth}: S{node.state} --{node.action}--> S{node.state}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.decision.experience_transition_graph import DecisionExperienceTransitionGraph
from src.decision.etg_beam_search import (
    BeamSearchResult,
    _compute_path_composite,
    get_beam_paths,
    plan_action,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RolloutNode:
    id: int
    parent_id: Optional[int]
    children_ids: List[int] = field(default_factory=list)

    state: int = 0
    action: Optional[str] = None
    beam_id: Optional[int] = None

    quality_score: float = 0.0
    win_rate: float = 0.0
    avg_future_reward: float = 0.0
    avg_step_reward: float = 0.0
    visits: int = 0

    transition_prob: float = 0.0
    cumulative_probability: float = 1.0

    rollout_depth: int = -1
    is_on_chosen_path: bool = False
    is_terminal: bool = False
    is_beam_root: bool = False


@dataclass
class RolloutResult:
    nodes: Dict[int, RolloutNode] = field(default_factory=dict)
    root_id: int = 0
    chosen_path_ids: List[int] = field(default_factory=list)
    termination_reason: str = ""
    beam_results_by_step: Dict[int, List[Dict]] = field(default_factory=dict)
    rollout_mode: str = "single_step"
    plan_segments: List[Dict] = field(default_factory=list)
    total_re_searches: int = 0
    total_backup_switches: int = 0
    switch_points_by_segment: Dict[int, List[Dict]] = field(default_factory=dict)

    def get_chosen_node_at_step(self, rollout_step: int) -> Optional[RolloutNode]:
        for nid in self.chosen_path_ids:
            n = self.nodes[nid]
            if n.rollout_depth == rollout_step:
                return n
        return None

    def get_beam_subtree_at_step(self, rollout_step: int) -> List[int]:
        root_node = self.get_chosen_node_at_step(rollout_step)
        if root_node is None:
            return []
        ids: List[int] = []
        stack = list(root_node.children_ids)
        while stack:
            cid = stack.pop()
            if cid in self.nodes:
                ids.append(cid)
                stack.extend(self.nodes[cid].children_ids)
        return ids

    def compute_subtree_metrics(self, node_id: int) -> Dict[str, Any]:
        node = self.nodes[node_id]
        all_ids: List[int] = []
        stack = list(node.children_ids)
        while stack:
            cid = stack.pop()
            if cid in self.nodes:
                all_ids.append(cid)
                stack.extend(self.nodes[cid].children_ids)
        all_nodes = [self.nodes[nid] for nid in all_ids]
        if not all_nodes:
            return {
                "beam_rank": node.beam_id if node.beam_id is not None else 0,
                "max_quality": node.quality_score,
                "avg_win_rate": node.win_rate,
                "avg_future_reward": node.avg_future_reward,
                "n_descendants": 0,
            }
        return {
            "beam_rank": min(n.beam_id for n in all_nodes if n.beam_id is not None),
            "max_quality": max(n.quality_score for n in all_nodes),
            "avg_win_rate": float(np.mean([n.win_rate for n in all_nodes])),
            "avg_future_reward": float(
                np.mean([n.avg_future_reward for n in all_nodes])
            ),
            "n_descendants": len(all_nodes),
        }

    def get_action_candidates_at_step(self, rollout_step: int) -> List[Dict]:
        root_node = self.get_chosen_node_at_step(rollout_step)
        if root_node is None:
            return []
        children = [self.nodes[cid] for cid in root_node.children_ids]
        groups: Dict[str, List[RolloutNode]] = defaultdict(list)
        for c in children:
            if c.action:
                groups[c.action].append(c)
        result = []
        for action, child_nodes in groups.items():
            all_ids: List[int] = []
            stack = [n.id for n in child_nodes]
            while stack:
                cid = stack.pop()
                if cid in self.nodes:
                    all_ids.append(cid)
                    stack.extend(self.nodes[cid].children_ids)
            all_nodes = [self.nodes[nid] for nid in all_ids]
            if not all_nodes:
                all_nodes = child_nodes
            result.append(
                {
                    "action": action,
                    "child_node_ids": [n.id for n in child_nodes],
                    "beam_rank": min(
                        n.beam_id for n in child_nodes if n.beam_id is not None
                    ),
                    "max_quality": max(n.quality_score for n in all_nodes),
                    "avg_win_rate": float(np.mean([n.win_rate for n in all_nodes])),
                    "avg_future_reward": float(
                        np.mean([n.avg_future_reward for n in all_nodes])
                    ),
                    "n_descendants": len(all_ids),
                }
            )
        return result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACTION_STRATEGIES = [
    "best_beam",
    "best_subtree_quality",
    "best_subtree_winrate",
    "highest_transition_prob",
    "random_beam",
    "epsilon_greedy",
]

_NEXT_STATE_MODES = ["sample", "highest_prob"]


@dataclass
class SwitchPoint:
    plan_position: int
    predicted_state: int
    backup_path_idx: int
    backup_step_in_path: int
    remaining_actions: List[str] = field(default_factory=list)
    match_type: str = "exact"


@dataclass
class PlanSegment:
    segment_type: str
    source_path_idx: int
    actions_planned: List[str] = field(default_factory=list)
    divergence_step: int = -1
    divergence_type: str = "none"


def _find_fork_step(
    main_path: List[BeamSearchResult], candidate_path: List[BeamSearchResult]
) -> int:
    limit = min(len(main_path), len(candidate_path))
    for i in range(limit):
        if main_path[i].state != candidate_path[i].state:
            return i
    return limit


def build_switching_map(
    main_path: List[BeamSearchResult],
    all_paths: List[List[BeamSearchResult]],
    score_threshold: float = 0.3,
    max_per_fork: int = 3,
) -> Dict[int, List[SwitchPoint]]:
    main_score = _compute_path_composite(main_path)
    threshold = main_score * score_threshold

    scored = []
    for idx, p in enumerate(all_paths):
        if p is main_path:
            continue
        s = _compute_path_composite(p)
        if s >= threshold:
            fork = _find_fork_step(main_path, p)
            scored.append((idx, p, s, fork))

    fork_groups: Dict[int, List[Tuple[int, List[BeamSearchResult], float, int]]] = (
        defaultdict(list)
    )
    for item in scored:
        fork_groups[item[3]].append(item)

    result: Dict[int, List[SwitchPoint]] = defaultdict(list)
    for fork_step, items in fork_groups.items():
        items.sort(key=lambda x: x[2], reverse=True)
        for idx, p, s, fork in items[:max_per_fork]:
            if fork < len(p) and fork < len(main_path):
                for step_in_path in range(fork, len(p)):
                    plan_pos = fork
                    predicted_state = main_path[fork].state
                    remaining = [
                        p[j].action
                        for j in range(step_in_path + 1, len(p))
                        if p[j].action
                    ]
                    sp = SwitchPoint(
                        plan_position=plan_pos,
                        predicted_state=predicted_state,
                        backup_path_idx=idx,
                        backup_step_in_path=step_in_path,
                        remaining_actions=remaining,
                        match_type="exact",
                    )
                    result[plan_pos].append(sp)
    return dict(result)


def find_closest_switch_point(
    actual_state: int,
    switching_points: List[SwitchPoint],
    dist_matrix: Optional[np.ndarray] = None,
    distance_threshold: float = 0.2,
) -> Optional[SwitchPoint]:
    for sp in switching_points:
        if sp.predicted_state == actual_state:
            return sp
    if dist_matrix is not None:
        best_sp = None
        best_dist = distance_threshold
        for sp in switching_points:
            try:
                d = float(dist_matrix[actual_state, sp.predicted_state])
                if not np.isfinite(d) or np.isnan(d):
                    continue
                if d < best_dist:
                    best_dist = d
                    best_sp = sp
            except (IndexError, TypeError):
                continue
        if best_sp is not None:
            return replace(best_sp, match_type="distance")
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pick_next_state(
    next_states: Dict[int, int],
    mode: str,
    rng: np.random.RandomState,
    visited_counts: Dict[int, int],
    max_state_revisits: int,
) -> Tuple[Optional[int], float]:
    if not next_states:
        return None, 0.0
    filtered = {
        ns: count
        for ns, count in next_states.items()
        if visited_counts.get(ns, 0) < max_state_revisits
    }
    if not filtered:
        return None, 0.0
    total = sum(filtered.values())
    if total <= 0:
        return None, 0.0
    if mode == "highest_prob":
        ns = max(filtered, key=lambda k: filtered[k])
        return ns, filtered[ns] / total
    states = list(filtered.keys())
    counts = list(filtered.values())
    probs = [c / total for c in counts]
    idx = rng.choice(len(states), p=probs)
    return states[idx], probs[idx]


def _mount_beam_results(
    beam_results: List[BeamSearchResult],
    parent_node_id: int,
    nodes: Dict[int, RolloutNode],
    next_id: int,
) -> Tuple[int, Dict[int, int]]:
    if not beam_results:
        return next_id, {}

    idx_to_node_id: Dict[int, int] = {0: parent_node_id}

    for i, r in enumerate(beam_results):
        if r.step == 0:
            continue

        parent_nid = idx_to_node_id.get(r.parent_idx)
        if parent_nid is None:
            continue

        nid = next_id
        next_id += 1
        idx_to_node_id[i] = nid

        node = RolloutNode(
            id=nid,
            parent_id=parent_nid,
            state=r.state,
            action=r.action,
            beam_id=r.beam_id,
            quality_score=r.quality_score,
            win_rate=r.win_rate,
            avg_future_reward=r.avg_future_reward,
            avg_step_reward=r.avg_step_reward,
            visits=0,
            cumulative_probability=r.cumulative_probability,
        )
        nodes[nid] = node
        if nid not in nodes[parent_nid].children_ids:
            nodes[parent_nid].children_ids.append(nid)

    idx_map: Dict[int, int] = {}
    for i, r in enumerate(beam_results):
        if i in idx_to_node_id:
            idx_map[i] = idx_to_node_id[i]

    return next_id, idx_map


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _advance_to_next_state(
    result: RolloutResult,
    current_id: int,
    current_state: int,
    chosen_action: str,
    chosen_next_state: int,
    chosen_trans_prob: float,
    ETG: DecisionExperienceTransitionGraph,
    transitions: Dict[int, Dict[str, Dict]],
    next_id: int,
    cum_prob: float,
    rollout_depth: int,
    visited_counts: Dict[int, int],
    is_on_chosen_path: bool = True,
) -> Tuple[int, int, float]:
    quality = ETG.get_action_quality(current_state, chosen_action)
    qs = quality.get("quality_score", 0.0) if quality else 0.0
    wr = quality.get("win_rate", 0.0) if quality else 0.0
    afr = quality.get("avg_future_reward", 0.0) if quality else 0.0
    asr_val = quality.get("avg_step_reward", 0.0) if quality else 0.0
    vis = quality.get("visits", 0) if quality else 0

    next_trans = transitions.get(chosen_next_state, {})
    is_terminal = bool(next_trans.get("__terminal__"))

    visited_counts[chosen_next_state] = visited_counts.get(chosen_next_state, 0) + 1

    current_node = result.nodes[current_id]
    beam_predicted_child: Optional[RolloutNode] = None
    for cid in current_node.children_ids:
        child = result.nodes[cid]
        if child.action == chosen_action and child.state == chosen_next_state:
            beam_predicted_child = child
            break

    if beam_predicted_child is not None:
        beam_predicted_child.is_on_chosen_path = is_on_chosen_path
        beam_predicted_child.rollout_depth = rollout_depth
        beam_predicted_child.is_beam_root = True
        beam_predicted_child.transition_prob = chosen_trans_prob
        beam_predicted_child.visits = vis
        new_id = beam_predicted_child.id
    else:
        nid = next_id
        next_id += 1
        new_node = RolloutNode(
            id=nid,
            parent_id=current_id,
            state=chosen_next_state,
            action=chosen_action,
            beam_id=None,
            quality_score=round(qs, 4),
            win_rate=round(wr, 4),
            avg_future_reward=round(afr, 4),
            avg_step_reward=round(asr_val, 4),
            visits=vis,
            transition_prob=round(chosen_trans_prob, 4),
            cumulative_probability=round(cum_prob, 6),
            rollout_depth=rollout_depth,
            is_on_chosen_path=is_on_chosen_path,
            is_terminal=is_terminal,
            is_beam_root=True,
        )
        result.nodes[nid] = new_node
        current_node.children_ids.append(nid)
        new_id = nid

    if is_on_chosen_path:
        result.chosen_path_ids.append(new_id)

    if is_terminal:
        result.nodes[new_id].is_terminal = True

    return new_id, next_id, cum_prob


def _execute_single_action(
    state_trans: Dict[str, Dict],
    planned_action: str,
    next_state_mode: str,
    rng: np.random.RandomState,
    visited_counts: Dict[int, int],
    max_state_revisits: int,
) -> Tuple[Optional[str], Optional[int], float]:
    trans_info = state_trans.get(planned_action, {})
    if not trans_info:
        return None, None, 0.0
    next_states = trans_info.get("next_states", {})
    if not next_states:
        return None, None, 0.0
    ns, tp = _pick_next_state(
        next_states, next_state_mode, rng, visited_counts, max_state_revisits
    )
    if ns is not None:
        return planned_action, ns, tp
    return None, None, 0.0


def _chain_rollout_single_step(
    ETG: DecisionExperienceTransitionGraph,
    transitions: Dict[int, Dict[str, Dict]],
    start_state: int,
    result: RolloutResult,
    next_id: int,
    rng: np.random.RandomState,
    score_mode: str,
    action_strategy: str,
    next_state_mode: str,
    beam_width: int,
    lookahead_steps: int,
    max_rollout_steps: int,
    min_visits: int,
    min_cum_prob: float,
    max_state_revisits: int,
    discount_factor: float,
    epsilon: float,
) -> RolloutResult:
    current_id = result.root_id
    visited_counts: Dict[int, int] = {start_state: 1}
    cum_prob = 1.0
    termination_reason = "无可用动作"

    for step in range(max_rollout_steps):
        current_node = result.nodes[current_id]
        current_state = current_node.state

        state_trans = transitions.get(current_state, {})
        if not state_trans or state_trans.get("__terminal__"):
            current_node.is_terminal = True
            termination_reason = "终端状态"
            break

        plan = plan_action(
            ETG,
            transitions,
            current_state,
            beam_width=beam_width,
            max_steps=lookahead_steps,
            min_visits=min_visits,
            min_cum_prob=min_cum_prob,
            score_mode=score_mode,
            max_state_revisits=max_state_revisits,
            discount_factor=discount_factor,
            action_strategy=action_strategy,
            epsilon=epsilon,
            rng_seed=None,
        )

        if plan.recommended_action is None:
            termination_reason = "无可用动作"
            break

        beam_results = plan.beam_results

        result.beam_results_by_step[step] = [
            {
                "step": r.step,
                "state": r.state,
                "action": r.action,
                "cumulative_probability": r.cumulative_probability,
                "quality_score": r.quality_score,
                "win_rate": r.win_rate,
                "avg_step_reward": r.avg_step_reward,
                "avg_future_reward": r.avg_future_reward,
                "beam_id": r.beam_id,
                "parent_idx": r.parent_idx,
            }
            for r in beam_results
        ]

        next_id, _ = _mount_beam_results(
            beam_results, current_id, result.nodes, next_id
        )

        ordered_actions = plan.ranked_actions

        if not ordered_actions:
            termination_reason = "无可用动作"
            break

        chosen_action = None
        chosen_next_state = None
        chosen_trans_prob = 0.0

        for candidate_action in ordered_actions:
            trans_info = state_trans.get(candidate_action, {})
            if not trans_info:
                continue
            next_states = trans_info.get("next_states", {})
            if not next_states:
                continue
            ns, tp = _pick_next_state(
                next_states, next_state_mode, rng, visited_counts, max_state_revisits
            )
            if ns is not None:
                chosen_action = candidate_action
                chosen_next_state = ns
                chosen_trans_prob = tp
                break

        if chosen_action is None:
            termination_reason = "无可用动作"
            break

        cum_prob = cum_prob * chosen_trans_prob * discount_factor
        if cum_prob < min_cum_prob:
            termination_reason = "累积概率过低"
            break

        current_id, next_id, cum_prob = _advance_to_next_state(
            result,
            current_id,
            current_state,
            chosen_action,
            chosen_next_state,
            chosen_trans_prob,
            ETG,
            transitions,
            next_id,
            cum_prob,
            step + 1,
            visited_counts,
        )

        if result.nodes[current_id].is_terminal:
            termination_reason = "终端状态"
            break

    if step + 1 >= max_rollout_steps and termination_reason == "无可用动作":
        termination_reason = "达到最大推演步数"

    result.termination_reason = termination_reason
    return result


def _chain_rollout_multi_step(
    ETG: DecisionExperienceTransitionGraph,
    transitions: Dict[int, Dict[str, Dict]],
    start_state: int,
    result: RolloutResult,
    next_id: int,
    rng: np.random.RandomState,
    score_mode: str,
    action_strategy: str,
    next_state_mode: str,
    beam_width: int,
    lookahead_steps: int,
    max_rollout_steps: int,
    min_visits: int,
    min_cum_prob: float,
    max_state_revisits: int,
    discount_factor: float,
    epsilon: float,
    enable_backup: bool,
    score_threshold: float,
    distance_threshold: float,
    dist_matrix: Optional[np.ndarray],
) -> RolloutResult:
    current_id = result.root_id
    current_state = start_state
    visited_counts: Dict[int, int] = {start_state: 1}
    cum_prob = 1.0
    termination_reason = "无可用动作"
    total_steps = 0
    segment_count = 0
    max_segments = max_rollout_steps

    while segment_count < max_segments:
        state_trans = transitions.get(current_state, {})
        if not state_trans or state_trans.get("__terminal__"):
            result.nodes[current_id].is_terminal = True
            termination_reason = "终端状态"
            break

        plan = plan_action(
            ETG,
            transitions,
            current_state,
            beam_width=beam_width,
            max_steps=lookahead_steps,
            min_visits=min_visits,
            min_cum_prob=min_cum_prob,
            score_mode=score_mode,
            max_state_revisits=max_state_revisits,
            discount_factor=discount_factor,
            action_strategy=action_strategy,
            epsilon=epsilon,
            rng_seed=None,
        )

        if plan.recommended_action is None:
            termination_reason = "无可用动作"
            break

        beam_results = plan.beam_results

        result.beam_results_by_step[total_steps] = [
            {
                "step": r.step,
                "state": r.state,
                "action": r.action,
                "cumulative_probability": r.cumulative_probability,
                "quality_score": r.quality_score,
                "win_rate": r.win_rate,
                "avg_step_reward": r.avg_step_reward,
                "avg_future_reward": r.avg_future_reward,
                "beam_id": r.beam_id,
                "parent_idx": r.parent_idx,
            }
            for r in beam_results
        ]

        next_id, _ = _mount_beam_results(
            beam_results, current_id, result.nodes, next_id
        )

        all_paths = plan.beam_paths
        if not all_paths:
            termination_reason = "无搜索路径"
            break

        main_path_idx = plan.best_path_index
        main_path = all_paths[main_path_idx]

        switching_map: Dict[int, List[SwitchPoint]] = {}
        if enable_backup:
            switching_map = build_switching_map(
                main_path, all_paths, score_threshold=score_threshold
            )

        seg_type = "re_search" if segment_count > 0 else "initial_plan"
        segment = PlanSegment(
            segment_type=seg_type,
            source_path_idx=main_path_idx,
        )
        actions_planned: List[str] = []
        divergence_step = -1
        divergence_type = "none"
        did_backup_switch = False
        re_search_needed = True

        for plan_step in range(len(main_path) - 1):
            if total_steps >= max_rollout_steps:
                re_search_needed = False
                termination_reason = "达到最大推演步数"
                break

            planned_node = main_path[plan_step + 1]
            planned_action = planned_node.action
            predicted_next = planned_node.state
            actions_planned.append(planned_action)

            trans_at_current = transitions.get(current_state, {})
            chosen_action, chosen_next_state, chosen_trans_prob = (
                _execute_single_action(
                    trans_at_current,
                    planned_action,
                    next_state_mode,
                    rng,
                    visited_counts,
                    max_state_revisits,
                )
            )

            if chosen_action is None:
                divergence_step = plan_step
                divergence_type = "no_valid_transition"
                break

            cum_prob = cum_prob * chosen_trans_prob * discount_factor
            if cum_prob < min_cum_prob:
                divergence_step = plan_step
                divergence_type = "low_cum_prob"
                break

            current_id, next_id, cum_prob = _advance_to_next_state(
                result,
                current_id,
                current_state,
                chosen_action,
                chosen_next_state,
                chosen_trans_prob,
                ETG,
                transitions,
                next_id,
                cum_prob,
                total_steps + 1,
                visited_counts,
            )
            total_steps += 1
            current_state = chosen_next_state

            if result.nodes[current_id].is_terminal:
                termination_reason = "终端状态"
                re_search_needed = False
                break

            if chosen_next_state == predicted_next:
                continue

            points_at_pos = switching_map.get(plan_step, [])
            if points_at_pos:
                sp = find_closest_switch_point(
                    chosen_next_state,
                    points_at_pos,
                    dist_matrix,
                    distance_threshold,
                )
                if sp is not None:
                    result.total_backup_switches += 1
                    did_backup_switch = True
                    divergence_step = plan_step
                    divergence_type = "backup_switch"
                    backup_path = all_paths[sp.backup_path_idx]
                    result.switch_points_by_segment[segment_count] = [
                        {
                            "plan_position": s.plan_position,
                            "predicted_state": s.predicted_state,
                            "backup_path_idx": s.backup_path_idx,
                            "backup_step_in_path": s.backup_step_in_path,
                            "remaining_actions": s.remaining_actions,
                            "match_type": s.match_type,
                        }
                        for s in points_at_pos
                    ]

                    for rem_step in range(sp.backup_step_in_path + 1, len(backup_path)):
                        if total_steps >= max_rollout_steps:
                            termination_reason = "达到最大推演步数"
                            re_search_needed = False
                            break

                        rem_node = backup_path[rem_step]
                        if not rem_node.action:
                            break

                        trans_at_current = transitions.get(current_state, {})
                        rem_action, rem_ns, rem_tp = _execute_single_action(
                            trans_at_current,
                            rem_node.action,
                            next_state_mode,
                            rng,
                            visited_counts,
                            max_state_revisits,
                        )
                        if rem_action is None:
                            break

                        cum_prob = cum_prob * rem_tp * discount_factor
                        if cum_prob < min_cum_prob:
                            break

                        actions_planned.append(rem_action)
                        current_id, next_id, cum_prob = _advance_to_next_state(
                            result,
                            current_id,
                            current_state,
                            rem_action,
                            rem_ns,
                            rem_tp,
                            ETG,
                            transitions,
                            next_id,
                            cum_prob,
                            total_steps + 1,
                            visited_counts,
                        )
                        total_steps += 1
                        current_state = rem_ns

                        if result.nodes[current_id].is_terminal:
                            termination_reason = "终端状态"
                            re_search_needed = False
                            break
                    break

            divergence_step = plan_step
            divergence_type = "re_search"
            break

        segment.actions_planned = actions_planned
        segment.divergence_step = divergence_step
        segment.divergence_type = divergence_type
        result.plan_segments.append(
            {
                "segment_type": segment.segment_type,
                "source_path_idx": segment.source_path_idx,
                "actions_planned": segment.actions_planned,
                "divergence_step": segment.divergence_step,
                "divergence_type": segment.divergence_type,
            }
        )

        if not re_search_needed or termination_reason in (
            "终端状态",
            "达到最大推演步数",
            "累积概率过低",
        ):
            break

        if divergence_type == "re_search":
            result.total_re_searches += 1

        segment_count += 1

    if total_steps >= max_rollout_steps and termination_reason == "无可用动作":
        termination_reason = "达到最大推演步数"

    result.termination_reason = termination_reason
    return result


def chain_rollout(
    ETG: DecisionExperienceTransitionGraph,
    transitions: Dict[int, Dict[str, Dict]],
    start_state: int,
    score_mode: str = "quality",
    action_strategy: str = "best_beam",
    next_state_mode: str = "sample",
    beam_width: int = 3,
    lookahead_steps: int = 5,
    max_rollout_steps: int = 50,
    min_visits: int = 1,
    min_cum_prob: float = 0.01,
    max_state_revisits: int = 2,
    discount_factor: float = 0.9,
    epsilon: float = 0.1,
    rng_seed: Optional[int] = None,
    rollout_mode: str = "single_step",
    enable_backup: bool = False,
    score_threshold: float = 0.3,
    distance_threshold: float = 0.2,
    dist_matrix: Optional[np.ndarray] = None,
) -> RolloutResult:
    rng = np.random.RandomState(rng_seed)
    result = RolloutResult()
    result.rollout_mode = rollout_mode
    next_id = 0

    root_node = RolloutNode(
        id=next_id,
        parent_id=None,
        state=start_state,
        rollout_depth=0,
        is_on_chosen_path=True,
        is_beam_root=True,
        cumulative_probability=1.0,
    )
    result.nodes[next_id] = root_node
    result.root_id = next_id
    result.chosen_path_ids.append(next_id)
    next_id += 1

    if rollout_mode == "multi_step":
        return _chain_rollout_multi_step(
            ETG,
            transitions,
            start_state,
            result,
            next_id,
            rng,
            score_mode,
            action_strategy,
            next_state_mode,
            beam_width,
            lookahead_steps,
            max_rollout_steps,
            min_visits,
            min_cum_prob,
            max_state_revisits,
            discount_factor,
            epsilon,
            enable_backup,
            score_threshold,
            distance_threshold,
            dist_matrix,
        )

    return _chain_rollout_single_step(
        ETG,
        transitions,
        start_state,
        result,
        next_id,
        rng,
        score_mode,
        action_strategy,
        next_state_mode,
        beam_width,
        lookahead_steps,
        max_rollout_steps,
        min_visits,
        min_cum_prob,
        max_state_revisits,
        discount_factor,
        epsilon,
    )
