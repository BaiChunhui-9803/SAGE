"""
KG Beam Search — Chain Reasoning Prediction Engine

Performs beam search over the knowledge graph to predict multi-step
state-action trajectories, starting from a given state.

Core idea:
  At each step, expand all active beams by considering candidate actions
  (from KG quality ranking) and their transition probabilities (from
  transitions dict).  Keep only the top-B beams by a configurable scoring
  metric, pruning branches whose cumulative probability drops below a
  threshold.

Unified planning entry point:
    from src.decision.kg_beam_search import plan_action

    plan = plan_action(kg, transitions, state=42,
                       action_strategy="best_beam")
    print(plan.recommended_action, plan.action_plan)

Backward-compatible shortcut:
    action, info = find_optimal_action(kg, transitions, state=42)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from src.decision.knowledge_graph import DecisionKnowledgeGraph


_COMPOSITE_WEIGHTS: Tuple[float, float, float, float] = (0.30, 0.30, 0.30, 0.10)


@dataclass
class BeamSearchResult:
    step: int
    state: int
    action: str
    cumulative_probability: float
    quality_score: float
    win_rate: float
    avg_step_reward: float
    avg_future_reward: float
    beam_id: int
    parent_idx: Optional[int]


@dataclass
class PlanResult:
    recommended_action: Optional[str] = None
    ranked_actions: List[str] = field(default_factory=list)
    action_plan: List[str] = field(default_factory=list)
    planned_states: List[int] = field(default_factory=list)
    beam_results: List[BeamSearchResult] = field(default_factory=list)
    beam_paths: List[List[BeamSearchResult]] = field(default_factory=list)
    best_path_index: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


class _BeamNode:
    __slots__ = (
        "state",
        "action",
        "cum_prob",
        "score",
        "step",
        "win_rate",
        "avg_step_reward",
        "avg_future_reward",
        "path",
        "parent_idx",
        "visited_counts",
    )

    def __init__(
        self,
        state: int,
        action: Optional[str],
        cum_prob: float,
        score: float,
        step: int,
        win_rate: float = 0.0,
        avg_step_reward: float = 0.0,
        avg_future_reward: float = 0.0,
        path: Optional[List[_BeamNode]] = None,
        parent_idx: Optional[int] = None,
        visited_counts: Optional[Dict[int, int]] = None,
    ):
        self.state = state
        self.action = action
        self.cum_prob = cum_prob
        self.score = score
        self.step = step
        self.win_rate = win_rate
        self.avg_step_reward = avg_step_reward
        self.avg_future_reward = avg_future_reward
        self.path = path if path is not None else []
        self.parent_idx = parent_idx
        self.visited_counts = visited_counts if visited_counts is not None else {}


def _get_score(
    quality: Dict[str, Any],
    score_mode: str,
) -> float:
    if score_mode == "future_reward":
        return quality.get("avg_future_reward", 0.0)
    if score_mode == "win_rate":
        return quality.get("win_rate", 0.0)
    return quality.get("quality_score", 0.0)


def beam_search_predict(
    kg: DecisionKnowledgeGraph,
    transitions: Dict[int, Dict[str, Dict]],
    start_state: int,
    beam_width: int = 3,
    max_steps: int = 5,
    min_visits: int = 1,
    min_cum_prob: float = 0.01,
    score_mode: str = "quality",
    max_state_revisits: int = 2,
    discount_factor: float = 0.9,
    finetune_model: Any = None,
    dist_matrix: Any = None,
) -> List[BeamSearchResult]:
    """
    Beam search over the KG transition graph.

    Args:
        kg: DecisionKnowledgeGraph instance.
        transitions: {state: {action: {"next_states": {s: count}, ...}, ...}}
        start_state: Starting state ID.
        beam_width: Number of beams to keep at each step.
        max_steps: Maximum prediction horizon.
        min_visits: Minimum visit count for an action to be considered.
        min_cum_prob: Prune branches whose cumulative probability < threshold.
        score_mode: "quality" | "future_reward" | "win_rate"
        max_state_revisits: Max times a single state can appear in one beam path.
            Set to 1 to completely forbid revisits, 2 to allow one revisit, etc.
        discount_factor: Per-step discount applied to cumulative probability.
            1.0 = no discount, 0.9 = 10% decay per step.

    Returns:
        Flat list of BeamSearchResult for every expanded node (across all beams),
        with beam_id and parent_idx so that callers can reconstruct the tree.
    """
    candidates_per_action: int = max(beam_width * 3, 9)

    all_nodes: List[BeamSearchResult] = []

    def _record(node: _BeamNode, beam_id: int, parent_idx: Optional[int]):
        all_nodes.append(
            BeamSearchResult(
                step=node.step,
                state=node.state,
                action=node.action or "",
                cumulative_probability=round(node.cum_prob, 6),
                quality_score=round(node.score, 4),
                win_rate=round(node.win_rate, 4),
                avg_step_reward=round(node.avg_step_reward, 4),
                avg_future_reward=round(node.avg_future_reward, 4),
                beam_id=beam_id,
                parent_idx=parent_idx,
            )
        )

    root = _BeamNode(
        state=start_state,
        action=None,
        cum_prob=1.0,
        score=0.0,
        step=0,
        path=[],
        parent_idx=None,
        visited_counts={start_state: 1},
    )
    _record(root, beam_id=0, parent_idx=None)

    beam_heads: List[Tuple[int, _BeamNode]] = [(0, root)]

    for step in range(max_steps):
        expanded: List[Tuple[int, _BeamNode]] = []

        for head_idx, beam in beam_heads:
            top_actions = kg.get_top_k_actions(
                state=beam.state,
                k=candidates_per_action,
                min_visits=min_visits,
                metric="quality_score",
            )

            if not top_actions and finetune_model is not None:
                ft_ranked = finetune_model.rank_actions_by_finetune(
                    beam.state, dist_matrix
                )
                if ft_ranked:
                    top_actions = [
                        (
                            ac,
                            {
                                "quality_score": finetune_model.smooth_q(
                                    beam.state, ac, dist_matrix
                                )
                            },
                        )
                        for ac in ft_ranked[:candidates_per_action]
                    ]

            if not top_actions:
                continue

            for action, quality in top_actions:
                state_trans = transitions.get(beam.state, {})
                if not state_trans or state_trans.get("__terminal__"):
                    continue

                trans_info = state_trans.get(action, {})
                if not trans_info:
                    continue

                next_states = trans_info.get("next_states", {})
                if not next_states:
                    continue

                total_count = sum(next_states.values())
                score = _get_score(quality, score_mode)

                for ns, count in sorted(
                    next_states.items(), key=lambda x: x[1], reverse=True
                ):
                    current_visits = beam.visited_counts.get(ns, 0)
                    if current_visits >= max_state_revisits:
                        continue

                    prob = count / total_count if total_count > 0 else 0.0
                    cum_prob = beam.cum_prob * prob * (discount_factor ** (step + 1))
                    if cum_prob < min_cum_prob:
                        continue

                    ns_quality = kg.get_action_quality(beam.state, action)
                    new_visited = dict(beam.visited_counts)
                    new_visited[ns] = new_visited.get(ns, 0) + 1
                    node = _BeamNode(
                        state=ns,
                        action=action,
                        cum_prob=cum_prob,
                        score=score,
                        step=step + 1,
                        win_rate=ns_quality["win_rate"] if ns_quality else 0.0,
                        avg_step_reward=ns_quality["avg_step_reward"]
                        if ns_quality
                        else 0.0,
                        avg_future_reward=ns_quality["avg_future_reward"]
                        if ns_quality
                        else 0.0,
                        path=beam.path + [beam],
                        parent_idx=None,
                        visited_counts=new_visited,
                    )
                    expanded.append((head_idx, node))

        if not expanded:
            break

        expanded.sort(key=lambda x: (x[1].score, x[1].cum_prob), reverse=True)

        trimmed = expanded[:beam_width]

        beam_heads = []
        for parent_head_idx, beam in trimmed:
            new_idx = len(all_nodes)
            beam_id = beam_heads.__len__()
            _record(beam, beam_id=beam_id, parent_idx=parent_head_idx)
            beam_heads.append((new_idx, beam))

    return all_nodes


def _compute_path_composite(path: List[BeamSearchResult]) -> float:
    if not path:
        return 0.0
    non_root = path[1:] if len(path) > 1 else path
    if not non_root:
        return 1.0
    cum_prob = path[-1].cumulative_probability
    avg_quality = float(np.mean([r.quality_score for r in non_root]))
    avg_win_rate = float(np.mean([r.win_rate for r in non_root]))
    avg_reward = float(np.mean([r.avg_future_reward for r in non_root]))
    vals = [cum_prob, avg_quality, avg_win_rate, avg_reward]
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return 0.5
    normed = [(v - mn) / (mx - mn) for v in vals]
    w_conf, w_qual, w_wr, w_rwd = _COMPOSITE_WEIGHTS
    return (
        w_conf * normed[0] + w_qual * normed[1] + w_wr * normed[2] + w_rwd * normed[3]
    )


def _rank_actions_from_beam(
    beam_results: List[BeamSearchResult],
    state_trans: Dict[str, Dict],
    action_strategy: str = "best_beam",
    epsilon: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
) -> List[str]:
    if not beam_results or len(beam_results) <= 1:
        return []

    if rng is None:
        rng = np.random.RandomState()

    root = beam_results[0]
    direct_children: Dict[int, BeamSearchResult] = {}
    for i, r in enumerate(beam_results):
        if i == 0:
            continue
        if r.parent_idx == 0:
            direct_children[i] = r

    groups: Dict[str, List[BeamSearchResult]] = defaultdict(list)
    all_descendants: Dict[str, List[int]] = defaultdict(list)

    child_indices = set(direct_children.keys())
    for i, r in enumerate(beam_results):
        if i == 0:
            continue
        current = r
        chain = [i]
        ancestor = current.parent_idx
        while ancestor is not None and ancestor != 0:
            chain.append(ancestor)
            if ancestor < len(beam_results):
                ancestor = beam_results[ancestor].parent_idx
            else:
                break
        for ci in child_indices:
            if ci in chain:
                action = direct_children[ci].action
                if action:
                    groups[action].append(r)
                    all_descendants[action].extend(chain)
                break

    for ci, child in direct_children.items():
        action = child.action
        if action and action not in groups:
            groups[action] = []
            all_descendants[action] = [ci]

    if not groups:
        seen = set()
        for ci, child in direct_children.items():
            if child.action and child.action not in seen:
                seen.add(child.action)
        return list(seen)

    action_metrics: Dict[str, Dict[str, Any]] = {}
    for action, child_nodes in groups.items():
        desc_ids = list(set(all_descendants.get(action, [])))
        desc_nodes = [beam_results[d] for d in desc_ids if d < len(beam_results)]
        all_nodes = desc_nodes if desc_nodes else child_nodes

        trans_total = 0
        ti = state_trans.get(action, {})
        ns = ti.get("next_states", {})
        if ns:
            trans_total = sum(ns.values())

        beam_ids = [n.beam_id for n in child_nodes]
        action_metrics[action] = {
            "beam_rank": min(beam_ids) if beam_ids else 999,
            "max_quality": max(n.quality_score for n in all_nodes),
            "avg_win_rate": float(np.mean([n.win_rate for n in all_nodes])),
            "avg_future_reward": float(
                np.mean([n.avg_future_reward for n in all_nodes])
            ),
            "trans_total": trans_total,
        }

    items = list(action_metrics.items())

    if action_strategy == "best_beam":
        items.sort(key=lambda x: x[1]["beam_rank"])
    elif action_strategy == "best_subtree_quality":
        items.sort(key=lambda x: x[1]["max_quality"], reverse=True)
    elif action_strategy == "best_subtree_winrate":
        items.sort(key=lambda x: x[1]["avg_win_rate"], reverse=True)
    elif action_strategy == "highest_transition_prob":
        items.sort(key=lambda x: x[1]["trans_total"], reverse=True)
    elif action_strategy == "random_beam":
        rng.shuffle(items)
    elif action_strategy == "epsilon_greedy":
        if rng.random() < epsilon and len(items) > 1:
            rng.shuffle(items)
        else:
            items.sort(key=lambda x: x[1]["beam_rank"])

    return [action for action, _ in items]


def plan_action(
    kg: DecisionKnowledgeGraph,
    transitions: Dict[int, Dict[str, Dict]],
    state: int,
    beam_width: int = 3,
    max_steps: int = 5,
    min_visits: int = 1,
    min_cum_prob: float = 0.01,
    score_mode: str = "quality",
    max_state_revisits: int = 2,
    discount_factor: float = 0.9,
    action_strategy: str = "best_beam",
    epsilon: float = 0.1,
    rng_seed: Optional[int] = None,
    finetune_model: Any = None,
    dist_matrix: Any = None,
) -> PlanResult:
    rng = np.random.RandomState(rng_seed)

    beam_results = beam_search_predict(
        kg,
        transitions,
        state,
        beam_width=beam_width,
        max_steps=max_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        score_mode=score_mode,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
        finetune_model=finetune_model,
        dist_matrix=dist_matrix,
    )

    if not beam_results:
        return PlanResult(
            recommended_action=None,
            metrics={"reason": "no_actions"},
        )

    beam_paths = get_beam_paths(beam_results)

    if not beam_paths:
        return PlanResult(
            recommended_action=None,
            beam_results=beam_results,
            metrics={"reason": "no_paths"},
        )

    composites = [_compute_path_composite(p) for p in beam_paths]
    best_path_index = int(np.argmax(composites))
    best_path = beam_paths[best_path_index]

    action_plan: List[str] = []
    planned_states: List[int] = []
    for node in best_path:
        planned_states.append(node.state)
        if node.action:
            action_plan.append(node.action)

    ranked_actions: List[str] = []
    recommended_action: Optional[str] = None
    if len(beam_results) > 1:
        state_trans = transitions.get(state, {})
        ranked_actions = _rank_actions_from_beam(
            beam_results,
            state_trans,
            action_strategy=action_strategy,
            epsilon=epsilon,
            rng=rng,
        )
        recommended_action = ranked_actions[0] if ranked_actions else None

    if recommended_action is None and action_plan:
        recommended_action = action_plan[0]

    non_root = best_path[1:] if len(best_path) > 1 else []
    avg_reward = (
        float(np.mean([r.avg_future_reward for r in non_root])) if non_root else 0.0
    )
    avg_wr = float(np.mean([r.win_rate for r in non_root])) if non_root else 0.0
    cum_prob = best_path[-1].cumulative_probability if best_path else 0.0

    metrics: Dict[str, Any] = {
        "recommended_action": recommended_action,
        "expected_cumulative_reward": round(avg_reward, 4),
        "expected_win_rate": round(avg_wr, 4),
        "best_beam_cum_prob": round(cum_prob, 6),
        "best_beam_length": len(best_path) - 1,
    }

    return PlanResult(
        recommended_action=recommended_action,
        ranked_actions=ranked_actions,
        action_plan=action_plan,
        planned_states=planned_states,
        beam_results=beam_results,
        beam_paths=beam_paths,
        best_path_index=best_path_index,
        metrics=metrics,
    )


def find_optimal_action(
    kg: DecisionKnowledgeGraph,
    transitions: Dict[int, Dict[str, Dict]],
    state: int,
    beam_width: int = 3,
    max_steps: int = 5,
    min_visits: int = 1,
    min_cum_prob: float = 0.01,
    score_mode: str = "quality",
    max_state_revisits: int = 2,
    discount_factor: float = 0.9,
    action_strategy: str = "best_beam",
    epsilon: float = 0.1,
    rng_seed: Optional[int] = None,
    finetune_model: Any = None,
    dist_matrix: Any = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    plan = plan_action(
        kg,
        transitions,
        state,
        beam_width=beam_width,
        max_steps=max_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        score_mode=score_mode,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
        action_strategy=action_strategy,
        epsilon=epsilon,
        rng_seed=rng_seed,
        finetune_model=finetune_model,
        dist_matrix=dist_matrix,
    )
    info: Dict[str, Any] = dict(plan.metrics)
    info["all_results"] = plan.beam_results
    info["ranked_actions"] = plan.ranked_actions
    info["action_plan"] = plan.action_plan
    info["planned_states"] = plan.planned_states
    info["beam_paths"] = plan.beam_paths
    info["best_path_index"] = plan.best_path_index
    return plan.recommended_action, info


def get_beam_paths(
    results: List[BeamSearchResult],
) -> List[List[BeamSearchResult]]:
    """
    Trace actual root-to-leaf paths via parent_idx.

    Returns:
        List of paths, each path is [root, step1, step2, ..., leaf].
    """
    if not results:
        return []

    parent_indices = {r.parent_idx for r in results if r.parent_idx is not None}
    all_indices = set(range(len(results)))
    leaf_indices = sorted(all_indices - parent_indices)

    paths: List[List[BeamSearchResult]] = []
    for leaf_idx in leaf_indices:
        path: List[BeamSearchResult] = []
        current = leaf_idx
        while current is not None:
            path.append(results[current])
            current = results[current].parent_idx
        path.reverse()
        paths.append(path)

    return paths
