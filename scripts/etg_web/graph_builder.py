from typing import Dict, List, Optional, Set, Tuple
from heapq import nlargest


def build_graph_preview(
    etg_data: Dict,
    transitions: Dict,
    focus_state: int,
    max_nodes: int = 24,
    max_neighbors: int = 6,
    focus_hops: int = 1,
    min_visits: int = 1,
    min_transition_prob: float = 0.05,
    max_edges: int = 60,
):
    """Select a bounded outgoing neighborhood before building any visual data.

    Transition probabilities retain their full empirical denominators. The
    supplied graph and transition counts are never modified by display limits.
    """
    if max_nodes < 1 or max_neighbors < 1 or focus_hops < 0 or max_edges < 0:
        raise ValueError("Preview limits must be positive (hops/edges may be zero)")
    if etg_data.get("use_context", False):
        raise ValueError("The scenario preview expects an abstract-state graph")
    sam = etg_data["state_action_map"]
    if focus_state not in sam:
        return [], [], {}, False
    selected = {focus_state: 0}
    discovery_links = set()
    node_budget = min(max_nodes, max_edges + 1)
    frontier = [focus_state]
    for depth in range(focus_hops):
        next_frontier = []
        for state in frontier:
            actions = sam[state]
            outgoing = transitions.get(state, {})
            if outgoing.get("__terminal__"):
                continue
            counts = {}
            for action, stats in actions.items():
                if stats.visits < min_visits:
                    continue
                destinations = outgoing.get(action, {}).get("next_states", {})
                total = sum(destinations.values())
                for target, count in destinations.items():
                    if target != state and target in sam and total and count / total >= min_transition_prob:
                        counts[target] = counts.get(target, 0) + count
            neighbors = nlargest(max_neighbors, counts, key=lambda target: (counts[target], -target))
            for target in neighbors:
                if target not in selected and len(selected) < node_budget:
                    selected[target] = depth + 1
                    discovery_links.add((state, target))
                    next_frontier.append(target)
        if len(selected) >= node_budget or not next_frontier:
            break
        frontier = next_frontier
    preview_data = {"state_action_map": {s: sam[s] for s in selected}, "use_context": False}
    nodes, edges, stats, exists = build_graph_data(
        preview_data, transitions, min_visits=min_visits, max_nodes=max_nodes
    )
    for node in nodes:
        node["level"] = selected[node["id"]]
    ranked = sorted(
        (edge for edge in edges if edge["transition_prob"] >= min_transition_prob),
        key=lambda edge: (-edge["transition_count"], edge["from"], edge["to"], str(edge["action"])),
    )
    # Keep the strongest discovery edge for every included neighbor so that
    # frequent self-loops cannot displace its connection to the focus state.
    edges, remaining = [], []
    for edge in ranked:
        pair = (edge["from"], edge["to"])
        if pair in discovery_links:
            edges.append(edge)
            discovery_links.remove(pair)
        else:
            remaining.append(edge)
    edges.extend(remaining[:max_edges - len(edges)])
    return sorted(nodes, key=lambda node: (node["level"], node["id"])), edges, stats, exists


def build_graph_data(
    etg_data: Dict,
    transitions: Dict,
    min_visits: int = 1,
    min_quality: float = -100,
    max_quality: float = 100,
    max_nodes: int = 200,
    focus_state: Optional[int] = None,
    focus_hops: int = 2,
    focus_forward: bool = True,
    focus_backward: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict[int, Dict], bool]:
    sam = etg_data["state_action_map"]

    state_stats: Dict[int, Dict] = {}
    for key, actions in sam.items():
        state_id = key if isinstance(key, int) else key[0]
        total_visits = sum(a.visits for a in actions.values())
        best_quality = max((a.quality_score for a in actions.values()), default=0)
        best_win_rate = max((a.win_rate for a in actions.values()), default=0)
        n_actions = len(actions)
        state_stats[state_id] = {
            "total_visits": total_visits,
            "best_quality": best_quality,
            "best_win_rate": best_win_rate,
            "n_actions": n_actions,
            "is_terminal": state_id not in transitions
            or transitions[state_id].get("__terminal__", False),
        }

    all_states: Set[int] = set(state_stats.keys())

    if focus_state is not None:
        if focus_state not in all_states:
            return [], [], state_stats, False
        selected = set()
        frontier = {focus_state}
        for _ in range(focus_hops):
            next_frontier = set()
            for s in frontier:
                if focus_forward and s in transitions:
                    if transitions[s].get("__terminal__"):
                        pass
                    else:
                        for a_info in transitions[s].values():
                            for ns in a_info.get("next_states", {}):
                                if ns in all_states:
                                    next_frontier.add(ns)
                if focus_backward:
                    for other_s in all_states:
                        if other_s in transitions and not transitions[other_s].get(
                            "__terminal__"
                        ):
                            for a_info in transitions[other_s].values():
                                if (
                                    s in a_info.get("next_states", {})
                                    and other_s not in selected
                                ):
                                    next_frontier.add(other_s)
            selected |= frontier
            frontier = next_frontier - selected
        selected |= frontier
        filtered_states = selected & all_states
    else:
        filtered_states = all_states

    edges: List[Dict] = []
    for state_id in filtered_states:
        key = state_id if not etg_data["use_context"] else (state_id, ())
        actions = sam.get(key, {})
        for action, stats in actions.items():
            if stats.visits < min_visits:
                continue
            if stats.quality_score < min_quality or stats.quality_score > max_quality:
                continue

            if state_id in transitions and action in transitions[state_id]:
                next_states = transitions[state_id][action].get("next_states", {})
                total_count = sum(next_states.values())
                for next_state, count in next_states.items():
                    prob = count / total_count if total_count > 0 else 0
                    edges.append(
                        {
                            "from": state_id,
                            "to": next_state,
                            "action": action,
                            "visits": stats.visits,
                            "quality_score": stats.quality_score,
                            "win_rate": stats.win_rate,
                            "avg_step_reward": stats.avg_step_reward,
                            "avg_future_reward": stats.avg_future_reward,
                            "transition_count": count,
                            "transition_prob": prob,
                        }
                    )
            else:
                edges.append(
                    {
                        "from": state_id,
                        "to": -1,
                        "action": action,
                        "visits": stats.visits,
                        "quality_score": stats.quality_score,
                        "win_rate": stats.win_rate,
                        "avg_step_reward": stats.avg_step_reward,
                        "avg_future_reward": stats.avg_future_reward,
                        "transition_count": 0,
                        "transition_prob": 0,
                    }
                )

    valid_edges = [e for e in edges if e["to"] in all_states]

    if len(filtered_states) > max_nodes:
        sorted_states = sorted(
            filtered_states,
            key=lambda s: state_stats[s]["total_visits"],
            reverse=True,
        )[:max_nodes]
        top_set = set(sorted_states)
        valid_edges = [
            e for e in valid_edges if e["from"] in top_set and e["to"] in top_set
        ]
        filtered_states = top_set

    nodes = []
    for s in filtered_states:
        st_info = state_stats.get(s, {})
        nodes.append(
            {
                "id": s,
                "total_visits": st_info.get("total_visits", 0),
                "best_quality": st_info.get("best_quality", 0),
                "best_win_rate": st_info.get("best_win_rate", 0),
                "n_actions": st_info.get("n_actions", 0),
                "is_terminal": st_info.get("is_terminal", False),
            }
        )

    final_state_ids = {n["id"] for n in nodes}
    valid_edges = [
        e
        for e in valid_edges
        if e["from"] in final_state_ids and e["to"] in final_state_ids
    ]

    return nodes, valid_edges, state_stats, True
