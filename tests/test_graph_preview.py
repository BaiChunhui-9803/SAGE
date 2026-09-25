"""Display sampling preserves graph identities and empirical probabilities."""
from copy import deepcopy
from types import SimpleNamespace

from scripts.etg_web.graph_builder import build_graph_preview


def example():
    stat = SimpleNamespace(visits=100, quality_score=5, win_rate=.8,
                           avg_step_reward=1, avg_future_reward=2)
    data = {"use_context": False, "state_action_map": {i: {"a": stat} for i in range(1001)}}
    transitions = {0: {"a": {"next_states": {i: i for i in range(1, 1001)}}}}
    transitions.update({i: {"a": {"next_states": {0: 1}}} for i in range(1, 1001)})
    return data, transitions


def test_preview_bounds_high_degree_neighborhood_and_keeps_focus():
    data, transitions = example()
    original = deepcopy(transitions)
    nodes, edges, stats, found = build_graph_preview(data, transitions, 0,
        max_neighbors=3, max_nodes=4, focus_hops=3, min_transition_prob=0, max_edges=5)
    assert found
    assert {n["id"] for n in nodes} == {0, 998, 999, 1000}
    assert set(stats) == {0, 998, 999, 1000}
    assert len(edges) <= 5
    assert all(e["from"] in stats and e["to"] in stats for e in edges)
    assert {e["to"] for e in edges if e["from"] == 0} == {998, 999, 1000}
    assert transitions == original
    for edge in edges:
        if edge["from"] == 0:
            assert edge["transition_prob"] == edge["to"] / sum(range(1, 1001))


def test_preview_filters_before_expansion_and_handles_missing_or_terminal_state():
    data, transitions = example()
    nodes, edges, _, _ = build_graph_preview(data, transitions, 0, min_transition_prob=.9)
    assert [n["id"] for n in nodes] == [0]
    assert not edges
    assert build_graph_preview(data, transitions, -1)[-1] is False
    transitions[0] = {"__terminal__": True}
    nodes, edges, _, found = build_graph_preview(data, transitions, 0)
    assert found and nodes[0]["is_terminal"] and not edges
