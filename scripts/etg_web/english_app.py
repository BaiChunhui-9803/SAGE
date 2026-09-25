"""English presentation layer; algorithms and frozen inputs are shared with the original app."""
from dataclasses import asdict
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from paper_assets.aiide26_sage.evidence import PACK, ROOT, SCENARIOS, read_json
from src.decision.experience_transition_graph import DecisionExperienceTransitionGraph, load_compatible_pickle
from src.decision.etg_beam_search import plan_action
from src.decision.chain_rollout import chain_rollout
from src.utils.distance_assets import load_distances
from scripts.run_paper_sage import normalize_source_names
from etg_web.graph_builder import build_graph_preview
from etg_web.paper_tab import render_paper_tab


@st.cache_resource(max_entries=2)
def graph_inputs(scenario):
    folder = ROOT / "cache/experience_transition_graph" / (SCENARIOS[scenario] + "_augmented")
    graph = DecisionExperienceTransitionGraph.load(str(folder / "etg_simple.pkl"))
    with (folder / "etg_simple_transitions.pkl").open("rb") as stream:
        transitions = load_compatible_pickle(stream)
    return graph, transitions


def recorded_parameters(scenario, method):
    for path in sorted((PACK / "raw/evaluations").glob("*/final_eval_summary.json")):
        meta = read_json(path)
        if meta["map_key"] == scenario and bool(meta["action_tuning_enabled"]) == (method == "Full SAGE"):
            return normalize_source_names(meta["repeats"][0]["params"])
    raise ValueError("No recorded configuration for this scenario and method")


def network_html(nodes, edges):
    """Render data from the original graph builder with explicit English tooltips."""
    net = Network(height="540px", width="100%", directed=True, cdn_resources="in_line")
    for node in nodes:
        sid = int(node["id"])
        net.add_node(sid, label=f"S{sid}", level=node.get("level", 0),
            color="#f59e0b" if node["is_terminal"] else "#60a5fa",
            title=(f"State: {sid}<br>Visits: {node['total_visits']}"
                   f"<br>Best quality: {node['best_quality']:.2f}"
                   f"<br>Best win rate: {node['best_win_rate']:.1%}"
                   f"<br>Terminal: {bool(node['is_terminal'])}"))
    connections = {}
    for edge in edges:
        connections.setdefault((int(edge["from"]), int(edge["to"])), []).append(edge)
    for (source, target), actions in connections.items():
        label = str(actions[0]["action"]) if len(actions) == 1 else f"{len(actions)} actions"
        tooltip = "<br>".join(f"Action {e['action']}: P={e['transition_prob']:.3f}, count={e['transition_count']}, quality={e['quality_score']:.2f}" for e in actions)
        net.add_edge(source, target, label=label, title=tooltip, arrows="to")
    net.set_options(json.dumps({"physics": {"enabled": False},
        "layout": {"hierarchical": {"enabled": True, "direction": "LR", "nodeSpacing": 110, "levelSeparation": 180}},
        "interaction": {"hover": True}, "edges": {"smooth": {"enabled": True, "type": "cubicBezier"}}}))
    return net.generate_html(notebook=False)


def graph_page(graph, transitions, start):
    st.caption("Explore a small outgoing neighborhood in the selected scenario's Experience Transition Graph. Node IDs belong to that scenario.")
    cols = st.columns(3)
    hops = cols[0].slider("Neighborhood hops", 1, 3, 2)
    neighbors = cols[1].slider("Neighbors per state", 1, 12, 6)
    limit = cols[2].slider("Maximum rendered nodes", 10, 100, 24)
    visits = st.number_input("Minimum state-action visits", 1, value=1)
    probability = st.slider("Minimum displayed transition probability", 0.0, 1.0, .05, .01)
    data = {"state_action_map": graph.state_action_map, "use_context": graph.use_context}
    nodes, edges, _, exists = build_graph_preview(data, transitions, min_visits=visits,
        max_nodes=limit, max_neighbors=neighbors, focus_state=start, focus_hops=hops,
        min_transition_prob=probability, max_edges=60)
    if not exists:
        st.warning("The selected state is not present in this graph.")
        return
    st.caption(f"Displayed states: {len(nodes)} · Action transitions: {len(edges)}. Up to 60 transitions are included. Parallel actions share one connection; hover for their individual probabilities. Orange nodes are terminal. Drag nodes or zoom to inspect.")
    components.html(network_html(nodes, edges), height=560, scrolling=False)
    with st.expander("Displayed transitions"):
        frame = pd.DataFrame(edges)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button("Download displayed transitions", frame.to_csv(index=False).encode(),
            file_name="displayed_transitions.csv", mime="text/csv")


def planner_settings(params):
    c1, c2, c3 = st.columns(3)
    width = c1.number_input("Beam width", min_value=1, max_value=30, value=int(params["beam_width"]))
    depth = c2.number_input("Look-ahead steps", min_value=1, max_value=30, value=int(params["lookahead_steps"]))
    visits = c3.number_input("Minimum visits", min_value=1, value=int(params["min_visits"]))
    return dict(beam_width=width, min_visits=visits, min_cum_prob=float(params["min_cum_prob"]),
        score_mode=params["score_mode"], max_state_revisits=int(params["max_state_revisits"]),
        discount_factor=float(params["discount_factor"]), action_strategy=params["action_strategy"],
        epsilon=float(params.get("epsilon", .1)), rng_seed=42), depth


def planning_page(graph, transitions, start, params):
    settings, depth = planner_settings(params)
    st.caption("Uses the recorded planner configuration with the controls above. This view inspects graph planning; it does not run Full SAGE's live observation gates.")
    if st.button("Run beam search", type="primary"):
        plan = plan_action(graph, transitions, start, max_steps=depth, **settings)
        if plan.recommended_action is None:
            st.warning("No admissible action under these visit and probability filters. Choose another state or relax the filters.")
            return
        st.success(f"Recommended graph action: {plan.recommended_action}")
        st.dataframe(pd.DataFrame([asdict(node) for node in plan.beam_results]), hide_index=True, use_container_width=True)
        st.json({"action_plan": plan.action_plan, "planned_states": plan.planned_states})


def rollout_page(graph, transitions, start, params, scenario):
    settings, depth = planner_settings(params)
    c1, c2, c3 = st.columns(3)
    steps = c1.slider("Maximum rollout steps", 1, 100, 20)
    mode = c2.selectbox("Planning mode", ["multi_step", "single_step"],
        format_func=lambda x: {"multi_step": "Follow a multi-step plan", "single_step": "Replan at each step"}[x])
    backup = c3.checkbox("Enable backup switching", value=bool(params.get("enable_backup", True)))
    transition = st.radio("Next-state rule", ["highest_prob", "sample"], horizontal=True,
        format_func=lambda x: {"highest_prob": "Most likely transition", "sample": "Sample the empirical distribution"}[x])
    st.caption("Graph simulation with seed 42. It does not simulate new SC2 observations or perform live gated exploration.")
    if st.button("Run graph rollout", type="primary"):
        result = chain_rollout(graph, transitions, start, lookahead_steps=depth,
            max_rollout_steps=steps, rollout_mode=mode, enable_backup=backup,
            next_state_mode=transition, dist_matrix=load_distances(SCENARIOS[scenario], "augmented_1"),
            score_threshold=float(params.get("backup_score_threshold", .3)),
            distance_threshold=float(params.get("backup_distance_threshold", .2)), **settings)
        st.success(f"Completed {max(0, len(result.chosen_path_ids)-1)} transitions; backup switches: {result.total_backup_switches}; replans: {result.total_re_searches}.")
        frame = pd.DataFrame([asdict(result.nodes[key]) for key in result.chosen_path_ids])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        if len(frame):
            st.line_chart(frame.set_index("rollout_depth")[["quality_score", "avg_future_reward"]])
        st.download_button("Download rollout trace", frame.to_csv(index=False).encode(), file_name="rollout.csv", mime="text/csv")


def experiments_page():
    st.markdown("Prepare or run **Graph-only SAGE** and **Full SAGE** using each scenario's recorded settings. Live runs require the separate SC2 environment and the six bundled maps.")
    scene = st.selectbox("Scenario", list(SCENARIOS))
    method = st.selectbox("Method", ["graph-only", "full"])
    episodes = st.number_input("Episodes", min_value=1, max_value=10000, value=1)
    command = f"python scripts/run_paper_sage.py --scenario {scene} --method {method} --episodes {episodes}"
    st.code(command, language="bash")
    st.caption("Preparation writes parameters and a manifest to a new output/paper_live/ directory. It does not start the game.")
    st.code(command + " --run", language="bash")
    st.caption("Run this command in the SC2 environment to play new episodes. Full SAGE updates a run-local copy of its initial model.")
    st.markdown("See `ARTIFACT.md` for installation, experiment commands, and output files.")


def main():
    st.set_page_config(page_title="SAGE Artifact Explorer", page_icon="🕸️", layout="wide")
    st.session_state["ui_language"] = "en"
    st.title("SAGE Artifact Explorer")
    st.caption("Switch-Aware Graph Planning Integrated with Gated Exploration on Unseen States")
    page = st.sidebar.radio("Explore", ["Paper results", "Project overview", "Experience Transition Graph", "Beam-search planning", "Graph rollout", "Fresh experiments"], key="english_page")
    if page == "Paper results":
        render_paper_tab()
    elif page == "Project overview":
        st.markdown("""SAGE stores historical micromanagement experience in an **Experience Transition Graph (ETG)**.

**Graph-only SAGE** plans on this graph and switches to retained backup branches when observations diverge.
**Full SAGE** adds gated exploration and a local action-value model for uncertain or unseen states.

Explore paper results, six scenario graphs, beam-search plans and graph rollouts using the SAGE planning modules.

The paper page displays results, figures and an input-directory browser. Each graph page loads the selected scenario independently. See `README.md`, `ARTIFACT.md`, and `paper_assets/aiide26_sage/README.md` for installation and experiment commands.""")
    elif page == "Fresh experiments":
        experiments_page()
    else:
        scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS), key="english_scenario")
        st.sidebar.caption(SCENARIOS[scenario] + "_augmented")
        with st.spinner("Loading graph assets..."):
            graph, transitions = graph_inputs(scenario)
        start = st.sidebar.number_input("Initial abstract state ID", min_value=0, value=1, key=f"english_state_{scenario}")
        st.sidebar.caption(f"Graph states: {len(graph.state_action_map):,}. Default initial state ID: 1.")
        if page == "Experience Transition Graph":
            graph_page(graph, transitions, start)
        else:
            method = st.sidebar.selectbox("Recorded configuration", ["Graph-only SAGE", "Full SAGE"], key="english_method")
            params = recorded_parameters(scenario, method)
            with st.sidebar.expander("Recorded planner settings"):
                st.json({k: params[k] for k in ["beam_width", "lookahead_steps", "min_visits", "score_mode", "min_cum_prob", "discount_factor", "action_strategy"]})
            if page == "Beam-search planning":
                planning_page(graph, transitions, start, params)
            else:
                rollout_page(graph, transitions, start, params, scenario)
