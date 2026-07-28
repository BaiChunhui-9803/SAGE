import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pickle
import json
import base64
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.decision.knowledge_graph import DecisionKnowledgeGraph
from src.decision.kg_beam_search import (
    get_beam_paths,
    BeamSearchResult,
)
from src.decision.chain_rollout import chain_rollout, RolloutNode
from src.decision.beam_matcher import match_beam_paths
from kg_web.constants import KG_DIR, NPY_DIR
from kg_web.loaders import load_episode_data, load_distance_matrix_np
from kg_web.beam_utils import (
    _compute_composite_scores,
    _build_rec_rows,
    _build_path_detail_rows,
    _results_to_json,
)
from kg_web.pyvis_renderer import render_beam_tree


@st.cache_data(max_entries=200, show_spinner=False)
def _cached_rollout_results(
    _kg_path: str,
    _trans_path: str,
    start_state: int,
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
    rng_seed,
    rollout_mode: str = "single_step",
    enable_backup: bool = False,
    score_threshold: float = 0.3,
    distance_threshold: float = 0.2,
):
    kg = DecisionKnowledgeGraph.load(str(KG_DIR / _kg_path))
    with open(str(KG_DIR / _trans_path), "rb") as f:
        transitions = pickle.load(f)

    dist_matrix = None
    if enable_backup:
        map_id = _kg_path.split("_")[0] if "_" in _kg_path else ""
        data_id = _kg_path.split("_")[1] if "_" in _kg_path else ""
        if map_id and data_id:
            dm_path = NPY_DIR / f"state_distance_matrix_{map_id}_{data_id}.npy"
            if dm_path.exists():
                dist_matrix = np.load(str(dm_path))

    result = chain_rollout(
        kg,
        transitions,
        start_state,
        score_mode=score_mode,
        action_strategy=action_strategy,
        next_state_mode=next_state_mode,
        beam_width=beam_width,
        lookahead_steps=lookahead_steps,
        max_rollout_steps=max_rollout_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
        epsilon=epsilon,
        rng_seed=rng_seed,
        rollout_mode=rollout_mode,
        enable_backup=enable_backup,
        score_threshold=score_threshold,
        distance_threshold=distance_threshold,
        dist_matrix=dist_matrix,
    )

    nodes_ser = {}
    for nid, n in result.nodes.items():
        nodes_ser[str(nid)] = {
            "id": n.id,
            "parent_id": n.parent_id,
            "children_ids": n.children_ids,
            "state": n.state,
            "action": n.action,
            "beam_id": n.beam_id,
            "quality_score": n.quality_score,
            "win_rate": n.win_rate,
            "avg_future_reward": n.avg_future_reward,
            "avg_step_reward": n.avg_step_reward,
            "visits": n.visits,
            "transition_prob": n.transition_prob,
            "cumulative_probability": n.cumulative_probability,
            "rollout_depth": n.rollout_depth,
            "is_on_chosen_path": n.is_on_chosen_path,
            "is_terminal": n.is_terminal,
            "is_beam_root": n.is_beam_root,
        }

    return {
        "nodes": nodes_ser,
        "root_id": result.root_id,
        "chosen_path_ids": result.chosen_path_ids,
        "termination_reason": result.termination_reason,
        "beam_results_by_step": {
            str(k): v for k, v in result.beam_results_by_step.items()
        },
        "rollout_mode": result.rollout_mode,
        "plan_segments": result.plan_segments,
        "total_re_searches": result.total_re_searches,
        "total_backup_switches": result.total_backup_switches,
        "switch_points_by_segment": {
            str(k): v for k, v in result.switch_points_by_segment.items()
        },
    }


@st.fragment
def _run_rollout(kg_data, transitions, kg_entry, kg):
    start_state = st.session_state.get("roll_state", 0)
    score_mode = st.session_state.get("roll_sm", "quality")
    beam_width = st.session_state.get("roll_bw", 3)
    lookahead_steps = st.session_state.get("roll_la", 5)
    min_cum_prob = st.session_state.get("roll_mcp", 0.01)
    min_visits_roll = st.session_state.get("roll_mv", 1)
    max_state_revisits = st.session_state.get("roll_msr", 2)
    discount_factor = st.session_state.get("roll_df", 0.9)
    action_strategy = st.session_state.get("roll_as", "best_beam")
    next_state_mode = st.session_state.get("roll_nsm", "sample")
    epsilon = st.session_state.get("roll_eps", 0.1)
    max_rollout_steps = st.session_state.get("roll_mrs", 50)
    rng_seed_str = st.session_state.get("roll_seed", "42")
    rollout_mode = st.session_state.get("roll_mode", "单步推演")
    enable_backup = st.session_state.get("roll_backup", False)
    backup_score_threshold = st.session_state.get("roll_backup_st", 0.3)
    backup_dist_threshold = st.session_state.get("roll_backup_dt", 0.2)

    unique_states = kg_data.get("unique_states", set())
    if isinstance(unique_states, set):
        if start_state not in unique_states:
            st.warning(
                f"状态 {start_state} 不存在于当前经验转移图中。请选择一个有效状态。"
            )
            return

    if start_state not in transitions:
        st.warning(f"状态 {start_state} 没有转移数据，无法进行推演。")
        return

    kg_file = kg_entry.get("file", "")
    trans_file = kg_entry.get("transitions", "")

    try:
        rng_seed = int(rng_seed_str) if rng_seed_str.strip() else None
    except ValueError:
        rng_seed = None

    actual_rollout_mode = "multi_step" if rollout_mode == "多步推演" else "single_step"

    cached = _cached_rollout_results(
        kg_file,
        trans_file,
        start_state,
        score_mode,
        action_strategy,
        next_state_mode,
        beam_width,
        lookahead_steps,
        max_rollout_steps,
        min_visits_roll,
        min_cum_prob,
        max_state_revisits,
        discount_factor,
        epsilon,
        rng_seed,
        rollout_mode=actual_rollout_mode,
        enable_backup=enable_backup,
        score_threshold=backup_score_threshold,
        distance_threshold=backup_dist_threshold,
    )

    nodes = cached["nodes"]
    chosen_path_ids = cached["chosen_path_ids"]
    termination_reason = cached["termination_reason"]

    if len(chosen_path_ids) <= 1:
        st.error("推演无结果：该状态无可用动作或为终端状态。")
        return

    path_nodes = [nodes[str(nid)] for nid in chosen_path_ids]
    n_steps = len(path_nodes) - 1

    step_data = []
    for i in range(n_steps):
        prev = path_nodes[i]
        curr = path_nodes[i + 1]
        child_actions = set()
        for cid in prev.get("children_ids", []):
            c = nodes.get(str(cid))
            if c and c.get("action"):
                child_actions.add(c["action"])
        step_data.append(
            {
                "step": i + 1,
                "state": prev["state"],
                "action": curr["action"],
                "next_state": curr["state"],
                "win_rate": curr["win_rate"],
                "quality_score": curr["quality_score"],
                "avg_future_reward": curr["avg_future_reward"],
                "avg_step_reward": curr["avg_step_reward"],
                "transition_prob": curr["transition_prob"],
                "cumulative_probability": curr["cumulative_probability"],
                "visits": curr["visits"],
                "n_candidates": len(child_actions),
                "is_terminal": curr["is_terminal"],
            }
        )

    st.markdown(f"**起始状态**: `{start_state}`")

    col_summary, col_mid, col_traj = st.columns([0.2, 0.6, 0.8])

    with col_summary:
        st.markdown("**📋 推演摘要**")
        last_step = step_data[-1]
        st.metric("总步数", n_steps)
        st.metric("终止原因", termination_reason)
        st.metric(
            "起止状态",
            f"{path_nodes[0]['state']} → {last_step['next_state']}",
        )
        avg_wr = sum(s["win_rate"] for s in step_data) / n_steps
        avg_qs = sum(s["quality_score"] for s in step_data) / n_steps
        st.metric("平均胜率", f"{avg_wr:.2%}")
        st.metric("平均 Quality", f"{avg_qs:.1f}")

        if actual_rollout_mode == "multi_step":
            st.divider()
            st.caption("**多步推演统计**")
            plan_segments = cached.get("plan_segments", [])
            st.metric("规划段数", len(plan_segments))
            st.metric("重新规划次数", cached.get("total_re_searches", 0))
            st.metric("备选切换次数", cached.get("total_backup_switches", 0))
            main_steps = 0
            total_planned = 0
            for seg in plan_segments:
                total_planned += len(seg.get("actions_planned", []))
                if seg.get("divergence_type") in ("none", "backup_switch"):
                    main_steps += len(seg.get("actions_planned", []))
                elif seg.get("divergence_step", -1) >= 0:
                    main_steps += seg["divergence_step"]
            if total_planned > 0:
                st.metric("路径命中率", f"{main_steps / total_planned:.0%}")

    with col_mid:
        st.markdown("**📈 胜率/质量趋势**")
        step_nums = [s["step"] for s in step_data]
        win_rates = [s["win_rate"] * 100 for s in step_data]
        qualities = [s["quality_score"] for s in step_data]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=step_nums,
                y=win_rates,
                mode="lines+markers",
                name="Win Rate (%)",
                line=dict(color="#4CAF50", width=2),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=step_nums,
                y=qualities,
                mode="lines+markers",
                name="Quality Score",
                line=dict(color="#2196F3", width=2),
                marker=dict(size=4),
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[step_nums[0], step_nums[-1]],
                y=[50, 50],
                mode="lines",
                name="50% 基准",
                line=dict(color="#4CAF50", width=1, dash="dash"),
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[step_nums[0], step_nums[-1]],
                y=[0, 0],
                mode="lines",
                name="0 基准",
                line=dict(color="#4CAF50", width=1, dash="dash"),
                hoverinfo="skip",
                yaxis="y2",
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=50, r=50, t=30, b=40),
            xaxis=dict(title="Step", gridcolor="#e0e0e0"),
            yaxis=dict(title="Win Rate (%)", gridcolor="#e0e0e0", domain=[0, 1]),
            yaxis2=dict(
                title="Quality",
                gridcolor="#e0e0e0",
                overlaying="y",
                side="right",
                anchor="x",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            font=dict(family="SimHei, Microsoft YaHei, sans-serif", size=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True, height=320)

        st.markdown("**🔍 片段匹配**")
        map_id = kg_entry.get("map_id", "")
        data_id = kg_entry.get("data_id", "")
        match_rows = []
        dist_mat = None

        if map_id and data_id:
            with st.spinner("匹配原始对局片段..."):
                ep_data = load_episode_data(map_id, data_id)
                dist_mat = load_distance_matrix_np(map_id, data_id)

            if ep_data["n_episodes"] > 0:
                state_seq = [pn["state"] for pn in path_nodes]
                action_seq = [s["action"] for s in step_data]

                rollout_beam_paths = {
                    0: [
                        BeamSearchResult(
                            step=i,
                            state=state_seq[i],
                            action=action_seq[i] if i < len(action_seq) else "",
                            cumulative_probability=path_nodes[i][
                                "cumulative_probability"
                            ],
                            quality_score=path_nodes[i]["quality_score"],
                            win_rate=path_nodes[i]["win_rate"],
                            avg_step_reward=path_nodes[i]["avg_step_reward"],
                            avg_future_reward=path_nodes[i]["avg_future_reward"],
                            beam_id=0,
                            parent_idx=i - 1 if i > 0 else None,
                        )
                        for i in range(len(state_seq))
                    ]
                }

                match_results = match_beam_paths(
                    rollout_beam_paths,
                    ep_data["states"],
                    ep_data["actions"],
                    ep_data["outcomes"],
                    ep_data["scores"],
                    dist_mat,
                    top_k=5,
                )

                for bid in sorted(match_results.keys()):
                    for rank, m in enumerate(match_results[bid]):
                        match_rows.append(
                            {
                                "Rank": rank + 1,
                                "Episode": m.episode_id,
                                "位置": f"{m.start_pos}~{m.end_pos}",
                                "状态相似度": f"{m.state_similarity:.1%}",
                                "动作匹配率": f"{m.action_match_rate:.0%}",
                                "综合置信度": f"{m.combined_score:.1%}",
                                "结果": m.outcome,
                                "得分": m.episode_score,
                            }
                        )

        if match_rows:
            st.dataframe(
                match_rows,
                use_container_width=True,
                height=min(len(match_rows) * 35 + 50, 300),
            )
            wins = sum(1 for r in match_rows if r["结果"] == "Win")
            total_m = len(match_rows)
            st.caption(
                f"共 {total_m} 条匹配 | Win: {wins} ({wins / total_m:.0%}) | "
                f"距离矩阵: {'有' if dist_mat is not None else '无(精确匹配)'}"
            )
        else:
            st.info("无匹配结果。原始对局数据可能未加载。")

    with col_traj:
        st.markdown("**📋 推演轨迹**")

        plan_segments = cached.get("plan_segments", [])
        step_segment_map = {}
        if actual_rollout_mode == "multi_step" and plan_segments:
            global_step = 0
            for seg_idx, seg in enumerate(plan_segments):
                actions = seg.get("actions_planned", [])
                div_step = seg.get("divergence_step", -1)
                div_type = seg.get("divergence_type", "none")
                for act_idx in range(len(actions)):
                    step_segment_map[global_step] = {
                        "segment": seg_idx,
                        "seg_type": seg.get("segment_type", ""),
                        "divergence": act_idx == div_step
                        if div_type != "none"
                        else False,
                        "div_type": div_type if act_idx == div_step else "",
                    }
                    global_step += 1

        _SEG_TYPE_LABELS = {
            "initial_plan": "初始规划",
            "re_search": "重新规划",
            "backup_switch": "备选切换",
        }

        traj_rows = []
        for idx, s in enumerate(step_data):
            seg_info = step_segment_map.get(idx, {})
            row = {
                "Step": s["step"],
                "State": s["state"],
                "Action": s["action"],
                "Next State": s["next_state"],
                "Win Rate": f"{s['win_rate']:.2%}",
                "Quality": f"{s['quality_score']:.1f}",
                "Future Reward": f"{s['avg_future_reward']:.2f}",
                "Cum. Prob": f"{s['cumulative_probability']:.4f}",
                "Trans. Prob": f"{s['transition_prob']:.2%}",
                "Visits": s["visits"],
                "Candidates": s["n_candidates"],
                "Terminal": "⭕" if s["is_terminal"] else "",
            }
            if actual_rollout_mode == "multi_step" and seg_info:
                row["推演段"] = seg_info.get("segment", "")
                seg_type = seg_info.get("seg_type", "")
                row["段类型"] = _SEG_TYPE_LABELS.get(seg_type, seg_type)
                if seg_info.get("divergence"):
                    div_t = seg_info.get("div_type", "")
                    div_label = {
                        "re_search": "🔄 重规划",
                        "backup_switch": "🔀 备选切换",
                        "no_valid_transition": "❌ 无转移",
                        "low_cum_prob": "⚠️ 低概率",
                    }.get(div_t, div_t)
                    row["段类型"] += f" [{div_label}]"
            traj_rows.append(row)

        st.dataframe(
            traj_rows,
            use_container_width=True,
            height=min(len(traj_rows) * 35 + 50, 600),
        )

    st.divider()
    st.subheader("🔍 束搜索追溯")

    step_options = []
    for i, s in enumerate(step_data):
        label = f"Step {s['step']}: S{s['state']} →({s['action']})→ S{s['next_state']}"
        if s["is_terminal"]:
            label += " [终端]"
        step_options.append(label)

    if not step_options:
        st.info("无推演步骤可供追溯。")
        return

    col_steps, col_tree, col_rec = st.columns([0.2, 0.5, 0.8])

    with col_steps:
        selected_step = st.radio(
            "选择步骤",
            options=list(range(len(step_options))),
            format_func=lambda x: step_options[x],
            key="beam_trace_step",
        )

    beam_results_by_step = cached.get("beam_results_by_step", {})
    raw_beam = beam_results_by_step.get(str(selected_step), [])

    beam_results_list = (
        [
            BeamSearchResult(
                step=r["step"],
                state=r["state"],
                action=r["action"],
                cumulative_probability=r["cumulative_probability"],
                quality_score=r["quality_score"],
                win_rate=r["win_rate"],
                avg_step_reward=r["avg_step_reward"],
                avg_future_reward=r["avg_future_reward"],
                beam_id=r["beam_id"],
                parent_idx=r["parent_idx"],
            )
            for r in raw_beam
        ]
        if raw_beam
        else []
    )

    sub_beam_paths = get_beam_paths(beam_results_list) if beam_results_list else []
    sub_composites, sub_path_metrics = (
        _compute_composite_scores(sub_beam_paths) if sub_beam_paths else ([], [])
    )
    sub_sorted_indices = (
        sorted(
            range(len(sub_beam_paths)),
            key=lambda i: sub_composites[i],
            reverse=True,
        )
        if sub_composites
        else []
    )

    _roll_sel_key = "roll_beam_selected_path"
    _roll_sel_step_key = "roll_beam_selected_step"
    prev_sel_step = st.session_state.get(_roll_sel_step_key, -1)
    if prev_sel_step != selected_step:
        if sub_sorted_indices:
            st.session_state[_roll_sel_key] = sub_sorted_indices[0]
        else:
            st.session_state[_roll_sel_key] = 0
        st.session_state[_roll_sel_step_key] = selected_step
    if _roll_sel_key not in st.session_state:
        st.session_state[_roll_sel_key] = (
            sub_sorted_indices[0] if sub_sorted_indices else 0
        )

    highlight_set = set()
    if sub_beam_paths and st.session_state[_roll_sel_key] < len(sub_beam_paths):
        sel_path = sub_beam_paths[st.session_state[_roll_sel_key]]
        highlight_set = {i for i, r in enumerate(beam_results_list) if r in sel_path}

    chosen_action = step_data[selected_step]["action"]
    chosen_next_state = step_data[selected_step]["next_state"]

    with col_tree:
        if beam_results_list:
            html = render_beam_tree(beam_results_list, highlight_indices=highlight_set)
            html_b64 = base64.b64encode(html.encode()).decode()
            html_uri = f"data:text/html;charset=utf-8;base64,{html_b64}#v=1"
            sd = step_data[selected_step]
            subtree_json = _results_to_json(
                beam_results_list,
                beam_width,
                lookahead_steps,
                min_cum_prob,
                score_mode,
                sd["state"],
            )
            json_b64 = base64.b64encode(subtree_json.encode()).decode()
            json_uri = f"data:application/json;base64,{json_b64}"
            st.markdown(
                f"**🌳 子树路径图**  "
                f'<a href="{html_uri}" '
                f'download="beam_subtree_step{sd["step"]}_S{sd["state"]}.html" '
                f'style="font-size:0.85em;color:#4CAF50;text-decoration:none;margin-right:12px;">📥 导出 HTML</a>'
                f'<a href="{json_uri}" '
                f'download="beam_subtree_step{sd["step"]}_S{sd["state"]}.json" '
                f'style="font-size:0.85em;color:#2196F3;text-decoration:none;">📥 导出 JSON</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="overflow:hidden;width:100%;height:520px;'
                f'border:1px solid #444;border-radius:4px;">'
                f'<iframe src="{html_uri}" width="100%" height="100%" '
                f'style="border:none;display:block;"></iframe>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**🌳 子树路径图**")
            st.info("该步骤无束搜索子树数据。")

    with col_rec:
        st.markdown("**📊 路径推荐**")
        if sub_beam_paths:
            rec_rows = _build_rec_rows(
                sub_sorted_indices, sub_beam_paths, sub_composites, sub_path_metrics
            )
            event = st.dataframe(
                rec_rows,
                use_container_width=True,
                height=min(len(rec_rows) * 35 + 50, 500),
                on_select="rerun",
                selection_mode="single-row",
            )
            if event and event.selection.rows:
                new_path = rec_rows[event.selection.rows[0]]["Path"]
                old_path = st.session_state[_roll_sel_key]
                if new_path != old_path:
                    st.session_state[_roll_sel_key] = new_path
                    st.rerun()
        else:
            st.info("无路径数据。")

    st.divider()
    st.markdown("**📋 路径详情**")
    if sub_beam_paths:
        detail_rows = _build_path_detail_rows(sub_beam_paths)
        chosen_path_idx = st.session_state.get(_roll_sel_key, 0)
        st.caption(f"当前选中路径: **#{chosen_path_idx}**")
        st.dataframe(
            detail_rows,
            use_container_width=True,
            height=min(len(detail_rows) * 35 + 50, 500),
        )
    else:
        st.info("无路径数据。")

    st.divider()

    with st.expander("🌐 完整推演树全局视图", expanded=False):
        import networkx as nx
        from pyvis.network import Network

        g = nx.DiGraph()
        chosen_set = set(chosen_path_ids)
        max_beam_id = 0
        for nid_key, n in nodes.items():
            bid = n.get("beam_id")
            if bid is not None and bid > max_beam_id:
                max_beam_id = bid

        for nid_key, n in nodes.items():
            nid = n["id"]
            is_chosen = nid in chosen_set
            is_term = n.get("is_terminal", False)

            if is_term:
                node_color = "rgba(255, 80, 80, 0.9)"
                shape = "diamond"
                bw = 3
            elif is_chosen:
                node_color = "#4FC3F7"
                shape = "dot"
                bw = 2
            else:
                node_color = "#999999"
                shape = "dot"
                bw = 1

            label = f"S{n['state']}"
            if n.get("action"):
                label += f"\n{n['action']}"

            depth_tag = f"D{n['rollout_depth']}" if n["rollout_depth"] >= 0 else ""
            title_parts = [
                f"Node ID: {nid}",
                f"State: {n['state']}",
                f"Action: {n.get('action', '-')}",
                f"Beam: {n.get('beam_id', '-')}",
                f"Depth: {depth_tag}",
                f"Quality: {n['quality_score']:.1f}",
                f"Win Rate: {n['win_rate']:.2%}",
                f"Cum Prob: {n['cumulative_probability']:.4f}",
                f"Chosen: {'Yes' if is_chosen else 'No'}",
            ]
            if is_term:
                title_parts.insert(1, "Terminal")
            if n.get("is_beam_root"):
                title_parts.append("Beam Root")

            g.add_node(
                nid,
                label=label,
                color=node_color,
                shape=shape,
                size=20 if is_chosen else 12,
                borderWidth=bw,
                title="\n".join(title_parts),
                font={"size": 10, "color": "white" if is_chosen else "#cccccc"},
            )

        for nid_key, n in nodes.items():
            parent_id = n.get("parent_id")
            if parent_id is not None and str(parent_id) in nodes:
                child_nid = n["id"]
                is_chosen_edge = parent_id in chosen_set and child_nid in chosen_set
                edge_color = "#4FC3F7" if is_chosen_edge else "#666666"
                edge_width = 2 if is_chosen_edge else 0.5
                action_label = n.get("action", "")
                g.add_edge(
                    parent_id,
                    child_nid,
                    label=action_label,
                    color=edge_color,
                    width=edge_width,
                    arrows="to",
                    font={"size": 8, "color": "#aaaaaa"},
                    smooth={"type": "continuous"},
                )

        net_full = Network(height="700px", width="100%", directed=True, notebook=False)
        for nid, ndata in g.nodes(data=True):
            net_full.add_node(nid, **ndata)
        for src, tgt, edata in g.edges(data=True):
            net_full.add_edge(src, tgt, **edata)

        options_full = {
            "physics": {"enabled": False},
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "LR",
                    "sortMethod": "directed",
                    "levelSeparation": 200,
                    "nodeSpacing": 120,
                    "blockShifting": True,
                    "edgeMinimization": True,
                }
            },
            "edges": {"smooth": {"type": "continuous", "roundness": 0.15}},
            "interaction": {"hover": True, "tooltipDelay": 50},
        }
        net_full.set_options(json.dumps(options_full))

        raw_html_full = net_full.generate_html(notebook=False)
        resize_full = """<script>
(function() {
    document.querySelectorAll('center').forEach(function(el) { el.remove(); });
    document.documentElement.style.cssText = 'height:100%; margin:0; overflow:hidden;';
    document.body.style.cssText = 'height:100%; margin:0; padding:0; overflow:hidden;';
    var card = document.querySelector('.card');
    if (card) card.style.cssText = 'height:100%; width:100%; padding:0; margin:0;';
    var c = document.getElementById('mynetwork');
    if (!c) return;
    c.style.cssText = 'width:100%; height:100%; position:relative; float:none; border:none;';
    setTimeout(function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    }, 500);
})();
</script>"""
        full_tree_html = raw_html_full.replace("</body>", resize_full + "</body>")
        st.components.v1.html(full_tree_html, height=720)

        st.caption(
            "蓝色=主路径 | 灰色=探索分支 | 红色菱形=终端 | "
            f"总节点: {len(nodes)} | 主路径: {len(chosen_path_ids)} 节点"
        )

    with st.expander("💾 导出推演树 JSON", expanded=False):
        tree_json_nodes = []
        for nid_key in sorted(
            nodes.keys(), key=lambda k: int(k) if str(k).isdigit() else k
        ):
            n = nodes[nid_key]
            tree_json_nodes.append(
                {
                    "id": n["id"],
                    "parent_id": n["parent_id"],
                    "children_ids": n["children_ids"],
                    "state": n["state"],
                    "action": n["action"],
                    "beam_id": n["beam_id"],
                    "quality_score": n["quality_score"],
                    "win_rate": n["win_rate"],
                    "avg_future_reward": n["avg_future_reward"],
                    "avg_step_reward": n["avg_step_reward"],
                    "visits": n["visits"],
                    "transition_prob": n["transition_prob"],
                    "cumulative_probability": n["cumulative_probability"],
                    "rollout_depth": n["rollout_depth"],
                    "is_on_chosen_path": n["is_on_chosen_path"],
                    "is_terminal": n["is_terminal"],
                    "is_beam_root": n["is_beam_root"],
                }
            )

        tree_json = json.dumps(
            {
                "meta": {
                    "start_state": start_state,
                    "action_strategy": action_strategy,
                    "score_mode": score_mode,
                    "beam_width": beam_width,
                    "lookahead_steps": lookahead_steps,
                    "max_rollout_steps": max_rollout_steps,
                    "rng_seed": rng_seed,
                    "total_nodes": len(nodes),
                    "chosen_path_length": len(chosen_path_ids),
                    "termination_reason": termination_reason,
                },
                "chosen_path_ids": chosen_path_ids,
                "nodes": tree_json_nodes,
            },
            indent=2,
            ensure_ascii=False,
        )

        b64 = base64.b64encode(tree_json.encode("utf-8")).decode()
        dl_href = f'<a href="data:application/json;base64,{b64}" download="rollout_tree_S{start_state}.json">📥 下载 rollout_tree_S{start_state}.json</a>'
        st.markdown(dl_href, unsafe_allow_html=True)
        st.caption(
            f"JSON 大小: {len(tree_json):,} 字符 ({len(tree_json_nodes)} 个节点)"
        )
        st.code(tree_json[:2000] + ("..." if len(tree_json) > 2000 else ""))


def _render_rollout_tab(kg_data, transitions, kg_entry, kg):
    st.markdown("### 从指定起始状态出发，按策略逐步滚动推演至终端状态。")
    _run_rollout(kg_data, transitions, kg_entry, kg)
