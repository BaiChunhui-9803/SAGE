import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import base64
import streamlit as st
from kg_web.graph_builder import build_graph_data
from kg_web.pyvis_renderer import render_pyvis_html


def _render_visualization_tab(
    kg_data,
    transitions,
    kg_entry,
    min_visits,
    min_quality,
    max_quality,
    max_nodes,
    focus_state,
    focus_hops,
    focus_forward,
    focus_backward,
    focus_enabled,
    highlight_terminal,
    edge_smooth_type,
    edge_roundness,
    layout_algorithm,
    freeze_layout,
):
    n_states = len(kg_data.get("unique_states", set()))
    n_actions = len(kg_data.get("unique_actions", set()))
    total_visits = kg_data.get("total_visits", 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("状态节点", n_states)
    col2.metric("动作种类", n_actions)
    col3.metric("总访问次数", total_visits)

    with st.spinner("构建图谱..."):
        nodes, edges, state_stats, state_found = build_graph_data(
            kg_data,
            transitions,
            min_visits=min_visits,
            min_quality=min_quality,
            max_quality=max_quality,
            max_nodes=max_nodes,
            focus_state=focus_state,
            focus_hops=focus_hops,
            focus_forward=focus_forward,
            focus_backward=focus_backward,
        )

    if focus_enabled and not state_found:
        st.toast(f"状态 {focus_state} 不存在于经验转移图中", icon="⚠️")
        st.info("当前渲染: **0 个节点**, **0 条边**")
        st.stop()

    st.info(f"当前渲染: **{len(nodes)} 个节点**, **{len(edges)} 条边**")

    if not nodes:
        st.warning("没有满足条件的节点，请放宽筛选条件。")
        st.stop()

    with st.spinner("渲染可视化（首次可能需要几秒）..."):
        html = render_pyvis_html(
            nodes,
            edges,
            state_stats,
            highlight_terminal,
            edge_smooth_type=edge_smooth_type,
            edge_roundness=edge_roundness,
            layout_algorithm=layout_algorithm,
            freeze_layout=freeze_layout,
            canvas_height=750,
        )

    b64 = base64.b64encode(html.encode()).decode()
    data_uri = (
        f"data:text/html;charset=utf-8;base64,{b64}#v={st.session_state.render_key}"
    )
    st.markdown(
        f'''<div style="resize:both; overflow:hidden; width:100%; height:800px; max-width:100%; border:2px solid #444; border-radius:4px;">
        <iframe src="{data_uri}" width="100%" height="100%" style="border:none; display:block;"></iframe>
      </div>''',
        unsafe_allow_html=True,
    )

    st.divider()

    with st.expander("📊 节点统计表（按访问量排序）"):
        table_data = sorted(nodes, key=lambda x: x["total_visits"], reverse=True)
        st.dataframe(
            [
                {
                    "State": n["id"],
                    "Visits": n["total_visits"],
                    "Actions": n["n_actions"],
                    "Best Win Rate": f"{n['best_win_rate'] * 100:.1f}%",
                    "Best Quality": f"{n['best_quality']:.1f}",
                }
                for n in table_data
            ],
            use_container_width=True,
        )

    with st.expander("📊 边统计表（按质量排序 Top 50）"):
        sorted_edges = sorted(edges, key=lambda x: x["quality_score"], reverse=True)[
            :50
        ]
        st.dataframe(
            [
                {
                    "From": e["from"],
                    "To": e["to"],
                    "Action": e["action"],
                    "Visits": e["visits"],
                    "Quality": f"{e['quality_score']:.1f}",
                    "Win Rate": f"{e['win_rate'] * 100:.1f}%",
                    "Step Reward": f"{e['avg_step_reward']:.2f}",
                    "Future Reward": f"{e['avg_future_reward']:.2f}",
                    "Trans Prob": f"{e['transition_prob'] * 100:.1f}%",
                }
                for e in sorted_edges
            ],
            use_container_width=True,
        )

    with st.expander("🔍 状态搜索"):
        search_id = st.number_input(
            "输入状态 ID", min_value=0, max_value=99999, value=0, key="search"
        )
        if st.button("搜索", key="search_btn"):
            if search_id in state_stats:
                info = state_stats[search_id]
                st.json(
                    {
                        "state_id": search_id,
                        "total_visits": info["total_visits"],
                        "n_available_actions": info["n_actions"],
                        "best_quality_score": round(info["best_quality"], 2),
                        "best_win_rate": round(info["best_win_rate"], 4),
                    }
                )

                key = search_id if not kg_data["use_context"] else (search_id, ())
                actions = kg_data["state_action_map"].get(key, {})
                if actions:
                    st.subheader("该状态的所有动作统计")
                    action_rows = []
                    for act, stats in sorted(
                        actions.items(), key=lambda x: x[1].quality_score, reverse=True
                    ):
                        action_rows.append(
                            {
                                "Action": act,
                                "Visits": stats.visits,
                                "Quality": f"{stats.quality_score:.2f}",
                                "Win Rate": f"{stats.win_rate * 100:.1f}%",
                                "Avg Step Reward": f"{stats.avg_step_reward:.2f}",
                                "Avg Future Reward": f"{stats.avg_future_reward:.2f}",
                            }
                        )
                    st.dataframe(action_rows, use_container_width=True)

                    if search_id in transitions:
                        st.subheader("该状态的转移概率")
                        trans_rows = []
                        for act, t_info in transitions[search_id].items():
                            ns_dict = t_info.get("next_states", {})
                            total_c = sum(ns_dict.values())
                            for ns, c in sorted(
                                ns_dict.items(), key=lambda x: x[1], reverse=True
                            ):
                                trans_rows.append(
                                    {
                                        "Action": act,
                                        "Next State": ns,
                                        "Count": c,
                                        "Probability": f"{c / total_c * 100:.1f}%"
                                        if total_c
                                        else "N/A",
                                    }
                                )
                        st.dataframe(trans_rows, use_container_width=True)
                else:
                    st.info("该状态在当前 KG 中无动作记录。")
            else:
                st.warning(f"状态 {search_id} 不存在于经验转移图中。")
