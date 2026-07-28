import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pickle
import base64
import streamlit as st
import numpy as np
from src.decision.knowledge_graph import DecisionKnowledgeGraph
from src.decision.kg_beam_search import (
    plan_action,
    get_beam_paths,
    BeamSearchResult,
)
from src.decision.beam_matcher import match_beam_paths
from kg_web.constants import KG_DIR
from kg_web.loaders import load_episode_data, load_distance_matrix_np
from kg_web.beam_utils import (
    _compute_composite_scores,
    _build_rec_rows,
    _build_path_detail_rows,
    _results_to_json,
)
from kg_web.pyvis_renderer import render_beam_tree


@st.cache_data(max_entries=200, show_spinner=False)
def _cached_beam_results(
    _kg_path: str,
    _trans_path: str,
    start_state: int,
    beam_width: int,
    max_steps: int,
    min_visits: int,
    min_cum_prob: float,
    score_mode: str,
    max_state_revisits: int,
    discount_factor: float,
):
    kg = DecisionKnowledgeGraph.load(str(KG_DIR / _kg_path))
    with open(str(KG_DIR / _trans_path), "rb") as f:
        transitions = pickle.load(f)

    plan = plan_action(
        kg,
        transitions,
        start_state,
        beam_width=beam_width,
        max_steps=max_steps,
        min_visits=min_visits,
        min_cum_prob=min_cum_prob,
        score_mode=score_mode,
        max_state_revisits=max_state_revisits,
        discount_factor=discount_factor,
    )

    if plan.recommended_action is None:
        return {"action": None}

    results = plan.beam_results

    results_ser = [
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
        for r in results
    ]

    beam_paths = plan.beam_paths
    beam_paths_ser = []
    for path in beam_paths:
        beam_paths_ser.append(
            [
                {
                    "step": r.step,
                    "state": r.state,
                    "action": r.action,
                    "cumulative_probability": r.cumulative_probability,
                    "quality_score": r.quality_score,
                    "win_rate": r.win_rate,
                    "avg_step_reward": r.avg_step_reward,
                    "avg_future_reward": r.avg_future_reward,
                }
                for r in path
            ]
        )

    return {
        "action": plan.recommended_action,
        "info": {
            "expected_cumulative_reward": plan.metrics.get(
                "expected_cumulative_reward", 0.0
            ),
            "expected_win_rate": plan.metrics.get("expected_win_rate", 0.0),
            "best_beam_cum_prob": plan.metrics.get("best_beam_cum_prob", 0.0),
            "best_beam_length": plan.metrics.get("best_beam_length", 0),
            "reason": plan.metrics.get("reason"),
        },
        "results": results_ser,
        "beam_paths": beam_paths_ser,
    }


@st.fragment
def _run_prediction(kg_data, transitions, kg_entry, kg):
    start_state = st.session_state.get("pred_state", 0)
    score_mode = st.session_state.get("sm", "quality")
    beam_width = st.session_state.get("beam_w", 3)
    max_steps = st.session_state.get("max_s", 5)
    min_cum_prob = st.session_state.get("mcp", 0.01)
    min_visits_pred = st.session_state.get("mv_pred", 1)
    max_state_revisits = st.session_state.get("msr", 2)
    discount_factor = st.session_state.get("df", 0.9)

    unique_states = kg_data.get("unique_states", set())
    if isinstance(unique_states, set):
        if start_state not in unique_states:
            st.toast(f"状态 {start_state} 不存在于经验转移图中", icon="⚠️")
            st.warning(
                f"状态 {start_state} 不存在于当前经验转移图中。请选择一个有效状态。"
            )
            return

    if start_state not in transitions:
        st.warning(f"状态 {start_state} 没有转移数据，无法进行规划。")
        return

    kg_file = kg_entry.get("file", "")
    trans_file = kg_entry.get("transitions", "")

    cached = _cached_beam_results(
        kg_file,
        trans_file,
        start_state,
        beam_width,
        max_steps,
        min_visits_pred,
        min_cum_prob,
        score_mode,
        max_state_revisits,
        discount_factor,
    )

    if cached["action"] is None:
        st.error("无法规划：该状态无可用动作。")
        return

    action = cached["action"]
    info = cached["info"]
    results = [BeamSearchResult(**r) for r in cached["results"]]

    beam_paths = get_beam_paths(results)
    if not beam_paths:
        st.info("无搜索结果。")
        return

    composites, path_metrics = _compute_composite_scores(beam_paths)

    sorted_indices = sorted(
        range(len(beam_paths)), key=lambda i: composites[i], reverse=True
    )

    if (
        "pred_selected_path" not in st.session_state
        or st.session_state.pred_selected_path >= len(beam_paths)
    ):
        st.session_state.pred_selected_path = sorted_indices[0] if sorted_indices else 0

    selected_path_idx = st.session_state.pred_selected_path
    selected_path = beam_paths[selected_path_idx]
    highlight_set = {i for i, r in enumerate(results) if r in selected_path}

    col_action, col_tree, col_rec = st.columns([0.15, 0.85, 0.8])

    with col_action:
        st.markdown("**🎯 最优动作推荐**")
        st.metric("推荐动作", action)
        st.metric("预期累积奖励", f"{info['expected_cumulative_reward']:.3f}")
        st.metric("预期胜率", f"{info['expected_win_rate']:.2%}")
        st.metric("最优路径累积概率", f"{info['best_beam_cum_prob']:.4f}")
        st.metric("最优路径步数", info["best_beam_length"])

        if info.get("reason") == "no_transitions":
            st.caption("注意：该状态无转移数据，推荐结果仅基于单步质量评分。")

    with col_tree:
        with st.spinner("生成路径树..."):
            tree_html = render_beam_tree(results, highlight_indices=highlight_set)

        tree_b64 = base64.b64encode(tree_html.encode()).decode()
        tree_uri = f"data:text/html;charset=utf-8;base64,{tree_b64}#v=1"
        json_b64 = base64.b64encode(
            _results_to_json(
                results, beam_width, max_steps, min_cum_prob, score_mode, start_state
            ).encode()
        ).decode()
        json_uri = f"data:application/json;base64,{json_b64}"
        st.markdown(
            f"**🌳 路径树图**  "
            f'<a href="{tree_uri}" '
            f'download="beam_tree_state{start_state}.html" '
            f'style="font-size:0.85em;color:#4CAF50;text-decoration:none;margin-right:12px;">📥 导出 HTML</a>'
            f'<a href="{json_uri}" '
            f'download="beam_tree_state{start_state}.json" '
            f'style="font-size:0.85em;color:#2196F3;text-decoration:none;">📥 导出 JSON</a>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'''<div style="overflow:hidden; width:100%; height:500px; border:2px solid #444; border-radius:4px;">
        <iframe src="{tree_uri}" width="100%" height="100%" style="border:none; display:block;"></iframe>
      </div>''',
            unsafe_allow_html=True,
        )
        st.caption("节点颜色对应不同 Beam；鼠标悬停查看详情。")

    with col_rec:
        st.markdown("**📊 路径推荐**")

        rec_rows = _build_rec_rows(sorted_indices, beam_paths, composites, path_metrics)

        event = st.dataframe(
            rec_rows,
            use_container_width=True,
            height=min(len(rec_rows) * 35 + 50, 400),
            on_select="rerun",
            selection_mode="single-row",
        )

        if event and event.selection.rows:
            new_path = rec_rows[event.selection.rows[0]]["Path"]
            if new_path != st.session_state.pred_selected_path:
                st.session_state.pred_selected_path = new_path
                st.rerun()

    col_detail, col_frag = st.columns([1, 0.8])

    with col_detail:
        st.markdown("**📋 路径详情**")

        all_rows = _build_path_detail_rows(beam_paths)

        st.dataframe(
            all_rows, use_container_width=True, height=min(len(all_rows) * 35 + 50, 500)
        )

    with col_frag:
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
                match_results = match_beam_paths(
                    {i: path for i, path in enumerate(beam_paths)},
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
                                "Path": bid,
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
                height=min(len(match_rows) * 35 + 50, 500),
            )
            wins = sum(1 for r in match_rows if r["结果"] == "Win")
            total_m = len(match_rows)
            st.caption(
                f"共 {total_m} 条匹配 | Win: {wins} ({wins / total_m:.0%}) | "
                f"距离矩阵: {'有' if dist_mat is not None else '无(精确匹配)'}"
            )
        else:
            st.info("无匹配结果。原始对局数据可能未加载。")


def _render_prediction_tab(kg_data, transitions, kg_entry, kg):
    st.markdown("### 从指定起始状态出发，用 Beam Search 在转移图上进行多步规划搜索。")
    _run_prediction(kg_data, transitions, kg_entry, kg)
