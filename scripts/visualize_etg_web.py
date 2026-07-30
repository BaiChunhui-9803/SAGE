#!/usr/bin/env python
"""
Experience Transition Graph Interactive Web Visualizer
=====================================================

Streamlit app for interactive visualization of experience transition graphs,
beam search planning, chain rollout simulation, raw data exploration, and
live game control.

Usage:  streamlit run scripts/visualize_etg_web.py
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from etg_web.i18n import render_language_switcher, st

from etg_web.constants import _ACTION_STRATEGY_LABELS, _NEXT_STATE_MODE_LABELS
from etg_web.loaders import load_etg_catalog, load_etg, load_transitions, load_etg_object
from etg_web.viz_tab import _render_visualization_tab
from etg_web.prediction_tab import _render_prediction_tab
from etg_web.rollout_tab import _render_rollout_tab
from etg_web.raw_data_tab import _render_raw_data_tab
from etg_web.live_game_tab import _render_live_game_sidebar, _render_live_game_content
from etg_web.results_tab import _render_results_tab
from etg_web.learner_tab import _render_learner_tab, _render_learner_sidebar
from etg_web.experiment_compare_tab import _render_experiment_compare_tab
from etg_web.data_filter_tab import _render_data_filter_tab


def main():
    st.set_page_config(
        page_title="Experience Transition Graph Visualizer",
        page_icon="🕸️",
        layout="wide",
    )

    st.title("🕸️ Experience Transition Graph Explorer")
    st.markdown("交互式浏览经验转移图 + 图上束搜索规划。")

    _TAB_OPTIONS = [
        "项目介绍",
        "转移图可视化",
        "束搜索规划",
        "滚动推演",
        "原始数据",
        "实时对局",
        "结果分析",
        "参数寻优",
        "批量实验",
        "数据筛选",
    ]
    _TAB_ICONS = ["📋", "🕸️", "🔮", "🎮", "📊", "⚡", "📈", "🧪", "🗂️", "🧹"]

    with st.sidebar:
        render_language_switcher()
        _sel = st.segmented_control(
            "功能",
            _TAB_OPTIONS,
            default=_TAB_OPTIONS[0],
            key="tab_selector",
        )
        if _sel:
            _sel_val = _sel[0] if isinstance(_sel, list) else _sel
            st.session_state._tab_value = _sel_val
        _sel_val = st.session_state.get("_tab_value", _TAB_OPTIONS[0])
        active_tab = _TAB_OPTIONS.index(_sel_val)
        st.session_state.active_tab = active_tab

        etg_entry = {}
        etg_data = {}
        transitions = {}
        etg_obj = None

        tabs_requiring_etg_entry = {1, 2, 3, 4, 5, 7}
        tabs_requiring_etg_data = {1, 2, 3}
        requires_etg_entry = active_tab in tabs_requiring_etg_entry
        requires_etg_data = active_tab in tabs_requiring_etg_data

        if requires_etg_entry:
            st.divider()

            catalog = load_etg_catalog()
            if not catalog:
                st.error("经验转移图目录为空，请检查 configs/etg_catalog.yaml")
                st.stop()

            maps: Dict[str, List[Dict]] = {}
            for entry in catalog:
                maps.setdefault(entry["map_id"], []).append(entry)

            map_ids = list(maps.keys())
            selected_map = st.selectbox("地图", map_ids)

            map_entries = maps[selected_map]
            etg_names = [e["name"] for e in map_entries]
            selected_etg_idx = st.selectbox(
                "经验转移图",
                options=range(len(map_entries)),
                format_func=lambda i: etg_names[i],
            )
            etg_entry = map_entries[selected_etg_idx]
            st.caption(
                f"📋 类型: {etg_entry.get('type', '-')}  |  "
                f"窗口: {etg_entry.get('context_window', 0)}  |  "
                f"data_id: {etg_entry.get('data_id', '-')}"
            )

            if requires_etg_data:
                etg_data, quality_min, quality_max = load_etg(etg_entry["file"])
                transitions = load_transitions(etg_entry.get("transitions", ""))
                etg_obj = load_etg_object(etg_entry["file"])

                if "quality_low" not in st.session_state:
                    st.session_state.quality_low = quality_min
                if "quality_high" not in st.session_state:
                    st.session_state.quality_high = quality_max

            st.divider()

        _prev_tab = st.session_state.get("_prev_tab", -1)
        tab_just_switched = _prev_tab != active_tab
        st.session_state._prev_tab = active_tab

        if requires_etg_entry and not tab_just_switched:
            if active_tab == 1:
                focus_enabled = st.checkbox(
                    "🎯 聚焦模式",
                    value=True,
                    help="仅展示指定状态的邻居子图",
                    key="viz_focus",
                )
                focus_state = None
                focus_hops = 2
                focus_forward = True
                focus_backward = True
                if focus_enabled:
                    focus_state = st.number_input(
                        "聚焦状态 ID",
                        min_value=0,
                        max_value=99999,
                        value=0,
                        key="viz_focus_state",
                    )
                    focus_hops = st.slider("扩展跳数", 1, 5, 2, key="viz_focus_hops")
                    focus_direction = st.radio(
                        "扩展方向",
                        ["双选", "作为源节点", "作为目标节点"],
                        index=0,
                        horizontal=True,
                        help="双选：同时扩展前后；源节点：仅后继；目标节点：仅前驱",
                        key="viz_focus_dir",
                    )
                    focus_forward = focus_direction in ("双选", "作为源节点")
                    focus_backward = focus_direction in ("双选", "作为目标节点")

                st.divider()

                min_visits = st.slider(
                    "最小访问次数",
                    1,
                    50,
                    1,
                    help="过滤低频 state-action 对",
                    key="viz_min_visits",
                )

                st.markdown("**Quality Score 范围**")
                quality_range = st.slider(
                    "Quality Score",
                    min_value=float(quality_min),
                    max_value=float(quality_max),
                    value=(
                        float(st.session_state.quality_low),
                        float(st.session_state.quality_high),
                    ),
                    step=0.5,
                    label_visibility="collapsed",
                    help="过滤 state-action 对的 Quality Score 范围",
                    key="viz_quality",
                )
                st.session_state.quality_low = quality_range[0]
                st.session_state.quality_high = quality_range[1]

                min_quality, max_quality = quality_range

                max_nodes = st.slider(
                    "最大节点数",
                    20,
                    500,
                    200,
                    help="限制渲染规模，避免卡顿",
                    key="viz_max_nodes",
                )

                st.divider()
                highlight_terminal = st.checkbox(
                    "高亮终端状态",
                    value=True,
                    help="Win终端=绿色, Loss终端=红色",
                    key="viz_hl_term",
                )

                st.divider()
                st.subheader("🎨 渲染设置")

                _EDGE_SMOOTH_OPTIONS = [
                    "continuous",
                    "manual_arc",
                ]
                _EDGE_SMOOTH_LABELS = {
                    "continuous": "直线",
                    "manual_arc": "弧线",
                }
                edge_smooth_type = st.selectbox(
                    "边样式",
                    options=_EDGE_SMOOTH_OPTIONS,
                    index=0,
                    format_func=lambda t: _EDGE_SMOOTH_LABELS[t],
                    help="直线: 简洁清晰; 弧线: 同一对节点间的多条边自动扇形分散",
                    key="viz_edge_smooth",
                )
                edge_roundness = st.slider(
                    "弯曲度",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    disabled=(edge_smooth_type != "manual_arc"),
                    help="手动弧线模式下，弧线弯曲的基础倍率",
                    key="viz_edge_round",
                )

                _LAYOUT_OPTIONS = [
                    "force_atlas_2based",
                    "barnes_hut",
                    "repulsion",
                    "hrepulsion",
                ]
                _LAYOUT_LABELS = {
                    "force_atlas_2based": "Force Atlas 2 (默认)",
                    "barnes_hut": "Barnes-Hut",
                    "repulsion": "Repulsion",
                    "hrepulsion": "Hierarchical Repulsion",
                }
                st.caption("**布局算法**")
                col_layout, col_btn = st.columns([5, 2])
                with col_layout:
                    layout_algorithm = st.selectbox(
                        "布局算法",
                        options=_LAYOUT_OPTIONS,
                        index=0,
                        format_func=lambda a: _LAYOUT_LABELS[a],
                        label_visibility="collapsed",
                        help="不同力导向算法影响节点分布方式",
                        key="viz_layout",
                    )
                with col_btn:
                    render_clicked = st.button(
                        "🔄 渲染", use_container_width=True, key="viz_render_btn"
                    )

                if "render_key" not in st.session_state:
                    st.session_state.render_key = 0
                if render_clicked:
                    st.session_state.render_key += 1

                freeze_layout = st.checkbox(
                    "🔒 冻结布局（拖拽不回弹）", value=False, key="viz_freeze"
                )

                st.divider()
                st.caption("数据来源: `cache/experience_transition_graph/`")

            elif active_tab == 2:
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        "起始状态 ID",
                        min_value=0,
                        max_value=99999,
                        value=0,
                        key="pred_state",
                    )
                with c2:
                    st.selectbox(
                        "评分策略",
                        options=["quality", "future_reward", "win_rate"],
                        format_func={
                            "quality": "Quality Score",
                            "future_reward": "Future Reward",
                            "win_rate": "Win Rate",
                        }.get,
                        key="sm",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    st.slider("Beam Width", 1, 10, 3, key="beam_w")
                with c4:
                    st.slider("最大步数", 1, 15, 5, key="max_s")

                c5, c6 = st.columns(2)
                with c5:
                    st.slider("最低访问次数", 1, 10, 1, key="mv_pred")
                with c6:
                    st.slider(
                        "最大状态重复",
                        1,
                        5,
                        2,
                        key="msr",
                        help="每条 beam 路径中同一状态最多出现的次数。设为 1 则完全禁止重复访问。",
                    )

                c7, c8 = st.columns(2)
                with c7:
                    st.slider(
                        "累积概率阈值",
                        0.001,
                        0.1,
                        0.01,
                        step=0.001,
                        format="%.3f",
                        key="mcp",
                    )
                with c8:
                    st.slider(
                        "折扣因子",
                        0.5,
                        1.0,
                        0.9,
                        step=0.05,
                        format="%.2f",
                        key="df",
                        help="每步累积概率乘以此值的步数次幂。1.0=无折扣，越低越惩罚过深路径。",
                    )

                st.button(
                    "🔮 开始规划",
                    type="primary",
                    use_container_width=True,
                    key="pred_btn",
                )

            elif active_tab == 3:
                st.caption("单步规划参数（每步束搜索）")
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        "起始状态 ID",
                        min_value=0,
                        max_value=99999,
                        value=0,
                        key="roll_state",
                    )
                with c2:
                    st.selectbox(
                        "评分策略",
                        options=["quality", "future_reward", "win_rate"],
                        format_func={
                            "quality": "Quality Score",
                            "future_reward": "Future Reward",
                            "win_rate": "Win Rate",
                        }.get,
                        key="roll_sm",
                    )

                c3, c4 = st.columns(2)
                with c3:
                    st.slider("Beam Width", 1, 10, 3, key="roll_bw")
                with c4:
                    st.slider("前瞻步数", 1, 15, 5, key="roll_la")

                c5, c6 = st.columns(2)
                with c5:
                    st.slider("最低访问次数", 1, 10, 1, key="roll_mv")
                with c6:
                    st.slider(
                        "最大状态重复",
                        1,
                        5,
                        2,
                        key="roll_msr",
                        help="同一条推演路径中同一状态最多出现的次数。设为 1 则完全禁止重复访问。",
                    )

                c7, c8 = st.columns(2)
                with c7:
                    st.slider(
                        "累积概率阈值",
                        0.001,
                        0.1,
                        0.01,
                        step=0.001,
                        format="%.3f",
                        key="roll_mcp",
                    )
                with c8:
                    st.slider(
                        "折扣因子",
                        0.5,
                        1.0,
                        0.9,
                        step=0.05,
                        format="%.2f",
                        key="roll_df",
                        help="每步累积概率乘以此值的步数次幂。1.0=无折扣，越低越惩罚过深路径。",
                    )

                st.caption("滚动推演参数")
                c9, c10, c11 = st.columns([6, 4, 1.5])
                with c9:
                    st.selectbox(
                        "动作选择",
                        options=list(_ACTION_STRATEGY_LABELS.keys()),
                        format_func=lambda x: _ACTION_STRATEGY_LABELS[x],
                        key="roll_as",
                    )
                with c10:
                    st.selectbox(
                        "状态转移",
                        options=list(_NEXT_STATE_MODE_LABELS.keys()),
                        format_func=lambda x: _NEXT_STATE_MODE_LABELS[x],
                        key="roll_nsm",
                    )
                with c11:
                    st.text_input("seed", value="42", key="roll_seed")

                c12, c13 = st.columns(2)
                with c12:
                    st.slider(
                        "ε",
                        0.01,
                        0.5,
                        0.1,
                        step=0.01,
                        format="%.2f",
                        key="roll_eps",
                        disabled=(st.session_state.get("roll_as") != "epsilon_greedy"),
                    )
                with c13:
                    st.slider("最大推演步数", 1, 100, 50, key="roll_mrs")

                st.markdown("**推演模式**")
                st.radio(
                    "mode",
                    options=["单步推演", "多步推演"],
                    index=0,
                    horizontal=True,
                    key="roll_mode",
                    label_visibility="collapsed",
                )

                if st.session_state.get("roll_mode") == "多步推演":
                    st.toggle("启用备选路径", value=False, key="roll_backup")
                    if st.session_state.get("roll_backup", False):
                        st.slider(
                            "备选评分阈值",
                            0.0,
                            1.0,
                            0.3,
                            step=0.05,
                            format="%.2f",
                            key="roll_backup_st",
                        )
                        st.slider(
                            "模糊匹配距离阈值",
                            0.0,
                            1.0,
                            0.2,
                            step=0.05,
                            format="%.2f",
                            key="roll_backup_dt",
                        )

                st.button(
                    "🎲 开始推演",
                    type="primary",
                    use_container_width=True,
                    key="roll_btn",
                )

            elif active_tab == 5:
                _render_live_game_sidebar(etg_entry)

            elif active_tab == 7:
                _render_learner_sidebar(etg_entry)

    st.markdown(f"### {_TAB_ICONS[active_tab]} {_TAB_OPTIONS[active_tab]}")

    if tab_just_switched:
        st.spinner("")
        st.rerun()

    if active_tab == 0:
        readme_path = Path(__file__).parent.parent / "README.md"
        if readme_path.exists():
            st.markdown(readme_path.read_text(encoding="utf-8"))
        else:
            st.info("README.md 尚未创建。请在项目根目录放置 README.md 文件。")

    elif active_tab == 1:
        _render_visualization_tab(
            etg_data,
            transitions,
            etg_entry,
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
        )

    elif active_tab == 2:
        _render_prediction_tab(etg_data, transitions, etg_entry, etg_obj)

    elif active_tab == 3:
        _render_rollout_tab(etg_data, transitions, etg_entry, etg_obj)

    elif active_tab == 4:
        _render_raw_data_tab(etg_entry)

    elif active_tab == 5:
        _render_live_game_content()

    elif active_tab == 6:
        _render_results_tab()

    elif active_tab == 7:
        _render_learner_tab()

    elif active_tab == 8:
        _render_experiment_compare_tab()

    elif active_tab == 9:
        _render_data_filter_tab()


if __name__ == "__main__":
    main()
