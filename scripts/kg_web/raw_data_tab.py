import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import numpy as np
from kg_web.constants import NPY_DIR
from kg_web.loaders import (
    load_episode_data,
    load_distance_matrix_np,
    load_state_hp_data,
)
from kg_web.raw_data_utils import compute_mds, _build_hp_diff_array, plot_mds_terrain


@st.fragment
def _run_episode_query(n_ep, outcomes, scores, ep_states, ep_actions):
    st.subheader("📋 Episode 查询")

    col_ep, col_btn = st.columns([4, 1])
    with col_ep:
        ep_id = st.number_input(
            "Episode ID", min_value=0, max_value=n_ep - 1, value=0, key="raw_ep"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 查询", type="primary", use_container_width=True, key="raw_btn")

    if ep_id >= n_ep:
        st.error(f"Episode ID 超出范围 (0~{n_ep - 1})")
    else:
        outcome = outcomes[ep_id] if ep_id < len(outcomes) else "Unknown"
        score = scores[ep_id] if ep_id < len(scores) else 0.0
        states = ep_states[ep_id] if ep_id < len(ep_states) else []
        actions = ep_actions[ep_id] if ep_id < len(ep_actions) else []
        n_steps = len(states)

        st.markdown(
            f"**Episode #{ep_id}**  |  {outcome}  |  得分: {score}  |  步数: {n_steps}"
        )

        if n_steps > 0:
            rows = []
            for i in range(n_steps):
                act = actions[i] if i < len(actions) else ""
                rows.append(
                    {
                        "Step": i,
                        "State": states[i],
                        "Action": act,
                    }
                )

            st.dataframe(
                rows, use_container_width=True, height=min(n_steps * 35 + 50, 600)
            )


@st.fragment
def _run_distance_query(map_id, data_id):
    st.subheader("🔬 状态距离查询")

    dist_mat = load_distance_matrix_np(map_id, data_id)

    if dist_mat is None:
        st.warning("当前地图无距离矩阵缓存。")
        return

    n_states = dist_mat.shape[0]

    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        query_state = st.number_input(
            "查询状态 ID",
            min_value=0,
            max_value=n_states - 1,
            value=0,
            key="dist_state",
        )
    with col_q2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "🔍 查询距离", type="primary", use_container_width=True, key="dist_btn"
        )

    col_m1, col_m2 = st.columns([1.5, 6])
    with col_m1:
        query_mode = st.segmented_control(
            "查询模式",
            options=["距离阈值", "Top-K"],
            key="dist_mode",
            label_visibility="collapsed",
        )
    with col_m2:
        threshold = 1.0
        top_k = 10
        if query_mode == "距离阈值":
            threshold = st.slider(
                "距离阈值",
                0.0,
                float(np.max(dist_mat)),
                1.0,
                step=0.1,
                format="%.1f",
                key="dist_thresh",
                label_visibility="collapsed",
            )
        else:
            top_k = st.slider(
                "Top-K", 1, 50, 10, key="dist_topk", label_visibility="collapsed"
            )

    if query_state >= n_states:
        st.error(f"状态 ID 超出范围（0~{n_states - 1}）")
    else:
        row = dist_mat[query_state]

        if query_mode == "距离阈值":
            indices = np.where(
                (row <= threshold) & (np.arange(n_states) != query_state)
            )[0]
            dists = row[indices]
            order = np.argsort(dists)
            indices = indices[order]
            dists = dists[order]
        else:
            sorted_idx = np.argsort(row)
            indices = sorted_idx[1 : top_k + 1]
            dists = row[indices]

        st.info(
            f"状态 {query_state} 在 {n_states} 个状态中"
            f"{'，阈值 ' + str(threshold) + ' 内' if query_mode == '距离阈值' else '，Top-' + str(top_k)}"
            f"有 {len(indices)} 个匹配"
        )

        dist_rows = [
            {"State": int(idx), "Distance": round(float(d), 4)}
            for idx, d in zip(indices, dists)
        ]
        st.dataframe(
            dist_rows,
            use_container_width=True,
            height=min(len(dist_rows) * 35 + 50, 500),
        )


@st.fragment
def _run_hp_query(map_id, data_id):
    st.subheader("❤️ 状态 HP 查询")

    hp_data = load_state_hp_data(map_id, data_id)
    n_hp_states = hp_data["n_states"]

    if n_hp_states == 0:
        st.warning(
            "无 HP 数据。请确认 data/ 目录下有 bktree/ 和 graph/state_node.txt。"
        )
        return

    col_hp1, col_hp2 = st.columns([4, 1])
    with col_hp1:
        hp_state_id = st.number_input(
            "查询状态 ID",
            min_value=0,
            max_value=n_hp_states - 1,
            value=0,
            key="hp_state",
        )
    with col_hp2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 查询 HP", type="primary", use_container_width=True, key="hp_btn")

    if hp_state_id >= n_hp_states:
        st.error(f"状态 ID 超出范围（0~{n_hp_states - 1}）")
    elif hp_state_id not in hp_data["hp_lookup"]:
        st.warning(f"状态 {hp_state_id} 在 BK-Tree 中无对应节点。")
    else:
        info = hp_data["hp_lookup"][hp_state_id]
        cluster = info["cluster"]
        score = info["score"]
        red = info["red_army"]
        blue = info["blue_army"]

        red_hp = [u[2] * 100 for u in red]
        blue_hp = [u[2] * 100 for u in blue]
        red_avg = sum(red_hp) / len(red_hp) if red_hp else 0
        blue_avg = sum(blue_hp) / len(blue_hp) if blue_hp else 0
        red_total = sum(red_hp)
        blue_total = sum(blue_hp)
        hp_diff = red_total - blue_total
        diff_label = (
            "红方优势" if hp_diff > 0 else "蓝方优势" if hp_diff < 0 else "持平"
        )

        st.markdown(
            f"**状态 {hp_state_id}**  |  聚类: `{cluster}`  |  Score: {score:.2f}"
        )

        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        c_h1.metric("红方单位数", len(red))
        c_h2.metric("红方平均 HP", f"{red_avg:.1f}%")
        c_h3.metric("蓝方单位数", len(blue))
        c_h4.metric("蓝方平均 HP", f"{blue_avg:.1f}%")

        c_h5, c_h6 = st.columns(2)
        c_h5.metric("HP 差值", f"{hp_diff:+.1f}%", diff_label)
        c_h6.metric("红/蓝总 HP", f"{red_total:.1f}% / {blue_total:.1f}%")

        with st.expander("红方单位详情"):
            red_rows = [
                {
                    "单位": i + 1,
                    "X": f"{u[0]:.3f}",
                    "Y": f"{u[1]:.3f}",
                    "HP%": f"{u[2] * 100:.1f}%",
                }
                for i, u in enumerate(red)
            ]
            st.dataframe(red_rows, use_container_width=True)

        with st.expander("蓝方单位详情"):
            blue_rows = [
                {
                    "单位": i + 1,
                    "X": f"{u[0]:.3f}",
                    "Y": f"{u[1]:.3f}",
                    "HP%": f"{u[2] * 100:.1f}%",
                }
                for i, u in enumerate(blue)
            ]
            st.dataframe(blue_rows, use_container_width=True)


@st.fragment
def _run_mds_terrain(map_id, data_id, n_states_kg):
    st.subheader("🗺️ 状态地形图（MDS 降维 + HP 插值）")

    dist_path = NPY_DIR / f"state_distance_matrix_{map_id}_{data_id}.npy"
    if not dist_path.exists():
        st.warning("当前地图无距离矩阵缓存，无法计算 MDS。")
        return

    dist_mat = np.load(str(dist_path))
    n_states_dist = dist_mat.shape[0]

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("状态数", n_states_dist)
    c2.metric("距离矩阵", f"{n_states_dist}x{n_states_dist}")

    cache_path = NPY_DIR / f"mds_{map_id}_{data_id}.npy"
    has_cache = cache_path.exists()

    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        do_compute = st.button(
            f"🗺️ {'加载' if has_cache else '计算'}地形图",
            type="primary",
            use_container_width=True,
            key="mds_btn",
        )

    if not do_compute:
        if has_cache:
            st.info(f"已有缓存: `cache/npy/mds_{map_id}_{data_id}.npy`，点击加载。")
        else:
            n_est = max(1, int(n_states_dist * n_states_dist * 8e-9))
            st.info(f"首次计算预计需要 ~{n_est} 分钟，计算结果将缓存到本地。")
        return

    coords = compute_mds(map_id, data_id)
    if coords is None:
        st.error("MDS 计算失败：无法加载距离矩阵。")
        return

    hp_data = load_state_hp_data(map_id, data_id)
    hp_diff = _build_hp_diff_array(hp_data, n_states_dist)

    with st.spinner("正在生成地形图..."):
        fig = plot_mds_terrain(coords, hp_diff)

    st.markdown(
        '<div class="mds-sq" style="height:0;overflow:hidden"></div>'
        "<style>"
        ".stVerticalBlock:has(.mds-sq) [data-testid='stPlotlyChart'] {"
        "  width: min(100%, 85vh) !important;"
        "  aspect-ratio: 1/1 !important;"
        "  margin: 0 auto !important;"
        "}"
        ".stVerticalBlock:has(.mds-sq) .js-plotly-plot,"
        ".stVerticalBlock:has(.mds-sq) .plot-container,"
        ".stVerticalBlock:has(.mds-sq) .svg-container {"
        "  width: 100% !important; height: 100% !important;"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, height=700)
    st.caption(
        "X/Y: MDS 降维坐标 | 颜色: HP 差值（红方优势=红色, 蓝方优势=蓝色） | 支持拖拽/缩放/悬停查看"
    )


def _render_raw_data_tab(kg_entry):
    map_id = kg_entry.get("map_id", "")
    data_id = kg_entry.get("data_id", "")

    if not map_id or not data_id:
        st.warning("当前 KG 条目缺少 map_id 或 data_id。")
        return

    ep_data = load_episode_data(map_id, data_id)
    n_ep = ep_data["n_episodes"]
    outcomes = ep_data["outcomes"]
    scores = ep_data["scores"]
    ep_states = ep_data["states"]
    ep_actions = ep_data["actions"]

    if n_ep == 0:
        st.error("未加载到原始对局数据。请确认 data/ 目录下有对应文件。")
        return

    n_win = sum(1 for o in outcomes if o == "Win")
    n_loss = sum(1 for o in outcomes if o == "Loss")
    avg_len = sum(len(s) for s in ep_states) / n_ep if n_ep > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总 Episodes", n_ep)
    c2.metric("Win", n_win, f"{n_win / n_ep:.1%}" if n_ep else "")
    c3.metric("Loss", n_loss, f"{n_loss / n_ep:.1%}" if n_ep else "")
    c4.metric("平均步数 / 平均得分", f"{avg_len:.1f}", f"{avg_score:.1f}")

    c_ep, c_dist = st.columns(2)
    with c_ep:
        _run_episode_query(n_ep, outcomes, scores, ep_states, ep_actions)
    with c_dist:
        _run_distance_query(map_id, data_id)

    c_hp, c_mds = st.columns(2)
    with c_hp:
        _run_hp_query(map_id, data_id)
    with c_mds:
        _run_mds_terrain(map_id, data_id, len(ep_data.get("states", [])))
