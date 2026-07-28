import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kg_web.constants import RESULTS_DIR


@st.cache_data(max_entries=50, show_spinner=False)
def _load_result_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _scan_results():
    if not RESULTS_DIR.exists():
        return []
    items = []
    for jf in sorted(
        RESULTS_DIR.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            with open(str(jf), "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            params = data.get("params", {})
            mode_text = meta.get("mode", "?")
            if meta.get("backup_enabled") and mode_text not in (
                "single_step",
                "replay",
            ):
                mode_text = f"{mode_text}_backup"
            param_tags = []
            sm = params.get("score_mode", "")
            if sm:
                param_tags.append(f"sm={sm}")
            bw = params.get("beam_width")
            if bw is not None:
                param_tags.append(f"bw={bw}")
            la = params.get("max_steps", params.get("lookahead_steps"))
            if la is not None:
                param_tags.append(f"la={la}")
            mv = params.get("min_visits")
            if mv is not None:
                param_tags.append(f"mv={mv}")
            msr = params.get("max_state_revisits")
            if msr is not None:
                param_tags.append(f"msr={msr}")
            mcp = params.get("min_cum_prob")
            if mcp is not None:
                param_tags.append(f"mcp={mcp}")
            df = params.get("discount_factor")
            if df is not None:
                param_tags.append(f"df={df}")
            as_ = params.get("action_strategy", "")
            if as_:
                param_tags.append(f"as={as_}")
            if as_ == "epsilon_greedy":
                eps = params.get("epsilon")
                if eps is not None:
                    param_tags.append(f"eps={eps}")
            if meta.get("backup_enabled") and mode_text.endswith("_backup"):
                bst = params.get("backup_score_threshold")
                if bst is not None:
                    param_tags.append(f"bp={bst}")
                bdt = params.get("backup_distance_threshold")
                if bdt is not None:
                    param_tags.append(f"bd={bdt}")
            param_str = " ".join(param_tags) if param_tags else ""
            label = (
                f"{mode_text} | {param_str} | {meta.get('timestamp', '?')} | {meta.get('num_episodes', 0)}局"
                if param_str
                else f"{mode_text} | {meta.get('timestamp', '?')} | {meta.get('num_episodes', 0)}局"
            )
            items.append(
                {
                    "path": str(jf),
                    "label": label,
                    "map_id": meta.get("map_id", ""),
                    "mode": meta.get("mode", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "num_episodes": meta.get("num_episodes", 0),
                }
            )
        except Exception:
            continue
    return items


def _extract_stats(data):
    episodes = data.get("episodes", [])
    rows = []
    for ep in episodes:
        frames = ep.get("frames", [])
        score = ep.get("score", 0)
        if not score and frames:
            last = frames[-1]
            score = last.get("hp_my", 0) - last.get("hp_enemy", 0)
        result = ep.get("result", "Unknown")

        action_sources = {}
        hp_diff_curve = []
        for fr in frames:
            src = fr.get("action_source", "unknown")
            action_sources[src] = action_sources.get(src, 0) + 1
            hp_diff_curve.append(fr.get("hp_my", 0) - fr.get("hp_enemy", 0))

        total_actions = sum(action_sources.values()) or 1
        action_pcts = {k: v / total_actions for k, v in action_sources.items()}

        rows.append(
            {
                "episode_id": ep.get("episode_id", 0),
                "result": result,
                "score": score,
                "num_frames": len(frames),
                "win": result == "Win",
                "action_sources": action_sources,
                "action_pcts": action_pcts,
                "hp_diff_curve": hp_diff_curve,
            }
        )
    return rows


def _plot_score_chart(all_stats, labels):
    fig = go.Figure()

    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

    for idx, (stats, label) in enumerate(zip(all_stats, labels)):
        scores = [r["score"] for r in stats]
        episodes = list(range(1, len(scores) + 1))
        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=scores,
                mode="lines+markers",
                name=label,
                line=dict(color=color),
                marker=dict(
                    color=["#2ecc71" if r["win"] else "#e74c3c" for r in stats],
                    size=8,
                    line=dict(width=1, color="white"),
                ),
            )
        )

        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = variance**0.5
            upper = [mean + std] * len(episodes)
            lower = [mean - std] * len(episodes)

            fig.add_trace(
                go.Scatter(
                    x=episodes + episodes[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    name=f"{label} ±1σ",
                )
            )

            fig.add_hline(
                y=mean,
                line_dash="dash",
                line_color=color,
                opacity=0.5,
                annotation_text=f"{label}: μ={mean:.1f}",
                annotation_position="top left",
            )

    fig.update_layout(
        title="得分折线图（带均值误差带）",
        xaxis_title="Episode",
        yaxis_title="得分 (HP差值)",
        height=400,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def _plot_winrate_chart(all_stats, labels):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("胜率统计", "累积胜率趋势"),
        column_widths=[0.4, 0.6],
    )

    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
    ]

    bar_data = []
    for idx, (stats, label) in enumerate(zip(all_stats, labels)):
        wins = sum(1 for r in stats if r["win"])
        total = len(stats)
        wr = wins / total if total > 0 else 0
        avg_score = sum(r["score"] for r in stats) / total if total > 0 else 0
        bar_data.append(
            {
                "label": label[:20],
                "win_rate": wr,
                "total": total,
                "wins": wins,
                "avg_score": avg_score,
                "color": colors[idx % len(colors)],
            }
        )

    fig.add_trace(
        go.Bar(
            x=[d["label"] for d in bar_data],
            y=[d["win_rate"] * 100 for d in bar_data],
            marker_color=[d["color"] for d in bar_data],
            text=[
                f"{d['win_rate']:.0%}<br>({d['wins']}/{d['total']})" for d in bar_data
            ],
            textposition="auto",
            name="胜率",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="胜率 (%)", range=[0, 100], row=1, col=1)

    for idx, (stats, label) in enumerate(zip(all_stats, labels)):
        cum_wins = 0
        cum_wr = []
        for i, r in enumerate(stats):
            cum_wins += 1 if r["win"] else 0
            cum_wr.append(cum_wins / (i + 1) * 100)
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(stats) + 1)),
                y=cum_wr,
                mode="lines",
                name=label[:20],
                line=dict(color=colors[idx % len(colors)]),
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="累积胜率 (%)", range=[0, 100], row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=1, col=2)

    fig.update_layout(height=350, showlegend=True)
    return fig


def _plot_hp_curve(all_stats, labels):
    fig = go.Figure()

    colors = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
    ]

    for idx, (stats, label) in enumerate(zip(all_stats, labels)):
        color = colors[idx % len(colors)]
        all_curves = []
        for r in stats:
            curve = r["hp_diff_curve"]
            if len(curve) > 1:
                norm_x = [i / (len(curve) - 1) * 100 for i in range(len(curve))]
                all_curves.append((norm_x, curve, r["win"]))

        for norm_x, curve, is_win in all_curves:
            fig.add_trace(
                go.Scatter(
                    x=norm_x,
                    y=curve,
                    mode="lines",
                    line=dict(
                        color="#2ecc71" if is_win else "#e74c3c",
                        width=1,
                    ),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if all_curves:
            max_len = max(len(c[1]) for c in all_curves)
            mean_curve = []
            std_upper = []
            std_lower = []
            for i in range(max_len):
                vals = []
                for norm_x, curve, _ in all_curves:
                    idx_f = i / (max_len - 1) * (len(curve) - 1)
                    lo = int(idx_f)
                    hi = min(lo + 1, len(curve) - 1)
                    frac = idx_f - lo
                    interp = curve[lo] * (1 - frac) + curve[hi] * frac
                    vals.append(interp)
                m = sum(vals) / len(vals)
                std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                mean_curve.append(m)
                std_upper.append(m + std)
                std_lower.append(m - std)

            norm_x_mean = [i / (max_len - 1) * 100 for i in range(max_len)]
            fig.add_trace(
                go.Scatter(
                    x=norm_x_mean + norm_x_mean[::-1],
                    y=std_upper + std_lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=norm_x_mean,
                    y=mean_curve,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label[:20],
                )
            )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="HP 差值曲线（归一化进度）",
        xaxis_title="进度 (%)",
        yaxis_title="HP 差值 (我方 - 敌方)",
        height=400,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def _plot_action_distribution(all_stats, labels):
    all_sources = set()
    for stats in all_stats:
        for r in stats:
            all_sources.update(r["action_sources"].keys())
    source_list = sorted(all_sources)

    source_colors = {
        "kg_plan": "#2ecc71",
        "kg_follow": "#27ae60",
        "diverge": "#e67e22",
        "fallback": "#e74c3c",
        "replay": "#3498db",
        "replay_fallback": "#9b59b6",
        "backup_switch": "#1abc9c",
        "external": "#95a5a6",
        "unknown": "#bdc3c7",
    }

    fig = go.Figure()
    for src in source_list:
        pcts = []
        for stats in all_stats:
            total = (
                sum(
                    sum(r["action_sources"].get(s, 0) for s in source_list)
                    for r in stats
                )
                or 1
            )
            src_total = sum(r["action_sources"].get(src, 0) for r in stats)
            pcts.append(src_total / total * 100)
        fig.add_trace(
            go.Bar(
                name=src,
                x=labels,
                y=pcts,
                marker_color=source_colors.get(src, "#bdc3c7"),
                text=[f"{p:.1f}%" for p in pcts],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="动作来源分布",
        yaxis_title="占比 (%)",
        barmode="stack",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_results_tab():
    st.markdown("### 选择本地保存的实验结果数据，进行多维度可视化分析。")

    available = _scan_results()
    if not available:
        st.info(
            "暂无保存的结果数据。请在「实时对局」Tab 中运行对局后点击「保存结果」。"
        )
        st.caption(f"数据目录: `{RESULTS_DIR}`")
        return

    with st.sidebar:
        st.caption(f"数据目录: `{RESULTS_DIR}`")

    file_labels = [item["label"] for item in available]
    selected_labels = st.multiselect(
        "选择数据集（可多选对比）",
        options=file_labels,
        default=[file_labels[0]] if file_labels else [],
    )

    if not selected_labels:
        st.warning("请至少选择一个数据集。")
        return

    selected_paths = []
    for lbl in selected_labels:
        for item in available:
            if item["label"] == lbl:
                selected_paths.append(item["path"])
                break

    result_filter = st.radio(
        "结果过滤",
        options=["全部", "仅 Win", "仅 Loss"],
        index=0,
        horizontal=True,
    )

    all_stats = []
    display_labels = []
    for path, label in zip(selected_paths, selected_labels):
        data = _load_result_file(path)
        stats = _extract_stats(data)
        if result_filter == "仅 Win":
            stats = [r for r in stats if r["win"]]
        elif result_filter == "仅 Loss":
            stats = [r for r in stats if not r["win"]]
        all_stats.append(stats)
        parts = label.split("|")
        mode_part = parts[0].strip()
        core_parts = []
        if len(parts) > 1:
            for tag in parts[1].split():
                if (
                    tag.startswith("sm=")
                    or tag.startswith("bw=")
                    or tag.startswith("la=")
                    or tag.startswith("as=")
                ):
                    core_parts.append(tag)
        core = (mode_part + " | " + " ".join(core_parts)) if core_parts else mode_part
        display_labels.append(core)

    total_eps = sum(len(s) for s in all_stats)
    total_wins = sum(sum(1 for r in s if r["win"]) for s in all_stats)
    st.markdown(
        f"**{len(selected_labels)} 个数据集 | {total_eps} 局 | "
        f"胜率 {total_wins}/{total_eps} ({total_wins / max(total_eps, 1):.0%})**"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _plot_score_chart(all_stats, display_labels), use_container_width=True
        )
    with col2:
        st.plotly_chart(
            _plot_hp_curve(all_stats, display_labels), use_container_width=True
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            _plot_winrate_chart(all_stats, display_labels), use_container_width=True
        )
    with col4:
        st.plotly_chart(
            _plot_action_distribution(all_stats, display_labels),
            use_container_width=True,
        )

    st.divider()
    st.markdown("**📋 Episode 汇总表**")

    summary_rows = []
    for stats, label in zip(all_stats, display_labels):
        for r in stats:
            summary_rows.append(
                {
                    "数据集": label,
                    "Episode": r["episode_id"],
                    "结果": r["result"],
                    "得分": r["score"],
                    "帧数": r["num_frames"],
                    "主要来源": max(r["action_sources"], key=r["action_sources"].get)
                    if r["action_sources"]
                    else "-",
                }
            )

    if summary_rows:
        st.dataframe(
            summary_rows,
            use_container_width=True,
            height=min(len(summary_rows) * 35 + 50, 400),
        )
