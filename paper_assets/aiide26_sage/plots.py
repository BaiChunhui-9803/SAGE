"""Portable redraws from frozen measurements; camera-ready PDFs remain read-only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .evidence import DATA, GATE, PACK, SCENARIOS, iter_episodes, legacy, read_json

FIGURES = {"3": "fig3_parameter_correlation", "4": "fig4_backup_switching",
           "5a": "fig5a_score_distribution", "5b": "fig5b_decision_trace",
           "6": "fig6_state_quality", "7": "fig7_state_diagnostics"}


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def save(fig, out, name):
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / (name + ".png"), bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)


def figure03(out):
    module = legacy("figure03")
    module.CSV_PATH = DATA / "figure_03_parameter_correlation/parameter_numeric_correlations.csv"
    module.ALL_SCENES_DIR = out
    module.OUT_STEM = FIGURES["3"]
    module.main()


def figure04(out):
    module = legacy("figure04")
    module.DATA_CSV = DATA / "figure_04_backup_switch/fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    module.FIG_DIR = out
    module.OUT_STEM = FIGURES["4"]
    module.FIG_SIZE = (3.34, 2.2)
    # The compact camera-ready figure places its four legend entries vertically.
    module._save = lambda fig, stem: _save_switch(fig, out, stem)
    module.main()


def _save_switch(fig, out, stem):
    for legend in list(fig.legends): legend.remove()
    module = legacy("figure04")
    handles = [Patch(facecolor=module.CONFIG_COLOR[c], edgecolor=module.CONFIG_COLOR[c],
               alpha=module.BOX_ALPHA, label=module.CONFIG_LABEL[c]) for c in module.CONFIG_ORDER]
    fig.legend(handles=handles, ncol=1, loc="lower center", bbox_to_anchor=(0.53, -0.22),
               frameon=False, fontsize=6, handlelength=1.2, labelspacing=0.18)
    save(fig, out, stem)


def figure05a(out):
    data = pd.read_csv(GATE / "analysis_ready/boxplot_aligned_episode_table.csv")
    fig, ax = plt.subplots(figsize=(3.34, 1.62))
    values, positions = [], []
    for i, scene in enumerate(SCENARIOS):
        for j, method in enumerate(["ETG-only", "Synergy"]):
            values.append(data.loc[data.scenario.eq(scene) & data.method.eq(method), "score"].values)
            positions.append(i + [-0.16, 0.16][j])
    box = ax.boxplot(values, positions=positions, widths=0.25, patch_artist=True, showfliers=False,
        showmeans=True, meanprops={"marker": "D", "markersize": 2.7, "markerfacecolor": "white", "markeredgecolor": "#334155"},
        medianprops={"color": "#334155", "linewidth": .7}, whiskerprops={"color": "#475569", "linewidth": .7},
        capprops={"color": "#475569", "linewidth": .7})
    for i, patch in enumerate(box["boxes"]):
        patch.set(facecolor=["#ABC0D9", "#B7DEC0"][i % 2], edgecolor="#475569", linewidth=.6,
                  hatch="///" if i % 2 else "")
    ax.set_xticks(range(6), [s.replace("-", "") for s in SCENARIOS], fontsize=6.5)
    ax.set_xlabel("Scenario", fontsize=7); ax.set_ylabel("Episode score", fontsize=7)
    ax.tick_params(axis="y", labelsize=7); ax.grid(axis="y", alpha=.2, linewidth=.4)
    ax.legend(handles=[Patch(facecolor="#ABC0D9", edgecolor="#475569", label="Graph-only SAGE"),
        Patch(facecolor="#B7DEC0", edgecolor="#475569", hatch="///", label="Full SAGE")],
        loc="lower right", ncol=2, fontsize=6.2, fancybox=False, borderpad=.25, columnspacing=.7)
    save(fig, out, FIGURES["5a"])


def _trace(module, case, side):
    run = case[side + "_experiment_id"]; episode_id = int(case[side + "_episode_id"])
    source = PACK / "raw/evaluations" / run / "episodes.jsonl.gz"
    for _, episode in iter_episodes(source):
        if int(episode["episode_id"]) != episode_id: continue
        frames = episode["frames"]
        method = "etg-only" if side == "left" else "Synergy"
        return module.EpisodeTrace(uid=case[side + "_uid"], method_group="ETG-only" if side == "left" else "synergy",
            method=method, experiment_id=run, map_key=case["map_key"], episode_id=episode_id,
            score=float(episode["score"]), result=episode["result"], frames=frames,
            state_keys=[str(f.get("state_key", "")) for f in frames],
            actions=[str(f.get("action_code") or f.get("shadow_selected_action_code") or "") for f in frames],
            changed=[module.truthy(f.get("shadow_mechanism_changed_action")) for f in frames],
            hp_delta=[float(f.get("hp_delta") or 0) for f in frames],
            hp_my=[float(f.get("hp_my") or 0) for f in frames],
            hp_enemy=[float(f.get("hp_enemy") or 0) for f in frames],
            states=[module.frame_to_project_state(f) for f in frames])
    raise ValueError(f"Missing branch episode: {run}/{episode_id}")


def figure05b(out):
    module = legacy("branch_analysis")
    case = read_json(GATE / "branch_case/sage_gate_branch_divergence_no_score_source.json")
    left, right = _trace(module, case, "left"), _trace(module, case, "right")
    if left.score != case["left_score"] or right.score != case["right_score"]:
        raise ValueError("Branch episode scores differ from frozen case")
    module.setup_style()
    module.plot_case_graph(left, right, pd.Series(case), out / FIGURES["5b"], horizon=10, paper_labels=True)


def figure06(out):
    module = legacy("gate_analysis")
    states = pd.read_csv(GATE / "analysis_ready/boxplot_aligned_state_table.csv")
    module.setup_style()
    module.save = lambda fig, directory, stem: save(fig, directory, FIGURES["6"])
    module.draw_hp_margin_progress_with_score(states, out, min_frames=1)


def figure07(out):
    """Redraw frozen MDS coordinates and state-transition records, without refitting MDS."""
    upstream = DATA / "figure_07_state_diagnostics/upstream_sce2"
    points = pd.read_csv(upstream / "bktree_sequence_landscape/sampled_episode_points.csv")
    nodes = pd.read_csv(upstream / "bktree_transition_graph/state_transition_nodes.csv")
    edges = pd.read_csv(upstream / "bktree_transition_graph/state_transition_edges.csv")
    colors = {"ETG-only": "#8DB7E8", "Synergy": "#1F5A9D", "QMIX": "#F28E2B", "QTRAN": "#9467BD",
              "VDN": "#D62728", "MAPPO": "#17BECF", "RMAPPO": "#8C564B", "MAT": "#E377C2"}
    fig, (ax, graph) = plt.subplots(1, 2, figsize=(6.9, 3.05), gridspec_kw={"wspace": .14})
    x, y, score = (points[c].to_numpy() for c in ["mds_x", "mds_y", "score"])
    px, py = np.ptp(x) * .05, np.ptp(y) * .05
    gx, gy = np.meshgrid(np.linspace(x.min()-px, x.max()+px, 160), np.linspace(y.min()-py, y.max()+py, 160))
    gz = griddata((x, y), score, (gx, gy), method="linear")
    gz = np.where(np.isfinite(gz), gz, griddata((x, y), score, (gx, gy), method="nearest"))
    gz = np.clip(gz, score.min(), score.max())
    lo, hi = min(float(gz.min()), 0), max(float(gz.max()), 0)
    cmap = LinearSegmentedColormap.from_list("score", [(0, "#5c7ee6"), (-lo/(hi-lo), "#ebebeb"), (1, "#b62d0a")])
    cs = ax.contourf(gx, gy, gz, levels=np.linspace(lo, hi, 25), cmap=cmap, norm=Normalize(lo, hi), alpha=.68)
    ax.contour(gx, gy, gz, levels=np.linspace(lo, hi, 13), colors="#5b6b7a", linewidths=.15, alpha=.35)
    for method, group in points.groupby("method"):
        ax.scatter(group.mds_x, group.mds_y, s=5.3, c=colors[method], edgecolors="white", linewidths=.1, alpha=.88)
    bar = fig.colorbar(cs, ax=ax, fraction=.035, pad=.045); bar.set_ticks([])
    ax.set_xticks([]); ax.set_yticks([])
    coords = nodes.set_index("state_id")[["x", "y"]]
    for row in edges.itertuples():
        pair = coords.loc[[row.source, row.target]]
        ratio = row.transition_count / edges.transition_count.max()
        graph.plot(pair.x, pair.y, c="#2f6da8" if row.category == "calp" else "#666666",
                   alpha=.15+.4*ratio, linewidth=.12+.45*np.sqrt(ratio), zorder=1)
    sizes = 3 + 13 * np.sqrt(nodes.visits.to_numpy() / nodes.visits.max())
    for category, group in nodes.groupby("category", sort=False):
        sage = category.startswith("calp")
        graph.scatter(group.x, group.y, s=sizes[group.index], c=group.color, marker=group.marker.iloc[0],
                      edgecolors="black" if sage else "none", linewidths=.25 if sage else 0, zorder=3)
    initial = nodes.loc[nodes.state_id.eq(0)].iloc[0]
    graph.annotate("Unified initial state", xy=(initial.x, initial.y), xycoords="data", xytext=(.12,.94),
                   textcoords="axes fraction", fontsize=8, color="#006b53",
                   arrowprops={"arrowstyle":"->", "lw":.5, "connectionstyle":"arc3,rad=-.45"})
    graph.axis("off")
    handles = [Line2D([], [], linestyle="", marker="o", color=color, markersize=5,
                     label={"ETG-only":"Graph-only SAGE", "Synergy":"Full SAGE"}.get(m,m)) for m,color in colors.items()]
    fig.legend(handles=handles, ncol=4, loc="upper center", fontsize=7.3, frameon=False,
               columnspacing=.9, handletextpad=.3)
    fig.subplots_adjust(top=.78, bottom=.13, left=.025, right=.995)
    fig.text(.245,.03,"(a) Episode sequence landscape", ha="center", fontsize=8)
    fig.text(.765,.03,"(b) Abstract state-transition graph", ha="center", fontsize=8)
    save(fig, out, FIGURES["7"])


def reproduce(selection, output):
    out = output / "figures"; out.mkdir(parents=True, exist_ok=True)
    functions = {"3": figure03, "4": figure04, "5a": figure05a, "5b": figure05b, "6": figure06, "7": figure07}
    for key in functions if selection == "all" else [selection]:
        style(); functions[key](out)
    (output / "figure_scope.json").write_text(json.dumps({
        "source": "Frozen measurements and original plotting routines, with portable paths and compact layout wrappers",
        "reference": "paper_assets/aiide26_sage/reference contains byte-identical camera-ready figures",
        "redraws": "Numeric inputs are preserved; PDF layout/font bytes may differ from the final manually adjusted paper figures",
        "figure7": "Uses archived MDS coordinates and node/edge tables; does not rerun full upstream fusion and MDS fitting",
    }, indent=2) + "\n", encoding="utf-8")
