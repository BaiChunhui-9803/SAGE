from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D


PROJECT = Path(r"C:\Users\Binich\PycharmProjects\PredictionRTS")
ANALYSIS = (
    PROJECT
    / "output"
    / "learner_results"
    / "all_data"
    / "_batch_final_eval_logs"
    / "batch_fixed_pool_switch_pair_web_fixed_pool_switch_pair_20260625_093920"
    / "switch_grid_analysis"
)
BATCH = ANALYSIS.parent
OUT = PROJECT / "output" / "figures" / "fixed_pool_switch_pair_full_v2_20260625"
FIG = OUT / "figures"
TABLE = OUT / "tables"

SCENE_ORDER = ["sce-1", "sce-1m", "sce-2", "sce-2m", "sce-3", "sce-3m"]
SCENE_LABEL = {"sce-1": "sce1", "sce-1m": "sce1m", "sce-2": "sce2", "sce-2m": "sce2m", "sce-3": "sce3", "sce-3m": "sce3m"}
LABEL_TO_SCENE = {value: key for key, value in SCENE_LABEL.items()}
CONFIG_ORDER = ["ms_ns", "ms_ex", "ms_fz020", "ms_fz050"]
CONFIG_LABEL = {"ms_ns": "No switch", "ms_ex": "Exact", "ms_fz020": "Fuzzy 0.20", "ms_fz050": "Fuzzy 0.50"}
CONFIG_COLOR = {"ms_ns": "#9CA3AF", "ms_ex": "#2563EB", "ms_fz020": "#16A34A", "ms_fz050": "#F97316"}
BACKUP_TO_CONFIG = {
    "fixed_pool_multi_step_no_switch": "ms_ns",
    "fixed_pool_multi_step_switch_exact": "ms_ex",
    "fixed_pool_multi_step_switch_fuzzy020": "ms_fz020",
    "fixed_pool_multi_step_switch_fuzzy050": "ms_fz050",
}
SWITCH_CONFIGS = ["ms_ex", "ms_fz020", "ms_fz050"]
SELECTED_GROUPS = {"sce1": 1, "sce1m": 1, "sce2": 8, "sce2m": 6, "sce3": 4, "sce3m": 2}


plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.45,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def scene_sort_key(values: pd.Series) -> pd.Series:
    order = {scene: idx for idx, scene in enumerate(SCENE_ORDER)}
    return values.map(lambda value: order.get(value, 99))


def load_override_params() -> dict[tuple[str, str], dict]:
    params: dict[tuple[str, str], dict] = {}
    override_dir = BATCH / "variant_overrides"
    for file_path in override_dir.glob("*_override.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "fixed_pool_base_rank" not in data or "fixed_pool_base_trial" not in data:
            continue
        name = file_path.stem.replace("_override", "")
        experiment_id = name.split("_rank")[0]
        rank = int(data["fixed_pool_base_rank"])
        trial = int(data["fixed_pool_base_trial"])
        base_variant_id = f"rank{rank:02d}_trial{trial:04d}"
        params[(experiment_id, base_variant_id)] = data
    return params


def build_matrix(comparison: pd.DataFrame) -> pd.DataFrame:
    override_params = load_override_params()
    rows = []
    for (experiment_id, base_variant_id), group in comparison.groupby(["experiment_id", "base_variant_id"], sort=False):
        base = group[group["backup_condition"] == "fixed_pool_multi_step_no_switch"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        row = {
            "experiment_id": experiment_id,
            "map_key": base_row["map_key"],
            "scene": SCENE_LABEL.get(base_row["map_key"], base_row["map_key"]),
            "base_variant_id": base_variant_id,
            "rank": int(base_row["base_variant_rank"]),
            "trial": int(base_row["base_variant_trial"]),
            "ms_ns": 0.0,
            "ms_ns_switch_rate": 0.0,
            "base_score": float(base_row["score_mean"]),
        }
        params = override_params.get((experiment_id, base_variant_id), {})
        for key in ["beam_width", "lookahead_steps", "min_visits", "score_mode", "action_strategy", "discount_factor", "min_cum_prob"]:
            row[key] = params.get(key, np.nan if key not in ["score_mode", "action_strategy"] else "")
        for backup_condition, config in BACKUP_TO_CONFIG.items():
            cond_row = group[group["backup_condition"] == backup_condition]
            if cond_row.empty:
                row[config] = np.nan
                row[f"{config}_switch_rate"] = np.nan
                row[f"{config}_score"] = np.nan
            else:
                cond_row = cond_row.iloc[0]
                row[config] = float(cond_row["score_delta_vs_multi_step_no_switch"])
                row[f"{config}_switch_rate"] = float(cond_row["switch_rate_per_divergence"])
                row[f"{config}_score"] = float(cond_row["score_mean"])
        deltas = {config: row[config] for config in SWITCH_CONFIGS if pd.notna(row.get(config))}
        row["best_cond"] = max(deltas, key=deltas.get)
        row["best_delta"] = deltas[row["best_cond"]]
        row["worst_delta"] = min(deltas.values())
        row["any_positive"] = any(value > 0 for value in deltas.values())
        row["all_positive"] = all(value > 0 for value in deltas.values())
        row["all_negative"] = all(value < 0 for value in deltas.values())
        rows.append(row)
    matrix = pd.DataFrame(rows)
    matrix["_order"] = scene_sort_key(matrix["map_key"])
    return matrix.sort_values(["_order", "rank"]).drop(columns="_order")


def build_effect(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for map_key, sub in matrix.groupby("map_key", sort=False):
        for config in CONFIG_ORDER:
            values = sub[config].dropna()
            rows.append(
                {
                    "map_key": map_key,
                    "scene": SCENE_LABEL.get(map_key, map_key),
                    "config": config,
                    "mean_delta": float(values.mean()) if len(values) else np.nan,
                    "median_delta": float(values.median()) if len(values) else np.nan,
                    "positive_rate": float((values > 0).mean()) if len(values) else np.nan,
                    "n": int(len(values)),
                }
            )
    effect = pd.DataFrame(rows)
    effect["_order"] = scene_sort_key(effect["map_key"])
    effect["_config"] = effect["config"].map({config: idx for idx, config in enumerate(CONFIG_ORDER)})
    return effect.sort_values(["_order", "_config"]).drop(columns=["_order", "_config"])


def plot_heatmap_with_baseline(matrix: pd.DataFrame) -> None:
    scenes = [scene for scene in SCENE_ORDER if scene in set(matrix["map_key"])]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.55), constrained_layout=True)
    all_values = matrix[CONFIG_ORDER].to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanpercentile(np.abs(all_values), 95)))
    vmax = min(vmax, 12.0)
    image = None
    for ax, scene in zip(axes.ravel(), scenes):
        sub = matrix[matrix["map_key"] == scene].sort_values("rank")
        values = sub[CONFIG_ORDER].to_numpy(dtype=float)
        image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(CONFIG_ORDER)), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=22, ha="right")
        ax.set_yticks(range(len(sub)), [f"R{int(rank)}" for rank in sub["rank"]])
        ax.set_xlabel("Switch configuration")
        ax.set_ylabel("Parameter group")
        ax.text(
            0.02,
            0.98,
            SCENE_LABEL[scene],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.4),
        )
        for row_idx, (_, row) in enumerate(sub.iterrows()):
            best_col = CONFIG_ORDER.index(row["best_cond"])
            for col_idx, config in enumerate(CONFIG_ORDER):
                value = row[config]
                color = "white" if abs(value) > vmax * 0.45 else "#111827"
                ax.text(col_idx, row_idx, f"{value:+.1f}", ha="center", va="center", fontsize=7.1, color=color)
            ax.add_patch(Rectangle((best_col - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="#111827", linewidth=1.65))
            ax.add_patch(Rectangle((-0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="#6B7280", linewidth=0.75, linestyle=":"))
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.01)
        colorbar.set_label("Score delta vs no-switch")
    save(fig, "fig1_param_group_config_delta_heatmap_with_no_switch")


def plot_scene_config_summary_with_baseline(effect: pd.DataFrame) -> None:
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(effect["map_key"])]
    x = np.arange(len(scenes))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.45), constrained_layout=True)
    for idx, config in enumerate(CONFIG_ORDER):
        sub = effect[effect["config"] == config]
        values = [float(sub[sub["scene"] == scene]["mean_delta"].iloc[0]) for scene in scenes]
        rates = [float(sub[sub["scene"] == scene]["positive_rate"].iloc[0]) * 100 for scene in scenes]
        offset = (idx - 1.5) * width
        axes[0].bar(x + offset, values, width=width, color=CONFIG_COLOR[config], alpha=0.86, label=CONFIG_LABEL[config])
        axes[1].bar(x + offset, rates, width=width, color=CONFIG_COLOR[config], alpha=0.86, label=CONFIG_LABEL[config])
    axes[0].axhline(0, color="#111827", lw=0.8, ls="--")
    axes[0].set_ylabel("Mean score delta")
    axes[0].set_xlabel("Scenario")
    axes[0].set_xticks(x, scenes)
    axes[1].set_ylabel("Positive parameter groups (%)")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x, scenes)
    axes[0].legend(frameon=False, ncol=4, loc="upper left")
    save(fig, "fig2_scene_config_mean_delta_positive_rate_with_no_switch")


def plot_delta_distribution_with_baseline(matrix: pd.DataFrame) -> None:
    rows = []
    for _, row in matrix.iterrows():
        for config in CONFIG_ORDER:
            rows.append({"scene": row["scene"], "map_key": row["map_key"], "rank": row["rank"], "config": config, "delta": row[config]})
    long = pd.DataFrame(rows)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(long["map_key"])]
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 3.85), constrained_layout=True)
    values = []
    positions = []
    colors = []
    base_positions = np.arange(len(scenes)) * 4.7
    offsets = [-0.84, -0.28, 0.28, 0.84]
    for scene_idx, scene in enumerate(scenes):
        for config_idx, config in enumerate(CONFIG_ORDER):
            values.append(long[(long["scene"] == scene) & (long["config"] == config)]["delta"].to_numpy())
            positions.append(base_positions[scene_idx] + offsets[config_idx])
            colors.append(CONFIG_COLOR[config])
    box = ax.boxplot(values, positions=positions, widths=0.42, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.42)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.15)
    for part in box["whiskers"] + box["caps"]:
        part.set_color("#6B7280")
    ax.axhline(0, color="#111827", lw=0.8, ls="--")
    ax.set_xticks(base_positions, scenes)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Score delta vs no-switch")
    ax.legend(
        handles=[Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], alpha=0.42, label=CONFIG_LABEL[c]) for c in CONFIG_ORDER],
        frameon=False,
        ncol=4,
        loc="upper left",
    )
    save(fig, "fig4_switch_delta_distribution_with_no_switch")


def plot_filtered_score_distribution_with_baseline(matrix: pd.DataFrame, episode: pd.DataFrame) -> None:
    filtered = matrix[~matrix["all_negative"]].copy()
    filtered.to_csv(TABLE / "filtered_non_all_negative_param_groups_for_fig4.csv", index=False, encoding="utf-8-sig")

    rows = []
    for _, row in filtered.iterrows():
        rows.extend(
            [
                {"scene": row["scene"], "map_key": row["map_key"], "rank": row["rank"], "config": "ms_ns", "score": row["base_score"]},
                {"scene": row["scene"], "map_key": row["map_key"], "rank": row["rank"], "config": "ms_ex", "score": row["ms_ex_score"]},
                {"scene": row["scene"], "map_key": row["map_key"], "rank": row["rank"], "config": "ms_fz020", "score": row["ms_fz020_score"]},
                {"scene": row["scene"], "map_key": row["map_key"], "rank": row["rank"], "config": "ms_fz050", "score": row["ms_fz050_score"]},
            ]
        )
    long = pd.DataFrame(rows)
    long.to_csv(TABLE / "fig4_filtered_param_group_mean_scores.csv", index=False, encoding="utf-8-sig")

    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(long["map_key"])]
    base_positions = np.arange(len(scenes)) * 4.7
    offsets = [-0.84, -0.28, 0.28, 0.84]

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 3.9), constrained_layout=True)
    values, positions, colors = [], [], []
    for scene_idx, scene in enumerate(scenes):
        for config_idx, config in enumerate(CONFIG_ORDER):
            values.append(long[(long["scene"] == scene) & (long["config"] == config)]["score"].to_numpy())
            positions.append(base_positions[scene_idx] + offsets[config_idx])
            colors.append(CONFIG_COLOR[config])
    box = ax.boxplot(values, positions=positions, widths=0.42, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.42)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.15)
    for part in box["whiskers"] + box["caps"]:
        part.set_color("#6B7280")
    ax.set_xticks(base_positions, scenes)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean score across retained parameter groups")
    ax.legend(
        handles=[Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], alpha=0.42, label=CONFIG_LABEL[c]) for c in CONFIG_ORDER],
        frameon=False,
        ncol=4,
        loc="upper left",
    )
    save(fig, "fig4_filtered_score_distribution_with_no_switch")

    keep_keys = set(zip(filtered["map_key"], filtered["rank"]))
    episode_view = episode.copy()
    episode_view["rank"] = episode_view["base_variant_rank"].astype(int)
    episode_view["config"] = episode_view["backup_condition"].map(BACKUP_TO_CONFIG)
    episode_view["scene"] = episode_view["map_key"].map(SCENE_LABEL)
    episode_view = episode_view[episode_view.apply(lambda row: (row["map_key"], row["rank"]) in keep_keys, axis=1)].copy()
    episode_view.to_csv(TABLE / "fig4_filtered_episode_scores.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 3.9), constrained_layout=True)
    values, positions, colors = [], [], []
    for scene_idx, scene in enumerate(scenes):
        for config_idx, config in enumerate(CONFIG_ORDER):
            values.append(episode_view[(episode_view["scene"] == scene) & (episode_view["config"] == config)]["score"].to_numpy())
            positions.append(base_positions[scene_idx] + offsets[config_idx])
            colors.append(CONFIG_COLOR[config])
    box = ax.boxplot(values, positions=positions, widths=0.42, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.15)
    for part in box["whiskers"] + box["caps"]:
        part.set_color("#6B7280")
    ax.set_xticks(base_positions, scenes)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Episode score in retained parameter groups")
    ax.legend(
        handles=[Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], alpha=0.35, label=CONFIG_LABEL[c]) for c in CONFIG_ORDER],
        frameon=False,
        ncol=4,
        loc="upper left",
    )
    save(fig, "fig4b_filtered_episode_score_distribution_with_no_switch")


def plot_filtered_score_distribution_independent_axes() -> None:
    long_path = TABLE / "fig4_filtered_param_group_mean_scores.csv"
    if not long_path.exists():
        return
    long = pd.read_csv(long_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(long["map_key"])]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.2), constrained_layout=True)
    for ax, scene in zip(axes.ravel(), scenes):
        sub = long[long["scene"] == scene]
        values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
        box = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False)
        for patch, config in zip(box["boxes"], CONFIG_ORDER):
            patch.set_facecolor(CONFIG_COLOR[config])
            patch.set_edgecolor(CONFIG_COLOR[config])
            patch.set_alpha(0.42)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.15)
        for part in box["whiskers"] + box["caps"]:
            part.set_color("#6B7280")
        ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=22, ha="right")
        ax.set_ylabel("Mean score")
        ax.text(
            0.02,
            0.98,
            scene,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.2),
        )
        all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
        low, high = float(np.min(all_values)), float(np.max(all_values))
        pad = max(1.0, (high - low) * 0.18)
        ax.set_ylim(low - pad, high + pad)
    handles = [Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], alpha=0.42, label=CONFIG_LABEL[c]) for c in CONFIG_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    save(fig, "fig4c_filtered_score_distribution_independent_axes")


def plot_filtered_score_distribution_independent_axes_trimmed() -> None:
    long_path = TABLE / "fig4_filtered_param_group_mean_scores.csv"
    if not long_path.exists():
        return
    long = pd.read_csv(long_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(long["map_key"])]
    kept_parts = []
    for scene in scenes:
        sub = long[long["scene"] == scene].copy()
        pivot = sub.pivot_table(index=["map_key", "rank"], columns="config", values="score", aggfunc="mean").reset_index()
        center = pivot[CONFIG_ORDER].mean(axis=1)
        q1, q3 = center.quantile(0.25), center.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 1e-9:
            keep = pivot
        else:
            keep = pivot[(center >= q1 - 1.5 * iqr) & (center <= q3 + 1.5 * iqr)]
        keep_keys = set(zip(keep["map_key"], keep["rank"]))
        kept_parts.append(sub[sub.apply(lambda row: (row["map_key"], row["rank"]) in keep_keys, axis=1)])
    trimmed = pd.concat(kept_parts, ignore_index=True)
    trimmed.to_csv(TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.2), constrained_layout=True)
    for ax, scene in zip(axes.ravel(), scenes):
        sub = trimmed[trimmed["scene"] == scene]
        values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
        box = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False)
        for patch, config in zip(box["boxes"], CONFIG_ORDER):
            patch.set_facecolor(CONFIG_COLOR[config])
            patch.set_edgecolor(CONFIG_COLOR[config])
            patch.set_alpha(0.42)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.15)
        for part in box["whiskers"] + box["caps"]:
            part.set_color("#6B7280")
        ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=22, ha="right")
        ax.set_ylabel("Mean score")
        n_groups = int(sub[["map_key", "rank"]].drop_duplicates().shape[0])
        ax.text(
            0.02,
            0.98,
            f"{scene} (n={n_groups})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.2),
        )
        all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
        low, high = float(np.min(all_values)), float(np.max(all_values))
        pad = max(0.8, (high - low) * 0.18)
        ax.set_ylim(low - pad, high + pad)
    handles = [Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], alpha=0.42, label=CONFIG_LABEL[c]) for c in CONFIG_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    save(fig, "fig4d_filtered_score_distribution_independent_axes_iqr_trimmed")


def plot_filtered_score_distribution_independent_axes_trimmed_row() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    fig, axes = plt.subplots(1, 6, figsize=(18.0, 3.25), constrained_layout=True)
    for ax, scene in zip(axes.ravel(), scenes):
        sub = trimmed[trimmed["scene"] == scene]
        values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
        box = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False)
        for patch, config in zip(box["boxes"], CONFIG_ORDER):
            patch.set_facecolor(CONFIG_COLOR[config])
            patch.set_edgecolor(CONFIG_COLOR[config])
            patch.set_alpha(0.42)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.15)
        for part in box["whiskers"] + box["caps"]:
            part.set_color("#6B7280")
        ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=28, ha="right")
        n_groups = int(sub[["map_key", "rank"]].drop_duplicates().shape[0])
        ax.text(
            0.02,
            0.98,
            f"{scene} (n={n_groups})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.1),
        )
        all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
        low, high = float(np.min(all_values)), float(np.max(all_values))
        pad = max(0.8, (high - low) * 0.18)
        ax.set_ylim(low - pad, high + pad)
        ax.set_ylabel("Mean score")
    handles = [Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], alpha=0.42, label=CONFIG_LABEL[c]) for c in CONFIG_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig4e_filtered_score_distribution_independent_axes_iqr_trimmed_row")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(19.5, 3.55), constrained_layout=True)
        for ax, scene in zip(axes.ravel(), scenes):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, widths=0.42, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.65)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.8)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.45)
            ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=30, ha="right")
            ax.set_title(scene, fontweight="bold", pad=4)
            ax.set_ylabel("Mean score")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.65, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        save(fig, "fig4f_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper_compact() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    compact_positions = [1.00, 1.52, 2.04, 2.56]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(18.0, 3.55), constrained_layout=True)
        for ax, scene in zip(axes.ravel(), scenes):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, positions=compact_positions, widths=0.28, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.75)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.9)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.55)
            ax.set_xticks(compact_positions, [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=30, ha="right")
            ax.set_xlim(0.72, 2.84)
            ax.set_title(scene, fontweight="bold", pad=4)
            ax.set_ylabel("Mean score")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.75, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        save(fig, "fig4g_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper_compact")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper_narrow() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(15.2, 3.35), constrained_layout=True)
        for ax, scene in zip(axes.ravel(), scenes):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, widths=0.24, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.55)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.75)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.35)
            ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=32, ha="right")
            ax.set_title(scene, fontweight="bold", pad=4)
            ax.set_ylabel("Mean score")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.55, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        save(fig, "fig4h_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper_narrow")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper_narrow_widebox() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(15.2, 3.35), constrained_layout=True)
        for ax, scene in zip(axes.ravel(), scenes):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, widths=0.36, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.55)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.75)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.35)
            ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=32, ha="right")
            ax.set_title(scene, fontweight="bold", pad=4)
            ax.set_ylabel("Mean score")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.55, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        save(fig, "fig4i_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper_narrow_boxw036")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper_narrow_widebox_close() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    positions = [1.00, 1.78, 2.56, 3.34]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(15.2, 3.35), constrained_layout=True)
        for ax, scene in zip(axes.ravel(), scenes):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, positions=positions, widths=0.36, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.55)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.75)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.35)
            ax.set_xticks(positions, [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=32, ha="right")
            ax.set_xlim(0.68, 3.66)
            ax.set_title(scene, fontweight="bold", pad=4)
            ax.set_ylabel("Mean score")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.55, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        save(fig, "fig4j_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper_narrow_boxw036_close")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper_tight_axes() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    positions = [1.00, 1.78, 2.56, 3.34]
    short_labels = ["No", "Exact", "Fz.20", "Fz.50"]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(13.2, 3.45), constrained_layout=False)
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.24, top=0.78, wspace=0.22)
        for axis_index, (ax, scene) in enumerate(zip(axes.ravel(), scenes)):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, positions=positions, widths=0.36, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.55)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.75)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.35)
            ax.set_xticks(positions, short_labels, rotation=25, ha="right")
            ax.set_xlim(0.68, 3.66)
            ax.set_title(scene, fontweight="bold", pad=4)
            if axis_index == 0:
                ax.set_ylabel("Mean score")
            else:
                ax.set_ylabel("")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.55, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995))
        save(fig, "fig4k_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper_tight_axes")


def plot_filtered_score_distribution_independent_axes_trimmed_row_paper_tight_axes_full_labels() -> None:
    trimmed_path = TABLE / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
    if not trimmed_path.exists():
        return
    trimmed = pd.read_csv(trimmed_path)
    scenes = [SCENE_LABEL[s] for s in SCENE_ORDER if s in set(trimmed["map_key"])]
    positions = [1.00, 1.70, 2.40, 3.10]
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.linewidth": 0.55,
        }
    ):
        fig, axes = plt.subplots(1, 6, figsize=(13.7, 3.55), constrained_layout=False)
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.28, top=0.78, wspace=0.18)
        for axis_index, (ax, scene) in enumerate(zip(axes.ravel(), scenes)):
            sub = trimmed[trimmed["scene"] == scene]
            values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
            box = ax.boxplot(values, positions=positions, widths=0.32, patch_artist=True, showfliers=False)
            for patch, config in zip(box["boxes"], CONFIG_ORDER):
                patch.set_facecolor(CONFIG_COLOR[config])
                patch.set_edgecolor(CONFIG_COLOR[config])
                patch.set_alpha(0.40)
                patch.set_linewidth(1.55)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.75)
            for part in box["whiskers"] + box["caps"]:
                part.set_color("#4B5563")
                part.set_linewidth(1.35)
            ax.set_xticks(positions, [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=34, ha="right")
            ax.set_xlim(0.72, 3.38)
            ax.set_title(scene, fontweight="bold", pad=4)
            ax.set_ylabel("Mean score" if axis_index == 0 else "")
            all_values = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([0.0])
            low, high = float(np.min(all_values)), float(np.max(all_values))
            pad = max(0.8, (high - low) * 0.18)
            ax.set_ylim(low - pad, high + pad)
        handles = [
            Patch(facecolor=CONFIG_COLOR[c], edgecolor=CONFIG_COLOR[c], linewidth=1.55, alpha=0.40, label=CONFIG_LABEL[c])
            for c in CONFIG_ORDER
        ]
        fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995))
        save(fig, "fig4l_filtered_score_distribution_independent_axes_iqr_trimmed_row_paper_tight_axes_full_labels")


def selected_group_rows(episode: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_records = []
    selected_episode_parts = []
    for scene, rank in SELECTED_GROUPS.items():
        map_key = LABEL_TO_SCENE[scene]
        matrix_row = matrix[(matrix["map_key"] == map_key) & (matrix["rank"] == rank)]
        if matrix_row.empty:
            continue
        matrix_row = matrix_row.iloc[0]
        selected_records.append(matrix_row)
        episode_part = episode[(episode["map_key"] == map_key) & (episode["base_variant_rank"] == rank)].copy()
        episode_part["scene"] = scene
        episode_part["config"] = episode_part["backup_condition"].map(BACKUP_TO_CONFIG)
        episode_part["config_label"] = episode_part["config"].map(CONFIG_LABEL)
        selected_episode_parts.append(episode_part)
    selected_matrix = pd.DataFrame(selected_records)
    selected_episode = pd.concat(selected_episode_parts, ignore_index=True)
    return selected_matrix, selected_episode


def plot_selected_groups_box_and_bar(episode: pd.DataFrame, matrix: pd.DataFrame) -> None:
    selected_matrix, selected_episode = selected_group_rows(episode, matrix)
    selected_matrix.to_csv(TABLE / "selected_best_param_group_delta_summary.csv", index=False, encoding="utf-8-sig")
    selected_episode.to_csv(TABLE / "selected_best_param_group_episode_scores.csv", index=False, encoding="utf-8-sig")

    scenes = list(SELECTED_GROUPS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.2), sharey=False, constrained_layout=True)
    for ax, scene in zip(axes.ravel(), scenes):
        sub = selected_episode[selected_episode["scene"] == scene]
        values = [sub[sub["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]
        box = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.58)
        for patch, config in zip(box["boxes"], CONFIG_ORDER):
            patch.set_facecolor(CONFIG_COLOR[config])
            patch.set_edgecolor(CONFIG_COLOR[config])
            patch.set_alpha(0.45)
        for median in box["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.15)
        ax.set_xticks(range(1, len(CONFIG_ORDER) + 1), [CONFIG_LABEL[c] for c in CONFIG_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Episode score")
        ax.text(
            0.02,
            0.98,
            f"{scene} R{SELECTED_GROUPS[scene]}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.3),
        )
        means = [float(np.mean(v)) if len(v) else np.nan for v in values]
        for idx, mean in enumerate(means, start=1):
            ax.scatter(idx, mean, marker="D", s=24, color="#111827", zorder=3)
    save(fig, "fig7_selected_param_group_episode_score_boxplots")

    rows = []
    for scene in scenes:
        sub = selected_episode[selected_episode["scene"] == scene]
        for config in CONFIG_ORDER:
            scores = sub[sub["config"] == config]["score"].to_numpy()
            rows.append({"scene": scene, "config": config, "mean": float(scores.mean()), "std": float(scores.std(ddof=1))})
    stat = pd.DataFrame(rows)
    stat.to_csv(TABLE / "selected_best_param_group_score_bar_stats.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(1, 1, figsize=(12.0, 3.8), constrained_layout=True)
    x = np.arange(len(scenes))
    width = 0.18
    for idx, config in enumerate(CONFIG_ORDER):
        sub = stat[stat["config"] == config]
        means = [sub[sub["scene"] == scene]["mean"].iloc[0] for scene in scenes]
        stds = [sub[sub["scene"] == scene]["std"].iloc[0] for scene in scenes]
        ax.bar(x + (idx - 1.5) * width, means, width=width, yerr=stds, capsize=2.5, color=CONFIG_COLOR[config], alpha=0.86, label=CONFIG_LABEL[config])
    ax.set_xticks(x, [f"{scene}\\nR{SELECTED_GROUPS[scene]}" for scene in scenes])
    ax.set_xlabel("Selected scenario-parameter group")
    ax.set_ylabel("Episode score")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    save(fig, "fig8_selected_param_group_score_mean_bar")


def plot_parameter_diversity_sce2m(matrix: pd.DataFrame) -> None:
    scene = "sce2m"
    sub = matrix[matrix["scene"] == scene].sort_values("rank").copy()
    sub.to_csv(TABLE / "sce2m_parameter_group_diversity_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)
    x = np.arange(len(sub))
    axes[0].bar(x - 0.18, sub["base_score"], width=0.36, color="#9CA3AF", alpha=0.80, label="No switch score")
    axes[0].bar(x + 0.18, sub["base_score"] + sub["best_delta"], width=0.36, color=[CONFIG_COLOR[c] for c in sub["best_cond"]], alpha=0.86, label="Best switch score")
    axes[0].set_xticks(x, [f"R{int(rank)}" for rank in sub["rank"]])
    axes[0].set_ylabel("Mean score")
    axes[0].set_xlabel("sce2m parameter group")
    axes[0].legend(frameon=False, loc="upper left")

    parameter_cols = ["beam_width", "lookahead_steps", "min_visits", "discount_factor", "min_cum_prob"]
    norm = sub[parameter_cols].astype(float).copy()
    for col in parameter_cols:
        low, high = norm[col].min(), norm[col].max()
        norm[col] = 0.5 if high == low else (norm[col] - low) / (high - low)
    for _, row in norm.iterrows():
        original = sub.loc[row.name]
        axes[1].plot(parameter_cols, row[parameter_cols], marker="o", lw=1.3, alpha=0.80, color=CONFIG_COLOR[original["best_cond"]])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Normalized parameter value")
    axes[1].set_xlabel("Planning parameter")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].scatter(sub["base_score"], sub["best_delta"], s=70, c=[CONFIG_COLOR[c] for c in sub["best_cond"]], edgecolor="white", linewidth=0.7)
    for _, row in sub.iterrows():
        axes[2].text(row["base_score"], row["best_delta"] + 0.35, f"R{int(row['rank'])}", ha="center", fontsize=7.5)
    axes[2].axhline(0, color="#111827", lw=0.8, ls="--")
    axes[2].set_xlabel("No-switch mean score")
    axes[2].set_ylabel("Best switch gain")
    handles = [Patch(facecolor=CONFIG_COLOR[c], label=CONFIG_LABEL[c]) for c in SWITCH_CONFIGS]
    axes[2].legend(handles=handles, frameon=False, loc="upper left")
    save(fig, "fig9_sce2m_parameter_group_diversity_and_portfolio")


def plot_parameter_pool_portfolio(matrix: pd.DataFrame) -> None:
    rows = []
    for _, row in matrix.iterrows():
        rows.append(
            {
                "scene": row["scene"],
                "rank": row["rank"],
                "base_score": row["base_score"],
                "best_switch_score": max(row["ms_ex_score"], row["ms_fz020_score"], row["ms_fz050_score"]),
                "best_any_score": max(row["base_score"], row["ms_ex_score"], row["ms_fz020_score"], row["ms_fz050_score"]),
                "best_cond": row["best_cond"],
            }
        )
    portfolio = pd.DataFrame(rows)
    portfolio.to_csv(TABLE / "parameter_pool_portfolio_summary.csv", index=False, encoding="utf-8-sig")

    scene_order = [SCENE_LABEL[s] for s in SCENE_ORDER]
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 3.7), constrained_layout=True)
    x = np.arange(len(scene_order))
    width = 0.25
    no_values = []
    switch_values = []
    any_values = []
    for scene in scene_order:
        sub = portfolio[portfolio["scene"] == scene]
        no_values.append(float(sub["base_score"].max()))
        switch_values.append(float(sub["best_switch_score"].max()))
        any_values.append(float(sub["best_any_score"].max()))
    ax.bar(x - width, no_values, width=width, color="#9CA3AF", alpha=0.86, label="Best no-switch group")
    ax.bar(x, switch_values, width=width, color="#2563EB", alpha=0.86, label="Best switch-only group")
    ax.bar(x + width, any_values, width=width, color="#F97316", alpha=0.86, label="Best portfolio group")
    for xpos, base, best in zip(x, no_values, any_values):
        ax.text(xpos + width, best + 0.7, f"{best-base:+.1f}", ha="center", fontsize=7.5, color="#111827")
    ax.set_xticks(x, scene_order)
    ax.set_ylabel("Best mean score across parameter groups")
    ax.set_xlabel("Scenario")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, "fig10_all_scene_parameter_pool_portfolio_envelope")


def plot_parameter_diversity_sce3(matrix: pd.DataFrame) -> None:
    scene = "sce3"
    sub = matrix[matrix["scene"] == scene].sort_values("rank").copy()
    sub["best_switch_score"] = sub[["ms_ex_score", "ms_fz020_score", "ms_fz050_score"]].max(axis=1)
    sub["best_any_score"] = sub[["base_score", "ms_ex_score", "ms_fz020_score", "ms_fz050_score"]].max(axis=1)
    sub.to_csv(TABLE / "sce3_parameter_group_diversity_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)
    x = np.arange(len(sub))
    axes[0].bar(x - 0.18, sub["base_score"], width=0.36, color="#9CA3AF", alpha=0.80, label="No switch score")
    axes[0].bar(x + 0.18, sub["best_switch_score"], width=0.36, color=[CONFIG_COLOR[c] for c in sub["best_cond"]], alpha=0.86, label="Best switch score")
    axes[0].set_xticks(x, [f"R{int(rank)}" for rank in sub["rank"]])
    axes[0].set_ylabel("Mean score")
    axes[0].set_xlabel("sce3 parameter group")
    axes[0].legend(frameon=False, loc="upper left")

    parameter_cols = ["beam_width", "lookahead_steps", "min_visits", "discount_factor", "min_cum_prob"]
    norm = sub[parameter_cols].astype(float).copy()
    for col in parameter_cols:
        low, high = norm[col].min(), norm[col].max()
        norm[col] = 0.5 if high == low else (norm[col] - low) / (high - low)
    for _, row in norm.iterrows():
        original = sub.loc[row.name]
        axes[1].plot(parameter_cols, row[parameter_cols], marker="o", lw=1.3, alpha=0.80, color=CONFIG_COLOR[original["best_cond"]])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("Normalized parameter value")
    axes[1].set_xlabel("Planning parameter")
    axes[1].tick_params(axis="x", rotation=25)

    cumulative_no = sub["base_score"].cummax()
    cumulative_switch = sub["best_switch_score"].cummax()
    axes[2].plot(sub["rank"], cumulative_no, marker="o", color="#9CA3AF", lw=1.8, label="Cumulative best no-switch")
    axes[2].plot(sub["rank"], cumulative_switch, marker="o", color="#2563EB", lw=1.8, label="Cumulative best switch")
    axes[2].fill_between(sub["rank"], cumulative_no, cumulative_switch, where=cumulative_switch >= cumulative_no, color="#2563EB", alpha=0.14)
    axes[2].set_xlabel("Parameter groups included by rank")
    axes[2].set_ylabel("Portfolio upper-envelope score")
    axes[2].legend(frameon=False, loc="lower right")
    save(fig, "fig11_sce3_parameter_group_diversity_portfolio_envelope")


def write_report(matrix: pd.DataFrame, effect: pd.DataFrame) -> None:
    selected_lines = []
    for scene, rank in SELECTED_GROUPS.items():
        row = matrix[(matrix["scene"] == scene) & (matrix["rank"] == rank)].iloc[0]
        selected_lines.append(
            f"- `{scene} R{rank}`：no-switch 均分 `{row['base_score']:.2f}`；"
            f"Exact `{row['ms_ex_score']:.2f}` ({row['ms_ex']:+.2f})，"
            f"Fuzzy 0.20 `{row['ms_fz020_score']:.2f}` ({row['ms_fz020']:+.2f})，"
            f"Fuzzy 0.50 `{row['ms_fz050_score']:.2f}` ({row['ms_fz050']:+.2f})。"
        )

    scene_summary = []
    for scene in [SCENE_LABEL[s] for s in SCENE_ORDER]:
        sub = matrix[matrix["scene"] == scene]
        scene_summary.append(
            f"- `{scene}`：{int(sub['any_positive'].sum())}/8 个参数组至少一种 switch 配置优于 no-switch，"
            f"最佳 switch 平均收益 `{sub['best_delta'].mean():+.2f}`。"
        )

    lines = [
        "# Fixed-pool switch-grid full v2 figures",
        "",
        f"- 数据来源：`{ANALYSIS}`",
        f"- 输出目录：`{OUT}`",
        "- 本版图表在全局矩阵、场景配置统计和收益分布中显式加入 `No switch` 基准列，其 delta 固定为 0。",
        "",
        "## 图表索引",
        "",
        "- `fig1_param_group_config_delta_heatmap_with_no_switch.pdf`：场景 × 参数组 × no-switch/三类 switch 配置的 delta 热图。",
        "- `fig2_scene_config_mean_delta_positive_rate_with_no_switch.pdf`：场景级配置均值与正收益比例，含 no-switch 基准。",
        "- `fig4_switch_delta_distribution_with_no_switch.pdf`：各场景参数组收益分布，含 no-switch 基准分布。",
        "- `fig7_selected_param_group_episode_score_boxplots.pdf`：指定场景最优/代表参数组的 50 episode score 箱线图。",
        "- `fig8_selected_param_group_score_mean_bar.pdf`：同一批指定参数组的均值±标准差柱状图。",
        "- `fig9_sce2m_parameter_group_diversity_and_portfolio.pdf`：以 sce2m 为例展示参数组多样性、最佳 switch 收益和参数空间分布。",
        "- `fig10_all_scene_parameter_pool_portfolio_envelope.pdf`：比较每个场景中 no-switch 最优参数组、switch-only 最优参数组和参数池组合上界。",
        "- `fig11_sce3_parameter_group_diversity_portfolio_envelope.pdf`：以 sce3 为例展示多参数组保留如何提高 switch portfolio 上界。",
        "",
        "## 场景级结论",
        "",
        *scene_summary,
        "",
        "## 指定参数组实例结论",
        "",
        *selected_lines,
        "",
        "## 多样性解释",
        "",
        "- `sce2m` 的 8 个高质量参数组不是同一种解的重复：它们在 `beam_width`、`lookahead_steps`、`min_visits`、`discount_factor`、`min_cum_prob` 上形成不同组合。",
        "- `sce3` 更适合说明参数池组合上界：单个 no-switch 最优参数组并不能覆盖 switch 后的最佳解，多参数组保留使 switch-only 上界进一步提高。",
        "- 不同参数组对应的最佳 switch 阈值不同，说明 Switch-Aware 机制更像一个参数组条件下的路径承接模块，而不是单一固定阈值模块。",
        "- 因此，多参数组池的价值在于保留多类规划逻辑；后续 Synergy 或最终选择阶段可以从这些不同规划逻辑中选择更适合当前场景随机性的配置。",
    ]
    (OUT / "README_FIXED_POOL_SWITCH_PAIR_FULL_V2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(ANALYSIS / "switch_grid_comparison_vs_no_switch.csv")
    episode = pd.read_csv(ANALYSIS / "switch_grid_episode_summary.csv")
    matrix = build_matrix(comparison)
    effect = build_effect(matrix)
    matrix.to_csv(TABLE / "param_group_switch_config_matrix_with_no_switch.csv", index=False, encoding="utf-8-sig")
    effect.to_csv(TABLE / "scene_config_effect_with_no_switch.csv", index=False, encoding="utf-8-sig")
    plot_heatmap_with_baseline(matrix)
    plot_scene_config_summary_with_baseline(effect)
    plot_delta_distribution_with_baseline(matrix)
    plot_filtered_score_distribution_with_baseline(matrix, episode)
    plot_filtered_score_distribution_independent_axes()
    plot_filtered_score_distribution_independent_axes_trimmed()
    plot_filtered_score_distribution_independent_axes_trimmed_row()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper_compact()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper_narrow()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper_narrow_widebox()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper_narrow_widebox_close()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper_tight_axes()
    plot_filtered_score_distribution_independent_axes_trimmed_row_paper_tight_axes_full_labels()
    plot_selected_groups_box_and_bar(episode, matrix)
    plot_parameter_diversity_sce2m(matrix)
    plot_parameter_pool_portfolio(matrix)
    plot_parameter_diversity_sce3(matrix)
    write_report(matrix, effect)
    print(OUT)


if __name__ == "__main__":
    main()
