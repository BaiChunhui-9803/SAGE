from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\Binich\PycharmProjects\PredictionRTS")
ANALYSIS_DIR = ROOT / "output" / "figures" / "etg-only-planning-parameter-analysis"
ALL_SCENES_DIR = ANALYSIS_DIR / "all_scenes"
CSV_PATH = ANALYSIS_DIR / "parameter_numeric_correlations.csv"

OUT_STEM = "fig8_numeric_correlation_lollipop_2x3_single_column_marked"

SCENE_GRID = [
    ["sce-1", "sce-2", "sce-3"],
    ["sce-1m", "sce-2m", "sce-3m"],
]
SCENE_ORDER = [scene for row in SCENE_GRID for scene in row]
SCENE_LABEL = {
    "sce-1": "sce1",
    "sce-1m": "sce1m",
    "sce-2": "sce2",
    "sce-2m": "sce2m",
    "sce-3": "sce3",
    "sce-3m": "sce3m",
}

PARAM_ORDER = [
    "min_visits",
    "beam_width",
    "lookahead_steps",
    "enable_backup",
    "backup_score_threshold",
    "backup_distance_threshold",
]
PARAM_LABEL = {
    "min_visits": "min visits",
    "beam_width": "beam width",
    "lookahead_steps": "lookahead",
    "enable_backup": "switch enabled",
    "backup_score_threshold": "switch score th.",
    "backup_distance_threshold": "switch dist. th.",
}

PARAM_GROUP = {
    "min_visits": "1. experience support",
    "beam_width": "2. beam/lookahead",
    "lookahead_steps": "2. beam/lookahead",
    "enable_backup": "3. backup switch",
    "backup_score_threshold": "3. backup switch",
    "backup_distance_threshold": "3. backup switch",
}
GROUP_COLOR = {
    "1. experience support": "#FFF1B8",
    "2. beam/lookahead": "#CDEAFE",
    "3. backup switch": "#FFD6C9",
}

FIG_SIZE = (3.45, 2.75)
SUBPLOT_ADJUST = {
    "left": 0.265,
    "right": 0.985,
    "bottom": 0.115,
    "top": 0.87,
    "wspace": 0.24,
    "hspace": 0.18,
}
X_LIM = (-0.62, 0.62)
X_TICKS = [-0.5, 0.0, 0.5]


def _load_data() -> pd.DataFrame:
    data = pd.read_csv(CSV_PATH)
    data = data[
        (data["metric"] == "avg_score")
        & (data["map_key"].isin(SCENE_ORDER))
        & (data["parameter"].isin(PARAM_ORDER))
    ].copy()
    data["scene_order"] = data["map_key"].map({scene: idx for idx, scene in enumerate(SCENE_ORDER)})
    data["param_order"] = data["parameter"].map({param: idx for idx, param in enumerate(PARAM_ORDER)})
    return data.sort_values(["scene_order", "param_order"])


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_panel(ax: plt.Axes, panel: pd.DataFrame, scene: str, show_y_labels: bool, show_x_label: bool) -> None:
    y_positions = {param: len(PARAM_ORDER) - 1 - idx for idx, param in enumerate(PARAM_ORDER)}

    for param in PARAM_ORDER:
        y = y_positions[param]
        group = PARAM_GROUP[param]
        ax.axhspan(y - 0.38, y + 0.38, color=GROUP_COLOR[group], alpha=0.38, zorder=0)

    ax.axvline(0.0, color="#6B7280", linestyle="--", linewidth=0.75, zorder=1)
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.38, alpha=0.55, zorder=0)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.36, alpha=0.45, zorder=0)

    for _, row in panel.iterrows():
        param = row["parameter"]
        value = float(row["pearson"])
        y = y_positions[param]
        color = "#16A34A" if value >= 0 else "#DC2626"
        marker = "D" if PARAM_GROUP[param] == "3. backup switch" else "o"
        ax.plot([0.0, value], [y, y], color=color, linewidth=1.35, solid_capstyle="round", zorder=2)
        ax.scatter([value], [y], s=15, color=color, marker=marker, edgecolor="white", linewidth=0.35, zorder=3)

        text = f"{value:+.2f}"
        if value >= 0:
            x_text = min(value + 0.022, X_LIM[1] - 0.005)
            ha = "left"
        else:
            x_text = max(value - 0.022, X_LIM[0] + 0.005)
            ha = "right"
        ax.text(x_text, y, text, va="center", ha=ha, fontsize=5.25, color="#374151", zorder=4)

    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(-0.65, len(PARAM_ORDER) - 0.35)
    ax.set_title(SCENE_LABEL[scene], fontsize=6.8, fontweight="normal", pad=3.0)

    y_ticks = [y_positions[param] for param in PARAM_ORDER]
    ax.set_yticks(y_ticks)
    if show_y_labels:
        ax.set_yticklabels([PARAM_LABEL[param] for param in PARAM_ORDER], fontsize=5.8)
        ax.tick_params(axis="y", length=2.4, width=0.65, pad=2.0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.tick_params(axis="x", length=2.4, width=0.65, pad=1.3)
    if not show_x_label:
        ax.set_xticklabels([])


def main() -> None:
    _setup_style()
    data = _load_data()

    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, sharey=False)
    for row_idx, row in enumerate(SCENE_GRID):
        for col_idx, scene in enumerate(row):
            ax = axes[row_idx, col_idx]
            panel = data[data["map_key"] == scene]
            _draw_panel(
                ax,
                panel,
                scene,
                show_y_labels=(col_idx == 0),
                show_x_label=(row_idx == len(SCENE_GRID) - 1),
            )

    handles = [
        mpatches.Patch(facecolor=GROUP_COLOR[group], edgecolor="none", alpha=0.55, label=group)
        for group in ["1. experience support", "2. beam/lookahead", "3. backup switch"]
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.985),
        handlelength=1.2,
        columnspacing=0.85,
    )
    fig.text(0.61, 0.035, "Pearson r", ha="center", va="center", fontsize=6.4)
    fig.subplots_adjust(**SUBPLOT_ADJUST)

    png_path = ALL_SCENES_DIR / f"{OUT_STEM}.png"
    pdf_path = ALL_SCENES_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
