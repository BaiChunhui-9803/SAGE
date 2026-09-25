from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


SHOW_FIGURE = os.environ.get("FIG4_2X3_SHOW", "0").lower() not in {"0", "false", "no"}
if not SHOW_FIGURE:
    matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "tables" / "fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv"
FIG_DIR = BASE_DIR / "figures"
OUT_STEM = "fig4_switch_aware_backup_score_distribution_by_scene_2x3_single_column"

SCENE_GRID = [
    ["sce1", "sce2", "sce3"],
    ["sce1m", "sce2m", "sce3m"],
]
CONFIG_ORDER = ["ms_ns", "ms_ex", "ms_fz020", "ms_fz050"]
CONFIG_LABEL = {
    "ms_ns": "No switch",
    "ms_ex": "Backup switch - exact",
    "ms_fz020": "Backup switch - fuzzy 0.20",
    "ms_fz050": "Backup switch - fuzzy 0.50",
}
CONFIG_COLOR = {
    "ms_ns": "#9CA3AF",
    "ms_ex": "#2563EB",
    "ms_fz020": "#16A34A",
    "ms_fz050": "#F97316",
}

FIG_SIZE = (5.0, 2.65)
SUBPLOT_ADJUST = {
    "left": 0.085,
    "right": 0.995,
    "bottom": 0.095,
    "top": 0.878,
    "wspace": 0.32,
    "hspace": 0.34,
}
BOX_WIDTH = 0.20
BOX_POSITIONS = [1.00, 1.4, 1.8, 2.2]
X_LIMIT = (0.74, 2.4)

FONT_SIZE = 6.8
AXIS_LABEL_SIZE = 6.8
TICK_LABEL_SIZE = 6.2
LEGEND_SIZE = 7.2
SCENE_TITLE_SIZE = 6.8
BOX_ALPHA = 0.40
BOX_EDGE_WIDTH = 1.05
MEDIAN_WIDTH = 1.15
WHISKER_WIDTH = 0.95


def _save(fig: plt.Figure, output_stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{output_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(FIG_DIR / f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.018)


def _draw_panel(ax: plt.Axes, data: pd.DataFrame, scene: str, show_ylabel: bool, show_xticks: bool) -> None:
    scene_data = data[data["scene"] == scene]
    values = [scene_data[scene_data["config"] == config]["score"].to_numpy() for config in CONFIG_ORDER]

    box = ax.boxplot(
        values,
        positions=BOX_POSITIONS,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
    )

    for patch, config in zip(box["boxes"], CONFIG_ORDER):
        patch.set_facecolor(CONFIG_COLOR[config])
        patch.set_edgecolor(CONFIG_COLOR[config])
        patch.set_alpha(BOX_ALPHA)
        patch.set_linewidth(BOX_EDGE_WIDTH)

    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(MEDIAN_WIDTH)

    for part in box["whiskers"] + box["caps"]:
        part.set_color("#4B5563")
        part.set_linewidth(WHISKER_WIDTH)

    ax.set_xlim(*X_LIMIT)
    ax.set_title(scene, fontweight="normal", fontsize=SCENE_TITLE_SIZE, pad=2.5)
    ax.set_ylabel("Score" if show_ylabel else "", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, width=0.65, length=2.4, pad=1.5)
    ax.set_xticks(BOX_POSITIONS)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color="#D1D5DB", alpha=0.28, linewidth=0.40)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_CSV}")

    data = pd.read_csv(DATA_CSV)

    with plt.rc_context(
        {
            "font.size": FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.40,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, constrained_layout=False)
        fig.subplots_adjust(**SUBPLOT_ADJUST)

        for row_idx, row in enumerate(SCENE_GRID):
            for col_idx, scene in enumerate(row):
                _draw_panel(
                    axes[row_idx, col_idx],
                    data,
                    scene,
                    show_ylabel=(col_idx == 0),
                    show_xticks=False,
                )

        handles = [
            Patch(
                facecolor=CONFIG_COLOR[config],
                edgecolor=CONFIG_COLOR[config],
                linewidth=BOX_EDGE_WIDTH,
                alpha=BOX_ALPHA,
                label=CONFIG_LABEL[config],
            )
            for config in CONFIG_ORDER
        ]
        fig.legend(
            handles=handles,
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
            columnspacing=0.85,
            handlelength=1.15,
            handletextpad=0.42,
        )

        _save(fig, OUT_STEM)
        if SHOW_FIGURE:
            fig.show()
            plt.show()
        else:
            plt.close(fig)

    print(FIG_DIR / f"{OUT_STEM}.png")
    print(FIG_DIR / f"{OUT_STEM}.pdf")


if __name__ == "__main__":
    main()
