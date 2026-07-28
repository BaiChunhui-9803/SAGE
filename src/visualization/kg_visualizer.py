"""
Knowledge Graph Visualizer Module

This module provides static visualization tools for decision knowledge graphs.
Generates various charts and reports to validate and analyze the knowledge graph.

Author: PredictionRTS Team
Date: 2026-03-23
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from src.decision.knowledge_graph import DecisionKnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeGraphVisualizer:
    """
    Static visualizer for Decision Knowledge Graph.

    Provides multiple visualization types:
    - Action quality heatmap
    - Top-K action distribution
    - Action frequency distribution
    - Win rate distribution
    - Quality vs visits scatter plot
    - State coverage analysis

    Example:
        kg = DecisionKnowledgeGraph.load("cache/knowledge_graph/kg_simple.pkl")
        viz = KnowledgeGraphVisualizer(kg)
        viz.plot_action_quality_heatmap(top_states=50, save_path="heatmap.png")
        viz.generate_summary_report(output_dir="output/figures/knowledge_graph")
    """

    ACTION_COLORS = {
        "a": "#1f77b4",
        "b": "#ff7f0e",
        "c": "#2ca02c",
        "d": "#d62728",
        "e": "#9467bd",
        "f": "#8c564b",
        "g": "#e377c2",
        "h": "#7f7f7f",
        "i": "#bcbd22",
        "j": "#17becf",
        "k": "#aec7e8",
    }

    def __init__(self, kg: DecisionKnowledgeGraph):
        self.kg = kg
        self.setup_style()

    def setup_style(self):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 150
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3

    def _get_action_letter(self, action: str) -> str:
        if len(action) >= 1:
            return action[-1]
        return action

    def _extract_all_stats(
        self,
    ) -> Tuple[List[int], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = []
        actions = []
        visits_list = []
        quality_list = []
        win_rate_list = []
        future_reward_list = []

        for key, action_dict in self.kg.state_action_map.items():
            state = key if isinstance(key, int) else key[0]
            for action, stats in action_dict.items():
                states.append(state)
                actions.append(action)
                visits_list.append(stats.visits)
                quality_list.append(stats.quality_score)
                win_rate_list.append(stats.win_rate)
                future_reward_list.append(stats.avg_future_reward)

        return (
            states,
            actions,
            np.array(visits_list),
            np.array(quality_list),
            np.array(win_rate_list),
            np.array(future_reward_list),
        )

    def plot_action_quality_heatmap(
        self,
        top_states: int = 50,
        top_actions: int = 15,
        metric: str = "quality_score",
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot action quality heatmap for top states and actions.

        Args:
            top_states: Number of top states (by visits) to include
            top_actions: Number of top actions to include
            metric: Metric to display ('quality_score', 'win_rate', 'avg_future_reward')
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        state_visits = Counter()
        action_visits = Counter()
        data_matrix = {}

        for key, action_dict in self.kg.state_action_map.items():
            state = key if isinstance(key, int) else key[0]
            for action, stats in action_dict.items():
                state_visits[state] += stats.visits
                action_visits[action] += stats.visits
                data_matrix[(state, action)] = getattr(stats, metric)

        top_state_ids = [s for s, _ in state_visits.most_common(top_states)]
        top_action_ids = [a for a, _ in action_visits.most_common(top_actions)]

        matrix = np.full((len(top_state_ids), len(top_action_ids)), np.nan)
        for i, state in enumerate(top_state_ids):
            for j, action in enumerate(top_action_ids):
                if (state, action) in data_matrix:
                    matrix[i, j] = data_matrix[(state, action)]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=top_action_ids,
            yticklabels=top_state_ids,
            cmap="RdYlGn",
            annot=False,
            fmt=".1f",
            cbar_kws={"label": metric.replace("_", " ").title()},
            mask=np.isnan(matrix),
        )

        ax.set_xlabel("Action", fontsize=12)
        ax.set_ylabel("State ID", fontsize=12)
        ax.set_title(
            f"Action {metric.replace('_', ' ').title()} Heatmap (Top {top_states} States)",
            fontsize=14,
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved heatmap to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_top_k_distribution(
        self,
        k: int = 5,
        top_states: int = 20,
        metric: str = "quality_score",
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot top-k action distribution for top states.

        Args:
            k: Number of top actions per state
            top_states: Number of top states to display
            metric: Metric for ranking actions
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        state_visits = Counter()
        for key in self.kg.state_action_map.keys():
            state = key if isinstance(key, int) else key[0]
            for action, stats in self.kg.state_action_map[key].items():
                state_visits[state] += stats.visits

        top_state_ids = [s for s, _ in state_visits.most_common(top_states)]

        n_cols = 4
        n_rows = (len(top_state_ids) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, state in enumerate(top_state_ids):
            ax = axes[idx]

            top_actions = self.kg.get_top_k_actions(
                state=state, k=k, metric=metric, min_visits=1
            )

            if not top_actions:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"State {state}")
                continue

            actions = [a for a, _ in top_actions]
            scores = [s[metric] for _, s in top_actions]
            win_rates = [s["win_rate"] for _, s in top_actions]

            colors = [
                self.ACTION_COLORS.get(self._get_action_letter(a), "#333333")
                for a in actions
            ]

            bars = ax.barh(range(len(actions)), scores, color=colors, alpha=0.8)

            for i, (bar, wr) in enumerate(zip(bars, win_rates)):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{wr * 100:.0f}%",
                    va="center",
                    fontsize=8,
                )

            ax.set_yticks(range(len(actions)))
            ax.set_yticklabels(actions)
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_title(f"State {state}", fontsize=10)
            ax.invert_yaxis()

        for idx in range(len(top_state_ids), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"Top-{k} Actions by {metric.replace('_', ' ').title()} (Top {top_states} States)",
            fontsize=14,
        )
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved top-k distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_action_frequency(
        self,
        top_actions: int = 20,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot action frequency distribution.

        Args:
            top_actions: Number of top actions to display
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        action_counts = Counter()
        for key, action_dict in self.kg.state_action_map.items():
            for action, stats in action_dict.items():
                action_counts[action] += stats.visits

        top_actions_list = action_counts.most_common(top_actions)
        actions = [a for a, _ in top_actions_list]
        counts = [c for _, c in top_actions_list]

        fig, ax = plt.subplots(figsize=figsize)

        colors = [
            self.ACTION_COLORS.get(self._get_action_letter(a), "#333333")
            for a in actions
        ]
        bars = ax.bar(range(len(actions)), counts, color=colors, alpha=0.8)

        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions, rotation=45, ha="right")
        ax.set_xlabel("Action", fontsize=12)
        ax.set_ylabel("Total Visits", fontsize=12)
        ax.set_title(f"Action Frequency Distribution (Top {top_actions})", fontsize=14)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved action frequency to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_win_rate_distribution(
        self,
        bins: int = 20,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot win rate distribution histogram.

        Args:
            bins: Number of histogram bins
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        win_rates = []
        for key, action_dict in self.kg.state_action_map.items():
            for stats in action_dict.values():
                win_rates.append(stats.win_rate)

        win_rates = np.array(win_rates)

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(win_rates, bins=bins, color="#2ca02c", alpha=0.7, edgecolor="black")

        ax.axvline(
            np.mean(win_rates),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(win_rates):.2f}",
        )
        ax.axvline(
            np.median(win_rates),
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Median: {np.median(win_rates):.2f}",
        )

        ax.set_xlabel("Win Rate", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Win Rate Distribution of State-Action Pairs", fontsize=14)
        ax.legend()

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved win rate distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_quality_vs_visits(
        self,
        min_visits: int = 1,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot quality score vs visits scatter plot.

        Args:
            min_visits: Minimum visits threshold
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        visits = []
        quality_scores = []
        win_rates = []

        for key, action_dict in self.kg.state_action_map.items():
            for stats in action_dict.values():
                if stats.visits >= min_visits:
                    visits.append(stats.visits)
                    quality_scores.append(stats.quality_score)
                    win_rates.append(stats.win_rate)

        visits = np.array(visits)
        quality_scores = np.array(quality_scores)
        win_rates = np.array(win_rates)

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(
            visits,
            quality_scores,
            c=win_rates,
            cmap="RdYlGn",
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidths=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Win Rate", fontsize=11)

        ax.set_xlabel("Visits (log scale)", fontsize=12)
        ax.set_ylabel("Quality Score", fontsize=12)
        ax.set_title("Quality Score vs Visits (color = Win Rate)", fontsize=14)
        ax.set_xscale("log")

        corr = np.corrcoef(visits, quality_scores)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved quality vs visits to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_state_coverage(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot state coverage (number of actions per state).

        Args:
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        state_action_counts = Counter()
        for key, action_dict in self.kg.state_action_map.items():
            state = key if isinstance(key, int) else key[0]
            state_action_counts[state] = len(action_dict)

        action_counts = list(state_action_counts.values())

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax1 = axes[0]
        ax1.hist(action_counts, bins=30, color="#9467bd", alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Number of Actions per State", fontsize=12)
        ax1.set_ylabel("Frequency (Number of States)", fontsize=12)
        ax1.set_title("Distribution of Actions per State", fontsize=12)
        ax1.axvline(
            np.mean(action_counts),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(action_counts):.1f}",
        )

        ax2 = axes[1]
        top_states = state_action_counts.most_common(30)
        states = [str(s) for s, _ in top_states]
        counts = [c for _, c in top_states]

        ax2.barh(range(len(states)), counts, color="#8c564b", alpha=0.8)
        ax2.set_yticks(range(len(states)))
        ax2.set_yticklabels(states)
        ax2.set_xlabel("Number of Unique Actions", fontsize=12)
        ax2.set_ylabel("State ID", fontsize=12)
        ax2.set_title("Top 30 States by Action Diversity", fontsize=12)
        ax2.invert_yaxis()

        fig.suptitle("State Coverage Analysis", fontsize=14)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved state coverage to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_metrics_summary(
        self,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot comprehensive metrics summary dashboard.

        Args:
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        win_rates = []
        for key, action_dict in self.kg.state_action_map.items():
            for stats in action_dict.values():
                win_rates.append(stats.win_rate)
        ax1.hist(win_rates, bins=20, color="#2ca02c", alpha=0.7)
        ax1.set_xlabel("Win Rate")
        ax1.set_ylabel("Count")
        ax1.set_title("Win Rate Distribution")

        ax2 = fig.add_subplot(gs[0, 1])
        quality_scores = []
        for key, action_dict in self.kg.state_action_map.items():
            for stats in action_dict.values():
                quality_scores.append(stats.quality_score)
        ax2.hist(quality_scores, bins=20, color="#1f77b4", alpha=0.7)
        ax2.set_xlabel("Quality Score")
        ax2.set_ylabel("Count")
        ax2.set_title("Quality Score Distribution")

        ax3 = fig.add_subplot(gs[0, 2])
        action_counts = Counter()
        for key, action_dict in self.kg.state_action_map.items():
            for action, stats in action_dict.items():
                action_counts[action] += stats.visits
        top_10 = action_counts.most_common(10)
        ax3.bar(
            [a for a, _ in top_10], [c for _, c in top_10], color="#ff7f0e", alpha=0.7
        )
        ax3.set_xlabel("Action")
        ax3.set_ylabel("Visits")
        ax3.set_title("Top 10 Actions")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax4 = fig.add_subplot(gs[1, :2])
        state_action_counts = Counter()
        for key, action_dict in self.kg.state_action_map.items():
            state = key if isinstance(key, int) else key[0]
            state_action_counts[state] = len(action_dict)
        top_20_states = state_action_counts.most_common(20)
        ax4.bar(
            range(len(top_20_states)),
            [c for _, c in top_20_states],
            color="#9467bd",
            alpha=0.7,
        )
        ax4.set_xticks(range(len(top_20_states)))
        ax4.set_xticklabels([str(s) for s, _ in top_20_states], rotation=45, ha="right")
        ax4.set_xlabel("State ID")
        ax4.set_ylabel("Number of Actions")
        ax4.set_title("Top 20 States by Action Diversity")

        ax5 = fig.add_subplot(gs[1, 2])
        visits_list = []
        quality_list = []
        for key, action_dict in self.kg.state_action_map.items():
            for stats in action_dict.values():
                visits_list.append(stats.visits)
                quality_list.append(stats.quality_score)
        ax5.scatter(visits_list, quality_list, alpha=0.3, s=10)
        ax5.set_xlabel("Visits")
        ax5.set_ylabel("Quality Score")
        ax5.set_title("Quality vs Visits")
        ax5.set_xscale("log")

        ax6 = fig.add_subplot(gs[2, :])
        stats = self.kg.get_statistics()
        metrics_text = f"""
Knowledge Graph Statistics Summary
{"=" * 50}
Total Visits:              {stats["total_visits"]:,}
Total Trajectories:        {stats["total_trajectories"]:,}
Unique States:             {stats["unique_states"]:,}
Unique Actions:            {stats["unique_actions"]:,}
Total Keys:                {stats["total_keys"]:,}
States with Multiple Actions: {stats["states_with_multiple_actions"]:,}
Use Context:               {stats["use_context"]}
Context Window:            {stats["context_window"] if stats["use_context"] else "N/A"}
        """
        ax6.text(
            0.1,
            0.5,
            metrics_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
        )
        ax6.axis("off")

        fig.suptitle(
            "Knowledge Graph Metrics Dashboard", fontsize=16, fontweight="bold"
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved metrics summary to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def generate_summary_report(self, output_dir: Path) -> Path:
        """
        Generate a markdown summary report.

        Args:
            output_dir: Directory to save the report

        Returns:
            Path to the generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = self.kg.get_statistics()

        win_rates = []
        quality_scores = []
        future_rewards = []

        for key, action_dict in self.kg.state_action_map.items():
            for s in action_dict.values():
                win_rates.append(s.win_rate)
                quality_scores.append(s.quality_score)
                future_rewards.append(s.avg_future_reward)

        action_counts = Counter()
        for key, action_dict in self.kg.state_action_map.items():
            for action, s in action_dict.items():
                action_counts[action] += s.visits

        report_path = output_dir / "kg_summary_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Knowledge Graph Summary Report\n\n")
            f.write(f"Generated: 2026-03-23\n\n")

            f.write("## Basic Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Visits | {stats['total_visits']:,} |\n")
            f.write(f"| Total Trajectories | {stats['total_trajectories']:,} |\n")
            f.write(f"| Unique States | {stats['unique_states']:,} |\n")
            f.write(f"| Unique Actions | {stats['unique_actions']:,} |\n")
            f.write(f"| Total Keys | {stats['total_keys']:,} |\n")
            f.write(f"| Use Context | {stats['use_context']} |\n")

            f.write("\n## Quality Metrics\n\n")
            f.write(f"| Metric | Mean | Std | Min | Max |\n")
            f.write(f"|--------|------|-----|-----|-----|\n")
            f.write(
                f"| Win Rate | {np.mean(win_rates):.3f} | {np.std(win_rates):.3f} | {np.min(win_rates):.3f} | {np.max(win_rates):.3f} |\n"
            )
            f.write(
                f"| Quality Score | {np.mean(quality_scores):.2f} | {np.std(quality_scores):.2f} | {np.min(quality_scores):.2f} | {np.max(quality_scores):.2f} |\n"
            )
            f.write(
                f"| Future Reward | {np.mean(future_rewards):.2f} | {np.std(future_rewards):.2f} | {np.min(future_rewards):.2f} | {np.max(future_rewards):.2f} |\n"
            )

            f.write("\n## Top 10 Actions\n\n")
            f.write("| Action | Visits | Percentage |\n")
            f.write("|--------|--------|------------|\n")
            total = sum(action_counts.values())
            for action, count in action_counts.most_common(10):
                f.write(f"| {action} | {count:,} | {count / total * 100:.1f}% |\n")

            f.write("\n## Visualization Files\n\n")
            f.write("The following visualizations were generated:\n\n")
            f.write("1. `01_action_quality_heatmap.png` - Action quality heatmap\n")
            f.write("2. `02_top_k_distribution.png` - Top-K action distribution\n")
            f.write("3. `03_action_frequency.png` - Action frequency distribution\n")
            f.write("4. `04_win_rate_histogram.png` - Win rate distribution\n")
            f.write("5. `05_quality_vs_visits.png` - Quality vs visits scatter plot\n")
            f.write("6. `06_state_coverage.png` - State coverage analysis\n")
            f.write("7. `07_metrics_dashboard.png` - Comprehensive metrics dashboard\n")

            f.write("\n---\n")
            f.write("*Generated by PredictionRTS Knowledge Graph Visualizer*\n")

        logger.info(f"Generated summary report: {report_path}")
        return report_path

    def generate_all_visualizations(
        self,
        output_dir: Path,
        top_states: int = 50,
        top_actions: int = 15,
        top_k: int = 5,
        show: bool = False,
    ) -> List[Path]:
        """
        Generate all visualization types and a summary report.

        Args:
            output_dir: Directory to save all outputs
            top_states: Number of top states for visualizations
            top_actions: Number of top actions for visualizations
            top_k: K value for top-k distribution
            show: Whether to display figures

        Returns:
            List of paths to generated files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        logger.info(f"Generating all visualizations to {output_dir}")

        generated_files.append(
            self.plot_action_quality_heatmap(
                top_states=top_states,
                top_actions=top_actions,
                save_path=output_dir / "01_action_quality_heatmap.png",
                show=show,
            )
        )

        generated_files.append(
            self.plot_top_k_distribution(
                k=top_k,
                top_states=20,
                save_path=output_dir / "02_top_k_distribution.png",
                show=show,
            )
        )

        generated_files.append(
            self.plot_action_frequency(
                top_actions=top_actions,
                save_path=output_dir / "03_action_frequency.png",
                show=show,
            )
        )

        generated_files.append(
            self.plot_win_rate_distribution(
                save_path=output_dir / "04_win_rate_histogram.png",
                show=show,
            )
        )

        generated_files.append(
            self.plot_quality_vs_visits(
                save_path=output_dir / "05_quality_vs_visits.png",
                show=show,
            )
        )

        generated_files.append(
            self.plot_state_coverage(
                save_path=output_dir / "06_state_coverage.png",
                show=show,
            )
        )

        generated_files.append(
            self.plot_metrics_summary(
                save_path=output_dir / "07_metrics_dashboard.png",
                show=show,
            )
        )

        report_path = self.generate_summary_report(output_dir)
        generated_files.append(report_path)

        logger.info(f"Generated {len(generated_files)} files")

        return generated_files
