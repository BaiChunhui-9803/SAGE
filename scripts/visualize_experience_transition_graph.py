"""
Visualize Experience Transition Graph

This script generates static visualizations for the decision experience transition graph.
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import get_config, ROOT_DIR
from src.decision.experience_transition_graph import DecisionExperienceTransitionGraph
from src.visualization.etg_visualizer import ExperienceTransitionGraphVisualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize Decision Experience Transition Graph")
    parser.add_argument(
        "--etg-type", type=str, default="simple", choices=["simple", "context"]
    )
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--top-states", type=int, default=50)
    parser.add_argument("--top-actions", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    etg_dir = ROOT_DIR / "cache" / "experience_transition_graph"

    if args.etg_type == "simple":
        etg_path = etg_dir / "etg_simple.pkl"
        etg_name = "simple"
    else:
        etg_path = etg_dir / f"etg_context_{args.context_window}.pkl"
        etg_name = f"context_{args.context_window}"

    if not etg_path.exists():
        logger.error(f"Experience Transition Graph not found: {etg_path}")
        logger.error("Please build the experience transition graph first:")
        logger.error("  python scripts/build_experience_transition_graph.py --etg-type simple")
        sys.exit(1)

    logger.info(f"Loading experience transition graph from {etg_path}")
    ETG = DecisionExperienceTransitionGraph.load(etg_path)
    stats = ETG.get_statistics()
    logger.info(
        f"Loaded ETG: {stats['unique_states']} states, {stats['unique_actions']} actions"
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT_DIR / "output" / "figures" / "experience_transition_graph" / etg_name
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = ExperienceTransitionGraphVisualizer(ETG)

    logger.info("Generating all visualizations...")

    visualizer.plot_action_quality_heatmap(
        top_states=args.top_states,
        top_actions=args.top_actions,
        save_path=output_dir / "action_quality_heatmap.png",
        show=args.show,
    )

    visualizer.plot_action_frequency(
        top_actions=args.top_actions,
        save_path=output_dir / "action_frequency.png",
        show=args.show,
    )

    visualizer.plot_win_rate_distribution(
        save_path=output_dir / "win_rate_histogram.png", show=args.show
    )

    visualizer.plot_quality_vs_visits(
        save_path=output_dir / "quality_vs_visits.png", show=args.show
    )

    visualizer.plot_state_coverage(
        save_path=output_dir / "state_coverage.png", show=args.show
    )

    visualizer.plot_metrics_summary(
        save_path=output_dir / "metrics_dashboard.png", show=args.show
    )

    visualizer.generate_summary_report(output_dir=output_dir)

    logger.info(f"All visualizations saved to {output_dir}")
    print(f"\nVisualization complete! Check output in: {output_dir}")


if __name__ == "__main__":
    main()
