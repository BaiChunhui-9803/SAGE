"""
Visualize Knowledge Graph

This script generates static visualizations for the decision knowledge graph.
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import get_config, ROOT_DIR
from src.decision.knowledge_graph import DecisionKnowledgeGraph
from src.visualization.kg_visualizer import KnowledgeGraphVisualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize Decision Knowledge Graph")
    parser.add_argument(
        "--kg-type", type=str, default="simple", choices=["simple", "context"]
    )
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--top-states", type=int, default=50)
    parser.add_argument("--top-actions", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    kg_dir = ROOT_DIR / "cache" / "knowledge_graph"

    if args.kg_type == "simple":
        kg_path = kg_dir / "kg_simple.pkl"
        kg_name = "simple"
    else:
        kg_path = kg_dir / f"kg_context_{args.context_window}.pkl"
        kg_name = f"context_{args.context_window}"

    if not kg_path.exists():
        logger.error(f"Knowledge graph not found: {kg_path}")
        logger.error("Please build the knowledge graph first:")
        logger.error("  python scripts/build_knowledge_graph.py --kg-type simple")
        sys.exit(1)

    logger.info(f"Loading knowledge graph from {kg_path}")
    kg = DecisionKnowledgeGraph.load(kg_path)
    stats = kg.get_statistics()
    logger.info(
        f"Loaded KG: {stats['unique_states']} states, {stats['unique_actions']} actions"
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ROOT_DIR / "output" / "figures" / "knowledge_graph" / kg_name
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = KnowledgeGraphVisualizer(kg)

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
