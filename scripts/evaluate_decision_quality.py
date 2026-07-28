#!/usr/bin/env python
"""
Evaluate Decision Quality

This script evaluates the quality of decision recommendations using the knowledge graph.

Metrics:
  - Quality Score: Average future_reward of recommended actions
  - Win Rate: Average win_rate of recommended actions
  - Coverage: How many recommended actions are in knowledge graph top-k
  - Diversity: Entropy of recommended actions
  - Confidence: Average frequency of recommended actions

Usage:
    python scripts/evaluate_decision_quality.py --kg-type simple
    python scripts/evaluate_decision_quality.py --kg-type context --context-window 5

Author: PredictionRTS Team
Date: 2026-03-21
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_config, set_seed, ROOT_DIR
from src.data.loader import DataLoader
from src.decision.knowledge_graph import DecisionKnowledgeGraph

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_name: str,
    test_states: list,
    test_actions: list,
    kg: DecisionKnowledgeGraph,
    top_k: int = 5,
    verbose: bool = False,
) -> dict:
    """
    Evaluate a model's decision quality

    Args:
        model_name: Name of the model (for reporting)
        test_states: List of state sequences
        test_actions: List of action sequences
        kg: Knowledge graph
        top_k: Number of top actions to consider
        verbose: Print details

    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "model": model_name,
        "quality_scores": [],
        "win_rates": [],
        "coverages": [],
        "diversities": [],
        "confidences": [],
        "total_evaluated": 0,
    }

    all_recommended_actions = []

    for ep_idx in range(len(test_states)):
        states = test_states[ep_idx]
        actions = test_actions[ep_idx]

        for t in range(len(states)):
            state = states[t]
            true_action = actions[t]

            # Get top-k recommendations from knowledge graph
            kg_top_actions = kg.get_top_k_actions(
                state=state, k=top_k, metric="quality_score", min_visits=3
            )

            if not kg_top_actions:
                continue

            results["total_evaluated"] += 1

            # Extract recommended actions and their stats
            recommended = [action for action, _ in kg_top_actions]
            all_recommended_actions.extend(recommended)

            # 1. Quality Score
            avg_quality = np.mean(
                [stats["quality_score"] for _, stats in kg_top_actions]
            )
            results["quality_scores"].append(avg_quality)

            # 2. Win Rate
            avg_win_rate = np.mean([stats["win_rate"] for _, stats in kg_top_actions])
            results["win_rates"].append(avg_win_rate)

            # 3. Coverage
            kg_action_set = {action for action, _ in kg_top_actions}
            coverage = len(kg_action_set) / top_k
            results["coverages"].append(coverage)

            # 4. Diversity (entropy of recommended actions)
            if len(recommended) > 0:
                action_counts = Counter(recommended)
                probs = [count / len(recommended) for count in action_counts.values()]
                diversity = -sum(p * np.log2(p) for p in probs if p > 0)
            else:
                diversity = 0
            results["diversities"].append(diversity)

            # 5. Confidence
            avg_confidence = np.mean(
                [
                    kg.get_action_confidence(state=state, action=action)
                    for action in recommended
                ]
            )
            results["confidences"].append(avg_confidence)

    # Compute averages
    if results["total_evaluated"] > 0:
        results["avg_quality_score"] = np.mean(results["quality_scores"])
        results["avg_win_rate"] = np.mean(results["win_rates"])
        results["avg_coverage"] = np.mean(results["coverages"])
        results["avg_diversity"] = np.mean(results["diversities"])
        results["avg_confidence"] = np.mean(results["confidences"])
    else:
        results["avg_quality_score"] = 0
        results["avg_win_rate"] = 0
        results["avg_coverage"] = 0
        results["avg_diversity"] = 0
        results["avg_confidence"] = 0

    # Action distribution
    results["action_distribution"] = Counter(all_recommended_actions)

    return results


def generate_report(
    results: dict, kg_stats: dict, output_path: Path, kg_type: str, context_window: int
):
    """Generate markdown evaluation report"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Decision Quality Evaluation Report\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Knowledge Graph Type**: {kg_type}\n")
        if context_window > 0:
            f.write(f"- **Context Window**: {context_window}\n")
        f.write(f"- **Total Evaluated**: {results['total_evaluated']}\n")
        f.write(f"- **Unique States in KG**: {kg_stats['unique_states']}\n")
        f.write(f"- **Unique Actions in KG**: {kg_stats['unique_actions']}\n\n")

        # Metrics
        f.write("## Evaluation Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|-------|-------|\n")
        f.write(f"| Quality Score | {results['avg_quality_score']:.2f} |\n")
        f.write(f"| Win Rate | {results['avg_win_rate'] * 100:.1f}% |\n")
        f.write(f"| Coverage | {results['avg_coverage'] * 100:.1f}% |\n")
        f.write(f"| Diversity | {results['avg_diversity']:.2f} |\n")
        f.write(f"| Confidence | {results['avg_confidence'] * 100:.1f}% |\n\n")

        # Action Distribution
        f.write("## Top Recommended Actions\n\n")
        f.write("| Action | Count |\n")
        f.write("|--------|-------|\n")
        for action, count in results["action_distribution"].most_common(10):
            f.write(f"| {action} | {count} |\n")

        f.write("\n---\n")
        f.write(
            "*Report generated by PredictionRTS Decision Quality Evaluation System*\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate decision quality")
    parser.add_argument(
        "--kg-type", type=str, default="simple", choices=["simple", "context"]
    )
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    # Load knowledge graph
    kg_dir = ROOT_DIR / "cache" / "knowledge_graph"
    if args.kg_type == "simple":
        kg_path = kg_dir / "kg_simple.pkl"
        context_window = 0
    else:
        kg_path = kg_dir / f"kg_context_{args.context_window}.pkl"
        context_window = args.context_window

    if not kg_path.exists():
        logger.error(f"Knowledge graph not found: {kg_path}")
        logger.error("Please run: python scripts/build_knowledge_graph.py")
        return

    kg = DecisionKnowledgeGraph.load(kg_path)
    kg_stats = kg.get_statistics()

    # Load test data
    cfg = get_config()
    loader = DataLoader(cfg)
    dt_data = loader.dt_data

    # Split test data
    test_start = int(len(dt_data["states"]) * (1 - args.test_size))
    test_states = dt_data["states"][test_start:]
    test_actions = dt_data["actions"][test_start:]

    logger.info(f"Test episodes: {len(test_states)}")

    # Load raw action data
    import csv

    data_root = cfg.get(
        "data_root", "D:/白春辉/实验平台/pymarl/results_HRL_new/Q-bktree"
    )
    map_id = cfg.get("map_id", "MarineMicro_MvsM_4")
    data_id = cfg.get("data_id", "6")

    action_log_path = f"{data_root}/{map_id}/{data_id}/action_log.csv"

    raw_action_episodes = []
    with open(action_log_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                raw = row[0]
                actions = [raw[j : j + 2] for j in range(0, len(raw), 2)]
                raw_action_episodes.append(actions)

    # Convert test actions to raw format
    test_actions_raw = []
    for ep_idx in range(len(test_states)):
        if test_start + ep_idx < len(raw_action_episodes):
            test_actions_raw.append(raw_action_episodes[test_start + ep_idx])
        else:
            test_actions_raw.append([])

    # Evaluate
    logger.info("Evaluating decision quality...")
    results = evaluate_model(
        model_name=f"Knowledge Graph ({args.kg_type})",
        test_states=test_states,
        test_actions=test_actions_raw,
        kg=kg,
        top_k=args.top_k,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("       DECISION QUALITY EVALUATION RESULTS")
    print("=" * 60)

    context_str = f" (context_window={context_window})" if context_window > 0 else ""
    print(f"\nKnowledge Graph: {args.kg_type}{context_str}")
    print(f"Test Episodes: {len(test_states)}")
    print(f"Total Evaluated: {results['total_evaluated']}")
    print()

    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Quality Score':<20} {results['avg_quality_score']:.2f}")
    print(f"{'Win Rate':<20} {results['avg_win_rate'] * 100:.1f}%")
    print(f"{'Coverage':<20} {results['avg_coverage'] * 100:.1f}%")
    print(f"{'Diversity':<20} {results['avg_diversity']:.2f}")
    print(f"{'Confidence':<20} {results['avg_confidence'] * 100:.1f}%")
    print()

    # Save report
    output_dir = ROOT_DIR / "output" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"decision_quality_{timestamp}.md"

    generate_report(
        results=results,
        kg_stats=kg_stats,
        output_path=report_path,
        kg_type=args.kg_type,
        context_window=context_window,
    )

    logger.info(f"\nEvaluation complete!")
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
