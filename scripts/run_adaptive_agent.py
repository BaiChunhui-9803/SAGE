#!/usr/bin/env python
"""
Adaptive Decision Agent Evaluation Script
Tests and compares all decision modes
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader as DataLoaderClass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_pool(device):
    """Load all models into model pool"""
    from src.decision.model_pool import ModelPoolManager
    from src.decision.strategy_router import StrategyRouter

    model_dir = ROOT_DIR / "cache" / "model"

    config = {
        "q_only": {
            "model_type": "q_network",
            "model_path": str(model_dir / "q_network.pth"),
            "context_window": 0,
            "min_history": 0,
            "max_history": 4,
            "prediction_steps": [1],
            "fallback": None,
            "strategy": "q_only",
            "description": "Q-value only",
        },
        "DT_ctx5": {
            "model_type": "decision_transformer",
            "model_path": str(model_dir / "dt_ctx5.pth"),
            "context_window": 5,
            "min_history": 5,
            "max_history": 9,
            "prediction_steps": [1, 3],
            "fallback": "q_only",
            "strategy": "hybrid",
            "description": "DT context 5",
        },
        "DT_ctx10": {
            "model_type": "decision_transformer",
            "model_path": str(model_dir / "dt_ctx10.pth"),
            "context_window": 10,
            "min_history": 10,
            "max_history": 19,
            "prediction_steps": [1, 3, 5],
            "fallback": "DT_ctx5",
            "strategy": "hybrid",
            "description": "DT context 10",
        },
        "DT_ctx20": {
            "model_type": "decision_transformer",
            "model_path": str(model_dir / "dt_ctx20_v2.pth"),
            "context_window": 20,
            "min_history": 20,
            "max_history": 1000,
            "prediction_steps": [1, 3, 5],
            "fallback": "DT_ctx10",
            "strategy": "hybrid",
            "description": "DT context 20 (retrained)",
        },
        "state_predictor": {
            "model_type": "state_predictor",
            "model_path": str(model_dir / "state_predictor.pth"),
            "context_window": 0,
            "min_history": 0,
            "max_history": 1000,
            "prediction_steps": [1],
            "fallback": None,
            "strategy": "none",
            "description": "State transition predictor",
        },
    }

    pool = ModelPoolManager(config, model_dir, device)
    pool.preload_all()

    router = StrategyRouter(pool)

    return pool, router


def create_adaptive_agent(pool, router, action_vocab, device):
    """Create adaptive decision agent"""
    from src.env.agents.AdaptiveDecisionAgent import AdaptiveDecisionAgent

    return AdaptiveDecisionAgent(
        model_pool=pool,
        strategy_router=router,
        action_vocab=action_vocab,
        device=device,
    )


def evaluate_by_history_range(agent, test_data, r_log, device):
    """Evaluate agent performance by history length ranges"""
    results = defaultdict(
        lambda: {"correct": 0, "total": 0, "q_pred_sum": 0.0, "q_true_sum": 0.0}
    )

    for ep_idx in range(len(test_data["states"])):
        states = test_data["states"][ep_idx]
        actions = test_data["actions"][ep_idx]
        rewards = r_log[ep_idx] if ep_idx < len(r_log) else []

        agent.reset(target_return=sum(rewards) if rewards else 1.0)

        for t in range(len(states)):
            available_history = t

            # Determine which range this falls into
            if available_history < 5:
                range_name = "0-4 (Q-only)"
            elif available_history < 10:
                range_name = "5-9 (DT_ctx5)"
            elif available_history < 20:
                range_name = "10-19 (DT_ctx10)"
            else:
                range_name = "20+ (DT_ctx20)"

            # Get action
            last_reward = rewards[t - 1] if t > 0 and rewards else None
            action_id, action_name, result = agent.get_action(states[t], last_reward)

            true_action = actions[t]

            # Track results
            results[range_name]["total"] += 1
            if action_id == true_action:
                results[range_name]["correct"] += 1

            # Track Q-values
            if result.q_values:
                q_pred = result.q_values.get(action_id, 0)
                q_true = result.q_values.get(true_action, 0)
                results[range_name]["q_pred_sum"] += q_pred
                results[range_name]["q_true_sum"] += q_true

    return dict(results)


def main():
    logger.info("=" * 60)
    logger.info("Adaptive Decision Agent Evaluation")
    logger.info("=" * 60)

    set_seed(42)
    device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load data
    cfg = get_config()
    data_loader = DataLoaderClass(cfg)
    action_vocab = data_loader.action_vocab
    dt_data = data_loader.dt_data
    r_log = data_loader.r_log

    # Split test data
    test_size = 0.2
    test_start = int(len(dt_data["states"]) * (1 - test_size))
    test_data = {
        "states": dt_data["states"][test_start:],
        "actions": dt_data["actions"][test_start:],
    }
    test_rewards = r_log[test_start:]

    logger.info(f"Test episodes: {len(test_data['states'])}")

    # Load model pool
    logger.info("Loading model pool...")
    pool, router = load_model_pool(device)

    # Create adaptive agent
    logger.info("Creating adaptive agent...")
    agent = create_adaptive_agent(pool, router, action_vocab, device)

    # Evaluate
    logger.info("Evaluating adaptive agent...")
    results = evaluate_by_history_range(agent, test_data, test_rewards, device)

    # Print results
    print("\n" + "=" * 60)
    print("       ADAPTIVE AGENT EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Range':<20} {'Accuracy':<12} {'Avg Q(pred)':<12} {'Avg Q(true)':<12}")
    print("-" * 60)

    total_correct = 0
    total_all = 0
    total_q_pred = 0.0
    total_q_true = 0.0

    for range_name in [
        "0-4 (Q-only)",
        "5-9 (DT_ctx5)",
        "10-19 (DT_ctx10)",
        "20+ (DT_ctx20)",
    ]:
        if range_name in results:
            r = results[range_name]
            acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
            avg_q_pred = r["q_pred_sum"] / r["total"] if r["total"] > 0 else 0
            avg_q_true = r["q_true_sum"] / r["total"] if r["total"] > 0 else 0

            print(
                f"{range_name:<20} {acc:>10.2f}%   {avg_q_pred:>12.2f} {avg_q_true:>12.2f}"
            )

            total_correct += r["correct"]
            total_all += r["total"]
            total_q_pred += r["q_pred_sum"]
            total_q_true += r["q_true_sum"]

    print("-" * 60)
    overall_acc = total_correct / total_all * 100 if total_all > 0 else 0
    overall_q_pred = total_q_pred / total_all if total_all > 0 else 0
    overall_q_true = total_q_true / total_all if total_all > 0 else 0
    print(
        f"{'OVERALL':<20} {overall_acc:>10.2f}%   {overall_q_pred:>12.2f} {overall_q_true:>12.2f}"
    )

    q_improvement = overall_q_pred - overall_q_true
    print(f"\nQ-value Improvement: {q_improvement:+.4f}")

    print("=" * 60)

    # Save results
    output_dir = ROOT_DIR / "output" / "adaptive_agent"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"adaptive_results_{timestamp}.txt"

    with open(result_file, "w") as f:
        f.write("Adaptive Decision Agent Results\n")
        f.write("=" * 40 + "\n\n")
        for range_name in [
            "0-4 (Q-only)",
            "5-9 (DT_ctx5)",
            "10-19 (DT_ctx10)",
            "20+ (DT_ctx20)",
        ]:
            if range_name in results:
                r = results[range_name]
                acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
                f.write(f"{range_name}: {acc:.2f}% ({r['correct']}/{r['total']})\n")

    logger.info(f"Results saved to {result_file}")
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
