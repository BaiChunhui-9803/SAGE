import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader
from src.models.DecisionTransformer import DecisionTransformer
from src.models.QNetwork import QNetwork
from src.models.StateTransitionPredictor import StateTransitionPredictor

# Import directly from file to avoid src/env/__init__.py which requires pysc2
import importlib.util

spec = importlib.util.spec_from_file_location(
    "OptimalDecisionAgent",
    ROOT_DIR / "src" / "env" / "agents" / "OptimalDecisionAgent.py",
)
OptimalDecisionAgent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OptimalDecisionAgent)
DecisionLogger = OptimalDecisionAgent.DecisionLogger
DecisionMode = OptimalDecisionAgent.DecisionMode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_models(cfg, data_loader, device):
    """Load all trained models"""

    state_dim = len(data_loader.state_node_dict)
    action_dim = len(data_loader.action_vocab)
    action_vocab = data_loader.action_vocab

    model_dir = ROOT_DIR / "cache" / "model"

    # Load Decision Transformer
    dt_path = model_dir / "best_model.pth"
    if not dt_path.exists():
        raise FileNotFoundError(f"DT model not found: {dt_path}")

    dt_model = DecisionTransformer(
        state_dim=state_dim,
        act_vocab_size=action_dim,
        n_layer=4,
        n_head=4,
        n_embd=128,
        max_len=2048,
    )
    dt_checkpoint = torch.load(dt_path, map_location="cpu")
    dt_model.load_state_dict(dt_checkpoint["model_state_dict"])
    dt_model.to(device)
    dt_model.eval()
    logger.info(f"Loaded DecisionTransformer from {dt_path}")

    # Load Q-Network
    q_path = model_dir / "q_network.pth"
    q_network = QNetwork(state_dim, action_dim)
    if q_path.exists():
        q_checkpoint = torch.load(q_path, map_location="cpu")
        q_network.load_state_dict(q_checkpoint["model_state_dict"])
        logger.info(f"Loaded Q-Network from {q_path}")
    else:
        logger.warning(f"Q-Network not found at {q_path}, using untrained network")
    q_network.to(device)
    q_network.eval()

    # Load State Predictor
    sp_path = model_dir / "state_predictor.pth"
    state_predictor = StateTransitionPredictor(state_dim, action_dim)
    if sp_path.exists():
        sp_checkpoint = torch.load(sp_path, map_location="cpu")
        state_predictor.load_state_dict(sp_checkpoint["model_state_dict"])
        logger.info(f"Loaded State Predictor from {sp_path}")
    else:
        logger.warning(
            f"State Predictor not found at {sp_path}, using untrained network"
        )
    state_predictor.to(device)
    state_predictor.eval()

    return dt_model, q_network, state_predictor, action_vocab


class SimulatedSC2Environment:
    """
    Simulated SC2 environment using historical data
    For offline evaluation without running actual SC2
    """

    def __init__(self, data_loader, test_ratio=0.2):
        self.data_loader = data_loader
        self.dt_data = data_loader.dt_data
        self.r_log = data_loader.r_log

        # Split into train/test
        total_episodes = len(self.dt_data["states"])
        test_start = int(total_episodes * (1 - test_ratio))

        self.test_states = self.dt_data["states"][test_start:]
        self.test_actions = self.dt_data["actions"][test_start:]
        self.test_rtgs = self.dt_data["rtgs"][test_start:]
        self.test_rewards = self.r_log[test_start:]

        logger.info(f"Simulated environment: {len(self.test_states)} test episodes")

    def get_episode(self, idx):
        """Get a specific episode"""
        return {
            "states": self.test_states[idx],
            "actions": self.test_actions[idx],
            "rtgs": self.test_rtgs[idx],
            "rewards": self.test_rewards[idx],
        }

    def sample_episode(self):
        """Randomly sample an episode"""
        idx = np.random.randint(len(self.test_states))
        return self.get_episode(idx), idx


def run_offline_evaluation(
    agent: OptimalDecisionAgent,
    env: SimulatedSC2Environment,
    num_episodes: int,
    decision_logger: DecisionLogger,
    mode: str,
):
    """
    Run offline evaluation using simulated environment

    Simulates the agent making decisions and compares with ground truth
    """
    logger.info("=" * 60)
    logger.info(f"Running Offline Evaluation (mode={mode})")
    logger.info("=" * 60)

    results = {
        "episodes": [],
        "total_correct": 0,
        "total_decisions": 0,
        "total_q_pred": 0.0,
        "total_q_true": 0.0,
    }

    for ep_idx in range(num_episodes):
        episode, original_idx = env.sample_episode()
        states = episode["states"]
        actions = episode["actions"]
        rewards = episode["rewards"]

        agent.reset(target_return=sum(rewards) if rewards else 0)

        episode_correct = 0
        episode_total = 0
        episode_q_pred = 0.0
        episode_q_true = 0.0

        for t in range(len(states) - 1):
            current_state = states[t]
            true_action = actions[t]
            last_reward = rewards[t - 1] if t > 0 else None

            # Get agent's decision
            pred_action, action_name, decision_info = agent.get_action(
                current_state=current_state,
                last_reward=last_reward,
                episode_id=ep_idx,
                true_action=true_action,
            )

            # Track metrics
            if pred_action == true_action:
                episode_correct += 1
                results["total_correct"] += 1
            episode_total += 1
            results["total_decisions"] += 1

            # Q-value comparison
            q_pred = decision_info.q_values.get(pred_action, 0)
            q_true = decision_info.q_values.get(true_action, 0)
            episode_q_pred += q_pred
            episode_q_true += q_true
            results["total_q_pred"] += q_pred
            results["total_q_true"] += q_true

        episode_accuracy = episode_correct / episode_total if episode_total > 0 else 0

        results["episodes"].append(
            {
                "episode_idx": ep_idx,
                "original_idx": original_idx,
                "accuracy": episode_accuracy,
                "correct": episode_correct,
                "total": episode_total,
                "avg_q_pred": episode_q_pred / episode_total
                if episode_total > 0
                else 0,
                "avg_q_true": episode_q_true / episode_total
                if episode_total > 0
                else 0,
            }
        )

        if (ep_idx + 1) % 10 == 0:
            logger.info(f"Completed {ep_idx + 1}/{num_episodes} episodes")

    # Calculate overall metrics
    results["overall_accuracy"] = results["total_correct"] / results["total_decisions"]
    results["avg_q_pred"] = results["total_q_pred"] / results["total_decisions"]
    results["avg_q_true"] = results["total_q_true"] / results["total_decisions"]
    results["q_improvement"] = results["avg_q_pred"] - results["avg_q_true"]

    return results


def print_results(results: dict):
    """Print evaluation results"""
    print("\n" + "=" * 60)
    print("        OPTIMAL DECISION AGENT EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {results['overall_accuracy'] * 100:.2f}%")
    print(f"  Correct: {results['total_correct']}/{results['total_decisions']}")
    print(f"  Avg Q(pred): {results['avg_q_pred']:.4f}")
    print(f"  Avg Q(true): {results['avg_q_true']:.4f}")
    print(f"  Q Improvement: {results['q_improvement']:.4f}")

    # Per-episode summary
    print(f"\nPer-Episode Summary (first 10):")
    print(f"{'Ep':<4} {'Acc':<8} {'Correct':<10} {'Q(pred)':<10} {'Q(true)':<10}")
    print("-" * 50)

    for ep in results["episodes"][:10]:
        print(
            f"{ep['episode_idx']:<4} {ep['accuracy'] * 100:>6.2f}%  "
            f"{ep['correct']:>3}/{ep['total']:<5} "
            f"{ep['avg_q_pred']:>8.4f}  {ep['avg_q_true']:>8.4f}"
        )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SC2 Optimal Decision Agent Runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["dt_only", "q_only", "hybrid", "adaptive"],
        help="Decision mode",
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of episodes to run"
    )
    parser.add_argument(
        "--context_window", type=int, default=20, help="Context window size"
    )
    parser.add_argument(
        "--dt_weight", type=float, default=0.3, help="DT weight in hybrid mode"
    )
    parser.add_argument(
        "--q_weight", type=float, default=0.7, help="Q weight in hybrid mode"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for logs"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed decision info"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load configuration and data
    cfg = get_config()
    data_loader = DataLoader(cfg)

    # Load models
    dt_model, q_network, state_predictor, action_vocab = load_models(
        cfg, data_loader, device
    )

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT_DIR / "output" / "optimal_agent" / f"{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create decision logger
    decision_logger = DecisionLogger(log_dir=output_dir)

    # Get RTG statistics for target return
    rtg_stats = data_loader.r_log
    if rtg_stats:
        all_returns = [sum(ep) for ep in rtg_stats]
        target_return = np.percentile(all_returns, 90)
    else:
        target_return = 1.0

    # Create agent
    agent = OptimalDecisionAgent.OptimalDecisionAgent(
        dt_model=dt_model,
        q_network=q_network,
        state_predictor=state_predictor,
        action_vocab=action_vocab,
        device=device,
        context_window=args.context_window,
        target_return=target_return,
        mode=args.mode,
        dt_weight=args.dt_weight,
        q_weight=args.q_weight,
        logger_instance=decision_logger,
    )

    logger.info(f"Agent created with mode={args.mode}")

    # Create simulated environment
    env = SimulatedSC2Environment(data_loader)

    # Run evaluation
    results = run_offline_evaluation(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        decision_logger=decision_logger,
        mode=args.mode,
    )

    # Print results
    print_results(results)

    # Save results
    import json

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert numpy values to native Python types
        serializable_results = {
            "mode": args.mode,
            "episodes": args.episodes,
            "overall_accuracy": float(results["overall_accuracy"]),
            "total_correct": int(results["total_correct"]),
            "total_decisions": int(results["total_decisions"]),
            "avg_q_pred": float(results["avg_q_pred"]),
            "avg_q_true": float(results["avg_q_true"]),
            "q_improvement": float(results["q_improvement"]),
            "episode_results": [
                {
                    "episode_idx": int(ep["episode_idx"]),
                    "accuracy": float(ep["accuracy"]),
                    "correct": int(ep["correct"]),
                    "total": int(ep["total"]),
                    "avg_q_pred": float(ep["avg_q_pred"]),
                    "avg_q_true": float(ep["avg_q_true"]),
                }
                for ep in results["episodes"]
            ],
        }
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Save decision logs
    decision_logger.save_all("all_decisions.json")

    # Get and print decision statistics
    stats = decision_logger.get_statistics()
    logger.info(f"Decision Statistics: {stats}")

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
