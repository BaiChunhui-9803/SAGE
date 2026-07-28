#!/usr/bin/env python
"""
Prediction script for PredictionRTS
Samples from real test data and predicts next K actions

Usage:
    python scripts/run_predict.py
    python scripts/run_predict.py mode=single_step
    python scripts/run_predict.py mode=long_seq samples=200
"""

import logging
import sys
import os
import random
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PREDICTION_MODES = {
    "single_step": 1,
    "short_seq": 3,
    "long_seq": 5,
}


class DTPredictor:
    def __init__(self, model, action_vocab, device, context_window=20):
        self.model = model
        self.device = device
        self.context_window = context_window
        self.id_to_action = {v: k for k, v in action_vocab.items()}
        self.action_vocab = action_vocab

    def predict(self, states, actions, rtgs, k):
        """
        Predict next k actions given history

        Args:
            states: List[int] - state sequence
            actions: List[int] - action sequence
            rtgs: List[float] - RTG sequence
            k: number of steps to predict

        Returns:
            predicted_actions: List[int] - predicted action IDs
        """
        self.model.eval()

        current_states = list(states[-self.context_window :])
        current_actions = list(actions[-self.context_window :])
        current_rtgs = list(rtgs[-self.context_window :])

        predictions = []

        with torch.no_grad():
            for step in range(k):
                seq_len = len(current_states)

                states_tensor = torch.tensor([current_states], dtype=torch.long).to(
                    self.device
                )
                actions_tensor = torch.tensor([current_actions], dtype=torch.long).to(
                    self.device
                )
                rtgs_tensor = torch.tensor([current_rtgs], dtype=torch.float).to(
                    self.device
                )
                timesteps_tensor = torch.arange(seq_len).unsqueeze(0).to(self.device)

                logits = self.model(
                    states_tensor, actions_tensor, rtgs_tensor, timesteps_tensor
                )

                pred_logits = logits[0, -1, :]
                pred_action = torch.argmax(pred_logits).item()
                predictions.append(pred_action)

                if step < k - 1:
                    current_states = current_states[1:] + [current_states[-1]]
                    current_actions = current_actions[1:] + [pred_action]
                    if len(current_rtgs) > 1:
                        current_rtgs = current_rtgs[1:] + [current_rtgs[-1]]

        return predictions

    def predict_with_probs(self, states, actions, rtgs, k):
        """Predict with probability distribution for top-k accuracy"""
        self.model.eval()

        current_states = list(states[-self.context_window :])
        current_actions = list(actions[-self.context_window :])
        current_rtgs = list(rtgs[-self.context_window :])

        predictions = []
        all_probs = []

        with torch.no_grad():
            for step in range(k):
                seq_len = len(current_states)

                states_tensor = torch.tensor([current_states], dtype=torch.long).to(
                    self.device
                )
                actions_tensor = torch.tensor([current_actions], dtype=torch.long).to(
                    self.device
                )
                rtgs_tensor = torch.tensor([current_rtgs], dtype=torch.float).to(
                    self.device
                )
                timesteps_tensor = torch.arange(seq_len).unsqueeze(0).to(self.device)

                logits = self.model(
                    states_tensor, actions_tensor, rtgs_tensor, timesteps_tensor
                )

                pred_logits = logits[0, -1, :]
                probs = torch.softmax(pred_logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

                pred_action = torch.argmax(pred_logits).item()
                predictions.append(pred_action)

                if step < k - 1:
                    current_states = current_states[1:] + [current_states[-1]]
                    current_actions = current_actions[1:] + [pred_action]
                    if len(current_rtgs) > 1:
                        current_rtgs = current_rtgs[1:] + [current_rtgs[-1]]

        return predictions, all_probs


def sample_test_data(dt_data, num_samples, min_length, seed=42):
    """
    Sample from test data

    Returns:
        List of (idx, states, actions, rtgs)
    """
    random.seed(seed)

    valid_indices = []
    for i, (states, actions, rtgs) in enumerate(
        zip(dt_data["states"], dt_data["actions"], dt_data["rtgs"])
    ):
        if len(states) >= min_length:
            valid_indices.append(i)

    if len(valid_indices) < num_samples:
        logger.warning(
            f"Only {len(valid_indices)} valid samples, requested {num_samples}"
        )
        num_samples = len(valid_indices)

    sampled_indices = random.sample(valid_indices, num_samples)

    samples = []
    for idx in sampled_indices:
        samples.append(
            {
                "idx": idx,
                "states": dt_data["states"][idx],
                "actions": dt_data["actions"][idx],
                "rtgs": dt_data["rtgs"][idx],
            }
        )

    return samples


def evaluate_predictions(samples, predictor, k, context_window):
    """
    Evaluate predictions on samples

    Returns:
        metrics dict and example predictions
    """
    total_correct = defaultdict(int)
    total_predictions = defaultdict(int)
    top3_correct = defaultdict(int)

    examples = []

    for sample in samples:
        states = sample["states"]
        actions = sample["actions"]
        rtgs = sample["rtgs"]
        idx = sample["idx"]

        if len(states) < context_window + k:
            continue

        cutoff = len(states) - k

        history_states = states[:cutoff]
        history_actions = actions[:cutoff]
        history_rtgs = rtgs[:cutoff]

        true_actions = actions[cutoff : cutoff + k]

        pred_actions, all_probs = predictor.predict_with_probs(
            history_states, history_actions, history_rtgs, k
        )

        for step in range(min(k, len(true_actions), len(pred_actions))):
            total_predictions[step] += 1

            if pred_actions[step] == true_actions[step]:
                total_correct[step] += 1

            top3_indices = np.argsort(all_probs[step])[-3:][::-1]
            if true_actions[step] in top3_indices:
                top3_correct[step] += 1

        if len(examples) < 5:
            examples.append(
                {
                    "idx": idx,
                    "history_length": len(history_states),
                    "true_actions": true_actions,
                    "pred_actions": pred_actions,
                    "match": pred_actions == list(true_actions),
                }
            )

    metrics = {
        "step_accuracy": {},
        "top3_accuracy": {},
        "overall_accuracy": 0.0,
        "overall_top3": 0.0,
    }

    total_correct_all = sum(total_correct.values())
    total_pred_all = sum(total_predictions.values())
    total_top3_all = sum(top3_correct.values())

    for step in range(k):
        if total_predictions[step] > 0:
            metrics["step_accuracy"][step] = (
                total_correct[step] / total_predictions[step]
            )
            metrics["top3_accuracy"][step] = (
                top3_correct[step] / total_predictions[step]
            )

    if total_pred_all > 0:
        metrics["overall_accuracy"] = total_correct_all / total_pred_all
        metrics["overall_top3"] = total_top3_all / total_pred_all

    return metrics, examples


def print_results(
    mode, k, num_samples, context_window, metrics, examples, id_to_action
):
    """Print formatted results"""

    print("\n" + "=" * 60)
    print("           PREDICTION RESULTS")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  - Mode: {mode} (K={k})")
    print(f"  - Test Samples: {num_samples}")
    print(f"  - Context Window: {context_window}")

    print(f"\nOverall Results:")
    print(f"  - Accuracy (Top-1): {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"  - Accuracy (Top-3): {metrics['overall_top3'] * 100:.2f}%")

    print(f"\nStep-wise Accuracy:")
    for step in range(k):
        if step in metrics["step_accuracy"]:
            acc = metrics["step_accuracy"][step] * 100
            top3 = metrics["top3_accuracy"].get(step, 0) * 100
            print(f"  - Step {step + 1}: {acc:.2f}% (Top-3: {top3:.2f}%)")

    print(f"\nExample Predictions (first 5):")
    for i, ex in enumerate(examples):
        print(f"\n  Sample #{ex['idx']}:")
        print(f"    History Length: {ex['history_length']}")

        true_actions = [id_to_action.get(a, str(a)) for a in ex["true_actions"]]
        pred_actions = [id_to_action.get(a, str(a)) for a in ex["pred_actions"]]

        print(f"    True Actions:  {true_actions}")
        print(f"    Pred Actions:  {pred_actions}")

        if ex["match"]:
            print(f"    Result: MATCH")
        else:
            matches = sum(
                1 for t, p in zip(ex["true_actions"], ex["pred_actions"]) if t == p
            )
            print(f"    Result: {matches}/{len(ex['true_actions'])} steps matched")

    print("\n" + "=" * 60)


def main():
    cfg = get_config()

    set_seed(cfg.get("seed", 42))
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    data_loader = DataLoader(cfg)

    model_name = cfg.get("model", {}).get("name", "DecisionTransformer")
    checkpoint_name = cfg.get("checkpoint_name", "best_model.pth")
    checkpoint_path = ROOT_DIR / "cache" / "model" / checkpoint_name

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    mode = cfg.get("prediction", {}).get("mode", "single_step")
    k = PREDICTION_MODES.get(mode, 1)

    num_samples = cfg.get("prediction", {}).get("num_samples", 100)
    context_window = cfg.get("prediction", {}).get("context_window", 20)
    min_length = context_window + k

    logger.info(f"Prediction mode: {mode}, K={k}")
    logger.info(f"Loading model from {checkpoint_path}")

    if model_name == "DecisionTransformer":
        from src.models.DecisionTransformer import DecisionTransformer

        state_dim = len(data_loader.state_node_dict)
        act_vocab_size = len(data_loader.action_vocab)

        model = DecisionTransformer(
            state_dim=state_dim,
            act_vocab_size=act_vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=128,
            max_len=2048,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        logger.info("Model loaded successfully")

        logger.info("Loading test data...")
        dt_data = data_loader.dt_data
        action_vocab = data_loader.action_vocab

        test_size = 0.2
        total_samples = len(dt_data["states"])
        test_start = int(total_samples * (1 - test_size))

        test_data = {
            "states": dt_data["states"][test_start:],
            "actions": dt_data["actions"][test_start:],
            "rtgs": dt_data["rtgs"][test_start:],
        }

        logger.info(f"Test set size: {len(test_data['states'])}")

        logger.info(f"Sampling {num_samples} samples from test data...")
        samples = sample_test_data(test_data, num_samples, min_length)
        logger.info(f"Sampled {len(samples)} valid samples")

        predictor = DTPredictor(model, action_vocab, device, context_window)

        logger.info("Running predictions...")
        metrics, examples = evaluate_predictions(samples, predictor, k, context_window)

        id_to_action = {v: k for k, v in action_vocab.items()}
        print_results(
            mode, k, len(samples), context_window, metrics, examples, id_to_action
        )

        output_dir = ROOT_DIR / "output" / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"prediction_{mode}_results.txt"

        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Prediction Results - {mode}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Mode: {mode} (K={k})\n")
            f.write(f"  Samples: {len(samples)}\n")
            f.write(f"  Context Window: {context_window}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Overall Accuracy: {metrics['overall_accuracy'] * 100:.2f}%\n")
            f.write(f"  Top-3 Accuracy: {metrics['overall_top3'] * 100:.2f}%\n\n")
            f.write(f"Step-wise Accuracy:\n")
            for step in range(k):
                if step in metrics["step_accuracy"]:
                    f.write(
                        f"  Step {step + 1}: {metrics['step_accuracy'][step] * 100:.2f}%\n"
                    )

        logger.info(f"Results saved to {result_file}")

    else:
        logger.error(f"Unsupported model: {model_name}")
        sys.exit(1)

    logger.info("Prediction completed!")


if __name__ == "__main__":
    main()
