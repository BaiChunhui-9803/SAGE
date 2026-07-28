#!/usr/bin/env python
"""
Complete Decision System Training and Evaluation Script
Trains Q-Network and State Predictor, then evaluates decision quality

Usage:
    python scripts/run_decision_system.py --train_all
    python scripts/run_decision_system.py --evaluate_only
"""

import argparse
import logging
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader as DataLoaderClass
from src.models.QNetwork import QNetwork
from src.models.StateTransitionPredictor import StateTransitionPredictor
from src.models.DecisionTransformer import DecisionTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["state"], sample["action"], sample["q_value"]


class StateDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["state"], sample["action"], sample["next_state"]


def prepare_q_data(dt_data, r_log):
    samples = []
    for episode_idx in range(len(dt_data["states"])):
        states = dt_data["states"][episode_idx]
        actions = dt_data["actions"][episode_idx]
        rewards = r_log[episode_idx]
        for t in range(len(states)):
            q_value = sum(rewards[t:])
            samples.append(
                {"state": states[t], "action": actions[t], "q_value": float(q_value)}
            )
    return samples


def prepare_state_data(dt_data):
    samples = []
    for episode_idx in range(len(dt_data["states"])):
        states = dt_data["states"][episode_idx]
        actions = dt_data["actions"][episode_idx]
        for t in range(len(states) - 1):
            samples.append(
                {"state": states[t], "action": actions[t], "next_state": states[t + 1]}
            )
    return samples


def train_q_network(data_loader, cfg, device, epochs=30, batch_size=64, lr=1e-4):
    logger.info("=" * 60)
    logger.info("Training Q-Network")
    logger.info("=" * 60)

    state_dim = len(data_loader.state_node_dict)
    action_dim = len(data_loader.action_vocab)
    dt_data = data_loader.dt_data
    r_log = data_loader.r_log

    logger.info("Preparing Q-Network training data...")
    q_samples = prepare_q_data(dt_data, r_log)

    train_samples, test_samples = train_test_split(
        q_samples, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    train_dataset = QDataset(train_samples)
    test_dataset = QDataset(test_samples)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size)

    model = QNetwork(state_dim, action_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for states, actions, q_targets in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            q_targets = q_targets.float().to(device)

            optimizer.zero_grad()
            q_preds = model(states, actions)
            loss = criterion(q_preds, q_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for states, actions, q_targets in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                q_targets = q_targets.float().to(device)
                q_preds = model(states, actions)
                loss = criterion(q_preds, q_targets)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = ROOT_DIR / "cache" / "model" / "q_network.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "epoch": epoch,
                    "test_loss": avg_test_loss,
                },
                save_path,
            )

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
            )

    logger.info(f"Q-Network training completed. Best test loss: {best_test_loss:.4f}")
    return model


def train_state_predictor(data_loader, cfg, device, epochs=30, batch_size=64, lr=1e-4):
    logger.info("=" * 60)
    logger.info("Training State Transition Predictor")
    logger.info("=" * 60)

    state_dim = len(data_loader.state_node_dict)
    action_dim = len(data_loader.action_vocab)
    dt_data = data_loader.dt_data

    logger.info("Preparing State Predictor training data...")
    state_samples = prepare_state_data(dt_data)

    train_samples, test_samples = train_test_split(
        state_samples, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    train_dataset = StateDataset(train_samples)
    test_dataset = StateDataset(test_samples)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size)

    model = StateTransitionPredictor(state_dim, action_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for states, actions, next_states in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)

            optimizer.zero_grad()
            logits = model(states, actions)
            loss = criterion(logits, next_states)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for states, actions, next_states in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                next_states = next_states.to(device)
                logits = model(states, actions)
                loss = criterion(logits, next_states)
                test_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == next_states).sum().item()
                total += next_states.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total if total > 0 else 0

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = ROOT_DIR / "cache" / "model" / "state_predictor.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "epoch": epoch,
                    "test_loss": avg_test_loss,
                    "accuracy": accuracy,
                },
                save_path,
            )

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {accuracy * 100:.2f}%"
            )

    logger.info(
        f"State Predictor training completed. Best accuracy: {accuracy * 100:.2f}%"
    )
    return model


def evaluate_decision_system(
    dt_model,
    q_network,
    state_predictor,
    data_loader,
    device,
    num_samples=100,
    context_window=20,
):
    logger.info("=" * 60)
    logger.info("Evaluating Decision System")
    logger.info("=" * 60)

    dt_data = data_loader.dt_data
    r_log = data_loader.r_log
    action_vocab = data_loader.action_vocab
    id_to_action = {v: k for k, v in action_vocab.items()}

    test_size = 0.2
    test_start = int(len(dt_data["states"]) * (1 - test_size))
    test_data = {
        "states": dt_data["states"][test_start:],
        "actions": dt_data["actions"][test_start:],
        "rtgs": dt_data["rtgs"][test_start:],
    }

    valid_samples = []
    for i, (s, a, r) in enumerate(
        zip(test_data["states"], test_data["actions"], test_data["rtgs"])
    ):
        if len(s) >= context_window + 1:
            valid_samples.append(i)

    random.seed(42)
    sample_indices = random.sample(valid_samples, min(num_samples, len(valid_samples)))

    results = []
    metrics = defaultdict(lambda: defaultdict(int))

    for idx in sample_indices:
        states = test_data["states"][idx]
        actions = test_data["actions"][idx]
        rtgs = test_data["rtgs"][idx]

        cutoff = context_window
        history_states = states[:cutoff]
        history_actions = actions[:cutoff]
        history_rtgs = rtgs[:cutoff]

        true_action = actions[cutoff]
        current_state = history_states[-1]

        dt_model.eval()
        with torch.no_grad():
            s_tensor = torch.tensor([history_states], dtype=torch.long).to(device)
            a_tensor = torch.tensor([history_actions], dtype=torch.long).to(device)
            r_tensor = torch.tensor([history_rtgs], dtype=torch.float).to(device)
            t_tensor = torch.arange(len(history_states)).unsqueeze(0).to(device)

            dt_logits = dt_model(s_tensor, a_tensor, r_tensor, t_tensor)
            dt_probs = F.softmax(dt_logits[0, -1, :], dim=-1)
            dt_pred_action = torch.argmax(dt_probs).item()

        topk_probs, topk_indices = torch.topk(dt_probs, 3)
        dt_top3_actions = topk_indices.cpu().numpy().tolist()

        q_network.eval()
        q_values = {}
        with torch.no_grad():
            state_t = torch.tensor([current_state], dtype=torch.long).to(device)
            for a in range(len(action_vocab)):
                action_t = torch.tensor([a], dtype=torch.long).to(device)
                q_values[a] = q_network(state_t, action_t).item()

        q_pred_action = max(q_values, key=q_values.get)

        dt_weight = 0.3
        q_weight = 0.7
        q_array = np.array([q_values[a] for a in dt_top3_actions])
        q_min, q_max = q_array.min(), q_array.max()
        q_norm = (
            (q_array - q_min) / (q_max - q_min + 1e-8)
            if q_max > q_min
            else np.ones_like(q_array)
        )
        dt_array = topk_probs.cpu().numpy()
        combined = dt_weight * dt_array + q_weight * q_norm
        hybrid_action = dt_top3_actions[np.argmax(combined)]

        state_predictor.eval()
        with torch.no_grad():
            state_t = torch.tensor([current_state], dtype=torch.long).to(device)
            for pred_action, name in [
                (dt_pred_action, "dt"),
                (q_pred_action, "q"),
                (hybrid_action, "hybrid"),
            ]:
                action_t = torch.tensor([pred_action], dtype=torch.long).to(device)
                logits = state_predictor(state_t, action_t)
                pred_state = torch.argmax(logits, dim=-1).item()

                if cutoff + 1 < len(states):
                    true_next_state = states[cutoff + 1]
                    state_match = pred_state == true_next_state
                else:
                    state_match = None

                metrics[name]["q_pred"] += q_values[pred_action]
                metrics[name]["q_true"] += q_values[true_action]
                metrics[name]["correct"] += pred_action == true_action
                metrics[name]["total"] += 1

        result = {
            "sample_idx": idx,
            "current_state": current_state,
            "true_action": true_action,
            "true_action_name": id_to_action.get(true_action, str(true_action)),
            "dt_pred": dt_pred_action,
            "dt_pred_name": id_to_action.get(dt_pred_action, str(dt_pred_action)),
            "q_pred": q_pred_action,
            "q_pred_name": id_to_action.get(q_pred_action, str(q_pred_action)),
            "hybrid_pred": hybrid_action,
            "hybrid_pred_name": id_to_action.get(hybrid_action, str(hybrid_action)),
            "dt_top3": [id_to_action.get(a, str(a)) for a in dt_top3_actions],
            "q_values": {id_to_action.get(a, str(a)): q for a, q in q_values.items()},
            "dt_prob": {
                id_to_action.get(a, str(a)): float(p)
                for a, p in zip(dt_top3_actions, dt_array)
            },
        }
        results.append(result)

    logger.info("\n" + "=" * 80)
    logger.info("DECISION SYSTEM EVALUATION RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Test samples: {len(results)}")
    logger.info(f"  - Context window: {context_window}")
    logger.info(f"  - DT weight: {dt_weight}, Q weight: {q_weight}")

    logger.info(f"\nOverall Results:")
    for name in ["dt", "q", "hybrid"]:
        acc = metrics[name]["correct"] / metrics[name]["total"] * 100
        avg_q_pred = metrics[name]["q_pred"] / metrics[name]["total"]
        avg_q_true = metrics[name]["q_true"] / metrics[name]["total"]
        logger.info(
            f"  {name.upper():8s}: Acc={acc:.2f}% | Avg Q(pred)={avg_q_pred:.2f} | Avg Q(true)={avg_q_true:.2f}"
        )

    logger.info(f"\nSample Details (first 10):")
    for i, r in enumerate(results[:10]):
        logger.info(f"\n  Sample #{r['sample_idx']}:")
        logger.info(f"    State: {r['current_state']}")
        logger.info(f"    True Action: {r['true_action_name']}")
        logger.info(f"    DT Prediction: {r['dt_pred_name']} (Top3: {r['dt_top3']})")
        logger.info(f"    Q Prediction: {r['q_pred_name']}")
        logger.info(f"    Hybrid Prediction: {r['hybrid_pred_name']}")
        logger.info(f"    Q Values: {r['q_values']}")

    output_dir = ROOT_DIR / "output" / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "decision_system_results.txt"

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Decision System Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Samples: {len(results)}\n")
        f.write(f"  Context Window: {context_window}\n\n")
        f.write(f"Results:\n")
        for name in ["dt", "q", "hybrid"]:
            acc = metrics[name]["correct"] / metrics[name]["total"] * 100
            f.write(f"  {name.upper()}: Accuracy={acc:.2f}%\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("\nSample Details:\n\n")
        for r in results:
            f.write(f"Sample #{r['sample_idx']}:\n")
            f.write(f"  True: {r['true_action_name']}\n")
            f.write(
                f"  DT: {r['dt_pred_name']} | Q: {r['q_pred_name']} | Hybrid: {r['hybrid_pred_name']}\n"
            )
            f.write(f"  Q Values: {r['q_values']}\n\n")

    logger.info(f"\nDetailed results saved to: {result_file}")

    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Decision System Training and Evaluation"
    )
    parser.add_argument("--train_all", action="store_true", help="Train all models")
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Skip training, only evaluate"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--samples", type=int, default=100, help="Evaluation samples")
    args = parser.parse_args()

    cfg = get_config()
    set_seed(cfg.get("seed", 42))
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    data_loader = DataLoaderClass(cfg)

    if args.train_all:
        q_network = train_q_network(data_loader, cfg, device, epochs=args.epochs)
        state_predictor = train_state_predictor(
            data_loader, cfg, device, epochs=args.epochs
        )
    else:
        q_path = ROOT_DIR / "cache" / "model" / "q_network.pth"
        sp_path = ROOT_DIR / "cache" / "model" / "state_predictor.pth"
        dt_path = ROOT_DIR / "cache" / "model" / "best_model.pth"

        if not q_path.exists():
            logger.error(
                f"Q-Network not found at {q_path}. Run with --train_all first."
            )
            return
        if not sp_path.exists():
            logger.error(
                f"State Predictor not found at {sp_path}. Run with --train_all first."
            )
            return

        state_dim = len(data_loader.state_node_dict)
        action_dim = len(data_loader.action_vocab)

        q_network = QNetwork(state_dim, action_dim).to(device)
        q_checkpoint = torch.load(q_path, map_location="cpu")
        q_network.load_state_dict(q_checkpoint["model_state_dict"])
        logger.info("Loaded Q-Network")

        state_predictor = StateTransitionPredictor(state_dim, action_dim).to(device)
        sp_checkpoint = torch.load(sp_path, map_location="cpu")
        state_predictor.load_state_dict(sp_checkpoint["model_state_dict"])
        logger.info("Loaded State Predictor")

    dt_model = DecisionTransformer(
        state_dim=len(data_loader.state_node_dict),
        act_vocab_size=len(data_loader.action_vocab),
        n_layer=4,
        n_head=4,
        n_embd=128,
        max_len=2048,
    ).to(device)
    dt_checkpoint = torch.load(
        ROOT_DIR / "cache" / "model" / "best_model.pth", map_location="cpu"
    )
    dt_model.load_state_dict(dt_checkpoint["model_state_dict"])
    logger.info("Loaded Decision Transformer")

    evaluate_decision_system(
        dt_model,
        q_network,
        state_predictor,
        data_loader,
        device,
        num_samples=args.samples,
    )

    logger.info("\nDecision System Evaluation Complete!")


if __name__ == "__main__":
    main()
