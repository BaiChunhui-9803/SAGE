#!/usr/bin/env python
"""
Decision Transformer Training Script with Context Window Support

Usage:
    python scripts/run_train_dt.py --context 20 --epochs 50
    python scripts/run_train_dt.py --context 5 --epochs 30 --output dt_ctx5.pth
"""

import argparse
import logging
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader as DataLoaderClass
from src.models.DecisionTransformer import DecisionTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DTDatasetWithContext(Dataset):
    """Dataset for DT with specific context window"""

    def __init__(self, samples, context_window, pad_value=0):
        self.samples = samples
        self.context_window = context_window
        self.pad_value = pad_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        history_states = sample["history_states"]
        history_actions = sample["history_actions"]
        history_rtgs = sample["history_rtgs"]
        target_action = sample["target_actions"][0]  # Single step prediction

        # Pad to context_window if needed
        if len(history_states) < self.context_window:
            pad_len = self.context_window - len(history_states)
            history_states = [self.pad_value] * pad_len + history_states
            history_actions = [self.pad_value] * pad_len + history_actions
            history_rtgs = [0.0] * pad_len + history_rtgs
        else:
            # Use last context_window
            history_states = history_states[-self.context_window :]
            history_actions = history_actions[-self.context_window :]
            history_rtgs = history_rtgs[-self.context_window :]

        return {
            "states": torch.tensor(history_states, dtype=torch.long),
            "actions": torch.tensor(history_actions, dtype=torch.long),
            "rtgs": torch.tensor(history_rtgs, dtype=torch.float),
            "target": torch.tensor(target_action, dtype=torch.long),
        }


def prepare_samples(dt_data, r_log, context_window, min_history=1):
    """Prepare training samples for specific context window"""
    samples = []

    for episode_idx in range(len(dt_data["states"])):
        states = dt_data["states"][episode_idx]
        actions = dt_data["actions"][episode_idx]
        rtgs = dt_data["rtgs"][episode_idx]
        rewards = r_log[episode_idx] if episode_idx < len(r_log) else []

        # Need at least min_history steps
        if len(states) < min_history + 1:
            continue

        for t in range(min_history, len(states)):
            # Use up to context_window history
            start_t = max(0, t - context_window)

            history_states = states[start_t:t]
            history_actions = actions[start_t:t]
            history_rtgs = rtgs[start_t:t]

            # Target is the action at time t
            target_action = actions[t]

            samples.append(
                {
                    "episode_idx": episode_idx,
                    "timestep": t,
                    "history_states": history_states,
                    "history_actions": history_actions,
                    "history_rtgs": history_rtgs,
                    "target_actions": [target_action],
                    "q_value": sum(rewards[t:]) if rewards else 0,
                }
            )

    return samples


def train_dt(args):
    """Train Decision Transformer with specific context window"""

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Context window: {args.context}")

    # Load data
    cfg = get_config()
    data_loader = DataLoaderClass(cfg)

    dt_data = data_loader.dt_data
    r_log = data_loader.r_log
    action_vocab = data_loader.action_vocab

    state_dim = len(data_loader.state_node_dict)
    action_dim = len(action_vocab)

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Prepare samples
    logger.info("Preparing training samples...")
    all_samples = prepare_samples(
        dt_data, r_log, context_window=args.context, min_history=args.min_history
    )
    logger.info(f"Total samples: {len(all_samples)}")

    if len(all_samples) == 0:
        logger.error("No samples generated! Check min_history parameter.")
        return

    # Split
    train_samples, test_samples = train_test_split(
        all_samples, test_size=0.2, random_state=args.seed
    )
    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Create datasets
    train_dataset = DTDatasetWithContext(train_samples, args.context)
    test_dataset = DTDatasetWithContext(test_samples, args.context)

    train_loader = TorchDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_vocab_size=action_dim,
        n_layer=4,
        n_head=4,
        n_embd=128,
        max_len=args.context,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Output path
    output_name = args.output if args.output else f"dt_ctx{args.context}.pth"
    output_path = ROOT_DIR / "cache" / "model" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_test_loss = float("inf")

    logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtgs = batch["rtgs"].to(device)
            targets = batch["target"].to(device)

            timesteps = (
                torch.arange(states.size(1))
                .unsqueeze(0)
                .expand(states.size(0), -1)
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(states, actions, rtgs, timesteps)

            # Predict last position
            loss = criterion(logits[:, -1, :], targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(logits[:, -1, :], dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rtgs = batch["rtgs"].to(device)
                targets = batch["target"].to(device)

                timesteps = (
                    torch.arange(states.size(1))
                    .unsqueeze(0)
                    .expand(states.size(0), -1)
                    .to(device)
                )

                logits = model(states, actions, rtgs, timesteps)
                loss = criterion(logits[:, -1, :], targets)

                test_loss += loss.item()

                preds = torch.argmax(logits[:, -1, :], dim=-1)
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "context_window": args.context,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "n_embd": args.n_embd,
                    },
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                },
                output_path,
            )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
                f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc * 100:.2f}%"
            )

    logger.info(f"Training completed. Best test loss: {best_test_loss:.4f}")
    logger.info(f"Model saved to: {output_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")

    # Model parameters
    parser.add_argument("--context", type=int, default=20, help="Context window size")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Data parameters
    parser.add_argument(
        "--min-history", type=int, default=1, help="Minimum history length for sample"
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None, help="Output model filename"
    )

    args = parser.parse_args()

    train_dt(args)


if __name__ == "__main__":
    main()
