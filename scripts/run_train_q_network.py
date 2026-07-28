#!/usr/bin/env python
"""
Q-Network Training Script
Trains Q(s, a) to predict cumulative rewards
"""

import logging
import sys
import random
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader
from src.models.QNetwork import QNetwork, QNetworkTrainer

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
        return (sample["state"], sample["action"], sample["q_value"])


def prepare_q_data(dt_data, r_log):
    """
    Prepare training data for Q-Network
    Q(s_t, a_t) = sum of rewards from t to end
    """
    samples = []

    for episode_idx in range(len(dt_data["states"])):
        states = dt_data["states"][episode_idx]
        actions = dt_data["actions"][episode_idx]
        rewards = r_log[episode_idx]

        for t in range(len(states)):
            q_value = sum(rewards[t:])
            samples.append(
                {
                    "state": states[t],
                    "action": actions[t],
                    "q_value": float(q_value),
                    "episode_idx": episode_idx,
                    "timestep": t,
                }
            )

    return samples


def train_q_network(cfg, data_loader):
    """Train Q-Network"""

    model_cfg = cfg.get("model", {}).get("training", {})
    epochs = model_cfg.get("epochs", 30)
    batch_size = model_cfg.get("batch_size", 64)
    lr = model_cfg.get("lr", 1e-4)

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    state_dim = len(data_loader.state_node_dict)
    action_dim = len(data_loader.action_vocab)

    dt_data = data_loader.dt_data
    r_log = data_loader.r_log

    logger.info("Preparing Q-Network training data...")
    all_samples = prepare_q_data(dt_data, r_log)
    logger.info(f"Total samples: {len(all_samples)}")

    q_values = [s["q_value"] for s in all_samples]
    logger.info(
        f"Q-value stats: min={min(q_values):.2f}, max={max(q_values):.2f}, mean={np.mean(q_values):.2f}"
    )

    train_samples, test_samples = train_test_split(
        all_samples, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    train_dataset = QDataset(train_samples)
    test_dataset = QDataset(test_samples)

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = QNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=128)
    model.to(device)

    trainer = QNetworkTrainer(model, lr=lr, weight_decay=0.01)

    cache_dir = ROOT_DIR / "cache" / "model"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "q_network.pth"

    best_test_loss = float("inf")

    logger.info(f"Starting Q-Network training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for states, actions, q_targets in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            q_targets = q_targets.float().to(device)

            loss = trainer.train_step(states, actions, q_targets)
            train_losses.append(loss)

        avg_train_loss = np.mean(train_losses)

        # Evaluation
        test_losses = []
        model.eval()
        with torch.no_grad():
            for states, actions, q_targets in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                q_targets = q_targets.float().to(device)

                loss, _ = trainer.evaluate(states, actions, q_targets)
                test_losses.append(loss)

        avg_test_loss = np.mean(test_losses)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "hidden_dim": 128,
                    },
                    "epoch": epoch,
                    "test_loss": avg_test_loss,
                },
                model_path,
            )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test Loss: {avg_test_loss:.4f} | "
                f"Best: {best_test_loss:.4f}"
            )

    logger.info(f"Q-Network training completed. Best test loss: {best_test_loss:.4f}")
    logger.info(f"Model saved to: {model_path}")

    return model, best_test_loss


def main():
    cfg = get_config()
    set_seed(cfg.get("seed", 42))

    logger.info("Loading data...")
    data_loader = DataLoader(cfg)

    logger.info("Training Q-Network...")
    model, test_loss = train_q_network(cfg, data_loader)

    logger.info("Q-Network training completed!")


if __name__ == "__main__":
    main()
