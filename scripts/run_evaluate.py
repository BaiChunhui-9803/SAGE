#!/usr/bin/env python
"""
Evaluation script for PredictionRTS
Usage:
    python scripts/run_evaluate.py model=decision_transformer checkpoint=best_model.pth
"""

import logging
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR
from src.data.loader import DataLoader
from src.utils.path_utils import get_output_paths
from src.utils.metrics import calculate_accuracy, calculate_stepwise_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_decision_transformer(model, data_loader, device):
    dt_data = data_loader.dt_data
    action_vocab = data_loader.action_vocab

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        states_list = dt_data["states"]
        actions_list = dt_data["actions"]
        rtgs_list = dt_data["rtgs"]

        for i in range(len(states_list)):
            states = torch.tensor([states_list[i]], dtype=torch.long).to(device)
            actions = torch.tensor([actions_list[i]], dtype=torch.long).to(device)
            rtgs = torch.tensor([rtgs_list[i]], dtype=torch.float).to(device)
            seq_len = states.shape[1]
            timesteps = torch.arange(seq_len).unsqueeze(0).to(device)

            logits = model(states, actions, rtgs, timesteps)
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
            targets = actions.squeeze().cpu().numpy()

            correct += (preds == targets).sum()
            total += len(targets)

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Decision Transformer Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.from_preconfig(ROOT_DIR / "configs")
    except ImportError:
        from src import get_config

        cfg = get_config()

    set_seed(cfg.get("seed", 42))
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    data_loader = DataLoader(cfg)

    model_name = cfg.get("model", {}).get("name", "DecisionTransformer")
    checkpoint_name = cfg.get("checkpoint_name", "best_model.pth")
    checkpoint_path = ROOT_DIR / "cache" / "model" / checkpoint_name

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

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
    else:
        logger.error(f"Unknown model: {model_name}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info(f"Evaluating {model_name}...")

    if model_name == "DecisionTransformer":
        accuracy = evaluate_decision_transformer(model, data_loader, device)
        logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
