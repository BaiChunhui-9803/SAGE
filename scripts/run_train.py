#!/usr/bin/env python
"""
Main training script for PredictionRTS
Usage:
    python scripts/run_train.py
"""

import logging
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader as TorchDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import set_seed, ROOT_DIR, get_config
from src.data.loader import DataLoader
from src.utils.path_utils import get_output_paths
from src.utils.model_utils import DTDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_model_config(cfg):
    """Get model configuration from config dict."""
    if hasattr(cfg, "model"):
        return cfg.model
    if isinstance(cfg, dict):
        return cfg.get("model", {})
    return {}


def train_decision_transformer_full(cfg, data_loader):
    from src.models.DecisionTransformer import DecisionTransformer

    model_cfg = get_model_config(cfg)
    training_cfg = model_cfg.get("training", {})

    # 获取训练参数
    d_model = model_cfg.get("d_model", 128)
    epochs = training_cfg.get("epochs", 50)
    batch_size = training_cfg.get("batch_size", 64)
    lr = training_cfg.get("lr", 1e-4)
    context_window = model_cfg.get("max_len", 20)

    # 设备
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 获取数据
    state_dim = len(data_loader.state_node_dict)
    dt_data = data_loader.dt_data
    action_vocab = data_loader.action_vocab
    act_vocab_size = len(action_vocab)
    state_distance_matrix = data_loader.state_distance_matrix
    r_log = data_loader.r_log

    logger.info(f"State dim: {state_dim}, Action vocab size: {act_vocab_size}")
    logger.info(f"Total episodes: {len(dt_data['states'])}")

    # 计算 RTG 分布
    traj_returns = [sum(traj) for traj in r_log]
    rtg_stats = {
        "max": float(np.max(traj_returns)),
        "min": float(np.min(traj_returns)),
        "mean": float(np.mean(traj_returns)),
        "p90": float(np.percentile(traj_returns, 90)),
        "std": float(np.std(traj_returns)),
    }
    logger.info(
        f"RTG stats: max={rtg_stats['max']:.2f}, mean={rtg_stats['mean']:.2f}, std={rtg_stats['std']:.2f}"
    )

    # 划分数据集
    indices = np.arange(len(dt_data["states"]))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_data = {k: [v[i] for i in train_idx] for k, v in dt_data.items()}
    test_data = {k: [v[i] for i in test_idx] for k, v in dt_data.items()}

    logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # 创建数据集和数据加载器
    train_dataset = DTDataset(train_data, context_window=context_window)
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DTDataset(test_data, context_window=context_window)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = DecisionTransformer(
        state_dim=state_dim,
        act_vocab_size=act_vocab_size,
        n_layer=model_cfg.get("n_layer", 4),
        n_head=model_cfg.get("n_head", 4),
        n_embd=d_model,
        max_len=2048,
    ).to(device)

    logger.info(f"Model created: DecisionTransformer")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # MDS 初始化 (可选)
    params = model_cfg.get("params", {})
    if params.get("mdl_init_embedding_freeze") or params.get(
        "mdl_init_embedding_train"
    ):
        from src.train.init import init_dt_state_embedding

        init_dt_state_embedding(model, state_distance_matrix, d_model)
        if params.get("mdl_init_embedding_freeze"):
            model.embed_state.weight.requires_grad = False
            logger.info("MDS State Embedding fixed.")

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    loss_history = []
    best_test_loss = float("inf")

    cache_dir = ROOT_DIR / "cache" / "model"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "best_model.pth"

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for states, actions, rtgs, timesteps in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)

            optimizer.zero_grad()

            action_logits = model(states, actions, rtgs, timesteps)

            loss = criterion(
                action_logits.view(-1, action_logits.size(-1)), actions.view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_cfg.get("grad_clip", 1.0)
            )
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # 验证
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for states, actions, rtgs, timesteps in test_loader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)

                action_logits = model(states, actions, rtgs, timesteps)

                loss = criterion(
                    action_logits.view(-1, action_logits.size(-1)), actions.view(-1)
                )

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)

        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "action_vocab": action_vocab,
                    "rtg_stats": rtg_stats,
                    "config": cfg,
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                },
                model_path,
            )

        # 日志
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test Loss: {avg_test_loss:.4f} | "
                f"Best: {best_test_loss:.4f}"
            )

    logger.info(f"Training completed. Best test loss: {best_test_loss:.4f}")
    logger.info(f"Model saved to: {model_path}")

    return model, loss_history


def train_trajectory_transformer_full(cfg, data_loader):
    from src.models.Transformer import TrajectoryTransformer

    model_cfg = get_model_config(cfg)
    training_cfg = model_cfg.get("training", {})

    vocab_size = len(data_loader.state_node_dict)
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )

    model = TrajectoryTransformer(
        vocab_size=vocab_size,
        d_model=model_cfg.get("d_model", 128),
        nhead=model_cfg.get("n_head", 8),
        num_layers=model_cfg.get("n_layer", 4),
        max_len=model_cfg.get("max_len", 512),
    ).to(device)

    logger.info(f"TrajectoryTransformer created with vocab_size={vocab_size}")

    return model


def main():
    cfg = get_config()

    set_seed(cfg.get("seed", 42) if isinstance(cfg, dict) else 42)

    data_loader = DataLoader(cfg)

    model_cfg = get_model_config(cfg)
    model_name = (
        model_cfg.get("name", "DecisionTransformer")
        if isinstance(model_cfg, dict)
        else "DecisionTransformer"
    )

    logger.info(f"Training {model_name}...")
    logger.info(f"Config: {cfg}")

    if model_name == "DecisionTransformer":
        model, loss_history = train_decision_transformer_full(cfg, data_loader)
    elif model_name in ["TrajectoryTransformer", "trajTransformer"]:
        model = train_trajectory_transformer_full(cfg, data_loader)
    else:
        logger.warning(f"Unknown model: {model_name}, using DecisionTransformer")
        model, loss_history = train_decision_transformer_full(cfg, data_loader)

    output_paths = get_output_paths(cfg)
    logger.info(f"Output paths: {output_paths}")
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
