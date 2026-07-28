"""
Base Trainer class for PredictionRTS
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(self, model: nn.Module, cfg: Dict[str, Any], device: str = "cuda"):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = cfg.get("training", {}).get("epochs", 20)
        self.batch_size = cfg.get("training", {}).get("batch_size", 32)
        self.lr = cfg.get("training", {}).get("lr", 1e-4)
        self.weight_decay = cfg.get("training", {}).get("weight_decay", 0.01)
        self.grad_clip = cfg.get("training", {}).get("grad_clip", 1.0)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.train_losses = []
        self.val_losses = []

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> float:
        pass

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> float:
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[Path] = None,
        save_every: int = 10,
    ) -> Dict[str, Any]:
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_dir:
                        self.save_checkpoint(save_dir / "best_model.pth")

                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}"
                )

            if save_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch + 1}.pth")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
        }

    def save_checkpoint(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            filepath,
        )
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: Path):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Checkpoint loaded from {filepath}")
