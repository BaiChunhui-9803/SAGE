"""
Visualization utilities for PredictionRTS
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def plot_fitness_landscape(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Fitness Landscape",
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(x, y, z, cmap="viridis", alpha=0.8)
    ax.set_xlabel("X (MDS)")
    ax.set_ylabel("Y (MDS)")
    ax.set_zlabel("Fitness")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_state_landscape(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "State Landscape",
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(x, y, z, cmap="plasma", alpha=0.8)
    ax.set_xlabel("X (MDS)")
    ax.set_ylabel("Y (MDS)")
    ax.set_zlabel("State Value")
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_loss(
    losses: List[float], save_path: Optional[Path] = None, title: str = "Training Loss"
):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_trajectory_comparison(
    trajectories: List[List[int]],
    pred_trajectory: List[int],
    state_distance_matrix: np.ndarray,
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    for i, traj in enumerate(trajectories):
        ax1.plot(range(len(traj)), traj, alpha=0.5, label=f"Traj {i + 1}")
    ax1.plot(
        range(len(pred_trajectory)),
        pred_trajectory,
        "r-",
        linewidth=2,
        label="Predicted",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("State ID")
    ax1.set_title("Trajectory Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    distances = []
    for traj in trajectories:
        dtw_dist = 0
        min_len = min(len(traj), len(pred_trajectory))
        for i in range(min_len):
            dtw_dist += state_distance_matrix[traj[i]][pred_trajectory[i]]
        distances.append(dtw_dist / min_len if min_len > 0 else 0)

    ax2.bar(range(len(distances)), distances)
    ax2.set_xlabel("Trajectory")
    ax2.set_ylabel("Avg Distance")
    ax2.set_title("Average Distance from Prediction")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
