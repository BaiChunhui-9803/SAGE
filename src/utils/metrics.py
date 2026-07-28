"""
Metrics utilities for PredictionRTS
"""

import numpy as np
from typing import List, Dict, Any


def calculate_accuracy(predictions: List, targets: List) -> float:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions) if predictions else 0.0


def calculate_stepwise_metrics(
    pred_seq: List[int], true_seq: List[int], state_distance_matrix: np.ndarray
) -> Dict[str, float]:
    metrics = {
        "exact_match": 0.0,
        "avg_distance": 0.0,
        "max_distance": 0.0,
        "min_distance": 0.0,
    }

    if not pred_seq or not true_seq:
        return metrics

    exact_matches = sum(p == t for p, t in zip(pred_seq, true_seq))
    metrics["exact_match"] = exact_matches / len(pred_seq)

    distances = []
    for p, t in zip(pred_seq, true_seq):
        if p < len(state_distance_matrix) and t < len(state_distance_matrix):
            distances.append(state_distance_matrix[p][t])

    if distances:
        metrics["avg_distance"] = np.mean(distances)
        metrics["max_distance"] = np.max(distances)
        metrics["min_distance"] = np.min(distances)

    return metrics


def calculate_dtw_distance(
    seq1: List[int], seq2: List[int], state_distance_matrix: np.ndarray
) -> float:
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = state_distance_matrix[seq1[i - 1]][seq2[j - 1]]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m]
