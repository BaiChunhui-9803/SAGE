"""
Custom state distance calculations
Migrated from structure/state_distance/
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def multi_distance(
    unit1: Dict, unit2: Dict, weights: Dict[str, float] = None
) -> Dict[str, float]:
    if weights is None:
        weights = {"position": 1.0, "health": 0.5, "type": 0.3}

    distances = {}

    if "x" in unit1 and "y" in unit1 and "x" in unit2 and "y" in unit2:
        pos_dist = euclidean_distance(
            (unit1["x"], unit1["y"]), (unit2["x"], unit2["y"])
        )
        distances["position"] = pos_dist * weights.get("position", 1.0)

    if "hp" in unit1 and "hp" in unit2:
        hp_dist = abs(unit1["hp"] - unit2["hp"])
        distances["health"] = hp_dist * weights.get("health", 0.5)

    return distances


def custom_distance(state1: Dict, state2: Dict) -> float:
    if "state" not in state1 or "state" not in state2:
        return 0.0

    units1 = state1["state"]
    units2 = state2["state"]

    if not units1 or not units2:
        return 0.0

    if len(units1) != len(units2):
        return float("inf")

    n = len(units1)
    cost_matrix = np.zeros((n, n))

    for i, u1 in enumerate(units1):
        for j, u2 in enumerate(units2):
            dists = multi_distance(u1, u2)
            cost_matrix[i, j] = sum(dists.values())

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost / n if n > 0 else 0.0


def hungarian_distance(units1: List[Dict], units2: List[Dict]) -> float:
    if not units1 or not units2:
        return 0.0

    n1, n2 = len(units1), len(units2)
    n = max(n1, n2)

    cost_matrix = np.full((n, n), 1e6)

    for i, u1 in enumerate(units1):
        for j, u2 in enumerate(units2):
            dists = multi_distance(u1, u2)
            cost_matrix[i, j] = sum(dists.values())

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost / max(n1, n2) if max(n1, n2) > 0 else 0.0


def calculate_state_distance_matrix(
    states: List[Dict], distance_func: callable = custom_distance
) -> np.ndarray:
    n = len(states)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_func(states[i], states[j])
            matrix[i, j] = dist
            matrix[j, i] = dist

    return matrix
