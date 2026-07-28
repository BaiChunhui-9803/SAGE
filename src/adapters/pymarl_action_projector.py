"""Project PyMARL/SMAC primitive joint actions to ETG script actions.

The ETG runtime currently consumes compact script codes such as ``4c``.  PyMARL
policies instead emit one primitive action per agent.  This adapter keeps the
ETG-facing action space stable by assigning each micro-action pattern to the
closest existing script action, while preserving confidence and diagnostic
metadata for later analysis.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_ACTIONS: Dict[str, str] = {
    "a": "action_ATK_nearest",
    "b": "action_ATK_clu_nearest",
    "c": "action_ATK_nearest_weakest",
    "d": "action_ATK_clu_nearest_weakest",
    "e": "action_ATK_threatening",
    "f": "action_DEF_clu_nearest",
    "g": "action_MIX_gather",
    "h": "action_MIX_lure",
    "i": "action_MIX_sacrifice_lure",
    "j": "do_randomly",
    "k": "do_nothing",
}

SMAC_NO_OP = 0
SMAC_STOP = 1
SMAC_MOVE_NORTH = 2
SMAC_MOVE_SOUTH = 3
SMAC_MOVE_EAST = 4
SMAC_MOVE_WEST = 5
SMAC_ATTACK_OFFSET = 6


@dataclass
class ProjectionResult:
    action_code: str
    action_letter: str
    cluster_digit: int
    confidence: float
    reason: str
    features: Dict[str, Any]
    micro_action_signature: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_state_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def extract_state(frame: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a normalized state dict from a teacher frame if present."""

    for key in (
        "norm_state",
        "normalized_state",
        "state_norm",
        "state",
        "raw_state",
        "attr_state",
        "state_dict",
    ):
        state = _as_state_dict(frame.get(key))
        if state is not None and (
            "blue_army" in state or "red_army" in state or "allies" in state
        ):
            return normalize_army_state(state)
    return None


def normalize_army_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common SC2/SMAC state variants to ``blue_army``/``red_army``."""

    if "blue_army" in state and "red_army" in state:
        return {
            "blue_army": [_unit_to_triplet(unit) for unit in state.get("blue_army", [])],
            "red_army": [_unit_to_triplet(unit) for unit in state.get("red_army", [])],
        }
    if "allies" in state and "enemies" in state:
        return {
            "blue_army": [_unit_to_triplet(unit) for unit in state.get("allies", [])],
            "red_army": [_unit_to_triplet(unit) for unit in state.get("enemies", [])],
        }
    return dict(state)


def _unit_to_triplet(unit: Any) -> Tuple[float, float, float]:
    if isinstance(unit, Mapping):
        x = unit.get("x", unit.get("pos_x", unit.get("x_norm", 0.0)))
        y = unit.get("y", unit.get("pos_y", unit.get("y_norm", 0.0)))
        hp = unit.get("health", unit.get("hp", unit.get("health_ratio", 0.0)))
        return (float(x), float(y), float(hp))
    if isinstance(unit, Sequence) and not isinstance(unit, (str, bytes)):
        values = list(unit)
        x = values[0] if len(values) > 0 else 0.0
        y = values[1] if len(values) > 1 else 0.0
        hp = values[2] if len(values) > 2 else 0.0
        return (float(x), float(y), float(hp))
    return (0.0, 0.0, 0.0)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _nearest_enemy_indices(allies: Sequence[Sequence[float]], enemies: Sequence[Sequence[float]]) -> List[Optional[int]]:
    result: List[Optional[int]] = []
    for ally in allies:
        if not enemies:
            result.append(None)
            continue
        result.append(min(range(len(enemies)), key=lambda idx: _distance(ally, enemies[idx])))
    return result


def _weakest_enemy_index(enemies: Sequence[Sequence[float]]) -> Optional[int]:
    if not enemies:
        return None
    return min(range(len(enemies)), key=lambda idx: float(enemies[idx][2]))


def _center(units: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not units:
        return (0.0, 0.0)
    return (
        sum(float(unit[0]) for unit in units) / len(units),
        sum(float(unit[1]) for unit in units) / len(units),
    )


def _mean_distance_to_center(units: Sequence[Sequence[float]]) -> float:
    if len(units) <= 1:
        return 0.0
    center = _center(units)
    return sum(_distance(unit, center) for unit in units) / len(units)


def _coerce_actions(actions: Any) -> List[int]:
    if actions is None:
        return []
    if hasattr(actions, "tolist"):
        actions = actions.tolist()
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except Exception:
            return []
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
        if len(actions) == 1 and isinstance(actions[0], Sequence) and not isinstance(actions[0], (str, bytes)):
            actions = actions[0]
        result: List[int] = []
        for item in actions:
            try:
                result.append(int(item))
            except Exception:
                continue
        return result
    return []


def project_pymarl_actions(
    pymarl_actions: Any,
    state: Optional[Mapping[str, Any]] = None,
    available_actions: Optional[Any] = None,
) -> ProjectionResult:
    """Project a PyMARL joint action to one ETG script code.

    The projection is intentionally conservative: when micro actions are
    ambiguous, it prefers attack scripts over random/no-op scripts unless the
    observed action pattern is dominated by explicit no-op/stop.
    """

    actions = _coerce_actions(pymarl_actions)
    norm_state = normalize_army_state(state or {})
    allies = list(norm_state.get("blue_army", []) or [])
    enemies = list(norm_state.get("red_army", []) or [])
    active_actions = actions[: len(allies)] if allies else list(actions)

    attack_targets = [
        action - SMAC_ATTACK_OFFSET
        for action in active_actions
        if int(action) >= SMAC_ATTACK_OFFSET
    ]
    attack_targets = [idx for idx in attack_targets if idx >= 0]
    move_actions = [action for action in active_actions if action in {2, 3, 4, 5}]
    stop_count = sum(1 for action in active_actions if action == SMAC_STOP)
    no_op_count = sum(1 for action in active_actions if action == SMAC_NO_OP)
    total = len(active_actions)

    target_counts = Counter(attack_targets)
    top_target, top_count = (None, 0)
    if target_counts:
        top_target, top_count = target_counts.most_common(1)[0]
    focus_ratio = _safe_ratio(top_count, len(attack_targets))
    attack_rate = _safe_ratio(len(attack_targets), total)
    move_rate = _safe_ratio(len(move_actions), total)
    stop_rate = _safe_ratio(stop_count, total)
    no_op_rate = _safe_ratio(no_op_count, total)

    weakest_idx = _weakest_enemy_index(enemies)
    weakest_agreement = _safe_ratio(
        sum(1 for target in attack_targets if target == weakest_idx),
        len(attack_targets),
    )
    nearest_indices = _nearest_enemy_indices(allies, enemies)
    nearest_agreement = 0.0
    if attack_targets and nearest_indices:
        matched = 0
        for agent_idx, action in enumerate(active_actions):
            if int(action) >= SMAC_ATTACK_OFFSET:
                target_idx = int(action) - SMAC_ATTACK_OFFSET
                if agent_idx < len(nearest_indices) and target_idx == nearest_indices[agent_idx]:
                    matched += 1
        nearest_agreement = _safe_ratio(matched, len(attack_targets))

    ally_spread = _mean_distance_to_center(allies)
    enemy_spread = _mean_distance_to_center(enemies)
    cluster_digit = _choose_cluster_digit(focus_ratio, ally_spread, enemy_spread)

    if total == 0:
        letter, confidence, reason = "k", 0.2, "empty_joint_action"
    elif no_op_rate >= 0.75:
        letter, confidence, reason = "k", min(1.0, 0.5 + no_op_rate / 2), "dominant_no_op"
    elif stop_rate >= 0.75 and attack_rate < 0.2:
        letter, confidence, reason = "k", min(0.85, 0.45 + stop_rate / 2), "dominant_stop"
    elif attack_rate >= 0.55:
        letter, confidence, reason = _project_attack_letter(
            focus_ratio=focus_ratio,
            weakest_agreement=weakest_agreement,
            nearest_agreement=nearest_agreement,
            move_rate=move_rate,
        )
    elif move_rate >= 0.55:
        letter, confidence, reason = _project_movement_letter(
            move_actions=move_actions,
            allies=allies,
            enemies=enemies,
        )
    elif attack_rate > 0:
        letter, confidence, reason = "e", 0.45 + 0.25 * attack_rate, "mixed_sparse_attack"
    else:
        letter, confidence, reason = "j", 0.35, "unstructured_non_attack"

    confidence = max(0.0, min(1.0, float(confidence)))
    action_code = f"{int(cluster_digit)}{letter}"
    features = {
        "total_agents": total,
        "attack_rate": attack_rate,
        "move_rate": move_rate,
        "stop_rate": stop_rate,
        "no_op_rate": no_op_rate,
        "focus_ratio": focus_ratio,
        "weakest_agreement": weakest_agreement,
        "nearest_agreement": nearest_agreement,
        "ally_spread": ally_spread,
        "enemy_spread": enemy_spread,
        "top_target": top_target,
        "target_histogram": dict(target_counts),
    }
    signature = {
        "pymarl_actions": active_actions,
        "available_actions": available_actions,
        "attack_targets": attack_targets,
        "script_name": SCRIPT_ACTIONS.get(letter, "unknown"),
    }
    return ProjectionResult(
        action_code=action_code,
        action_letter=letter,
        cluster_digit=int(cluster_digit),
        confidence=confidence,
        reason=reason,
        features=features,
        micro_action_signature=signature,
    )


def _choose_cluster_digit(focus_ratio: float, ally_spread: float, enemy_spread: float) -> int:
    if focus_ratio >= 0.85:
        return 4
    if focus_ratio >= 0.65:
        return 3
    if ally_spread > enemy_spread * 1.25:
        return 1
    if ally_spread < enemy_spread * 0.75:
        return 2
    return 2


def _project_attack_letter(
    focus_ratio: float,
    weakest_agreement: float,
    nearest_agreement: float,
    move_rate: float,
) -> Tuple[str, float, str]:
    if weakest_agreement >= 0.65 and focus_ratio >= 0.65:
        return "c", 0.72 + 0.18 * weakest_agreement, "focused_weakest_attack"
    if weakest_agreement >= 0.55 and nearest_agreement >= 0.45:
        return "d", 0.68 + 0.16 * min(weakest_agreement, nearest_agreement), "clustered_nearest_weakest_attack"
    if nearest_agreement >= 0.65 and focus_ratio < 0.75:
        return "b", 0.66 + 0.18 * nearest_agreement, "per_unit_nearest_attack"
    if nearest_agreement >= 0.45:
        return "a", 0.58 + 0.18 * nearest_agreement, "nearest_attack"
    if move_rate >= 0.25:
        return "e", 0.55 + 0.12 * focus_ratio, "attack_with_movement_pressure"
    return "c" if focus_ratio >= 0.55 else "e", 0.52 + 0.2 * focus_ratio, "generic_attack"


def _project_movement_letter(
    move_actions: Iterable[int],
    allies: Sequence[Sequence[float]],
    enemies: Sequence[Sequence[float]],
) -> Tuple[str, float, str]:
    counts = Counter(move_actions)
    dominant_move, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
    dominance = _safe_ratio(dominant_count, sum(counts.values()))
    if not allies or not enemies:
        return "g", 0.45 + 0.25 * dominance, "movement_without_full_state"
    ally_center = _center(allies)
    enemy_center = _center(enemies)
    before_dist = _distance(ally_center, enemy_center)
    dx, dy = {
        SMAC_MOVE_NORTH: (0.0, 1.0),
        SMAC_MOVE_SOUTH: (0.0, -1.0),
        SMAC_MOVE_EAST: (1.0, 0.0),
        SMAC_MOVE_WEST: (-1.0, 0.0),
    }.get(dominant_move, (0.0, 0.0))
    after_dist = _distance((ally_center[0] + dx, ally_center[1] + dy), enemy_center)
    if after_dist > before_dist:
        return "h", 0.55 + 0.25 * dominance, "movement_away_lure"
    if after_dist < before_dist:
        return "g", 0.55 + 0.25 * dominance, "movement_toward_gather"
    return "f", 0.5 + 0.2 * dominance, "lateral_defensive_movement"


def project_frame(frame: Mapping[str, Any]) -> ProjectionResult:
    actions = (
        frame.get("pymarl_actions")
        if "pymarl_actions" in frame
        else frame.get("actions", frame.get("joint_action"))
    )
    return project_pymarl_actions(
        actions,
        state=extract_state(frame),
        available_actions=frame.get("available_actions", frame.get("avail_actions")),
    )


def summarize_projection_results(results: Sequence[ProjectionResult]) -> Dict[str, Any]:
    if not results:
        return {
            "frames": 0,
            "mean_confidence": None,
            "low_confidence_ratio": None,
            "action_counts": {},
            "reason_counts": {},
        }
    confidences = [float(item.confidence) for item in results]
    return {
        "frames": len(results),
        "mean_confidence": sum(confidences) / len(confidences),
        "low_confidence_ratio": sum(1 for value in confidences if value < 0.5) / len(confidences),
        "action_counts": dict(Counter(item.action_code for item in results).most_common(20)),
        "reason_counts": dict(Counter(item.reason for item in results).most_common(20)),
    }

