import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.structure.custom_distance_sc2 import CustomDistance


SCENARIO_ORDER = ["sce-1", "sce-2", "sce-3", "sce-1m", "sce-2m", "sce-3m"]
METHOD_DISPLAY = {"etg-only": "etg-only", "synergy": "Synergy"}
METHOD_COLOR = {"etg-only": "#4C78A8", "Synergy": "#E45756"}
PROJECT_STATE_DISTANCE = CustomDistance(threshold=0.5).multi_distance
PROJECT_HP_WEIGHT = 1.0
DEFAULT_MAP_RESOLUTION = 128.0


@dataclass
class EpisodeTrace:
    uid: str
    method_group: str
    method: str
    experiment_id: str
    map_key: str
    episode_id: int
    score: float
    result: str
    frames: list[dict]
    state_keys: list[str]
    actions: list[str]
    changed: list[bool]
    hp_delta: list[float]
    hp_my: list[float]
    hp_enemy: list[float]
    states: list[dict | None]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, path_base: Path):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    def long_path(path: Path) -> str:
        text = str(path.resolve())
        if len(text) >= 240 and not text.startswith("\\\\?\\"):
            return "\\\\?\\" + text
        return text

    fig.savefig(long_path(path_base.with_suffix(".pdf")))
    fig.savefig(long_path(path_base.with_suffix(".png")))
    plt.close(fig)


def clean_norm_state(state) -> dict | None:
    if not isinstance(state, dict):
        return None
    cleaned = {}
    for side in ("red_army", "blue_army"):
        units = state.get(side)
        if not isinstance(units, list):
            return None
        clean_units = []
        for unit in units:
            if not isinstance(unit, (list, tuple)) or len(unit) < 3:
                return None
            try:
                clean_units.append([float(unit[0]), float(unit[1]), float(unit[2])])
            except (TypeError, ValueError):
                return None
        cleaned[side] = clean_units
    return cleaned


def raw_frame_to_norm_state(frame: dict, map_resolution: float = DEFAULT_MAP_RESOLUTION) -> dict | None:
    if not isinstance(frame, dict):
        return None

    def normalize_units(units) -> list[list[float]] | None:
        if not isinstance(units, list):
            return None
        half = map_resolution / 2.0
        clean_units = []
        for unit in units:
            if not isinstance(unit, dict):
                return None
            try:
                x = float(unit["x"])
                y = float(unit["y"])
                hp = float(unit["hp"])
            except (KeyError, TypeError, ValueError):
                return None
            clean_units.append([x / half - 1.0, 1.0 - y / half, hp / 45.0])
        return clean_units

    red = normalize_units(frame.get("my_units_pos"))
    blue = normalize_units(frame.get("enemy_units_pos"))
    if red is None or blue is None:
        return None
    return {"red_army": red, "blue_army": blue}


def frame_to_project_state(frame: dict) -> dict | None:
    state = clean_norm_state(frame.get("eval_bktree_norm_state"))
    if state is not None:
        return state
    return raw_frame_to_norm_state(frame)


def project_state_distance(left: dict | None, right: dict | None, hp_weight: float = PROJECT_HP_WEIGHT) -> float:
    if left is None or right is None:
        return math.inf
    try:
        dist_pos, dist_hp = PROJECT_STATE_DISTANCE(left, right)
    except Exception:
        return math.inf
    return float(dist_pos) + hp_weight * float(dist_hp)


def prefix_state_distance(left: EpisodeTrace, right: EpisodeTrace, prefix_len: int) -> float:
    distances = []
    for idx in range(prefix_len):
        distances.append(project_state_distance(left.states[idx], right.states[idx]))
    if any(not math.isfinite(item) for item in distances):
        return math.inf
    return float(np.mean(distances))


def dtw_node_path(left_states: list[dict | None], right_states: list[dict | None]) -> list[tuple[int, int, float]]:
    rows = len(left_states)
    cols = len(right_states)
    if rows == 0 or cols == 0:
        return []
    dist = np.full((rows, cols), np.inf, dtype=float)
    for row in range(rows):
        for col in range(cols):
            dist[row, col] = project_state_distance(left_states[row], right_states[col])
    cost = np.full((rows, cols), np.inf, dtype=float)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    for row in range(rows):
        for col in range(cols):
            candidates = []
            if row > 0:
                candidates.append((cost[row - 1, col], (row - 1, col)))
            if col > 0:
                candidates.append((cost[row, col - 1], (row, col - 1)))
            if row > 0 and col > 0:
                candidates.append((cost[row - 1, col - 1], (row - 1, col - 1)))
            if not candidates:
                cost[row, col] = dist[row, col]
                parent[(row, col)] = None
            else:
                best_cost, best_parent = min(candidates, key=lambda item: item[0])
                cost[row, col] = dist[row, col] + best_cost
                parent[(row, col)] = best_parent
    path = []
    cursor = (rows - 1, cols - 1)
    while cursor is not None:
        row, col = cursor
        path.append((row, col, float(dist[row, col])))
        cursor = parent.get(cursor)
    path.reverse()
    return path


def truthy(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def load_batch_items(root: Path, batch_tag: str) -> list[dict]:
    result_path = root / "_batch_final_eval_logs" / batch_tag / "batch_final_eval_results.json"
    with result_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("results", [])


def load_traces(root: Path, batch_tag: str) -> list[EpisodeTrace]:
    traces = []
    for item in load_batch_items(root, batch_tag):
        repeat_dir = Path(item["output_dir"]) / "repeats" / "repeat_001"
        episodes_path = repeat_dir / "episodes.jsonl"
        if not episodes_path.exists():
            continue
        method = METHOD_DISPLAY.get(item["method_group"], item["method_group"])
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                frames = episode.get("frames") or []
                if len(frames) < 2:
                    continue
                state_keys = [str(frame.get("state_key", "")) for frame in frames]
                actions = [str(frame.get("action_code") or frame.get("shadow_selected_action_code") or "") for frame in frames]
                changed = [truthy(frame.get("shadow_mechanism_changed_action")) for frame in frames]
                hp_delta = [float(frame.get("hp_delta") or 0.0) for frame in frames]
                hp_my = [float(frame.get("hp_my") or 0.0) for frame in frames]
                hp_enemy = [float(frame.get("hp_enemy") or 0.0) for frame in frames]
                states = []
                for frame in frames:
                    states.append(frame_to_project_state(frame))
                episode_id = int(episode.get("episode_id"))
                traces.append(
                    EpisodeTrace(
                        uid=f"{item['method_group']}::{item['experiment_id']}::ep{episode_id}",
                        method_group=item["method_group"],
                        method=method,
                        experiment_id=item["experiment_id"],
                        map_key=item["map_key"],
                        episode_id=episode_id,
                        score=float(episode.get("score", np.nan)),
                        result=str(episode.get("result")),
                        frames=frames,
                        state_keys=state_keys,
                        actions=actions,
                        changed=changed,
                        hp_delta=hp_delta,
                        hp_my=hp_my,
                        hp_enemy=hp_enemy,
                        states=states,
                    )
                )
    return traces


def select_traces(traces: list[EpisodeTrace], sample_mode: str) -> list[EpisodeTrace]:
    selected = []
    groups: dict[tuple[str, str, str], list[EpisodeTrace]] = {}
    for trace in traces:
        groups.setdefault((trace.method_group, trace.map_key, trace.experiment_id), []).append(trace)
    for group in groups.values():
        group = sorted(group, key=lambda item: item.score, reverse=True)
        if sample_mode == "all300":
            part = group
        elif sample_mode == "top100":
            part = group[:100]
        elif sample_mode == "middle100":
            part = group[100:200]
        elif sample_mode == "bottom100":
            part = group[-100:]
        elif sample_mode == "small_all_large_top100":
            part = group[:100] if group[0].map_key in {"sce-3", "sce-3m"} else group
        elif sample_mode == "small_top100_large_all":
            part = group if group[0].map_key in {"sce-3", "sce-3m"} else group[:100]
        else:
            raise ValueError(sample_mode)
        selected.extend(part)
    return selected


def longest_common_prefix(left: EpisodeTrace, right: EpisodeTrace, max_len: int | None = None) -> int:
    limit = min(len(left.state_keys), len(right.state_keys))
    if max_len is not None:
        limit = min(limit, max_len)
    count = 0
    for idx in range(limit):
        if left.state_keys[idx] != right.state_keys[idx]:
            break
        count += 1
    return count


def first_action_divergence(left: EpisodeTrace, right: EpisodeTrace, start: int, end: int) -> int | None:
    limit = min(len(left.actions), len(right.actions), end)
    for idx in range(max(start, 0), limit):
        if left.actions[idx] and right.actions[idx] and left.actions[idx] != right.actions[idx]:
            return idx
    return None


def case_row(left: EpisodeTrace, right: EpisodeTrace, prefix_len: int, match_type: str, distance: float, pair_type: str) -> dict | None:
    lcp = longest_common_prefix(left, right)
    if match_type == "exact" and lcp < prefix_len:
        return None
    decision_step = first_action_divergence(left, right, prefix_len - 1, max(prefix_len, lcp) + 1)
    if decision_step is None:
        return None

    left_is_syn = left.method == "Synergy"
    right_is_syn = right.method == "Synergy"
    synergy = left if left_is_syn else right if right_is_syn else None
    other = right if left_is_syn else left if right_is_syn else None
    synergy_step_changed = False
    synergy_action = ""
    no_mech_action = ""
    if synergy is not None and decision_step < len(synergy.frames):
        frame = synergy.frames[decision_step]
        synergy_step_changed = truthy(frame.get("shadow_mechanism_changed_action"))
        synergy_action = str(frame.get("shadow_selected_action_code") or synergy.actions[decision_step])
        no_mech_action = str(frame.get("shadow_no_mechanism_action_code") or "")

    if synergy is not None and other is not None:
        score_delta_synergy = synergy.score - other.score
        end_hp_delta_synergy = synergy.hp_delta[-1] - other.hp_delta[-1]
        terminal_hp_synergy = synergy.hp_delta[-1]
        terminal_hp_other = other.hp_delta[-1]
    else:
        score_delta_synergy = np.nan
        end_hp_delta_synergy = np.nan
        terminal_hp_synergy = np.nan
        terminal_hp_other = np.nan

    return {
        "pair_type": pair_type,
        "match_type": match_type,
        "prefix_len": prefix_len,
        "prefix_distance": distance,
        "map_key": left.map_key,
        "left_method": left.method,
        "left_experiment_id": left.experiment_id,
        "left_episode_id": left.episode_id,
        "left_score": left.score,
        "left_result": left.result,
        "right_method": right.method,
        "right_experiment_id": right.experiment_id,
        "right_episode_id": right.episode_id,
        "right_score": right.score,
        "right_result": right.result,
        "score_delta_right_minus_left": right.score - left.score,
        "score_delta_synergy_minus_other": score_delta_synergy,
        "end_hp_delta_synergy_minus_other": end_hp_delta_synergy,
        "terminal_hp_delta_synergy": terminal_hp_synergy,
        "terminal_hp_delta_other": terminal_hp_other,
        "lcp_exact": lcp,
        "decision_step": decision_step,
        "state_key_at_decision_left": left.state_keys[decision_step] if decision_step < len(left.state_keys) else "",
        "state_key_at_decision_right": right.state_keys[decision_step] if decision_step < len(right.state_keys) else "",
        "left_action": left.actions[decision_step] if decision_step < len(left.actions) else "",
        "right_action": right.actions[decision_step] if decision_step < len(right.actions) else "",
        "left_hp_delta_at_decision": left.hp_delta[decision_step] if decision_step < len(left.hp_delta) else np.nan,
        "right_hp_delta_at_decision": right.hp_delta[decision_step] if decision_step < len(right.hp_delta) else np.nan,
        "left_hp_delta_next": left.hp_delta[min(decision_step + 1, len(left.hp_delta) - 1)],
        "right_hp_delta_next": right.hp_delta[min(decision_step + 1, len(right.hp_delta) - 1)],
        "synergy_step_changed": synergy_step_changed,
        "synergy_selected_action": synergy_action,
        "synergy_no_mechanism_action": no_mech_action,
        "synergy_episode_id": synergy.episode_id if synergy is not None else "",
        "other_episode_id": other.episode_id if other is not None else "",
        "left_uid": left.uid,
        "right_uid": right.uid,
    }


def exact_candidate_pairs(traces: list[EpisodeTrace], prefix_len: int, include_within: bool) -> Iterable[tuple[EpisodeTrace, EpisodeTrace, float, str]]:
    by_scenario: dict[str, list[EpisodeTrace]] = {}
    for trace in traces:
        if len(trace.state_keys) >= prefix_len:
            by_scenario.setdefault(trace.map_key, []).append(trace)
    for scenario, scenario_traces in by_scenario.items():
        buckets: dict[tuple[str, ...], list[EpisodeTrace]] = {}
        for trace in scenario_traces:
            buckets.setdefault(tuple(trace.state_keys[:prefix_len]), []).append(trace)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            for idx, base_left in enumerate(bucket):
                for base_right in bucket[idx + 1 :]:
                    left = base_left
                    right = base_right
                    if left.method != right.method:
                        if left.method == "Synergy":
                            left, right = right, left
                        yield left, right, 0.0, "cross_method"
                    elif include_within:
                        yield left, right, 0.0, f"within_{left.method}"


def similar_candidate_pairs(
    traces: list[EpisodeTrace],
    prefix_len: int,
    threshold: float,
    include_within: bool,
    max_pairs_per_scenario: int,
) -> Iterable[tuple[EpisodeTrace, EpisodeTrace, float, str]]:
    by_scenario: dict[str, list[EpisodeTrace]] = {}
    for trace in traces:
        if len(trace.states) >= prefix_len and all(trace.states[idx] is not None for idx in range(prefix_len)):
            by_scenario.setdefault(trace.map_key, []).append(trace)
    for scenario, scenario_traces in by_scenario.items():
        emitted = 0
        for idx, base_left in enumerate(scenario_traces):
            for base_right in scenario_traces[idx + 1 :]:
                left = base_left
                right = base_right
                if not include_within and left.method == right.method:
                    continue
                pair_type = "cross_method" if left.method != right.method else f"within_{left.method}"
                if left.method == "Synergy" and right.method != "Synergy":
                    left, right = right, left
                dist = prefix_state_distance(left, right, prefix_len)
                if dist <= threshold:
                    yield left, right, dist, pair_type
                    emitted += 1
                    if emitted >= max_pairs_per_scenario:
                        break
            if emitted >= max_pairs_per_scenario:
                break


def build_cases(
    traces: list[EpisodeTrace],
    prefix_len: int,
    match_type: str,
    threshold: float,
    include_within: bool,
    max_pairs_per_scenario: int,
) -> pd.DataFrame:
    rows = []
    if match_type == "exact":
        iterator = exact_candidate_pairs(traces, prefix_len, include_within)
    else:
        iterator = similar_candidate_pairs(traces, prefix_len, threshold, include_within, max_pairs_per_scenario)
    for left, right, distance, pair_type in iterator:
        row = case_row(left, right, prefix_len, match_type, distance, pair_type)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def hp_color(value, vmin, vmax):
    if not math.isfinite(value):
        return "#BDBDBD"
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    return plt.cm.coolwarm(float(np.clip(t, 0, 1)))


def frame_gate_metrics(frame: dict) -> dict:
    shadow = frame.get("mechanism_shadow")
    tuning = shadow.get("tuning") if isinstance(shadow, dict) else {}
    resolution = shadow.get("nid_resolution") if isinstance(shadow, dict) else {}
    if not isinstance(tuning, dict):
        tuning = {}
    if not isinstance(resolution, dict):
        resolution = {}

    def as_float(value):
        try:
            if value is None or value == "":
                return np.nan
            return float(value)
        except Exception:
            return np.nan

    def as_int(value):
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except Exception:
            return None

    confidence = as_float(tuning.get("confidence"))
    advantage = as_float(tuning.get("advantage"))
    candidate_visits = as_int(tuning.get("candidate_visits"))
    threshold_confidence = as_float(tuning.get("threshold_confidence"))
    threshold_advantage = as_float(tuning.get("threshold_advantage"))
    threshold_visits = as_int(tuning.get("threshold_visits"))
    model_total_visits = as_int(shadow.get("model_total_visits") if isinstance(shadow, dict) else None)
    is_ood = truthy(frame.get("nid_is_ood")) or truthy(resolution.get("is_ood"))
    nid_status = str(frame.get("nid_status") or resolution.get("status") or "")
    tuning_reason = str(tuning.get("reason") or "")
    return {
        "confidence": confidence,
        "advantage": advantage,
        "candidate_visits": candidate_visits,
        "threshold_confidence": threshold_confidence,
        "threshold_advantage": threshold_advantage,
        "threshold_visits": threshold_visits,
        "model_total_visits": model_total_visits,
        "is_ood": is_ood,
        "nid_status": nid_status,
        "tuning_reason": tuning_reason,
        "source": str(tuning.get("source") or ""),
        "candidate_action": str(tuning.get("candidate_action") or ""),
        "etg_action": str(tuning.get("etg_action") or ""),
        "selected_action": str(tuning.get("action") or ""),
        "candidate_eligible": truthy(tuning.get("candidate_eligible")),
    }


def gate_metric_has_values(metrics: dict) -> bool:
    return (
        math.isfinite(metrics.get("confidence", np.nan))
        or math.isfinite(metrics.get("advantage", np.nan))
        or metrics.get("candidate_visits") is not None
    )


def gate_metric_is_advantageous(metrics: dict) -> bool:
    confidence = metrics.get("confidence", np.nan)
    advantage = metrics.get("advantage", np.nan)
    visits = metrics.get("candidate_visits")
    threshold_confidence = metrics.get("threshold_confidence", np.nan)
    threshold_advantage = metrics.get("threshold_advantage", np.nan)
    threshold_visits = metrics.get("threshold_visits")
    confidence_ok = math.isfinite(confidence) and confidence > 0.0
    advantage_ok = math.isfinite(advantage) and advantage > 0.0
    visits_ok = visits is not None and visits > 0
    if math.isfinite(threshold_confidence):
        confidence_ok = confidence_ok and confidence >= threshold_confidence
    if math.isfinite(threshold_advantage):
        advantage_ok = advantage_ok and advantage >= threshold_advantage
    if threshold_visits is not None:
        visits_ok = visits_ok and visits >= threshold_visits
    reason = str(metrics.get("tuning_reason") or "")
    return bool(
        (confidence_ok and advantage_ok and visits_ok)
        or metrics.get("candidate_eligible")
        or "confident_advantage" in reason
    )


def gate_label(metrics: dict, compact: bool = False, evidence_offset: int | None = None) -> str:
    parts = []
    confidence = metrics.get("confidence", np.nan)
    advantage = metrics.get("advantage", np.nan)
    candidate_visits = metrics.get("candidate_visits")
    model_total_visits = metrics.get("model_total_visits")
    if evidence_offset is not None and evidence_offset != 0:
        parts.append(f"near {evidence_offset:+d}")
    if math.isfinite(confidence):
        parts.append(f"C={confidence:.2f}")
    if math.isfinite(advantage):
        parts.append(f"A={advantage:.1f}")
    if candidate_visits is not None:
        if model_total_visits is not None and not compact:
            parts.append(f"v={candidate_visits}/{model_total_visits}")
        else:
            parts.append(f"v={candidate_visits}")
    if metrics.get("is_ood"):
        parts.append("OOD")
    if not parts:
        return ""
    return "\n".join(parts[:2] if compact else parts)


def nearby_gate_evidence(trace: EpisodeTrace | None, decision_step: int, radius: int = 10) -> dict:
    if trace is None:
        return {"status": "missing_synergy_trace", "step": None, "offset": None, "metrics": {}, "label": ""}
    candidates = []
    fallback_candidates = []
    for step in range(max(0, decision_step - radius), min(len(trace.frames), decision_step + radius + 1)):
        metrics = frame_gate_metrics(trace.frames[step])
        has_values = gate_metric_has_values(metrics)
        advantageous = gate_metric_is_advantageous(metrics)
        changed = bool(step < len(trace.changed) and trace.changed[step])
        if not has_values and not advantageous and not changed:
            continue
        offset = step - decision_step
        candidate = {
            "step": step,
            "offset": offset,
            "metrics": metrics,
            "changed": changed,
            "advantageous": advantageous,
            "has_values": has_values,
        }
        if advantageous:
            candidates.append(candidate)
        else:
            fallback_candidates.append(candidate)
    if not candidates:
        if fallback_candidates:
            def fallback_rank(candidate: dict):
                return (
                    0 if candidate["changed"] else 1,
                    abs(candidate["offset"]),
                    0 if candidate["offset"] >= 0 else 1,
                )

            best_fallback = sorted(fallback_candidates, key=fallback_rank)[0]
            return {
                "status": "no_advantageous_gate_evidence",
                "step": best_fallback["step"],
                "offset": best_fallback["offset"],
                "metrics": best_fallback["metrics"],
                "changed": best_fallback["changed"],
                "advantageous": False,
                "label": "",
                "fallback_has_values": best_fallback["has_values"],
            }
        return {"status": "no_nearby_gate_evidence", "step": None, "offset": None, "metrics": {}, "label": ""}

    def rank(candidate: dict):
        return (
            0 if candidate["changed"] else 1,
            abs(candidate["offset"]),
            0 if candidate["offset"] >= 0 else 1,
        )

    best = sorted(candidates, key=rank)[0]
    status = "exact_advantage_gate_evidence" if best["offset"] == 0 else "nearby_advantage_gate_evidence"
    label = gate_label(best["metrics"], compact=False, evidence_offset=best["offset"])
    return {
        "status": status,
        "step": best["step"],
        "offset": best["offset"],
        "metrics": best["metrics"],
        "changed": best["changed"],
        "advantageous": best["advantageous"],
        "label": label,
    }


def display_state_key(state_key) -> str:
    text = str(state_key)
    if text.lower().startswith("ood:"):
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
        return f"OOD-{digest}"
    if len(text) > 18:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
        return f"state-{digest}"
    return text


def plot_case_graph(
    left: EpisodeTrace,
    right: EpisodeTrace,
    case: pd.Series,
    output_path: Path,
    horizon: int = 10,
    paper_labels: bool = False,
):
    def method_label(method: str) -> str:
        if not paper_labels:
            return method
        return {"etg-only": "Graph-only SAGE", "Synergy": "Full SAGE"}.get(method, method)

    decision_step = int(case["decision_step"])
    prefix_len = int(case["prefix_len"])
    start = max(0, decision_step - prefix_len + 1)
    end = min(max(len(left.frames), len(right.frames)), decision_step + horizon + 1)
    graph = nx.DiGraph()
    positions = {}
    node_values = []
    node_to_trace_step = {}
    synergy_trace = left if left.method == "Synergy" else right if right.method == "Synergy" else None
    gate_evidence = nearby_gate_evidence(synergy_trace, decision_step)
    decision_gate_label = str(gate_evidence.get("label") or "")

    common_until = min(decision_step, len(left.frames) - 1, len(right.frames) - 1)
    common_nodes = {}
    for step in range(start, common_until + 1):
        node = f"common:{step}"
        if synergy_trace is not None and step < len(synergy_trace.frames):
            metrics = frame_gate_metrics(synergy_trace.frames[step])
        else:
            metrics = frame_gate_metrics(left.frames[step])
        is_decision = step == decision_step
        label = display_state_key(left.state_keys[step])
        if is_decision and decision_gate_label:
            label = f"{label}\n{decision_gate_label}"
        graph.add_node(
            node,
            label=label,
            method="Common prefix",
            hp_delta=left.hp_delta[step],
            changed=bool(synergy_trace and step < len(synergy_trace.changed) and synergy_trace.changed[step]),
            action=left.actions[step],
            step=step,
            common=True,
            decision=is_decision,
            is_ood=metrics.get("is_ood", False),
            gate=metrics,
        )
        positions[node] = (step - start, 0.5)
        common_nodes[step] = node
        if step > start:
            graph.add_edge(common_nodes[step - 1], node, method="Common prefix", action=left.actions[step - 1])

    branch_specs = [(left, 0.86), (right, 0.14)]
    branch_nodes: dict[str, list[tuple[int, str]]] = {left.uid: [], right.uid: []}
    for trace, y in branch_specs:
        previous = common_nodes.get(decision_step)
        for step in range(decision_step + 1, min(end, len(trace.frames))):
            node = f"{trace.method}:{trace.episode_id}:{step}"
            metrics = frame_gate_metrics(trace.frames[step])
            label = f"{display_state_key(trace.state_keys[step])}\nΔHP={trace.hp_delta[step]:.0f}"
            graph.add_node(
                node,
                label=label,
                method=trace.method,
                hp_delta=trace.hp_delta[step],
                changed=trace.changed[step],
                action=trace.actions[step],
                step=step,
                common=False,
                decision=False,
                is_ood=metrics.get("is_ood", False),
                gate=metrics,
            )
            positions[node] = (step - start, y)
            node_to_trace_step[node] = (trace, step)
            branch_nodes[trace.uid].append((step, node))
            node_values.append(trace.hp_delta[step])
            if previous is not None:
                action_step = max(step - 1, 0)
                graph.add_edge(previous, node, method=trace.method, action=trace.actions[action_step])
            previous = node

    vmin = min(node_values) if node_values else -1
    vmax = max(node_values) if node_values else 1
    fig, ax = plt.subplots(figsize=(9.2, 2.6))

    left_branch = branch_nodes.get(left.uid, [])
    right_branch = branch_nodes.get(right.uid, [])
    left_states = [left.states[step] if step < len(left.states) else None for step, _ in left_branch]
    right_states = [right.states[step] if step < len(right.states) else None for step, _ in right_branch]
    dtw_path = dtw_node_path(left_states, right_states)
    finite_distances = [dist for _, _, dist in dtw_path if math.isfinite(dist)]
    max_dtw_distance = max(finite_distances) if finite_distances else 1.0
    max_dtw_distance = max(max_dtw_distance, 1e-6)
    used_label_slots = {}
    for row, col, distance in dtw_path:
        if row >= len(left_branch) or col >= len(right_branch) or not math.isfinite(distance):
            continue
        left_node = left_branch[row][1]
        right_node = right_branch[col][1]
        x1, y1 = positions[left_node]
        x2, y2 = positions[right_node]
        distance_ratio = float(np.clip(distance / max_dtw_distance, 0.0, 1.0))
        gray = 0.66 - 0.36 * distance_ratio
        line_color = (gray, gray, gray)
        line_width = 0.7 + 2.7 * distance_ratio
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=line_color,
            lw=line_width,
            ls=(0, (3, 3)),
            alpha=0.56 + 0.26 * distance_ratio,
            zorder=2.2,
        )
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        slot_key = round(mid_x, 1)
        slot_count = used_label_slots.get(slot_key, 0)
        used_label_slots[slot_key] = slot_count + 1
        label_y = mid_y + (0.035 if slot_count % 2 == 0 else -0.035)
        ax.text(
            mid_x,
            label_y,
            f"{distance:.3f}",
            fontsize=6.4,
            color="#4D4D4D",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.25},
            zorder=4.5,
        )

    for edge in graph.edges(data=True):
        source, target, data = edge
        color = METHOD_COLOR.get(data["method"], "#777777")
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(source, target)],
            ax=ax,
            edge_color=color,
            arrows=True,
            arrowsize=9,
            width=1.5 if data["method"] == "Synergy" else 1.0,
            alpha=0.78 if data["method"] != "Common prefix" else 0.55,
            connectionstyle="arc3,rad=0.03",
        )
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, data.get("action", ""), fontsize=7.2, color=color, ha="center")

    for node, data in graph.nodes(data=True):
        if data.get("common"):
            marker = "*" if data.get("changed") else "D" if data.get("decision") else "o"
            size = 260 if data.get("changed") else 170 if data.get("decision") else 110
            color = "#F0F0F0" if data.get("decision") else "#D9D9D9"
            edge_color = "#2E7D32" if data.get("decision") else "#777777"
            label_y_offset = -0.16 if data.get("decision") else -0.13
            label_va = "top"
        else:
            marker = "*" if data["changed"] else "o"
            size = 250 if data["changed"] else 150
            color = hp_color(data["hp_delta"], vmin, vmax)
            edge_color = METHOD_COLOR.get(data["method"], "black")
            if data["method"] == left.method:
                label_y_offset = 0.15
                label_va = "bottom"
            else:
                label_y_offset = -0.18
                label_va = "top"
        if data.get("is_ood") and not data.get("changed"):
            edge_color = "#F28E2B"
            size = max(size, 190)
        ax.scatter(
            [positions[node][0]],
            [positions[node][1]],
            s=size,
            marker=marker,
            color=color,
            edgecolor=edge_color,
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            positions[node][0],
            positions[node][1] + label_y_offset,
            data["label"],
            fontsize=6.8,
            ha="center",
            va=label_va,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.45, "pad": 0.15} if not data.get("common") else None,
            zorder=4,
        )

    ax.axvline(decision_step - start, color="#2E7D32", lw=1.0, ls="--", alpha=0.32, zorder=0)
    title = (
        f"{case['map_key']} | prefix={case['prefix_len']} | backup switch = {case['match_type']} | "
        f"prefixDist={case['prefix_distance']:.3f} | "
        f"{method_label('Synergy')} ΔterminalHP={case['end_hp_delta_synergy_minus_other']:.1f} | "
        f"changed={case['synergy_step_changed']}"
    )
    ax.set_title(title, fontsize=12.4)
    ax.set_yticks([0.14, 0.5, 0.86])
    ax.set_yticklabels([method_label(right.method), "Common prefix", method_label(left.method)])
    ax.tick_params(axis="both", labelsize=8.8)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, max(1, end - start - 0.5))
    ax.set_ylim(-0.55, 1.45)
    ax.grid(axis="x", alpha=0.15)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=8.2)
    cbar.set_label("HP advantage", fontsize=9.2)
    save_fig(fig, output_path)


def trace_lookup(traces: list[EpisodeTrace]) -> dict[str, EpisodeTrace]:
    return {trace.uid: trace for trace in traces}


def write_config_report(cases: pd.DataFrame, out_dir: Path, name: str):
    lines = [f"# Sequence divergence cases: {name}", ""]
    lines.append(f"- Total cases: `{len(cases)}`")
    if not cases.empty:
        lines.append(f"- Cross-method cases: `{int(cases['pair_type'].eq('cross_method').sum())}`")
        lines.append(f"- Synergy-changed cases: `{int(cases['synergy_step_changed'].sum())}`")
        lines.append("")
        lines.append("## Scenario counts")
        lines.append("")
        counts = cases.groupby(["map_key", "pair_type"]).size().reset_index(name="count")
        lines.append(counts.to_markdown(index=False))
        lines.append("")
        lines.append("## Top positive Synergy cases")
        lines.append("")
        cols = [
            "map_key",
            "left_method",
            "left_episode_id",
            "left_score",
            "right_method",
            "right_episode_id",
            "right_score",
            "score_delta_synergy_minus_other",
            "decision_step",
            "left_action",
            "right_action",
            "synergy_step_changed",
        ]
        top = cases.sort_values(["synergy_step_changed", "score_delta_synergy_minus_other"], ascending=[False, False]).head(20)
        lines.append(top[cols].to_markdown(index=False))
    (out_dir / "README_cases.md").write_text("\n".join(lines), encoding="utf-8")


def gate_evidence_row(case: pd.Series, lookup: dict[str, EpisodeTrace]) -> dict:
    left = lookup.get(case["left_uid"])
    right = lookup.get(case["right_uid"])
    synergy_trace = left if left is not None and left.method == "Synergy" else right if right is not None and right.method == "Synergy" else None
    decision_step = int(case["decision_step"])
    evidence = nearby_gate_evidence(synergy_trace, decision_step)
    metrics = evidence.get("metrics") or {}
    return {
        "map_key": case.get("map_key", ""),
        "match_type": case.get("match_type", ""),
        "prefix_len": int(case.get("prefix_len", 0)),
        "prefix_distance": float(case.get("prefix_distance", np.nan)),
        "left_method": case.get("left_method", ""),
        "left_episode_id": case.get("left_episode_id", ""),
        "right_method": case.get("right_method", ""),
        "right_episode_id": case.get("right_episode_id", ""),
        "decision_step": decision_step,
        "evidence_status": evidence.get("status", ""),
        "evidence_step": evidence.get("step", ""),
        "evidence_offset": evidence.get("offset", ""),
        "evidence_changed_action": bool(evidence.get("changed", False)),
        "evidence_advantageous": bool(evidence.get("advantageous", False)),
        "confidence": metrics.get("confidence", np.nan),
        "advantage": metrics.get("advantage", np.nan),
        "candidate_visits": metrics.get("candidate_visits", ""),
        "model_total_visits": metrics.get("model_total_visits", ""),
        "threshold_confidence": metrics.get("threshold_confidence", np.nan),
        "threshold_advantage": metrics.get("threshold_advantage", np.nan),
        "threshold_visits": metrics.get("threshold_visits", ""),
        "tuning_reason": metrics.get("tuning_reason", ""),
        "source": metrics.get("source", ""),
        "candidate_action": metrics.get("candidate_action", ""),
        "etg_action": metrics.get("etg_action", ""),
        "selected_action": metrics.get("selected_action", ""),
        "terminal_hp_delta_synergy_minus_other": case.get("end_hp_delta_synergy_minus_other", np.nan),
        "score_delta_synergy_minus_other": case.get("score_delta_synergy_minus_other", np.nan),
    }


def case_has_advantage_gate_evidence(case: pd.Series, lookup: dict[str, EpisodeTrace]) -> bool:
    left = lookup.get(case["left_uid"])
    right = lookup.get(case["right_uid"])
    synergy_trace = left if left is not None and left.method == "Synergy" else right if right is not None and right.method == "Synergy" else None
    evidence = nearby_gate_evidence(synergy_trace, int(case["decision_step"]))
    return bool(evidence.get("advantageous", False))


def write_gate_evidence_report(plot_cases: pd.DataFrame, lookup: dict[str, EpisodeTrace], out_dir: Path):
    rows = [gate_evidence_row(row, lookup) for _, row in plot_cases.iterrows()]
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "gate_evidence_report.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Gate evidence report",
        "",
        "- Figures only display confidence, advantage, and visit statistics at the divergence node.",
        "- If the exact divergence frame has no advantageous gate statistics, the plot uses the nearest frame within ten steps that satisfies the confidence, advantage, and visit gates.",
        "- Frames with zero or non-advantageous statistics are retained in the CSV report but are not displayed as gate evidence in the figure.",
        "- `evidence_offset` records the replacement distance from the divergence frame. A value of `0` means exact-frame evidence.",
        "- This substitution is used only for annotation. It does not change case selection, terminal HP comparison, or plotted trajectories.",
        "",
    ]
    if not report.empty:
        status_counts = report["evidence_status"].value_counts().reset_index()
        status_counts.columns = ["evidence_status", "cases"]
        map_counts = report.groupby(["map_key", "evidence_status"]).size().reset_index(name="cases")
        adv_rate = float(report["evidence_advantageous"].mean()) if len(report) else 0.0
        lines.extend(
            [
                f"- Plotted cases: `{len(report)}`",
                f"- Cases with advantageous gate evidence: `{int(report['evidence_advantageous'].sum())}` ({adv_rate:.1%})",
                "",
                "## Evidence status counts",
                "",
                status_counts.to_markdown(index=False),
                "",
                "## Scenario by evidence status",
                "",
                map_counts.to_markdown(index=False),
            ]
        )
    else:
        lines.append("- No plotted cases.")
    (out_dir / "README_gate_evidence.md").write_text("\n".join(lines), encoding="utf-8")


def process_config(
    traces: list[EpisodeTrace],
    lookup: dict[str, EpisodeTrace],
    out_root: Path,
    prefix_len: int,
    match_type: str,
    threshold: float,
    sample_mode: str,
    include_within: bool,
    max_pairs_per_scenario: int,
    max_plots: int,
    max_plots_per_scenario: int,
    positive_synergy_only: bool,
    positive_terminal_hp_only: bool,
    changed_only: bool,
    plot_all_filtered: bool,
):
    sample_label = {
        "all300": "all300",
        "top100": "top100",
        "middle100": "mid100",
        "bottom100": "bot100",
        "small_all_large_top100": "smallAll_largeTop100",
        "small_top100_large_all": "smallTop100_largeAll",
    }.get(sample_mode, sample_mode)
    match_label = {"exact": "ex", "similar": "sim"}.get(match_type, match_type)
    label = f"{sample_label}_p{prefix_len}_{match_label}"
    if match_type == "similar":
        label += f"_thr{threshold:.3f}".replace(".", "p")
    out_dir = out_root / label
    figures = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    cases = build_cases(traces, prefix_len, match_type, threshold, include_within, max_pairs_per_scenario)
    if not cases.empty:
        cases = cases.sort_values(
            ["synergy_step_changed", "score_delta_synergy_minus_other", "prefix_distance"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    cases.to_csv(out_dir / "cases_all.csv", index=False, encoding="utf-8-sig")
    filtered_cases = cases.copy()
    if positive_synergy_only and not filtered_cases.empty:
        filtered_cases = filtered_cases[filtered_cases["score_delta_synergy_minus_other"] > 0].copy()
    if positive_terminal_hp_only and not filtered_cases.empty:
        filtered_cases = filtered_cases[filtered_cases["end_hp_delta_synergy_minus_other"] > 0].copy()
    if changed_only and not filtered_cases.empty:
        filtered_cases = filtered_cases[filtered_cases["synergy_step_changed"].fillna(False).astype(bool)].copy()
    if not filtered_cases.empty:
        filtered_cases["has_advantage_gate_evidence"] = filtered_cases.apply(
            lambda row: case_has_advantage_gate_evidence(row, lookup),
            axis=1,
        )
        if positive_terminal_hp_only:
            sort_cols = ["has_advantage_gate_evidence", "end_hp_delta_synergy_minus_other", "score_delta_synergy_minus_other", "prefix_distance"]
            ascending = [False, False, False, True]
        else:
            sort_cols = ["has_advantage_gate_evidence", "score_delta_synergy_minus_other", "prefix_distance"]
            ascending = [False, False, True]
        filtered_cases = filtered_cases.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    filtered_cases.to_csv(out_dir / "cases_selected_for_plot.csv", index=False, encoding="utf-8-sig")
    write_config_report(filtered_cases, out_dir, label)
    if cases.empty or filtered_cases.empty:
        return

    summary = cases.groupby(["map_key", "pair_type"]).agg(
        cases=("pair_type", "size"),
        synergy_changed_cases=("synergy_step_changed", "sum"),
        mean_synergy_score_delta=("score_delta_synergy_minus_other", "mean"),
        max_synergy_score_delta=("score_delta_synergy_minus_other", "max"),
    ).reset_index()
    summary.to_csv(out_dir / "summary_by_scenario.csv", index=False, encoding="utf-8-sig")

    if plot_all_filtered:
        plot_cases = filtered_cases
    elif max_plots_per_scenario > 0:
        plot_cases = (
            filtered_cases.groupby("map_key", group_keys=False)
            .head(max_plots_per_scenario)
            .reset_index(drop=True)
        )
    else:
        plot_cases = filtered_cases.head(max_plots)
    plot_cases.to_csv(out_dir / "cases_plotted.csv", index=False, encoding="utf-8-sig")
    write_gate_evidence_report(plot_cases, lookup, out_dir)
    for idx, row in plot_cases.iterrows():
        left = lookup[row["left_uid"]]
        right = lookup[row["right_uid"]]
        left_tag = "etg" if row["left_method"] == "etg-only" else "syn"
        right_tag = "etg" if row["right_method"] == "etg-only" else "syn"
        case_payload = f"{row['map_key']}|{left_tag}{int(row['left_episode_id'])}|{right_tag}{int(row['right_episode_id'])}|{row['prefix_len']}|{row['match_type']}|{row['prefix_distance']}"
        case_digest = hashlib.md5(case_payload.encode("utf-8")).hexdigest()[:8]
        case_name = f"case_{idx+1:03d}_{left_tag}{int(row['left_episode_id'])}_{right_tag}{int(row['right_episode_id'])}_{case_digest}"
        scenario_figures = figures / str(row["map_key"])
        plot_case_graph(left, right, row, scenario_figures / case_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-data-root", default=r"C:\Users\Binich\PycharmProjects\PredictionRTS\output\learner_results\all_data")
    parser.add_argument("--batch-tag", default="web_batch_20260616_212212")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--sample-mode", action="append", default=None)
    parser.add_argument("--prefix-len", action="append", type=int, default=None)
    parser.add_argument("--match-type", action="append", choices=["exact", "similar"], default=None)
    parser.add_argument("--similar-threshold", type=float, default=0.05)
    parser.add_argument("--include-within", action="store_true")
    parser.add_argument("--max-pairs-per-scenario", type=int, default=25000)
    parser.add_argument("--max-plots", type=int, default=40)
    parser.add_argument("--max-plots-per-scenario", type=int, default=0)
    parser.add_argument("--positive-synergy-only", action="store_true")
    parser.add_argument("--positive-terminal-hp-only", action="store_true")
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--plot-all-filtered", action="store_true")
    args = parser.parse_args()

    setup_style()
    root = Path(args.all_data_root)
    output_name = args.output_name or f"mechanism_sequence_divergence_cases_{args.batch_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = root / output_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("[LOAD] traces")
    all_traces = load_traces(root, args.batch_tag)
    pd.DataFrame(
        [
            {
                "uid": trace.uid,
                "method": trace.method,
                "experiment_id": trace.experiment_id,
                "map_key": trace.map_key,
                "episode_id": trace.episode_id,
                "score": trace.score,
                "result": trace.result,
                "frames": len(trace.frames),
                "changed_frames": int(sum(trace.changed)),
            }
            for trace in all_traces
        ]
    ).to_csv(out_root / "episode_trace_index.csv", index=False, encoding="utf-8-sig")

    sample_modes = args.sample_mode or ["all300", "top100", "middle100", "small_all_large_top100"]
    prefix_lens = args.prefix_len or [3, 5, 7]
    match_types = args.match_type or ["exact", "similar"]
    manifest = {
        "batch_tag": args.batch_tag,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_modes": sample_modes,
        "prefix_lens": prefix_lens,
        "match_types": match_types,
        "similar_threshold": args.similar_threshold,
        "include_within": args.include_within,
        "max_plots": args.max_plots,
        "max_plots_per_scenario": args.max_plots_per_scenario,
        "positive_synergy_only": args.positive_synergy_only,
        "positive_terminal_hp_only": args.positive_terminal_hp_only,
        "changed_only": args.changed_only,
        "plot_all_filtered": args.plot_all_filtered,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for sample_mode in sample_modes:
        selected = select_traces(all_traces, sample_mode)
        lookup = trace_lookup(selected)
        print(f"[SAMPLE] {sample_mode}: traces={len(selected)}")
        for prefix_len in prefix_lens:
            for match_type in match_types:
                print(f"[CONFIG] {sample_mode} prefix={prefix_len} match={match_type}")
                process_config(
                    selected,
                    lookup,
                    out_root,
                    prefix_len,
                    match_type,
                    args.similar_threshold,
                    sample_mode,
                    args.include_within,
                    args.max_pairs_per_scenario,
                    args.max_plots,
                    args.max_plots_per_scenario,
                    args.positive_synergy_only,
                    args.positive_terminal_hp_only,
                    args.changed_only,
                    args.plot_all_filtered,
                )

    lines = [
        "# Sequence divergence case mining",
        "",
        f"- Source batch: `{args.batch_tag}`",
        f"- Generated at: `{manifest['generated_at']}`",
        "- Each subdirectory corresponds to one sampling mode, prefix length, and matching rule.",
        "- `cases_all.csv` stores all matched divergence cases before filtering.",
        "- `cases_selected_for_plot.csv` stores cases after positive or changed-action filters.",
        "- `cases_plotted.csv` stores the cases for which transition graphs were rendered.",
        "- Node color encodes HP advantage, blue is lower and red is higher.",
        "- Star markers indicate Synergy frames where the shadow logger says the mechanism changed the action.",
        "- DTW guide-line labels use the same project BKTree state metric as the state-transition graph.",
        "- The state metric is distribution distance plus health distance with hp_weight 1.0, both returned by CustomDistance.multi_distance.",
        "",
        "## Important interpretation",
        "",
        "- These are matched trajectory case studies, not strict paired simulator rollouts.",
        "- A positive Synergy score delta indicates that the matched Synergy episode ended better than the matched comparator episode.",
        "- The transition graph is intended to identify plausible mechanism-driven divergence points for follow-up paired replay.",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_root}")


if __name__ == "__main__":
    main()
