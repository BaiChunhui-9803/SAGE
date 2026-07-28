#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate an archived experiment with its fixed best parameters.

The script is intended for final, fair comparison after parameter search:
it reads an archived experiment directory, extracts the best trial parameters,
optionally reuses the archived ActionTuningModel, runs fresh episodes, and
writes an independent final-evaluation summary under the archive.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from scripts.parameter_learner import (
    ParameterLearner,
    _analyze_local_result,
    _load_config,
    _pause_game,
)
from scripts.validate_final_eval_start_state import validate_final_eval_start_state


_DEFAULT_CONFIG = ROOT_DIR / "configs" / "learner_config.yaml"
_MANIFEST_NAME = "experiment_manifest.json"
_SUMMARY_NAME = "study_summary.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    with open(str(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"json root must be an object: {path}")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{idx:03d}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"unable to allocate unique output directory near {path}")


def _is_etg_only_manifest(manifest: Dict[str, Any]) -> bool:
    experiment_type = str(manifest.get("experiment_type", "")).lower()
    method = str(manifest.get("method", "")).lower()
    return (
        "etg_only" in experiment_type
        or "etg-only" in experiment_type
        or "etg-only" in method
        or "etg only" in method
    )


def _best_trial_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    best_trial = summary.get("best_trial")
    for trial in summary.get("trials", []):
        if isinstance(trial, dict) and trial.get("number") == best_trial:
            return trial
    return {}


def _load_best_params(exp_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    best_trial = summary.get("best_trial")
    if best_trial is None:
        raise ValueError("study_summary.json does not contain best_trial")

    run_path = exp_dir / "runs" / f"trial_{int(best_trial):04d}_run.json"
    if run_path.exists():
        run_data = _read_json(run_path)
        params = run_data.get("params")
        if isinstance(params, dict) and params:
            return dict(params)

    trial = _best_trial_record(summary)
    params = trial.get("params") if isinstance(trial, dict) else None
    if isinstance(params, dict) and params:
        return dict(params)

    params = summary.get("best_params")
    if isinstance(params, dict) and params:
        return dict(params)

    raise ValueError("failed to locate best parameters from run record or summary")


def _normalize_cfg(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
    output_dir: Path,
    episodes: int,
    timeout_minutes: Optional[int],
    enable_action_tuning: bool,
) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("game", {})
    cfg.setdefault("execution", {})
    cfg.setdefault("bktree", {})
    cfg.setdefault("action_tuning", {})
    cfg.setdefault("phased_optimization", {})
    cfg.setdefault("incremental_layer", {})
    cfg.setdefault("storage", {})

    cfg["game"] = dict(cfg["game"])
    cfg["execution"] = dict(cfg["execution"])
    cfg["bktree"] = dict(cfg["bktree"])
    cfg["action_tuning"] = dict(cfg["action_tuning"])
    cfg["phased_optimization"] = dict(cfg["phased_optimization"])
    cfg["incremental_layer"] = dict(cfg["incremental_layer"])
    cfg["storage"] = dict(cfg["storage"])

    cfg["game"]["map_key"] = manifest.get("map_key", cfg["game"].get("map_key", "sce-1"))
    if manifest.get("kg_file"):
        cfg["game"]["kg_file"] = manifest["kg_file"]
    if manifest.get("data_dir"):
        cfg["game"]["data_dir"] = manifest["data_dir"]
    cfg["game"]["api_load_kg"] = False
    cfg["game"]["autopilot_mode"] = cfg["game"].get("autopilot_mode", "multi_step")

    bktree = manifest.get("bktree") if isinstance(manifest.get("bktree"), dict) else {}
    if bktree.get("primary_threshold") is not None:
        cfg["bktree"]["primary_threshold"] = float(bktree["primary_threshold"])
    if bktree.get("secondary_threshold") is not None:
        cfg["bktree"]["secondary_threshold"] = float(bktree["secondary_threshold"])

    cfg["execution"]["episodes_per_trial"] = int(episodes)
    if timeout_minutes is not None:
        cfg["execution"]["completion_timeout_minutes"] = int(timeout_minutes)
    cfg["execution"].setdefault("startup_wait_seconds", 120)
    cfg["execution"].setdefault("game_ready_wait_seconds", 240)

    cfg["action_tuning"]["enabled"] = bool(enable_action_tuning)
    cfg["phased_optimization"]["enabled"] = False
    cfg["incremental_layer"]["enabled"] = False
    cfg["storage"]["results_dir"] = str(output_dir)
    return cfg


def _prepare_params(
    params: Dict[str, Any],
    manifest: Dict[str, Any],
    enable_action_tuning: bool,
) -> Dict[str, Any]:
    fixed = dict(params)
    if _is_etg_only_manifest(manifest) or not enable_action_tuning:
        fixed["phase"] = "etg_only"
        fixed["enable_action_tuning"] = False
        fixed["tuning_force_explore"] = False
        fixed["tuning_explore_ood"] = False
        fixed["tuning_explore_rate"] = 0.0
        fixed["tuning_explore_sources"] = []
        fixed["exclude_from_parameter_optimization"] = False
        return fixed

    fixed["phase"] = "synergy"
    fixed["enable_action_tuning"] = True
    fixed.setdefault("tuning_force_explore", False)
    fixed.setdefault("tuning_explore_ood", False)
    fixed.setdefault("tuning_explore_rate", 0.0)
    fixed.setdefault("exclude_from_parameter_optimization", False)
    return fixed


def _load_param_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    path = getattr(args, "beam_param_override_file", None)
    if path:
        data = _read_json(Path(path))
        if not isinstance(data, dict):
            raise ValueError(f"beam param override file must contain a JSON object: {path}")
        overrides.update(data)
    raw = getattr(args, "beam_param_override_json", None)
    if raw:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("--beam-param-override-json must be a JSON object")
        overrides.update(data)
    return overrides


def _objective(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    obj_cfg = cfg.get("objective", {})
    alpha = float(obj_cfg.get("stability_alpha", 0.2))
    cap = float(obj_cfg.get("stability_cap", 8.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    avg_score = float(metrics.get("avg_score", 0.0))
    stability = float(metrics.get("stability", 0.0))
    stability_norm = min(stability / cap, 1.0) if cap > 0 else 0.0
    penalty_factor = max(1 - alpha * stability_norm, 0.0)
    return {
        "penalty_factor": penalty_factor,
        "objective": win_rate * avg_score * penalty_factor,
    }


def _aggregate(repeats: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [r.get("metrics", {}) for r in repeats if r.get("metrics")]
    if not metrics:
        return {}

    keys = ["win_rate", "avg_score", "score_std", "stability", "objective", "penalty_factor"]
    aggregate: Dict[str, Any] = {}
    for key in keys:
        values = [float(m.get(key, 0.0)) for m in metrics if m.get(key) is not None]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else None
        aggregate[key] = {
            "mean": mean,
            "std": std,
            "ci95": ci95,
        }

    total_episodes = int(sum(int(m.get("num_episodes", 0)) for m in metrics))
    total_wins = int(round(sum(float(m.get("win_rate", 0.0)) * int(m.get("num_episodes", 0)) for m in metrics)))
    aggregate["total_episodes"] = total_episodes
    aggregate["total_wins"] = total_wins
    aggregate["repeat_count"] = len(metrics)
    return aggregate


def _repeat_startup_cfg(cfg: Dict[str, Any], params_file: Path) -> Dict[str, Any]:
    repeat_cfg = dict(cfg)
    repeat_cfg["game"] = dict(cfg.get("game", {}) or {})
    repeat_cfg["game"]["initial_beam_params_file"] = str(params_file)
    return repeat_cfg


def _read_progress_count(progress_file: Path) -> int:
    if not progress_file.exists():
        return 0
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        return int(data.get("completed", 0) or 0)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_planning_switch_candidates(repeat_dir: Path) -> Dict[str, Any]:
    repeat_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = repeat_dir / "episodes.jsonl"
    out_csv = repeat_dir / "planning_switch_candidates.csv"
    summary_json = repeat_dir / "planning_switch_candidates_summary.json"
    if not episodes_path.exists():
        summary = {
            "status": "missing_episodes",
            "candidate_csv": "",
            "candidate_rows": 0,
            "plan_frames": 0,
            "unique_plans": 0,
        }
        _write_json(summary_json, summary)
        return summary

    rows: List[Dict[str, Any]] = []
    plan_frames = 0
    plans = set()
    switched_frames = 0
    diverged_frames = 0
    with open(str(episodes_path), "r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            frames = episode.get("frames") or []
            if not isinstance(frames, list):
                continue
            episode_id = episode.get("episode_id", line_idx)
            score = _safe_float(episode.get("score"))
            result = episode.get("result")
            for step, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    continue
                plan_id = frame.get("plan_id")
                candidates = frame.get("plan_switch_candidates") or []
                if plan_id:
                    plans.add(str(plan_id))
                if candidates:
                    plan_frames += 1
                switch_event = frame.get("switch_event") if isinstance(frame.get("switch_event"), dict) else {}
                path_event = (
                    frame.get("path_follow_event")
                    if isinstance(frame.get("path_follow_event"), dict)
                    else {}
                )
                switch_type = switch_event.get("type") or path_event.get("type")
                if switch_type in ("backup_switch_exact", "backup_switch_fuzzy"):
                    switched_frames += 1
                if str(path_event.get("type") or "").startswith("diverged"):
                    diverged_frames += 1
                for cand_idx, cand in enumerate(candidates):
                    if not isinstance(cand, dict):
                        continue
                    row = {
                        "episode_id": episode_id,
                        "step": step,
                        "score": score,
                        "result": result,
                        "frame_state_key": frame.get("state_key"),
                        "eval_state_id": frame.get("eval_state_id"),
                        "action_source": frame.get("action_source"),
                        "action_code": frame.get("action_code"),
                        "plan_id": plan_id or cand.get("plan_id"),
                        "candidate_index": cand_idx,
                        "fork_index": cand.get("fork_index"),
                        "predicted_state": cand.get("predicted_state"),
                        "main_state": cand.get("main_state"),
                        "candidate_state": cand.get("candidate_state"),
                        "main_action": cand.get("main_action"),
                        "candidate_action": cand.get("candidate_action"),
                        "backup_path_idx": cand.get("backup_path_idx"),
                        "backup_step_in_path": cand.get("backup_step_in_path"),
                        "match_type": cand.get("match_type"),
                        "remaining_actions": _csv_value(cand.get("remaining_actions")),
                        "main_path_score": cand.get("main_path_score"),
                        "candidate_path_score": cand.get("candidate_path_score"),
                        "planning_score_gain": cand.get("planning_score_gain"),
                        "main_future_reward": cand.get("main_future_reward"),
                        "candidate_future_reward": cand.get("candidate_future_reward"),
                        "planning_future_reward_gain": cand.get("planning_future_reward_gain"),
                        "main_cum_prob": cand.get("main_cum_prob"),
                        "candidate_cum_prob": cand.get("candidate_cum_prob"),
                        "main_states": _csv_value(cand.get("main_states")),
                        "candidate_states": _csv_value(cand.get("candidate_states")),
                        "main_actions": _csv_value(cand.get("main_actions")),
                        "candidate_actions": _csv_value(cand.get("candidate_actions")),
                        "switch_event_type": switch_event.get("type"),
                        "switch_event_distance": switch_event.get("distance"),
                        "switch_event_selected_backup_action": switch_event.get("selected_backup_action"),
                        "path_follow_event_type": path_event.get("type"),
                        "path_expected_state": path_event.get("expected_state"),
                        "path_actual_state": path_event.get("actual_state"),
                        "path_in_any_beam_state": path_event.get("in_any_beam_state"),
                        "path_has_backup_continuation": path_event.get("has_backup_continuation"),
                        "path_switch_distance": path_event.get("switch_distance"),
                    }
                    rows.append(row)

    if rows:
        with open(str(out_csv), "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    gains = [_safe_float(row.get("planning_score_gain")) for row in rows]
    gains = [value for value in gains if value is not None]
    future_gains = [_safe_float(row.get("planning_future_reward_gain")) for row in rows]
    future_gains = [value for value in future_gains if value is not None]
    summary = {
        "status": "ok",
        "candidate_csv": str(out_csv) if rows else "",
        "candidate_rows": len(rows),
        "plan_frames": plan_frames,
        "unique_plans": len(plans),
        "switched_frames": switched_frames,
        "diverged_frames": diverged_frames,
        "mean_planning_score_gain": float(np.mean(gains)) if gains else None,
        "positive_planning_score_gain_ratio": (
            float(sum(1 for value in gains if value > 0) / len(gains)) if gains else None
        ),
        "mean_planning_future_reward_gain": float(np.mean(future_gains)) if future_gains else None,
        "positive_future_reward_gain_ratio": (
            float(sum(1 for value in future_gains if value > 0) / len(future_gains))
            if future_gains
            else None
        ),
        "note": (
            "Rows are planning-time backup continuation candidates logged from beam paths. "
            "planning_*_gain compares a backup path with the selected main path inside the same beam plan."
        ),
    }
    _write_json(summary_json, summary)
    return summary


def _write_mechanism_shadow_summary(repeat_dir: Path) -> Dict[str, Any]:
    repeat_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = repeat_dir / "episodes.jsonl"
    rows: List[Dict[str, Any]] = []
    if not episodes_path.exists():
        return {
            "status": "missing_episodes",
            "frame_csv": "",
            "state_csv": "",
            "total_shadow_frames": 0,
        }

    with open(str(episodes_path), "r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            frames = episode.get("frames") or []
            if not isinstance(frames, list):
                continue
            episode_id = episode.get("episode_id", line_idx)
            score = _safe_float(episode.get("score"))
            result = episode.get("result")
            for step, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    continue
                shadow = frame.get("mechanism_shadow")
                if not isinstance(shadow, dict):
                    continue
                tuning = shadow.get("tuning") if isinstance(shadow.get("tuning"), dict) else {}
                nid_resolution = (
                    shadow.get("nid_resolution")
                    if isinstance(shadow.get("nid_resolution"), dict)
                    else {}
                )
                current_hp_delta = _safe_float(frame.get("hp_delta"))
                next_hp_delta = None
                observed_next_hp_gain = None
                if step + 1 < len(frames) and isinstance(frames[step + 1], dict):
                    next_hp_delta = _safe_float(frames[step + 1].get("hp_delta"))
                    if current_hp_delta is not None and next_hp_delta is not None:
                        observed_next_hp_gain = next_hp_delta - current_hp_delta
                rows.append(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "score": score,
                        "result": result,
                        "state_key": frame.get("state_key"),
                        "eval_state_id": frame.get("eval_state_id"),
                        "state_regime": shadow.get("state_regime"),
                        "nid_status": frame.get("nid_status"),
                        "nid_reason": frame.get("nid_reason"),
                        "nid_candidate": frame.get("nid_candidate"),
                        "nid_distance": frame.get("nid_distance"),
                        "nid_hp_distance": frame.get("nid_hp_distance"),
                        "nid_is_ood": frame.get("nid_is_ood"),
                        "baseline_action_code": shadow.get("baseline_action_code"),
                        "baseline_source": shadow.get("baseline_source"),
                        "selected_action_code": shadow.get("selected_action_code"),
                        "selected_source": shadow.get("selected_source"),
                        "mechanism_changed_action": shadow.get("mechanism_changed_action"),
                        "tuning_source": tuning.get("source"),
                        "tuning_reason": tuning.get("reason"),
                        "candidate_action": tuning.get("candidate_action"),
                        "candidate_visits": tuning.get("candidate_visits"),
                        "confidence": tuning.get("confidence"),
                        "advantage": tuning.get("advantage"),
                        "threshold_confidence": tuning.get("threshold_confidence"),
                        "threshold_advantage": tuning.get("threshold_advantage"),
                        "threshold_visits": tuning.get("threshold_visits"),
                        "model_total_visits": shadow.get("model_total_visits"),
                        "current_hp_delta": current_hp_delta,
                        "next_hp_delta": next_hp_delta,
                        "observed_selected_next_hp_gain": observed_next_hp_gain,
                        "my_count": frame.get("my_count"),
                        "enemy_count": frame.get("enemy_count"),
                        "game_loop": frame.get("game_loop"),
                        "plan_id": frame.get("plan_id"),
                        "plan_switch_candidate_count": len(frame.get("plan_switch_candidates") or []),
                        "switch_event_type": (
                            frame.get("switch_event", {}).get("type")
                            if isinstance(frame.get("switch_event"), dict)
                            else None
                        ),
                        "switch_event_distance": (
                            frame.get("switch_event", {}).get("distance")
                            if isinstance(frame.get("switch_event"), dict)
                            else None
                        ),
                        "switch_event_selected_backup_action": (
                            frame.get("switch_event", {}).get("selected_backup_action")
                            if isinstance(frame.get("switch_event"), dict)
                            else None
                        ),
                        "no_switch_shadow_action": (
                            frame.get("no_switch_shadow", {}).get("recommended_action")
                            if isinstance(frame.get("no_switch_shadow"), dict)
                            else None
                        ),
                        "path_follow_event_type": (
                            frame.get("path_follow_event", {}).get("type")
                            if isinstance(frame.get("path_follow_event"), dict)
                            else None
                        ),
                        "path_expected_state": (
                            frame.get("path_follow_event", {}).get("expected_state")
                            if isinstance(frame.get("path_follow_event"), dict)
                            else None
                        ),
                        "path_actual_state": (
                            frame.get("path_follow_event", {}).get("actual_state")
                            if isinstance(frame.get("path_follow_event"), dict)
                            else None
                        ),
                        "path_in_any_beam_state": (
                            frame.get("path_follow_event", {}).get("in_any_beam_state")
                            if isinstance(frame.get("path_follow_event"), dict)
                            else None
                        ),
                        "path_has_backup_continuation": (
                            frame.get("path_follow_event", {}).get("has_backup_continuation")
                            if isinstance(frame.get("path_follow_event"), dict)
                            else None
                        ),
                        "path_switch_distance": (
                            frame.get("path_follow_event", {}).get("switch_distance")
                            if isinstance(frame.get("path_follow_event"), dict)
                            else None
                        ),
                        "decision_elapsed_ms": (
                            frame.get("planning_timing", {}).get("decision_elapsed_ms")
                            if isinstance(frame.get("planning_timing"), dict)
                            else None
                        ),
                        "beam_plan_ms": (
                            frame.get("planning_timing", {}).get("beam_plan_ms")
                            if isinstance(frame.get("planning_timing"), dict)
                            else None
                        ),
                        "beam_result_count": (
                            frame.get("planning_timing", {}).get("beam_result_count")
                            if isinstance(frame.get("planning_timing"), dict)
                            else None
                        ),
                        "beam_path_count": (
                            frame.get("planning_timing", {}).get("beam_path_count")
                            if isinstance(frame.get("planning_timing"), dict)
                            else None
                        ),
                        "no_switch_shadow_elapsed_ms": (
                            frame.get("no_switch_shadow", {}).get("elapsed_ms")
                            if isinstance(frame.get("no_switch_shadow"), dict)
                            else None
                        ),
                        "paired_counterfactual_id": frame.get("paired_counterfactual_id"),
                        "nid_resolution_status": nid_resolution.get("status"),
                        "nid_resolution_distance": nid_resolution.get("distance"),
                    }
                )

    frame_csv = repeat_dir / "mechanism_shadow_frames.csv"
    state_csv = repeat_dir / "mechanism_shadow_state_summary.csv"
    summary_json = repeat_dir / "mechanism_shadow_summary.json"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(str(frame_csv), "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("state_key"))
        item = grouped.setdefault(
            key,
            {
                "state_key": key,
                "frames": 0,
                "episodes": set(),
                "changed_frames": 0,
                "ood_frames": 0,
                "selected_gain_values": [],
                "model_total_visits_values": [],
                "candidate_visits_values": [],
                "confidence_values": [],
                "advantage_values": [],
                "baseline_actions": {},
                "selected_actions": {},
            },
        )
        item["frames"] += 1
        item["episodes"].add(row.get("episode_id"))
        if str(row.get("mechanism_changed_action")).lower() == "true":
            item["changed_frames"] += 1
        if str(row.get("nid_is_ood")).lower() == "true" or row.get("state_regime") == "ood":
            item["ood_frames"] += 1
        for dst, src in [
            ("selected_gain_values", "observed_selected_next_hp_gain"),
            ("model_total_visits_values", "model_total_visits"),
            ("candidate_visits_values", "candidate_visits"),
            ("confidence_values", "confidence"),
            ("advantage_values", "advantage"),
        ]:
            value = _safe_float(row.get(src))
            if value is not None:
                item[dst].append(value)
        for dst, src in [
            ("baseline_actions", "baseline_action_code"),
            ("selected_actions", "selected_action_code"),
        ]:
            action = str(row.get(src) or "")
            if action:
                item[dst][action] = item[dst].get(action, 0) + 1

    state_rows: List[Dict[str, Any]] = []
    for item in grouped.values():
        def _mean(values: List[float]) -> Optional[float]:
            return float(np.mean(values)) if values else None

        state_rows.append(
            {
                "state_key": item["state_key"],
                "frames": item["frames"],
                "episodes": len(item["episodes"]),
                "changed_frames": item["changed_frames"],
                "changed_rate": item["changed_frames"] / max(item["frames"], 1),
                "ood_frames": item["ood_frames"],
                "mean_observed_selected_next_hp_gain": _mean(item["selected_gain_values"]),
                "mean_model_total_visits": _mean(item["model_total_visits_values"]),
                "mean_candidate_visits": _mean(item["candidate_visits_values"]),
                "mean_confidence": _mean(item["confidence_values"]),
                "mean_advantage": _mean(item["advantage_values"]),
                "top_baseline_action": max(item["baseline_actions"], key=item["baseline_actions"].get)
                if item["baseline_actions"]
                else "",
                "top_selected_action": max(item["selected_actions"], key=item["selected_actions"].get)
                if item["selected_actions"]
                else "",
            }
        )
    state_rows.sort(
        key=lambda row: (int(row.get("changed_frames") or 0), int(row.get("frames") or 0)),
        reverse=True,
    )
    if state_rows:
        with open(str(state_csv), "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(state_rows[0].keys()))
            writer.writeheader()
            writer.writerows(state_rows)

    summary = {
        "status": "ok",
        "frame_csv": str(frame_csv) if rows else "",
        "state_csv": str(state_csv) if state_rows else "",
        "total_shadow_frames": len(rows),
        "changed_frames": int(
            sum(1 for row in rows if str(row.get("mechanism_changed_action")).lower() == "true")
        ),
        "ood_shadow_frames": int(
            sum(
                1
                for row in rows
                if str(row.get("nid_is_ood")).lower() == "true" or row.get("state_regime") == "ood"
            )
        ),
        "unique_shadow_states": len(state_rows),
        "note": (
            "observed_selected_next_hp_gain is the realized next-frame HP-margin change "
            "after the actually executed action. The baseline action is logged as a "
            "shadow decision and does not have same-branch outcome unless paired replay is added."
        ),
    }
    _write_json(summary_json, summary)
    return summary


def _wait_for_repeat_progress(
    learner: ParameterLearner,
    repeat_dir: Path,
    target: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    poll_interval = float(cfg.get("execution", {}).get("completion_poll_interval", 3))
    timeout_minutes = float(cfg.get("execution", {}).get("completion_timeout_minutes", 60))
    deadline = time.time() + timeout_minutes * 60
    progress_file = repeat_dir / "progress.json"
    last_logged = 0
    last_done = -1
    last_progress_time = time.time()
    no_progress_timeout = float(
        cfg.get("execution", {}).get("no_progress_timeout_seconds", 180)
    )

    while time.time() < deadline:
        done = _read_progress_count(progress_file)
        if done != last_done:
            last_done = done
            last_progress_time = time.time()
        if done >= int(target):
            print(f"  repeat done: {done} episodes")
            return {"completed": True, "status": "completed", "completed_episodes": done}
        if done >= last_logged + 10:
            print(f"  progress: {done}/{target}")
            last_logged = done

        proc = getattr(learner, "_current_proc", None)
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                done = _read_progress_count(progress_file)
                status = "completed" if done >= int(target) else "process_exited_before_target"
                print(f"  [ERROR] game process exited rc={rc}, progress={done}/{target}")
                return {
                    "completed": done >= int(target),
                    "status": status,
                    "completed_episodes": done,
                    "process_returncode": rc,
                }
        if no_progress_timeout > 0 and time.time() - last_progress_time > no_progress_timeout:
            done = _read_progress_count(progress_file)
            print(
                f"  [ERROR] no progress for {no_progress_timeout:g}s, "
                f"progress={done}/{target}"
            )
            return {
                "completed": False,
                "status": "no_progress_timeout",
                "completed_episodes": done,
            }
        time.sleep(poll_interval)

    done = _read_progress_count(progress_file)
    print(f"  [ERROR] timeout ({timeout_minutes:g} min), progress={done}/{target}")
    return {"completed": False, "status": "timeout", "completed_episodes": done}


def _run_repeat(
    params: Dict[str, Any],
    repeat_dir: Path,
    repeat_index: int,
    episodes: int,
    source_model: Optional[Path],
    cfg: Dict[str, Any],
    eval_bktree_normalization: str,
    enable_mechanism_shadow_logging: bool,
    enable_planning_switch_logging: bool,
) -> Dict[str, Any]:
    repeat_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "episodes_hp.jsonl", "progress.json", "plan.log"):
        path = repeat_dir / name
        if path.exists():
            path.unlink()
    bktree_dir = repeat_dir / "bktree"
    if bktree_dir.exists():
        shutil.rmtree(str(bktree_dir))

    send_params = dict(params)
    bktree_cfg = cfg.get("bktree", {}) or {}
    send_params["bktree_primary_threshold"] = float(bktree_cfg.get("primary_threshold", 1.0))
    send_params["bktree_secondary_threshold"] = float(bktree_cfg.get("secondary_threshold", 0.5))
    send_params["local_result_dir"] = str(repeat_dir)
    send_params["target_episodes"] = int(episodes)
    send_params["trial_number"] = int(repeat_index)
    send_params["plan_log_path"] = str(repeat_dir / "plan.log")
    send_params["eval_bktree_mode"] = "fresh_eval"
    send_params["eval_bktree_source_mode"] = "fresh"
    send_params["eval_bktree_id_mode"] = "local_compact"
    send_params["eval_bktree_normalization"] = str(eval_bktree_normalization)
    send_params["enable_eval_bktree"] = True
    send_params["force_initial_debug_spawn"] = False
    send_params["reset_between_episodes"] = False
    send_params["validate_manual_spawn"] = False
    send_params["stop_when_target_reached"] = True
    send_params["enable_mechanism_shadow_logging"] = bool(enable_mechanism_shadow_logging)
    send_params["enable_planning_switch_logging"] = bool(enable_planning_switch_logging)
    send_params.setdefault("planning_switch_score_threshold", 0.0)
    send_params.setdefault("planning_switch_max_per_fork", 6)
    send_params.setdefault("planning_switch_max_candidates", 24)

    if source_model is not None:
        model_path = repeat_dir / "action_tuning_model.pkl"
        shutil.copy2(str(source_model), str(model_path))
        send_params["action_tuning_model_path"] = str(model_path)

    record = {
        "repeat": repeat_index,
        "target_episodes": episodes,
        "params": send_params,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    _write_json(repeat_dir / "repeat_run.json", record)
    params_file = repeat_dir / "startup_beam_params.json"
    _write_json(params_file, send_params)

    repeat_cfg = _repeat_startup_cfg(cfg, params_file)
    learner = ParameterLearner(repeat_cfg, run_dir=str(repeat_dir))
    wait_result: Dict[str, Any] = {"completed": False, "status": "not_started"}
    try:
        learner._startup()
        wait_result = _wait_for_repeat_progress(
            learner,
            repeat_dir,
            int(episodes),
            cfg,
        )
        if wait_result.get("completed"):
            _pause_game(learner._port)
    except Exception as exc:
        wait_result = {
            "completed": False,
            "status": "startup_or_runtime_error",
            "completed_episodes": _read_progress_count(repeat_dir / "progress.json"),
            "error": str(exc),
        }
        print(f"  [ERROR] repeat failed: {exc}", flush=True)
    finally:
        learner._shutdown()

    stability_segments = int(cfg.get("objective", {}).get("stability_segments", 5))
    metrics = _analyze_local_result(
        repeat_dir,
        stability_segments,
        expected_trial=int(repeat_index),
    )
    metrics.update(_objective(metrics, cfg))
    mechanism_shadow_summary = _write_mechanism_shadow_summary(repeat_dir)
    planning_switch_candidate_summary = _write_planning_switch_candidates(repeat_dir)

    record["status"] = str(wait_result.get("status") or "timeout")
    record["completed_episodes"] = int(wait_result.get("completed_episodes", 0) or 0)
    if wait_result.get("process_returncode") is not None:
        record["process_returncode"] = wait_result.get("process_returncode")
    if wait_result.get("error") is not None:
        record["error"] = wait_result.get("error")
    record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["metrics"] = metrics
    record["mechanism_shadow_summary"] = mechanism_shadow_summary
    record["planning_switch_candidate_summary"] = planning_switch_candidate_summary
    _write_json(repeat_dir / "repeat_run.json", record)
    return record


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_absolute():
        exp_dir = ROOT_DIR / exp_dir
    exp_dir = exp_dir.resolve()

    manifest = _read_json(exp_dir / _MANIFEST_NAME)
    summary = _read_json(exp_dir / _SUMMARY_NAME)
    params = _load_best_params(exp_dir, summary)

    is_etg_only = _is_etg_only_manifest(manifest)
    archived_model = exp_dir / "action_tuning_model.pkl"
    enable_action_tuning = bool(args.enable_action_tuning)
    if args.enable_action_tuning is None:
        enable_action_tuning = (not is_etg_only) and archived_model.exists()
    if enable_action_tuning and not archived_model.exists():
        raise FileNotFoundError(f"action tuning requested but model is missing: {archived_model}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "final_eval" / f"eval_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir = _unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(str(args.config))
    cfg = _normalize_cfg(
        cfg,
        manifest,
        output_dir,
        args.episodes,
        args.timeout_minutes,
        enable_action_tuning,
    )
    fixed_params = _prepare_params(params, manifest, enable_action_tuning)
    beam_param_overrides = _load_param_overrides(args)
    if beam_param_overrides:
        fixed_params.update(beam_param_overrides)

    source_model = None
    if enable_action_tuning:
        source_model = archived_model
        shutil.copy2(str(archived_model), str(output_dir / "action_tuning_model.pkl"))

    metadata = {
        "experiment_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "experiment_id": manifest.get("experiment_id", exp_dir.name),
        "method": manifest.get("method"),
        "experiment_type": manifest.get("experiment_type"),
        "map_key": manifest.get("map_key"),
        "map_id": manifest.get("map_id"),
        "kg_file": manifest.get("kg_file"),
        "data_dir": manifest.get("data_dir"),
        "best_trial": summary.get("best_trial"),
        "best_value": summary.get("best_value"),
        "episodes_per_repeat": int(args.episodes),
        "requested_repeats": int(args.repeats),
        "action_tuning_enabled": bool(enable_action_tuning),
        "action_tuning_model": str(archived_model) if enable_action_tuning else None,
        "param_injection": "startup_beam_params_file",
        "eval_bktree_mode": "fresh_eval",
        "eval_bktree_state_id_mode": "local_compact",
        "eval_bktree_normalization": str(args.eval_bktree_normalization),
        "mechanism_shadow_logging": bool(args.enable_mechanism_shadow_logging),
        "planning_switch_logging": bool(args.enable_planning_switch_logging),
        "beam_param_overrides": beam_param_overrides,
        "batch_tag": str(args.batch_tag or ""),
        "force_initial_debug_spawn": False,
        "reset_between_episodes": False,
        "validate_manual_spawn": False,
        "eval_bktree_outputs": [
            "repeats/<repeat_id>/bktree/node_log.txt",
            "repeats/<repeat_id>/bktree/state_node.txt",
            "repeats/<repeat_id>/bktree/primary_bktree.json",
            "repeats/<repeat_id>/bktree/secondary_bktree_*.json",
            "repeats/<repeat_id>/episode_start_state_validation.json",
            "repeats/<repeat_id>/episode_start_state_validation.csv",
        ],
        "seed_control": "not_available_in_current_sc2_launcher",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    _write_json(output_dir / "final_eval_summary.json", {**metadata, "repeats": []})
    _write_json(output_dir / "fixed_params.json", fixed_params)

    repeats: List[Dict[str, Any]] = []
    for i in range(1, int(args.repeats) + 1):
        repeat_dir = output_dir / "repeats" / f"repeat_{i:03d}"
        repeat = _run_repeat(
            fixed_params,
            repeat_dir,
            i,
            int(args.episodes),
            source_model,
            cfg,
            str(args.eval_bktree_normalization),
            bool(args.enable_mechanism_shadow_logging),
            bool(args.enable_planning_switch_logging),
        )
        repeats.append(repeat)
        partial = {
            **metadata,
            "status": "running",
            "repeats": repeats,
            "aggregate": _aggregate(repeats),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(output_dir / "final_eval_summary.json", partial)

    incomplete_repeats = [
        repeat
        for repeat in repeats
        if str(repeat.get("status") or "") != "completed"
        or int(repeat.get("completed_episodes") or 0) < int(args.episodes)
    ]
    result_status = "completed" if not incomplete_repeats else "failed"
    result = {
        **metadata,
        "status": result_status,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repeats": repeats,
        "aggregate": _aggregate(repeats),
        "incomplete_repeat_count": len(incomplete_repeats),
    }
    try:
        validation = validate_final_eval_start_state(output_dir, exp_dir)
        result["start_state_validation"] = {
            "status": validation.get("overall_status"),
            "baseline_count": validation.get("baseline_count"),
            "report_json": str(output_dir / "start_state_validation.json"),
            "report_md": str(output_dir / "start_state_validation.md"),
            "episode_start_report_json": validation.get("episode_start_report_json"),
            "episode_start_report_csv": validation.get("episode_start_report_csv"),
            "episode_start_analysis_json": validation.get("episode_start_analysis_json"),
            "episode_start_analysis_csv": validation.get("episode_start_analysis_csv"),
        }
    except Exception as exc:
        result["start_state_validation"] = {
            "status": "error",
            "error": str(exc),
        }
    _write_json(output_dir / "final_eval_summary.json", result)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Final evaluation for archived experiments")
    parser.add_argument("--experiment-dir", required=True, help="Archived experiment directory")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per repeat")
    parser.add_argument("--repeats", type=int, default=1, help="Number of independent repeats")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to <experiment>/final_eval/eval_<timestamp>")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG), help="Base learner config")
    parser.add_argument("--timeout-minutes", type=int, default=None, help="Completion timeout per repeat")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="Deprecated; final eval now injects fixed params before game startup",
    )
    parser.add_argument(
        "--eval-bktree-normalization",
        choices=["pymarl_compatible", "pymarl", "onpolicy", "decision", "live", "predictionrts"],
        default="decision",
        help="Normalization used only for final-eval BKTree/state-id recording",
    )
    parser.add_argument(
        "--beam-param-override-json",
        default=None,
        help="JSON object merged into fixed beam params before final eval startup",
    )
    parser.add_argument(
        "--beam-param-override-file",
        default=None,
        help="JSON file merged into fixed beam params before final eval startup",
    )
    parser.add_argument(
        "--batch-tag",
        default="",
        help="Optional batch id/tag written into final_eval_summary.json for later analysis grouping",
    )
    parser.set_defaults(enable_mechanism_shadow_logging=True)
    parser.add_argument(
        "--enable-mechanism-shadow-logging",
        dest="enable_mechanism_shadow_logging",
        action="store_true",
        help=(
            "Record per-frame no-mechanism shadow decisions for mechanism analysis. "
            "This logs candidate actions and tuning gates but does not change executed actions."
        ),
    )
    parser.add_argument(
        "--disable-mechanism-shadow-logging",
        "--no-enable-mechanism-shadow-logging",
        dest="enable_mechanism_shadow_logging",
        action="store_false",
        help="Disable per-frame no-mechanism shadow decision logging",
    )
    parser.set_defaults(enable_planning_switch_logging=True)
    parser.add_argument(
        "--enable-planning-switch-logging",
        dest="enable_planning_switch_logging",
        action="store_true",
        help="Record plan_id, switch candidates, switch events, and no-switch shadow fields",
    )
    parser.add_argument(
        "--disable-planning-switch-logging",
        "--no-enable-planning-switch-logging",
        dest="enable_planning_switch_logging",
        action="store_false",
        help="Disable planning switch diagnostic logging",
    )
    parser.add_argument(
        "--enable-action-tuning",
        dest="enable_action_tuning",
        action="store_true",
        default=None,
        help="Force enable ActionTuningModel",
    )
    parser.add_argument(
        "--disable-action-tuning",
        dest="enable_action_tuning",
        action="store_false",
        default=None,
        help="Force disable ActionTuningModel",
    )
    args = parser.parse_args()

    result = evaluate(args)
    print(json.dumps(result.get("aggregate", {}), ensure_ascii=False, indent=2))
    if result.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
