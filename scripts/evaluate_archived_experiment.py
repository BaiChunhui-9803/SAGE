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
    _resume_game,
    _set_beam_params,
    _wait_for_file_progress,
)


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


def _run_repeat(
    learner: ParameterLearner,
    params: Dict[str, Any],
    repeat_dir: Path,
    repeat_index: int,
    episodes: int,
    source_model: Optional[Path],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    repeat_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "episodes_hp.jsonl", "progress.json", "plan.log"):
        path = repeat_dir / name
        if path.exists():
            path.unlink()

    send_params = dict(params)
    bktree_cfg = cfg.get("bktree", {}) or {}
    send_params["bktree_primary_threshold"] = float(bktree_cfg.get("primary_threshold", 1.0))
    send_params["bktree_secondary_threshold"] = float(bktree_cfg.get("secondary_threshold", 0.5))
    send_params["local_result_dir"] = str(repeat_dir)
    send_params["target_episodes"] = int(episodes)
    send_params["trial_number"] = int(repeat_index)
    send_params["plan_log_path"] = str(repeat_dir / "plan.log")

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

    sent = False
    for _ in range(5):
        if _set_beam_params(learner._port, send_params):
            sent = True
            break
        time.sleep(5)
    if not sent:
        _pause_game(learner._port)
        record["status"] = "failed_to_set_params"
        _write_json(repeat_dir / "repeat_run.json", record)
        return record

    _resume_game(learner._port)
    completed = _wait_for_file_progress(
        repeat_dir,
        int(episodes),
        cfg,
        expected_trial=int(repeat_index),
    )
    _pause_game(learner._port)

    stability_segments = int(cfg.get("objective", {}).get("stability_segments", 5))
    metrics = _analyze_local_result(
        repeat_dir,
        stability_segments,
        expected_trial=int(repeat_index),
    )
    metrics.update(_objective(metrics, cfg))

    record["status"] = "completed" if completed else "timeout"
    record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["metrics"] = metrics
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
        "seed_control": "not_available_in_current_sc2_launcher",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    _write_json(output_dir / "final_eval_summary.json", {**metadata, "repeats": []})
    _write_json(output_dir / "fixed_params.json", fixed_params)

    learner = ParameterLearner(cfg, run_dir=str(output_dir))
    repeats: List[Dict[str, Any]] = []
    try:
        learner._startup()
        time.sleep(float(args.settle_seconds))
        for i in range(1, int(args.repeats) + 1):
            repeat_dir = output_dir / "repeats" / f"repeat_{i:03d}"
            repeat = _run_repeat(
                learner,
                fixed_params,
                repeat_dir,
                i,
                int(args.episodes),
                source_model,
                cfg,
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
    finally:
        learner._shutdown()

    result = {
        **metadata,
        "status": "completed",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repeats": repeats,
        "aggregate": _aggregate(repeats),
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
    parser.add_argument("--settle-seconds", type=float, default=10.0, help="Seconds to wait after game startup")
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


if __name__ == "__main__":
    main()
