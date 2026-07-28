#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parameter Learner — 基于贝叶斯优化的 Beam Search 参数自动寻优

架构：单次启动 SC2，通过 HTTP API 热切换参数，无需每轮重启。

Usage:
    python scripts/parameter_learner.py --config configs/learner_config.yaml
    python scripts/parameter_learner.py --trials 30 --episodes 50 --kg_file MarineMicro_MvsM_4_augmented/kg_simple.pkl
"""

import sys
import os
import json
import time
import signal
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import optuna
import requests
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR

optuna.logging.set_verbosity(optuna.logging.INFO)

_DEFAULT_CONFIG = ROOT_DIR / "configs" / "learner_config.yaml"

_ETG_PARAM_KEYS = (
    "action_strategy",
    "mode",
    "beam_width",
    "lookahead_steps",
    "score_mode",
    "min_visits",
    "max_state_revisits",
    "min_cum_prob",
    "discount_factor",
    "enable_backup",
    "backup_score_threshold",
    "backup_distance_threshold",
    "epsilon",
    "masked_actions",
)


def _find_free_port(exclude=None):
    exclude = set(exclude or [])
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if port not in exclude:
                return port
    raise RuntimeError("Cannot find free port")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _tail_file(path: Path, max_lines: int = 80) -> str:
    try:
        with open(str(path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except Exception as e:
        return f"<failed to read log tail: {e}>"


def _terminate_process_tree(proc: subprocess.Popen):
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_for_server(
    port: int,
    timeout: int = 120,
    proc: subprocess.Popen = None,
    log_path: Path = None,
) -> bool:
    url = f"http://127.0.0.1:{port}/game/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            print(
                f"  [ERROR] server process exited before startup "
                f"(returncode={proc.returncode})"
            )
            if log_path is not None:
                print("  [ERROR] child log tail:")
                print(_tail_file(log_path))
            return False
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        except requests.RequestException:
            pass
        time.sleep(1)
    if log_path is not None:
        print(f"  [ERROR] server startup timeout after {timeout}s; child log tail:")
        print(_tail_file(log_path))
    return False


def _wait_for_game_ready(
    port: int,
    timeout: int = 180,
    proc: subprocess.Popen = None,
    log_path: Path = None,
) -> bool:
    url = f"http://127.0.0.1:{port}/game/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            print(
                f"  [ERROR] game process exited before ready "
                f"(returncode={proc.returncode})"
            )
            if log_path is not None:
                print("  [ERROR] child log tail:")
                print(_tail_file(log_path))
            return False
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                status = r.json()
                if status.get("running") is True:
                    return True
        except requests.RequestException:
            pass
        except ValueError:
            pass
        time.sleep(1)
    if log_path is not None:
        print(f"  [ERROR] game startup timeout after {timeout}s; child log tail:")
        print(_tail_file(log_path))
    return False


def _save_results(
    port: int, cfg: dict, start_count: int = 0, target_episodes: int = 100
) -> dict:
    url = f"http://127.0.0.1:{port}/game/results/save"
    timeout = cfg["execution"].get("save_timeout_seconds", 30)
    try:
        r = requests.post(
            url,
            json={"start_count": start_count, "target_episodes": target_episodes},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()
        else:
            print(f"  [WARN] save failed: {r.status_code} {r.text[:200]}")
            return {}
    except requests.RequestException as e:
        print(f"  [WARN] save error: {e}")
        return {}


def _set_beam_params(port: int, params: dict) -> bool:
    url = f"http://127.0.0.1:{port}/game/beam_params"
    try:
        r = requests.post(url, json=params, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _pause_game(port: int):
    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "pause"},
            timeout=5,
        )
    except Exception:
        pass


def _resume_game(port: int):
    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "resume"},
            timeout=5,
        )
    except Exception:
        pass


def _confirm_params_applied(
    port: int, expected_trial: int, timeout: float = 10.0
) -> bool:
    url = f"http://127.0.0.1:{port}/game/param_confirm"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                data = r.json()
                confirmed = data.get("confirmed")
                if confirmed == expected_trial:
                    return True
        except requests.RequestException:
            pass
        time.sleep(0.2)
    return False


def _wait_for_file_progress(
    trial_dir: Path, target: int, cfg: dict, expected_trial: int = -1
) -> bool:
    poll_interval = cfg["execution"].get("completion_poll_interval", 3)
    timeout_minutes = cfg["execution"].get("completion_timeout_minutes", 60)
    deadline = time.time() + timeout_minutes * 60
    progress_file = trial_dir / "progress.json"
    ep_file = trial_dir / "episodes.jsonl"
    last_logged = 0
    first_checked = False

    while time.time() < deadline:
        try:
            if not first_checked and expected_trial >= 0 and ep_file.exists():
                first_checked = True
                with open(str(ep_file), "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if first_line:
                    rec = json.loads(first_line)
                    tn = rec.get("trial_number")
                    if tn is not None and tn != expected_trial:
                        print(
                            f"  [WARN] first episode trial_number={tn}, expected={expected_trial}, truncating stale data"
                        )
                        valid_lines = []
                        with open(str(ep_file), "r", encoding="utf-8") as rf:
                            for ln in rf:
                                ln_s = ln.strip()
                                if not ln_s:
                                    continue
                                try:
                                    r = json.loads(ln_s)
                                    if r.get("trial_number") in (
                                        expected_trial,
                                        None,
                                    ):
                                        valid_lines.append(ln_s)
                                except json.JSONDecodeError:
                                    pass
                        with open(str(ep_file), "w", encoding="utf-8") as wf:
                            for valid_ln in valid_lines:
                                wf.write(valid_ln + "\n")
                        progress_file.write_text(
                            json.dumps({"completed": len(valid_lines)}),
                            encoding="utf-8",
                        )

            if progress_file.exists():
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                done = data.get("completed", 0)
                if done >= target:
                    print(f"  trial done: {done} episodes")
                    return True
                if done >= last_logged + 10:
                    print(f"  progress: {done}/{target}")
                    last_logged = done
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(poll_interval)

    print(f"  [ERROR] timeout ({timeout_minutes} min)")
    return False


def _compute_stability(episodes_results: list, num_segments: int) -> float:
    if len(episodes_results) < num_segments:
        return 0.0
    n = len(episodes_results)
    seg_size = n // num_segments
    if seg_size < 1:
        return 0.0

    win_rates = []
    avg_scores = []
    for i in range(num_segments):
        seg = episodes_results[i * seg_size : (i + 1) * seg_size]
        if not seg:
            continue
        win_rates.append(sum(1 for r in seg if r.get("result") == "Win") / len(seg))
        avg_scores.append(float(np.mean([r.get("score", 0) for r in seg])))

    wr_std = float(np.std(win_rates)) if win_rates else 0.0
    sc_std = float(np.std(avg_scores)) if avg_scores else 0.0
    return wr_std + sc_std


def _analyze_local_result(
    trial_dir: Path, num_segments: int, expected_trial: int = -1
) -> dict:
    ep_file = trial_dir / "episodes.jsonl"
    if not ep_file.exists():
        return {
            "win_rate": 0.0,
            "avg_score": 0.0,
            "score_std": 0.0,
            "stability": 0.0,
            "num_episodes": 0,
        }

    episodes = []
    with open(str(ep_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ep = json.loads(line)
                    if expected_trial >= 0:
                        tn = ep.get("trial_number")
                        if tn is not None and tn != expected_trial:
                            continue
                    episodes.append(ep)
                except json.JSONDecodeError:
                    continue

    if not episodes:
        return {
            "win_rate": 0.0,
            "avg_score": 0.0,
            "score_std": 0.0,
            "stability": 0.0,
            "num_episodes": 0,
        }

    wins = sum(1 for ep in episodes if ep.get("result") == "Win")
    scores = [ep.get("score", 0) for ep in episodes]
    stability = _compute_stability(episodes, num_segments)

    return {
        "win_rate": wins / len(episodes),
        "avg_score": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "stability": stability,
        "num_episodes": len(episodes),
    }


def _sample_params(trial: optuna.Trial, space: dict, cfg: dict = None) -> dict:
    action_strategy = trial.suggest_categorical(
        "action_strategy", space["action_strategy"]
    )
    agent_mode = trial.suggest_categorical("mode", space["mode"])

    params = {
        "beam_width": trial.suggest_int(
            "beam_width", space["beam_width"][0], space["beam_width"][1]
        ),
        "lookahead_steps": trial.suggest_int(
            "lookahead_steps", space["lookahead_steps"][0], space["lookahead_steps"][1]
        ),
        "score_mode": trial.suggest_categorical("score_mode", space["score_mode"]),
        "action_strategy": action_strategy,
        "mode": agent_mode,
        "min_visits": trial.suggest_int(
            "min_visits", space["min_visits"][0], space["min_visits"][1]
        ),
        "max_state_revisits": trial.suggest_int(
            "max_state_revisits",
            space["max_state_revisits"][0],
            space["max_state_revisits"][1],
        ),
        "min_cum_prob": trial.suggest_float(
            "min_cum_prob", space["min_cum_prob"][0], space["min_cum_prob"][1], log=True
        ),
        "discount_factor": trial.suggest_float(
            "discount_factor", space["discount_factor"][0], space["discount_factor"][1]
        ),
    }

    if agent_mode == "multi_step":
        enable_backup = trial.suggest_categorical("enable_backup", [True, False])
        params["enable_backup"] = enable_backup

        if enable_backup:
            params["backup_score_threshold"] = trial.suggest_float(
                "backup_score_threshold",
                space["backup_score_threshold"][0],
                space["backup_score_threshold"][1],
            )
            params["backup_distance_threshold"] = trial.suggest_float(
                "backup_distance_threshold",
                space["backup_distance_threshold"][0],
                space["backup_distance_threshold"][1],
            )
        else:
            params["backup_score_threshold"] = 0.3
            params["backup_distance_threshold"] = 0.2
    else:
        params["enable_backup"] = False
        params["backup_score_threshold"] = 0.3
        params["backup_distance_threshold"] = 0.2

    if action_strategy == "epsilon_greedy":
        params["epsilon"] = trial.suggest_float(
            "epsilon", space["epsilon"][0], space["epsilon"][1]
        )
    else:
        params["epsilon"] = 0.1

    if "masked_count" in space:
        masked_count = trial.suggest_int(
            "masked_count", space["masked_count"][0], space["masked_count"][1]
        )
        _ACTION_LETTERS = list("abcdefghijk")
        masked_letters = []
        for i in range(masked_count):
            choice = trial.suggest_int(f"mask_{i}", 0, len(_ACTION_LETTERS) - 1)
            letter = _ACTION_LETTERS[choice]
            if letter not in masked_letters:
                masked_letters.append(letter)
        masked_actions = []
        for letter in masked_letters:
            for c in range(5):
                masked_actions.append(f"{c}{letter}")
        params["masked_actions"] = masked_actions
    else:
        params["masked_actions"] = []

    if cfg and cfg.get("_fixed_masked_actions"):
        params["masked_actions"] = cfg["_fixed_masked_actions"]

    return params


def _fixed_etg_params_from_cfg(cfg: dict) -> dict:
    search_space = cfg.get("search_space", {})
    best_params = cfg.get("game", {}).get("best_params", {}) or {}
    params = {}
    defaults = {
        "action_strategy": "best_beam",
        "mode": "multi_step",
        "beam_width": 3,
        "lookahead_steps": 5,
        "score_mode": "quality",
        "min_visits": 1,
        "max_state_revisits": 2,
        "min_cum_prob": 0.01,
        "discount_factor": 0.9,
        "enable_backup": False,
        "backup_score_threshold": 0.3,
        "backup_distance_threshold": 0.2,
        "epsilon": 0.1,
        "masked_actions": [],
    }
    for key, default in defaults.items():
        if key in best_params:
            params[key] = best_params[key]
        elif key in search_space and isinstance(search_space[key], list) and search_space[key]:
            values = search_space[key]
            params[key] = values[0] if not all(isinstance(v, (int, float)) for v in values) else values[len(values) // 2]
        else:
            params[key] = default
    return params


class ParameterLearner:
    def __init__(self, cfg: dict, run_dir: str = None):
        self.cfg = cfg
        if run_dir:
            self.results_dir = Path(run_dir)
            self.runs_dir = self.results_dir / "runs"
            self.trials_dir = self.results_dir / "trials"
        else:
            self.results_dir = Path(
                cfg["storage"].get("results_dir", "output/learner_results")
            )
            self.runs_dir = self.results_dir / "runs"
            self.trials_dir = self.results_dir / "trials"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self._all_metrics = []
        self._current_proc = None
        self._port = None
        self._source_trial = None
        self._override_cfg = self.cfg.get("action_override", {})
        self._cf_enabled = self._override_cfg.get("enabled", False)
        self._finetune_interval = self._override_cfg.get("finetune_interval", 25)
        self._override_model_path = self.results_dir / "action_override_model.pkl"
        self._tuning_cfg = self.cfg.get("action_tuning", {})
        self._action_tuning_enabled = self._tuning_cfg.get("enabled", False)
        self._action_tuning_model_path = self.results_dir / "action_tuning_model.pkl"
        self._incremental_cfg = self.cfg.get("incremental_layer", {})
        self._phase_cfg = self.cfg.get("phased_optimization", {}) or {}
        self._phase_enabled = bool(self._phase_cfg.get("enabled", False))
        self._current_phase = (
            str(self._phase_cfg.get("default_phase", "synergy"))
            if self._phase_enabled
            else "etg_only"
        )
        self._phase_history = []
        stages = self._phase_cfg.get("stages", []) or []
        self._adaptive_phase = str(stages[0].get("name", "etg_only")) if stages else "synergy"
        self._adaptive_phase_start = 0
        self._best_etg_params = None
        self._best_etg_value = -float("inf")
        self._etg_param_pool = []
        self._fixed_etg_param_pool = self._load_fixed_etg_param_pool()

    def _load_fixed_etg_param_pool(self) -> list:
        raw_pool = (
            self._phase_cfg.get("fixed_etg_param_pool")
            or self._phase_cfg.get("fixed_synergy_etg_params")
            or []
        )
        if not isinstance(raw_pool, list):
            return []
        pool = []
        for idx, item in enumerate(raw_pool):
            if not isinstance(item, dict):
                continue
            raw_params = item.get("params")
            if not isinstance(raw_params, dict):
                raw_params = item.get("override")
            if not isinstance(raw_params, dict):
                continue
            params = {k: raw_params[k] for k in _ETG_PARAM_KEYS if k in raw_params}
            if not params:
                continue
            params["mode"] = "multi_step"
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            value = metrics.get("avg_score", item.get("value", item.get("score", 0.0)))
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            pool.append(
                {
                    "trial": int(item.get("rank", idx + 1) or idx + 1),
                    "value": value,
                    "params": params,
                    "source": str(item.get("name") or item.get("selection_reason") or "fixed_pool"),
                }
            )
        pool.sort(key=lambda entry: entry.get("value", -float("inf")), reverse=True)
        if pool:
            print(f"  [PHASE] loaded fixed ETG param pool: {len(pool)} candidates", flush=True)
        return pool

    def _select_synergy_etg_params(self) -> dict:
        if not self._phase_cfg.get("synergy_use_best_etg_params", True):
            return {}
        source_pool = self._fixed_etg_param_pool or self._etg_param_pool
        if not source_pool:
            return self._best_etg_params or _fixed_etg_params_from_cfg(self.cfg)
        pool_size = int(self._phase_cfg.get("synergy_etg_pool_size", 3) or 1)
        pool = source_pool[: max(pool_size, 1)]
        selection = str(self._phase_cfg.get("synergy_etg_selection", "weighted"))
        if selection == "best":
            item = pool[0]
        elif selection == "weighted":
            weights = self._phase_cfg.get("synergy_etg_weights", [0.6, 0.25, 0.15])
            if not isinstance(weights, list) or not weights:
                weights = [0.6, 0.25, 0.15]
            numeric_weights = []
            for idx in range(len(pool)):
                try:
                    numeric_weights.append(float(weights[idx]))
                except Exception:
                    numeric_weights.append(0.0)
            if sum(numeric_weights) <= 0:
                numeric_weights = [1.0 / len(pool)] * len(pool)
            total_weight = sum(numeric_weights)
            if self._fixed_etg_param_pool:
                used = len(
                    [
                        h
                        for h in self._phase_history
                        if h["phase"] in ("exploration_only", "synergy")
                    ]
                )
            else:
                used = len([h for h in self._phase_history if h["phase"] == "synergy"])
            slot = ((used * 37) % 100) / 100.0
            cumulative = 0.0
            item = pool[-1]
            for candidate, weight in zip(pool, numeric_weights):
                cumulative += weight / total_weight
                if slot <= cumulative:
                    item = candidate
                    break
        else:
            if self._fixed_etg_param_pool:
                used = len(
                    [
                        h
                        for h in self._phase_history
                        if h["phase"] in ("exploration_only", "synergy")
                    ]
                )
            else:
                used = len([h for h in self._phase_history if h["phase"] == "synergy"])
            item = pool[used % len(pool)]
        params = dict(item.get("params", {}) or {})
        params["synergy_etg_source_trial"] = int(item.get("trial", -1))
        params["synergy_etg_source_value"] = float(item.get("value", 0.0))
        return params

    def _stage_cfg(self, phase: str) -> dict:
        for stage in self._phase_cfg.get("stages", []) or []:
            if str(stage.get("name", "")) == phase:
                return stage
        return {}

    def _next_phase(self, phase: str) -> str:
        stages = [str(s.get("name", "")) for s in self._phase_cfg.get("stages", []) or [] if s.get("name")]
        if not stages:
            return "synergy"
        if phase not in stages:
            return stages[0]
        return stages[(stages.index(phase) + 1) % len(stages)]

    def _phase_for_index(self, index: int) -> str:
        if not self._phase_enabled:
            return "etg_only"
        if str(self._phase_cfg.get("mode", "cycle")) == "adaptive":
            return self._adaptive_phase
        stages = self._phase_cfg.get("stages", []) or []
        if self._phase_cfg.get("cycle", True):
            cycle_len = sum(int(stage.get("trials", 0) or 0) for stage in stages)
            if cycle_len > 0:
                index = index % cycle_len
        cursor = 0
        for stage in stages:
            name = str(stage.get("name", "synergy"))
            trials = int(stage.get("trials", 0) or 0)
            if trials <= 0:
                continue
            if cursor <= index < cursor + trials:
                return name
            cursor += trials
        return str(self._phase_cfg.get("default_phase", "synergy"))

    def _apply_phase_to_params(self, params: dict, phase: str) -> dict:
        phased = dict(params)
        phased["tuning_ood_key_mode"] = str(
            self._tuning_cfg.get("ood_key_mode", "aggregate")
        )
        phased["tuning_ood_distance_bucket"] = float(
            self._tuning_cfg.get("ood_distance_bucket", 0.5)
        )
        if phase == "etg_only":
            phased["enable_action_tuning"] = False
            phased["tuning_force_explore"] = False
            phased["tuning_explore_ood"] = False
            phased["exclude_from_parameter_optimization"] = False
            phased["tuning_explore_rate"] = 0.0
            phased["tuning_explore_sources"] = []
            phased["phase"] = phase
        elif phase == "exploration_only":
            phased.update(self._select_synergy_etg_params())
            phased["enable_action_tuning"] = True
            phased["tuning_force_explore"] = True
            phased["tuning_explore_ood"] = True
            phased["exclude_from_parameter_optimization"] = bool(
                self._phase_cfg.get("exclude_exploration_from_optimization", True)
            )
            phased["tuning_explore_rate"] = max(
                float(self._tuning_cfg.get("explore_rate", 0.05)),
                float(self._phase_cfg.get("exploration_min_rate", 0.20)),
            )
            phased["tuning_explore_sources"] = [
                "kg_plan",
                "kg_follow",
                "fallback",
                "ft_plan",
                "kg_relaxed",
                "fuzzy_plan",
                "ood",
            ]
            phased["phase"] = phase
        else:
            phased.update(self._select_synergy_etg_params())
            phased["enable_action_tuning"] = bool(self._action_tuning_enabled)
            phased["tuning_force_explore"] = False
            phased["tuning_explore_ood"] = False
            phased["tuning_etg_first"] = bool(
                self._phase_cfg.get("synergy_etg_first", True)
            )
            phased["tuning_etg_protected_sources"] = list(
                self._phase_cfg.get(
                    "synergy_etg_protected_sources",
                    ["kg_plan", "kg_follow"],
                )
                or []
            )
            phased["exclude_from_parameter_optimization"] = False
            phased["tuning_explore_rate"] = float(
                self._phase_cfg.get("synergy_explore_rate", 0.0)
            )
            phased["tuning_explore_sources"] = list(
                self._phase_cfg.get("synergy_explore_sources", []) or []
            )
            phased["tuning_validation_sources"] = list(
                self._phase_cfg.get(
                    "synergy_validation_sources",
                    ["ood", "fallback", "kg_relaxed"],
                )
                or []
            )
            phased["tuning_validation_min_confidence"] = float(
                self._phase_cfg.get(
                    "synergy_validation_min_confidence",
                    self._tuning_cfg.get("validation_min_confidence", 0.35),
                )
            )
            phased["tuning_validation_min_advantage"] = float(
                self._phase_cfg.get(
                    "synergy_validation_min_advantage",
                    self._tuning_cfg.get("validation_min_advantage", 5.0),
                )
            )
            phased["tuning_validation_min_visits"] = int(
                self._phase_cfg.get(
                    "synergy_validation_min_visits",
                    self._tuning_cfg.get("validation_min_visits", 8),
                )
            )
            phased["tuning_validation_profiles"] = dict(
                self._phase_cfg.get(
                    "synergy_validation_profiles",
                    {
                        "ood": {
                            "min_confidence": 0.30,
                            "min_advantage": 4.0,
                            "min_visits": 8,
                        },
                        "fallback": {
                            "min_confidence": 0.30,
                            "min_advantage": 4.0,
                            "min_visits": 8,
                        },
                        "kg_relaxed": {
                            "min_confidence": 0.45,
                            "min_advantage": 8.0,
                            "min_visits": 12,
                        },
                        "diverge": {
                            "min_confidence": 0.45,
                            "min_advantage": 8.0,
                            "min_visits": 12,
                        },
                        "fuzzy_plan": {
                            "min_confidence": 0.45,
                            "min_advantage": 8.0,
                            "min_visits": 12,
                        },
                    },
                )
                or {}
            )
            phased["phase"] = phase
        return phased

    def _apply_non_phased_params(self, params: dict) -> dict:
        phased = dict(params)
        phased["tuning_ood_key_mode"] = str(
            self._tuning_cfg.get("ood_key_mode", "aggregate")
        )
        phased["tuning_ood_distance_bucket"] = float(
            self._tuning_cfg.get("ood_distance_bucket", 0.5)
        )
        phased["enable_action_tuning"] = False
        phased["tuning_force_explore"] = False
        phased["tuning_explore_ood"] = False
        phased["exclude_from_parameter_optimization"] = False
        phased["tuning_explore_rate"] = 0.0
        phased["tuning_explore_sources"] = []
        phased["phase"] = "etg_only"
        return phased

    def _record_phase_result(self, phase: str, value: float, metrics: dict, trial_number: int) -> None:
        self._phase_history.append(
            {
                "phase": phase,
                "value": float(value),
                "avg_score": float(metrics.get("avg_score", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "trial": int(trial_number),
            }
        )
        if phase == "etg_only":
            run_path = self.runs_dir / f"trial_{trial_number:04d}_run.json"
            try:
                data = json.loads(run_path.read_text(encoding="utf-8"))
                params = data.get("params", {}) or {}
                etg_params = {k: params[k] for k in _ETG_PARAM_KEYS if k in params}
                pool_item = {
                    "trial": int(trial_number),
                    "value": float(value),
                    "params": dict(etg_params),
                }
                self._etg_param_pool = [
                    item for item in self._etg_param_pool if item.get("trial") != trial_number
                ]
                self._etg_param_pool.append(pool_item)
                self._etg_param_pool.sort(key=lambda item: item.get("value", -float("inf")), reverse=True)
                pool_size = int(self._phase_cfg.get("synergy_etg_pool_size", 3) or 1)
                self._etg_param_pool = self._etg_param_pool[: max(pool_size, 1)]
                if value > self._best_etg_value:
                    self._best_etg_value = float(value)
                    self._best_etg_params = dict(etg_params)
                    print(
                        f"  [PHASE] best ETG params updated from trial {trial_number}: {value:.4f}",
                        flush=True,
                    )
            except Exception:
                pass
        if str(self._phase_cfg.get("mode", "cycle")) != "adaptive":
            return
        stage = self._stage_cfg(phase)
        max_trials = int(stage.get("max_trials", stage.get("trials", 50)) or 50)
        min_trials = int(stage.get("min_trials", max(1, min(max_trials, 10))) or 1)
        phase_items = [h for h in self._phase_history if h["phase"] == phase and h["trial"] >= self._adaptive_phase_start]
        count = len(phase_items)
        if count < min_trials:
            return
        should_advance = count >= max_trials
        if phase == "etg_only":
            target = float(stage.get("target_objective", self._phase_cfg.get("etg_target_objective", 35.0)))
            should_advance = should_advance or max(h["value"] for h in phase_items) >= target
        elif phase == "exploration_only":
            target = float(stage.get("target_avg_score", self._phase_cfg.get("exploration_target_avg_score", 10.0)))
            recent = phase_items[-min(5, len(phase_items)):]
            should_advance = should_advance or float(np.mean([h["avg_score"] for h in recent])) >= target
        if should_advance:
            next_phase = self._next_phase(phase)
            print(f"  [PHASE] {phase} -> {next_phase} after {count} trials", flush=True)
            self._adaptive_phase = next_phase
            self._adaptive_phase_start = trial_number + 1

    def run(self, n_trials: int = None, resume: bool = False):
        total = n_trials or self.cfg["execution"]["total_trials"]
        db_path = self.results_dir / "study.db"
        study_db = f"sqlite:///{db_path}"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if resume:
            study = optuna.load_study(study_name="beam_search", storage=study_db)
            print(f"loaded study, {len(study.trials)} trials completed")
        else:
            study = optuna.create_study(
                study_name="beam_search",
                storage=study_db,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                load_if_exists=True,
            )
            print(f"new study: {study_db}")

        existing_batch = 0
        for t in study.trials:
            b = t.user_attrs.get("batch", 0)
            if isinstance(b, int) and b > existing_batch:
                existing_batch = b
        self._batch = existing_batch + 1
        print(f"  current batch: {self._batch}")

        signal.signal(signal.SIGINT, self._signal_handler)

        restart_interval = self.cfg["execution"].get("restart_interval", 0)
        restart_on_phase_change = bool(
            self.cfg["execution"].get("restart_on_phase_change", True)
        )
        last_phase = None

        try:
            self._startup()
            for i in range(total):
                self._current_phase = self._phase_for_index(i)
                if (
                    restart_on_phase_change
                    and last_phase is not None
                    and self._current_phase != last_phase
                ):
                    completed_so_far = len(
                        [t for t in study.trials if t.state.name == "COMPLETE"]
                    )
                    print(
                        f"  [PHASE-RESTART] {last_phase} -> {self._current_phase}; "
                        f"{completed_so_far} completed trials, restarting game client..."
                    )
                    self._shutdown()
                    time.sleep(2)
                    self._startup()
                if restart_interval > 0 and i > 0 and i % restart_interval == 0:
                    completed_so_far = len(
                        [t for t in study.trials if t.state.name == "COMPLETE"]
                    )
                    print(
                        f"  [RESTART] {completed_so_far} trials completed, "
                        f"restarting game client..."
                    )
                    self._shutdown()
                    time.sleep(2)
                    self._startup()
                last_phase = self._current_phase
                study.optimize(
                    lambda trial: self._objective(trial, study),
                    n_trials=1,
                    show_progress_bar=False,
                )
                if (
                    self._cf_enabled
                    and self._finetune_interval > 0
                    and (i + 1) % self._finetune_interval == 0
                ):
                    self._run_finetune_phase(i + 1, study)
        except KeyboardInterrupt:
            print("\ninterrupted, saving progress...")
        finally:
            self._shutdown()

        self._print_best(study)
        self._save_summary(study)

    def _startup(self):
        self._action_tuning_enabled = self._tuning_cfg.get("enabled", False) or bool(
            self._phase_cfg.get("enabled", False)
        )
        port = _find_free_port(exclude={8000, 8501, 8502})
        self._port = port

        game = self.cfg.get("game", {})
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_live_game.py"),
            "--mode",
            "all",
            "--port",
            str(port),
            "--map_key",
            game.get("map_key", "sce-1"),
            "--max_episodes",
            "0",
            "--autopilot_mode",
            game.get("autopilot_mode", "multi_step"),
        ]
        if game.get("kg_file"):
            cmd.extend(["--kg_file", game["kg_file"]])
        if game.get("data_dir"):
            cmd.extend(["--data_dir", game["data_dir"]])
        if not bool(game.get("api_load_kg", False)):
            cmd.append("--skip_api_kg")
        if game.get("fallback_action"):
            cmd.extend(["--fallback_action", game["fallback_action"]])
        if game.get("initial_beam_params_file"):
            cmd.extend(["--beam_params_file", str(game["initial_beam_params_file"])])
        if game.get("initial_beam_params_json"):
            cmd.extend(["--beam_params_json", str(game["initial_beam_params_json"])])
        bktree_cfg = self.cfg.get("bktree", {}) or {}
        primary_threshold = float(bktree_cfg.get("primary_threshold", 1.0))
        secondary_threshold = float(bktree_cfg.get("secondary_threshold", 0.5))
        cmd.extend(["--primary_threshold", str(primary_threshold)])
        cmd.extend(["--secondary_threshold", str(secondary_threshold)])
        if self._action_tuning_enabled:
            cmd.append("--enable_action_tuning")
            cmd.extend(["--action_tuning_model_path", str(self._action_tuning_model_path)])
            cmd.extend(
                [
                    "--tuning_explore_rate",
                    str(self._tuning_cfg.get("explore_rate", 0.05)),
                    "--tuning_min_confidence",
                    str(self._tuning_cfg.get("min_confidence", 0.35)),
                    "--tuning_min_advantage",
                    str(self._tuning_cfg.get("min_advantage", 1.0)),
                    "--tuning_ucb_c",
                    str(self._tuning_cfg.get("ucb_c", 1.4)),
                    "--tuning_target_visits",
                    str(self._tuning_cfg.get("target_visits", 10)),
                    "--tuning_min_visits",
                    str(self._tuning_cfg.get("min_visits", 3)),
                    "--tuning_credit_mode",
                    str(self._tuning_cfg.get("credit_mode", "every_visit")),
                    "--tuning_discount_factor",
                    str(self._tuning_cfg.get("discount_factor", 0.95)),
                    "--tuning_outcome_bonus",
                    str(self._tuning_cfg.get("outcome_bonus", 50.0)),
                    "--tuning_confidence_return_scale",
                    str(self._tuning_cfg.get("confidence_return_scale", 50.0)),
                    "--tuning_ood_key_mode",
                    str(self._tuning_cfg.get("ood_key_mode", "aggregate")),
                    "--tuning_ood_distance_bucket",
                    str(self._tuning_cfg.get("ood_distance_bucket", 0.5)),
                    "--max_nid_fallback_dist",
                    str(self._tuning_cfg.get("max_nid_fallback_dist", 0.75)),
                    "--max_nid_fallback_hp_dist",
                    str(self._tuning_cfg.get("max_nid_fallback_hp_dist", 1.5)),
                ]
            )
            guard_cfg = self._tuning_cfg.get("restart_guard", {}) or {}
            if guard_cfg.get("enabled", True):
                cmd.append("--restart_guard_enabled")
                cmd.extend(
                    [
                        "--restart_warmup_episodes",
                        str(guard_cfg.get("warmup_episodes", 10)),
                        "--restart_guard_max_ood_ratio",
                        str(guard_cfg.get("max_ood_ratio", 0.30)),
                        "--restart_guard_max_ood_mc_ratio",
                        str(guard_cfg.get("max_ood_mc_ratio", 0.30)),
                        "--restart_guard_max_episode_frames",
                        str(guard_cfg.get("max_episode_frames", 80)),
                        "--restart_guard_high_score_ood_min_score",
                        str(guard_cfg.get("high_score_ood_min_score", 24.0)),
                    ]
                )
                if guard_cfg.get("allow_high_score_ood_update", True):
                    cmd.append("--restart_guard_allow_high_score_ood_update")
                if guard_cfg.get("skip_model_update", True):
                    cmd.append("--restart_guard_skip_model_update")
                if guard_cfg.get("skip_bad_results", True):
                    cmd.append("--restart_guard_skip_bad_results")
                if guard_cfg.get("disable_ood_explore_on_violation", True):
                    cmd.append("--restart_guard_disable_ood_explore")
        if self._incremental_cfg.get("enabled", False):
            cmd.append("--enable_incremental_layer")
            if self._incremental_cfg.get("update_bktree", False):
                cmd.append("--incremental_update_bktree")
            if self._incremental_cfg.get("update_etg_delta", False):
                cmd.append("--incremental_update_etg_delta")
            if self._incremental_cfg.get("use_delta_for_planning", False):
                cmd.append("--incremental_use_delta_for_planning")
            delta_dir = self._incremental_cfg.get(
                "delta_dir", str(self.results_dir / "incremental_layer")
            )
            if not Path(delta_dir).is_absolute():
                delta_dir = str(self.results_dir / delta_dir)
            cmd.extend(["--incremental_delta_dir", delta_dir])
            cmd.extend(
                [
                    "--incremental_persist_interval",
                    str(self._incremental_cfg.get("persist_interval_episodes", 10)),
                    "--incremental_min_new_state_distance",
                    str(self._incremental_cfg.get("min_new_state_distance", 1.0)),
                ]
            )

        log_path = self.trials_dir / "learner.log"
        log_file = open(str(log_path), "a", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32":
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        self._current_proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=str(ROOT_DIR),
            creationflags=flags,
        )

        print(f"  SC2 process started (PID={self._current_proc.pid}, port={port})")
        startup_wait = self.cfg["execution"].get("startup_wait_seconds", 120)
        if not _wait_for_server(
            port,
            timeout=startup_wait,
            proc=self._current_proc,
            log_path=log_path,
        ):
            _terminate_process_tree(self._current_proc)
            raise RuntimeError("server startup timeout")
        game_ready_wait = self.cfg["execution"].get("game_ready_wait_seconds", 180)
        if game_ready_wait and not _wait_for_game_ready(
            port,
            timeout=game_ready_wait,
            proc=self._current_proc,
            log_path=log_path,
        ):
            _terminate_process_tree(self._current_proc)
            raise RuntimeError("game startup timeout")

    def _load_kg(self):
        game = self.cfg.get("game", {})
        kg_file = game.get("kg_file")
        if not kg_file:
            return
        port = self._port
        if not port:
            return
        try:
            requests.post(
                f"http://127.0.0.1:{port}/game/load_kg",
                params={
                    "kg_file": kg_file,
                    "data_dir": game.get("data_dir") or "",
                },
                timeout=30,
            )
            print(f"  KG loaded: {kg_file}")
        except requests.RequestException as e:
            print(f"  [WARN] KG load failed: {e}")

    def _run_finetune_phase(self, completed_trials: int, study):
        try:
            from scripts.counterfactual_simulator import CounterfactualSimulator
        except ImportError:
            print("  [CF-FINETUNE] counterfactual_simulator not available, skipping")
            return

        recent_count = self._override_cfg.get("recent_trials_for_analysis", 20)
        completed = [t.number for t in study.trials if t.state.name == "COMPLETE"]
        recent_trials = completed[-recent_count:]
        if not recent_trials:
            return

        try:
            simulator = CounterfactualSimulator(
                self.cfg, self.results_dir, self.results_dir
            )
            result = simulator.run_finetune_phase(completed_trials, recent_trials)
            print(
                f"  [CF-FINETUNE] Result: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )
            if self._override_model_path.exists():
                self._send_override_model_to_agent()
        except Exception as e:
            print(f"  [CF-FINETUNE] Error: {e}")

    def _send_override_model_to_agent(self):
        if not self._port:
            return
        try:
            requests.post(
                f"http://127.0.0.1:{self._port}/game/beam_params",
                json={"override_model_path": str(self._override_model_path)},
                timeout=10,
            )
            print(f"  Override model sent to agent: {self._override_model_path}")
        except Exception as e:
            print(f"  [WARN] Failed to send override model: {e}")

    def _shutdown(self):
        if self._port:
            try:
                requests.post(
                    f"http://127.0.0.1:{self._port}/game/shutdown",
                    timeout=5,
                )
            except Exception:
                pass
        if self._current_proc and self._current_proc.poll() is None:
            _terminate_process_tree(self._current_proc)
            print("  SC2 process stopped")

    def _signal_handler(self, sig, frame):
        print("\nstopping...")
        if self._current_proc and self._current_proc.poll() is None:
            _terminate_process_tree(self._current_proc)
        raise KeyboardInterrupt

    def _objective(self, trial: optuna.Trial, study: optuna.Study) -> float:
        space = self.cfg["search_space"]
        params = _sample_params(trial, space, self.cfg)
        if self._phase_enabled:
            params = self._apply_phase_to_params(params, self._current_phase)
        else:
            params = self._apply_non_phased_params(params)
        value = self._execute_trial(trial, params)
        if params.get("exclude_from_parameter_optimization", False):
            trial.set_user_attr("probe_objective", value)
            trial.set_user_attr("status", "exploration_probe")
            raise optuna.exceptions.TrialPruned(
                "exploration-only probe excluded from parameter optimization"
            )
        return value

    def _execute_trial(self, trial: optuna.Trial, params: dict) -> float:
        target_episodes = self.cfg["execution"]["episodes_per_trial"]
        port = self._port

        print(f"\n{'=' * 60}")
        print(f"Trial #{trial.number}")
        print(f"  phase: {params.get('phase', self._current_phase)}")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"  target: {target_episodes} episodes")
        print(f"{'=' * 60}")

        trial_dir = self.trials_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ep_file = trial_dir / "episodes.jsonl"
        if ep_file.exists():
            ep_file.write_text("", encoding="utf-8")

        send_params = dict(params)
        bktree_cfg = self.cfg.get("bktree", {}) or {}
        send_params["bktree_primary_threshold"] = float(
            bktree_cfg.get("primary_threshold", 1.0)
        )
        send_params["bktree_secondary_threshold"] = float(
            bktree_cfg.get("secondary_threshold", 0.5)
        )
        send_params["local_result_dir"] = str(trial_dir)
        send_params["target_episodes"] = target_episodes
        send_params["trial_number"] = trial.number
        send_params["stop_when_target_reached"] = False

        sent = False
        for _attempt in range(3):
            if _set_beam_params(port, send_params):
                sent = True
                break
            print(f"  [WARN] set_beam_params failed, retrying...")
            time.sleep(3)
        if not sent:
            print("  [ERROR] failed to set beam params after 3 attempts")
            _pause_game(port)
            trial.set_user_attr("status", "error")
            raise optuna.exceptions.TrialPruned("failed to set beam params")

        _resume_game(port)

        run_record = {
            "trial": trial.number,
            "port": port,
            "target_episodes": target_episodes,
            "params": {k: v for k, v in params.items()},
            "phase": params.get("phase", self._current_phase),
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "batch": self._batch,
            "source_trial": self._source_trial,
        }
        run_path = self.runs_dir / f"trial_{trial.number:04d}_run.json"
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        time.sleep(1)
        completed = _wait_for_file_progress(
            trial_dir, target_episodes, self.cfg, expected_trial=trial.number
        )
        _pause_game(port)

        time.sleep(1)
        obj_cfg = self.cfg.get("objective", {})
        stability_segments = obj_cfg.get("stability_segments", 5)
        metrics = _analyze_local_result(
            trial_dir, stability_segments, expected_trial=trial.number
        )

        win_rate = metrics["win_rate"]
        avg_score = metrics["avg_score"]
        stability = metrics["stability"]
        score_std = metrics["score_std"]
        n_eps = metrics["num_episodes"]

        self._all_metrics.append(metrics)

        alpha = obj_cfg.get("stability_alpha", 0.2)
        cap = obj_cfg.get("stability_cap", 8.0)

        stability_norm = min(stability / cap, 1.0) if cap > 0 else 0.0
        penalty_factor = max(1 - alpha * stability_norm, 0.0)

        objective = win_rate * avg_score * penalty_factor
        self._record_phase_result(
            params.get("phase", self._current_phase), objective, metrics, trial.number
        )

        trial.set_user_attr("status", "completed")
        trial.set_user_attr("phase", params.get("phase", self._current_phase))
        if "synergy_etg_source_trial" in params:
            trial.set_user_attr(
                "synergy_etg_source_trial", int(params["synergy_etg_source_trial"])
            )
            trial.set_user_attr(
                "synergy_etg_source_value", float(params.get("synergy_etg_source_value", 0.0))
            )
        trial.set_user_attr("batch", self._batch)
        if self._source_trial is not None:
            trial.set_user_attr("source_trial", self._source_trial)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("avg_score", avg_score)
        trial.set_user_attr("score_std", score_std)
        trial.set_user_attr("stability", stability)
        trial.set_user_attr("penalty_factor", penalty_factor)
        trial.set_user_attr("num_episodes", n_eps)
        trial.set_user_attr("result_file", str(trial_dir))
        trial.set_user_attr(
            "exclude_from_parameter_optimization",
            bool(params.get("exclude_from_parameter_optimization", False)),
        )

        print(
            f"  win_rate: {win_rate:.2%}  avg_score: {avg_score:.1f}  stability: {stability:.4f}  penalty: {penalty_factor:.2f}"
        )
        print(f"  objective: {objective:.4f}")

        run_record["status"] = "completed"
        run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_record["metrics"] = metrics
        run_record["objective"] = objective
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        if not completed:
            trial.set_user_attr("status", "timeout")
            run_record["status"] = "timeout"
            run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(str(run_path), "w", encoding="utf-8") as f:
                json.dump(run_record, f, ensure_ascii=False, indent=2)
            raise optuna.exceptions.TrialPruned("trial timeout")

        return objective

    def _rerun(self, source_trial_number: int, n_times: int):
        import sqlite3

        db_path = self.results_dir / "study.db"
        study_db = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name="beam_search", storage=study_db)
        print(f"loaded study, {len(study.trials)} trials")

        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                try:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                    print(f"  cleaned stale RUNNING trial #{t.number}")
                except Exception:
                    pass

        source = None
        for t in study.trials:
            if t.number == source_trial_number:
                source = t
                break
        if source is None:
            print(f"[ERROR] Trial #{source_trial_number} not found")
            return

        params = dict(source.params)
        print(f"rerun source: Trial #{source_trial_number}")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print(f"  repeat: {n_times} times")

        self._source_trial = source_trial_number

        existing_batch = 0
        for t in study.trials:
            b = t.user_attrs.get("batch", 0)
            if isinstance(b, int) and b > existing_batch:
                existing_batch = b
        self._batch = existing_batch + 1
        print(f"  batch: {self._batch}")

        self._startup()
        print("  waiting for game to fully initialize...")
        time.sleep(10)

        next_number = max(t.number for t in study.trials) + 1
        db_path = study_db.replace("sqlite:///", "")

        try:
            for i in range(n_times):
                trial_number = next_number + i
                value = self._execute_trial_simple(trial_number, params)

                self._record_trial_to_db(db_path, study, trial_number, params, value)

                print(f"  [{i + 1}/{n_times}] Trial #{trial_number}: obj={value:.4f}")
        except KeyboardInterrupt:
            print("\ninterrupted, saving progress...")
        finally:
            self._shutdown()

        self._save_summary(study)

    def _execute_trial_simple(self, trial_number: int, params: dict) -> float:
        target_episodes = self.cfg["execution"]["episodes_per_trial"]
        port = self._port

        print(f"\n{'=' * 60}")
        print(f"Trial #{trial_number} (rerun)")
        print(f"  target: {target_episodes} episodes")
        print(f"{'=' * 60}")

        trial_dir = self.trials_dir / f"trial_{trial_number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        for fname in ("episodes.jsonl", "progress.json"):
            fp = trial_dir / fname
            if fp.exists():
                if fname == "episodes.jsonl":
                    fp.write_text("", encoding="utf-8")
                else:
                    fp.unlink()

        send_params = dict(params)
        bktree_cfg = self.cfg.get("bktree", {}) or {}
        send_params["bktree_primary_threshold"] = float(
            bktree_cfg.get("primary_threshold", 1.0)
        )
        send_params["bktree_secondary_threshold"] = float(
            bktree_cfg.get("secondary_threshold", 0.5)
        )
        send_params["local_result_dir"] = str(trial_dir)
        send_params["target_episodes"] = target_episodes
        send_params["trial_number"] = trial_number
        send_params["plan_log_path"] = str(trial_dir / "plan.log")
        send_params["stop_when_target_reached"] = False

        sent = False
        for _attempt in range(5):
            if _set_beam_params(port, send_params):
                sent = True
                break
            print(f"  [WARN] set_beam_params failed, retrying...")
            time.sleep(5)
        if not sent:
            print("  [ERROR] failed to set beam params")
            _pause_game(port)
            return 0.0

        _resume_game(port)

        run_record = {
            "trial": trial_number,
            "port": port,
            "target_episodes": target_episodes,
            "params": {k: v for k, v in params.items()},
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "batch": self._batch,
            "source_trial": self._source_trial,
        }
        run_path = self.runs_dir / f"trial_{trial_number:04d}_run.json"
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        time.sleep(1)
        completed = _wait_for_file_progress(
            trial_dir, target_episodes, self.cfg, expected_trial=trial_number
        )
        _pause_game(port)

        time.sleep(1)
        obj_cfg = self.cfg.get("objective", {})
        stability_segments = obj_cfg.get("stability_segments", 5)
        metrics = _analyze_local_result(
            trial_dir, stability_segments, expected_trial=trial_number
        )

        win_rate = metrics["win_rate"]
        avg_score = metrics["avg_score"]
        stability = metrics["stability"]

        self._all_metrics.append(metrics)

        alpha = obj_cfg.get("stability_alpha", 0.2)
        cap = obj_cfg.get("stability_cap", 8.0)

        stability_norm = min(stability / cap, 1.0) if cap > 0 else 0.0
        penalty_factor = max(1 - alpha * stability_norm, 0.0)
        objective = win_rate * avg_score * penalty_factor

        run_record["status"] = "completed" if completed else "timeout"
        run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_record["metrics"] = metrics
        run_record["objective"] = objective
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        status = "completed" if completed else "timeout"
        print(
            f"  win_rate: {win_rate:.2%}  avg_score: {avg_score:.1f}  "
            f"stability: {stability:.4f}  penalty: {penalty_factor:.2f}"
        )
        print(f"  objective: {objective:.4f}  status: {status}")

        return objective

    def _record_trial_to_db(self, db_path, study, trial_number, params, value):
        import sqlite3

        attrs = {
            "status": "completed" if value > 0 else "timeout",
            "batch": self._batch,
            "win_rate": self._all_metrics[-1]["win_rate"] if self._all_metrics else 0,
            "avg_score": self._all_metrics[-1]["avg_score"] if self._all_metrics else 0,
            "score_std": self._all_metrics[-1]["score_std"] if self._all_metrics else 0,
            "stability": self._all_metrics[-1]["stability"] if self._all_metrics else 0,
            "num_episodes": self._all_metrics[-1]["num_episodes"]
            if self._all_metrics
            else 0,
        }
        if self._source_trial is not None:
            attrs["source_trial"] = self._source_trial

        try:
            db = sqlite3.connect(str(db_path))
            cur = db.cursor()
            cur.execute("SELECT study_id FROM studies WHERE study_name = 'beam_search'")
            row = cur.fetchone()
            if not row:
                db.close()
                return
            study_id = row[0]

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            cur.execute(
                "INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete) "
                "VALUES (?, ?, 'COMPLETE', ?, ?)",
                (trial_number, study_id, now, now),
            )
            trial_id = cur.lastrowid

            cur.execute(
                "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (?, 0, ?, 'FINITE')",
                (trial_id, value),
            )

            if self._source_trial is not None:
                cur.execute(
                    "SELECT trial_id FROM trials WHERE number = ?",
                    (self._source_trial,),
                )
                src_row = cur.fetchone()
                if src_row:
                    cur.execute(
                        "SELECT param_name, param_value, distribution_json "
                        "FROM trial_params WHERE trial_id = ?",
                        (src_row[0],),
                    )
                    for pname, pval, dist_json in cur.fetchall():
                        cur.execute(
                            "INSERT INTO trial_params "
                            "(trial_id, param_name, param_value, distribution_json) "
                            "VALUES (?, ?, ?, ?)",
                            (trial_id, pname, pval, dist_json),
                        )
            else:
                for k, v in params.items():
                    cur.execute(
                        "INSERT INTO trial_params "
                        "(trial_id, param_name, param_value, distribution_json) "
                        "VALUES (?, ?, ?, ?)",
                        (trial_id, k, str(v), json.dumps({"name": k})),
                    )

            for k, v in attrs.items():
                cur.execute(
                    "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, ?, ?)",
                    (trial_id, k, json.dumps(v)),
                )

            db.commit()
            db.close()
            print(f"  recorded Trial #{trial_number} to DB (trial_id={trial_id})")
        except Exception as e:
            print(f"  [ERROR] failed to record trial to DB: {e}")
            import traceback

            traceback.print_exc()

    def _print_best(self, study: optuna.Study):
        best = study.best_trial
        print(f"\n{'=' * 60}")
        print(f"Best (Trial #{best.number})")
        print(f"  objective: {best.value:.4f}")
        print(f"  params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"  win_rate: {best.user_attrs.get('win_rate', 'N/A')}")
        print(f"  avg_score: {best.user_attrs.get('avg_score', 'N/A')}")
        print(f"  stability: {best.user_attrs.get('stability', 'N/A')}")
        print(f"{'=' * 60}")

    def _save_summary(self, study: optuna.Study):
        trials_data = []
        for t in study.trials:
            trials_data.append(
                {
                    "number": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    "params": t.params,
                    "user_attrs": dict(t.user_attrs),
                }
            )

        summary = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "total_trials": len(study.trials),
            "completed_trials": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ),
            "trials": trials_data,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        out_path = self.results_dir / "study_summary.json"
        with open(str(out_path), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"summary saved: {out_path}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="PredictionRTS Parameter Learner")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG), help="config path")
    parser.add_argument("--trials", type=int, default=None, help="total trials")
    parser.add_argument("--episodes", type=int, default=None, help="episodes per trial")
    parser.add_argument("--map_key", default=None, help="Map config key")
    parser.add_argument("--kg_file", default=None, help="KG pickle file")
    parser.add_argument("--data_dir", default=None, help="data dir")
    parser.add_argument("--resume", action="store_true", help="resume from last")
    parser.add_argument(
        "--rerun",
        type=int,
        default=None,
        help="rerun specified trial with fixed params",
    )
    parser.add_argument(
        "--masked_actions",
        default=None,
        help="comma-separated masked action codes, e.g. '4a,4k'",
    )
    parser.add_argument(
        "--restart_interval",
        type=int,
        default=None,
        help="restart game client every N trials (0=never)",
    )
    parser.add_argument(
        "--restart_on_phase_change",
        action="store_true",
        help="restart game client whenever phased optimization switches phase",
    )
    parser.add_argument(
        "--no_restart_on_phase_change",
        action="store_true",
        help="disable phase-change restart",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="training run directory (e.g. output/learner_results/training_runs/run_0002). "
        "If set, uses an isolated study.db and trial directory.",
    )
    parser.add_argument(
        "--enable_counterfactual",
        action="store_true",
        help="Enable counterfactual action override during parameter search",
    )
    parser.add_argument(
        "--enable_action_tuning",
        action="store_true",
        help="Enable Monte Carlo action tuning during parameter search",
    )
    parser.add_argument(
        "--auto_archive",
        action="store_true",
        help="Archive completed run into output/learner_results/all_data with manifest",
    )
    parser.add_argument(
        "--archive_root",
        default=str(ROOT_DIR / "output" / "learner_results" / "all_data"),
        help="Archive root for --auto_archive",
    )
    parser.add_argument(
        "--archive_overwrite",
        action="store_true",
        help="Overwrite existing archive destination when --auto_archive is enabled",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.trials is not None:
        cfg["execution"]["total_trials"] = args.trials
    if args.episodes is not None:
        cfg["execution"]["episodes_per_trial"] = args.episodes
    if args.map_key is not None:
        cfg.setdefault("game", {})["map_key"] = args.map_key
    if args.kg_file is not None:
        cfg.setdefault("game", {})["kg_file"] = args.kg_file
    if args.data_dir is not None:
        cfg.setdefault("game", {})["data_dir"] = args.data_dir
    if args.restart_interval is not None:
        cfg["execution"]["restart_interval"] = args.restart_interval
    if args.restart_on_phase_change:
        cfg.setdefault("execution", {})["restart_on_phase_change"] = True
    if args.no_restart_on_phase_change:
        cfg.setdefault("execution", {})["restart_on_phase_change"] = False
    if args.enable_counterfactual:
        cfg.setdefault("action_override", {})["enabled"] = True
    if args.enable_action_tuning:
        cfg.setdefault("action_tuning", {})["enabled"] = True
    if cfg.get("phased_optimization", {}).get("enabled", False):
        cfg.setdefault("action_tuning", {})["enabled"] = True

    if args.masked_actions:
        cfg["_fixed_masked_actions"] = [
            a.strip() for a in args.masked_actions.split(",") if a.strip()
        ]
    else:
        cfg["_fixed_masked_actions"] = []

    if args.run_dir:
        run_dir_path = Path(args.run_dir)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        try:
            with open(str(run_dir_path / "learner_config_snapshot.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"  [WARN] failed to save config snapshot: {e}")

    print("=" * 60)
    print("PredictionRTS Parameter Learner (single-start mode)")
    print(f"  config: {args.config}")
    print(f"  total trials: {cfg['execution']['total_trials']}")
    print(f"  episodes/trial: {cfg['execution']['episodes_per_trial']}")
    print(f"  KG file: {cfg['game'].get('kg_file', '(auto)')}")
    print(f"  data dir: {cfg['game'].get('data_dir', '(auto)')}")
    print(f"  restart interval: {cfg['execution'].get('restart_interval', 0)} trials")
    print(
        f"  restart on phase change: {cfg['execution'].get('restart_on_phase_change', True)}"
    )
    if args.run_dir:
        print(f"  run dir: {args.run_dir}")
    cf_enabled = cfg.get("action_override", {}).get("enabled", False)
    print(f"  counterfactual: {'enabled' if cf_enabled else 'disabled'}")
    if cf_enabled:
        fi = cfg["action_override"].get("finetune_interval", 25)
        print(f"  finetune_interval: {fi} trials")
    tuning_enabled = cfg.get("action_tuning", {}).get("enabled", False)
    print(f"  action_tuning: {'enabled' if tuning_enabled else 'disabled'}")
    print("=" * 60)

    results_dir = Path(cfg["storage"].get("results_dir", "output/learner_results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    pid_file = results_dir / ".learner_pid"
    pid_file.write_text(str(os.getpid()))

    try:
        learner = ParameterLearner(cfg, run_dir=args.run_dir)
        if args.rerun is not None:
            learner._rerun(args.rerun, n_times=1)
        else:
            learner.run(n_trials=cfg["execution"]["total_trials"], resume=args.resume)
        if args.auto_archive and args.run_dir and args.rerun is None:
            try:
                from scripts.archive_learner_run import archive_run

                dest = archive_run(
                    Path(args.run_dir),
                    archive_root=Path(args.archive_root),
                    overwrite=bool(args.archive_overwrite),
                    cfg=cfg,
                )
                print(f"archived run: {dest}")
            except Exception as e:
                print(f"  [WARN] auto archive failed: {e}")
    finally:
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
