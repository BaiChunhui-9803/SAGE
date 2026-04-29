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


def _wait_for_server(port: int, timeout: int = 30) -> bool:
    url = f"http://127.0.0.1:{port}/game/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        except requests.RequestException:
            pass
        time.sleep(1)
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

        try:
            self._startup()
            self._load_kg()
            for i in range(total):
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
                study.optimize(
                    lambda trial: self._objective(trial, study),
                    n_trials=1,
                    show_progress_bar=False,
                )
        except KeyboardInterrupt:
            print("\ninterrupted, saving progress...")
        finally:
            self._shutdown()

        self._print_best(study)
        self._save_summary(study)

    def _startup(self):
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
        if game.get("fallback_action"):
            cmd.extend(["--fallback_action", game["fallback_action"]])

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
        startup_wait = self.cfg["execution"].get("startup_wait_seconds", 30)
        if not _wait_for_server(port, timeout=startup_wait):
            self._current_proc.terminate()
            raise RuntimeError("server startup timeout")

        if game.get("kg_file"):
            self._load_kg()

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
            self._current_proc.terminate()
            try:
                self._current_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._current_proc.kill()
            print("  SC2 process stopped")

    def _signal_handler(self, sig, frame):
        print("\nstopping...")
        if self._current_proc and self._current_proc.poll() is None:
            self._current_proc.terminate()
            try:
                self._current_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._current_proc.kill()
        raise KeyboardInterrupt

    def _objective(self, trial: optuna.Trial, study: optuna.Study) -> float:
        space = self.cfg["search_space"]
        params = _sample_params(trial, space, self.cfg)
        return self._execute_trial(trial, params)

    def _execute_trial(self, trial: optuna.Trial, params: dict) -> float:
        target_episodes = self.cfg["execution"]["episodes_per_trial"]
        port = self._port

        print(f"\n{'=' * 60}")
        print(f"Trial #{trial.number}")
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
        send_params["local_result_dir"] = str(trial_dir)
        send_params["target_episodes"] = target_episodes
        send_params["trial_number"] = trial.number

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

        trial.set_user_attr("status", "completed")
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
        send_params["local_result_dir"] = str(trial_dir)
        send_params["target_episodes"] = target_episodes
        send_params["trial_number"] = trial_number
        send_params["plan_log_path"] = str(trial_dir / "plan.log")

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
        "--run_dir",
        default=None,
        help="training run directory (e.g. output/learner_results/training_runs/run_0002). "
        "If set, uses an isolated study.db and trial directory.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.trials is not None:
        cfg["execution"]["total_trials"] = args.trials
    if args.episodes is not None:
        cfg["execution"]["episodes_per_trial"] = args.episodes
    if args.kg_file is not None:
        cfg["game"]["kg_file"] = args.kg_file
    if args.data_dir is not None:
        cfg["game"]["data_dir"] = args.data_dir
    if args.restart_interval is not None:
        cfg["execution"]["restart_interval"] = args.restart_interval

    if args.masked_actions:
        cfg["_fixed_masked_actions"] = [
            a.strip() for a in args.masked_actions.split(",") if a.strip()
        ]
    else:
        cfg["_fixed_masked_actions"] = []

    print("=" * 60)
    print("PredictionRTS Parameter Learner (single-start mode)")
    print(f"  config: {args.config}")
    print(f"  total trials: {cfg['execution']['total_trials']}")
    print(f"  episodes/trial: {cfg['execution']['episodes_per_trial']}")
    print(f"  KG file: {cfg['game'].get('kg_file', '(auto)')}")
    print(f"  data dir: {cfg['game'].get('data_dir', '(auto)')}")
    print(f"  restart interval: {cfg['execution'].get('restart_interval', 0)} trials")
    if args.run_dir:
        print(f"  run dir: {args.run_dir}")
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
    finally:
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
