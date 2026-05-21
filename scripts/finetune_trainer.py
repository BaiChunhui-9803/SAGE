#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finetune Trainer - Online Collaborative Exploration (v3)

Beam search + finetune model collaborative online learning.
All trials share one evolving finetune model, updated every step.

Usage:
    python scripts/finetune_trainer.py --n_trials 20 --episodes_per_trial 100 --kg_file ... --data_dir ...
"""

import sys
import os
import json
import time
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from src.decision.finetune_model import FinetuneModel, save_atomic

_DEFAULT_CONFIG = ROOT_DIR / "configs" / "learner_config.yaml"
_RESULTS_DIR = ROOT_DIR / "output" / "learner_results"
_TRIALS_DIR = _RESULTS_DIR / "trials"
_FINETUNE_DIR = _RESULTS_DIR / "finetune_runs"
_SHARED_MODEL_PATH = _RESULTS_DIR / "shared_finetune_model.pkl"


def _find_free_port(exclude=None):
    exclude = set(exclude or [])
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if port not in exclude:
                return port
    raise RuntimeError("Cannot find free port")


def _wait_for_server(port, timeout=30):
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


def _set_beam_params(port, params):
    url = f"http://127.0.0.1:{port}/game/beam_params"
    try:
        r = requests.post(url, json=params, timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _pause_game(port):
    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "pause"},
            timeout=5,
        )
    except Exception:
        pass


def _resume_game(port):
    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "resume"},
            timeout=5,
        )
    except Exception:
        pass


def _wait_for_file_progress(trial_dir, target, timeout_minutes=60, poll_interval=3):
    deadline = time.time() + timeout_minutes * 60
    progress_file = trial_dir / "progress.json"
    last_logged = 0

    while time.time() < deadline:
        try:
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


def _ensure_shared_model() -> str:
    if not _SHARED_MODEL_PATH.exists():
        model = FinetuneModel()
        model.save(str(_SHARED_MODEL_PATH))
        print(f"  Created shared model: {_SHARED_MODEL_PATH}")
    return str(_SHARED_MODEL_PATH)


class FinetuneTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.n_trials = config.get("n_trials", 20)
        self.episodes_per_trial = config.get("episodes_per_trial", 100)
        self.sigma = config.get("sigma", 0.5)
        self.target_visits = config.get("target_visits", 10)
        self._port = None
        self._proc = None
        self._finetune_id = 0
        self._sample_dir: Optional[Path] = None
        self._kg_file = config.get("kg_file")
        self._data_dir = config.get("data_dir")
        self._map_key = config.get("map_key")
        self._reward_mode = config.get("reward_mode", "hp_episodic")

    def run(self):
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
        _TRIALS_DIR.mkdir(parents=True, exist_ok=True)

        existing = list(_FINETUNE_DIR.glob("finetune_*_run.json"))
        self._finetune_id = len(existing) + 1

        self._sample_dir = _FINETUNE_DIR / f"finetune_{self._finetune_id:04d}"
        self._sample_dir.mkdir(parents=True, exist_ok=True)

        shared_model_path = _ensure_shared_model()

        print(f"\n{'=' * 60}")
        print(f"Online Collaborative Exploration #{self._finetune_id:04d}")
        print(f"  n_trials: {self.n_trials}")
        print(f"  episodes/trial: {self.episodes_per_trial}")
        print(f"  sigma: {self.sigma}, target_visits: {self.target_visits}")
        print(f"  shared model: {shared_model_path}")
        print(f"  reward_mode: {self._reward_mode}")
        print(f"{'=' * 60}")

        run_record = {
            "finetune_id": self._finetune_id,
            "config": self.config,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running",
            "shared_model_path": shared_model_path,
        }
        run_path = _FINETUNE_DIR / f"finetune_{self._finetune_id:04d}_run.json"
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(run_record, f, ensure_ascii=False, indent=2)

        try:
            self._startup()

            study = optuna.create_study(
                study_name="online_collab",
                storage=f"sqlite:///{_RESULTS_DIR / 'online_study.db'}",
                direction="maximize",
                load_if_exists=True,
            )

            study.optimize(
                lambda trial: self._objective(trial, shared_model_path),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )

            run_record["status"] = "completed"
            run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_record["best_params"] = study.best_params
            run_record["best_value"] = study.best_value
            self._save_run_record(run_record, run_path)

            print(f"\n{'=' * 60}")
            print("Online Collaborative Exploration Complete")
            print(f"  Best value: {study.best_value:.4f}")
            print(f"  Best params: {study.best_params}")
            if _SHARED_MODEL_PATH.exists():
                model = FinetuneModel.load(str(_SHARED_MODEL_PATH))
                stats = model.get_overall_stats()
                print(
                    f"  Shared model: {stats['explored_states']}/{stats['total_states']} states, "
                    f"RS={stats['avg_replacement_score']:.3f}, "
                    f"visits={stats['total_visits']}"
                )
            print(f"{'=' * 60}")

        except KeyboardInterrupt:
            print("\ninterrupted, saving progress...")
            run_record["status"] = "interrupted"
            run_record["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._save_run_record(run_record, run_path)
        finally:
            self._shutdown()

    def _objective(self, trial, shared_model_path: str) -> float:
        min_visits = trial.suggest_int("min_visits", 1, 10)
        beam_width = trial.suggest_int("beam_width", 2, 8)
        lookahead_steps = trial.suggest_int("lookahead_steps", 3, 20)

        trial_dir = self._sample_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("episodes.jsonl", "progress.json"):
            fp = trial_dir / fname
            if fp.exists():
                fp.unlink()

        send_params = {
            "min_visits": min_visits,
            "beam_width": beam_width,
            "lookahead_steps": lookahead_steps,
            "local_result_dir": str(trial_dir),
            "target_episodes": self.episodes_per_trial,
            "trial_number": trial.number,
            "finetune_model_path": shared_model_path,
            "reward_mode": self._reward_mode,
            "kg_file": self._kg_file,
            "data_dir": self._data_dir,
        }

        sent = False
        for _attempt in range(5):
            if _set_beam_params(self._port, send_params):
                sent = True
                break
            print(f"    [WARN] set_beam_params failed, retrying...")
            time.sleep(5)
        if not sent:
            print(f"    [ERROR] failed to set beam params for trial #{trial.number}")
            return 0.0

        _resume_game(self._port)
        time.sleep(1)
        completed = _wait_for_file_progress(
            trial_dir, self.episodes_per_trial, timeout_minutes=120
        )
        _pause_game(self._port)

        if not completed:
            print(f"    [WARN] trial #{trial.number} timed out")
            return 0.0

        progress = self._read_trial_metrics(trial_dir)
        results = self._parse_trial_results(trial_dir)

        win_rate = results.get("win_rate", 0.0)
        avg_score = results.get("avg_score", 0.0)
        fallback_ratio = progress.get("fallback_ratio", 1.0)

        score_norm = 1.0 / (1.0 + np.exp(-avg_score / 50.0))

        objective = win_rate * 0.5 + score_norm * 0.3 + (1.0 - fallback_ratio) * 0.2

        print(
            f"  Trial #{trial.number}: wr={win_rate:.3f} "
            f"score={avg_score:.1f} fb={fallback_ratio:.3f} "
            f"obj={objective:.4f}"
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "trial": trial.number,
            "params": {
                "min_visits": min_visits,
                "beam_width": beam_width,
                "lookahead_steps": lookahead_steps,
            },
            "target_episodes": self.episodes_per_trial,
            "start_time": now,
            "status": "completed",
            "progress": progress,
            "results": results,
            "objective": round(objective, 4),
        }
        run_path = trial_dir / "run.json"
        with open(str(run_path), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return objective

    def _parse_trial_results(self, trial_dir: Path) -> dict:
        ep_file = trial_dir / "episodes.jsonl"
        if not ep_file.exists():
            return {}
        wins = 0
        losses = 0
        total_score = 0.0
        count = 0
        try:
            with open(str(ep_file), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ep = json.loads(line)
                        result = ep.get("result", "")
                        score = ep.get("score", 0.0)
                        total_score += float(score)
                        count += 1
                        if result == "Win":
                            wins += 1
                        elif result == "Loss":
                            losses += 1
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError:
            return {}
        if count == 0:
            return {}
        return {
            "win_rate": round(wins / count, 4),
            "loss_rate": round(losses / count, 4),
            "avg_score": round(total_score / count, 2),
            "total_episodes": count,
        }

    def _read_trial_metrics(self, trial_dir: Path) -> dict:
        progress_file = trial_dir / "progress.json"
        if progress_file.exists():
            try:
                return json.loads(progress_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_run_record(self, record, path):
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def _startup(self):
        if self._port and self._proc and self._proc.poll() is None:
            return

        port = _find_free_port(exclude={8000, 8501, 8502})
        self._port = port

        cfg_path = _DEFAULT_CONFIG
        game_cfg = {}
        if cfg_path.exists():
            try:
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f)
                game_cfg = full_cfg.get("game", {})
            except Exception:
                pass

        _kg_file = self._kg_file or game_cfg.get("kg_file")
        _data_dir = self._data_dir or game_cfg.get("data_dir")
        _map_key = self._map_key or game_cfg.get("map_key", "sce-1")
        if not _kg_file or not _data_dir:
            print("=" * 60)
            print("[ERROR] ETG parameters required but not provided:")
            print(f"  --kg_file  = {_kg_file!r}")
            print(f"  --data_dir = {_data_dir!r}")
            print()
            print("  Usage:")
            print(
                "    python scripts/finetune_trainer.py"
                " --n_trials 20"
                " --kg_file MarineMicro_MvsM_4_augmented/kg_simple.pkl"
                " --data_dir data/MarineMicro_MvsM_4/augmented_1"
            )
            print("=" * 60)
            raise RuntimeError(
                "ETG parameters (--kg_file and --data_dir) are required. "
                "Aborting without starting SC2 process."
            )

        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_live_game.py"),
            "--mode",
            "all",
            "--port",
            str(port),
            "--map_key",
            _map_key,
            "--max_episodes",
            "0",
            "--autopilot_mode",
            game_cfg.get("autopilot_mode", "multi_step"),
            "--kg_file",
            _kg_file,
            "--data_dir",
            _data_dir,
        ]
        if game_cfg.get("fallback_action"):
            cmd.extend(["--fallback_action", game_cfg["fallback_action"]])

        log_path = (
            self._sample_dir / "trainer.log"
            if self._sample_dir
            else _TRIALS_DIR / "trainer.log"
        )
        log_file = open(str(log_path), "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32":
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            cwd=str(ROOT_DIR),
            creationflags=flags,
        )

        print(f"  SC2 process started (PID={self._proc.pid}, port={port})")
        if not _wait_for_server(port, timeout=30):
            self._proc.terminate()
            raise RuntimeError("server startup timeout")

        if _kg_file:
            try:
                requests.post(
                    f"http://127.0.0.1:{port}/game/load_kg",
                    json={
                        "kg_file": _kg_file,
                        "data_dir": _data_dir,
                    },
                    timeout=30,
                )
                print(f"  KG loaded: {_kg_file}")
            except requests.RequestException as e:
                print(f"  [WARN] KG load failed: {e}")

    def _shutdown(self):
        if self._port:
            try:
                requests.post(f"http://127.0.0.1:{self._port}/game/shutdown", timeout=5)
            except Exception:
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="PredictionRTS Online Collaborative Exploration"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="number of optuna trials"
    )
    parser.add_argument("--episodes_per_trial", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.5, help="k-NN distance decay")
    parser.add_argument(
        "--target_visits", type=int, default=10, help="confidence target"
    )
    parser.add_argument("--kg_file", type=str, default=None, help="KG pickle file")
    parser.add_argument("--data_dir", type=str, default=None, help="Training data dir")
    parser.add_argument("--map_key", type=str, default=None, help="Map config key")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="hp_episodic",
        choices=["hp", "hp_episodic", "etg_correct", "etg_offline"],
        help="Reward mode for online finetune updates",
    )
    args = parser.parse_args()

    config = {
        "n_trials": args.n_trials,
        "episodes_per_trial": args.episodes_per_trial,
        "sigma": args.sigma,
        "target_visits": args.target_visits,
        "kg_file": args.kg_file,
        "data_dir": args.data_dir,
        "map_key": args.map_key,
        "reward_mode": args.reward_mode,
    }

    trainer = FinetuneTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
