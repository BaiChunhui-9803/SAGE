#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CounterfactualSimulator -- 反事实动作覆盖模拟编排器

从参数寻优的差 episode 中识别关键决策点，启动独立游戏实例执行
反事实模拟（重放 + 动作替换 + beam search 接管），评估替换收益，
更新 ActionOverrideModel。

Usage:
    Called by parameter_learner.py during finetune phases.
    Can also be run standalone for testing.
"""

from __future__ import annotations

import sys
import os
import json
import time
import signal
import socket
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from src.decision.action_override_model import ActionOverrideModel


def _find_free_port(exclude=None):
    exclude = set(exclude or [])
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if port not in exclude:
                return port
    raise RuntimeError("Cannot find free port")


def _wait_for_server(port: int, timeout: float = 60.0) -> bool:
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/game/status", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _set_beam_params(port: int, params: dict) -> bool:
    import requests

    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/game/beam_params", json=params, timeout=10
        )
        return r.status_code == 200
    except Exception:
        return False


def _resume_game(port: int):
    import requests

    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "resume"},
            timeout=5,
        )
    except Exception:
        pass


def _pause_game(port: int):
    import requests

    try:
        requests.post(
            f"http://127.0.0.1:{port}/game/control",
            json={"command": "pause"},
            timeout=5,
        )
    except Exception:
        pass


def _wait_for_file_progress(trial_dir: Path, target: int, cfg: dict) -> bool:
    poll_interval = cfg["execution"].get("completion_poll_interval", 3)
    timeout_minutes = cfg["execution"].get("completion_timeout_minutes", 60)
    deadline = time.time() + timeout_minutes * 60
    progress_file = trial_dir / "progress.json"
    last_logged = 0
    while time.time() < deadline:
        try:
            if progress_file.exists():
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                done = data.get("completed", 0)
                if done >= target:
                    return True
                if done >= last_logged + 5:
                    print(f"    cf progress: {done}/{target}")
                    last_logged = done
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(poll_interval)
    return False


def _analyze_local_result(trial_dir: Path, num_segments: int) -> dict:
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
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not episodes:
        return {
            "win_rate": 0.0,
            "avg_score": 0.0,
            "score_std": 0.0,
            "stability": 0.0,
            "num_episodes": 0,
        }
    wins = sum(1 for e in episodes if e.get("result") == "Win")
    scores = [e.get("score", 0) for e in episodes]
    win_rate = wins / len(episodes)
    avg_score = float(np.mean(scores))
    score_std = float(np.std(scores))

    n = len(episodes)
    seg_size = max(n // num_segments, 1)
    win_rates = []
    avg_scores = []
    for i in range(num_segments):
        seg = episodes[i * seg_size : (i + 1) * seg_size]
        if seg:
            win_rates.append(sum(1 for e in seg if e.get("result") == "Win") / len(seg))
            avg_scores.append(float(np.mean([e.get("score", 0) for e in seg])))
    stability = (
        float(np.std(win_rates)) + float(np.std(avg_scores)) if win_rates else 0.0
    )
    return {
        "win_rate": win_rate,
        "avg_score": avg_score,
        "score_std": score_std,
        "stability": stability,
        "num_episodes": len(episodes),
    }


class CounterfactualSimulator:
    def __init__(self, cfg: dict, results_dir: Path, run_dir: Path):
        self._cfg = cfg
        self._results_dir = Path(results_dir)
        self._run_dir = Path(run_dir)
        self._override_model = ActionOverrideModel()
        self._override_model_path = self._run_dir / "action_override_model.pkl"
        if self._override_model_path.exists():
            try:
                self._override_model = ActionOverrideModel.load(
                    str(self._override_model_path)
                )
                print(
                    f"  Loaded existing override model: {len(self._override_model.get_all_entries())} rules"
                )
            except Exception as e:
                print(f"  [WARN] Failed to load override model: {e}")

    def run_finetune_phase(
        self, completed_trials: int, recent_trials: List[int]
    ) -> Dict:
        print(f"\n{'=' * 60}")
        print(
            f"[CF-FINETUNE] Phase after {completed_trials} trials, analyzing {len(recent_trials)} recent trials"
        )
        print(f"{'=' * 60}")

        bad_episodes = self._extract_bad_episodes(recent_trials)
        if not bad_episodes:
            print("  No bad episodes found, skipping.")
            return {
                "status": "no_bad_episodes",
                "model_entries": len(self._override_model.get_all_entries()),
            }

        print(f"  Found {len(bad_episodes)} bad episodes")

        divergence_points = self._identify_divergence_points(bad_episodes)
        if not divergence_points:
            print("  No divergence points found, skipping.")
            return {
                "status": "no_divergence_points",
                "model_entries": len(self._override_model.get_all_entries()),
            }

        print(f"  Identified {len(divergence_points)} divergence points")

        results = []
        for idx, dp in enumerate(divergence_points):
            print(
                f"\n  [{idx + 1}/{len(divergence_points)}] DP: nid={dp['nid']} "
                f"action={dp['original_action']} step={dp['frame_index']}"
            )
            dp_results = self._run_counterfactual(dp)
            results.append(dp_results)

        self._override_model.prune(min_confidence=0.1, min_cf_runs=1)
        self._override_model.save(str(self._override_model_path))

        summary = self._override_model.get_summary()
        print(f"\n[CF-FINETUNE] Complete: {summary}")
        return {
            "status": "completed",
            "bad_episodes": len(bad_episodes),
            "divergence_points": len(divergence_points),
            "cf_simulations": sum(r["runs"] for r in results),
            "model_entries": summary["total_rules"],
        }

    def _extract_bad_episodes(self, trial_numbers: List[int]) -> List[Dict]:
        override_cfg = self._cfg.get("action_override", {})
        percentile = override_cfg.get("bad_episode_percentile", 25.0)

        all_episodes = []
        for tn in trial_numbers:
            trial_dir = self._run_dir / "trials" / f"trial_{tn:04d}"
            ep_file = trial_dir / "episodes.jsonl"
            run_file = self._run_dir / "runs" / f"trial_{tn:04d}_run.json"
            if not ep_file.exists():
                continue

            params = {}
            if run_file.exists():
                try:
                    params = json.loads(run_file.read_text(encoding="utf-8")).get(
                        "params", {}
                    )
                except Exception:
                    pass

            try:
                with open(str(ep_file), "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ep = json.loads(line)
                        if ep.get("trial_number") != tn:
                            continue
                        frames = ep.get("frames", [])
                        if not frames:
                            continue
                        all_episodes.append(
                            {
                                "episode": ep,
                                "trial_number": tn,
                                "params": params,
                            }
                        )
            except Exception:
                pass

        if not all_episodes:
            return []

        scores = [e["episode"]["score"] for e in all_episodes]
        threshold = float(np.percentile(scores, percentile))
        bad = [e for e in all_episodes if e["episode"]["score"] <= threshold]

        print(
            f"  Score threshold (p{percentile}): {threshold:.1f}, "
            f"{len(bad)}/{len(all_episodes)} episodes below"
        )
        return bad

    def _identify_divergence_points(self, bad_episodes: List[Dict]) -> List[Dict]:
        override_cfg = self._cfg.get("action_override", {})
        max_per_episode = override_cfg.get("max_diverge_points_per_episode", 3)

        all_points = []
        for ep_data in bad_episodes:
            ep = ep_data["episode"]
            frames = ep.get("frames", [])
            if not frames:
                continue

            candidates = []
            for i, frame in enumerate(frames):
                if frame.get("action_source") == "terminal_fix":
                    continue
                if i < 3:
                    continue
                if frame.get("enemy_count", 0) <= 1:
                    continue

                nid = frame.get("nid")
                if nid is None:
                    continue

                action_code = frame.get("action_code", "")
                action_source = frame.get("action_source", "")
                hp_delta = frame.get("hp_delta", 0)

                priority = 0
                if action_source == "etg_plan" and hp_delta < -10:
                    priority = 3
                elif (
                    frame.get("my_count", 0) > frame.get("enemy_count", 0)
                    and action_code
                    and action_code[-1] not in ("a", "b", "c", "d", "e")
                ):
                    priority = 2
                elif action_source in (
                    "fallback",
                    "ft_plan",
                    "etg_relaxed",
                    "fuzzy_plan",
                ):
                    priority = 1

                if priority > 0:
                    candidates.append(
                        {
                            "episode_data": ep_data,
                            "frame_index": i,
                            "frame": frame,
                            "priority": priority,
                            "nid": nid,
                            "original_action": action_code,
                            "action_source": action_source,
                        }
                    )

            candidates.sort(key=lambda x: -x["priority"])
            all_points.extend(candidates[:max_per_episode])

        all_points.sort(key=lambda x: -x["priority"])
        return all_points

    def _get_replacement_candidates(self, dp: Dict) -> List[str]:
        original = dp["original_action"]
        candidates = []

        if original != "4b":
            candidates.append("4b")
        if original != "4c" and "4c" not in candidates:
            candidates.append("4c")

        cluster = original[0] if original else "4"
        for letter in ("a", "b", "c", "d", "e"):
            code = cluster + letter
            if code != original and code not in candidates:
                candidates.append(code)
                break

        for entry in self._override_model.get_all_entries():
            if (
                entry.state_id == dp["nid"]
                and entry.replacement_action not in candidates
            ):
                candidates.append(entry.replacement_action)
                break

        return candidates[:3]

    def _run_counterfactual(self, dp: Dict) -> Dict:
        override_cfg = self._cfg.get("action_override", {})
        cf_runs = override_cfg.get("cf_runs", 5)

        ep_data = dp["episode_data"]
        ep = ep_data["episode"]
        params = ep_data["params"]
        frames = ep.get("frames", [])
        diverge_step = dp["frame_index"]

        if diverge_step >= len(frames):
            return {"dp": dp, "candidates_results": [], "runs": 0}

        original_actions = [f["action_code"] for f in frames[:diverge_step]]
        if not original_actions:
            return {"dp": dp, "candidates_results": [], "runs": 0}

        candidates = self._get_replacement_candidates(dp)
        if not candidates:
            return {"dp": dp, "candidates_results": [], "runs": 0}

        self._override_model.record_original(
            dp["nid"],
            dp["original_action"],
            ep.get("result", "Unknown"),
            ep.get("score", 0),
            trial_number=ep_data["trial_number"],
        )

        results = []
        for repl_action in candidates:
            cf_run_id = (
                f"cf_t{ep_data['trial_number']}"
                f"_ep{ep.get('episode_id', 0)}"
                f"_s{diverge_step}_{repl_action}"
            )
            cf_dir = self._run_dir / "counterfactual" / cf_run_id
            cf_dir.mkdir(parents=True, exist_ok=True)

            port = _find_free_port()
            game = self._cfg.get("game", {})
            cmd = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "run_live_game.py"),
                "--mode",
                "all",
                "--port",
                str(port),
                "--map_key",
                game.get("map_key", "sce-1"),
                "--autopilot_mode",
                "counterfactual",
                "--max_episodes",
                str(cf_runs),
                "--cf_actions",
                ",".join(original_actions),
                "--cf_diverge_step",
                str(diverge_step),
                "--cf_replacement",
                repl_action,
                "--cf_run_id",
                cf_run_id,
                "--cf_runs",
                str(cf_runs),
            ]
            if game.get("etg_file"):
                cmd.extend(["--etg_file", game["etg_file"]])
            if game.get("data_dir"):
                cmd.extend(["--data_dir", game["data_dir"]])

            beam_params = dict(params)
            beam_params["local_result_dir"] = str(cf_dir)
            beam_params["target_episodes"] = cf_runs
            beam_params["mode"] = "counterfactual"

            print(f"    Starting cf simulation: {repl_action} (port={port})")

            log_path = cf_dir / "sim.log"
            log_file = open(str(log_path), "w", encoding="utf-8")
            flags = 0
            if sys.platform == "win32":
                flags = (
                    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                )

            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    cwd=str(ROOT_DIR),
                    creationflags=flags,
                )
                if not _wait_for_server(port, timeout=60):
                    print(f"      [ERROR] server startup timeout")
                    results.append(
                        {
                            "replacement_action": repl_action,
                            "runs": 0,
                            "error": "startup_timeout",
                        }
                    )
                    continue

                if not _set_beam_params(port, beam_params):
                    print(f"      [ERROR] set_beam_params failed")
                    results.append(
                        {
                            "replacement_action": repl_action,
                            "runs": 0,
                            "error": "params_failed",
                        }
                    )
                    continue

                _resume_game(port)
                completed = _wait_for_file_progress(cf_dir, cf_runs, self._cfg)
                _pause_game(port)

                if completed:
                    metrics = _analyze_local_result(cf_dir, 5)
                    ep_file = cf_dir / "episodes.jsonl"
                    if ep_file.exists():
                        with open(str(ep_file), "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    cf_ep = json.loads(line)
                                    self._override_model.update_counterfactual(
                                        dp["nid"],
                                        dp["original_action"],
                                        repl_action,
                                        cf_ep.get("result", "Unknown"),
                                        cf_ep.get("score", 0),
                                        trial_number=ep_data["trial_number"],
                                    )
                                except json.JSONDecodeError:
                                    pass

                    improvement = metrics["avg_score"] - ep.get("score", 0)
                    print(
                        f"      Result: wr={metrics['win_rate']:.2%} "
                        f"avg_score={metrics['avg_score']:.1f} "
                        f"improvement={improvement:+.1f}"
                    )
                    results.append(
                        {
                            "replacement_action": repl_action,
                            "runs": cf_runs,
                            "cf_win_rate": metrics["win_rate"],
                            "cf_avg_score": metrics["avg_score"],
                            "improvement": improvement,
                        }
                    )
                else:
                    print(f"      [WARN] cf simulation timeout")
                    results.append(
                        {
                            "replacement_action": repl_action,
                            "runs": 0,
                            "error": "timeout",
                        }
                    )
            except Exception as e:
                print(f"      [ERROR] {e}")
                results.append(
                    {
                        "replacement_action": repl_action,
                        "runs": 0,
                        "error": str(e),
                    }
                )
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                log_file.close()

        return {
            "dp": dp,
            "candidates_results": results,
            "runs": sum(r.get("runs", 0) for r in results),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Counterfactual Simulator (standalone test)"
    )
    parser.add_argument("--run_dir", required=True, help="Training run directory")
    parser.add_argument(
        "--config", default=str(ROOT_DIR / "configs" / "learner_config.yaml")
    )
    parser.add_argument("--recent_trials", type=int, default=20)
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_dir = Path(args.run_dir)

    import optuna

    study_db = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.load_study(study_name="beam_search", storage=study_db)
    completed = [t.number for t in study.trials if t.state.name == "COMPLETE"]
    recent_trials = completed[-args.recent_trials :]

    sim = CounterfactualSimulator(cfg, run_dir, run_dir)
    result = sim.run_finetune_phase(len(completed), recent_trials)
    print(json.dumps(result, indent=2, ensure_ascii=False))
