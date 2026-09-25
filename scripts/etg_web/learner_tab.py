import os
import re
import sys
import shutil
import subprocess
import time
import datetime
import pickle
import requests as _requests
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
from etg_web.i18n import st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna

from src import ROOT_DIR
from etg_web.constants import get_bktree_threshold_defaults, get_map_key_for_map_id

_RESULTS_DIR = ROOT_DIR / "output" / "learner_results"
_TRAINING_RUNS_DIR = _RESULTS_DIR / "training_runs"


@st.cache_data(ttl=300, show_spinner=False)
def _get_bktree_summary(etg_data_dir: str) -> Dict[str, int]:
    bkt_dir = ROOT_DIR / etg_data_dir / "bktree"
    primary_file = bkt_dir / "primary_bktree.json"
    primary_count = 0
    if primary_file.exists():
        try:
            with open(str(primary_file), "r", encoding="utf-8") as f:
                root = json.load(f)
            stack = [root]
            while stack:
                node = stack.pop()
                primary_count += 1
                stack.extend(node.get("children", {}).values())
        except Exception:
            primary_count = 0

    secondary_count = len(list(bkt_dir.glob("secondary_bktree_*.json")))

    state_node_file = ROOT_DIR / etg_data_dir / "graph" / "state_node.txt"
    state_node_count = 0
    if state_node_file.exists():
        try:
            with open(str(state_node_file), "r", encoding="utf-8") as f:
                state_node_count = sum(1 for line in f if line.strip())
        except Exception:
            state_node_count = 0

    return {
        "primary_count": primary_count,
        "secondary_count": secondary_count,
        "state_node_count": state_node_count,
    }


def _is_valid_action_code(action_code) -> bool:
    return (
        isinstance(action_code, str)
        and len(action_code) == 2
        and action_code[0] in "01234"
        and action_code[1] in "abcdefghijk"
    )


def _format_state_ref(state_id):
    state_ref = str(state_id)
    if state_ref.startswith("ood:"):
        parts = state_ref.split(":")
        base_nid = parts[1] if len(parts) > 1 else None
        cluster = parts[2] if len(parts) > 2 else None
        dist = parts[4][1:] if len(parts) > 4 and parts[4].startswith("d") else None
        return {
            "state_ref": state_ref,
            "state_kind": "ood",
            "base_nid": base_nid,
            "state_cluster": cluster,
            "ood_distance": dist,
        }
    try:
        base_nid = int(state_id)
    except Exception:
        base_nid = None
    return {
        "state_ref": state_ref,
        "state_kind": "etg",
        "base_nid": base_nid,
        "state_cluster": None,
        "ood_distance": None,
    }


def _get_all_runs():
    runs = []
    legacy_db = _RESULTS_DIR / "study.db"
    if legacy_db.exists():
        runs.append(("run_0001", _RESULTS_DIR, True))
    tr_dir = _TRAINING_RUNS_DIR
    if tr_dir.exists():
        for d in sorted(tr_dir.iterdir()):
            if d.is_dir() and d.name.startswith("run_"):
                db = d / "study.db"
                if db.exists():
                    runs.append((d.name, d, False))
    return runs


def _get_active_run_path():
    sel = st.session_state.get("_active_run", None)
    if not sel:
        return None
    if sel == "run_0001":
        return _RESULTS_DIR
    return _TRAINING_RUNS_DIR / sel


def _get_active_study_db():
    run_path = _get_active_run_path()
    if run_path is None:
        return None
    return run_path / "study.db"


def _get_active_runs_dir():
    run_path = _get_active_run_path()
    if run_path is None:
        return None
    return run_path / "runs"


def _get_active_trials_dir():
    run_path = _get_active_run_path()
    if run_path is None:
        return None
    return run_path / "trials"


_ACTION_STRATEGY_LABELS = {
    "best_beam": "Best Beam",
    "best_subtree_quality": "Best Subtree Quality",
    "best_subtree_winrate": "Best Subtree WinRate",
    "highest_transition_prob": "Highest Trans. Prob",
    "random_beam": "Random Beam",
    "epsilon_greedy": "Epsilon-Greedy",
}


def _load_summary():
    run_path = _get_active_run_path()
    if run_path is None:
        return None
    sp = run_path / "study_summary.json"
    if sp.exists():
        try:
            with open(str(sp), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


@st.cache_resource(ttl=60, show_spinner=False)
def _load_study(_run_key: str = ""):
    study_db = _get_active_study_db()
    if study_db and study_db.exists():
        try:
            storage = f"sqlite:///{study_db}"
            study = optuna.load_study(study_name="beam_search", storage=storage)
            return study
        except Exception:
            pass
    return None


@st.cache_data(ttl=10, show_spinner=False)
def _get_running_trial_number():
    if not _PID_FILE.exists():
        return None
    if not _is_learner_alive():
        return None
    study_db = _get_active_study_db()
    if not study_db:
        return None
    try:
        import sqlite3 as _sqlite

        db = _sqlite.connect(str(study_db))
        cur = db.cursor()
        cur.execute(
            "SELECT t.number FROM trials t WHERE t.state = 'RUNNING' ORDER BY t.number DESC LIMIT 1"
        )
        row = cur.fetchone()
        db.close()
        return row[0] if row else None
    except Exception:
        return None


def _get_running_trial(study):
    if not study:
        return None
    num = _get_running_trial_number()
    if num is None:
        return None
    for t in study.trials:
        if t.number == num:
            return t
    return None


_PID_FILE = _RESULTS_DIR / ".learner_pid"


@st.cache_data(ttl=15, show_spinner=False)
def _is_learner_alive():
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return False
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return str(pid) in r.stdout.decode(errors="replace")
    except Exception:
        return False


def _kill_learner_process():
    if not _PID_FILE.exists():
        return
    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    _is_learner_alive.clear()


def _kill_port_process(port: int):
    try:
        r = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        output = r.stdout.decode(errors="replace")
        for line in output.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit():
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", pid],
                            capture_output=True,
                            creationflags=subprocess.CREATE_NO_WINDOW
                            if sys.platform == "win32"
                            else 0,
                        )
    except Exception:
        pass


def _delete_trial(trial_number: int):
    runs_dir = _get_active_runs_dir()
    trials_dir = _get_active_trials_dir()
    if runs_dir:
        run_json = runs_dir / f"trial_{trial_number:04d}_run.json"
        run_log = runs_dir / f"trial_{trial_number:04d}.log"
        run_json.unlink(missing_ok=True)
        run_log.unlink(missing_ok=True)
    if trials_dir:
        trial_dir = trials_dir / f"trial_{trial_number:04d}"
        if trial_dir.is_dir():
            shutil.rmtree(trial_dir, ignore_errors=True)

    study = _load_study(st.session_state.get("_active_run", ""))
    if not study:
        return
    for t in study.trials:
        if t.number == trial_number:
            try:
                study._storage.delete_trial(t._trial_id)
            except Exception:
                try:
                    study._storage.set_trial_state(
                        t._trial_id, optuna.trial.TrialState.FAIL
                    )
                except Exception:
                    pass
            break


@st.cache_data(ttl=120, show_spinner=False)
def _get_run_info():
    runs = _get_all_runs()
    result = {}
    for run_name, run_path, is_legacy in runs:
        study_db = run_path / "study.db"
        if not study_db.exists():
            continue
        import sqlite3 as _sqlite

        try:
            db = _sqlite.connect(str(study_db))
            cur = db.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM trials WHERE state IN ('COMPLETE', 'FAIL')"
            )
            count = cur.fetchone()[0]
            cur.execute(
                "SELECT MIN(number), MAX(number) FROM trials WHERE state IN ('COMPLETE', 'FAIL')"
            )
            row = cur.fetchone()
            min_num = row[0] if row and row[0] is not None else 0
            max_num = row[1] if row and row[1] is not None else 0
            db.close()
            result[run_name] = {
                "count": count,
                "min_trial": min_num,
                "max_trial": max_num,
                "path": run_path,
                "is_legacy": is_legacy,
            }
        except Exception:
            pass
    return result


def _delete_run(run_name: str):
    import gc

    run_info = _get_run_info()
    if run_name not in run_info:
        return
    info = run_info[run_name]
    run_path = info["path"]

    if _is_learner_alive():
        _kill_learner_process()
        time.sleep(1)
    if "learner_proc" in st.session_state:
        proc = st.session_state.get("learner_proc")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        del st.session_state["learner_proc"]

    _clear_all_cache()
    gc.collect()
    time.sleep(0.3)

    if run_path == _RESULTS_DIR:
        study_db = run_path / "study.db"
        if study_db.exists():
            study_db.unlink(missing_ok=True)
        runs_dir = run_path / "runs"
        if runs_dir.exists():
            for f in runs_dir.glob("trial_*_run.json"):
                f.unlink(missing_ok=True)
        trials_dir = run_path / "trials"
        if trials_dir.exists():
            for d in trials_dir.glob("trial_*"):
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
        sp = run_path / "study_summary.json"
        sp.unlink(missing_ok=True)
    else:
        study_db = run_path / "study.db"
        for _attempt in range(5):
            if not study_db.exists():
                break
            try:
                study_db.unlink()
            except Exception:
                gc.collect()
                time.sleep(0.5)
        for _attempt in range(5):
            if not run_path.exists():
                break
            try:
                shutil.rmtree(str(run_path))
            except Exception:
                gc.collect()
                time.sleep(0.5)
        if run_path.exists():
            for f in run_path.rglob("*"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                run_path.rmdir()
            except Exception:
                pass
    _clear_all_cache()


def _clear_all_cache():
    for fn in (
        _get_run_info,
        _load_finetune_runs,
    ):
        try:
            fn.clear()
        except Exception:
            pass


def _export_all_data():
    study_db = _get_active_study_db()
    if not study_db or not study_db.exists():
        st.toast("无数据可导出")
        return None
    import sqlite3 as _sqlite

    try:
        db = _sqlite.connect(str(study_db))
        cur = db.cursor()
        cur.execute("""
            SELECT t.number, t.state, tv.value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            ORDER BY t.number
        """)
        trial_values_rows = cur.fetchall()

        cur.execute("""
            SELECT t.number, tp.param_name, tp.param_value
            FROM trials t
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            ORDER BY t.number
        """)
        params_rows = cur.fetchall()

        cur.execute("""
            SELECT t.number, tua.key, tua.value_json
            FROM trials t
            JOIN trial_user_attributes tua ON t.trial_id = tua.trial_id
            ORDER BY t.number
        """)
        attrs_rows = cur.fetchall()
        db.close()
    except Exception:
        st.toast("导出失败：数据库读取错误")
        return None

    params_map: dict = {}
    for num, pname, pval in params_rows:
        params_map.setdefault(num, {})[pname] = pval

    attrs_map: dict = {}
    for num, key, val_json in attrs_rows:
        attrs_map.setdefault(num, {})[key] = val_json

    value_map: dict = {}
    for num, state, val in trial_values_rows:
        if val is not None:
            value_map[num] = val

    trials = []
    for num in sorted(
        set(list(params_map.keys()) + list(attrs_map.keys()) + list(value_map.keys()))
    ):
        params = params_map.get(num, {})
        raw_attrs = attrs_map.get(num, {})
        attrs = {}
        for k, v in raw_attrs.items():
            try:
                attrs[k] = json.loads(v)
            except Exception:
                attrs[k] = v

        trial_data = {
            "trial": num,
            "objective": value_map.get(num),
            "params": params,
            "metrics": {
                "win_rate": attrs.get("win_rate", 0),
                "avg_score": attrs.get("avg_score", 0),
                "score_std": attrs.get("score_std", 0),
                "stability": attrs.get("stability", 0),
                "num_episodes": attrs.get("num_episodes", 0),
            },
            "user_attrs": {
                "status": attrs.get("status"),
                "batch": attrs.get("batch"),
                "source_trial": attrs.get("source_trial"),
                "penalty_factor": attrs.get("penalty_factor"),
                "result_file": attrs.get("result_file"),
            },
        }
        trials.append(trial_data)

    export = {
        "export_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_trials": len(trials),
        "trials": trials,
    }
    return json.dumps(export, ensure_ascii=False, indent=2)


def _render_export_button():
    if "learner_export_data" not in st.session_state:
        st.session_state.learner_export_data = None
    if st.button("导出数据", key="learner_export_btn"):
        data = _export_all_data()
        if data:
            st.session_state.learner_export_data = data
    if st.session_state.learner_export_data:
        st.download_button(
            "下载 JSON",
            data=st.session_state.learner_export_data.encode("utf-8"),
            file_name=f"learner_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="learner_download_btn",
        )


def _start_learner():
    if _is_learner_alive():
        st.error("参数寻优正在运行中，请先停止。")
        return

    episodes = st.session_state.get("learner_episodes", 100)
    trials = st.session_state.get("learner_trials", 50)
    masked = st.session_state.get("learner_masked_actions", [])
    expanded_masked = []
    for letter in masked:
        for c in range(5):
            expanded_masked.append(f"{c}{letter}")

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "parameter_learner.py"),
        "--trials",
        str(trials),
        "--episodes",
        str(episodes),
        "--auto_archive",
    ]
    etg_file = st.session_state.get("_learner_etg_file", "")
    etg_data_dir = st.session_state.get("_learner_etg_data_dir", "")
    etg_map_id = st.session_state.get("_learner_etg_map_id", "")
    map_key = st.session_state.get("_learner_map_key") or get_map_key_for_map_id(etg_map_id)
    cmd.extend(["--map_key", map_key])
    if etg_file:
        cmd.extend(["--etg_file", etg_file])
    if etg_data_dir:
        cmd.extend(["--data_dir", etg_data_dir])
    if expanded_masked:
        cmd.extend(["--masked_actions", ",".join(expanded_masked)])

    restart_interval = st.session_state.get("learner_restart_interval", 0)
    if restart_interval > 0:
        cmd.extend(["--restart_interval", str(restart_interval)])

    if st.session_state.get("learner_use_counterfactual", False):
        cmd.append("--enable_counterfactual")
    if st.session_state.get("learner_use_action_tuning", False):
        cmd.append("--enable_action_tuning")

    alpha = st.session_state.get("learner_alpha", 0.2)
    cap = st.session_state.get("learner_cap", 8.0)
    _save_obj_config(alpha, cap)
    try:
        import yaml as _yaml

        cfg_path = ROOT_DIR / "configs" / "learner_config.yaml"
        cfg = {}
        if cfg_path.exists():
            with open(str(cfg_path), "r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f) or {}
        tuning_enabled = st.session_state.get("learner_use_action_tuning", False)
        game_cfg = cfg.setdefault("game", {})
        game_cfg["map_key"] = map_key
        game_cfg["etg_file"] = etg_file or None
        game_cfg["data_dir"] = etg_data_dir or None
        exec_cfg = cfg.setdefault("execution", {})
        exec_cfg["restart_interval"] = int(st.session_state.get("learner_restart_interval", 0))
        exec_cfg["restart_on_phase_change"] = bool(
            st.session_state.get("learner_restart_on_phase_change", True)
        )
        bktree_cfg = cfg.setdefault("bktree", {})
        bktree_cfg["primary_threshold"] = float(
            st.session_state.get("learner_bktree_primary_threshold", 1.0)
        )
        bktree_cfg["secondary_threshold"] = float(
            st.session_state.get("learner_bktree_secondary_threshold", 0.5)
        )
        tuning_cfg = cfg.setdefault("action_tuning", {})
        tuning_cfg["enabled"] = bool(tuning_enabled)
        tuning_cfg["explore_rate"] = st.session_state.get(
            "learner_tuning_explore_rate", tuning_cfg.get("explore_rate", 0.05)
        )
        tuning_cfg["min_confidence"] = st.session_state.get(
            "learner_tuning_min_confidence", tuning_cfg.get("min_confidence", 0.35)
        )
        tuning_cfg["min_advantage"] = st.session_state.get(
            "learner_tuning_min_advantage", tuning_cfg.get("min_advantage", 1.0)
        )
        tuning_cfg["ucb_c"] = st.session_state.get(
            "learner_tuning_ucb_c", tuning_cfg.get("ucb_c", 1.4)
        )
        tuning_cfg["target_visits"] = st.session_state.get(
            "learner_tuning_target_visits", tuning_cfg.get("target_visits", 10)
        )
        tuning_cfg["confidence_return_scale"] = st.session_state.get(
            "learner_tuning_confidence_return_scale",
            tuning_cfg.get("confidence_return_scale", 50.0),
        )
        tuning_cfg["validation_min_confidence"] = st.session_state.get(
            "learner_tuning_validation_min_confidence",
            tuning_cfg.get("validation_min_confidence", 0.10),
        )
        tuning_cfg["validation_min_advantage"] = st.session_state.get(
            "learner_tuning_validation_min_advantage",
            tuning_cfg.get("validation_min_advantage", 0.0),
        )
        tuning_cfg["validation_min_visits"] = st.session_state.get(
            "learner_tuning_validation_min_visits",
            tuning_cfg.get("validation_min_visits", 2),
        )
        tuning_cfg["ood_key_mode"] = st.session_state.get(
            "learner_tuning_ood_key_mode",
            tuning_cfg.get("ood_key_mode", "aggregate"),
        )
        tuning_cfg["ood_distance_bucket"] = st.session_state.get(
            "learner_tuning_ood_distance_bucket",
            tuning_cfg.get("ood_distance_bucket", 0.5),
        )
        tuning_cfg["max_nid_fallback_dist"] = st.session_state.get(
            "learner_max_nid_fallback_dist",
            tuning_cfg.get("max_nid_fallback_dist", 0.75),
        )
        tuning_cfg["max_nid_fallback_hp_dist"] = st.session_state.get(
            "learner_max_nid_fallback_hp_dist",
            tuning_cfg.get("max_nid_fallback_hp_dist", 1.5),
        )
        guard_cfg = tuning_cfg.setdefault("restart_guard", {})
        guard_cfg["enabled"] = bool(
            st.session_state.get("learner_restart_guard_enabled", True)
        )
        guard_cfg["warmup_episodes"] = int(
            st.session_state.get("learner_restart_warmup_episodes", 10)
        )
        guard_cfg["max_ood_ratio"] = float(
            st.session_state.get("learner_restart_guard_max_ood_ratio", 0.30)
        )
        guard_cfg["max_ood_mc_ratio"] = float(
            st.session_state.get("learner_restart_guard_max_ood_mc_ratio", 0.30)
        )
        guard_cfg["max_episode_frames"] = int(
            st.session_state.get("learner_restart_guard_max_episode_frames", 80)
        )
        guard_cfg["skip_model_update"] = bool(
            st.session_state.get("learner_restart_guard_skip_model_update", True)
        )
        guard_cfg["skip_bad_results"] = bool(
            st.session_state.get("learner_restart_guard_skip_bad_results", True)
        )
        guard_cfg["disable_ood_explore_on_violation"] = bool(
            st.session_state.get("learner_restart_guard_disable_ood_explore", True)
        )
        guard_cfg["allow_high_score_ood_update"] = bool(
            st.session_state.get("learner_restart_guard_allow_high_score_ood_update", True)
        )
        guard_cfg["high_score_ood_min_score"] = float(
            st.session_state.get("learner_restart_guard_high_score_ood_min_score", 24.0)
        )
        phase_cfg = cfg.setdefault("phased_optimization", {})
        phase_cfg["enabled"] = bool(st.session_state.get("learner_use_phased_optimization", False))
        phase_cfg["mode"] = st.session_state.get("learner_phase_mode", "cycle")
        phase_cfg["cycle"] = phase_cfg["mode"] == "cycle"
        phase_cfg["exclude_exploration_from_optimization"] = bool(
            st.session_state.get("learner_phase_exclude_explore", True)
        )
        phase_cfg["exploration_min_rate"] = float(
            st.session_state.get("learner_phase_exploration_min_rate", 0.20)
        )
        phase_cfg["etg_target_objective"] = float(
            st.session_state.get("learner_phase_etg_target_objective", 35.0)
        )
        phase_cfg["exploration_target_avg_score"] = float(
            st.session_state.get("learner_phase_exploration_target_avg_score", 10.0)
        )
        phase_cfg["synergy_etg_pool_size"] = int(
            st.session_state.get("learner_phase_synergy_etg_pool_size", 3)
        )
        phase_cfg["synergy_etg_selection"] = st.session_state.get(
            "learner_phase_synergy_etg_selection", "weighted"
        )
        phase_cfg["synergy_etg_weights"] = [0.6, 0.25, 0.15]
        phase_cfg["synergy_validation_min_confidence"] = float(
            st.session_state.get("learner_phase_synergy_validation_min_confidence", 0.10)
        )
        phase_cfg["synergy_validation_min_advantage"] = float(
            st.session_state.get("learner_phase_synergy_validation_min_advantage", 0.0)
        )
        phase_cfg["synergy_validation_min_visits"] = int(
            st.session_state.get("learner_phase_synergy_validation_min_visits", 2)
        )
        phase_cfg["synergy_validation_sources"] = [
            "ood",
            "fallback",
            "etg_relaxed",
            "fuzzy_plan",
            "diverge",
        ]
        phase_cfg["synergy_validation_profiles"] = {
            "ood": {"min_confidence": 0.08, "min_advantage": 0.0, "min_visits": 2},
            "fallback": {"min_confidence": 0.08, "min_advantage": 0.0, "min_visits": 2},
            "etg_relaxed": {"min_confidence": 0.15, "min_advantage": 0.5, "min_visits": 3},
            "diverge": {"min_confidence": 0.15, "min_advantage": 0.5, "min_visits": 3},
            "fuzzy_plan": {"min_confidence": 0.15, "min_advantage": 0.5, "min_visits": 3},
        }
        phase_cfg["default_phase"] = "synergy"
        phase_cfg["stages"] = [
            {
                "name": "etg_only",
                "trials": int(st.session_state.get("learner_phase_etg_trials", 50)),
                "min_trials": int(st.session_state.get("learner_phase_etg_min_trials", 20)),
                "max_trials": int(st.session_state.get("learner_phase_etg_trials", 50)),
                "target_objective": float(st.session_state.get("learner_phase_etg_target_objective", 35.0)),
            },
            {
                "name": "exploration_only",
                "trials": int(st.session_state.get("learner_phase_explore_trials", 50)),
                "min_trials": int(st.session_state.get("learner_phase_explore_min_trials", 20)),
                "max_trials": int(st.session_state.get("learner_phase_explore_trials", 50)),
                "target_avg_score": float(st.session_state.get("learner_phase_exploration_target_avg_score", 10.0)),
            },
            {
                "name": "synergy",
                "trials": int(st.session_state.get("learner_phase_synergy_trials", 100)),
                "min_trials": int(st.session_state.get("learner_phase_synergy_min_trials", 50)),
                "max_trials": int(st.session_state.get("learner_phase_synergy_trials", 100)),
            },
        ]
        inc_enabled = st.session_state.get("learner_use_incremental_layer", False)
        inc_cfg = cfg.setdefault("incremental_layer", {})
        inc_cfg["enabled"] = bool(inc_enabled)
        inc_cfg["update_bktree"] = bool(
            st.session_state.get("learner_incremental_update_bktree", False)
        )
        inc_cfg["update_etg_delta"] = bool(
            st.session_state.get("learner_incremental_update_etg_delta", False)
        )
        inc_cfg["use_delta_for_planning"] = bool(
            st.session_state.get("learner_incremental_use_delta_for_planning", False)
        )
        inc_cfg["persist_interval_episodes"] = st.session_state.get(
            "learner_incremental_persist_interval",
            inc_cfg.get("persist_interval_episodes", 10),
        )
        inc_cfg.setdefault("min_new_state_distance", 1.0)
        inc_cfg.setdefault("delta_dir", "incremental_layer")
        with open(str(cfg_path), "w", encoding="utf-8") as f:
            _yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    except Exception as e:
        st.warning(f"动作微调配置保存失败: {e}")

    _TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        d.name
        for d in _TRAINING_RUNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    next_id = max((int(n.split("_")[1]) for n in existing), default=1) + 1
    new_run_name = f"run_{next_id:04d}"
    new_run_dir = _TRAINING_RUNS_DIR / new_run_name
    st.session_state["_new_run_dir"] = str(new_run_dir)
    st.session_state["_pending_active_run"] = new_run_name

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = st.session_state.get("_new_run_dir", None)
    if run_dir:
        cmd.extend(["--run_dir", str(run_dir)])
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "runs").mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "trials").mkdir(parents=True, exist_ok=True)
        log_path = Path(run_dir) / "learner.log"
    else:
        (_RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
        (_RESULTS_DIR / "trials").mkdir(parents=True, exist_ok=True)
        log_path = _RESULTS_DIR / "trials" / "learner.log"
    log_file = open(str(log_path), "w", encoding="utf-8")
    st.session_state.learner_log_file = log_file
    st.session_state._active_run_log = log_path
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=str(ROOT_DIR),
        creationflags=flags,
    )
    st.session_state.learner_proc = p
    _PID_FILE.write_text(str(p.pid))
    st.toast(f"参数寻优已启动 (PID: {p.pid})", icon="🚀")
    time.sleep(1)
    st.rerun()


def _start_rerun(trial_numbers):
    if _is_learner_alive():
        st.error("参数寻优正在运行中，请先停止。")
        return

    episodes = st.session_state.get("learner_episodes", 100)

    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "parameter_learner.py"),
        "--rerun",
        str(trial_numbers[0]),
        "--episodes",
        str(episodes),
        "--resume",
    ]
    etg_file = st.session_state.get("_learner_etg_file", "")
    etg_data_dir = st.session_state.get("_learner_etg_data_dir", "")
    etg_map_id = st.session_state.get("_learner_etg_map_id", "")
    map_key = st.session_state.get("_learner_map_key") or get_map_key_for_map_id(etg_map_id)
    cmd.extend(["--map_key", map_key])
    if etg_file:
        cmd.extend(["--etg_file", etg_file])
    if etg_data_dir:
        cmd.extend(["--data_dir", etg_data_dir])

    active_path = _get_active_run_path()
    if active_path and active_path != _RESULTS_DIR:
        cmd.extend(["--run_dir", str(active_path)])

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = _get_active_runs_dir()
    trials_dir = _get_active_trials_dir()
    if runs_dir:
        runs_dir.mkdir(parents=True, exist_ok=True)
    if trials_dir:
        trials_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        (trials_dir / "learner.log")
        if trials_dir
        else (_RESULTS_DIR / "trials" / "learner.log")
    )
    log_file = open(str(log_path), "w", encoding="utf-8")
    st.session_state.learner_log_file = log_file
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=str(ROOT_DIR),
        creationflags=flags,
    )
    st.session_state.learner_proc = p
    _PID_FILE.write_text(str(p.pid))
    st.toast(f"重跑已启动 (PID: {p.pid})", icon="🚀")
    time.sleep(1)
    st.rerun()


_CONFIG_PATH = ROOT_DIR / "configs" / "learner_config.yaml"

_DEFAULT_SEARCH_SPACE = {
    "mode": ["single_step", "multi_step"],
    "beam_width": [1, 10],
    "lookahead_steps": [1, 15],
    "score_mode": ["quality", "future_reward", "win_rate"],
    "action_strategy": [
        "best_beam",
        "best_subtree_quality",
        "best_subtree_winrate",
        "highest_transition_prob",
        "random_beam",
        "epsilon_greedy",
    ],
    "min_visits": [1, 10],
    "max_state_revisits": [1, 5],
    "min_cum_prob": [0.001, 0.1],
    "discount_factor": [0.5, 1.0],
    "enable_backup": [True, False],
    "epsilon": [0.01, 0.5],
    "backup_score_threshold": [0.0, 1.0],
    "backup_distance_threshold": [0.0, 1.0],
}

_INT_PARAMS = {"beam_width", "lookahead_steps", "min_visits", "max_state_revisits"}
_FLOAT_PARAMS = {
    "min_cum_prob",
    "discount_factor",
    "epsilon",
    "backup_score_threshold",
    "backup_distance_threshold",
}
_CATEGORY_PARAMS = {"score_mode", "action_strategy", "mode"}


def _load_config_space() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("search_space", {})
        except Exception:
            pass
    return {}


def _save_config_space(space: dict):
    cfg = {}
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    cfg["search_space"] = space
    with open(str(_CONFIG_PATH), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _save_obj_config(alpha, cap):
    cfg = {}
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    cfg.setdefault("objective", {})
    cfg["objective"]["stability_alpha"] = alpha
    cfg["objective"]["stability_cap"] = cap
    with open(str(_CONFIG_PATH), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _render_space_editor():
    space = _load_config_space()
    if not space:
        space = dict(_DEFAULT_SEARCH_SPACE)

    changed = False

    with st.container(border=True):
        st.markdown("**搜索空间**")

        for key in _INT_PARAMS:
            if key in space and isinstance(space[key], list) and len(space[key]) == 2:
                lo, hi = space[key]
                c1, c2 = st.columns(2)
                with c1:
                    new_lo = st.number_input(
                        f"{key} 最小", min_value=0, value=int(lo), key=f"sp_{key}_lo"
                    )
                with c2:
                    new_hi = st.number_input(
                        f"{key} 最大", min_value=0, value=int(hi), key=f"sp_{key}_hi"
                    )
                if new_lo != int(lo) or new_hi != int(hi):
                    space[key] = [int(new_lo), int(new_hi)]
                    changed = True

        for key in _FLOAT_PARAMS:
            if key in space and isinstance(space[key], list) and len(space[key]) == 2:
                lo, hi = space[key]
                c1, c2 = st.columns(2)
                with c1:
                    new_lo = st.number_input(
                        f"{key} 最小",
                        value=float(lo),
                        step=0.001,
                        format="%g",
                        key=f"sp_{key}_lo",
                    )
                with c2:
                    new_hi = st.number_input(
                        f"{key} 最大",
                        value=float(hi),
                        step=0.001,
                        format="%g",
                        key=f"sp_{key}_hi",
                    )
                if new_lo != float(lo) or new_hi != float(hi):
                    space[key] = [float(new_lo), float(new_hi)]
                    changed = True

        for key in _CATEGORY_PARAMS:
            if key in space and isinstance(space[key], list):
                val_str = ", ".join(str(v) for v in space[key])
                new_str = st.text_input(f"{key}", value=val_str, key=f"sp_{key}_cat")
                new_list = [v.strip() for v in new_str.split(",") if v.strip()]
                if new_list != space[key]:
                    space[key] = new_list
                    changed = True

        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button(
                "保存修改", use_container_width=True, key="sp_save", type="primary"
            ):
                _save_config_space(space)
                st.toast("搜索空间已保存", icon="✅")
        with col_reset:
            if st.button("恢复默认", use_container_width=True, key="sp_reset"):
                _save_config_space(dict(_DEFAULT_SEARCH_SPACE))
                st.toast("已恢复默认搜索空间", icon="🔄")
                st.rerun()


def _show_log(filename, base_dir=None):
    log_path = (base_dir or _RESULTS_DIR) / filename
    if not log_path.exists():
        st.info("日志文件不存在。")
        return
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
        lines = content.strip().split("\n")
        show_lines = "\n".join(lines[-50:]) if len(lines) > 50 else content
        st.text_area(
            "日志（最近 50 行）", show_lines, height=300, label_visibility="collapsed"
        )
    except Exception as e:
        st.error(f"读取日志失败: {e}")


def _render_learner_sidebar(etg_entry: Optional[Dict] = None):
    etg_file = etg_entry.get("file", "") if etg_entry else ""
    etg_name = etg_entry.get("name", "") if etg_entry else ""
    etg_data_dir = etg_entry.get("data_dir", "") if etg_entry else ""
    etg_map_id = etg_entry.get("map_id", "") if etg_entry else ""
    etg_map_key = get_map_key_for_map_id(etg_map_id)

    st.session_state["_learner_etg_file"] = etg_file
    st.session_state["_learner_etg_data_dir"] = etg_data_dir
    st.session_state["_learner_etg_map_id"] = etg_map_id
    st.session_state["_learner_map_key"] = etg_map_key

    if etg_file:
        st.caption(f"ETG: {etg_name}")
        st.caption(f"游戏场景: {etg_map_key} ({etg_map_id or '-'})")
        st.caption(f"路径: cache/experience_transition_graph/{etg_file}")
        if etg_data_dir:
            st.caption(f"数据目录: {etg_data_dir}")
            _bkt_summary = _get_bktree_summary(etg_data_dir)
            with st.expander("BKTree 详情", expanded=False):
                st.caption(f"Primary 节点: {_bkt_summary['primary_count']}")
                st.caption(f"Secondary 树: {_bkt_summary['secondary_count']}")
                st.caption(f"State 映射: {_bkt_summary['state_node_count']} 条")

    if "learner_proc" in st.session_state:
        proc = st.session_state.learner_proc
        if proc and proc.poll() is not None:
            del st.session_state.learner_proc

    study = _load_study(st.session_state.get("_active_run", ""))
    running_trial = _get_running_trial(study)

    threshold_defaults = get_bktree_threshold_defaults(etg_map_id)
    st.markdown("**BKTree 阈值**")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.number_input(
            "Primary",
            min_value=0.0,
            max_value=5.0,
            value=float(threshold_defaults["primary_threshold"]),
            step=0.05,
            key="learner_bktree_primary_threshold",
            help="规划决策阶段使用的 primary BKTree 最近邻接受阈值。",
        )
    with tc2:
        st.number_input(
            "Secondary",
            min_value=0.0,
            max_value=5.0,
            value=float(threshold_defaults["secondary_threshold"]),
            step=0.05,
            key="learner_bktree_secondary_threshold",
            help="规划决策阶段使用的 secondary BKTree 最近邻接受阈值。",
        )

    st.divider()

    if running_trial:
        st.warning(f"正在运行 Trial #{running_trial.number}")
        st.progress(0.5)
        if st.button(
            "停止并清理",
            use_container_width=True,
            key="learner_stop",
            type="secondary",
        ):
            _kill_learner_process()
            time.sleep(1)
            _PID_FILE.unlink(missing_ok=True)
            st.toast("已停止并清理", icon="🛑")
            st.rerun()
        if st.button("刷新进度", use_container_width=True, key="learner_refresh"):
            st.rerun()
        if st.button("查看日志", use_container_width=True, key="learner_show_log"):
            _show_log("trials/learner.log")
        return

    st.number_input(
        "每轮对局数", min_value=10, max_value=1000, value=100, key="learner_episodes"
    )

    total_default = 50
    if study:
        completed_count = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        total_default = max(50, completed_count + 10)
    st.number_input(
        "总试验轮数",
        min_value=1,
        max_value=5000,
        value=total_default,
        key="learner_trials",
    )

    st.number_input(
        "重启间隔（每N轮重启游戏客户端）",
        min_value=0,
        max_value=500,
        value=0,
        key="learner_restart_interval",
        help="0 = 不按固定 trial 间隔重启。建议改用阶段切换重启。",
    )
    st.toggle(
        "阶段切换时重启游戏客户端",
        value=True,
        key="learner_restart_on_phase_change",
        help="每次 ETG / 探索 / 协同阶段切换前重启客户端，确保新阶段参数完整加载。",
    )

    st.divider()
    st.markdown("**目标函数**")

    _obj_cfg = {}
    if _CONFIG_PATH.exists():
        try:
            with open(str(_CONFIG_PATH), "r", encoding="utf-8") as f:
                _full_cfg = yaml.safe_load(f) or {}
            _obj_cfg = _full_cfg.get("objective", {})
        except Exception:
            pass

    alpha = st.slider(
        "稳定性惩罚强度 (alpha)",
        0.0,
        1.0,
        _obj_cfg.get("stability_alpha", 0.5),
        step=0.05,
        key="learner_alpha",
        help="0=不惩罚稳定性，1=极不稳定时目标值完全归零",
    )
    cap = st.number_input(
        "稳定性归一化上限 (cap)",
        min_value=0.1,
        max_value=10.0,
        value=_obj_cfg.get("stability_cap", 5.0),
        step=0.1,
        format="%g",
        key="learner_cap",
        help="stability 归一化参考值，超过此值按 cap 计算",
    )
    st.caption("`win_rate x avg_score x max(1-alpha x min(stability/cap,1), 0)`")

    st.divider()
    st.markdown("**动作屏蔽**")
    _ACTION_NAMES = [
        "ATK_nearest",
        "ATK_clu_nearest",
        "ATK_nearest_weakest",
        "ATK_clu_nearest_weakest",
        "ATK_threatening",
        "DEF_clu_nearest",
        "MIX_gather",
        "MIX_lure",
        "MIX_sacrifice_lure",
        "do_randomly",
        "do_nothing",
    ]
    _MASKED_ACTION_OPTIONS = {
        chr(ord("a") + a): f"{chr(ord('a') + a)} - {_ACTION_NAMES[a]}"
        for a in range(len(_ACTION_NAMES))
    }
    st.multiselect(
        "手动屏蔽动作（按类别，覆盖所有聚类粒度）",
        options=list(_MASKED_ACTION_OPTIONS.keys()),
        format_func=lambda x: _MASKED_ACTION_OPTIONS.get(x, x),
        key="learner_masked_actions",
        help="选择动作类别后，该动作在所有5种聚类粒度下均被屏蔽",
    )

    st.divider()
    st.markdown("**状态微调机**")

    use_finetune = st.toggle(
        "启用状态微调机",
        value=False,
        key="learner_use_finetune",
        help="在规划阶段对低置信度状态的动作选择进行修正",
    )

    if use_finetune:
        model_options = ["(无模型)"]
        shared_model = _RESULTS_DIR / "shared_finetune_model.pkl"
        if shared_model.exists():
            model_options.append("shared_finetune_model.pkl")
        for p in sorted(_RESULTS_DIR.glob("finetune_model_group_*.pkl")):
            model_options.append(p.name)
        old_model_path = _RESULTS_DIR / "finetune_model.pkl"
        if old_model_path.exists():
            model_options.append("finetune_model.pkl")

        st.selectbox(
            "选择模型",
            model_options,
            key="learner_finetune_model",
        )

        st.slider(
            "微调阈值",
            0.1,
            1.0,
            0.4,
            step=0.05,
            key="learner_finetune_threshold",
            help="replacement_score 低于此值时用微调模型替换 beam search 建议",
        )

    st.divider()
    st.markdown("**蒙特卡洛动作微调探索**")

    st.toggle(
        "启用动作微调探索",
        value=False,
        key="learner_use_action_tuning",
        help="在参数寻优过程中维护独立动作微调模型，通过 UCB 探索与置信度路由判断使用 ETG 还是微调动作。",
    )
    if st.session_state.get("learner_use_action_tuning", False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.slider(
                "探索率",
                0.0,
                0.5,
                0.05,
                step=0.01,
                key="learner_tuning_explore_rate",
                help="非低置信状态下仍随机触发 UCB 探索的概率",
            )
            st.slider(
                "最小置信度",
                0.0,
                1.0,
                0.35,
                step=0.05,
                key="learner_tuning_min_confidence",
            )
        with col_t2:
            st.number_input(
                "最小优势",
                min_value=0.0,
                max_value=50.0,
                value=1.0,
                step=0.5,
                key="learner_tuning_min_advantage",
            )
            st.number_input(
                "UCB 系数",
                min_value=0.0,
                max_value=5.0,
                value=1.4,
                step=0.1,
                key="learner_tuning_ucb_c",
            )
        st.number_input(
            "目标访问次数",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="learner_tuning_target_visits",
        )
        st.number_input(
            "confidence return scale",
            min_value=1.0,
            max_value=200.0,
            value=50.0,
            step=5.0,
            key="learner_tuning_confidence_return_scale",
            help="Larger values reduce the penalty from high return variance when computing action-tuning confidence.",
        )
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.slider(
                "validation min confidence",
                0.0,
                1.0,
                0.10,
                step=0.05,
                key="learner_tuning_validation_min_confidence",
                help="Lower gate used only for OOD/fallback/etg_relaxed validation opportunities.",
            )
        with col_v2:
            st.number_input(
                "validation min advantage",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                key="learner_tuning_validation_min_advantage",
            )
        with col_v3:
            st.number_input(
                "validation min visits",
                min_value=1,
                max_value=100,
                value=2,
                step=1,
                key="learner_tuning_validation_min_visits",
            )
        col_key1, col_key2 = st.columns(2)
        with col_key1:
            st.selectbox(
                "OOD state key",
                ["aggregate", "exact"],
                index=0,
                key="learner_tuning_ood_key_mode",
                help="aggregate groups OOD states by candidate_nid + cluster + distance bucket; exact keeps hash keys.",
            )
        with col_key2:
            st.number_input(
                "OOD distance bucket",
                min_value=0.05,
                max_value=5.0,
                value=0.5,
                step=0.05,
                key="learner_tuning_ood_distance_bucket",
            )
        col_nid1, col_nid2 = st.columns(2)
        with col_nid1:
            st.number_input(
                "NID fallback 最大距离",
                min_value=0.0,
                max_value=5.0,
                value=0.75,
                step=0.05,
                key="learner_max_nid_fallback_dist",
                help="超过该距离时拒绝最近邻 nid，改走 OOD 动作微调通道。",
            )
        with col_nid2:
            st.number_input(
                "NID fallback 最大 HP 距离",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.1,
                key="learner_max_nid_fallback_hp_dist",
            )
        st.markdown("**RestartGuard / OOD 熔断**")
        st.toggle(
            "启用重启后保护",
            value=True,
            key="learner_restart_guard_enabled",
            help="重启后 warm-up 期间只记录不更新模型，并在 OOD 过高时熔断 OOD 探索。",
        )
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.number_input(
                "Warm-up episodes",
                min_value=0,
                max_value=100,
                value=10,
                step=1,
                key="learner_restart_warmup_episodes",
            )
            st.number_input(
                "最大 OOD 占比",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.05,
                key="learner_restart_guard_max_ood_ratio",
            )
        with col_g2:
            st.number_input(
                "最大 OOD 探索占比",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.05,
                key="learner_restart_guard_max_ood_mc_ratio",
            )
            st.number_input(
                "最长 episode 帧数",
                min_value=0,
                max_value=500,
                value=80,
                step=5,
                key="learner_restart_guard_max_episode_frames",
            )
        c_guard1, c_guard2, c_guard3 = st.columns(3)
        with c_guard1:
            st.toggle("跳过模型更新", value=True, key="learner_restart_guard_skip_model_update")
        with c_guard2:
            st.toggle("跳过坏结果更新", value=True, key="learner_restart_guard_skip_bad_results")
        with c_guard3:
            st.toggle("熔断 OOD 探索", value=True, key="learner_restart_guard_disable_ood_explore")
        c_guard4, c_guard5 = st.columns(2)
        with c_guard4:
            st.toggle(
                "allow high-score OOD update",
                value=True,
                key="learner_restart_guard_allow_high_score_ood_update",
                help="High-score OOD wins can still update action tuning instead of being discarded.",
            )
        with c_guard5:
            st.number_input(
                "high-score OOD min score",
                min_value=-100.0,
                max_value=200.0,
                value=24.0,
                step=1.0,
                key="learner_restart_guard_high_score_ood_min_score",
            )

    st.divider()
    st.markdown("**分阶段协同优化**")
    st.toggle(
        "启用 ETG / 探索 / 协同三阶段",
        value=False,
        key="learner_use_phased_optimization",
        help="阶段 A 仅 ETG 参数寻优；阶段 B 冻结 ETG 参数并强化动作微调探索；阶段 C 同时启用 ETG 与动作微调。",
    )
    if st.session_state.get("learner_use_phased_optimization", False):
        st.selectbox(
            "阶段调度模式",
            ["cycle", "adaptive"],
            index=0,
            key="learner_phase_mode",
            help="cycle=按 etg/探索/协同循环；adaptive=阶段达到目标或最大 trial 后切换。",
        )
        st.toggle(
            "探索阶段不更新参数寻优模型",
            value=True,
            key="learner_phase_exclude_explore",
            help="开启后 Exploration-only trial 只用于动作微调探索和日志，不作为 Optuna 参数优化样本。",
        )
        c_phase1, c_phase2, c_phase3 = st.columns(3)
        with c_phase1:
            st.number_input("etg-only trials", min_value=0, max_value=1000, value=20, step=5, key="learner_phase_etg_trials")
            st.number_input("ETG min trials", min_value=1, max_value=1000, value=10, step=5, key="learner_phase_etg_min_trials")
        with c_phase2:
            st.number_input("Exploration-only trials", min_value=0, max_value=1000, value=20, step=5, key="learner_phase_explore_trials")
            st.number_input("Explore min trials", min_value=1, max_value=1000, value=10, step=5, key="learner_phase_explore_min_trials")
        with c_phase3:
            st.number_input("Synergy trials", min_value=0, max_value=1000, value=50, step=5, key="learner_phase_synergy_trials")
            st.number_input("Synergy min trials", min_value=1, max_value=1000, value=20, step=5, key="learner_phase_synergy_min_trials")
            st.number_input("Synergy ETG pool size", min_value=1, max_value=20, value=3, step=1, key="learner_phase_synergy_etg_pool_size")
            st.selectbox("Synergy ETG selection", ["weighted", "round_robin", "best"], index=0, key="learner_phase_synergy_etg_selection")
        st.number_input(
            "探索阶段最小探索率",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            key="learner_phase_exploration_min_rate",
        )
        c_target1, c_target2 = st.columns(2)
        with c_target1:
            st.number_input("ETG 阶段目标 Objective", min_value=-100.0, max_value=200.0, value=35.0, step=1.0, key="learner_phase_etg_target_objective")
        with c_target2:
            st.number_input("探索阶段目标 Avg Score", min_value=-100.0, max_value=200.0, value=10.0, step=1.0, key="learner_phase_exploration_target_avg_score")

    st.divider()
    st.markdown("**增量层实验开关**")
    st.toggle(
        "启用增量层（默认不影响原 etg/BKTree）",
        value=False,
        key="learner_use_incremental_layer",
        help="启用后仅写入独立 delta 目录；除非开启 use_delta_for_planning，否则不参与规划。",
    )
    if st.session_state.get("learner_use_incremental_layer", False):
        st.checkbox(
            "写入 ETG delta",
            value=False,
            key="learner_incremental_update_etg_delta",
            help="将在线 episode 中的状态转移动作统计写入 etg_delta.pkl，不修改原 ETG。",
        )
        st.checkbox(
            "写入 BKTree delta（预留）",
            value=False,
            key="learner_incremental_update_bktree",
            help="当前仅保留开关与配置，后续实现新增 BKTree 节点写入。",
        )
        st.checkbox(
            "规划时使用 delta（预留）",
            value=False,
            key="learner_incremental_use_delta_for_planning",
            help="当前仅保留开关与配置，后续实现 base ETG + delta ETG 合并视图。",
        )
        st.number_input(
            "持久化间隔 episode",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="learner_incremental_persist_interval",
        )

    st.divider()

    if st.button(
        "查看/编辑搜索空间", use_container_width=True, key="learner_toggle_space"
    ):
        st.session_state._show_space_editor = not st.session_state.get(
            "_show_space_editor", False
        )

    if st.session_state.get("_show_space_editor", False):
        _render_space_editor()

    st.caption(f"数据目录: `{_RESULTS_DIR}`")

    if st.button(
        "启动参数寻优", type="primary", key="learner_start", use_container_width=True
    ):
        _start_learner()

    st.caption(
        "上方按钮: 独立参数寻优（仅优化 beam search 参数）。"
        "下方「训练记录与启动」视图: 在线协同训练（参数寻优 + 模型进化同步进行）。"
    )


def _get_filtered_trials(study):
    completed = _get_completed_trials(study, min_count=1)
    return completed


def _render_run_selector(key="_active_run"):
    pending = st.session_state.pop("_pending_active_run", None)
    if pending:
        st.session_state[key] = pending
        _clear_all_cache()
    run_info = _get_run_info()
    all_runs = _get_all_runs()
    if not all_runs:
        return None
    options = [r[0] for r in all_runs]
    labels = []
    for run_name, _, is_legacy in all_runs:
        info = run_info.get(run_name, {})
        count = info.get("count", 0)
        tag = " (legacy)" if is_legacy else ""
        labels.append(f"{run_name} ({count} trials){tag}")
    current = st.session_state.get(key, options[0])
    if current not in options:
        current = options[0]
        st.session_state[key] = current
    prev = current
    sel = st.radio(
        "Training Run",
        options,
        index=options.index(current),
        format_func=lambda x: labels[options.index(x)],
        horizontal=True,
        key=key,
    )
    if sel != prev:
        _clear_all_cache()
    return sel


def _render_conclusion_panel(filtered_trials):
    if not filtered_trials:
        st.info("无符合条件的数据。")
        return
    filtered_trials.sort(key=lambda t: t.value if t.value else 0, reverse=True)
    best = filtered_trials[0]
    threshold = filtered_trials[int(len(filtered_trials) * 0.9)]
    top_pct = filtered_trials[: max(1, int(len(filtered_trials) * 0.1))]
    threshold_val = threshold.value if threshold.value else 0

    st.subheader("结论")
    st.caption(
        f"基于 {len(filtered_trials)} 轮试验，Top-10% 阈值 objective >= {threshold_val:.4f}"
    )

    bc = st.columns(3)
    bc[0].metric("最佳 Objective", f"{best.value:.4f}", f"Trial #{best.number}")
    wr = best.user_attrs.get("win_rate", 0)
    sc = best.user_attrs.get("avg_score", 0)
    bc[1].metric("胜率", f"{wr:.1%}")
    bc[2].metric("平均得分", f"{sc:.1f}")

    st.markdown("**最优参数**")
    pc = st.columns(4)
    for i, (k, v) in enumerate(best.params.items()):
        label = _ACTION_STRATEGY_LABELS.get(v, v) if isinstance(v, str) else v
        with pc[i % 4]:
            st.metric(k, label)

    st.markdown("**Top-10% 参数区间**")
    numeric_keys, categorical_keys = _classify_params(top_pct)
    rows = []
    for k in numeric_keys:
        vals = [t.params[k] for t in top_pct if k in t.params]
        if vals:
            rows.append(
                {
                    "参数": k,
                    "Min": min(vals),
                    "Median": sorted(vals)[len(vals) // 2],
                    "Mean": round(sum(vals) / len(vals), 2),
                    "Max": max(vals),
                }
            )
    if rows:
        import pandas as _pd

        st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for k in categorical_keys:
        dist = {}
        for t in top_pct:
            v = t.params.get(k, "")
            dist[v] = dist.get(v, 0) + 1
        if dist:
            label_map = _ACTION_STRATEGY_LABELS if k == "action_strategy" else {}
            txt = ", ".join(
                f"{label_map.get(v, v)}: {c}"
                for v, c in sorted(dist.items(), key=lambda x: -x[1])
            )
            st.markdown(f"**{k}**: {txt}")

    st.markdown("**关键发现**")
    importance = _compute_importance(st.session_state.get("_active_run", ""))
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
        top_param = sorted_imp[0][0] if sorted_imp else ""
        top_vals = [t.params[top_param] for t in top_pct if top_param in t.params]
        if top_vals:
            median_val = sorted(top_vals)[len(top_vals) // 2]
            st.info(
                f"最关键参数: **{top_param}** (重要性 {sorted_imp[0][1]:.1%})，Top-10% 中位数: {median_val}"
            )
        top3 = ", ".join(f"{k} ({v:.1%})" for k, v in sorted_imp[:3])
        st.caption(f"参数重要性排名: {top3}")

    mask_mc = [t.params.get("masked_count", 0) for t in top_pct]
    if mask_mc:
        from collections import Counter

        _MASK_LETTERS = list("abcdefghijk")
        _MASK_NAMES = [
            "ATK_nearest",
            "ATK_clu_nearest",
            "ATK_nearest_weakest",
            "ATK_clu_nearest_weakest",
            "ATK_threatening",
            "DEF_clu_nearest",
            "MIX_gather",
            "MIX_lure",
            "MIX_sacrifice_lure",
            "do_randomly",
            "do_nothing",
        ]

        def _mask_label(v):
            if v is None or not isinstance(v, int):
                return "N/A"
            if v < len(_MASK_LETTERS):
                return f"{_MASK_LETTERS[v]} ({_MASK_NAMES[v]})"
            return str(v)

        mc_mode = Counter(mask_mc).most_common(1)[0]
        mask_1_vals = [
            t.params.get("mask_1")
            for t in top_pct
            if t.params.get("masked_count", 0) >= 2
            and t.params.get("mask_1") is not None
        ]
        mask_0_vals = [
            t.params.get("mask_0")
            for t in top_pct
            if t.params.get("masked_count", 0) >= 2
            and t.params.get("mask_0") is not None
        ]
        m1_mode = Counter(mask_1_vals).most_common(1)[0][0] if mask_1_vals else "N/A"
        m0_min = min(mask_0_vals) if mask_0_vals else "N/A"
        m0_max = max(mask_0_vals) if mask_0_vals else "N/A"
        if isinstance(m0_min, int) and isinstance(m0_max, int):
            m0_range = f"{_mask_label(m0_min)}-{_mask_label(m0_max)}"
        else:
            m0_range = "N/A"
        st.info(
            f"最优 mask 模式: **masked_count={mc_mode[0]}** ({mc_mode[1]}/{len(top_pct)} Top-10%), "
            f"mask_1={_mask_label(m1_mode)}, mask_0={m0_range}"
        )

    if st.button("导出最优配置到 learner_config.yaml", key="export_best_config"):
        try:
            import yaml as _yaml

            cfg_path = ROOT_DIR / "configs" / "learner_config.yaml"
            if cfg_path.exists():
                with open(str(cfg_path), "r", encoding="utf-8") as f:
                    cfg = _yaml.safe_load(f)
                if "game" not in cfg:
                    cfg["game"] = {}
                cfg["game"]["best_params"] = dict(best.params)
                cfg["game"]["best_objective"] = float(best.value)
                cfg["game"]["best_trial"] = best.number
                with open(str(cfg_path), "w", encoding="utf-8") as f:
                    _yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
                st.success(f"已导出到 {cfg_path}")
            else:
                st.error("learner_config.yaml 不存在")
        except Exception as e:
            st.error(f"导出失败: {e}")

    st.divider()


def _plot_objective_history(study):
    if not study or len(study.trials) < 2:
        st.info("试验数据不足，至少需要 2 轮完成的试验。")
        return

    fig = go.Figure()

    trial_numbers = []
    values = []
    best_so_far = []
    best_val = -float("inf")

    for t in study.trials:
        plot_value = t.value
        if plot_value is None and t.user_attrs.get("probe_objective") is not None:
            plot_value = t.user_attrs.get("probe_objective")
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) and plot_value is not None:
            trial_numbers.append(t.number)
            values.append(float(plot_value))
            if not t.user_attrs.get("exclude_from_parameter_optimization", False):
                best_val = max(best_val, float(plot_value))
            elif best_val == -float("inf"):
                best_val = float(plot_value)
            best_so_far.append(best_val)

    if not trial_numbers:
        st.info("没有完成的试验。")
        return

    phase_colors = {
        "etg_only": "#636EFA",
        "exploration_only": "#00CC96",
        "synergy": "#AB63FA",
        "etg_focus": "#636EFA",
        "explore_focus": "#00CC96",
        "synergy_focus": "#AB63FA",
    }
    phase_fill_colors = {
        "etg_only": "rgba(99, 110, 250, 0.12)",
        "exploration_only": "rgba(0, 204, 150, 0.14)",
        "synergy": "rgba(171, 99, 250, 0.12)",
        "etg_focus": "rgba(99, 110, 250, 0.10)",
        "explore_focus": "rgba(0, 204, 150, 0.12)",
        "synergy_focus": "rgba(171, 99, 250, 0.10)",
    }
    phase_labels = {
        "etg_only": "etg-only",
        "exploration_only": "Exploration-only",
        "synergy": "ETG + 微调协同",
        "etg_focus": "ETG 参数寻优占优",
        "explore_focus": "动作探索占优",
        "synergy_focus": "ETG + 微调协同",
    }
    phase_by_trial = {}
    for t in study.trials:
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            phase = t.user_attrs.get("phase")
            if phase:
                phase_by_trial[int(t.number)] = str(phase)
    source_stats_df = pd.DataFrame()
    try:
        source_stats_df = _collect_action_source_stats()
        source_df = source_stats_df if not phase_by_trial else pd.DataFrame()
        if not source_df.empty and "trial" in source_df:
            phase_points = []
            for r in source_df.sort_values("trial").itertuples(index=False):
                trial = int(getattr(r, "trial"))
                etg_ratio = float(getattr(r, "etg_plan_ratio", 0.0) or 0.0) + float(
                    getattr(r, "etg_follow_ratio", 0.0) or 0.0
                ) + float(getattr(r, "etg_relaxed_ratio", 0.0) or 0.0)
                explore_ratio = sum(
                    float(getattr(r, col, 0.0) or 0.0)
                    for col in (
                        "mc_explore_ratio",
                        "ood_mc_explore_ratio",
                        "ood_ratio",
                        "fallback_ratio",
                    )
                )
                tune_ratio = float(getattr(r, "tuning_ratio", 0.0) or 0.0) + float(
                    getattr(r, "ood_tuning_ratio", 0.0) or 0.0
                )
                if etg_ratio >= 0.50 and tune_ratio + explore_ratio < 0.35:
                    phase = "etg_focus"
                elif explore_ratio >= 0.50 and etg_ratio < 0.35:
                    phase = "explore_focus"
                else:
                    phase = "synergy_focus"
                phase_points.append((trial, phase))
            if phase_points:
                phase_by_trial = dict(phase_points)
    except Exception:
        phase_by_trial = {}

    def _add_phase_regions(target_fig, y0, y1):
        if not phase_by_trial:
            return
        phase_seen = set()
        sorted_trials = [t for t in trial_numbers if t in phase_by_trial]
        if sorted_trials:
            start_trial = sorted_trials[0]
            current_phase = phase_by_trial[start_trial]
            prev_trial = start_trial
            intervals = []
            for trial in sorted_trials[1:]:
                phase = phase_by_trial.get(trial)
                if phase != current_phase or trial != prev_trial + 1:
                    intervals.append((start_trial, prev_trial, current_phase))
                    start_trial = trial
                    current_phase = phase
                prev_trial = trial
            intervals.append((start_trial, prev_trial, current_phase))

            for start_trial, end_trial, phase_key in intervals:
                if phase_key not in phase_labels:
                    continue
                showlegend = phase_key not in phase_seen
                phase_seen.add(phase_key)
                fig.add_trace(
                    go.Scatter(
                        x=[start_trial - 0.5, end_trial + 0.5, end_trial + 0.5, start_trial - 0.5, start_trial - 0.5],
                        y=[y0, y0, y1, y1, y0],
                        fill="toself",
                        fillcolor=phase_fill_colors.get(phase_key, "rgba(200,200,200,0.10)"),
                        line=dict(width=0, color=phase_colors.get(phase_key, "rgba(200,200,200,0.2)")),
                        mode="lines",
                        name=phase_labels[phase_key],
                        legendgroup=f"phase_{phase_key}",
                        showlegend=showlegend,
                        hoverinfo="skip",
                    )
                )

    if phase_by_trial:
        y_min = min(values)
        y_max = max(values)
        y_margin = max((y_max - y_min) * 0.08, 1.0)
        y0 = y_min - y_margin
        y1 = y_max + y_margin
        _add_phase_regions(fig, y0, y1)

    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=values,
            mode="markers+lines",
            name="目标值",
            line=dict(color="#636EFA", width=1),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=best_so_far,
            mode="lines",
            name="最优目标值",
            line=dict(color="#EF553B", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="优化目标值变化",
        xaxis_title="Trial #",
        yaxis_title="Objective",
        height=350,
        margin=dict(l=50, r=30, t=50, b=40),
        yaxis=dict(range=[min(values) - max((max(values) - min(values)) * 0.08, 1.0), max(values) + max((max(values) - min(values)) * 0.08, 1.0)]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            groupclick="togglegroup",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not source_stats_df.empty and "trial" in source_stats_df:
        balance_df = source_stats_df.sort_values("trial").copy()
        etg_direct = (
            balance_df.get("etg_plan_ratio", 0)
            + balance_df.get("etg_follow_ratio", 0)
            + balance_df.get("etg_relaxed_ratio", 0)
        )
        raw_explore = (
            balance_df.get("mc_explore_ratio", 0)
            + balance_df.get("ood_mc_explore_ratio", 0)
        )
        guard_keep = 1.0 - balance_df.get("guard_skip_update_ratio", 0).clip(0, 1)
        explore_update = raw_explore * guard_keep
        tuning_use = (
            balance_df.get("tuning_ratio", 0)
            + balance_df.get("ood_tuning_ratio", 0)
        )
        denominator = (etg_direct + explore_update + tuning_use).replace(0, np.nan)
        balance_df["explore_etg_balance"] = (
            (explore_update - etg_direct) / denominator
        ).fillna(0).clip(-1, 1)
        balance_df["tuning_conversion"] = (
            tuning_use - raw_explore
        ).fillna(0).clip(-1, 1)
        balance_df["etg_external_signal"] = (
            balance_df.get("nid_ood_ratio", 0)
            + balance_df.get("fallback_ratio", 0)
            + balance_df.get("ood_ratio", 0)
        ).clip(0, 1)

        balance_fig = go.Figure()
        _add_phase_regions(balance_fig, -1.0, 1.0)
        balance_fig.add_trace(
            go.Scatter(
                x=balance_df["trial"],
                y=balance_df["explore_etg_balance"],
                mode="lines+markers",
                name="探索更新 - ETG利用平衡",
                line=dict(color="#00CC96", width=2),
            )
        )
        balance_fig.add_trace(
            go.Scatter(
                x=balance_df["trial"],
                y=balance_df["tuning_conversion"],
                mode="lines+markers",
                name="微调利用 - 主动探索转化",
                line=dict(color="#AB63FA", width=2),
            )
        )
        balance_fig.add_trace(
            go.Scatter(
                x=balance_df["trial"],
                y=balance_df["etg_external_signal"],
                mode="lines",
                name="ETG外/不确定信号",
                line=dict(color="#FFA15A", width=1.5, dash="dot"),
            )
        )
        if "tuning_opportunity_ratio" in balance_df:
            balance_fig.add_trace(
                go.Scatter(
                    x=balance_df["trial"],
                    y=balance_df["tuning_opportunity_ratio"],
                    mode="lines+markers",
                    name="tuning opportunity ratio",
                    line=dict(color="#19D3F3", width=2, dash="dash"),
                )
            )
        if "tuning_accept_per_opportunity" in balance_df:
            balance_fig.add_trace(
                go.Scatter(
                    x=balance_df["trial"],
                    y=balance_df["tuning_accept_per_opportunity"],
                    mode="lines+markers",
                    name="tuning accept/opportunity",
                    line=dict(color="#FF6692", width=2, dash="dash"),
                )
            )
        if "tuning_candidate_eligible_ratio" in balance_df:
            balance_fig.add_trace(
                go.Scatter(
                    x=balance_df["trial"],
                    y=balance_df["tuning_candidate_eligible_ratio"],
                    mode="lines+markers",
                    name="eligible tuning candidate ratio",
                    line=dict(color="#B6E880", width=2, dash="dot"),
                )
            )
        balance_fig.add_hline(y=0, line=dict(color="rgba(80,80,80,0.45)", width=1))
        balance_fig.add_hline(y=1, line=dict(color="rgba(80,80,80,0.20)", width=1, dash="dot"))
        balance_fig.add_hline(y=-1, line=dict(color="rgba(80,80,80,0.20)", width=1, dash="dot"))
        balance_fig.update_layout(
            title="ETG 利用 ↔ 探索微调平衡变化",
            xaxis_title="Trial #",
            yaxis_title="Balance Index (-1=ETG利用, +1=探索更新)",
            yaxis=dict(range=[-1.05, 1.05]),
            height=350,
            margin=dict(l=50, r=30, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                groupclick="togglegroup",
            ),
        )
        st.caption(
            "`探索更新 - ETG利用平衡` 越接近 +1 表示探索并可更新模型越多，越接近 -1 表示直接利用 ETG 越多；"
            "`微调利用 - 主动探索转化` 越高表示已学到的微调动作开始替代主动探索。"
        )
        st.plotly_chart(balance_fig, use_container_width=True)


@st.cache_data(ttl=120, show_spinner=False)
def _compute_importance(_run_key: str = ""):
    study = _load_study(_run_key)
    if not study or len(study.trials) < 5:
        return None
    try:
        return dict(optuna.importance.get_param_importances(study))
    except Exception:
        return None


def _plot_importance(study):
    if not study or len(study.trials) < 5:
        st.info("试验数据不足，至少需要 5 轮完成才能计算参数重要性。")
        return

    importance = _compute_importance(st.session_state.get("_active_run", ""))
    if importance is None:
        st.info("无法计算参数重要性（可能缺少足够的参数变化）。")
        return

    if not importance:
        st.info("参数重要性数据为空。")
        return

    params = list(importance.keys())
    values = list(importance.values())
    sorted_pairs = sorted(zip(values, params), reverse=True)
    values = [p[0] for p in sorted_pairs]
    params = [p[1] for p in sorted_pairs]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=params,
            orientation="h",
            marker_color="#636EFA",
        )
    )
    fig.update_layout(
        title="参数重要性",
        xaxis_title="重要性",
        yaxis_title="参数",
        height=680,
        margin=dict(l=150, r=30, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _get_completed_trials(study, min_count=5):
    if not study or len(study.trials) < min_count:
        return []
    return [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def _classify_params(completed):
    all_keys = []
    for t in completed:
        for k in t.params:
            if k not in all_keys:
                all_keys.append(k)

    numeric_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), (int, float))
        and all(t.params.get(k) is not None for t in completed)
    ]

    categorical_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), str)
        and all(t.params.get(k) is not None for t in completed)
    ]

    return numeric_keys, categorical_keys


_METRIC_KEYS = ["objective", "win_rate", "avg_score", "stability"]
_METRIC_LABELS = {
    "objective": "Objective",
    "win_rate": "Win Rate",
    "avg_score": "Avg Score",
    "stability": "Stability",
}
_METRIC_COLORS = {
    "objective": "#636EFA",
    "win_rate": "#00CC96",
    "avg_score": "#EF553B",
    "stability": "#AB63FA",
}


def _get_trial_metrics(t) -> dict:
    return {
        "objective": float(t.value) if t.value is not None else None,
        "win_rate": t.user_attrs.get("win_rate"),
        "avg_score": t.user_attrs.get("avg_score"),
        "stability": t.user_attrs.get("stability"),
    }


def _plot_numeric_correlation(study):
    completed = _get_filtered_trials(study)
    if len(completed) < 5:
        st.info("完成的试验不足 5 轮。")
        return

    numeric_keys, _ = _classify_params(completed)

    if not numeric_keys:
        st.info("无数值型参数。")
        return

    valid_metrics = []
    for mk in _METRIC_KEYS:
        vals = [_get_trial_metrics(t).get(mk) for t in completed]
        if all(v is not None for v in vals):
            valid_metrics.append(mk)

    if not valid_metrics:
        st.info("指标数据不完整。")
        return

    corr_results = {}
    for mk in valid_metrics:
        metric_vals = [_get_trial_metrics(t)[mk] for t in completed]
        data_matrix = []
        for t in completed:
            row = [float(t.params[k]) for k in numeric_keys]
            row.append(float(metric_vals[completed.index(t)]))
            data_matrix.append(row)

        arr = np.array(data_matrix)
        if arr.shape[0] < 2:
            continue
        corr = np.corrcoef(arr, rowvar=False)
        corr_results[mk] = corr[-1, :-1]

    if not corr_results:
        st.info("数据不足以计算相关性。")
        return

    first_mk = valid_metrics[0]
    sorted_pairs = sorted(
        zip(numeric_keys, corr_results[first_mk]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    param_names = [p[0] for p in sorted_pairs]

    fig = go.Figure()
    for mk in valid_metrics:
        corr_vals = [corr_results[mk][numeric_keys.index(p)] for p in param_names]
        fig.add_trace(
            go.Bar(
                name=_METRIC_LABELS[mk],
                x=corr_vals,
                y=param_names,
                orientation="h",
                marker_color=_METRIC_COLORS[mk],
                text=[f"{v:+.3f}" for v in corr_vals],
                textposition="outside",
                textfont=dict(size=9),
            )
        )

    n_groups = len(param_names)
    fig.update_layout(
        title="参数相关性",
        xaxis_title="相关系数",
        xaxis_range=[-1.3, 1.3],
        yaxis_title="参数",
        height=680,
        margin=dict(l=150, r=60, t=50, b=40),
        yaxis=dict(autorange="reversed"),
        barmode="group",
        bargap=0.3,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    st.plotly_chart(fig, use_container_width=True)


def _plot_categorical_effect(study, cat_key):
    completed = _get_filtered_trials(study)
    if len(completed) < 3:
        st.info("试验数据不足。")
        return

    cat_labels = {
        "score_mode": "Score Mode",
        "action_strategy": "Action Strategy",
        "mode": "Mode",
        "enable_backup": "Backup",
    }
    label = cat_labels.get(cat_key, cat_key)

    groups = {}
    for t in completed:
        val = t.params.get(cat_key)
        if val is None:
            continue
        display = (
            _ACTION_STRATEGY_LABELS.get(val, val)
            if cat_key == "action_strategy"
            else val
        )
        metrics = _get_trial_metrics(t)
        groups.setdefault(display, []).append(metrics)

    if len(groups) < 2:
        st.info(f"{label} 只有单一取值。")
        return

    valid_metrics = [
        mk
        for mk in _METRIC_KEYS
        if all(g[0].get(mk) is not None for g in groups.values() if g)
    ]
    if not valid_metrics:
        st.info("指标数据不完整。")
        return

    sorted_groups = sorted(
        groups.items(),
        key=lambda x: np.mean([m.get("objective", 0) or 0 for m in x[1]]),
        reverse=True,
    )
    names = [name for name, _ in sorted_groups]

    has_avg_score = "avg_score" in valid_metrics
    if has_avg_score:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.55, 0.45],
        )
    else:
        fig = go.Figure()

    for mk in valid_metrics:
        means = [np.mean([m[mk] for m in grp]) for _, grp in sorted_groups]
        stds = [np.std([m[mk] for m in grp]) for _, grp in sorted_groups]

        text_labels = []
        for v in means:
            if mk == "win_rate":
                text_labels.append(f"{v:.1%}")
            elif mk == "avg_score":
                text_labels.append(f"{v:.1f}")
            else:
                text_labels.append(f"{v:.3f}")

        trace = go.Bar(
            name=_METRIC_LABELS[mk],
            x=names,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=_METRIC_COLORS[mk],
            text=text_labels,
            textposition="outside",
            textfont=dict(size=9),
            legendgroup=_METRIC_LABELS[mk],
            showlegend=True,
        )
        if has_avg_score:
            row = 2 if mk == "avg_score" else 1
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    layout_kwargs = dict(
        title=f"{label}",
        height=340,
        margin=dict(l=60, r=30, t=40, b=30),
        xaxis_tickangle=-25,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if has_avg_score:
        fig.update_yaxes(title_text="Obj / WR / Stability", row=1, col=1)
        fig.update_yaxes(title_text="Avg Score", row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(tickangle=-25, row=2, col=1)
    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)


def _plot_parallel_coordinates(study):
    if not study:
        return

    completed = _get_filtered_trials(study)
    if len(completed) < 3:
        st.info("试验数据不足。")
        return

    all_keys = []
    for t in completed:
        for k in t.params:
            if k not in all_keys:
                all_keys.append(k)

    numeric_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), (int, float))
        and all(t.params.get(k) is not None for t in completed)
    ]

    categorical_keys = [
        k
        for k in all_keys
        if isinstance(completed[0].params.get(k), str)
        and all(t.params.get(k) is not None for t in completed)
    ]

    cat_display_order = ["mode", "score_mode", "action_strategy", "enable_backup"]
    categorical_keys = [k for k in cat_display_order if k in categorical_keys]

    if not numeric_keys and not categorical_keys:
        st.info("无可显示的参数。")
        return

    dimensions = []

    for key in categorical_keys:
        unique_vals = sorted(set(t.params[key] for t in completed))
        val_to_int = {v: i for i, v in enumerate(unique_vals)}
        display_vals = [
            _ACTION_STRATEGY_LABELS.get(v, v) if key == "action_strategy" else v
            for v in unique_vals
        ]
        dimensions.append(
            dict(
                label=key,
                values=[val_to_int[t.params[key]] for t in completed],
                range=[-0.5, len(unique_vals) - 0.5],
                ticktext=display_vals,
                tickvals=list(range(len(unique_vals))),
            )
        )

    for key in numeric_keys:
        vals = [t.params[key] for t in completed]
        v_min, v_max = min(vals), max(vals)
        padding = (v_max - v_min) * 0.05 if v_max > v_min else 0.5
        dimensions.append(
            dict(
                label=key,
                values=vals,
                range=[v_min - padding, v_max + padding],
            )
        )

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=[t.value for t in completed],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Objective"),
            ),
            dimensions=dimensions,
        )
    )
    fig.update_layout(
        height=400,
        margin=dict(l=80, r=50, t=50, b=40),
        font=dict(color="#333", size=14),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_slice(study):
    if not study:
        return
    completed = _get_filtered_trials(study)
    if len(completed) < 5:
        st.info("试验数据不足（至少 5 轮）。")
        return
    numeric_keys, categorical_keys = _classify_params(completed)
    n_params = len(numeric_keys) + len(categorical_keys)
    if n_params == 0:
        st.info("无可显示的参数。")
        return
    n_rows = (n_params + 1) // 2
    from plotly.subplots import make_subplots

    specs = [[{}, {}] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        specs=specs,
        subplot_titles=[k for k in numeric_keys + categorical_keys],
    )
    for i, k in enumerate(numeric_keys):
        vals = [t.params[k] for t in completed if k in t.params]
        objs = [t.value for t in completed if k in t.params and t.value is not None]
        fig.add_trace(
            go.Scatter(
                x=vals,
                y=objs,
                mode="markers",
                marker=dict(size=4, opacity=0.6, color="#636EFA"),
                name=k,
                showlegend=False,
            ),
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )
    for j, k in enumerate(categorical_keys):
        offset = len(numeric_keys)
        idx = offset + j
        label_map = _ACTION_STRATEGY_LABELS if k == "action_strategy" else {}
        groups = {}
        for t in completed:
            v = t.params.get(k, "")
            if v not in groups:
                groups[v] = []
            if t.value is not None:
                groups[v].append(t.value)
        sorted_keys = sorted(groups.keys())
        labels = [label_map.get(v, v) for v in sorted_keys]
        for ci, v in enumerate(sorted_keys):
            fig.add_trace(
                go.Box(
                    y=groups[v],
                    x=[ci] * len(groups[v]),
                    name=k,
                    boxpoints="all",
                    jitter=0.3,
                    marker=dict(size=3, opacity=0.6),
                    showlegend=False,
                ),
                row=(idx // 2) + 1,
                col=(idx % 2) + 1,
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(sorted_keys))),
            ticktext=labels,
            row=(idx // 2) + 1,
            col=(idx % 2) + 1,
        )
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        height=250 * n_rows,
        margin=dict(l=60, r=20, t=30, b=30),
        title_text="参数切片分析",
    )
    st.plotly_chart(fig, use_container_width=True)


def _compute_pair_importance(completed, importance):
    numeric_keys, _ = _classify_params(completed)
    all_pairs = []
    for i in range(len(numeric_keys)):
        for j in range(i + 1, len(numeric_keys)):
            p1, p2 = numeric_keys[i], numeric_keys[j]
            imp = importance.get(p1, 0) + importance.get(p2, 0)
            all_pairs.append((p1, p2, imp))
    all_pairs.sort(key=lambda x: -x[2])
    return all_pairs


def _plot_contour(study):
    if not study:
        return []
    completed = _get_filtered_trials(study)
    if len(completed) < 10:
        st.info("试验数据不足（至少 10 轮）。")
        return []
    importance = _compute_importance(st.session_state.get("_active_run", "")) or {}
    all_pairs = _compute_pair_importance(completed, importance)
    if not all_pairs:
        st.info("无可用参数对。")
        return []
    cols = st.columns(6)
    for idx, (p1, p2, imp_sum) in enumerate(all_pairs):
        if idx % 6 == 0:
            cols = st.columns(6)
        with cols[idx % 6]:
            x = [t.params[p1] for t in completed if p1 in t.params]
            y = [t.params[p2] for t in completed if p2 in t.params]
            z = [
                t.value
                for t in completed
                if p1 in t.params and p2 in t.params and t.value is not None
            ]
            if len(x) < 3 or len(z) < 3:
                continue
            fig = go.Figure(
                go.Histogram2dContour(
                    x=x,
                    y=y,
                    z=z,
                    colorscale="Viridis",
                    ncontours=15,
                    contours_coloring="heatmap",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=3, opacity=0.4, color="#333"),
                    showlegend=False,
                )
            )
            fig.update_layout(
                height=300,
                margin=dict(l=45, r=10, t=30, b=25),
                title_text=f"{p1} vs {p2}<br><sub>imp={imp_sum:.1%}</sub>",
            )
            fig.update_xaxes(title_text=p1)
            fig.update_yaxes(title_text=p2)
            st.plotly_chart(fig, use_container_width=True)
    return all_pairs


def _plot_mask_heatmap(study):
    _MASK_LETTERS = list("abcdefghijk")
    _MASK_NAMES = [
        "ATK_nearest",
        "ATK_clu_nearest",
        "ATK_nearest_weakest",
        "ATK_clu_nearest_weakest",
        "ATK_threatening",
        "DEF_clu_nearest",
        "MIX_gather",
        "MIX_lure",
        "MIX_sacrifice_lure",
        "do_randomly",
        "do_nothing",
    ]

    def _mask_label(v):
        if v < len(_MASK_LETTERS):
            return f"{_MASK_LETTERS[v]} ({_MASK_NAMES[v]})"
        return str(v)

    completed = _get_filtered_trials(study)
    masked = [
        t
        for t in completed
        if t.params.get("masked_count", 0) >= 2
        and t.params.get("mask_0") is not None
        and t.params.get("mask_1") is not None
        and t.value is not None
    ]
    if len(masked) < 5:
        st.caption("mask 组合数据不足。")
        return
    grid = {}
    for t in masked:
        m0 = t.params["mask_0"]
        m1 = t.params["mask_1"]
        key = (m0, m1)
        if key not in grid:
            grid[key] = []
        grid[key].append(t.value)
    x_labels = sorted(set(m0 for m0, _ in grid.keys()))
    y_labels = sorted(set(m1 for _, m1 in grid.keys()))
    z_data = []
    for m1 in y_labels:
        row = []
        for m0 in x_labels:
            vals = grid.get((m0, m1), [])
            row.append(round(sum(vals) / len(vals), 3) if vals else None)
        z_data.append(row)
    fig = go.Figure(
        go.Heatmap(
            z=z_data,
            x=[_mask_label(v) for v in x_labels],
            y=[_mask_label(v) for v in y_labels],
            colorscale="Viridis",
            text=[[f"{v:.3f}" if v else "" for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="mask_0=%{x}<br>mask_1=%{y}<br>obj=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=110, r=20, t=10, b=110),
        xaxis_title="mask_0",
        yaxis_title="mask_1",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "热力图: mask_0 x mask_1 的平均 objective（仅 masked_count>=2，按动作类别）"
    )


def _render_best_params(study, summary):
    best = None
    if study:
        try:
            best = study.best_trial
        except (ValueError, KeyError):
            best = None
    if best:
        st.subheader("最优参数")
        cols = st.columns(3)
        col_idx = 0
        for k, v in best.params.items():
            label = _ACTION_STRATEGY_LABELS.get(v, v) if isinstance(v, str) else v
            with cols[col_idx % 3]:
                st.metric(k, label)
            col_idx += 1
        st.divider()

        wr = best.user_attrs.get("win_rate", 0)
        sc = best.user_attrs.get("avg_score", 0)
        stab = best.user_attrs.get("stability", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("胜率", f"{wr:.1%}")
        c2.metric("平均得分", f"{sc:.1f}")
        c3.metric("稳定性 (越低越好)", f"{stab:.4f}")
    elif summary:
        st.subheader("最优参数 (摘要)")
        for k, v in summary.get("best_params", {}).items():
            label = _ACTION_STRATEGY_LABELS.get(v, v) if isinstance(v, str) else v
            st.write(f"**{k}**: {label}")


_PAGE_SIZE_TABLE = 100
_PAGE_SIZE_MONITOR = 50

_CATEGORICAL_PARAM_MAP = {
    "score_mode": {0: "quality", 1: "future_reward", 2: "win_rate"},
    "action_strategy": {
        0: "best_beam",
        1: "best_subtree_quality",
        2: "best_subtree_winrate",
        3: "highest_transition_prob",
        4: "random_beam",
        5: "epsilon_greedy",
    },
    "mode": {0: "single_step", 1: "multi_step"},
    "enable_backup": {0: False, 1: True},
}


@st.cache_data(ttl=120, show_spinner=False)
def _load_trials_from_db(_run_key: str = ""):
    study_db = _get_active_study_db()
    if not study_db or not study_db.exists():
        return []
    import sqlite3 as _sqlite

    try:
        db = _sqlite.connect(str(study_db))
        cur = db.cursor()
        cur.execute("""
            SELECT t.number, tv.value, tp.param_name, tp.param_value,
                   GROUP_CONCAT(tua.key || '||' || tua.value_json, '&&')
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            LEFT JOIN trial_user_attributes tua ON t.trial_id = tua.trial_id
            WHERE t.state = 'COMPLETE' AND tv.value IS NOT NULL
            GROUP BY t.trial_id
            ORDER BY t.number
        """)
        raw_rows = cur.fetchall()
        db.close()
    except Exception:
        return []

    rows = []
    for r in raw_rows:
        num, val, pname, pval, attrs_str = r
        attrs = {}
        if attrs_str:
            for pair in attrs_str.split("&&"):
                if "||" in pair:
                    k, v = pair.split("||", 1)
                    try:
                        attrs[k] = json.loads(v)
                    except Exception:
                        pass
        rows.append(
            {
                "trial": num,
                "params": {},
                "metrics": {
                    "win_rate": attrs.get("win_rate", 0),
                    "avg_score": attrs.get("avg_score", 0),
                    "stability": attrs.get("stability", 0),
                },
                "objective": val,
                "status": "completed",
                "batch": attrs.get("batch"),
            }
        )

    params_by_trial = {}
    if study_db and study_db.exists():
        db2 = _sqlite.connect(str(study_db))
        cur2 = db2.cursor()
        cur2.execute("""
            SELECT t.number, tp.param_name, tp.param_value
            FROM trials t
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY t.number
        """)
        for num, pname, pval in cur2.fetchall():
            params_by_trial.setdefault(num, {})[pname] = pval
        db2.close()

    for row in rows:
        row["params"] = params_by_trial.get(row["trial"], {})

    return rows


def _paginate(items, page_key, page_size):
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    st.session_state[page_key] = min(st.session_state[page_key], total_pages - 1)
    page = st.session_state[page_key]
    goto_key = f"{page_key}_goto"
    if goto_key not in st.session_state:
        st.session_state[goto_key] = page + 1
    start = page * page_size
    end = min(start + page_size, total)
    return page, total_pages, start, end


def _render_pagination(page, total_pages, page_key):
    goto_key = f"{page_key}_goto"

    def _on_prev():
        cur = st.session_state.get(page_key, 0)
        new_page = max(0, cur - 1)
        st.session_state[page_key] = new_page
        st.session_state[goto_key] = new_page + 1

    def _on_next():
        cur = st.session_state.get(page_key, 0)
        new_page = min(total_pages - 1, cur + 1)
        st.session_state[page_key] = new_page
        st.session_state[goto_key] = new_page + 1

    def _on_goto():
        st.session_state[page_key] = st.session_state[goto_key] - 1

    col_prev, col_info, col_next, col_goto = st.columns([1, 2, 1, 1.5])
    with col_prev:
        st.button(
            "◀ 上一页", key=f"{page_key}_prev", disabled=(page <= 0), on_click=_on_prev
        )
    with col_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:4px'>第 {page + 1} / {total_pages} 页</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        st.button(
            "下一页 ▶",
            key=f"{page_key}_next",
            disabled=(page >= total_pages - 1),
            on_click=_on_next,
        )
    with col_goto:
        st.number_input(
            "跳转到页",
            min_value=1,
            max_value=total_pages,
            key=goto_key,
            step=1,
            on_change=_on_goto,
            label_visibility="collapsed",
        )


def _filter_rows(rows, keyword):
    if not keyword.strip():
        return rows
    kw = keyword.strip().lower()
    return [r for r in rows if kw in str(r).lower()]


def _render_trials_table():
    study = _load_study(st.session_state.get("_active_run", ""))
    runs = _load_runs(st.session_state.get("_active_run", ""))

    all_rows = _load_trials_from_db(st.session_state.get("_active_run", ""))
    if not all_rows and runs:
        for run in runs:
            all_rows.append(
                {
                    "trial": run.get("trial", 0),
                    "params": run.get("params", {}),
                    "metrics": run.get("metrics", {}),
                    "objective": run.get("objective"),
                    "status": run.get("status", "?"),
                }
            )

    if not all_rows:
        st.info("暂无试验记录。")
        return

    all_rows.sort(key=lambda x: x.get("trial", 0), reverse=True)

    col_search, col_size = st.columns([2, 1])
    with col_search:
        keyword = st.text_input(
            "Trial 编号", key="table_search", placeholder="输入编号精确查找"
        )
    with col_size:
        _PAGE_SIZE_OPTIONS = [50, 100, 200, 500]
        ps = st.selectbox(
            "每页显示",
            _PAGE_SIZE_OPTIONS,
            index=_PAGE_SIZE_OPTIONS.index(_PAGE_SIZE_TABLE),
            key="table_page_size",
        )

    if keyword.strip().isdigit():
        target = int(keyword.strip())
        filtered = [r for r in all_rows if r.get("trial") == target]
    else:
        filtered = all_rows

    if not filtered:
        st.info("无匹配结果。")
        return

    st.caption(
        f"共 {len(filtered)} 条记录"
        + (f"（已过滤，全部 {len(all_rows)} 条）" if keyword.strip() else "")
    )

    page, total_pages, start, end = _paginate(filtered, "table_page", ps)
    _render_pagination(page, total_pages, "table_page")

    display_rows = []
    for r in filtered[start:end]:
        p = r.get("params", {})
        m = r.get("metrics", {})
        display_rows.append(
            {
                "_trial_num": r.get("trial", 0),
                "批次": r.get("batch", "-") or "-",
                "Trial": f"#{r.get('trial', '?')}",
                "score_mode": p.get("score_mode", ""),
                "beam_width": p.get("beam_width", ""),
                "lookahead": p.get("lookahead_steps", ""),
                "action_strategy": _ACTION_STRATEGY_LABELS.get(
                    p.get("action_strategy", ""), p.get("action_strategy", "")
                ),
                "mode": p.get("mode", ""),
                "min_visits": p.get("min_visits", ""),
                "max_revisits": p.get("max_state_revisits", ""),
                "min_cum_prob": f"{p.get('min_cum_prob', ''):.4f}"
                if isinstance(p.get("min_cum_prob"), float)
                else p.get("min_cum_prob", ""),
                "discount": f"{p.get('discount_factor', ''):.2f}"
                if isinstance(p.get("discount_factor"), float)
                else p.get("discount_factor", ""),
                "backup": "Yes" if p.get("enable_backup") else "No",
                "胜率": f"{m.get('win_rate', 0):.1%}"
                if m.get("win_rate") is not None
                else "-",
                "平均得分": f"{m.get('avg_score', 0):.1f}"
                if m.get("avg_score") is not None
                else "-",
                "稳定性": f"{m.get('stability', 0):.4f}"
                if m.get("stability") is not None
                else "-",
                "目标值": f"{r.get('objective', 0):.4f}"
                if r.get("objective") is not None
                else "-",
            }
        )

    df = pd.DataFrame(display_rows)
    df.insert(0, "选择", False)
    disabled_cols = [c for c in df.columns if c != "选择"]

    edited = st.data_editor(
        df,
        use_container_width=True,
        height=min(len(display_rows) * 35 + 50, 600),
        disabled=disabled_cols,
        hide_index=True,
        key="table_editor",
    )

    selected = edited[edited["选择"] == True]
    selected_trials = []
    if len(selected) > 0:
        for _, row in selected.iterrows():
            trial_str = str(row.get("Trial", ""))
            num_str = trial_str.replace("#", "")
            if num_str.isdigit():
                selected_trials.append(int(num_str))

    if selected_trials:
        st.caption(
            f"已选中 {len(selected_trials)} 条: {', '.join(f'#{t}' for t in selected_trials)}"
        )
        col_rerun, col_del, col_spacer = st.columns([1, 1, 3])
        with col_rerun:
            if st.button("重跑选中", key="table_rerun_selected", type="primary"):
                _start_rerun(selected_trials)
        with col_del:
            if st.button("删除选中", key="table_del_selected"):
                for tn in selected_trials:
                    _delete_trial(tn)
                st.toast(f"已删除 {len(selected_trials)} 条", icon="🗑️")
                _clear_all_cache()
                st.rerun()


@st.cache_data(ttl=30, show_spinner=False)
def _load_runs(_run_key: str = ""):
    runs = []
    runs_dir = _get_active_runs_dir()
    if not runs_dir or not runs_dir.exists():
        return runs
    for fp in sorted(runs_dir.glob("trial_*_run.json")):
        try:
            with open(str(fp), "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            continue
    return runs


def _check_port_alive(port):
    try:
        r = _requests.get(f"http://127.0.0.1:{port}/game/status", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False


def _read_trial_progress(trial_num, target_episodes):
    trials_dir = _get_active_trials_dir()
    if not trials_dir:
        return 0
    progress_file = trials_dir / f"trial_{trial_num:04d}" / "progress.json"
    if not progress_file.exists():
        return 0
    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
        return data.get("completed", 0)
    except (json.JSONDecodeError, OSError):
        return 0


def _render_run_monitor():
    runs = _load_runs(st.session_state.get("_active_run", ""))

    if not runs:
        st.info("暂无运行记录。启动参数寻优后将自动记录。")
        return

    runs.reverse()

    col_search, col_size = st.columns([2, 1])
    with col_search:
        keyword = st.text_input(
            "Trial 编号", key="monitor_search", placeholder="输入编号精确查找"
        )
    with col_size:
        _PAGE_SIZE_OPTIONS = [50, 100, 200, 500]
        ps = st.selectbox(
            "每页显示",
            _PAGE_SIZE_OPTIONS,
            index=_PAGE_SIZE_OPTIONS.index(_PAGE_SIZE_MONITOR),
            key="monitor_page_size",
        )

    if keyword.strip().isdigit():
        target = int(keyword.strip())
        filtered = [r for r in runs if r.get("trial") == target]
    else:
        filtered = runs

    if not filtered:
        st.info("无匹配结果。")
        return

    st.caption(
        f"共 {len(filtered)} 条记录"
        + (f"（已过滤，全部 {len(runs)} 条）" if keyword.strip() else "")
    )

    page, total_pages, start, end = _paginate(filtered, "monitor_page", ps)
    _render_pagination(page, total_pages, "monitor_page")

    header_cols = st.columns([0.5, 1, 1, 1.5, 2.5, 1.5, 1, 1, 0.6, 0.6])
    header_labels = [
        "Run",
        "Trial",
        "端口",
        "时间",
        "参数",
        "状态",
        "指标",
        "备注",
        "重跑",
        "删除",
    ]
    for col, label in zip(header_cols, header_labels):
        col.caption(f"**{label}**")

    for run in filtered[start:end]:
        trial_num = run.get("trial", "?")
        p = run.get("params", {})
        status = run.get("status", "unknown")
        port = run.get("port", 0)
        target_episodes = run.get("target_episodes", 0)

        if status == "running":
            if isinstance(trial_num, int):
                done = _read_trial_progress(trial_num, target_episodes)
                if done > 0:
                    status_display = "🟢 运行中"
                    progress_info = f" {done}/{target_episodes}"
                else:
                    alive = _check_port_alive(port)
                    status_display = "🟢 运行中" if alive else "🔴 已停止"
                    progress_info = ""
            else:
                status_display = "🟢 运行中"
                progress_info = ""
        else:
            progress_info = ""
            if status == "timeout":
                status_display = "🟡 超时"
            elif status == "completed":
                status_display = "✅ 已完成"
            else:
                status_display = "❓ 未知"

        param_summary = " ".join(
            f"{k}={v}"
            for k, v in p.items()
            if k in ("beam_width", "score_mode", "action_strategy", "lookahead_steps")
        )

        metrics = run.get("metrics", {})
        obj_val = run.get("objective", None)
        if metrics:
            metric_str = f"胜率 {metrics.get('win_rate', 0):.0%}"
            if obj_val is not None:
                metric_str += f" | {obj_val:.4f}"
        else:
            metric_str = "-"

        cols = st.columns([0.5, 1, 1, 1.5, 2.5, 1.5, 1, 1, 0.6, 0.6])
        cols[0].caption(st.session_state.get("_active_run", "-"))
        cols[1].caption(f"#{trial_num}")
        cols[2].caption(str(port))
        cols[3].caption(run.get("start_time", ""))
        cols[4].caption(param_summary)
        if status == "running" and progress_info:
            parts = progress_info.strip().split("/")
            done_val = int(parts[0]) if parts[0].isdigit() else 0
            pct = done_val / target_episodes if target_episodes > 0 else 0
            cols[5].caption(status_display)
            cols[5].progress(min(pct, 1.0))
            cols[5].caption(progress_info)
        else:
            cols[5].caption(status_display)
        cols[6].caption(metric_str)
        source = run.get("source_trial")
        remark = ""
        if source:
            remark = f"重跑自 #{source}"
        if p.get("exploration_targets"):
            remark = ("微调训练 " + remark).strip()
        cols[7].caption(remark)
        rerun_key = f"_rerun_trial_{trial_num}_{page}"
        if cols[8].button("▶", key=rerun_key):
            if isinstance(trial_num, int):
                _start_rerun([trial_num])
        del_key = f"_del_trial_{trial_num}_{page}"
        if cols[9].button("🗑", key=del_key):
            if isinstance(trial_num, int) and status == "running":
                _kill_port_process(port)
            if isinstance(trial_num, int):
                _delete_trial(trial_num)
            st.toast(f"Trial #{trial_num} 已删除", icon="🗑️")
            st.rerun()

    st.divider()

    learner_log_path = None
    trials_dir = _get_active_trials_dir()
    if trials_dir:
        learner_log_path = trials_dir / "learner.log"
    if learner_log_path and learner_log_path.exists():
        if st.button("查看 learner.log", key="monitor_learner_log"):
            active_run = st.session_state.get("_active_run", "")
            if active_run == "run_0001":
                _show_log("trials/learner.log")
            else:
                run_path = _get_active_run_path()
                _show_log("trials/learner.log", base_dir=run_path)

    ep_options = []
    if trials_dir:
        for run in runs:
            tn = run.get("trial")
            if isinstance(tn, int):
                ep_file = trials_dir / f"trial_{tn:04d}" / "episodes.jsonl"
                if ep_file.exists():
                    ep_options.append(tn)
    if ep_options:
        selected_tn = st.selectbox(
            "查看 Trial Episodes",
            ep_options,
            key="monitor_ep_select",
            format_func=lambda x: f"Trial #{x:04d}",
        )
        if selected_tn is not None and trials_dir:
            ep_path = trials_dir / f"trial_{selected_tn:04d}" / "episodes.jsonl"
            try:
                content = ep_path.read_text(encoding="utf-8", errors="replace")
                lines = content.strip().split("\n")
                show_lines = "\n".join(lines[-50:]) if len(lines) > 50 else content
                st.text_area(
                    f"episodes.jsonl (最近 50 行)",
                    show_lines,
                    height=300,
                    label_visibility="collapsed",
                )
            except Exception as e:
                st.error(f"读取失败: {e}")
    if not (learner_log_path and learner_log_path.exists()) and not ep_options:
        st.caption("暂无日志文件。")


def _load_action_tuning_model():
    run_path = _get_active_run_path()
    if run_path is None:
        return None, None
    model_path = run_path / "action_tuning_model.pkl"
    if not model_path.exists():
        return None, model_path
    try:
        from src.decision.action_tuning_model import ActionTuningModel

        return ActionTuningModel.load(str(model_path)), model_path
    except Exception as e:
        st.error(f"动作微调模型加载失败: {e}")
        return None, model_path


def _collect_action_source_stats():
    trials_dir = _get_active_trials_dir()
    if trials_dir is None or not trials_dir.exists():
        return pd.DataFrame()
    rows = []
    for ep_file in sorted(trials_dir.glob("trial_*/episodes.jsonl")):
        try:
            trial_number = int(ep_file.parent.name.split("_")[-1])
        except Exception:
            trial_number = None
        counters = {}
        nid_counters = {}
        guard_skip = 0
        guard_disable = 0
        tuning_decision = 0
        tuning_opportunity = 0
        tuning_accept = 0
        tuning_accept_without_opportunity = 0
        tuning_candidate_eligible = 0
        tuning_validation_opportunity = 0
        tuning_validation_accept = 0
        episodes = 0
        scores = []
        try:
            with open(str(ep_file), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    episodes += 1
                    scores.append(float(rec.get("score", 0.0)))
                    guard = rec.get("restart_guard") or {}
                    if guard.get("skip_update"):
                        guard_skip += 1
                    if guard.get("disable_ood_explore"):
                        guard_disable += 1
                    for fr in rec.get("frames", []):
                        src = fr.get("action_source", "unknown")
                        counters[src] = counters.get(src, 0) + 1
                        status = fr.get("nid_status") or "unknown"
                        nid_counters[status] = nid_counters.get(status, 0) + 1
                        plan = fr.get("plan") or {}
                        tuning_info = plan.get("action_tuning") or fr.get("action_tuning") or {}
                        if tuning_info:
                            tuning_decision += 1
                            has_opportunity = bool(tuning_info.get("opportunity"))
                            if tuning_info.get("opportunity"):
                                tuning_opportunity += 1
                            if tuning_info.get("candidate_eligible"):
                                tuning_candidate_eligible += 1
                            if tuning_info.get("validation"):
                                tuning_validation_opportunity += 1
                            if src in ("tuning", "ood_tuning"):
                                tuning_accept += 1
                                if not has_opportunity:
                                    tuning_accept_without_opportunity += 1
                                if tuning_info.get("validation"):
                                    tuning_validation_accept += 1
        except Exception:
            continue
        total = sum(counters.values())
        if total <= 0:
            continue
        row = {
            "trial": trial_number,
            "episodes": episodes,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "total_actions": total,
            "guard_skip_update": guard_skip,
            "guard_disable_ood": guard_disable,
            "guard_skip_update_ratio": guard_skip / max(episodes, 1),
            "guard_disable_ood_ratio": guard_disable / max(episodes, 1),
            "tuning_decision": tuning_decision,
            "tuning_opportunity": tuning_opportunity,
            "tuning_accept": tuning_accept,
            "tuning_accept_without_opportunity": tuning_accept_without_opportunity,
            "tuning_candidate_eligible": tuning_candidate_eligible,
            "tuning_validation_opportunity": tuning_validation_opportunity,
            "tuning_validation_accept": tuning_validation_accept,
            "tuning_opportunity_ratio": tuning_opportunity / total,
            "tuning_accept_ratio": tuning_accept / total,
            "tuning_accept_per_opportunity": tuning_accept
            / max(tuning_opportunity + tuning_accept_without_opportunity, 1),
            "tuning_accept_per_decision": tuning_accept / max(tuning_decision, 1),
            "tuning_candidate_eligible_ratio": tuning_candidate_eligible / total,
            "tuning_validation_opportunity_ratio": tuning_validation_opportunity / total,
            "tuning_validation_accept_ratio": tuning_validation_accept / total,
            "tuning_validation_accept_per_opportunity": tuning_validation_accept
            / max(tuning_validation_opportunity, 1),
        }
        for key in (
            "etg_plan",
            "etg_follow",
            "tuning",
            "mc_explore",
            "ood",
            "ood_mc_explore",
            "ood_tuning",
            "fallback",
            "ft_plan",
            "etg_relaxed",
            "fuzzy_plan",
        ):
            row[key] = counters.get(key, 0)
            row[f"{key}_ratio"] = counters.get(key, 0) / total
        for key in ("exact", "near_valid", "near_rejected", "missing"):
            row[f"nid_{key}"] = nid_counters.get(key, 0)
            row[f"nid_{key}_ratio"] = nid_counters.get(key, 0) / total
        row["nid_ood_ratio"] = (
            nid_counters.get("near_rejected", 0) + nid_counters.get("missing", 0)
        ) / total
        rows.append(row)
    return pd.DataFrame(rows)


def _load_trial_objectives() -> Dict[int, float]:
    study_db = _get_active_study_db()
    if study_db is None or not study_db.exists():
        return {}
    import sqlite3 as _sqlite

    try:
        db = _sqlite.connect(str(study_db))
        cur = db.cursor()
        cur.execute("""
            SELECT t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE tv.value IS NOT NULL
        """)
        rows = cur.fetchall()
        db.close()
        return {int(num): float(value) for num, value in rows}
    except Exception:
        return {}


def _collect_high_score_ood_candidates(top_quantile: float = 0.25, max_rows: int = 100):
    trials_dir = _get_active_trials_dir()
    if trials_dir is None or not trials_dir.exists():
        return pd.DataFrame()
    objectives = _load_trial_objectives()
    if not objectives:
        return pd.DataFrame()
    values = [v for v in objectives.values() if v is not None]
    if not values:
        return pd.DataFrame()
    threshold = float(np.quantile(values, max(0.0, min(1.0, 1.0 - top_quantile))))

    stats = {}
    action_counts = defaultdict(Counter)
    source_counts = defaultdict(Counter)
    trial_sets = defaultdict(set)
    distance_values = defaultdict(list)

    for ep_file in sorted(trials_dir.glob("trial_*/episodes.jsonl")):
        try:
            trial_number = int(ep_file.parent.name.split("_")[-1])
        except Exception:
            continue
        objective = objectives.get(trial_number)
        if objective is None or objective < threshold:
            continue
        try:
            with open(str(ep_file), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    episode_score = float(rec.get("score", 0.0))
                    for fr in rec.get("frames", []):
                        status = fr.get("nid_status")
                        if status not in ("near_rejected", "missing") and not fr.get("nid_is_ood"):
                            continue
                        state_key = fr.get("state_key") or fr.get("nid") or "unknown"
                        state_key = str(state_key)
                        row = stats.setdefault(
                            state_key,
                            {
                                "state_key": state_key,
                                "visits": 0,
                                "episodes": 0,
                                "score_sum": 0.0,
                                "objective_sum": 0.0,
                                "candidate_nid": fr.get("nid_candidate"),
                                "nid_status": status or "unknown",
                            },
                        )
                        row["visits"] += 1
                        row["score_sum"] += episode_score
                        row["objective_sum"] += objective
                        trial_sets[state_key].add(trial_number)
                        action_code = fr.get("action_code", "unknown")
                        if not _is_valid_action_code(action_code):
                            action_code = "invalid"
                        action_counts[state_key][action_code] += 1
                        source_counts[state_key][fr.get("action_source", "unknown")] += 1
                        if fr.get("nid_distance") is not None:
                            try:
                                distance_values[state_key].append(float(fr.get("nid_distance")))
                            except Exception:
                                pass
        except Exception:
            continue

    rows = []
    for state_key, row in stats.items():
        visits = max(int(row["visits"]), 1)
        valid_action_counts = Counter(
            {k: v for k, v in action_counts[state_key].items() if _is_valid_action_code(k)}
        )
        if valid_action_counts:
            top_action, top_action_count = valid_action_counts.most_common(1)[0]
        else:
            top_action, top_action_count = "invalid", action_counts[state_key].get("invalid", 0)
        top_source, top_source_count = source_counts[state_key].most_common(1)[0]
        distances = distance_values.get(state_key, [])
        rows.append(
            {
                "state_key": state_key,
                "visits": visits,
                "trial_count": len(trial_sets[state_key]),
                "avg_score": row["score_sum"] / visits,
                "avg_objective": row["objective_sum"] / visits,
                "top_action": top_action,
                "top_action_ratio": top_action_count / visits,
                "invalid_action_count": action_counts[state_key].get("invalid", 0),
                "top_source": top_source,
                "top_source_ratio": top_source_count / visits,
                "avg_nid_distance": float(np.mean(distances)) if distances else None,
                "candidate_nid": row.get("candidate_nid"),
                "nid_status": row.get("nid_status"),
                "phase_hint": "promote_candidate" if visits >= 10 and len(trial_sets[state_key]) >= 2 else "observe",
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["avg_score", "visits", "trial_count"], ascending=[False, False, False]
    ).head(max_rows)


def _render_action_tuning_tab():
    st.markdown("### 蒙特卡洛动作微调探索效果")
    model, model_path = _load_action_tuning_model()
    if model_path is not None:
        st.caption(f"模型路径: `{model_path}`")
    if model is None:
        st.info("当前 Run 暂无 `action_tuning_model.pkl`。启用动作微调探索并完成至少一个 episode 后生成。")
    else:
        summary = model.get_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("状态数", summary.get("total_states", 0))
        c2.metric("状态-动作对", summary.get("total_pairs", 0))
        c3.metric("总访问", summary.get("total_visits", 0))
        c4.metric("训练局数", summary.get("trained_episodes", 0))

        rows = []
        for sid, actions in model.state_action_stats.items():
            state_info = _format_state_ref(sid)
            for action, stats in actions.items():
                rows.append(
                    {
                        **state_info,
                        "action": action,
                        "visits": stats.visits,
                        "wins": stats.wins,
                        "mean_return": stats.mean_return,
                        "std_return": stats.std_return,
                        "confidence": stats.confidence,
                        "ucb_score": model.ucb_score(sid, action)
                        if stats.visits > 0
                        else None,
                    }
                )
        if rows:
            df = pd.DataFrame(rows)
            df["state_ref"] = df["state_ref"].astype(str)
            df["state_kind"] = df["state_kind"].astype(str)
            df["base_nid"] = pd.to_numeric(df["base_nid"], errors="coerce").astype("Int64")
            df["state_cluster"] = df["state_cluster"].astype("string")
            df["ood_distance"] = pd.to_numeric(df["ood_distance"], errors="coerce")
            st.markdown("**状态-动作价值表**")
            st.caption(
                "`state_kind=etg` 表示原始 etg/BKTree nid；`state_kind=ood` 表示当前状态未能可靠映射到原始 nid，"
                "`state_ref` 使用 `ood:<candidate_nid>:<cluster>:<hash>:d<distance>` 作为稳定临时键。"
            )
            sort_col = st.selectbox(
                "排序字段",
                ["confidence", "mean_return", "visits", "ucb_score", "std_return"],
                key="action_tuning_sort_col",
            )
            top_n = st.slider("显示 Top-N", 10, 500, 100, step=10, key="action_tuning_top_n")
            st.dataframe(
                df.sort_values(sort_col, ascending=False).head(top_n),
                use_container_width=True,
                hide_index=True,
            )

            high_conf = df[(df["visits"] >= 3) & (df["confidence"] >= 0.35)]
            if not high_conf.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=high_conf["visits"],
                        y=high_conf["mean_return"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=high_conf["confidence"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="confidence"),
                        ),
                        text=[
                            f"state={r.state_ref}, action={r.action}, conf={r.confidence:.3f}"
                            for r in high_conf.itertuples()
                        ],
                    )
                )
                fig.update_layout(
                    title="高置信动作：访问次数 vs 平均回报",
                    xaxis_title="visits",
                    yaxis_title="mean_return",
                    height=360,
                )
                st.plotly_chart(fig, use_container_width=True)

    source_df = _collect_action_source_stats()
    if source_df.empty:
        st.info("暂无 episode action_source 统计。")
        return

    st.markdown("**动作来源占比趋势**")
    fig = go.Figure()
    for col, label in (
        ("etg_plan_ratio", "etg_plan"),
        ("etg_follow_ratio", "etg_follow"),
        ("tuning_ratio", "tuning"),
        ("mc_explore_ratio", "mc_explore"),
        ("ood_ratio", "ood"),
        ("ood_tuning_ratio", "ood_tuning"),
        ("ood_mc_explore_ratio", "ood_mc_explore"),
        ("fallback_ratio", "fallback"),
    ):
        if col in source_df:
            fig.add_trace(
                go.Scatter(
                    x=source_df["trial"],
                    y=source_df[col],
                    mode="lines+markers",
                    name=label,
                )
            )
    fig.update_layout(
        xaxis_title="trial",
        yaxis_title="ratio",
        yaxis_tickformat=".0%",
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    if "guard_skip_update_ratio" in source_df or "guard_disable_ood_ratio" in source_df:
        st.markdown("**RestartGuard 触发趋势**")
        guard_fig = go.Figure()
        for col, label in (
            ("guard_skip_update_ratio", "skip model update"),
            ("guard_disable_ood_ratio", "disable OOD explore"),
        ):
            if col in source_df:
                guard_fig.add_trace(
                    go.Scatter(
                        x=source_df["trial"],
                        y=source_df[col],
                        mode="lines+markers",
                        name=label,
                    )
                )
        guard_fig.update_layout(
            xaxis_title="trial",
            yaxis_title="episode ratio",
            yaxis_tickformat=".0%",
            height=280,
        )
        st.plotly_chart(guard_fig, use_container_width=True)

    st.markdown("**微调使用率与得分关系**")
    if "nid_ood_ratio" in source_df:
        st.markdown("**NID 解析质量趋势**")
        nid_fig = go.Figure()
        for col, label in (
            ("nid_exact_ratio", "exact"),
            ("nid_near_valid_ratio", "near_valid"),
            ("nid_near_rejected_ratio", "near_rejected"),
            ("nid_missing_ratio", "missing"),
            ("nid_ood_ratio", "OOD total"),
        ):
            if col in source_df:
                nid_fig.add_trace(
                    go.Scatter(
                        x=source_df["trial"],
                        y=source_df[col],
                        mode="lines+markers",
                        name=label,
                    )
                )
        nid_fig.update_layout(
            xaxis_title="trial",
            yaxis_title="ratio",
            yaxis_tickformat=".0%",
            height=320,
        )
        st.plotly_chart(nid_fig, use_container_width=True)

    usage = (
        source_df.get("tuning_ratio", 0)
        + source_df.get("mc_explore_ratio", 0)
        + source_df.get("ood_ratio", 0)
        + source_df.get("ood_tuning_ratio", 0)
        + source_df.get("ood_mc_explore_ratio", 0)
    )
    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            x=usage,
            y=source_df["avg_score"],
            mode="markers+text",
            text=source_df["trial"].astype(str),
            textposition="top center",
        )
    )
    scatter.update_layout(
        xaxis_title="tuning + mc_explore ratio",
        yaxis_title="avg_score",
        height=340,
    )
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown("**高分 OOD 状态候选表**")
    col_ood1, col_ood2 = st.columns([1, 1])
    with col_ood1:
        top_quantile = st.slider(
            "高分 trial 范围",
            min_value=0.05,
            max_value=0.50,
            value=0.25,
            step=0.05,
            format="Top %.2f",
            key="high_score_ood_quantile",
            help="只统计 objective 位于该 Top 比例内的 trial，用于寻找高收益但 BKTree 未覆盖的 OOD 状态。",
        )
    with col_ood2:
        max_candidates = st.slider(
            "候选数量",
            min_value=20,
            max_value=300,
            value=100,
            step=20,
            key="high_score_ood_max_rows",
        )
    ood_candidates = _collect_high_score_ood_candidates(
        top_quantile=float(top_quantile), max_rows=int(max_candidates)
    )
    if ood_candidates.empty:
        st.info("暂无高分 OOD 候选。需要已完成 trial，且高分 trial 中存在 near_rejected/missing 状态。")
    else:
        st.caption(
            "用于识别“高分但未被 BKTree/etg 覆盖”的状态；phase_hint=promote_candidate 表示可优先考虑进入增量层候选。"
        )
        st.dataframe(
            ood_candidates,
            use_container_width=True,
            hide_index=True,
        )


def _render_learner_tab():
    st.markdown("### 在线协同训练：Beam Search 参数寻优 + 微调模型进化")

    if "_active_run" not in st.session_state:
        all_runs = _get_all_runs()
        st.session_state["_active_run"] = all_runs[0][0] if all_runs else None

    _render_run_selector()

    summary = _load_summary()
    study = _load_study(st.session_state.get("_active_run", ""))
    running_trial = _get_running_trial(study)

    if study:
        active_trials = [
            t
            for t in study.trials
            if not (
                t.state == optuna.trial.TrialState.RUNNING and not _is_learner_alive()
            )
        ]
        total = len(active_trials)
        completed = sum(
            1 for t in active_trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        running = sum(
            1 for t in active_trials if t.state == optuna.trial.TrialState.RUNNING
        )
        failed = sum(
            1 for t in active_trials if t.state == optuna.trial.TrialState.FAIL
        )

        if running:
            st.markdown(
                f"**{total} 轮试验 | {completed} 完成 |** :orange[{running} 运行中] **| {failed} 失败**"
            )
        else:
            st.markdown(f"**{total} 轮试验 | {completed} 完成 | {failed} 失败**")
    elif summary:
        st.markdown(
            f"**{summary.get('total_trials', 0)} 轮试验 | {summary.get('completed_trials', 0)} 完成**"
        )
    else:
        st.info("暂无优化数据。切换到「训练记录与启动」视图启动在线协同训练。")

    if study and len(study.trials) > 0:
        run_info = _get_run_info()

        with st.expander("Run 管理", expanded=False):
            if run_info:
                selected_runs = []
                for run_name in sorted(run_info.keys()):
                    info = run_info[run_name]
                    checked = st.checkbox(
                        f"{run_name}: {info['count']} trials "
                        f"(#{info['min_trial']} - #{info['max_trial']})",
                        key=f"_run_sel_{run_name}",
                    )
                    if checked:
                        selected_runs.append(run_name)

                if selected_runs:
                    st.caption(f"已选中 {len(selected_runs)} 个 run")
                    confirmed = st.checkbox(
                        f"确认删除选中的 {len(selected_runs)} 个 run",
                        key="_run_delete_confirm",
                    )
                    if st.button(
                        f"删除选中的 {len(selected_runs)} 个 run",
                        disabled=not confirmed,
                        type="primary" if confirmed else "secondary",
                        key="_run_delete_btn",
                    ):
                        for rn in selected_runs:
                            _delete_run(rn)
                        st.toast(
                            f"已删除 {len(selected_runs)} 个 run",
                            icon="🗑️",
                        )
                        st.rerun()
            else:
                st.info("暂无 run 信息。")

        col_reset, col_unlock, col_grefresh, col_export = st.columns([3, 2, 1, 1])
        with col_reset:
            active_run = st.session_state.get("_active_run")
            if st.button("重置当前 Run 数据", key="learner_reset_db"):
                study_db = _get_active_study_db()
                if study_db and study_db.exists():
                    try:
                        optuna.delete_study(
                            study_name="beam_search", storage=f"sqlite:///{study_db}"
                        )
                    except Exception:
                        pass
                    study_db.unlink(missing_ok=True)
                runs_dir = _get_active_runs_dir()
                if runs_dir:
                    for f in runs_dir.glob("trial_*_run.json"):
                        f.unlink(missing_ok=True)
                trials_dir = _get_active_trials_dir()
                if trials_dir:
                    for d in trials_dir.glob("trial_*"):
                        if d.is_dir():
                            shutil.rmtree(d, ignore_errors=True)
                run_path = _get_active_run_path()
                if run_path:
                    sp = run_path / "study_summary.json"
                    sp.unlink(missing_ok=True)
                st.toast("数据已重置", icon="🗑️")
                _clear_all_cache()
                st.rerun()
        with col_unlock:
            if st.button("清除进程锁定", key="learner_unlock"):
                if _PID_FILE.exists():
                    _PID_FILE.unlink()
                try:
                    _is_learner_alive.clear()
                except Exception:
                    pass
                for k in ("learner_proc", "finetune_proc"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.toast("进程锁定已清除", icon="🔓")
                st.rerun()
        with col_grefresh:
            if st.button("刷新状态", key="learner_refresh_status"):
                _clear_all_cache()
                st.rerun()
        with col_export:
            _render_export_button()

    _render_best_params(study, summary)

    _VIEW_OPTIONS = [
        "优化曲线",
        "参数分析",
        "参数关系",
        "试验记录",
        "运行状态监测",
        "动作微调效果",
        "训练记录与启动",
    ]
    _VIEW_KEY = "_learner_active_view"
    active_view = st.radio(
        "选择视图",
        _VIEW_OPTIONS,
        horizontal=True,
        key=_VIEW_KEY,
        label_visibility="collapsed",
    )

    if active_view == "优化曲线":
        _plot_objective_history(study)

    elif active_view == "参数分析":
        filtered = _get_filtered_trials(study)
        _render_conclusion_panel(filtered)
        col_left, col_right = st.columns([3, 2])
        with col_left:
            left_l, left_r = st.columns(2)
            with left_l:
                _plot_importance(study)
            with left_r:
                _plot_numeric_correlation(study)
        with col_right:
            cat_row1_a, cat_row1_b = st.columns(2)
            with cat_row1_a:
                _plot_categorical_effect(study, "mode")
            with cat_row1_b:
                _plot_categorical_effect(study, "score_mode")
            cat_row2_a, cat_row2_b = st.columns(2)
            with cat_row2_a:
                _plot_categorical_effect(study, "action_strategy")
            with cat_row2_b:
                _plot_categorical_effect(study, "enable_backup")
            cat_row3_a, cat_row3_b = st.columns(2)
            with cat_row3_a:
                _plot_categorical_effect(study, "masked_count")
            with cat_row3_b:
                _plot_mask_heatmap(study)

    elif active_view == "参数关系":
        _plot_parallel_coordinates(study)
        st.divider()
        _plot_slice(study)
        with st.expander("参数等高线图", expanded=False):
            importance = (
                _compute_importance(st.session_state.get("_active_run", "")) or {}
            )
            if importance:
                top_params = sorted(importance.items(), key=lambda x: -x[1])[:6]
                imp_txt = ", ".join(f"{k}={v:.1%}" for k, v in top_params)
                st.caption(f"选择依据（参数重要性，组合重要性=两参数之和）: {imp_txt}")
                import pandas as _pd

                single_rows = [
                    {"参数": k, "重要性": f"{v:.1%}"}
                    for k, v in sorted(importance.items(), key=lambda x: -x[1])
                ]
                st.markdown("**单参数重要性**")
                st.dataframe(
                    _pd.DataFrame(single_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            all_pairs = _plot_contour(study)
            if all_pairs:
                import pandas as _pd

                pair_rows = [
                    {
                        "排名": i + 1,
                        "参数对": f"{p1} vs {p2}",
                        "组合重要性": f"{imp:.1%}",
                    }
                    for i, (p1, p2, imp) in enumerate(all_pairs)
                ]
                st.markdown(
                    f"**全部参数对组合重要性（共 {len(all_pairs)} 对，全部已绘制）**"
                )
                st.dataframe(
                    _pd.DataFrame(pair_rows),
                    use_container_width=True,
                    hide_index=True,
                )

    elif active_view == "试验记录":
        _render_trials_table()

    elif active_view == "运行状态监测":
        _render_run_monitor()

    elif active_view == "动作微调效果":
        _render_action_tuning_tab()

    elif active_view == "训练记录与启动":
        _render_finetune_tab()


_FINETUNE_DIR = _RESULTS_DIR / "finetune_runs"
_FINETUNE_SAMPLES_DIR = _RESULTS_DIR / "finetune_samples"


@st.cache_data(ttl=60, show_spinner=False)
def _load_finetune_runs():
    runs = []
    if not _FINETUNE_DIR.exists():
        return runs
    for fp in sorted(_FINETUNE_DIR.glob("finetune_*_run.json")):
        try:
            with open(str(fp), "r", encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception:
            continue
    return runs


def _delete_finetune_run(finetune_id: int):
    run_json = _FINETUNE_DIR / f"finetune_{finetune_id:04d}_run.json"
    run_json.unlink(missing_ok=True)

    sample_dir = _FINETUNE_SAMPLES_DIR / f"finetune_{finetune_id:04d}"
    if sample_dir.is_dir():
        shutil.rmtree(sample_dir, ignore_errors=True)

    try:
        _load_finetune_runs.clear()
    except Exception:
        pass

    for p in _RESULTS_DIR.glob("finetune_model_group_*.pkl"):
        p.unlink(missing_ok=True)
    old_model = _RESULTS_DIR / "finetune_model.pkl"
    if old_model.exists():
        old_model.unlink()
    shared_model = _RESULTS_DIR / "shared_finetune_model.pkl"
    shared_backup = _RESULTS_DIR / "shared_finetune_model.pkl.backup"
    shared_model.unlink(missing_ok=True)
    shared_backup.unlink(missing_ok=True)


def _render_finetune_model_overview():
    shared_model = _RESULTS_DIR / "shared_finetune_model.pkl"
    group_models = sorted(_RESULTS_DIR.glob("finetune_model_group_*.pkl"))
    old_model = _RESULTS_DIR / "finetune_model.pkl"
    if old_model.exists():
        group_models.append(old_model)

    all_models = []
    if shared_model.exists():
        all_models.append(shared_model)
    all_models.extend(group_models)

    if not all_models:
        st.info("尚未训练模型。启动训练后将在此显示模型概览。")
        return

    try:
        import pickle
        from src.decision.finetune_model import FinetuneModel
    except ImportError:
        st.error("无法导入 FinetuneModel")
        return

    for model_path in all_models:
        try:
            with open(str(model_path), "rb") as f:
                model = pickle.load(f)
            if not isinstance(model, FinetuneModel):
                continue

            stats = model.get_overall_stats()
            name = model_path.stem
            with st.expander(f"模型 {name}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "已探索状态", f"{stats['explored_states']}/{stats['total_states']}"
                )
                col2.metric("平均置信度", f"{stats['explored_ratio']:.1%}")
                col3.metric("平均 RS", f"{stats['avg_replacement_score']:.3f}")
                col4.metric("训练 Episodes", stats["trained_episodes"])
                st.caption(f"最后训练: {stats.get('last_trained', 'N/A')}")

                all_scores = []
                for sid in model.q_table:
                    for ac in model.q_table[sid]:
                        all_scores.append(model.replacement_score(sid, ac))
                if all_scores:
                    import plotly.graph_objects as go

                    fig = go.Figure(
                        data=[
                            go.Histogram(
                                x=all_scores, nbinsx=20, marker_color="steelblue"
                            )
                        ]
                    )
                    fig.update_layout(
                        xaxis_title="Replacement Score",
                        yaxis_title="State-Action Pairs",
                        height=300,
                        margin=dict(l=40, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"加载 {model_path.name} 失败: {e}")


def _render_finetune_tab():
    st.subheader("状态微调机")

    col_overview, col_config = st.columns([3, 2])
    with col_overview:
        _render_finetune_model_overview()

    with col_config:
        st.markdown("**训练配置**")
        top_k = st.number_input("Top-K 参数组合", 1, 20, 5, key="ft_top_k")
        ep_per_cfg = st.number_input("每组 Episodes", 10, 200, 50, key="ft_ep_per_cfg")
        reward_mode = st.selectbox(
            "Reward 模式",
            ["hp_episodic", "hp", "etg_correct", "etg_offline"],
            index=0,
            key="ft_reward_mode",
            help="hp=纯hp差值; hp_episodic=hp+胜负回溯; etg_correct=ETG修正即时reward; etg_offline=定期ETG校正Q表",
        )
        sigma = st.slider("k-NN sigma", 0.1, 2.0, 0.5, step=0.1, key="ft_sigma")
        target_visits = st.number_input(
            "目标探索次数", 3, 100, 10, key="ft_target_visits"
        )

    st.divider()

    runs = _load_finetune_runs()

    if runs:
        st.markdown("**训练记录**")

        header_cols = st.columns([0.7, 2, 0.8, 0.8, 1, 1.2, 1.2, 0.8, 0.5])
        header_labels = [
            "ID",
            "Top-K Trials",
            "Ep/Cfg",
            "预算",
            "状态",
            "时间",
            "模型指标",
            "详情",
            "删除",
        ]
        for col, label in zip(header_cols, header_labels):
            col.caption(f"**{label}**")

        for run in reversed(runs):
            fid = run.get("finetune_id", "?")
            status = run.get("status", "unknown")
            cfg = run.get("config", {})
            top_trials = run.get("top_trials", [])
            model_stats = run.get("model_stats", {})

            if status == "running":
                status_display = "🟢 运行中"
            elif status == "completed":
                status_display = "✅ 已完成"
            elif status == "interrupted":
                status_display = "🟡 中断"
            elif status == "no_data":
                status_display = "⚠️ 无数据"
            else:
                status_display = f"❓ {status}"

            top_str = ", ".join(f"#{t}" for t in top_trials)

            metric_str = ""
            if model_stats:
                metric_str = (
                    f"{model_stats.get('explored_states', 0)}/{model_stats.get('total_states', 0)} "
                    f"| RS {model_stats.get('avg_replacement_score', 0):.3f}"
                )

            time_str = run.get("start_time", "?")
            end_time = run.get("end_time", "")
            if end_time and end_time != time_str:
                time_str = f"{time_str[:16]}~{end_time[:16]}"

            cols = st.columns([0.7, 2, 0.8, 0.8, 1, 1.2, 1.2, 0.8, 0.5])
            cols[0].caption(f"#{fid:04d}")
            cols[1].caption(top_str)
            cols[2].caption(str(cfg.get("episodes_per_config", "?")))
            cols[3].caption(str(cfg.get("explore_budget", "?")))
            cols[4].caption(status_display)
            cols[5].caption(time_str)
            cols[6].caption(metric_str if metric_str else "—")

            detail_key = f"ft_detail_{fid}"
            if cols[7].button("📋", key=detail_key):
                st.session_state[f"_ft_show_detail_{fid}"] = not st.session_state.get(
                    f"_ft_show_detail_{fid}", False
                )
                st.rerun()

            del_key = f"ft_del_{fid}"
            if cols[8].button("🗑", key=del_key):
                _delete_finetune_run(fid)
                st.toast(f"训练 #{fid:04d} 已删除", icon="🗑️")
                st.rerun()

            if st.session_state.get(f"_ft_show_detail_{fid}", False):
                with st.expander(f"训练 #{fid:04d} 详情", expanded=True):
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.caption("**配置**")
                        st.caption(f"Top-K Trials: {top_trials}")
                        st.caption(
                            f"Episodes/Config: {cfg.get('episodes_per_config', '?')}"
                        )
                        st.caption(f"探索预算: {cfg.get('explore_budget', '?')}")
                        st.caption(f"Sigma: {cfg.get('sigma', '?')}")
                        st.caption(f"目标探索次数: {cfg.get('target_visits', '?')}")
                        st.caption(f"收敛阈值: {cfg.get('convergence_threshold', '?')}")
                        st.caption(f"胜率阈值: {cfg.get('win_rate_threshold', '?')}")
                    with dc2:
                        st.caption("**结果**")
                        st.caption(f"开始: {run.get('start_time', '?')}")
                        st.caption(f"结束: {run.get('end_time', '—')}")
                        if model_stats:
                            st.caption(
                                f"已探索: {model_stats.get('explored_states', 0)}/{model_stats.get('total_states', 0)}"
                            )
                            st.caption(
                                f"探索比例: {model_stats.get('explored_ratio', 0):.2%}"
                            )
                            st.caption(
                                f"平均 RS: {model_stats.get('avg_replacement_score', 0):.4f}"
                            )
                            st.caption(
                                f"训练 Episodes: {model_stats.get('trained_episodes', 0)}"
                            )
                        else:
                            st.caption("无模型统计")

                    sample_dir = _FINETUNE_SAMPLES_DIR / f"finetune_{fid:04d}"
                    if sample_dir.is_dir():
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            base_ep = sample_dir / "base_episodes.jsonl"
                            if base_ep.exists():
                                try:
                                    line_count = sum(
                                        1 for _ in open(str(base_ep), encoding="utf-8")
                                    )
                                    st.caption(f"base_episodes.jsonl ({line_count} 行)")
                                except Exception:
                                    st.caption("base_episodes.jsonl")
                                if st.button(
                                    "查看基准样本",
                                    key=f"ft_base_{fid}",
                                    use_container_width=True,
                                ):
                                    _show_finetune_jsonl(
                                        base_ep, f"基准样本 (#{fid:04d})"
                                    )
                        with sc2:
                            explore_ep = sample_dir / "explore_episodes.jsonl"
                            if explore_ep.exists():
                                try:
                                    line_count = sum(
                                        1
                                        for _ in open(str(explore_ep), encoding="utf-8")
                                    )
                                    st.caption(
                                        f"explore_episodes.jsonl ({line_count} 行)"
                                    )
                                except Exception:
                                    st.caption("explore_episodes.jsonl")
                                if line_count > 0:
                                    if st.button(
                                        "查看探索样本",
                                        key=f"ft_explore_{fid}",
                                        use_container_width=True,
                                    ):
                                        _show_finetune_jsonl(
                                            explore_ep, f"探索样本 (#{fid:04d})"
                                        )
                                else:
                                    trainer_log = sample_dir / "trainer.log"
                                    reason = ""
                                    if trainer_log.exists():
                                        try:
                                            log_text = trainer_log.read_text(
                                                encoding="utf-8"
                                            )
                                            if "no more targets" in log_text:
                                                reason = "Phase B: 所有状态的替代动作已探索完毕"
                                            elif (
                                                "no ETG" in log_text
                                                or "no state_action_map" in log_text
                                            ):
                                                reason = "Phase B: ETG state_action_map 加载失败"
                                            elif "Phase B" not in log_text:
                                                reason = (
                                                    "Phase B 未执行（Phase A 无数据）"
                                                )
                                        except Exception:
                                            pass
                                    if reason:
                                        st.caption(f"探索为空 - {reason}")
                                    else:
                                        st.caption("探索样本为空")

                    qc_path = sample_dir / "q_table_snapshot.json"
                    if qc_path.exists():
                        if st.button(
                            "查看 Q-Table 快照",
                            key=f"ft_qtable_{fid}",
                            use_container_width=True,
                        ):
                            _show_qtable_snapshot(qc_path)

                    progress_path = sample_dir / "progress.json"
                    if progress_path.exists():
                        if st.button(
                            "查看训练进度",
                            key=f"ft_progress_{fid}",
                            use_container_width=True,
                        ):
                            _show_finetune_progress(progress_path)

                    trainer_log = sample_dir / "trainer.log"
                    if trainer_log.exists():
                        if st.button(
                            "查看训练日志",
                            key=f"ft_trainerlog_{fid}",
                            use_container_width=True,
                        ):
                            _show_log(
                                f"finetune_samples/finetune_{fid:04d}/trainer.log"
                            )

                    group_dirs = sorted(sample_dir.glob("group_*"))
                    if group_dirs:
                        for gd in group_dirs:
                            cand_path = gd / "candidates.json"
                            if cand_path.exists():
                                try:
                                    cands = json.loads(
                                        cand_path.read_text(encoding="utf-8")
                                    )
                                except Exception:
                                    cands = []
                                group_name = gd.name
                                if st.button(
                                    f"查看候选列表 ({group_name})",
                                    key=f"ft_cand_{fid}_{group_name}",
                                    use_container_width=True,
                                ):
                                    if cands:
                                        st.caption(
                                            f"共 {len(cands)} 个候选（ETG 反事实评估）"
                                        )
                                        df = pd.DataFrame(cands)
                                        st.dataframe(
                                            df, use_container_width=True, height=400
                                        )
                                    else:
                                        st.info("无候选（所有状态的替代动作收益 <= 0）")

                    base_ep = sample_dir / "base_episodes.jsonl"
                    qc_path = sample_dir / "q_table_snapshot.json"
                    if base_ep.exists() and qc_path.exists():
                        if st.button(
                            "动作切换收益分析",
                            key=f"ft_action_switch_{fid}",
                            use_container_width=True,
                        ):
                            _render_action_switch_analysis(sample_dir)
    else:
        st.info("暂无训练记录。")

    st.divider()

    if _is_learner_alive():
        st.warning("参数寻优正在运行中，请先停止后再启动训练。")
    else:
        start_ft = st.button(
            "启动在线协同训练", type="primary", key="ft_start", use_container_width=True
        )
        if start_ft:
            cmd = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "finetune_trainer.py"),
                "--n_trials",
                str(top_k),
                "--episodes_per_trial",
                str(ep_per_cfg),
                "--sigma",
                str(sigma),
                "--target_visits",
                str(target_visits),
                "--reward_mode",
                reward_mode,
            ]
            _etg_file = st.session_state.get("_learner_etg_file", "")
            _etg_data_dir = st.session_state.get("_learner_etg_data_dir", "")
            _etg_map_id = st.session_state.get("_learner_etg_map_id", "")
            _map_key = st.session_state.get("_learner_map_key") or get_map_key_for_map_id(
                _etg_map_id
            )
            cmd.extend(["--map_key", _map_key])
            if _etg_file:
                cmd.extend(["--etg_file", _etg_file])
            if _etg_data_dir:
                cmd.extend(["--data_dir", _etg_data_dir])
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            _FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
            _FINETUNE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            td = _get_active_trials_dir()
            if td:
                td.mkdir(parents=True, exist_ok=True)

            log_path = _FINETUNE_DIR / "finetune_trainer.log"
            log_file = open(str(log_path), "w", encoding="utf-8")
            st.session_state.finetune_log_file = log_file
            flags = 0
            if sys.platform == "win32":
                flags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            p = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                cwd=str(ROOT_DIR),
                creationflags=flags,
            )
            st.session_state.finetune_proc = p
            _PID_FILE.write_text(str(p.pid))
            st.toast(f"在线协同训练已启动 (PID: {p.pid})", icon="🚀")
            time.sleep(1)
            st.rerun()

    if "finetune_proc" in st.session_state:
        proc = st.session_state.finetune_proc
        if proc and proc.poll() is not None:
            del st.session_state.finetune_proc

    ft_log = _FINETUNE_DIR / "finetune_trainer.log"
    if ft_log.exists():
        if st.button("查看全局训练日志", key="ft_show_log"):
            _show_log("finetune_runs/finetune_trainer.log")


def _show_finetune_jsonl(file_path, title):
    try:
        lines = []
        with open(str(file_path), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, list):
                        lines.append(
                            f"[{i}] ({len(data)} 步) state={data[0].get('state_id', '?')} action={data[0].get('action_code', '?')} ..."
                        )
                    else:
                        lines.append(
                            f"[{i}] {json.dumps(data, ensure_ascii=False)[:200]}"
                        )
                except json.JSONDecodeError:
                    lines.append(f"[{i}] (解析失败)")
        show = "\n".join(lines[-50:]) if len(lines) > 50 else "\n".join(lines)
        st.text_area(title, show, height=300, label_visibility="collapsed")
    except Exception as e:
        st.error(f"读取失败: {e}")


def _show_qtable_snapshot(file_path):
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        summary_rows = []
        for sid_str, actions in data.items():
            total_v = sum(a.get("visits", 0) for a in actions.values())
            best_ac = (
                max(actions, key=lambda k: actions[k].get("avg_reward", 0))
                if actions
                else "?"
            )
            best_r = actions[best_ac].get("avg_reward", 0) if best_ac != "?" else 0
            summary_rows.append(
                {
                    "state_id": sid_str,
                    "actions": len(actions),
                    "total_visits": total_v,
                    "best_action": best_ac,
                    "best_reward": round(best_r, 2),
                }
            )
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("Q-Table 为空。")
    except Exception as e:
        st.error(f"读取失败: {e}")


def _show_finetune_progress(file_path):
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        st.json(data)
    except Exception as e:
        st.error(f"读取失败: {e}")


def _render_action_switch_analysis(sample_dir: Path):
    import pickle as _pickle

    etg_file = st.session_state.get("_learner_etg_file", "")
    if not etg_file:
        st.warning("未配置 ETG 文件路径，无法分析")
        return
    etg_path = ROOT_DIR / "cache" / "experience_transition_graph" / etg_file
    if not etg_path.exists():
        st.warning(f"ETG 文件不存在: {etg_path}")
        return

    try:
        with open(str(etg_path), "rb") as f:
            raw = _pickle.load(f)
        etg_sam = raw.get("state_action_map", {})
    except Exception as e:
        st.error(f"加载 ETG 失败: {e}")
        return

    q_path = sample_dir / "q_table_snapshot.json"
    base_path = sample_dir / "base_episodes.jsonl"
    try:
        q_table = json.loads(q_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"加载 Q-Table 失败: {e}")
        return

    rows = []
    ep_idx = 0
    with open(str(base_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                flow = json.loads(line)
            except json.JSONDecodeError:
                continue
            for step_idx, step in enumerate(flow):
                sid = step.get("state_id")
                current_ac = step.get("action_code")
                if sid is None or current_ac is None:
                    continue
                if abs(sid) > 10000:
                    continue
                sid_int = int(sid)
                if sid_int not in etg_sam:
                    continue
                etg_actions = etg_sam[sid_int]
                if not etg_actions:
                    continue
                current_quality = 0.0
                if current_ac in etg_actions:
                    current_quality = getattr(
                        etg_actions[current_ac], "quality_score", 0.0
                    )
                best_alt_ac = None
                best_alt_quality = -float("inf")
                for ac, stats in etg_actions.items():
                    if ac == current_ac:
                        continue
                    q = getattr(stats, "quality_score", 0.0)
                    if q > best_alt_quality:
                        best_alt_quality = q
                        best_alt_ac = ac
                if best_alt_ac is None:
                    continue
                gain = best_alt_quality - current_quality
                if gain > 0:
                    rows.append(
                        {
                            "Episode": ep_idx,
                            "Step": step_idx,
                            "State": sid_int,
                            "当前动作": current_ac,
                            "当前质量": round(current_quality, 2),
                            "最优替代": best_alt_ac,
                            "替代质量": round(best_alt_quality, 2),
                            "收益差": round(gain, 2),
                        }
                    )
            ep_idx += 1

    if not rows:
        st.info("未发现可提升的动作切换（当前动作已是 ETG 最优，或无替代动作）")
        return

    rows.sort(key=lambda x: x["收益差"], reverse=True)
    df = pd.DataFrame(rows)
    st.caption(f"共发现 {len(rows)} 个可提升的动作切换（按收益差降序）")
    st.dataframe(df, use_container_width=True, height=400)
