#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a historical-action replay baseline.

The baseline selects one or more action sequences from historical records,
executes those sequences directly in the SC2 environment, and archives the
result under output/learner_results/all_data/Replay-baseline/<experiment_id>.
It intentionally bypasses ETG planning, beam search, and action tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from scripts.parameter_learner import (
    _analyze_local_result,
    _compute_stability,
    _find_free_port,
    _load_config,
    _terminate_process_tree,
    _wait_for_file_progress,
    _wait_for_game_ready,
    _wait_for_server,
)


_DEFAULT_CONFIG = ROOT_DIR / "configs" / "learner_config.yaml"
_DEFAULT_ARCHIVE_ROOT = ROOT_DIR / "output" / "learner_results" / "all_data"
_METHOD_GROUP = "Replay-baseline"
_MANIFEST_NAME = "experiment_manifest.json"
_ACTION_CHARS = set("abcdefghijk")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _valid_action_code(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 2
        and value[0] in "01234"
        and value[1] in _ACTION_CHARS
    )


def _parse_action_string(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if "," in raw:
        actions = [part.strip() for part in raw.split(",")]
    else:
        actions = [raw[i : i + 2] for i in range(0, len(raw), 2)]
    return [action for action in actions if _valid_action_code(action)]


def _parse_result_line(line: str) -> Dict[str, Any]:
    parts = [part.strip() for part in line.strip().split("\t") if part.strip()]
    result = parts[0] if parts else None
    frames = None
    reward_d = None
    reward_a = None

    if len(parts) >= 2:
        frame_text = parts[1].strip().strip("[]")
        try:
            frames = int(float(frame_text))
        except ValueError:
            frames = parts[1]
    if len(parts) >= 3:
        try:
            reward_d = float(parts[2])
        except ValueError:
            reward_d = None
    if len(parts) >= 4:
        try:
            reward_a = float(parts[3])
        except ValueError:
            reward_a = None

    score = None
    if reward_d is not None and reward_a is not None:
        score = reward_d + reward_a

    return {
        "result": result,
        "score": score,
        "frames": frames,
        "reward_d": reward_d,
        "reward_a": reward_a,
        "raw": line.strip(),
    }


def _load_action_log(action_log: Path, result_log: Optional[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    if result_log and result_log.exists():
        results = [
            _parse_result_line(line)
            for line in result_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]

    with open(str(action_log), "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if not row:
                continue
            actions = _parse_action_string(row[0])
            if not actions:
                continue
            meta = results[idx] if idx < len(results) else {}
            rows.append(
                {
                    "source": "action_log",
                    "source_index": idx,
                    "actions": actions,
                    "source_result": meta.get("result"),
                    "source_score": meta.get("score"),
                    "source_frames": meta.get("frames"),
                    "source_reward_d": meta.get("reward_d"),
                    "source_reward_a": meta.get("reward_a"),
                    "source_raw_result": meta.get("raw"),
                }
            )
    return rows


def _episode_actions(record: Dict[str, Any]) -> List[str]:
    frames = record.get("frames") or record.get("steps") or []
    actions = []
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            code = frame.get("action_code")
            if _valid_action_code(code):
                actions.append(code)
    return actions


def _episode_candidate(record: Dict[str, Any], idx: int, source: str) -> Optional[Dict[str, Any]]:
    actions = _episode_actions(record)
    if not actions:
        return None
    score = record.get("score", record.get("final_score"))
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    return {
        "source": source,
        "source_index": idx,
        "actions": actions,
        "source_result": record.get("result"),
        "source_score": score,
        "source_frames": len(actions),
        "source_raw_result": None,
    }


def _iter_episode_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if isinstance(obj.get("frames"), list) or isinstance(obj.get("steps"), list):
            yield obj
            return
        for value in obj.values():
            yield from _iter_episode_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_episode_dicts(value)


def _load_episodes_jsonl(path: Path) -> List[Dict[str, Any]]:
    candidates = []
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                cand = _episode_candidate(record, idx, "episodes_jsonl")
                if cand:
                    candidates.append(cand)
    return candidates


def _load_episodes_pkl(path: Path) -> List[Dict[str, Any]]:
    with open(str(path), "rb") as f:
        obj = pickle.load(f)
    candidates = []
    for idx, record in enumerate(_iter_episode_dicts(obj)):
        cand = _episode_candidate(record, idx, "episodes_pkl")
        if cand:
            candidates.append(cand)
    return candidates


def _load_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.action_log:
        action_log = _resolve_path(args.action_log)
        result_log = _resolve_path(args.result_log) if args.result_log else action_log.with_name("game_result.txt")
        if not action_log.exists():
            raise FileNotFoundError(f"action log not found: {action_log}")
        return _load_action_log(action_log, result_log if result_log.exists() else None)
    if args.episodes_jsonl:
        path = _resolve_path(args.episodes_jsonl)
        if not path.exists():
            raise FileNotFoundError(f"episodes jsonl not found: {path}")
        return _load_episodes_jsonl(path)
    if args.episodes_pkl:
        path = _resolve_path(args.episodes_pkl)
        if not path.exists():
            raise FileNotFoundError(f"episodes pkl not found: {path}")
        return _load_episodes_pkl(path)
    raise ValueError("one of --action-log, --episodes-jsonl, or --episodes-pkl is required")


def _rank_key(candidate: Dict[str, Any]) -> Tuple[float, int, int, int]:
    is_win = 1 if str(candidate.get("source_result", "")).lower() == "win" else 0
    score = candidate.get("source_score")
    score_val = float(score) if score is not None else float("-inf")
    frames = candidate.get("source_frames")
    try:
        frame_val = int(frames)
    except (TypeError, ValueError):
        frame_val = 10**9
    return score_val, is_win, -frame_val, -int(candidate.get("source_index", 0))


def _select_candidates(
    candidates: List[Dict[str, Any]],
    selection: str,
    top_k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if selection == "first_n":
        selected = candidates[:top_k]
    elif selection == "random":
        rng = random.Random(seed)
        selected = list(candidates)
        rng.shuffle(selected)
        selected = selected[:top_k]
    elif selection == "top_score":
        selected = sorted(candidates, key=_rank_key, reverse=True)[:top_k]
    else:
        wins = [item for item in candidates if str(item.get("source_result", "")).lower() == "win"]
        pool = wins if wins else candidates
        selected = sorted(pool, key=_rank_key, reverse=True)[:top_k]

    return [
        {
            **item,
            "sequence_id": f"seq_{i + 1:03d}",
            "action_count": len(item.get("actions", [])),
        }
        for i, item in enumerate(selected)
    ]


def _allocate_repeats(
    selected: List[Dict[str, Any]],
    episodes: int,
    repeats_per_sequence: Optional[int],
    allocation: str,
    seed: int,
) -> List[int]:
    if not selected:
        return []
    if repeats_per_sequence is not None:
        return [int(repeats_per_sequence)] * len(selected)
    total = max(int(episodes), len(selected))
    if allocation == "random" and len(selected) > 1:
        rng = random.Random(seed)
        counts = [0] * len(selected)
        for _ in range(total):
            counts[rng.randrange(len(selected))] += 1
        return counts
    base = total // len(selected)
    extra = total % len(selected)
    return [base + (1 if i < extra else 0) for i in range(len(selected))]


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


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [record.get("metrics", {}) for record in records if record.get("metrics")]
    if not metrics:
        return {}
    keys = ["win_rate", "avg_score", "score_std", "stability", "objective", "penalty_factor"]
    aggregate: Dict[str, Any] = {}
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in metrics if item.get(key) is not None]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else None
        aggregate[key] = {"mean": mean, "std": std, "ci95": ci95}
    total_episodes = int(sum(int(item.get("num_episodes", 0)) for item in metrics))
    total_wins = int(round(sum(float(item.get("win_rate", 0.0)) * int(item.get("num_episodes", 0)) for item in metrics)))
    aggregate["total_episodes"] = total_episodes
    aggregate["total_wins"] = total_wins
    aggregate["sequence_count"] = len(records)
    return aggregate


def _resolve_path(value: Optional[str]) -> Path:
    if not value:
        return Path()
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _relative_or_str(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _deserialize_bktree_node(node_data: Optional[Dict[str, Any]]) -> Any:
    if node_data is None:
        return None
    from src.structure.BKTree_sc2 import ClusterNode

    node = ClusterNode(node_data["state"], node_data["cluster_id"])
    for dist_key, child_data in node_data.get("children", {}).items():
        dist_val = int(dist_key) if str(dist_key).isdigit() else float(dist_key)
        child_node = _deserialize_bktree_node(child_data)
        if child_node is not None:
            node.children[dist_val] = child_node
    return node


def _load_readonly_bktree_file(path: Path, distance_index: int) -> Any:
    from src.structure.BKTree_sc2 import BKTree, get_max_cluster_id
    from src.structure.custom_distance_sc2 import CustomDistance

    with open(str(path), "r", encoding="utf-8") as f:
        tree_data = json.load(f)
    tree = BKTree(CustomDistance(threshold=0.5).multi_distance, distance_index=distance_index)
    tree.root = _deserialize_bktree_node(tree_data)
    if tree.root is not None:
        tree.next_cluster_id = get_max_cluster_id(tree) + 1
    return tree


def _load_state_id_map(data_dir: Path) -> Dict[Tuple[int, int], int]:
    path = data_dir / "graph" / "state_node.txt"
    mapping: Dict[Tuple[int, int], int] = {}
    if not path.exists():
        return mapping
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            cluster_text = parts[0].strip().strip("()")
            cluster_parts = [part.strip() for part in cluster_text.split(",")]
            if len(cluster_parts) != 2:
                continue
            try:
                mapping[(int(cluster_parts[0]), int(cluster_parts[1]))] = int(float(parts[1]))
            except ValueError:
                continue
    return mapping


class _ReadonlyBKTreeStateProjector:
    def __init__(self, data_dir: Path, primary_threshold: float, secondary_threshold: float):
        self.data_dir = data_dir
        self.bktree_dir = data_dir / "bktree"
        self.primary_threshold = float(primary_threshold)
        self.secondary_threshold = float(secondary_threshold)
        self.state_id_map = _load_state_id_map(data_dir)
        self.primary = _load_readonly_bktree_file(self.bktree_dir / "primary_bktree.json", 0)
        self.secondary_cache: Dict[int, Optional[Any]] = {}

    @classmethod
    def maybe_create(
        cls,
        data_dir: Optional[Path],
        primary_threshold: float,
        secondary_threshold: float,
    ) -> Optional["_ReadonlyBKTreeStateProjector"]:
        if data_dir is None:
            return None
        bktree_dir = data_dir / "bktree"
        if not (bktree_dir / "primary_bktree.json").exists():
            return None
        return cls(data_dir, primary_threshold, secondary_threshold)

    def _secondary(self, primary_id: int) -> Optional[Any]:
        if primary_id in self.secondary_cache:
            return self.secondary_cache[primary_id]
        path = self.bktree_dir / f"secondary_bktree_{int(primary_id)}.json"
        if not path.exists():
            self.secondary_cache[primary_id] = None
            return None
        self.secondary_cache[primary_id] = _load_readonly_bktree_file(path, 1)
        return self.secondary_cache[primary_id]

    def project(self, norm_state: Mapping[str, Any]) -> Dict[str, Any]:
        primary_id, primary_dist = self.primary.query_nearest(norm_state)
        if primary_id is None:
            return {"state_key": None, "state_id": None, "nid": None, "bktree_match": {"rejected": True, "reason": "primary_query_failed"}}

        primary_id = int(primary_id)
        primary_dist = float(primary_dist)
        secondary_tree = self._secondary(primary_id)
        if secondary_tree is None or secondary_tree.root is None:
            secondary_id = 1
            secondary_dist = None
            rejected = primary_dist > self.primary_threshold
            reason = "secondary_missing" if not rejected else "distance_over_threshold"
        else:
            secondary_raw, secondary_raw_dist = secondary_tree.query_nearest(norm_state)
            secondary_id = int(secondary_raw) if secondary_raw is not None else 1
            secondary_dist = float(secondary_raw_dist)
            rejected = primary_dist > self.primary_threshold or secondary_dist > self.secondary_threshold
            reason = "distance_over_threshold" if rejected else "accepted"

        state_id = self.state_id_map.get((primary_id, secondary_id))
        if rejected:
            state_key = f"ood:{primary_id}:{secondary_id}:p{primary_dist:.3f}:s{secondary_dist:.3f}" if secondary_dist is not None else f"ood:{primary_id}:{secondary_id}:p{primary_dist:.3f}:sNA"
            nid = None
            state_id = None
        else:
            state_key = state_id if state_id is not None else f"{primary_id}:{secondary_id}"
            nid = state_id

        return {
            "state_key": state_key,
            "state_id": state_id,
            "nid": nid,
            "bktree_match": {
                "primary_id": primary_id,
                "secondary_id": secondary_id,
                "primary_distance": primary_dist,
                "secondary_distance": secondary_dist,
                "rejected": rejected,
                "reason": reason,
            },
        }


def _annotate_episode_frames(
    episodes: List[Dict[str, Any]],
    projector: Optional[_ReadonlyBKTreeStateProjector],
) -> None:
    if projector is None:
        return
    cache: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        frames = ep.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            norm_state = frame.get("norm_state")
            if not isinstance(norm_state, Mapping):
                continue
            cache_key = json.dumps(norm_state, ensure_ascii=False, sort_keys=True)
            projected = cache.get(cache_key)
            if projected is None:
                projected = projector.project(norm_state)
                cache[cache_key] = projected
            frame.update(projected)


def _run_sequence(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    exp_dir: Path,
    sequence: Dict[str, Any],
    repeats: int,
    sequence_index: int,
) -> Dict[str, Any]:
    seq_dir = exp_dir / "replay_runs" / sequence["sequence_id"]
    seq_dir.mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "episodes_hp.jsonl", "progress.json", "plan.log", "live_game.log"):
        path = seq_dir / name
        if path.exists():
            path.unlink()

    port = _find_free_port(exclude={8000, 8501, 8502})
    log_path = seq_dir / "live_game.log"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_live_game.py"),
        "--mode",
        "all",
        "--port",
        str(port),
        "--map_key",
        args.map_key,
        "--max_episodes",
        "0",
        "--autopilot_mode",
        "replay",
        "--replay_actions",
        ",".join(sequence["actions"]),
        "--replay_runs",
        str(int(repeats)),
        "--replay_exhaustion_mode",
        "end_episode",
        "--local_result_dir",
        str(seq_dir),
        "--target_episodes",
        str(int(repeats)),
        "--trial_number",
        str(sequence_index),
        "--plan_log_path",
        str(seq_dir / "plan.log"),
        "--skip_api_kg",
        "--skip_game_kg",
        "--primary_threshold",
        str(float(args.primary_threshold)),
        "--secondary_threshold",
        str(float(args.secondary_threshold)),
    ]
    if args.data_dir:
        cmd.extend(["--data_dir", args.data_dir])
    if args.fallback_action:
        cmd.extend(["--fallback_action", args.fallback_action])

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    record = {
        "sequence_id": sequence["sequence_id"],
        "source_index": sequence.get("source_index"),
        "source_result": sequence.get("source_result"),
        "source_score": sequence.get("source_score"),
        "source_reward_d": sequence.get("source_reward_d"),
        "source_reward_a": sequence.get("source_reward_a"),
        "source_frames": sequence.get("source_frames"),
        "action_count": sequence.get("action_count"),
        "repeats": int(repeats),
        "command": cmd,
        "status": "running",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(seq_dir / "sequence_run.json", record)

    log_file = open(str(log_path), "w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        creationflags=flags,
        env=env,
    )
    completed = False
    try:
        startup_wait = int(cfg.get("execution", {}).get("startup_wait_seconds", 120))
        ready_wait = int(cfg.get("execution", {}).get("game_ready_wait_seconds", 240))
        if not _wait_for_server(port, timeout=startup_wait, proc=proc, log_path=log_path):
            record["status"] = "server_startup_timeout"
            return record
        if ready_wait and not _wait_for_game_ready(port, timeout=ready_wait, proc=proc, log_path=log_path):
            record["status"] = "game_startup_timeout"
            return record
        completed = _wait_for_file_progress(seq_dir, int(repeats), cfg, expected_trial=sequence_index)
    finally:
        _terminate_process_tree(proc)
        log_file.close()

    stability_segments = int(cfg.get("objective", {}).get("stability_segments", 5))
    metrics = _analyze_local_result(seq_dir, stability_segments, expected_trial=sequence_index)
    metrics.update(_objective(metrics, cfg))
    record["status"] = "completed" if completed else "timeout"
    record["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["metrics"] = metrics
    _write_json(seq_dir / "sequence_run.json", record)
    return record


def _write_batch_replay_source(exp_dir: Path, selected: List[Dict[str, Any]], repeats: List[int]) -> Tuple[Path, List[Dict[str, Any]]]:
    source_dir = exp_dir / "replay_collector_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    action_log = source_dir / "action_log.csv"
    expanded_rows: List[Dict[str, Any]] = []
    with open(str(action_log), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for sequence, repeat_count in zip(selected, repeats):
            raw_actions = "".join(sequence.get("actions", []))
            for repeat_index in range(int(repeat_count)):
                writer.writerow([raw_actions])
                expanded_rows.append(
                    {
                        "row_index": len(expanded_rows),
                        "sequence_id": sequence.get("sequence_id"),
                        "source_index": sequence.get("source_index"),
                        "repeat_index": repeat_index,
                        "action_count": len(sequence.get("actions", [])),
                    }
                )
    _write_json(source_dir / "expanded_rows.json", {"rows": expanded_rows})
    return source_dir, expanded_rows


def _read_replay_collector_episodes(collector_dir: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for pkl_path in sorted(collector_dir.glob("episodes_*.pkl")):
        with open(str(pkl_path), "rb") as f:
            while True:
                try:
                    item = pickle.load(f)
                except EOFError:
                    break
                except Exception:
                    break
                if isinstance(item, dict):
                    episodes.append(item)
    return episodes


def _episodes_to_jsonl(episodes: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as f:
        for idx, ep in enumerate(episodes, start=1):
            frames = ep.get("frames", [])
            state_key_sequence = [
                frame.get("state_key")
                for frame in frames
                if isinstance(frame, dict) and frame.get("state_key") is not None
            ]
            state_id_sequence = [
                frame.get("state_id")
                for frame in frames
                if isinstance(frame, dict) and frame.get("state_id") is not None
            ]
            record = {
                "episode_id": idx,
                "source_idx": ep.get("source_idx"),
                "replay_idx": ep.get("replay_idx"),
                "result": ep.get("result", "Dogfall"),
                "score": ep.get("score", 0.0),
                "state_key_sequence": state_key_sequence,
                "state_id_sequence": state_id_sequence,
                "frames": frames,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _metrics_from_episodes(episodes: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not episodes:
        metrics = {
            "win_rate": 0.0,
            "avg_score": 0.0,
            "score_std": 0.0,
            "stability": 0.0,
            "num_episodes": 0,
        }
    else:
        scores = [float(ep.get("score", 0.0)) for ep in episodes]
        wins = sum(1 for ep in episodes if ep.get("result") == "Win")
        stability_segments = int(cfg.get("objective", {}).get("stability_segments", 5))
        metrics = {
            "win_rate": wins / len(episodes),
            "avg_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "stability": _compute_stability(episodes, stability_segments),
            "num_episodes": len(episodes),
        }
    metrics.update(_objective(metrics, cfg))
    return metrics


def _aggregate_single(metrics: Dict[str, Any]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for key in ("win_rate", "avg_score", "score_std", "stability", "objective", "penalty_factor"):
        if metrics.get(key) is not None:
            aggregate[key] = {"mean": float(metrics.get(key, 0.0)), "std": 0.0, "ci95": None}
    aggregate["total_episodes"] = int(metrics.get("num_episodes", 0))
    aggregate["total_wins"] = int(round(float(metrics.get("win_rate", 0.0)) * int(metrics.get("num_episodes", 0))))
    aggregate["sequence_count"] = 1
    return aggregate


def _wait_for_collector_output(proc: subprocess.Popen, collector_base: Path, expected: int, timeout_minutes: int, log_path: Path) -> Optional[Path]:
    import time

    deadline = time.time() + int(timeout_minutes) * 60
    last_logged = -1
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        candidates = sorted(
            [p for p in collector_base.glob("ep*_r*_p*_s*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            collector_dir = candidates[0]
            progress = collector_dir / "progress.json"
            if progress.exists():
                try:
                    data = json.loads(progress.read_text(encoding="utf-8"))
                    done = int(data.get("completed_episodes", 0))
                    if done != last_logged and (done == expected or done % 10 == 0):
                        print(f"  batch_replay progress: {done}/{expected}", flush=True)
                        last_logged = done
                    if done >= expected:
                        return collector_dir
                except Exception:
                    pass
            if list(collector_dir.glob("batch_stats_*.json")):
                return collector_dir
        time.sleep(2)

    candidates = sorted(
        [p for p in collector_base.glob("ep*_r*_p*_s*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    if log_path.exists():
        print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)
    return None


def _run_replay_collector_eval(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    exp_dir: Path,
    selected: List[Dict[str, Any]],
    repeats: List[int],
    final_eval_dir: Path,
) -> Dict[str, Any]:
    source_dir, expanded_rows = _write_batch_replay_source(exp_dir, selected, repeats)
    expected = len(expanded_rows)
    if expected <= 0:
        raise ValueError("no expanded replay rows to execute")

    collector_base = exp_dir / "replay_collector_output"
    collector_base.mkdir(parents=True, exist_ok=True)
    port = _find_free_port(exclude={8000, 8501, 8502})
    log_path = exp_dir / "replay_collector_run.log"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "run_live_game.py"),
        "--mode",
        "all",
        "--port",
        str(port),
        "--map_key",
        args.map_key,
        "--data_dir",
        str(source_dir),
        "--autopilot_mode",
        "batch_replay",
        "--batch_start",
        "0",
        "--batch_end",
        str(expected - 1),
        "--replay_count",
        "1",
        "--batch_output_dir",
        str(collector_base),
        "--primary_threshold",
        str(float(args.primary_threshold)),
        "--secondary_threshold",
        str(float(args.secondary_threshold)),
        "--skip_api_kg",
    ]

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log_file = open(str(log_path), "w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        creationflags=flags,
        env=env,
    )
    collector_dir: Optional[Path] = None
    try:
        startup_wait = int(cfg.get("execution", {}).get("startup_wait_seconds", 120))
        ready_wait = int(cfg.get("execution", {}).get("game_ready_wait_seconds", 240))
        if not _wait_for_server(port, timeout=startup_wait, proc=proc, log_path=log_path):
            raise RuntimeError("batch_replay server startup timeout")
        if ready_wait and not _wait_for_game_ready(port, timeout=ready_wait, proc=proc, log_path=log_path):
            raise RuntimeError("batch_replay game startup timeout")
        collector_dir = _wait_for_collector_output(
            proc,
            collector_base,
            expected,
            int(args.timeout_minutes),
            log_path,
        )
    finally:
        _terminate_process_tree(proc)
        log_file.close()

    if collector_dir is None:
        raise RuntimeError("ReplayCollector did not produce an output directory")

    episodes = _read_replay_collector_episodes(collector_dir)
    data_dir = _resolve_path(args.data_dir) if args.data_dir else None
    projector = _ReadonlyBKTreeStateProjector.maybe_create(
        data_dir,
        float(args.primary_threshold),
        float(args.secondary_threshold),
    )
    _annotate_episode_frames(episodes, projector)
    _episodes_to_jsonl(episodes, final_eval_dir / "episodes.jsonl")
    _write_json(
        exp_dir / "replay_collector_artifacts.json",
        {
            "collector_dir": str(collector_dir),
            "source_dir": str(source_dir),
            "expanded_episode_rows": expected,
            "command": cmd,
            "log_path": str(log_path),
            "state_id_projection": {
                "enabled": projector is not None,
                "data_dir": str(data_dir) if data_dir else "",
                "bktree_path": str(projector.bktree_dir) if projector else "",
                "state_id_map_entries": len(projector.state_id_map) if projector else 0,
            },
        },
    )
    metrics = _metrics_from_episodes(episodes, cfg)
    return {
        "status": "completed" if len(episodes) >= expected else "partial",
        "collector_dir": str(collector_dir),
        "metrics": metrics,
        "episodes": len(episodes),
        "expected_episodes": expected,
    }


def _build_manifest(args: argparse.Namespace, exp_dir: Path, selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    bktree_path = str(Path(args.data_dir) / "bktree") if args.data_dir else ""
    return {
        "experiment_id": args.experiment_id,
        "display_name": args.display_name or args.experiment_id,
        "map_key": args.map_key,
        "map_id": args.map_id,
        "experiment_type": "historical_replay_baseline",
        "method": "Historical Action Replay",
        "kg_name": args.kg_name or "",
        "kg_file": args.kg_file or "",
        "transitions": args.transitions or "",
        "data_dir": args.data_dir or "",
        "dataset_type": args.dataset_type,
        "replay_dataset_expansion": bool(args.replay_dataset_expansion),
        "bktree": {
            "primary_threshold": float(args.primary_threshold),
            "secondary_threshold": float(args.secondary_threshold),
            "path": bktree_path,
        },
        "source_run": args.source_run or "",
        "source_sequences": {
            "selection": args.selection,
            "ranking_metric": "source_score = reward_d + reward_a from game_result.txt columns 3 and 4; column 2 is episode frames",
            "sequence_allocation": args.sequence_allocation,
            "top_k": int(args.top_k),
            "selected_count": len(selected),
            "action_log": args.action_log or "",
            "result_log": args.result_log or "",
            "episodes_jsonl": args.episodes_jsonl or "",
            "episodes_pkl": args.episodes_pkl or "",
        },
        "notes": args.notes or "Historical action-sequence replay baseline; ETG planning and action tuning are disabled.",
        "archive_path": _relative_or_str(exp_dir),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(str(_resolve_path(args.config)))
    cfg.setdefault("execution", {})
    cfg.setdefault("objective", {})
    if args.timeout_minutes is not None:
        cfg["execution"]["completion_timeout_minutes"] = int(args.timeout_minutes)

    archive_root = _resolve_path(args.archive_root)
    exp_dir = archive_root / args.method_group / args.experiment_id
    existing_payload = []
    if exp_dir.exists():
        existing_payload = [
            path
            for path in exp_dir.iterdir()
            if path.name != "replay_baseline_launcher.log"
        ]
    if exp_dir.exists() and existing_payload and not args.overwrite:
        raise FileExistsError(f"experiment directory already exists; use --overwrite: {exp_dir}")
    if exp_dir.exists() and existing_payload and args.overwrite:
        resolved_exp = exp_dir.resolve()
        resolved_group = (archive_root / args.method_group).resolve()
        if resolved_group in resolved_exp.parents:
            shutil.rmtree(str(exp_dir))
    exp_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(args)
    selected = _select_candidates(candidates, args.selection, int(args.top_k), int(args.seed))
    if not selected:
        raise ValueError("no valid historical action sequence was selected")
    repeats = _allocate_repeats(
        selected,
        int(args.episodes),
        args.repeats_per_sequence,
        args.sequence_allocation,
        int(args.seed),
    )

    selected_export = []
    for item, repeat_count in zip(selected, repeats):
        export_item = {key: value for key, value in item.items() if key != "actions"}
        export_item["actions"] = item["actions"]
        export_item["repeats"] = repeat_count
        selected_export.append(export_item)
    _write_json(exp_dir / "selected_sequences.json", {"sequences": selected_export})

    manifest = _build_manifest(args, exp_dir, selected)
    _write_json(exp_dir / _MANIFEST_NAME, manifest)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_eval_dir = exp_dir / "final_eval" / f"eval_{timestamp}"
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_dir": str(exp_dir),
        "output_dir": str(final_eval_dir),
        "experiment_id": args.experiment_id,
        "method": "Historical Action Replay",
        "experiment_type": "historical_replay_baseline",
        "map_key": args.map_key,
        "map_id": args.map_id,
        "kg_file": args.kg_file or "",
        "data_dir": args.data_dir or "",
        "episodes_per_repeat": int(args.episodes),
        "requested_repeats": 1,
        "selected_sequences": len(selected),
        "action_tuning_enabled": False,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    _write_json(exp_dir / "replay_eval_summary.json", {**metadata, "sequences": [], "aggregate": {}})
    _write_json(final_eval_dir / "final_eval_summary.json", {**metadata, "sequences": [], "aggregate": {}})

    if args.dry_run:
        result = {**metadata, "status": "dry_run", "selected_sequences_detail": selected_export}
        _write_json(exp_dir / "replay_eval_summary.json", result)
        _write_json(final_eval_dir / "final_eval_summary.json", result)
        return result

    batch_record = _run_replay_collector_eval(
        args,
        cfg,
        exp_dir,
        selected,
        repeats,
        final_eval_dir,
    )
    aggregate = _aggregate_single(batch_record["metrics"])

    result = {
        **metadata,
        "status": batch_record.get("status", "completed"),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sequences": selected_export,
        "collector_dir": batch_record.get("collector_dir"),
        "metrics": batch_record.get("metrics", {}),
        "aggregate": aggregate,
    }
    _write_json(exp_dir / "replay_eval_summary.json", result)
    _write_json(final_eval_dir / "final_eval_summary.json", result)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate historical action replay baseline")
    parser.add_argument("--experiment-id", required=True, help="Archive experiment id")
    parser.add_argument("--display-name", default="", help="Display name in manifest")
    parser.add_argument("--map-key", required=True, help="Map config key, e.g. sce-1")
    parser.add_argument("--map-id", required=True, help="Map id, e.g. MarineMicro_MvsM_4")
    parser.add_argument("--action-log", default="", help="Historical action_log.csv")
    parser.add_argument("--result-log", default="", help="Optional paired game_result.txt")
    parser.add_argument("--episodes-jsonl", default="", help="Episode jsonl source with frame action_code")
    parser.add_argument("--episodes-pkl", default="", help="Episode pkl source with frame action_code")
    parser.add_argument(
        "--selection",
        choices=["best_pool"],
        default="best_pool",
        help="Historical sequence selection strategy; fixed to best_pool",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of historical sequences to replay")
    parser.add_argument("--episodes", type=int, default=100, help="Total target episodes across selected sequences")
    parser.add_argument("--repeats-per-sequence", type=int, default=None, help="Override repeats for each selected sequence")
    parser.add_argument(
        "--sequence-allocation",
        choices=["random", "even"],
        default="random",
        help="Allocation of total episodes over the selected best sequence pool",
    )
    parser.add_argument("--seed", type=int, default=0, help="Selection seed for random mode")
    parser.add_argument("--archive-root", default=str(_DEFAULT_ARCHIVE_ROOT), help="all_data root")
    parser.add_argument("--method-group", default=_METHOD_GROUP, help="Archive method group")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG), help="Base learner config for timeouts/objective")
    parser.add_argument("--timeout-minutes", type=int, default=90, help="Timeout per selected sequence")
    parser.add_argument("--kg-name", default="", help="Scenario ETG label for manifest only")
    parser.add_argument("--kg-file", default="", help="Scenario ETG file for manifest only")
    parser.add_argument("--transitions", default="", help="Scenario transition file for manifest only")
    parser.add_argument("--data-dir", default="", help="Scenario data directory for BKTree/action resolution")
    parser.add_argument("--dataset-type", default="historical", help="Dataset type label")
    parser.add_argument("--replay-dataset-expansion", action="store_true", help="Mark source as replay-expanded")
    parser.add_argument("--primary-threshold", type=float, default=0.7, help="BKTree primary threshold")
    parser.add_argument("--secondary-threshold", type=float, default=0.5, help="BKTree secondary threshold")
    parser.add_argument("--fallback-action", default="action_ATK_nearest_weakest", help="Fallback after sequence exhaustion")
    parser.add_argument("--source-run", default="", help="Optional source run identifier")
    parser.add_argument("--notes", default="", help="Manifest notes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in existing experiment directory")
    parser.add_argument("--dry-run", action="store_true", help="Only select sequences and write manifest/selection")
    args = parser.parse_args()

    result = evaluate(args)
    print(json.dumps(result.get("aggregate", result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
