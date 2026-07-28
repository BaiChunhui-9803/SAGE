#!/usr/bin/env python
"""Build a teacher-guided ETG from PyMARL step-level trajectories.

This script is intentionally separate from ``build_from_collected.py`` so that
existing augmented ETGs and ongoing experiments are not affected.  It consumes a
teacher JSONL file, projects PyMARL micro actions to ETG script codes when
needed, optionally mixes filtered replay episodes, and writes all artifacts to
new teacher-specific directories.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import build_from_collected as bfc
from src import ROOT_DIR
from src.adapters.pymarl_action_projector import (
    ProjectionResult,
    extract_state,
    project_frame,
    summarize_projection_results,
)
from src.structure.BKTree_sc2 import BKTree, ClusterNode, get_max_cluster_id
from src.structure.custom_distance_sc2 import CustomDistance


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("teacher_guided_etg")


LARGE_MAP_TOKENS = ("MvsM_8", "8_mirror")


def _resolve_root_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value or "").strip())
    return cleaned.strip("_") or "teacher"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(data, dict):
                episodes.append(data)
    return episodes


def _deserialize_bktree_node(node_data: Optional[Mapping[str, Any]]) -> Optional[ClusterNode]:
    if node_data is None:
        return None
    node = ClusterNode(node_data["state"], node_data["cluster_id"])
    for dist_key, child_data in node_data.get("children", {}).items():
        try:
            dist_val: Any = int(dist_key)
        except Exception:
            dist_val = float(dist_key)
        child = _deserialize_bktree_node(child_data)
        if child is not None:
            node.children[dist_val] = child
    return node


def _load_bktree_file(path: Path, distance_index: int) -> BKTree:
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


class ReadonlyBKTreeMatcher:
    def __init__(self, bktree_dir: Path, primary_threshold: float, secondary_threshold: float):
        self.bktree_dir = bktree_dir
        self.primary_threshold = float(primary_threshold)
        self.secondary_threshold = float(secondary_threshold)
        self.primary = _load_bktree_file(bktree_dir / "primary_bktree.json", 0)
        self.secondary_cache: Dict[int, Optional[BKTree]] = {}
        self.state_id_map = _load_state_id_map(bktree_dir.parent)

    def _secondary(self, primary_id: int) -> Optional[BKTree]:
        if primary_id in self.secondary_cache:
            return self.secondary_cache[primary_id]
        path = self.bktree_dir / f"secondary_bktree_{int(primary_id)}.json"
        if not path.exists():
            self.secondary_cache[primary_id] = None
            return None
        self.secondary_cache[primary_id] = _load_bktree_file(path, 1)
        return self.secondary_cache[primary_id]

    def query(self, norm_state: Mapping[str, Any]) -> Dict[str, Any]:
        primary_id, primary_dist = self.primary.query_nearest(norm_state)
        if primary_id is None:
            return {"cluster": None, "rejected": True, "reason": "primary_query_failed"}
        secondary_tree = self._secondary(int(primary_id))
        if secondary_tree is None or secondary_tree.root is None:
            rejected = float(primary_dist) > self.primary_threshold
            return {
                "cluster": (int(primary_id), 1),
                "state_id": self.state_id_for_cluster((int(primary_id), 1)) if not rejected else None,
                "primary_distance": float(primary_dist),
                "secondary_distance": None,
                "rejected": rejected,
                "reason": "distance_over_threshold" if rejected else "secondary_missing",
            }
        secondary_id, secondary_dist = secondary_tree.query_nearest(norm_state)
        secondary_id = int(secondary_id) if secondary_id is not None else 1
        rejected = (
            float(primary_dist) > self.primary_threshold
            or float(secondary_dist) > self.secondary_threshold
        )
        return {
            "cluster": (int(primary_id), secondary_id),
            "state_id": self.state_id_for_cluster((int(primary_id), secondary_id)) if not rejected else None,
            "primary_distance": float(primary_dist),
            "secondary_distance": float(secondary_dist),
            "rejected": rejected,
            "reason": "distance_over_threshold" if rejected else "accepted",
        }

    def state_id_for_cluster(self, cluster: Tuple[int, int]) -> Optional[int]:
        return self.state_id_map.get((int(cluster[0]), int(cluster[1])))

    def state_key_for_cluster(self, cluster: Tuple[int, int], rejected: bool = False) -> Any:
        state_id = self.state_id_for_cluster(cluster)
        if rejected:
            return f"ood:{int(cluster[0])}:{int(cluster[1])}"
        return state_id if state_id is not None else f"{int(cluster[0])}:{int(cluster[1])}"


def _normalize_bktree_dir(raw: str) -> Path:
    bktree_dir = _resolve_root_path(raw)
    if (bktree_dir / "bktree" / "primary_bktree.json").exists():
        bktree_dir = bktree_dir / "bktree"
    if not (bktree_dir / "primary_bktree.json").exists():
        raise FileNotFoundError(f"primary_bktree.json not found under {bktree_dir}")
    return bktree_dir


def _frame_cluster(frame: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    cluster = frame.get("state_cluster", frame.get("cluster"))
    if isinstance(cluster, str):
        text = cluster.strip().strip("()[]")
        parts = [p.strip() for p in re.split(r"[,:\s]+", text) if p.strip()]
        if len(parts) >= 2:
            return (int(float(parts[0])), int(float(parts[1])))
    if isinstance(cluster, Sequence) and not isinstance(cluster, (str, bytes)) and len(cluster) >= 2:
        return (int(cluster[0]), int(cluster[1]))
    return None


def _hp_sum_from_state(state: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    if not state:
        return None
    units = state.get(key, []) or []
    try:
        return float(sum(float(unit[2]) for unit in units))
    except Exception:
        return None


def _frame_hp(frame: Mapping[str, Any], state: Optional[Mapping[str, Any]]) -> Tuple[float, float]:
    hp_my = frame.get("hp_my", frame.get("sum_health_agents"))
    hp_enemy = frame.get("hp_enemy", frame.get("sum_health_enemies"))
    if hp_my is None:
        hp_my = _hp_sum_from_state(state, "blue_army")
    if hp_enemy is None:
        hp_enemy = _hp_sum_from_state(state, "red_army")
    return float(hp_my or 0.0), float(hp_enemy or 0.0)


def _episode_final_score(ep: Mapping[str, Any]) -> Optional[float]:
    for key in ("final_score", "score", "avg_score"):
        if ep.get(key) is not None:
            try:
                return float(ep[key])
            except Exception:
                pass
    frames = ep.get("frames", []) or []
    if frames:
        frame = frames[-1]
        state = extract_state(frame) if isinstance(frame, Mapping) else None
        hp_my, hp_enemy = _frame_hp(frame, state)
        return hp_my - hp_enemy
    return None


def _episode_result(ep: Mapping[str, Any], final_score: Optional[float]) -> str:
    result = str(ep.get("result", ep.get("outcome", ""))).strip()
    if result:
        return result
    if final_score is None:
        return "Unknown"
    if final_score > 0:
        return "Win"
    if final_score < 0:
        return "Loss"
    return "Dogfall"


def _normalize_teacher_episode(
    episode: Mapping[str, Any],
    matcher: ReadonlyBKTreeMatcher,
    projection_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], List[ProjectionResult], Counter]:
    final_score = _episode_final_score(episode)
    result = _episode_result(episode, final_score)
    frames_out: List[Dict[str, Any]] = []
    projections: List[ProjectionResult] = []
    counters: Counter = Counter()

    for frame_idx, frame in enumerate(episode.get("frames", []) or []):
        if not isinstance(frame, Mapping):
            continue
        norm_state = extract_state(frame)
        cluster = _frame_cluster(frame)
        match: Dict[str, Any] = {}
        if cluster is None and norm_state is not None:
            match = matcher.query(norm_state)
            cluster = match.get("cluster")
            if match.get("rejected"):
                counters["bktree_rejected"] += 1
        if cluster is None:
            counters["missing_cluster"] += 1
            continue

        action_code = frame.get("projected_action_code") or frame.get("action_code")
        projection: Optional[ProjectionResult] = None
        if not action_code:
            projection = project_frame(frame)
            action_code = projection.action_code
        else:
            if frame.get("pymarl_actions") is not None or frame.get("actions") is not None or frame.get("joint_action") is not None:
                projection = project_frame(frame)
            else:
                projection = ProjectionResult(
                    action_code=str(action_code),
                    action_letter=str(action_code)[1] if len(str(action_code)) >= 2 else "k",
                    cluster_digit=int(str(action_code)[0]) if str(action_code)[:1].isdigit() else 4,
                    confidence=float(frame.get("projection_confidence", 1.0)),
                    reason=str(frame.get("projection_reason", "preprojected_action_code")),
                    features={},
                    micro_action_signature=frame.get("micro_action_signature", {}),
                )
        if projection.confidence < projection_threshold:
            counters["low_confidence_filtered"] += 1
            continue

        hp_my, hp_enemy = _frame_hp(frame, norm_state)
        state_id = matcher.state_id_for_cluster((int(cluster[0]), int(cluster[1])))
        rejected = bool(match.get("rejected")) if match else False
        frame_out = {
            "state_cluster": [int(cluster[0]), int(cluster[1])],
            "state_key": matcher.state_key_for_cluster((int(cluster[0]), int(cluster[1])), rejected=rejected),
            "state_id": None if rejected else state_id,
            "nid": None if rejected else state_id,
            "action_code": str(action_code),
            "hp_my": hp_my,
            "hp_enemy": hp_enemy,
            "projection_confidence": projection.confidence,
            "projection_reason": projection.reason,
            "projected_action_code": str(action_code),
            "micro_action_signature": projection.micro_action_signature,
            "teacher_frame_idx": int(frame.get("frame_idx", frame_idx)),
        }
        if norm_state is not None:
            frame_out["norm_state"] = norm_state
        if match:
            frame_out["bktree_match"] = match
        frames_out.append(frame_out)
        projections.append(projection)

    if not frames_out:
        counters["empty_episode"] += 1
        return None, projections, counters

    return (
        {
            "episode_id": episode.get("episode_id", episode.get("episode", None)),
            "result": result,
            "final_score": final_score,
            "state_key_sequence": [
                frame.get("state_key")
                for frame in frames_out
                if frame.get("state_key") is not None
            ],
            "state_id_sequence": [
                frame.get("state_id")
                for frame in frames_out
                if frame.get("state_id") is not None
            ],
            "frames": frames_out,
            "teacher_meta": episode.get("teacher_meta", {}),
        },
        projections,
        counters,
    )


def _load_teacher_episodes(
    teacher_jsonl: Path,
    matcher: ReadonlyBKTreeMatcher,
    projection_threshold: float,
    max_episodes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_episodes = _read_jsonl(teacher_jsonl)
    if max_episodes > 0:
        raw_episodes = raw_episodes[:max_episodes]

    episodes: List[Dict[str, Any]] = []
    projections: List[ProjectionResult] = []
    counters: Counter = Counter()
    for raw_ep in raw_episodes:
        ep, ep_projections, ep_counters = _normalize_teacher_episode(
            raw_ep,
            matcher,
            projection_threshold,
        )
        projections.extend(ep_projections)
        counters.update(ep_counters)
        if ep is not None:
            episodes.append(ep)

    scores = [score for score in (_episode_final_score(ep) for ep in episodes) if score is not None]
    summary = {
        "input_episodes": len(raw_episodes),
        "valid_episodes": len(episodes),
        "projection": summarize_projection_results(projections),
        "counters": dict(counters),
        "final_score_mean": sum(scores) / len(scores) if scores else None,
        "final_score_min": min(scores) if scores else None,
        "final_score_max": max(scores) if scores else None,
    }
    return episodes, summary


def _filter_base_episodes(
    episodes: List[Dict[str, Any]],
    merge_mode: str,
    top_quantile: float,
    min_final_score: Optional[float],
    wins_only: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if merge_mode == "teacher_only":
        return [], {"selected": 0, "source": len(episodes)}

    selected = list(episodes)
    if wins_only:
        selected = [
            ep for ep in selected if str(ep.get("result", "")).lower().startswith("win")
        ]
    if min_final_score is not None:
        selected = [
            ep
            for ep in selected
            if (_episode_final_score(ep) is not None and _episode_final_score(ep) >= min_final_score)
        ]
    if merge_mode == "teacher_plus_top_replay" and selected:
        scored = [(ep, _episode_final_score(ep)) for ep in selected]
        scored = [(ep, score) for ep, score in scored if score is not None]
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        keep = max(1, int(len(scored) * max(0.0, min(1.0, float(top_quantile)))))
        selected = [ep for ep, _ in scored[:keep]]

    scores = [score for score in (_episode_final_score(ep) for ep in selected) if score is not None]
    return selected, {
        "source": len(episodes),
        "selected": len(selected),
        "final_score_mean": sum(scores) / len(scores) if scores else None,
        "final_score_min": min(scores) if scores else None,
        "final_score_max": max(scores) if scores else None,
    }


def _copy_bktree_to_data_dir(bktree_dir: Path, data_dir: Path) -> int:
    dest = data_dir / "bktree"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in [bktree_dir / "primary_bktree.json", *sorted(bktree_dir.glob("secondary_bktree_*.json"))]:
        if src.exists():
            shutil.copy2(src, dest / src.name)
            count += 1
    return count


def _copy_runtime_sparse_index(kg_dir: Path, map_id: str, data_id: str) -> Optional[str]:
    candidates = [kg_dir / "sparse_neighbors.pkl", kg_dir / "npy" / "sparse_neighbors.pkl"]
    src = next((path for path in candidates if path.exists()), None)
    if src is None:
        return None
    dest = ROOT_DIR / "cache" / "npy" / f"state_sparse_neighbors_{map_id}_{data_id}.pkl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)


def _write_prepared_teacher_jsonl(path: Path, episodes: Iterable[Dict[str, Any]]) -> None:
    with open(str(path), "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def _write_prepared_pickle(path: Path, episodes: Sequence[Dict[str, Any]]) -> None:
    with open(str(path), "wb") as f:
        for ep in episodes:
            pickle.dump(ep, f)


def _relative_to_cache_kg(path: Path) -> str:
    kg_root = ROOT_DIR / "cache" / "knowledge_graph"
    try:
        return str(path.relative_to(kg_root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _register_catalog_entry(
    name: str,
    map_id: str,
    data_id: str,
    kg_dir: Path,
    data_dir: Path,
    description: str,
    overwrite: bool,
) -> None:
    catalog_path = ROOT_DIR / "configs" / "kg_catalog.yaml"
    if catalog_path.exists():
        with open(str(catalog_path), "r", encoding="utf-8") as f:
            catalog = yaml.safe_load(f) or {}
    else:
        catalog = {}
    entries = catalog.setdefault("knowledge_graphs", [])
    entry = {
        "name": name,
        "file": f"{_relative_to_cache_kg(kg_dir)}/kg_simple.pkl",
        "transitions": f"{_relative_to_cache_kg(kg_dir)}/kg_simple_transitions.pkl",
        "data_dir": str(data_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
        "type": "teacher_guided",
        "context_window": 0,
        "map_id": map_id,
        "data_id": data_id,
        "description": description,
    }
    existing_idx = next((idx for idx, item in enumerate(entries) if item.get("name") == name), None)
    if existing_idx is not None:
        if not overwrite:
            logger.info("Catalog entry already exists and --overwrite is false: %s", name)
            return
        entries[existing_idx] = entry
    else:
        entries.insert(0, entry)
    with open(str(catalog_path), "w", encoding="utf-8") as f:
        yaml.safe_dump(catalog, f, allow_unicode=True, sort_keys=False)
    logger.info("Registered catalog entry: %s", name)


def _archive_teacher_build(
    experiment_id: str,
    map_key: str,
    map_id: str,
    data_id: str,
    teacher_method: str,
    source_run_id: str,
    checkpoint_path: str,
    teacher_episodes: int,
    kg_name: str,
    kg_dir: Path,
    data_dir: Path,
    bktree_dir: Path,
    summary: Dict[str, Any],
    overwrite: bool,
) -> Path:
    exp_dir = ROOT_DIR / "output" / "learner_results" / "all_data" / "Teacher-guided-ETG" / experiment_id
    if exp_dir.exists() and not overwrite:
        raise FileExistsError(f"archive directory already exists: {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": experiment_id,
        "display_name": f"{map_key} {teacher_method.upper()} Teacher-guided ETG",
        "map_key": map_key,
        "map_id": map_id,
        "experiment_type": "teacher_guided_etg_build",
        "method": f"Teacher-guided ETG ({teacher_method.upper()} projected-script)",
        "kg_name": kg_name,
        "kg_file": f"{_relative_to_cache_kg(kg_dir)}/kg_simple.pkl",
        "transitions": f"{_relative_to_cache_kg(kg_dir)}/kg_simple_transitions.pkl",
        "data_dir": str(data_dir.relative_to(ROOT_DIR)).replace("\\", "/"),
        "dataset_type": "teacher_guided",
        "replay_dataset_expansion": True,
        "teacher_policy": {
            "algorithm": teacher_method,
            "source_run_id": source_run_id,
            "checkpoint_path": checkpoint_path,
            "episodes": teacher_episodes,
        },
        "action_adaptation": {
            "mode": "projected_script",
            "micro_signature_saved": True,
        },
        "bktree": {
            "primary_threshold": summary.get("primary_threshold"),
            "secondary_threshold": summary.get("secondary_threshold"),
            "path": str((data_dir / "bktree").relative_to(ROOT_DIR)).replace("\\", "/"),
            "source_path": str(bktree_dir),
        },
        "source_run": source_run_id or "-",
        "notes": "Teacher-guided ETG build record; not an Optuna learner run.",
    }
    with open(str(exp_dir / "experiment_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(str(exp_dir / "teacher_build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return exp_dir


def build_teacher_guided_etg(args: argparse.Namespace) -> Dict[str, Any]:
    teacher_jsonl = _resolve_root_path(args.teacher_jsonl)
    if not teacher_jsonl.exists():
        raise FileNotFoundError(f"teacher JSONL not found: {teacher_jsonl}")

    bktree_dir = _normalize_bktree_dir(args.bktree_dir)
    matcher = ReadonlyBKTreeMatcher(
        bktree_dir,
        primary_threshold=args.primary_threshold,
        secondary_threshold=args.secondary_threshold,
    )
    teacher_episodes, teacher_summary = _load_teacher_episodes(
        teacher_jsonl,
        matcher,
        projection_threshold=args.projection_threshold,
        max_episodes=args.max_teacher_episodes,
    )
    if not teacher_episodes:
        raise RuntimeError("No valid teacher episodes after projection and BKTree matching.")

    base_episodes: List[Dict[str, Any]] = []
    base_summary: Dict[str, Any] = {"source": 0, "selected": 0}
    if args.base_replay_input and args.merge_mode != "teacher_only":
        replay_data = bfc.load_collected_data(str(_resolve_root_path(args.base_replay_input)))
        base_episodes, base_summary = _filter_base_episodes(
            replay_data["episodes"],
            args.merge_mode,
            args.top_replay_quantile,
            args.min_final_score,
            args.wins_only,
        )

    weighted_teacher = list(teacher_episodes) * max(1, int(args.teacher_weight))
    all_episodes = list(base_episodes) + weighted_teacher
    if args.preview_only:
        return {
            "preview_only": True,
            "teacher": teacher_summary,
            "base_replay": base_summary,
            "total_build_episodes": len(all_episodes),
        }

    output_dir = _resolve_root_path(args.output_dir) if args.output_dir else (
        ROOT_DIR
        / "cache"
        / "knowledge_graph"
        / f"{args.map_id}_{_sanitize_name(args.teacher_method)}_teacher_projected"
    )
    if output_dir.exists() and not args.overwrite and (output_dir / "kg_simple.pkl").exists():
        raise FileExistsError(f"KG output already exists; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT_DIR / "data" / args.map_id / args.data_id
    if data_dir.exists() and not args.overwrite and any(data_dir.iterdir()):
        raise FileExistsError(f"data_dir already exists; use --overwrite: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = data_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    _copy_bktree_to_data_dir(bktree_dir, data_dir)
    _write_prepared_teacher_jsonl(data_dir / "teacher_episodes_projected.jsonl", teacher_episodes)
    _write_prepared_pickle(prepared_dir / "episodes_teacher_guided.pkl", all_episodes)

    logger.info("Building teacher-guided ETG from %s episodes", len(all_episodes))
    state_to_node, node_to_state = bfc.build_cluster_mapping(all_episodes, str(bktree_dir))
    bfc.save_state_node_txt(state_to_node, output_dir)
    graph_dir = data_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "state_node.txt", graph_dir / "state_node.txt")

    state_episodes, action_episodes, reward_episodes, outcome_episodes = bfc.build_episodes_arrays(
        all_episodes,
        state_to_node,
    )
    transitions = bfc.build_transitions(
        state_episodes=state_episodes,
        action_episodes=action_episodes,
        outcome_episodes=outcome_episodes,
        output_dir=output_dir,
        unique_states=set(state_to_node.values()),
    )
    kg = bfc.build_knowledge_graph(
        state_episodes=state_episodes,
        action_episodes=action_episodes,
        reward_episodes=reward_episodes,
        outcome_episodes=outcome_episodes,
        output_dir=output_dir,
    )
    if args.validate:
        bfc.validate_knowledge_graph(kg)

    dense_ready = False
    skip_dense = args.skip_distance_matrix or (
        args.auto_skip_large_distance_matrix and any(token in args.map_id for token in LARGE_MAP_TOKENS)
    )
    if skip_dense:
        logger.info("Skipping dense distance matrix for teacher-guided ETG.")
    else:
        dense_ready = bfc.compute_distance_matrix(
            str(bktree_dir),
            output_dir,
            node_to_state,
            max_states=args.max_distance_matrix_states,
        )
    if not dense_ready and not args.skip_sparse_neighbors:
        bfc.compute_sparse_neighbor_index(
            str(bktree_dir),
            output_dir,
            node_to_state,
            bfc.collect_state_visits(kg),
            top_k=args.sparse_top_k,
            max_source_states=args.sparse_max_source_states,
            max_candidates_per_primary=args.sparse_max_candidates_per_primary,
        )
    sparse_runtime_path = _copy_runtime_sparse_index(output_dir, args.map_id, args.data_id)

    kg_name = args.kg_name or f"{args.map_id} - Teacher {args.teacher_method.upper()} Projected"
    summary = {
        "map_id": args.map_id,
        "map_key": args.map_key,
        "data_id": args.data_id,
        "teacher_method": args.teacher_method,
        "source_run_id": args.source_run_id,
        "checkpoint_path": args.checkpoint_path,
        "teacher_jsonl": str(teacher_jsonl),
        "bktree_dir": str(bktree_dir),
        "primary_threshold": args.primary_threshold,
        "secondary_threshold": args.secondary_threshold,
        "projection_threshold": args.projection_threshold,
        "merge_mode": args.merge_mode,
        "teacher_weight": args.teacher_weight,
        "teacher": teacher_summary,
        "base_replay": base_summary,
        "build_episodes": len(all_episodes),
        "unique_states": len(state_to_node),
        "transitions": len(transitions),
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "sparse_runtime_path": sparse_runtime_path,
    }
    with open(str(output_dir / "teacher_build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(str(data_dir / "teacher_dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.register_catalog:
        _register_catalog_entry(
            kg_name,
            args.map_id,
            args.data_id,
            output_dir,
            data_dir,
            f"Teacher-guided ETG ({args.teacher_method}, projected_script)",
            overwrite=args.overwrite,
        )
    if args.archive_manifest:
        experiment_id = args.experiment_id or f"{_sanitize_name(args.map_key)}_{_sanitize_name(args.teacher_method)}_teacher_etg"
        archive_dir = _archive_teacher_build(
            experiment_id,
            args.map_key,
            args.map_id,
            args.data_id,
            args.teacher_method,
            args.source_run_id,
            args.checkpoint_path,
            len(teacher_episodes),
            kg_name,
            output_dir,
            data_dir,
            bktree_dir,
            summary,
            overwrite=args.overwrite,
        )
        summary["archive_dir"] = str(archive_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher-guided ETG from PyMARL trajectories")
    parser.add_argument("--teacher-jsonl", required=True, help="PyMARL step-level teacher_episodes.jsonl")
    parser.add_argument("--map-id", required=True, help="Target map id")
    parser.add_argument("--map-key", default="", help="Scenario key, e.g. sce-3")
    parser.add_argument("--teacher-method", default="qmix", help="Teacher algorithm name")
    parser.add_argument("--source-run-id", default="", help="Source PyMARL run id")
    parser.add_argument("--checkpoint-path", default="", help="Source checkpoint path")
    parser.add_argument("--bktree-dir", required=True, help="Readonly BKTree directory or data_dir containing bktree/")
    parser.add_argument("--base-replay-input", default="", help="Optional original collected replay directory/pkl")
    parser.add_argument(
        "--merge-mode",
        choices=["teacher_only", "teacher_plus_replay", "teacher_plus_top_replay"],
        default="teacher_plus_top_replay",
    )
    parser.add_argument("--top-replay-quantile", type=float, default=0.25)
    parser.add_argument("--min-final-score", type=float, default=None)
    parser.add_argument("--wins-only", action="store_true")
    parser.add_argument("--teacher-weight", type=int, default=2)
    parser.add_argument("--projection-threshold", type=float, default=0.0)
    parser.add_argument("--max-teacher-episodes", type=int, default=0, help="0 means all")
    parser.add_argument("--data-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--kg-name", default="")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--primary-threshold", type=float, default=1.0)
    parser.add_argument("--secondary-threshold", type=float, default=0.5)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--register-catalog", action="store_true")
    parser.add_argument("--archive-manifest", action="store_true")
    parser.add_argument("--skip-distance-matrix", action="store_true")
    parser.add_argument("--auto-skip-large-distance-matrix", action="store_true", default=True)
    parser.add_argument("--max-distance-matrix-states", type=int, default=30000)
    parser.add_argument("--skip-sparse-neighbors", action="store_true")
    parser.add_argument("--sparse-top-k", type=int, default=32)
    parser.add_argument("--sparse-max-source-states", type=int, default=30000)
    parser.add_argument("--sparse-max-candidates-per-primary", type=int, default=512)
    args = parser.parse_args()
    if not args.map_key:
        args.map_key = "sce-3m" if args.map_id.endswith("_mirror") else "sce-3"
    if not args.data_id:
        args.data_id = f"teacher_{_sanitize_name(args.teacher_method)}_projected_1"
    return args


def main() -> None:
    args = parse_args()
    summary = build_teacher_guided_etg(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
