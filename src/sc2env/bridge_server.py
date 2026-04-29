"""
Bridge Server — FastAPI 桥接服务

在独立进程中运行，通过 GameBridge (multiprocessing.Queue) 与 SC2 游戏进程通信，
对外提供 REST API 供 Streamlit Web 前端调用。

端点:
    GET  /game/status           → 游戏运行状态
    GET  /game/observation      → 当前游戏观测
    GET  /game/logs             → 合并日志 (game事件 + API请求)
    POST /game/action           → 发送动作指令
    POST /game/fallback         → 设置默认回退策略
    POST /game/control          → 控制游戏 (pause/resume/stop/step)
    POST /game/start            → 启动游戏子进程
    GET  /game/process          → 游戏子进程状态
    POST /game/stop-process     → 终止游戏子进程 (保留API)
    GET  /game/events           → SSE 游戏事件流
    GET  /game/actions          → 可用动作列表
"""

from __future__ import annotations

import collections
import multiprocessing
import os
import sys
import json
import logging
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

import threading

from src.sc2env.bridge import GameBridge
from src import ROOT_DIR


def _game_worker(bridge, map_key, run_name, fallback_action):
    from absl import flags as absl_flags

    if not absl_flags.FLAGS.is_parsed():
        absl_flags.FLAGS(["run_live_game.py"])
    from src.sc2env.run_game import run_game

    run_game(
        map_key=map_key,
        run_name=run_name,
        bridge=bridge,
        agent_type="kg_guided",
        fallback_action=fallback_action,
    )


logger = logging.getLogger(__name__)

KG_DIR = ROOT_DIR / "cache" / "knowledge_graph"

_instance: Optional["BridgeServer"] = None

_MAX_LOG_BUFFER = 500


class ActionRequest(BaseModel):
    action: str


class ControlRequest(BaseModel):
    command: str


class FallbackRequest(BaseModel):
    action: str


class LogEntry:
    __slots__ = ("ts", "level", "source", "message", "seq")

    def __init__(self, level: str, source: str, message: str):
        self.ts = datetime.now().strftime("%H:%M:%S")
        self.level = level
        self.source = source
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "seq": self.seq,
        }


_cluster_to_nid_cache: Dict[Tuple[int, int], Optional[int]] = {}


def _parse_state_id(obs: Dict[str, Any]) -> Optional[int]:
    state_cluster = obs.get("state_cluster_str", "0")
    try:
        if isinstance(state_cluster, str):
            state_cluster = eval(state_cluster)
        if isinstance(state_cluster, (list, tuple)) and len(state_cluster) >= 2:
            key = (int(state_cluster[0]), int(state_cluster[1]))
            if _instance and _instance._state_id_map:
                nid = _instance._state_id_map.get(key)
                if nid is not None:
                    return nid
                cached = _cluster_to_nid_cache.get(key)
                if cached is not None:
                    return cached
                if key in _cluster_to_nid_cache:
                    return None
                norm_state = obs.get("norm_state")
                if norm_state and _instance._nid_norm_states:
                    from src.structure.custom_distance_sc2 import DistributionDistance

                    best_nid = None
                    best_dist = float("inf")
                    best_hp = float("inf")
                    for nid, stored in _instance._nid_norm_states.items():
                        try:
                            dd = DistributionDistance(norm_state, stored)
                            d, h = dd()
                            if d < best_dist or (d == best_dist and h < best_hp):
                                best_dist = d
                                best_hp = h
                                best_nid = nid
                        except Exception:
                            pass
                    if best_nid is not None:
                        _cluster_to_nid_cache[key] = best_nid
                        return best_nid
                _cluster_to_nid_cache[key] = None
                return None
            return int(state_cluster[0])
        elif isinstance(state_cluster, (list, tuple)) and len(state_cluster) == 1:
            return int(state_cluster[0])
        elif isinstance(state_cluster, (int, float)):
            return int(state_cluster)
    except Exception:
        pass
    return None


def _enrich_event(event: Dict[str, Any]) -> Dict[str, Any]:
    if event.get("level") == "debug" and "state_cluster" in event:
        p, s = event["state_cluster"]
        nid = None
        if _instance and _instance._state_id_map:
            nid = _instance._state_id_map.get((int(p), int(s)))
        msg = event.get("message", "")
        if nid is not None:
            event["message"] = f"nid={nid} {msg}"
        else:
            event["message"] = f"nid=? {msg}"
        del event["state_cluster"]
    return event


def _states_match(
    actual: int, expected: int, dist_matrix=None, threshold: float = 0.2
) -> bool:
    if actual == expected:
        return True
    if dist_matrix is not None:
        try:
            d = float(dist_matrix[actual, expected])
            if not np.isnan(d) and d < threshold:
                return True
        except (IndexError, TypeError, KeyError):
            pass
    return False


def _safe_dist(dm, shape, a: int, b: int) -> Optional[float]:
    if dm is None or not isinstance(a, int) or not isinstance(b, int):
        return None
    try:
        if 0 <= a < shape[0] and 0 <= b < shape[1]:
            d = float(dm[a, b])
            return None if np.isnan(d) else d
    except (IndexError, TypeError):
        pass
    return None


def _compute_deviations(events: List[Dict]) -> List[Dict]:
    dm = _instance._dist_matrix if _instance else None
    dm_shape = dm.shape if dm is not None else (0, 0)
    plan_id_counter = 0
    current_planned_states: List[int] = []
    plan_step_idx = 0
    current_plan_start_idx = 0
    is_multi = False
    for i, ev in enumerate(events):
        plan = ev.get("plan")
        et = ev.get("event_type", "")
        actual_state = ev.get("state_id")
        if not isinstance(actual_state, int):
            try:
                actual_state = int(actual_state)
            except (TypeError, ValueError):
                actual_state = None
        deviation = None
        planned_state = None
        prev_prediction_error = None
        if plan is not None:
            planned_states = plan.get("planned_states") or plan.get("action_plan", [])
            planned_states = [
                s for s in planned_states if isinstance(s, (int, float, np.integer))
            ]
            if planned_states:
                is_multi = plan.get("mode") == "multi_step"
                current_planned_states = [int(s) for s in planned_states]
                plan_step_idx = 0
                plan_id_counter += 1
                current_plan_start_idx = i
                if (
                    is_multi
                    and len(current_planned_states) > 1
                    and actual_state is not None
                ):
                    planned_state = current_planned_states[0]
                    deviation = _safe_dist(dm, dm_shape, actual_state, planned_state)
            else:
                current_planned_states = []
                plan_step_idx = 0
                plan_id_counter += 1
                current_plan_start_idx = i
        if et == "kg_follow" and current_planned_states:
            idx_in_plan = plan_step_idx
            if idx_in_plan < len(current_planned_states) and actual_state is not None:
                planned_state = current_planned_states[idx_in_plan]
                deviation = _safe_dist(dm, dm_shape, actual_state, planned_state)
            plan_step_idx += 1
        elif et == "kg_plan" and plan is None and current_planned_states:
            plan_step_idx += 1
        if i > 0 and actual_state is not None and prev_prediction_error is None:
            prev_ev = events[i - 1]
            prev_plan = prev_ev.get("plan")
            if prev_plan is not None:
                beam_paths = prev_plan.get("beam_paths")
                if beam_paths and len(beam_paths) > 0:
                    chosen = beam_paths[0]
                    steps = chosen.get("steps", [])
                    for step in steps[1:]:
                        if isinstance(step.get("state"), (int, float, np.integer)):
                            predicted = int(step["state"])
                            prev_prediction_error = _safe_dist(
                                dm, dm_shape, actual_state, predicted
                            )
                            break
        ev["deviation"] = float(deviation) if deviation is not None else None
        ev["planned_state"] = planned_state
        ev["plan_id"] = (
            plan_id_counter
            if plan is not None or current_planned_states
            else (plan_id_counter if et == "kg_follow" else 0)
        )
        ev["plan_step_idx"] = (
            plan_step_idx if (plan is not None or et == "kg_follow") else 0
        )
        ev["prediction_error"] = (
            float(prev_prediction_error) if prev_prediction_error is not None else None
        )
    return events


def _lookup_dist_map(dist_map: Dict, a, b) -> Optional[float]:
    if not dist_map or a is None or b is None:
        return None
    k1 = f"{a},{b}"
    k2 = f"{b},{a}"
    if k1 in dist_map:
        return dist_map[k1]
    if k2 in dist_map:
        return dist_map[k2]
    return None


def _build_fork_tree_data(events: List[Dict], dist_map: Dict) -> Optional[Dict]:
    from sklearn.manifold import MDS as _MDS

    n = min(len(events), 40)
    nodes = {}
    edges = []
    edge_set = set()

    for i in range(n):
        ev = events[i]
        sid = ev.get("state_id")
        if not isinstance(sid, int):
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                sid = None
        nid = f"A{i}"
        nodes[nid] = {
            "id": nid,
            "state": sid,
            "label": f"S{sid}" if sid is not None else "S?",
            "type": "actual",
            "frame": i,
            "et": ev.get("event_type", "no_action"),
            "planId": ev.get("plan_id", 0),
        }
        if i > 0:
            prev_sid = events[i - 1].get("state_id")
            if not isinstance(prev_sid, int):
                try:
                    prev_sid = int(prev_sid)
                except (TypeError, ValueError):
                    prev_sid = None
            d = _lookup_dist_map(dist_map, prev_sid, sid)
            ek = f"A{i - 1}->{nid}"
            if ek not in edge_set:
                edge_set.add(ek)
                edges.append(
                    {
                        "from": f"A{i - 1}",
                        "to": nid,
                        "type": "actual",
                        "et": ev.get("event_type", "no_action"),
                        "dist": d,
                    }
                )

        plan = ev.get("plan")
        if not plan or not plan.get("beam_paths"):
            continue
        bp_list = plan["beam_paths"]
        for pi, bp in enumerate(bp_list):
            steps = bp.get("steps", [])
            if not steps or len(steps) < 2:
                continue
            is_chosen = bool(bp.get("chosen", False))
            parent_id = f"A{i}"
            parent_state = sid
            for si in range(1, len(steps)):
                step = steps[si]
                step_state = step.get("state")
                if step_state is None:
                    continue
                try:
                    step_state = int(step_state)
                except (TypeError, ValueError):
                    continue
                beam_nid = f"P{i}_{si}_{step_state}"
                if beam_nid not in nodes:
                    nodes[beam_nid] = {
                        "id": beam_nid,
                        "state": step_state,
                        "label": f"S{step_state}",
                        "type": "beam",
                        "frame": i,
                        "stepIdx": si,
                    }
                nd = nodes[beam_nid]
                if "pathIndices" not in nd:
                    nd["pathIndices"] = []
                if pi not in nd["pathIndices"]:
                    nd["pathIndices"].append(pi)
                if is_chosen:
                    nd["chosen"] = True
                d = _lookup_dist_map(dist_map, parent_state, step_state)
                ek = f"{parent_id}->{beam_nid}"
                if ek not in edge_set:
                    edge_set.add(ek)
                    edges.append(
                        {
                            "from": parent_id,
                            "to": beam_nid,
                            "type": "beam",
                            "dist": d,
                            "chosen": is_chosen,
                            "pathIdx": pi,
                        }
                    )
                parent_id = beam_nid
                parent_state = step_state

    if len(nodes) < 2:
        return None

    ids = sorted(nodes.keys())
    N = len(ids)
    idx_map = {nid: idx for idx, nid in enumerate(ids)}
    D = np.full((N, N), 1.0)
    for i in range(N):
        D[i, i] = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            s1 = nodes[ids[i]].get("state")
            s2 = nodes[ids[j]].get("state")
            d = _lookup_dist_map(dist_map, s1, s2)
            if d is not None:
                v = max(d, 0.001)
            else:
                v = 1.0
            D[i, j] = v
            D[j, i] = v

    try:
        mds = _MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress=False,
            max_iter=300,
        )
        coords_2d = mds.fit_transform(D)
    except Exception:
        return {"nodes": list(nodes.values()), "edges": edges, "coords": {}}

    mins = coords_2d.min(axis=0)
    maxs = coords_2d.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    coords_2d = (coords_2d - mins) / ranges

    pad_l, pad_t = 60, 50
    draw_w, draw_h = 900, 450
    result_coords = {}
    for i, nid in enumerate(ids):
        result_coords[nid] = [
            round(pad_l + float(coords_2d[i, 0]) * draw_w, 1),
            round(pad_t + float(coords_2d[i, 1]) * draw_h, 1),
        ]

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "coords": result_coords,
    }


class BridgeServer:
    def __init__(self, bridge: GameBridge):
        self.bridge = bridge
        self.kg = None
        self.transitions = None
        self.kg_loaded = False
        self.kg_file: Optional[str] = None
        self._state_id_map: Dict[Tuple[int, int], int] = {}
        self._local_status: Dict[str, Any] = {
            "running": False,
            "paused": False,
            "frame": 0,
            "episode": 0,
            "map_key": "",
            "mode": "idle",
        }
        self._log_buffer: collections.deque = collections.deque(maxlen=_MAX_LOG_BUFFER)
        self._log_seq: int = 0

        self._dist_matrix: Optional[np.ndarray] = None
        self._nid_norm_states: Dict[int, dict] = {}

        self._history_store: Dict[int, List[Dict[str, Any]]] = {}
        self._history_meta: Dict[int, Dict[str, Any]] = {}
        self._history_max_episodes: int = 100
        self._ep_detail_cache: Dict[int, Dict[str, Any]] = {}
        self._total_completed: int = 0

    def add_log(self, level: str, source: str, message: str) -> None:
        self._log_seq += 1
        entry = LogEntry(level, source, message)
        entry.seq = self._log_seq
        self._log_buffer.append(entry)

    def drain_logs(self, after_seq: int = 0) -> List[Dict[str, Any]]:
        for event in self.bridge.get_events():
            event = _enrich_event(event)
            level = event.get("level", "info")
            source = event.get("source", "game")
            message = event.get("message", str(event))
            if isinstance(message, dict):
                message = json.dumps(message, default=str)
            self.add_log(level, source, message)

        result = []
        for entry in self._log_buffer:
            if hasattr(entry, "seq") and entry.seq > after_seq:
                result.append(entry.to_dict())
        return result

    def drain_all_logs(self) -> List[Dict[str, Any]]:
        for event in self.bridge.get_events():
            event = _enrich_event(event)
            level = event.get("level", "info")
            source = event.get("source", "game")
            message = event.get("message", str(event))
            if isinstance(message, dict):
                message = json.dumps(message, default=str)
            self.add_log(level, source, message)

        return [entry.to_dict() for entry in self._log_buffer]

    def clear_logs(self) -> None:
        self._log_buffer.clear()

    def _to_native(self, val):
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: self._to_native(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [self._to_native(v) for v in val]
        return val

    def _refresh_status(self) -> Dict[str, Any]:
        for upd in self.bridge.drain_status_updates():
            self._local_status.update(upd)
        result = {k: self._to_native(v) for k, v in self._local_status.items()}
        result["kg_loaded"] = self.kg_loaded
        result["kg_file"] = self.kg_file
        return result

    def load_kg(self, kg_file: str, data_dir: Optional[str] = None) -> bool:
        from src.decision.knowledge_graph import DecisionKnowledgeGraph

        path = KG_DIR / kg_file
        if not path.exists():
            logger.error(f"KG file not found: {path}")
            return False

        try:
            self.kg = DecisionKnowledgeGraph.load(str(path))
            self.kg_file = kg_file
            logger.info(f"Loaded KG from {path}")
        except Exception as e:
            logger.error(f"Failed to load KG: {e}")
            return False

        trans_file = kg_file.replace(".pkl", "_transitions.pkl")
        if not trans_file.endswith(".pkl"):
            trans_file = kg_file + "_transitions.pkl"
        trans_path = KG_DIR / trans_file
        if trans_path.exists():
            try:
                with open(trans_path, "rb") as f:
                    self.transitions = pickle.load(f)
                logger.info(f"Loaded transitions from {trans_path}")
            except Exception as e:
                logger.warning(f"Failed to load transitions: {e}")
                self.transitions = {}
        else:
            logger.warning(f"Transitions file not found: {trans_path}")
            self.transitions = {}

        self._dist_matrix = None
        map_id, data_id = "", ""
        if data_dir:
            dp = Path(data_dir)
            parts = dp.parts
            if len(parts) >= 2:
                map_id, data_id = parts[-2], parts[-1]
                npy_dir = ROOT_DIR / "cache" / "npy"
                dm_path = npy_dir / f"state_distance_matrix_{map_id}_{data_id}.npy"
                if dm_path.exists():
                    try:
                        self._dist_matrix = np.load(str(dm_path))
                        logger.info(
                            f"Loaded distance matrix from {dm_path} ({self._dist_matrix.shape})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load distance matrix: {e}")
                else:
                    logger.warning(f"Distance matrix not found: {dm_path}")

        self._state_id_map = {}
        if data_dir:
            sn_path = Path(data_dir) / "graph" / "state_node.txt"
            if not sn_path.exists():
                sn_path = ROOT_DIR / data_dir / "graph" / "state_node.txt"
            if sn_path.exists():
                try:
                    for line in sn_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            try:
                                key = eval(parts[0])
                                if isinstance(key, tuple) and len(key) == 2:
                                    self._state_id_map[(int(key[0]), int(key[1]))] = (
                                        int(parts[1])
                                    )
                            except Exception:
                                pass
                    logger.info(
                        f"Loaded state_id_map: {len(self._state_id_map)} entries from {sn_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load state_node.txt: {e}")
            else:
                logger.warning(f"state_node.txt not found: {sn_path}")

        self._nid_norm_states = {}
        if data_dir and self._state_id_map:
            self._load_known_states(data_dir)

        self._mds_coords = None
        if self._dist_matrix is not None and map_id and data_id:
            try:
                npy_dir = ROOT_DIR / "cache" / "npy"
                npy_dir.mkdir(parents=True, exist_ok=True)
                mds_path = npy_dir / f"state_mds_coords_{map_id}_{data_id}.npy"
                if mds_path.exists():
                    self._mds_coords = np.load(str(mds_path))
                    logger.info(
                        f"Loaded MDS coords from {mds_path} ({self._mds_coords.shape})"
                    )
                else:
                    dm = self._dist_matrix.copy()
                    nan_mask = np.isnan(dm)
                    if nan_mask.any():
                        valid_max = np.nanmax(dm)
                        dm[nan_mask] = valid_max * 1.5 if valid_max > 0 else 1.0
                    np.fill_diagonal(dm, 0)
                    dm = (dm + dm.T) / 2.0
                    N = dm.shape[0]
                    if N <= 2000:
                        from sklearn.manifold import MDS as _MDS

                        emb = _MDS(
                            n_components=2,
                            dissimilarity="precomputed",
                            random_state=42,
                            max_iter=300,
                            normalized_stress=False,
                        )
                        coords = emb.fit_transform(dm)
                    else:
                        from sklearn.manifold import SpectralEmbedding as _SE

                        med = np.median(dm[dm > 0])
                        if med <= 0:
                            med = 1.0
                        affinity = np.exp(-(dm**2) / (med**2))
                        np.fill_diagonal(affinity, 1.0)
                        emb = _SE(
                            n_components=2, affinity="precomputed", random_state=42
                        )
                        coords = emb.fit_transform(affinity)
                    mins = coords.min(axis=0)
                    maxs = coords.max(axis=0)
                    ranges = maxs - mins
                    ranges[ranges == 0] = 1.0
                    coords = (coords - mins) / ranges
                    np.save(str(mds_path), coords)
                    self._mds_coords = coords
                    logger.info(
                        f"Computed & saved MDS coords ({N} states) to {mds_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to compute MDS coords: {e}")

        self.kg_loaded = True
        return True

    def _load_known_states(self, data_dir: str) -> None:
        import json as _json

        bktree_dir = Path(data_dir) / "bktree"
        if not bktree_dir.exists():
            logger.warning(f"bktree dir not found: {bktree_dir}")
            return

        def _collect_nodes(node):
            nodes = []
            if node is not None:
                nodes.append(node)
                for child in node.get("children", {}).values():
                    nodes.extend(_collect_nodes(child))
            return nodes

        secondary_states = {}
        sec_files = list(bktree_dir.glob("secondary_bktree_*.json"))
        for sf in sec_files:
            cid_str = sf.stem.replace("secondary_bktree_", "")
            try:
                with open(str(sf), "r") as f:
                    root = _json.load(f)
                p_id = int(cid_str)
                for node in _collect_nodes(root):
                    s_id = node.get("cluster_id")
                    st = node.get("state")
                    if s_id is not None and st is not None:
                        secondary_states[(p_id, s_id)] = st
            except Exception:
                pass

        primary_file = bktree_dir / "primary_bktree.json"
        if primary_file.exists():
            try:
                with open(str(primary_file), "r") as f:
                    root = _json.load(f)
                for node in _collect_nodes(root):
                    p_id = node.get("cluster_id")
                    st = node.get("state")
                    if p_id is not None and st is not None:
                        for (pk, sk), nid in self._state_id_map.items():
                            if pk == p_id and (pk, sk) in secondary_states:
                                self._nid_norm_states[nid] = secondary_states[(pk, sk)]
            except Exception as e:
                logger.warning(f"Failed to load known states: {e}")

        logger.info(
            f"Loaded {len(self._nid_norm_states)} nid norm_states for distance matching"
        )

    def _drain_history(self) -> None:
        for ep_data in self.bridge.get_histories():
            self._total_completed += 1
            ep_id = ep_data["episode_id"]
            self._history_store[ep_id] = ep_data["frames"]
            self._history_meta[ep_id] = {
                "result": ep_data.get("result", "Unknown"),
                "score": ep_data.get("score", 0.0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": ep_data.get("mode", ""),
                "frame_count": len(ep_data["frames"]),
            }
            self._ep_detail_cache.pop(ep_id, None)
        while len(self._history_store) > self._history_max_episodes:
            oldest = min(self._history_store.keys())
            del self._history_store[oldest]
            if oldest in self._history_meta:
                del self._history_meta[oldest]
            self._ep_detail_cache.pop(oldest, None)

    def history_stats(self) -> Dict[str, int]:
        total_frames = sum(len(f) for f in self._history_store.values())
        return {
            "history_episodes": len(self._history_store),
            "history_frames": total_frames,
            "history_capacity": self._history_max_episodes,
            "total_completed": self._total_completed,
        }

    def ack_history(self, episode_ids: List[int]) -> None:
        for eid in episode_ids:
            self._history_store.pop(eid, None)
            self._history_meta.pop(eid, None)
            self._ep_detail_cache.pop(eid, None)

    def _build_episode_detail(self, ep_id: int) -> Optional[Dict[str, Any]]:
        cached = self._ep_detail_cache.get(ep_id)
        if cached is not None:
            return cached

        frames = self._history_store.get(ep_id)
        if frames is None:
            return None
        meta = self._history_meta.get(ep_id, {})

        markov_flow = []
        events_list = []
        for fr in frames:
            nid = fr.get("nid")
            if nid is None:
                sc = fr.get("state_cluster", (0, 0))
                nid = f"M({sc[0]},{sc[1]})"
            markov_flow.append([nid, fr.get("action_code", fr.get("action", ""))])
            events_list.append(
                {
                    "state_id": nid,
                    "state_cluster": fr.get("state_cluster"),
                    "action": fr.get("action", ""),
                    "action_code": fr.get("action_code", ""),
                    "event_type": fr.get("action_source", ""),
                    "my_count": fr.get("my_count", 0),
                    "enemy_count": fr.get("enemy_count", 0),
                    "hp_my": fr.get("hp_my", 0),
                    "hp_enemy": fr.get("hp_enemy", 0),
                    "game_loop": fr.get("game_loop", 0),
                    "end_game_flag": fr.get("end_game_flag", False),
                    "my_units_pos": self._to_native(fr.get("my_units_pos", [])),
                    "enemy_units_pos": self._to_native(fr.get("enemy_units_pos", [])),
                    "plan": self._to_native(fr.get("plan"))
                    if fr.get("plan") is not None
                    else None,
                }
            )
        events_list = _compute_deviations(events_list)
        _all_st = set()
        for _ev in events_list:
            _sid = _ev.get("state_id")
            if isinstance(_sid, int):
                _all_st.add(_sid)
            _ps = _ev.get("planned_state")
            if isinstance(_ps, int):
                _all_st.add(_ps)
            _pl = _ev.get("plan")
            if _pl and _pl.get("beam_paths"):
                for _bp in _pl["beam_paths"]:
                    for _st in _bp.get("steps", []):
                        _sv = _st.get("state")
                        if _sv is not None:
                            try:
                                _all_st.add(int(_sv))
                            except (TypeError, ValueError):
                                pass
        _dm_local = self._dist_matrix if self else None
        _dm_sh = _dm_local.shape if _dm_local is not None else (0, 0)
        _dist_map = {}
        for _s1 in _all_st:
            for _s2 in _all_st:
                if _s1 <= _s2:
                    _d = _safe_dist(_dm_local, _dm_sh, _s1, _s2)
                    if _d is not None:
                        _dist_map[f"{_s1},{_s2}"] = round(_d, 4)

        result = {
            "id": ep_id,
            "result": meta.get("result", "Unknown"),
            "score": meta.get("score", 0.0),
            "markov_flow": markov_flow,
            "events": events_list,
            "timestamp": meta.get("timestamp", ""),
            "mode": meta.get("mode", ""),
            "steps": len(frames),
            "match_mode": "",
            "source": "agent",
            "dist_map": _dist_map,
            "fork_tree": _build_fork_tree_data(events_list, _dist_map),
        }
        self._ep_detail_cache[ep_id] = result
        return result

    def event_generator(self):
        try:
            while True:
                events = self.bridge.get_events()
                for event in events:
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                if not events:
                    yield f": heartbeat\n\n"
                time.sleep(0.2)
        except GeneratorExit:
            pass


def create_app(bridge: GameBridge) -> FastAPI:
    global _instance
    _instance = BridgeServer(bridge)

    app = FastAPI(title="PredictionRTS Bridge Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        path = request.url.path
        if path in (
            "/game/status",
            "/game/observation",
            "/game/logs",
            "/game/events",
            "/game/actions",
            "/game/episodes",
            "/game/episodes/clear",
        ):
            return await call_next(request)

        start = time.time()
        response = await call_next(request)
        elapsed_ms = round((time.time() - start) * 1000, 1)
        method = request.method
        status = response.status_code

        if status >= 500:
            level = "error"
        elif status >= 400:
            level = "warn"
        else:
            level = "info"
        _instance.add_log(level, "api", f"{method} {path} {status} ({elapsed_ms}ms)")

        return response

    @app.get("/game/status")
    async def get_status():
        status = _instance._refresh_status()
        _instance._drain_history()
        status.update(_instance.history_stats())
        return status

    @app.get("/game/state_space")
    async def get_state_space():
        if not _instance or _instance._mds_coords is None:
            return {"coords": {}, "total": 0}
        coords = _instance._mds_coords
        N = coords.shape[0]
        result = {}
        for i in range(N):
            result[str(i)] = [
                round(float(coords[i, 0]), 4),
                round(float(coords[i, 1]), 4),
            ]
        return {"coords": result, "total": N}

    @app.get("/game/observation")
    async def get_observation():
        status = _instance._refresh_status()
        return {
            "my_total_hp": status.get("my_total_hp", 0),
            "enemy_total_hp": status.get("enemy_total_hp", 0),
            "last_action": status.get("last_action", "-"),
        }

    @app.get("/game/logs")
    async def get_logs(after_seq: int = Query(0)):
        logs = _instance.drain_logs(after_seq=after_seq)
        return {"logs": logs, "latest_seq": _instance._log_seq}

    @app.post("/game/logs/clear")
    async def clear_logs():
        _instance.clear_logs()
        return {"status": "cleared"}

    @app.get("/game/episodes")
    async def get_episodes(
        page: int = Query(1, ge=1),
        per_page: int = Query(10, ge=1, le=9999),
        sort: str = Query("id_desc"),
        search: Optional[str] = Query(None),
    ):
        _instance._drain_history()
        all_eps = []
        for ep_id in sorted(_instance._history_store.keys()):
            detail = _instance._build_episode_detail(ep_id)
            if detail is None:
                continue
            all_eps.append(detail)
        if search:
            search_lower = search.lower()
            all_eps = [
                ep
                for ep in all_eps
                if search_lower in ep["result"].lower()
                or search_lower in ep["mode"].lower()
            ]
        sort_key = {
            "id_desc": lambda e: -e["id"],
            "id_asc": lambda e: e["id"],
            "score_desc": lambda e: -e["score"],
            "score_asc": lambda e: e["score"],
        }.get(sort, lambda e: -e["id"])
        all_eps.sort(key=sort_key)
        total = len(all_eps)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = all_eps[start:end]
        return {
            "episodes": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    @app.post("/game/episodes/ack")
    async def ack_episodes(request: Request):
        body = await request.json()
        ids = body.get("ids", [])
        if ids:
            _instance.ack_history(ids)
        return {"ok": True, "acked": len(ids)}

    @app.post("/game/episodes/clear")
    async def clear_episodes(req: dict = None):
        req = req or {}
        if "max_history" in req:
            try:
                _instance._history_max_episodes = int(req["max_history"])
            except (TypeError, ValueError):
                pass
        _instance._history_store.clear()
        _instance._history_meta.clear()
        _instance._ep_detail_cache.clear()
        _instance.add_log("info", "system", "对局记录已清空")
        return {"ok": True}

    @app.post("/game/results/save")
    async def save_results(req: dict = None):
        max_count = req.get("max_count", 100) if req else 100
        target_episodes = req.get("target_episodes", 0) if req else 0
        _instance._drain_history()
        if not _instance._history_store:
            raise HTTPException(status_code=404, detail="没有可保存的对局记录")

        status = _instance._refresh_status()
        mode = status.get("agent_mode", status.get("mode", "unknown"))
        kg_file = status.get("kg_file", "")
        map_id = "unknown"
        if kg_file:
            parts = Path(kg_file).parent.name
            map_id = parts if parts else "unknown"

        all_ids = sorted(_instance._history_meta.keys())
        if target_episodes > 0:
            selected_ids = all_ids[-target_episodes:]
        else:
            selected_ids = all_ids[:max_count]

        episodes = []
        for ep_id in selected_ids:
            meta = _instance._history_meta.get(ep_id, {})
            frames_raw = _instance._history_store.get(ep_id, [])
            frames_light = []
            for fr in frames_raw:
                frames_light.append(
                    {
                        "state_cluster": fr.get("state_cluster"),
                        "nid": fr.get("nid"),
                        "action": fr.get("action", ""),
                        "action_code": fr.get("action_code", ""),
                        "action_source": fr.get("action_source", ""),
                        "my_count": fr.get("my_count", 0),
                        "enemy_count": fr.get("enemy_count", 0),
                        "hp_my": fr.get("hp_my", 0),
                        "hp_enemy": fr.get("hp_enemy", 0),
                        "game_loop": fr.get("game_loop", 0),
                        "end_game_flag": fr.get("end_game_flag", False),
                    }
                )
            score = 0
            if frames_light:
                last = frames_light[-1]
                score = last["hp_my"] - last["hp_enemy"]
            episodes.append(
                {
                    "episode_id": ep_id,
                    "result": meta.get("result", "Unknown"),
                    "score": meta.get("score", score),
                    "num_frames": len(frames_light),
                    "frames": frames_light,
                }
            )

        agent_params = status.get("agent_params", {})
        backup_enabled = agent_params.get("enable_backup", False)

        payload = {
            "metadata": {
                "map_id": map_id,
                "mode": mode,
                "kg_file": kg_file,
                "backup_enabled": backup_enabled,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_episodes": len(episodes),
                "source": "live_game",
            },
            "params": agent_params if agent_params else {},
            "episodes": episodes,
        }

        results_dir = ROOT_DIR / "output" / "live_results" / map_id
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_label = mode
        if backup_enabled and mode not in ("single_step", "replay"):
            mode_label = f"{mode}_backup"
        sm = agent_params.get("score_mode", "")
        bw = agent_params.get("beam_width", "")
        la = agent_params.get("max_steps", agent_params.get("lookahead_steps", ""))
        as_ = agent_params.get("action_strategy", "")
        mv = agent_params.get("min_visits")
        msr = agent_params.get("max_state_revisits")
        mcp = agent_params.get("min_cum_prob")
        df = agent_params.get("discount_factor")
        param_parts = []
        if sm:
            param_parts.append(f"sm{sm}")
        if bw is not None and bw != "":
            param_parts.append(f"bw{bw}")
        if la is not None and la != "":
            param_parts.append(f"la{la}")
        if mv is not None:
            param_parts.append(f"mv{mv}")
        if msr is not None:
            param_parts.append(f"msr{msr}")
        if mcp is not None:
            param_parts.append(f"mcp{mcp}")
        if df is not None:
            param_parts.append(f"df{df}")
        if as_:
            param_parts.append(f"as{as_}")
        if as_ == "epsilon_greedy":
            eps = agent_params.get("epsilon")
            if eps is not None:
                param_parts.append(f"eps{eps}")
        if backup_enabled and mode not in ("single_step", "replay"):
            bst = agent_params.get("backup_score_threshold")
            if bst is not None:
                param_parts.append(f"bp{bst}")
            bdt = agent_params.get("backup_distance_threshold")
            if bdt is not None:
                param_parts.append(f"bd{bdt}")
        param_str = "_".join(param_parts)
        filename = f"{mode_label}_{param_str}_{ts}.json"
        filepath = results_dir / filename
        with open(str(filepath), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        _instance.add_log(
            "success",
            "system",
            f"结果已保存: {filepath} ({len(episodes)} 局)",
        )
        return {
            "ok": True,
            "path": str(filepath),
            "num_episodes": len(episodes),
            "filename": filename,
        }

    @app.get("/game/results/list")
    async def list_results():
        results_root = ROOT_DIR / "output" / "live_results"
        if not results_root.exists():
            return {"results": []}
        items = []
        for jf in sorted(
            results_root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                with open(str(jf), "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                items.append(
                    {
                        "path": str(jf),
                        "filename": jf.name,
                        "map_id": meta.get("map_id", ""),
                        "mode": meta.get("mode", ""),
                        "timestamp": meta.get("timestamp", ""),
                        "num_episodes": meta.get("num_episodes", 0),
                    }
                )
            except Exception:
                continue
        return {"results": items}

    @app.post("/game/action")
    async def send_action(req: ActionRequest):
        _instance.bridge.put_action(
            {"action_code": req.action, "plan_snap": None, "event_type": "manual"}
        )
        return {"status": "queued", "action": req.action}

    @app.post("/game/fallback")
    async def set_fallback(req: FallbackRequest):
        _instance.bridge.put_action(
            {
                "action_code": f"__fallback:{req.action}",
                "plan_snap": None,
                "event_type": "manual",
            }
        )
        return {"status": "fallback_set", "action": req.action}

    @app.post("/game/control")
    async def control(req: ControlRequest):
        if req.command not in (
            "start",
            "pause",
            "resume",
            "stop",
            "step",
            "run_episode",
        ):
            raise HTTPException(
                status_code=400, detail=f"Invalid command: {req.command}"
            )
        if req.command == "run_episode":
            _instance.bridge.set_run_episode()
            _instance.add_log("info", "control", "单局运行: 将在下一局结束后暂停")
        _instance.bridge.send_control(req.command)
        return {"status": "sent", "command": req.command}

    @app.get("/game/events")
    async def game_events():
        return StreamingResponse(
            _instance.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/game/load_kg")
    async def load_kg(kg_file: str = Query(...), data_dir: Optional[str] = Query(None)):
        success = _instance.load_kg(kg_file, data_dir=data_dir or None)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load KG file.")
        return {"status": "loaded", "kg_file": kg_file}

    @app.get("/game/actions")
    async def list_actions():
        from src.sc2env.kg_guided_agent import ACTION_NAME_MAP, FALLBACK_ACTIONS

        return {
            "action_map": ACTION_NAME_MAP,
            "available_actions": FALLBACK_ACTIONS,
        }

    _FILTER_CFG_FILE = Path(
        os.environ.get(
            "LIVE_FILTER_CFG",
            os.path.join(
                str(Path(__file__).resolve().parent.parent.parent),
                ".live_filter_cfg.json",
            ),
        )
    )
    _FILTER_DEFAULTS = {
        "levels": ["info", "success", "warn", "error"],
        "sources": ["game", "api"],
        "types": ["info", "action", "fallback", "result", "episode", "control"],
    }

    @app.get("/game/filter-config")
    async def get_filter_config():
        if _FILTER_CFG_FILE.is_file():
            try:
                return json.loads(_FILTER_CFG_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return _FILTER_DEFAULTS

    @app.post("/game/filter-config")
    async def save_filter_config(req: Dict[str, Any] = None):
        cfg = req if req else {}
        for key in _FILTER_DEFAULTS:
            if key not in cfg or not isinstance(cfg[key], list):
                cfg[key] = _FILTER_DEFAULTS[key]
        try:
            _FILTER_CFG_FILE.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        return {"status": "saved", "config": cfg}

    @app.post("/game/beam_params")
    async def update_beam_params(req: dict):
        try:
            if not _instance.bridge.param_update_queue.empty():
                _instance.bridge.param_update_queue.get_nowait()
            _instance.bridge.param_update_queue.put_nowait(req)
        except Exception:
            pass
        return {"ok": True}

    @app.get("/game/param_confirm")
    async def get_param_confirm():
        trial_number = _instance.bridge.check_param_confirm()
        return {"confirmed": trial_number}

    @app.post("/game/shutdown")
    async def shutdown():
        _instance.bridge.request_stop()
        threading.Thread(target=lambda: os._exit(0), daemon=True).start()
        return {"status": "shutting_down"}

    @app.post("/game/window_pos")
    async def set_window_pos(req: dict):
        try:
            x = int(req.get("x", 50))
            y = int(req.get("y", 50))
            w = int(req.get("w", 640))
            h = int(req.get("h", 480))
        except (ValueError, TypeError):
            return {"ok": False, "error": "invalid params"}
        try:
            import ctypes
            import threading

            def _do_move():
                import time
                import ctypes

                deadline = time.time() + 5
                while time.time() < deadline:
                    hwnd = ctypes.windll.user32.FindWindowW(None, "StarCraft II")
                    if hwnd:
                        ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, w, h, 0x0040)
                        return
                    time.sleep(0.3)

            threading.Thread(target=_do_move, daemon=True).start()
            _instance.add_log("info", "system", f"窗口移动: ({x},{y}) {w}x{h}")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return app


def run_server(
    bridge: GameBridge,
    host: str = "0.0.0.0",
    port: int = 8000,
    kg_file: Optional[str] = None,
    data_dir: Optional[str] = None,
):
    import uvicorn

    app = create_app(bridge)
    if kg_file and _instance:
        _instance.load_kg(kg_file, data_dir=data_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
