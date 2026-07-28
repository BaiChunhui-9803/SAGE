#!/usr/bin/env python
"""
ReplayCollector: 批量回放 action 序列，收集 norm_state 数据并增量构建 BKTree。

继承 SmartAgent，复用其 BKTree 和动作执行能力，
在 step() 中只做：获取 norm_state → 聚类 → 执行 action → 记录帧。

不执行 Q-learning、不写文件、不依赖 KG/transitions/beam search。

支持增量保存与中断恢复：
- 每 episode 完成后立即 pickle append 到增量文件
- 每 5% 进度保存 BKTree checkpoint
- progress.json 记录已完成 episode，支持从中断处恢复
"""

import os
import sys
import json
import csv
import pickle
import logging
from typing import List, Dict, Optional, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from pysc2.lib import actions as sc2_actions

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sc2env.agent import SmartAgent
from src.sc2env.config import get_map_config
from src.structure.BKTree_sc2 import ClusterNode

_MAP_CONFIG, _MAP, _ENV_CONFIG, _ALG_CONFIG, _PATH_CONFIG = get_map_config("sce-1")

logger = logging.getLogger(__name__)


def _deserialize_bktree_node(node_data: Dict) -> Optional[ClusterNode]:
    if node_data is None:
        return None
    node = ClusterNode(node_data["state"], node_data["cluster_id"])
    for dist_key, child_data in node_data.get("children", {}).items():
        dist_val = int(dist_key) if dist_key.isdigit() else float(dist_key)
        child_node = _deserialize_bktree_node(child_data)
        if child_node is not None:
            node.children[dist_val] = child_node
    return node


class ReplayCollector(SmartAgent):
    def __init__(
        self,
        bridge=None,
        action_log_path: str = "",
        replay_count: int = 3,
        batch_start: int = 0,
        batch_end: Optional[int] = None,
        output_dir: Optional[str] = None,
        primary_threshold: float = 1.0,
        secondary_threshold: float = 0.5,
    ):
        self.bridge = bridge
        self._action_log_path = action_log_path
        self._replay_count = replay_count
        self._batch_start = batch_start
        self._batch_end = batch_end
        self._base_output_dir = output_dir or "output/collected_data"
        self._primary_threshold = primary_threshold
        self._secondary_threshold = secondary_threshold

        self._all_sequences: List[List[str]] = []
        self._current_seq_idx: int = batch_start
        self._current_run_idx: int = 0
        self._replay_actions: List[str] = []
        self._replay_idx: int = 0
        self._done: bool = False

        self._collected_episodes: List[Dict[str, Any]] = []
        self._current_frames: List[Dict[str, Any]] = []
        self._completed_episodes: int = 0
        self._total_episodes: int = 0
        self._seq_stats: Dict[int, Dict[str, list]] = defaultdict(
            lambda: {"results": [], "scores": [], "frame_counts": []}
        )

        self._completed_pairs: List[Tuple[int, int]] = []
        self._frames_file = None
        self._progress_path: Optional[Path] = None
        self._last_checkpoint_pct: int = -1
        self._incremental_ts: str = ""

        super().__init__()

        self._all_sequences = self._load_action_log(action_log_path)
        effective_end = (
            batch_end if batch_end is not None else len(self._all_sequences) - 1
        )
        effective_end = min(effective_end, len(self._all_sequences) - 1)
        self._batch_end = effective_end
        self._total_episodes = (effective_end - batch_start + 1) * replay_count

        subdir_name = f"ep{batch_start}-{effective_end}_r{replay_count}_p{primary_threshold:g}_s{secondary_threshold:g}"
        self._output_dir = str(Path(self._base_output_dir) / subdir_name)

        self._init_incremental_save()

        if not self._done:
            self._load_current_sequence()

        print(
            f"[ReplayCollector] sequences={batch_start}~{effective_end}, "
            f"replay_count={replay_count}, total_episodes={self._total_episodes}, "
            f"threshold=({primary_threshold}, {secondary_threshold})"
        )

    def _load_action_log(self, path: str) -> List[List[str]]:
        if not path or not os.path.exists(path):
            print(f"[ReplayCollector] Warning: action_log not found: {path}")
            return []
        sequences = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                raw = row[0].strip()
                if len(raw) < 2:
                    continue
                actions = [raw[i : i + 2] for i in range(0, len(raw), 2)]
                sequences.append(actions)
        print(f"[ReplayCollector] Loaded {len(sequences)} action sequences from {path}")
        return sequences

    def _load_current_sequence(self):
        if (
            self._batch_end is not None
            and self._current_seq_idx <= self._batch_end
            and self._current_seq_idx < len(self._all_sequences)
        ):
            self._replay_actions = list(self._all_sequences[self._current_seq_idx])
            self._replay_idx = 0
        else:
            self._replay_actions = []
            self._replay_idx = 0

    # ── 增量保存与恢复 ──────────────────────────────────────────────

    def _init_incremental_save(self):
        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path = output_dir / "progress.json"

        if self._progress_path.exists():
            self._load_progress()
        else:
            self._incremental_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._last_checkpoint_pct = -1
            pkl_path = output_dir / f"episodes_{self._incremental_ts}.pkl"
            self._frames_file = open(pkl_path, "ab")
            print(f"[ReplayCollector] 增量文件: {pkl_path}")

    def _load_progress(self):
        try:
            with open(self._progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except Exception as e:
            print(f"[ReplayCollector] Warning: progress.json 损坏 ({e}), 从头开始")
            self._clean_corrupt_progress()
            return

        completed_pairs_raw = progress.get("completed_pairs", [])
        self._completed_pairs = [tuple(p) for p in completed_pairs_raw]
        self._completed_episodes = len(self._completed_pairs)
        self._last_checkpoint_pct = progress.get("last_checkpoint_pct", -1)
        self._incremental_ts = progress.get("frames_file_ts", "")

        completed_meta = progress.get("completed_meta", [])
        for meta in completed_meta:
            sid = meta["source_idx"]
            self._seq_stats[sid]["results"].append(meta["result"])
            self._seq_stats[sid]["scores"].append(meta["score"])
            self._seq_stats[sid]["frame_counts"].append(meta["num_frames"])

        self._restore_bktree_checkpoint()

        expected_set = set(self._completed_pairs)
        found_next = False

        highest_seq = self._batch_start - 1
        for seq in range(self._batch_end, self._batch_start - 1, -1):
            if any((seq, run) in expected_set for run in range(self._replay_count)):
                highest_seq = seq
                break

        for seq in range(highest_seq, self._batch_end + 1):
            for run in range(self._replay_count):
                if (seq, run) not in expected_set:
                    self._current_seq_idx = seq
                    self._current_run_idx = run
                    found_next = True
                    break
            if found_next:
                break

        if not found_next:
            self._done = True
            print(
                f"[ReplayCollector] 所有 {self._completed_episodes} episodes 已完成, "
                f"跳过采集, 直接保存"
            )
            self._save_all()
            return

        output_dir = Path(self._output_dir)
        pkl_path = output_dir / f"episodes_{self._incremental_ts}.pkl"
        self._frames_file = open(pkl_path, "ab")

        print(
            f"[ReplayCollector] 恢复进度: {self._completed_episodes}/{self._total_episodes} "
            f"episodes 已完成, 从 seq={self._current_seq_idx} run={self._current_run_idx} 继续"
        )

    def _restore_bktree_checkpoint(self):
        output_dir = Path(self._output_dir)
        from src.structure.BKTree_sc2 import get_max_cluster_id

        primary_path = output_dir / "primary_bktree.json"
        if primary_path.exists():
            try:
                with open(primary_path, "r") as f:
                    primary_data = json.load(f)
                self.primary_bktree.root = _deserialize_bktree_node(primary_data)
                max_id = get_max_cluster_id(self.primary_bktree)
                self.primary_bktree.next_cluster_id = max_id + 1
                print(f"[ReplayCollector] 恢复 primary BKTree (next_id={max_id + 1})")
            except Exception as e:
                print(f"[ReplayCollector] Warning: primary BKTree 恢复失败 ({e})")

        for json_file in output_dir.glob("secondary_bktree_*.json"):
            cid_str = json_file.stem.replace("secondary_bktree_", "")
            try:
                cid = int(cid_str)
            except ValueError:
                continue
            try:
                with open(json_file, "r") as f:
                    sec_data = json.load(f)
                tree = self.secondary_bktree[cid]
                tree.root = _deserialize_bktree_node(sec_data)
                max_id = get_max_cluster_id(tree)
                tree.next_cluster_id = max_id + 1
            except Exception as e:
                print(
                    f"[ReplayCollector] Warning: secondary BKTree {cid} 恢复失败 ({e})"
                )

        sec_count = sum(1 for t in self.secondary_bktree.values() if t.root is not None)
        print(f"[ReplayCollector] 恢复 {sec_count} 个 secondary BKTrees")

    def _clean_corrupt_progress(self):
        if self._progress_path and self._progress_path.exists():
            self._progress_path.unlink()
        self._incremental_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self._output_dir)
        pkl_path = output_dir / f"episodes_{self._incremental_ts}.pkl"
        self._frames_file = open(pkl_path, "ab")

    def _save_progress(self):
        if self._progress_path is None:
            return
        meta_list = []
        for ep in self._collected_episodes:
            meta_list.append(
                {
                    "source_idx": ep["source_idx"],
                    "replay_idx": ep["replay_idx"],
                    "result": ep["result"],
                    "score": ep["score"],
                    "num_frames": ep["num_frames"],
                }
            )
        progress = {
            "batch_start": self._batch_start,
            "batch_end": self._batch_end,
            "replay_count": self._replay_count,
            "primary_threshold": self._primary_threshold,
            "secondary_threshold": self._secondary_threshold,
            "total_episodes": self._total_episodes,
            "completed_pairs": [list(p) for p in self._completed_pairs],
            "completed_episodes": self._completed_episodes,
            "completed_meta": meta_list,
            "last_checkpoint_pct": self._last_checkpoint_pct,
            "frames_file_ts": self._incremental_ts,
        }
        tmp_path = self._progress_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self._progress_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()

    def _save_bktree_checkpoint(self, pct: int):
        output_dir = Path(self._output_dir)
        self._save_bktree_to_dir(output_dir)
        self._last_checkpoint_pct = pct
        print(f"[ReplayCollector] BKTree checkpoint saved at {pct}%")
        self._save_progress()

    def _try_incremental_save_episode(self, episode: Dict[str, Any]):
        if self._frames_file is None:
            return False
        try:
            pickle.dump(episode, self._frames_file)
            self._frames_file.flush()
            return True
        except Exception as e:
            print(f"[ReplayCollector] Warning: 增量写入失败 ({e})")
            return False

    # ── 核心逻辑 ─────────────────────────────────────────────────────

    def get_state_cluster(self, norm_state):
        if self.primary_bktree.root is None:
            self.primary_bktree.root = ClusterNode(norm_state, 1)
            self.secondary_bktree[1].root = ClusterNode(norm_state, 1)
            return (1, 1)
        else:
            new_cluster_id = self.classify_new_state(
                norm_state, self.primary_bktree, threshold=self._primary_threshold
            )
            if self.secondary_bktree[new_cluster_id].root is None:
                self.secondary_bktree[new_cluster_id].root = ClusterNode(norm_state, 1)
                return (new_cluster_id, 1)
            else:
                new_sub_cluster_id = self.classify_new_state(
                    norm_state,
                    self.secondary_bktree[new_cluster_id],
                    threshold=self._secondary_threshold,
                )
                return (new_cluster_id, new_sub_cluster_id)

    def step(self, obs, env):
        super(SmartAgent, self).step(obs, env)

        if obs.last():
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if obs.first():
            if not self._initial_spawned:
                my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
                enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])
                self._initial_units_my = [(u.x, u.y) for u in my_units]
                self._initial_units_enemy = [(u.x, u.y) for u in enemy_units]
                self._initial_spawned = True

        my_units = self.get_my_units_by_type(obs, _MAP["unit_type"])
        enemy_units = self.get_enemy_units_by_type(obs, _MAP["unit_type"])

        if self._done:
            return sc2_actions.RAW_FUNCTIONS.no_op()

        if not self._termination_signaled:
            if not enemy_units and my_units:
                self.end_game_state = "Win"
                self.end_game_flag = True
                self._termination_signaled = True
                env.f_result = "win"
            elif not my_units and enemy_units:
                self.end_game_state = "Loss"
                self.end_game_flag = True
                self._termination_signaled = True
                env.f_result = "loss"

        if (my_units or enemy_units) and not self._done:
            norm_state = self.get_norm_state(obs)
            state_cluster = self.get_state_cluster(norm_state)

            action_code = None
            if self._replay_idx < len(self._replay_actions):
                action_code = self._replay_actions[self._replay_idx]
                self._replay_idx += 1

            action_name = self._resolve_action(action_code)

            self._pending_cluster = None
            if action_name and len(action_code or "") == 2 and action_code[0].isdigit():
                cluster_idx = int(action_code[0])
                if 0 <= cluster_idx < len(self.clusters):
                    self._pending_cluster = self.clusters[cluster_idx]

            result = None
            if self._pending_cluster and hasattr(self, self._pending_cluster):
                self.cluster_result = getattr(self, self._pending_cluster)(obs)
            if action_name and hasattr(self, action_name):
                result = getattr(self, action_name)(obs)

            hp_my = int(sum(u.health for u in my_units)) if my_units else 0
            hp_enemy = int(sum(u.health for u in enemy_units)) if enemy_units else 0

            self._current_frames.append(
                {
                    "norm_state": norm_state,
                    "state_cluster": (
                        int(state_cluster[0]),
                        int(state_cluster[1]),
                    ),
                    "action_code": action_code,
                    "hp_my": hp_my,
                    "hp_enemy": hp_enemy,
                    "game_loop": int(obs.observation.game_loop[0]),
                }
            )

            if result is not None:
                return result

        return sc2_actions.RAW_FUNCTIONS.no_op()

    def _resolve_action(self, raw_action: Optional[str]) -> Optional[str]:
        if raw_action is None or len(raw_action) < 2:
            return None
        letter = raw_action[1].lower()
        action_idx = ord(letter) - ord("a")
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        return None

    def new_game(self):
        if self._done:
            return

        if self._current_frames:
            result = self.end_game_state or "Dogfall"
            last_frame = self._current_frames[-1] if self._current_frames else {}
            score = last_frame.get("hp_my", 0) - last_frame.get("hp_enemy", 0)
            episode = {
                "source_idx": self._current_seq_idx,
                "replay_idx": self._current_run_idx,
                "result": result,
                "score": float(score),
                "num_frames": len(self._current_frames),
                "frames": list(self._current_frames),
            }

            self._try_incremental_save_episode(episode)

            del episode["frames"]
            self._collected_episodes.append(episode)
            self._current_frames = []
            self._completed_episodes += 1
            if hasattr(self, "ctx") and self.ctx:
                self.ctx.episode_count += 1

            pair = (self._current_seq_idx, self._current_run_idx)
            self._completed_pairs.append(pair)

            stats = self._seq_stats[self._current_seq_idx]
            stats["results"].append(result)
            stats["scores"].append(score)
            stats["frame_counts"].append(episode["num_frames"])

            self._save_progress()

            self._check_bktree_checkpoint()

        self.end_game_frames = _ENV_CONFIG["_MAX_STEP"] * _ENV_CONFIG["_STEP_MUL"]
        self.end_game_state = "Dogfall"
        self.end_game_flag = False
        self._termination_signaled = False
        self.action_queue.clear()

        self._current_run_idx += 1
        if self._current_run_idx < self._replay_count:
            self._replay_idx = 0
        else:
            self._current_run_idx = 0
            self._current_seq_idx += 1
            if self._current_seq_idx <= self._batch_end:
                self._load_current_sequence()
            else:
                self._done = True
                self._save_all()
                return

        self._report_progress()

    def _check_bktree_checkpoint(self):
        if self._total_episodes <= 0:
            return
        pct = int(self._completed_episodes / self._total_episodes * 100)
        next_checkpoint = (pct // 5) * 5
        if next_checkpoint == 0 and pct > 0:
            next_checkpoint = 5
        if next_checkpoint > self._last_checkpoint_pct:
            self._save_bktree_checkpoint(next_checkpoint)

    def _end_episode(self, obs):
        pass

    def _report_progress(self):
        if self._total_episodes <= 0:
            return
        done = self._completed_episodes
        total = self._total_episodes
        pct = done / total * 100 if total else 0

        if self.bridge:
            try:
                self.bridge.update_status(
                    frame=done,
                    batch_replay_progress=f"{done}/{total} ({pct:.1f}%)",
                    batch_replay_current_seq=self._current_seq_idx,
                )
            except Exception:
                pass
            if done % 100 == 0 or done == total:
                summary = self._compute_summary_stats()
                try:
                    self.bridge.put_event(
                        {
                            "level": "info",
                            "source": "game",
                            "message": f"进度: {done}/{total} ({pct:.1f}%)",
                            "batch_stats": summary,
                        }
                    )
                except Exception:
                    pass
        else:
            print(f"[ReplayCollector] 进度: {done}/{total} ({pct:.1f}%)")

    def _compute_summary_stats(self) -> Dict[str, Any]:
        import numpy as np

        all_results: List[str] = []
        all_scores: List[float] = []
        for stats in self._seq_stats.values():
            all_results.extend(stats["results"])
            all_scores.extend(stats["scores"])
        if not all_results:
            return {}
        win_count = sum(1 for r in all_results if r == "Win")
        return {
            "completed_sequences": len(self._seq_stats),
            "total_episodes": len(all_results),
            "win_rate": round(win_count / len(all_results), 4),
            "mean_score": round(float(np.mean(all_scores)), 2),
            "std_score": round(float(np.std(all_scores)), 2),
        }

    def _save_all(self):
        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._frames_file is not None:
            self._frames_file.close()
            self._frames_file = None

        self._save_bktree_to_dir(output_dir)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = self._compute_summary_stats()
        summary["primary_threshold"] = self._primary_threshold
        summary["secondary_threshold"] = self._secondary_threshold
        summary["frames_file"] = f"episodes_{self._incremental_ts}.pkl"
        summary["total_episodes"] = self._completed_episodes
        stats_path = output_dir / f"batch_stats_{ts}.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[ReplayCollector] Saved batch stats to {stats_path}")

        if self._progress_path and self._progress_path.exists():
            self._progress_path.unlink()

        if self.bridge:
            try:
                self.bridge.put_event(
                    {
                        "level": "success",
                        "source": "game",
                        "message": f"重采样数据集扩张完成: {self._completed_episodes} episodes, BKTree 已保存",
                        "batch_stats": summary,
                    }
                )
                self.bridge.update_status(batch_replay_done=True)
                self.bridge.send_control("pause")
            except Exception:
                pass
        else:
            print(
                f"[ReplayCollector] 全部完成: {self._completed_episodes} episodes, "
                f"Win rate: {summary.get('win_rate', 0):.1%}"
            )

    def _save_bktree_to_dir(self, output_dir: Path):
        def serialize_node(node):
            if node is None:
                return None
            node_info = {
                "state": node.state,
                "cluster_id": node.cluster_id,
                "children": {},
            }
            for dist, child in node.children.items():
                node_info["children"][str(dist)] = serialize_node(child)
            return node_info

        if self.primary_bktree.root is not None:
            primary_path = output_dir / "primary_bktree.json"
            with open(primary_path, "w") as f:
                json.dump(serialize_node(self.primary_bktree.root), f)
            print(f"[ReplayCollector] Saved primary BKTree to {primary_path}")

        for cluster_id, bktree in self.secondary_bktree.items():
            if bktree.root is not None:
                sec_path = output_dir / f"secondary_bktree_{cluster_id}.json"
                with open(sec_path, "w") as f:
                    json.dump(serialize_node(bktree.root), f)

        print(f"[ReplayCollector] Saved {len(self.secondary_bktree)} secondary BKTrees")

    @staticmethod
    def load_incremental_episodes(pkl_path: str) -> List[Dict[str, Any]]:
        episodes = []
        try:
            with open(pkl_path, "rb") as f:
                while True:
                    try:
                        ep = pickle.load(f)
                        episodes.append(ep)
                    except EOFError:
                        break
                    except Exception:
                        break
        except FileNotFoundError:
            pass
        return episodes
