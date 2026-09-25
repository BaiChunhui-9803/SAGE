#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate final-eval BKTree start-state compatibility with learner baselines.

The check is intentionally record-only: it does not mutate any experiment data.
For each final-eval repeat, it resolves local compact state id 0, compares it
with same-map PyMARL/OnPolicy start states, and reports whether both states
would be assigned to the same node in a fresh fused BKTree under the configured
scenario thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.fuse_bktree_state_sequences import (  # noqa: E402
    SourceBKTreeLookup,
    UnifiedBKTreeBuilder,
    _thresholds_for_map,
)
from src.structure.custom_distance_sc2 import DistributionDistance  # noqa: E402


ALL_DATA_ROOT = ROOT_DIR / "output" / "learner_results" / "all_data"
MANIFEST_NAME = "experiment_manifest.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(str(path), "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    def fmt_float(value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return "-"

    lines: List[str] = []
    lines.append("# Final Eval 起始状态一致性验证")
    lines.append("")
    lines.append(f"- 实验目录：`{report.get('experiment_dir', '')}`")
    lines.append(f"- 复评目录：`{report.get('eval_dir', '')}`")
    lines.append(f"- 地图：`{report.get('map_id', '')}`")
    lines.append(f"- 阈值：primary={report.get('primary_threshold')}，secondary={report.get('secondary_threshold')}")
    lines.append(f"- 结论：**{report.get('overall_status', 'unknown')}**")
    if report.get("scatter_plot"):
        lines.append(f"- 起始状态散点对比图：`{report.get('scatter_plot')}`")
    if report.get("scatter_data"):
        lines.append(f"- 起始状态散点数据：`{report.get('scatter_data')}`")
    if report.get("raw_coordinate_scatter_plot"):
        lines.append(f"- 反推原始坐标散点对比图：`{report.get('raw_coordinate_scatter_plot')}`")
    if report.get("raw_coordinate_scatter_data"):
        lines.append(f"- 反推原始坐标散点数据：`{report.get('raw_coordinate_scatter_data')}`")
    lines.append("")
    lines.append("## 检查含义")
    lines.append("")
    lines.append(
        "该检查只验证复评输出的 BKTree 记录口径，不影响 etg/Synergy 决策过程。"
        "它首先检查本次 final eval 的本地 0 号节点是否能与同地图 PyMARL/OnPolicy 起始状态对齐，"
        "随后逐 episode 检查真实写入 `state_id_sequence[0]` 的首帧状态。"
        "后者与 BKTree 状态序列记录时机一致，用于发现首局之后的手动 reset/spawn 偏差。"
    )
    lines.append("")
    for repeat in report.get("repeats", []):
        lines.append(f"## {repeat.get('repeat_id', '')}")
        lines.append("")
        lines.append(f"- 本地 BKTree：`{repeat.get('bktree_dir', '')}`")
        lines.append(f"- 本地起始状态可解析：{repeat.get('local_start_resolved')}")
        lines.append(f"- 匹配结论：**{repeat.get('status', 'unknown')}**")
        episode_validation = repeat.get("episode_start_validation") or {}
        if episode_validation:
            lines.append(
                "- 逐局首帧验证："
                f"episodes={episode_validation.get('episode_count', 0)}，"
                f"matched={episode_validation.get('matched_count', 0)}，"
                f"mismatch={episode_validation.get('mismatch_count', 0)}，"
                f"unresolved={episode_validation.get('unresolved_count', 0)}，"
                f"max_dist={fmt_float(episode_validation.get('max_distribution_distance'))}"
            )
            if episode_validation.get("reference"):
                ref = episode_validation["reference"]
                lines.append(
                    "- 逐局验证基准："
                    f"{ref.get('baseline_group', '')} {ref.get('method', '')} "
                    f"`{ref.get('bktree_dir', '')}`"
                )
            if episode_validation.get("report_json"):
                lines.append(f"- 逐局首帧 JSON：`{episode_validation.get('report_json')}`")
            if episode_validation.get("report_csv"):
                lines.append(f"- 逐局首帧 CSV：`{episode_validation.get('report_csv')}`")
            if episode_validation.get("scatter_dir"):
                lines.append(f"- 逐局首帧散点图目录：`{episode_validation.get('scatter_dir')}`")
        if repeat.get("scatter_plot"):
            lines.append(f"- 起始状态散点对比图：`{repeat.get('scatter_plot')}`")
        if repeat.get("scatter_data"):
            lines.append(f"- 起始状态散点数据：`{repeat.get('scatter_data')}`")
        if repeat.get("raw_coordinate_scatter_plot"):
            lines.append(f"- 反推原始坐标散点对比图：`{repeat.get('raw_coordinate_scatter_plot')}`")
        if repeat.get("raw_coordinate_scatter_data"):
            lines.append(f"- 反推原始坐标散点数据：`{repeat.get('raw_coordinate_scatter_data')}`")
        comparisons = repeat.get("comparisons", [])
        if comparisons:
            lines.append("")
            lines.append("| 基线 | 方法 | 起始ID | 同一融合节点 | 融合ID | 坐标距离 | HP距离 | 路径 |")
            lines.append("|---|---|---:|---|---:|---:|---:|---|")
            for row in comparisons:
                lines.append(
                    "| "
                    f"{row.get('baseline_group', '')} | "
                    f"{row.get('method', '')} | "
                    f"{row.get('baseline_start_state_id', '')} | "
                    f"{'是' if row.get('same_unified_state') else '否'} | "
                    f"{row.get('baseline_unified_id', '')} | "
                    f"{fmt_float(row.get('distribution_distance'))} | "
                    f"{fmt_float(row.get('health_distance'))} | "
                    f"`{row.get('bktree_dir', '')}` |"
                )
        else:
            lines.append("")
            lines.append("- 未找到可用的同地图 PyMARL/OnPolicy 基线 BKTree。")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_state_node(path: Path) -> Dict[int, Tuple[int, int]]:
    mapping: Dict[int, Tuple[int, int]] = {}
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
                mapping[int(float(parts[1]))] = (int(cluster_parts[0]), int(cluster_parts[1]))
            except ValueError:
                continue
    return mapping


def _parse_first_node_log_id(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with open(str(path), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                return int(float(parts[0]))
            except ValueError:
                return None
    return None


def _state_for_id(bktree_dir: Path, state_id: int, primary: float, secondary: float) -> Optional[Dict[str, Any]]:
    lookup = SourceBKTreeLookup(bktree_dir, primary, secondary)
    return lookup.state_for_id(state_id)


def _source_etg_bktree_dir(map_id: str) -> Optional[Path]:
    if not map_id:
        return None
    candidates = [
        ROOT_DIR / "data" / map_id / "augmented_1" / "bktree",
        ROOT_DIR / "data" / map_id / "bktree",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _army_points(state: Optional[Mapping[str, Any]], key: str) -> List[Dict[str, float]]:
    if not state:
        return []
    points: List[Dict[str, float]] = []
    for unit in state.get(key, []) or []:
        try:
            points.append({"x": float(unit[0]), "y": float(unit[1]), "hp": float(unit[2])})
        except Exception:
            continue
    return points


def _state_summary(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    red = _army_points(state, "red_army")
    blue = _army_points(state, "blue_army")
    return {
        "red_n": len(red),
        "blue_n": len(blue),
        "red_hp_sum": float(sum(p["hp"] for p in red)),
        "blue_hp_sum": float(sum(p["hp"] for p in blue)),
        "red_army": red,
        "blue_army": blue,
    }


def _read_first_raw_observation_state(repeat_dir: Path) -> Optional[Dict[str, Any]]:
    episodes_path = repeat_dir / "episodes.jsonl"
    if not episodes_path.exists():
        return None
    try:
        with open(str(episodes_path), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                frames = record.get("frames") or []
                if not frames:
                    return None
                frame = frames[0] if isinstance(frames[0], dict) else {}
                raw_obs = frame.get("raw_observation") if isinstance(frame, dict) else {}
                state = (raw_obs or {}).get("state_norm") or frame.get("state_norm")
            return state if isinstance(state, dict) else None
    except Exception:
        return None
    return None


def _episode_first_frame_records(repeat_dir: Path) -> List[Dict[str, Any]]:
    episodes_path = repeat_dir / "episodes.jsonl"
    records: List[Dict[str, Any]] = []
    if not episodes_path.exists():
        return records
    with open(str(episodes_path), "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                records.append(
                    {
                        "line_number": line_number,
                        "episode_id": None,
                        "resolved": False,
                        "error": "json_decode_error",
                    }
                )
                continue
            frames = episode.get("frames") or []
            frame = frames[0] if frames and isinstance(frames[0], dict) else {}
            state_id_sequence = episode.get("state_id_sequence") or []
            state_key_sequence = episode.get("state_key_sequence") or []
            raw_observation = frame.get("raw_observation") if isinstance(frame, dict) else {}
            raw_state = (raw_observation or {}).get("state_norm") if isinstance(raw_observation, dict) else None
            eval_state = frame.get("eval_bktree_norm_state") if isinstance(frame, dict) else None
            state = raw_state if isinstance(raw_state, dict) else eval_state
            source = "raw_observation.state_norm" if isinstance(raw_state, dict) else (
                "eval_bktree_norm_state" if isinstance(eval_state, dict) else ""
            )
            local_state_id = None
            if state_id_sequence:
                try:
                    local_state_id = int(state_id_sequence[0])
                except (TypeError, ValueError):
                    local_state_id = None
            frame_eval_state_id = None
            try:
                frame_eval_state_id = int(frame.get("eval_state_id")) if frame.get("eval_state_id") is not None else None
            except (TypeError, ValueError):
                frame_eval_state_id = None
            records.append(
                {
                    "line_number": line_number,
                    "episode_id": episode.get("episode_id", line_number),
                    "result": episode.get("result"),
                    "score": episode.get("score"),
                    "frame_count": len(frames),
                    "first_game_loop": frame.get("game_loop") if isinstance(frame, dict) else None,
                    "local_state_id_sequence_first": local_state_id,
                    "frame_eval_state_id": frame_eval_state_id,
                    "state_key_sequence_first": state_key_sequence[0] if state_key_sequence else None,
                    "frame_state_key": frame.get("state_key") if isinstance(frame, dict) else None,
                    "state_source": source,
                    "resolved": isinstance(state, dict),
                    "state": state if isinstance(state, dict) else None,
                    "raw_observation": raw_observation if isinstance(raw_observation, dict) else None,
                    "my_units_pos": frame.get("my_units_pos") if isinstance(frame, dict) else None,
                    "enemy_units_pos": frame.get("enemy_units_pos") if isinstance(frame, dict) else None,
                }
            )
    return records


def _resolve_baseline_start(
    baseline: Mapping[str, Any],
    primary_threshold: float,
    secondary_threshold: float,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    baseline_bktree = Path(baseline["bktree_dir"])
    baseline_start_id = _parse_first_node_log_id(baseline_bktree / "node_log.txt")
    if baseline_start_id is None:
        baseline_start_id = 0
    baseline_state = _state_for_id(
        baseline_bktree,
        int(baseline_start_id),
        primary_threshold,
        secondary_threshold,
    )
    return int(baseline_start_id), baseline_state


def _first_reference_baseline(
    baselines: List[Dict[str, Any]],
    primary_threshold: float,
    secondary_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[Dict[str, Any]]]:
    for baseline in baselines:
        baseline_start_id, baseline_state = _resolve_baseline_start(
            baseline,
            primary_threshold,
            secondary_threshold,
        )
        if baseline_state is not None:
            return baseline, baseline_start_id, baseline_state
    return None, None, None


def _write_episode_start_validation_files(
    repeat_dir: Path,
    rows: List[Dict[str, Any]],
    summary: Mapping[str, Any],
) -> Tuple[str, str]:
    json_path = repeat_dir / "episode_start_state_validation.json"
    csv_path = repeat_dir / "episode_start_state_validation.csv"
    _write_json(json_path, {"schema_version": 1, "summary": summary, "episodes": rows})
    csv_columns = [
        "episode_id",
        "line_number",
        "result",
        "score",
        "frame_count",
        "first_game_loop",
        "local_state_id_sequence_first",
        "frame_eval_state_id",
        "state_key_sequence_first",
        "frame_state_key",
        "state_source",
        "resolved",
        "same_unified_state_as_reference",
        "local_unified_id",
        "reference_unified_id",
        "distribution_distance",
        "health_distance",
        "within_primary_threshold",
        "within_secondary_threshold",
        "raw_start_delta_from_episode1",
        "first_frame_order_count",
        "start_state_raw_shifted",
        "start_state_has_orders",
        "exclude_for_start_state_analysis",
        "start_state_scatter_plot",
        "start_state_scatter_error",
        "raw_observation",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(csv_path), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["raw_observation"] = json.dumps(
                row.get("raw_observation"),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            writer.writerow(row_out)
    return str(json_path), str(csv_path)


def _stable_json_hash(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        payload = str(value)
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _rank_rows_by_score(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    sortable = []
    for idx, row in enumerate(rows):
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            score = float("-inf")
        sortable.append((idx, score))
    sortable.sort(key=lambda item: (-item[1], item[0]))
    return {id(rows[idx]): rank for rank, (idx, _) in enumerate(sortable, start=1)}


def _write_episode_start_analysis_files(
    repeat_dir: Path,
    rows: List[Dict[str, Any]],
    summary: Mapping[str, Any],
) -> Tuple[str, str]:
    json_path = repeat_dir / "episode_start_state_analysis.json"
    csv_path = repeat_dir / "episode_start_state_analysis.csv"
    all_ranks = _rank_rows_by_score(rows)
    legal_rows = [
        row
        for row in rows
        if row.get("resolved")
        and row.get("same_unified_state_as_reference") is not False
        and not row.get("exclude_for_start_state_analysis")
    ]
    legal_ranks = _rank_rows_by_score(legal_rows)
    legal_rank_by_row = {id(row): legal_ranks[id(row)] for row in legal_rows}

    records: List[Dict[str, Any]] = []
    for row in rows:
        raw_observation = row.get("raw_observation") if isinstance(row.get("raw_observation"), Mapping) else {}
        raw_state_norm = raw_observation.get("state_norm") if isinstance(raw_observation, Mapping) else None
        exclude_reasons = [
            reason
            for reason, enabled in (
                ("raw_start_shifted", row.get("start_state_raw_shifted")),
                ("first_frame_has_orders", row.get("start_state_has_orders")),
                ("unresolved_state", not row.get("resolved")),
                ("different_unified_start_state", row.get("same_unified_state_as_reference") is False),
            )
            if enabled
        ]
        record = {
            "episode_id": row.get("episode_id"),
            "line_number": row.get("line_number"),
            "result": row.get("result"),
            "final_score": row.get("score"),
            "fitness_rank": all_ranks.get(id(row)),
            "legal_fitness_rank": legal_rank_by_row.get(id(row)),
            "is_legal_for_start_state_analysis": id(row) in legal_rank_by_row,
            "exclude_for_start_state_analysis": bool(row.get("exclude_for_start_state_analysis")),
            "exclude_reasons": exclude_reasons,
            "first_game_loop": row.get("first_game_loop"),
            "first_frame_order_count": row.get("first_frame_order_count"),
            "raw_start_delta_from_episode1": row.get("raw_start_delta_from_episode1"),
            "raw_state_norm_hash": _stable_json_hash(raw_state_norm),
            "raw_observation_hash": _stable_json_hash(raw_observation),
            "raw_state_norm": raw_state_norm,
            "local_bktree_state_id": row.get("local_state_id_sequence_first"),
            "frame_eval_state_id": row.get("frame_eval_state_id"),
            "local_unified_id": row.get("local_unified_id"),
            "reference_unified_id": row.get("reference_unified_id"),
            "distribution_distance": row.get("distribution_distance"),
            "health_distance": row.get("health_distance"),
            "state_key_sequence_first": row.get("state_key_sequence_first"),
            "frame_state_key": row.get("frame_state_key"),
            "state_source": row.get("state_source"),
            "start_state_scatter_plot": row.get("start_state_scatter_plot"),
        }
        records.append(record)

    top_legal = sorted(
        (record for record in records if record.get("is_legal_for_start_state_analysis")),
        key=lambda item: int(item.get("legal_fitness_rank") or 10**9),
    )
    analysis_summary = {
        "schema_version": 1,
        "episode_count": len(records),
        "legal_episode_count": len(top_legal),
        "legal_top100_episode_ids": [record.get("episode_id") for record in top_legal[:100]],
        "filter_rule": summary.get("raw_start_filter_rule"),
        "source_validation_summary": {
            key: summary.get(key)
            for key in (
                "status",
                "episode_count",
                "matched_count",
                "mismatch_count",
                "unresolved_count",
                "raw_start_shift_count",
                "raw_start_order_count",
                "raw_start_exclude_count",
                "max_raw_start_delta_from_episode1",
                "mean_raw_start_delta_from_episode1",
            )
        },
    }
    _write_json(json_path, {"summary": analysis_summary, "episodes": records})

    csv_columns = [
        "episode_id",
        "line_number",
        "result",
        "final_score",
        "fitness_rank",
        "legal_fitness_rank",
        "is_legal_for_start_state_analysis",
        "exclude_for_start_state_analysis",
        "exclude_reasons",
        "first_game_loop",
        "first_frame_order_count",
        "raw_start_delta_from_episode1",
        "raw_state_norm_hash",
        "raw_observation_hash",
        "local_bktree_state_id",
        "frame_eval_state_id",
        "local_unified_id",
        "reference_unified_id",
        "distribution_distance",
        "health_distance",
        "state_key_sequence_first",
        "frame_state_key",
        "state_source",
        "start_state_scatter_plot",
    ]
    with open(str(csv_path), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row_out = dict(record)
            row_out["exclude_reasons"] = ";".join(str(v) for v in record.get("exclude_reasons", []))
            writer.writerow(row_out)
    return str(json_path), str(csv_path)


def _raw_units_from_observation(raw_observation: Optional[Mapping[str, Any]], key: str) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    if not isinstance(raw_observation, Mapping):
        return points
    for unit in raw_observation.get(key, []) or []:
        try:
            points.append(
                {
                    "x": float(unit.get("x", 0.0)),
                    "y": float(unit.get("y", 0.0)),
                    "hp": float(unit.get("health", unit.get("hp", 0.0))),
                    "order_length": float(unit.get("order_length", 0.0)),
                }
            )
        except Exception:
            continue
    return points


def _raw_summary_from_observation(raw_observation: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    red = _raw_units_from_observation(raw_observation, "my_units")
    blue = _raw_units_from_observation(raw_observation, "enemy_units")
    return {
        "red_n": len(red),
        "blue_n": len(blue),
        "red_hp_sum": float(sum(p["hp"] for p in red)),
        "blue_hp_sum": float(sum(p["hp"] for p in blue)),
        "red_army": red,
        "blue_army": blue,
    }


def _raw_start_max_delta(
    local_summary: Optional[Mapping[str, Any]],
    reference_summary: Optional[Mapping[str, Any]],
) -> Optional[float]:
    if not isinstance(local_summary, Mapping) or not isinstance(reference_summary, Mapping):
        return None
    max_delta = 0.0
    for key in ("red_army", "blue_army"):
        local_points = sorted(
            [
                (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("hp", 0.0)))
                for p in local_summary.get(key, []) or []
            ]
        )
        reference_points = sorted(
            [
                (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("hp", 0.0)))
                for p in reference_summary.get(key, []) or []
            ]
        )
        if len(local_points) != len(reference_points):
            return float("inf")
        for local_point, reference_point in zip(local_points, reference_points):
            max_delta = max(
                max_delta,
                abs(local_point[0] - reference_point[0]),
                abs(local_point[1] - reference_point[1]),
                abs(local_point[2] - reference_point[2]),
            )
    return max_delta


def _raw_start_order_count(raw_observation: Optional[Mapping[str, Any]]) -> int:
    if not isinstance(raw_observation, Mapping):
        return 0
    total = 0
    for key in ("my_units", "enemy_units"):
        for unit in raw_observation.get(key, []) or []:
            try:
                total += int(float(unit.get("order_length", 0.0)))
            except Exception:
                continue
    return total


def _raw_summary_from_norm_state(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = _state_summary(state)
    return {
        "red_n": summary.get("red_n", 0),
        "blue_n": summary.get("blue_n", 0),
        "red_hp_sum": float(summary.get("red_hp_sum", 0.0)) * 45.0,
        "blue_hp_sum": float(summary.get("blue_hp_sum", 0.0)) * 45.0,
        "red_army": [
            {"x": p["x"], "y": p["y"], "hp": p["hp"] * 45.0}
            for p in _norm_points_to_raw(summary, "red_army")
        ],
        "blue_army": [
            {"x": p["x"], "y": p["y"], "hp": p["hp"] * 45.0}
            for p in _norm_points_to_raw(summary, "blue_army")
        ],
    }


def _plot_raw_state_pair(
    local_summary: Mapping[str, Any],
    reference_summary: Optional[Mapping[str, Any]],
    out_png: Path,
    title: str,
    subtitle: str,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    panels = [("FinalEval episode first frame", local_summary)]
    if reference_summary is not None:
        panels.append(("Reference baseline start", reference_summary))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), squeeze=False)
    red_color = "#d62728"
    blue_color = "#1f77b4"
    all_x: List[float] = []
    all_y: List[float] = []
    for _, summary in panels:
        for key in ("red_army", "blue_army"):
            for point in summary.get(key, []) or []:
                all_x.append(float(point["x"]))
                all_y.append(float(point["y"]))
    x_pad = 2.0
    y_pad = 2.0
    xlim = (min(all_x) - x_pad, max(all_x) + x_pad) if all_x else (0, 128)
    ylim = (min(all_y) - y_pad, max(all_y) + y_pad) if all_y else (0, 128)
    for ax, (panel_title, summary) in zip(axes.ravel(), panels):
        red = summary.get("red_army", []) or []
        blue = summary.get("blue_army", []) or []
        if red:
            ax.scatter(
                [p["x"] for p in red],
                [p["y"] for p in red],
                c=red_color,
                marker="o",
                s=[max(35, float(p.get("hp", 0.0)) * 1.6) for p in red],
                label="self/red",
                alpha=0.85,
            )
            for idx, p in enumerate(red):
                ax.text(p["x"], p["y"], f"R{idx}:{p.get('hp', 0):.0f}", fontsize=7, color=red_color)
        if blue:
            ax.scatter(
                [p["x"] for p in blue],
                [p["y"] for p in blue],
                c=blue_color,
                marker="^",
                s=[max(35, float(p.get("hp", 0.0)) * 1.6) for p in blue],
                label="enemy/blue",
                alpha=0.85,
            )
            for idx, p in enumerate(blue):
                ax.text(p["x"], p["y"], f"B{idx}:{p.get('hp', 0):.0f}", fontsize=7, color=blue_color)
        ax.set_title(
            f"{panel_title}\nHP {summary.get('red_hp_sum', 0):.0f}/{summary.get('blue_hp_sum', 0):.0f}",
            fontsize=9,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("raw x")
        ax.set_ylabel("raw y")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout(rect=(0, 0.0, 1, 0.88))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=170)
    plt.close(fig)
    return None


def _write_episode_start_scatter_plots(
    repeat_dir: Path,
    rows: List[Dict[str, Any]],
    reference_state: Optional[Mapping[str, Any]],
) -> str:
    scatter_dir = repeat_dir / "episode_start_state_scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    reference_summary = _raw_summary_from_norm_state(reference_state) if reference_state is not None else None
    for row in rows:
        raw_observation = row.get("raw_observation")
        local_summary = _raw_summary_from_observation(raw_observation)
        episode_id = row.get("episode_id")
        try:
            episode_text = f"{int(episode_id):03d}"
        except Exception:
            episode_text = str(episode_id or row.get("line_number") or "unknown")
        if row.get("exclude_for_start_state_analysis"):
            status_text = "raw_shift"
        else:
            status_text = "match" if row.get("same_unified_state_as_reference") else "mismatch"
        out_png = scatter_dir / f"episode_{episode_text}_{status_text}_start_state.png"
        subtitle = (
            f"loop={row.get('first_game_loop')} "
            f"local_state={row.get('local_state_id_sequence_first')} "
            f"dist={row.get('distribution_distance', '-')}; "
            f"hp_dist={row.get('health_distance', '-')}; "
            f"raw_delta={row.get('raw_start_delta_from_episode1', '-')}; "
            f"orders={row.get('first_frame_order_count', '-')}"
        )
        error = _plot_raw_state_pair(
            local_summary,
            reference_summary,
            out_png,
            f"Episode {episode_id} first state ({status_text})",
            subtitle,
        )
        row["start_state_scatter_plot"] = str(out_png)
        if error:
            row["start_state_scatter_error"] = error
    return str(scatter_dir)


def _validate_episode_start_states(
    repeat_dir: Path,
    baselines: List[Dict[str, Any]],
    primary_threshold: float,
    secondary_threshold: float,
) -> Dict[str, Any]:
    reference, reference_start_id, reference_state = _first_reference_baseline(
        baselines,
        primary_threshold,
        secondary_threshold,
    )
    rows: List[Dict[str, Any]] = []
    unresolved_count = 0
    matched_count = 0
    mismatch_count = 0
    distances: List[float] = []
    hp_distances: List[float] = []

    for record in _episode_first_frame_records(repeat_dir):
        row = {k: v for k, v in record.items() if k != "state"}
        state = record.get("state")
        if reference is None or reference_state is None:
            row["same_unified_state_as_reference"] = None
        elif not isinstance(state, dict):
            unresolved_count += 1
            row["same_unified_state_as_reference"] = False
        else:
            same, local_unified, reference_unified = _same_unified_state(
                state,
                reference_state,
                primary_threshold,
                secondary_threshold,
            )
            dist, hp_dist = DistributionDistance(state, reference_state)()
            row.update(
                {
                    "same_unified_state_as_reference": bool(same),
                    "local_unified_id": int(local_unified),
                    "reference_unified_id": int(reference_unified),
                    "distribution_distance": float(dist),
                    "health_distance": float(hp_dist),
                    "within_primary_threshold": float(dist) <= float(primary_threshold),
                    "within_secondary_threshold": float(hp_dist) <= float(secondary_threshold),
                }
            )
            distances.append(float(dist))
            hp_distances.append(float(hp_dist))
            if same:
                matched_count += 1
            else:
                mismatch_count += 1
        rows.append(row)

    reference_raw_summary = None
    for row in rows:
        raw_summary = _raw_summary_from_observation(row.get("raw_observation"))
        if raw_summary.get("red_n") and raw_summary.get("blue_n"):
            reference_raw_summary = raw_summary
            break
    raw_shift_count = 0
    raw_order_count = 0
    raw_exclude_count = 0
    raw_deltas: List[float] = []
    for row in rows:
        raw_summary = _raw_summary_from_observation(row.get("raw_observation"))
        raw_delta = _raw_start_max_delta(raw_summary, reference_raw_summary)
        first_order_count = _raw_start_order_count(row.get("raw_observation"))
        raw_shifted = raw_delta is not None and raw_delta > 1e-9
        has_orders = first_order_count > 0
        exclude = bool(raw_shifted or has_orders)
        row.update(
            {
                "raw_start_delta_from_episode1": raw_delta,
                "first_frame_order_count": first_order_count,
                "start_state_raw_shifted": raw_shifted,
                "start_state_has_orders": has_orders,
                "exclude_for_start_state_analysis": exclude,
            }
        )
        if raw_delta is not None:
            raw_deltas.append(float(raw_delta))
        raw_shift_count += int(raw_shifted)
        raw_order_count += int(has_orders)
        raw_exclude_count += int(exclude)

    if reference is None or reference_state is None:
        status = "no_reference_baseline"
    elif not rows:
        status = "no_episode_records"
    elif unresolved_count or mismatch_count:
        status = "fail"
    else:
        status = "pass"

    summary: Dict[str, Any] = {
        "status": status,
        "episode_count": len(rows),
        "matched_count": matched_count,
        "mismatch_count": mismatch_count,
        "unresolved_count": unresolved_count,
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "reference": None,
        "max_distribution_distance": max(distances) if distances else None,
        "mean_distribution_distance": sum(distances) / len(distances) if distances else None,
        "max_health_distance": max(hp_distances) if hp_distances else None,
        "mean_health_distance": sum(hp_distances) / len(hp_distances) if hp_distances else None,
        "raw_start_shift_count": raw_shift_count,
        "raw_start_order_count": raw_order_count,
        "raw_start_exclude_count": raw_exclude_count,
        "max_raw_start_delta_from_episode1": max(raw_deltas) if raw_deltas else None,
        "mean_raw_start_delta_from_episode1": sum(raw_deltas) / len(raw_deltas) if raw_deltas else None,
        "raw_start_filter_rule": "exclude if first-frame raw coordinates differ from episode 1 or first-frame raw units already have orders",
    }
    if reference is not None:
        summary["reference"] = {
            "baseline_group": reference.get("baseline_group"),
            "method": reference.get("method"),
            "bktree_dir": str(reference.get("bktree_dir")),
            "eval_dir": str(reference.get("eval_dir")),
            "start_state_id": reference_start_id,
        }
    scatter_dir = _write_episode_start_scatter_plots(repeat_dir, rows, reference_state)
    summary["scatter_dir"] = scatter_dir
    json_path, csv_path = _write_episode_start_validation_files(repeat_dir, rows, summary)
    summary["report_json"] = json_path
    summary["report_csv"] = csv_path
    analysis_json_path, analysis_csv_path = _write_episode_start_analysis_files(
        repeat_dir,
        rows,
        summary,
    )
    summary["analysis_json"] = analysis_json_path
    summary["analysis_csv"] = analysis_csv_path
    return summary


def _norm_points_to_raw(summary: Mapping[str, Any], key: str) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for point in summary.get(key, []) or []:
        try:
            x_norm = float(point["x"])
            y_norm = float(point["y"])
            points.append(
                {
                    "x": (x_norm + 1.0) * 64.0,
                    "y": (1.0 - y_norm) * 64.0,
                    "hp": float(point.get("hp", 0.0)),
                }
            )
        except Exception:
            continue
    return points


def _plot_start_state_scatter(
    states: List[Dict[str, Any]],
    out_png: Path,
    out_json: Path,
    title: str,
) -> Optional[str]:
    payload = {
        "schema_version": 1,
        "title": title,
        "coordinate_space": "BKTree normalized state coordinates",
        "states": states,
    }
    _write_json(out_json, payload)
    valid_states = [item for item in states if item.get("resolved")]
    if not valid_states:
        return "no_resolved_state"
    try:
        import math
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        payload["plot_error"] = f"{type(exc).__name__}: {exc}"
        _write_json(out_json, payload)
        return str(exc)

    n = len(valid_states)
    cols = min(4, max(1, n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    red_color = "#d62728"
    blue_color = "#1f77b4"
    for ax in axes.ravel():
        ax.axis("off")
    for ax, item in zip(axes.ravel(), valid_states):
        summary = item["summary"]
        red = summary.get("red_army", [])
        blue = summary.get("blue_army", [])
        ax.axis("on")
        if red:
            ax.scatter([p["x"] for p in red], [p["y"] for p in red], c=red_color, marker="o", label="red/self", s=45)
            for idx, p in enumerate(red):
                ax.text(p["x"], p["y"], f"R{idx}", fontsize=7, color=red_color)
        if blue:
            ax.scatter([p["x"] for p in blue], [p["y"] for p in blue], c=blue_color, marker="^", label="blue/enemy", s=45)
            for idx, p in enumerate(blue):
                ax.text(p["x"], p["y"], f"B{idx}", fontsize=7, color=blue_color)
        ax.set_title(
            f"{item.get('label')}\nR{summary.get('red_n')}/B{summary.get('blue_n')} "
            f"HP {summary.get('red_hp_sum'):.1f}/{summary.get('blue_hp_sum'):.1f}",
            fontsize=9,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)
    return None


def _plot_start_state_raw_coordinate_scatter(
    states: List[Dict[str, Any]],
    out_png: Path,
    out_json: Path,
    title: str,
) -> Optional[str]:
    raw_states: List[Dict[str, Any]] = []
    for item in states:
        summary = item.get("summary") or {}
        raw_summary = {
            "red_n": summary.get("red_n", 0),
            "blue_n": summary.get("blue_n", 0),
            "red_hp_sum": summary.get("red_hp_sum", 0.0),
            "blue_hp_sum": summary.get("blue_hp_sum", 0.0),
            "red_army": _norm_points_to_raw(summary, "red_army"),
            "blue_army": _norm_points_to_raw(summary, "blue_army"),
        }
        raw_states.append({**item, "summary": raw_summary})
    payload = {
        "schema_version": 1,
        "title": title,
        "coordinate_space": "raw coordinates reconstructed from normalized state: x=(nx+1)*64, y=(1-ny)*64",
        "states": raw_states,
    }
    _write_json(out_json, payload)
    error = _plot_start_state_scatter(raw_states, out_png, out_json, title)
    _write_json(out_json, payload)
    return error


def _map_from_pymarl_eval(eval_dir: Path, archive_root: Path) -> str:
    summary = _read_json(eval_dir / "eval_summary.json")
    metrics = _read_json(eval_dir / "eval_metrics.json")
    if summary.get("map_name") or metrics.get("map_name"):
        return str(summary.get("map_name") or metrics.get("map_name"))
    name = archive_root.name
    patterns = [
        "MarineMicro_MvsM_4_dist_mirror",
        "MarineMicro_MvsM_4_dist",
        "MarineMicro_MvsM_4_mirror",
        "MarineMicro_MvsM_8_mirror",
        "MarineMicro_MvsM_8",
        "MarineMicro_MvsM_4",
    ]
    for pattern in patterns:
        if pattern in name:
            return pattern
    return ""


def _pymarl_archive_root(pymarl_root: Path, bktree_dir: Path) -> Path:
    try:
        rel = bktree_dir.relative_to(pymarl_root)
        return pymarl_root / rel.parts[0]
    except Exception:
        return pymarl_root


def _iter_baseline_bktrees(all_data_root: Path, map_id: str) -> Iterable[Dict[str, Any]]:
    pymarl_root = all_data_root / "pymarl"
    if pymarl_root.exists():
        for bktree_dir in sorted(p for p in pymarl_root.rglob("bktree") if (p / "node_log.txt").exists()):
            eval_dir = bktree_dir.parent
            archive_root = _pymarl_archive_root(pymarl_root, bktree_dir)
            detected_map = _map_from_pymarl_eval(eval_dir, archive_root)
            if detected_map != map_id:
                continue
            archive_manifest = _read_json(archive_root / "archive_manifest.json")
            eval_summary = _read_json(eval_dir / "eval_summary.json")
            yield {
                "baseline_group": "PyMARL",
                "method": str(archive_manifest.get("method") or eval_summary.get("method") or archive_root.name),
                "bktree_dir": bktree_dir,
                "eval_dir": eval_dir,
            }

    onpolicy_root = all_data_root / "onpolicy"
    if onpolicy_root.exists():
        for bktree_dir in sorted(p for p in onpolicy_root.rglob("bktree") if (p / "node_log.txt").exists()):
            eval_dir = bktree_dir.parent
            eval_summary = _read_json(eval_dir / "eval_summary.json")
            eval_metrics = _read_json(eval_dir / "eval_metrics.json")
            detected_map = str(eval_summary.get("map_name") or eval_metrics.get("map_name") or "")
            if not detected_map:
                try:
                    rel = eval_dir.relative_to(onpolicy_root).parts
                    detected_map = rel[0] if rel else ""
                except Exception:
                    detected_map = ""
            if detected_map != map_id:
                continue
            method = str(
                eval_summary.get("algorithm_name")
                or eval_metrics.get("algorithm_name")
                or _infer_onpolicy_method(onpolicy_root, eval_dir)
                or "onpolicy"
            )
            yield {
                "baseline_group": "OnPolicy",
                "method": method,
                "bktree_dir": bktree_dir,
                "eval_dir": eval_dir,
            }


def _infer_onpolicy_method(onpolicy_root: Path, eval_dir: Path) -> str:
    try:
        rel = eval_dir.relative_to(onpolicy_root).parts
        return rel[1] if len(rel) > 1 else ""
    except Exception:
        return ""


def _same_unified_state(
    local_state: Mapping[str, Any],
    baseline_state: Mapping[str, Any],
    primary_threshold: float,
    secondary_threshold: float,
) -> Tuple[bool, int, int]:
    builder = UnifiedBKTreeBuilder(primary_threshold, secondary_threshold)
    local_id, _ = builder.insert_or_query(local_state)
    baseline_id, _ = builder.insert_or_query(baseline_state)
    return local_id == baseline_id, local_id, baseline_id


def _repeat_dirs(eval_dir: Path) -> List[Path]:
    repeat_root = eval_dir / "repeats"
    repeats = sorted(p for p in repeat_root.glob("repeat_*") if p.is_dir()) if repeat_root.exists() else []
    return repeats or [eval_dir]


def validate_final_eval_start_state(
    eval_dir: Path,
    experiment_dir: Optional[Path] = None,
    all_data_root: Path = ALL_DATA_ROOT,
) -> Dict[str, Any]:
    eval_dir = eval_dir.resolve()
    experiment_dir = experiment_dir.resolve() if experiment_dir else eval_dir.parent.parent
    manifest = _read_json(experiment_dir / MANIFEST_NAME)
    summary = _read_json(eval_dir / "final_eval_summary.json")
    map_id = str(summary.get("map_id") or manifest.get("map_id") or "")
    primary_threshold, secondary_threshold = _thresholds_for_map(map_id, None, None)
    baselines = list(_iter_baseline_bktrees(all_data_root.resolve(), map_id))
    source_bktree_dir = _source_etg_bktree_dir(map_id)
    source_state = (
        _state_for_id(source_bktree_dir, 0, primary_threshold, secondary_threshold)
        if source_bktree_dir is not None
        else None
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "check": "final_eval_start_state_fused_bktree_compatibility",
        "experiment_dir": str(experiment_dir),
        "eval_dir": str(eval_dir),
        "map_id": map_id,
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "baseline_count": len(baselines),
        "repeats": [],
    }

    any_repeat_resolved = False
    repeat_statuses: List[str] = []
    for repeat_dir in _repeat_dirs(eval_dir):
        bktree_dir = repeat_dir / "bktree"
        local_state_id = 0
        local_state = _state_for_id(bktree_dir, local_state_id, primary_threshold, secondary_threshold)
        repeat_report: Dict[str, Any] = {
            "repeat_id": repeat_dir.name,
            "repeat_dir": str(repeat_dir),
            "bktree_dir": str(bktree_dir),
            "local_start_state_id": local_state_id,
            "local_start_cluster": list(_parse_state_node(bktree_dir / "state_node.txt").get(local_state_id, ())),
            "local_start_resolved": local_state is not None,
            "comparisons": [],
        }
        scatter_states: List[Dict[str, Any]] = [
            {
                "label": "FinalEval local",
                "group": "PredictionRTS",
                "method": "local",
                "path": str(bktree_dir),
                "state_id": local_state_id,
                "resolved": local_state is not None,
                "summary": _state_summary(local_state),
            }
        ]
        raw_observation_state = _read_first_raw_observation_state(repeat_dir)
        scatter_states.append(
            {
                "label": "FinalEval raw obs",
                "group": "PredictionRTS",
                "method": "raw_observation_state_norm",
                "path": str(repeat_dir / "episodes.jsonl"),
                "state_id": None,
                "resolved": raw_observation_state is not None,
                "summary": _state_summary(raw_observation_state),
            }
        )
        if source_bktree_dir is not None:
            scatter_states.append(
                {
                    "label": "Source ETG",
                    "group": "PredictionRTS",
                    "method": "source_etg",
                    "path": str(source_bktree_dir),
                    "state_id": 0,
                    "resolved": source_state is not None,
                    "summary": _state_summary(source_state),
                }
            )
        if local_state is not None:
            any_repeat_resolved = True
        for baseline in baselines:
            baseline_bktree = Path(baseline["bktree_dir"])
            baseline_start_id, baseline_state = _resolve_baseline_start(
                baseline,
                primary_threshold,
                secondary_threshold,
            )
            row: Dict[str, Any] = {
                "baseline_group": baseline.get("baseline_group"),
                "method": baseline.get("method"),
                "bktree_dir": str(baseline_bktree),
                "baseline_start_state_id": baseline_start_id,
                "baseline_start_resolved": baseline_state is not None,
                "same_unified_state": False,
            }
            if local_state is not None and baseline_state is not None:
                same, local_unified, baseline_unified = _same_unified_state(
                    local_state,
                    baseline_state,
                    primary_threshold,
                    secondary_threshold,
                )
                dist, hp_dist = DistributionDistance(local_state, baseline_state)()
                row.update(
                    {
                        "same_unified_state": bool(same),
                        "local_unified_id": int(local_unified),
                        "baseline_unified_id": int(baseline_unified),
                        "distribution_distance": float(dist),
                        "health_distance": float(hp_dist),
                        "within_primary_threshold": float(dist) <= float(primary_threshold),
                        "within_secondary_threshold": float(hp_dist) <= float(secondary_threshold),
                    }
                )
            repeat_report["comparisons"].append(row)
            if baseline_state is not None:
                scatter_states.append(
                    {
                        "label": f"{baseline.get('baseline_group')} {baseline.get('method')}",
                        "group": baseline.get("baseline_group"),
                        "method": baseline.get("method"),
                        "path": str(baseline_bktree),
                        "state_id": baseline_start_id,
                        "resolved": True,
                        "summary": _state_summary(baseline_state),
                    }
                )
        scatter_png = repeat_dir / "start_state_scatter_comparison.png"
        scatter_json = repeat_dir / "start_state_scatter_comparison.json"
        raw_scatter_png = repeat_dir / "start_state_raw_coordinate_comparison.png"
        raw_scatter_json = repeat_dir / "start_state_raw_coordinate_comparison.json"
        plot_error = _plot_start_state_scatter(
            scatter_states,
            scatter_png,
            scatter_json,
            f"{map_id} start-state comparison ({repeat_dir.name})",
        )
        raw_plot_error = _plot_start_state_raw_coordinate_scatter(
            scatter_states,
            raw_scatter_png,
            raw_scatter_json,
            f"{map_id} start-state raw-coordinate comparison ({repeat_dir.name})",
        )
        repeat_report["scatter_plot"] = str(scatter_png)
        repeat_report["scatter_data"] = str(scatter_json)
        repeat_report["raw_coordinate_scatter_plot"] = str(raw_scatter_png)
        repeat_report["raw_coordinate_scatter_data"] = str(raw_scatter_json)
        if plot_error:
            repeat_report["scatter_plot_error"] = plot_error
        if raw_plot_error:
            repeat_report["raw_coordinate_scatter_plot_error"] = raw_plot_error
        episode_start_validation = _validate_episode_start_states(
            repeat_dir,
            baselines,
            primary_threshold,
            secondary_threshold,
        )
        repeat_report["episode_start_validation"] = episode_start_validation
        if len(report["repeats"]) == 0:
            eval_scatter_png = eval_dir / "start_state_scatter_comparison.png"
            eval_scatter_json = eval_dir / "start_state_scatter_comparison.json"
            eval_raw_scatter_png = eval_dir / "start_state_raw_coordinate_comparison.png"
            eval_raw_scatter_json = eval_dir / "start_state_raw_coordinate_comparison.json"
            eval_episode_json = eval_dir / "episode_start_state_validation.json"
            eval_episode_csv = eval_dir / "episode_start_state_validation.csv"
            eval_analysis_json = eval_dir / "episode_start_state_analysis.json"
            eval_analysis_csv = eval_dir / "episode_start_state_analysis.csv"
            try:
                if scatter_png.exists():
                    shutil.copy2(str(scatter_png), str(eval_scatter_png))
                if scatter_json.exists():
                    shutil.copy2(str(scatter_json), str(eval_scatter_json))
                if raw_scatter_png.exists():
                    shutil.copy2(str(raw_scatter_png), str(eval_raw_scatter_png))
                if raw_scatter_json.exists():
                    shutil.copy2(str(raw_scatter_json), str(eval_raw_scatter_json))
                episode_json = Path(str(episode_start_validation.get("report_json", "")))
                episode_csv = Path(str(episode_start_validation.get("report_csv", "")))
                if episode_json.exists():
                    shutil.copy2(str(episode_json), str(eval_episode_json))
                    repeat_report["eval_level_episode_start_report_json"] = str(eval_episode_json)
                if episode_csv.exists():
                    shutil.copy2(str(episode_csv), str(eval_episode_csv))
                    repeat_report["eval_level_episode_start_report_csv"] = str(eval_episode_csv)
                analysis_json = Path(str(episode_start_validation.get("analysis_json", "")))
                analysis_csv = Path(str(episode_start_validation.get("analysis_csv", "")))
                if analysis_json.exists():
                    shutil.copy2(str(analysis_json), str(eval_analysis_json))
                    repeat_report["eval_level_episode_start_analysis_json"] = str(eval_analysis_json)
                if analysis_csv.exists():
                    shutil.copy2(str(analysis_csv), str(eval_analysis_csv))
                    repeat_report["eval_level_episode_start_analysis_csv"] = str(eval_analysis_csv)
                repeat_report["eval_level_scatter_plot"] = str(eval_scatter_png)
                repeat_report["eval_level_scatter_data"] = str(eval_scatter_json)
                repeat_report["eval_level_raw_coordinate_scatter_plot"] = str(eval_raw_scatter_png)
                repeat_report["eval_level_raw_coordinate_scatter_data"] = str(eval_raw_scatter_json)
            except Exception as exc:
                repeat_report["eval_level_scatter_copy_error"] = f"{type(exc).__name__}: {exc}"
        local_zero_status = (
            "pass"
            if any(row.get("same_unified_state") for row in repeat_report["comparisons"])
            else ("no_baseline" if not baselines else "fail")
        )
        repeat_report["local_zero_status"] = local_zero_status
        repeat_report["status"] = (
            "pass"
            if local_zero_status == "pass"
            and episode_start_validation.get("status") == "pass"
            else ("no_baseline" if not baselines else "fail")
        )
        repeat_statuses.append(str(repeat_report["status"]))
        _write_json(repeat_dir / "start_state_validation.json", repeat_report)
        _write_md(repeat_dir / "start_state_validation.md", {**report, "repeats": [repeat_report], "overall_status": repeat_report["status"]})
        report["repeats"].append(repeat_report)

    if not baselines:
        report["overall_status"] = "no_baseline"
    elif repeat_statuses and all(status == "pass" for status in repeat_statuses):
        report["overall_status"] = "pass"
    elif not any_repeat_resolved:
        report["overall_status"] = "local_start_unresolved"
    else:
        report["overall_status"] = "fail"
    first_repeat = report["repeats"][0] if report.get("repeats") else {}
    if first_repeat.get("eval_level_scatter_plot"):
        report["scatter_plot"] = first_repeat.get("eval_level_scatter_plot")
    if first_repeat.get("eval_level_scatter_data"):
        report["scatter_data"] = first_repeat.get("eval_level_scatter_data")
    if first_repeat.get("eval_level_raw_coordinate_scatter_plot"):
        report["raw_coordinate_scatter_plot"] = first_repeat.get("eval_level_raw_coordinate_scatter_plot")
    if first_repeat.get("eval_level_raw_coordinate_scatter_data"):
        report["raw_coordinate_scatter_data"] = first_repeat.get("eval_level_raw_coordinate_scatter_data")
    if first_repeat.get("eval_level_episode_start_report_json"):
        report["episode_start_report_json"] = first_repeat.get("eval_level_episode_start_report_json")
    if first_repeat.get("eval_level_episode_start_report_csv"):
        report["episode_start_report_csv"] = first_repeat.get("eval_level_episode_start_report_csv")
    if first_repeat.get("eval_level_episode_start_analysis_json"):
        report["episode_start_analysis_json"] = first_repeat.get("eval_level_episode_start_analysis_json")
    if first_repeat.get("eval_level_episode_start_analysis_csv"):
        report["episode_start_analysis_csv"] = first_repeat.get("eval_level_episode_start_analysis_csv")
    _write_json(eval_dir / "start_state_validation.json", report)
    _write_md(eval_dir / "start_state_validation.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final-eval start-state compatibility with baselines")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--all-data-root", default=str(ALL_DATA_ROOT))
    args = parser.parse_args()

    report = validate_final_eval_start_state(
        Path(args.eval_dir),
        Path(args.experiment_dir) if args.experiment_dir else None,
        Path(args.all_data_root),
    )
    print(json.dumps({k: report[k] for k in ("map_id", "baseline_count", "overall_status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
