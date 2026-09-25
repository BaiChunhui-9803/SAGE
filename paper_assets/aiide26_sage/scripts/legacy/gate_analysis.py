#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Boxplot-aligned Gated Exploration state-quality analysis.

This script is deliberately tied to the exact data used by
``gate_selected_corr_and_etg_delta``:

- ``cache/selected_gate_positive_episodes.csv`` for Synergy;
- ``cache/selected_etg_counterpart_episodes.csv`` for etg-only.

No training trials, no full final-eval episodes, and no additional top-k
selection are added.  Any state-level or gate-level conclusion from this script
therefore shares the same sample source and filtering rule as the boxplot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_DIR = (
    PROJECT_ROOT
    / "output"
    / "learner_results"
    / "all_data"
    / "gate_score_correlation_selected_web_batch_20260622_141137"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "figures" / "gate_boxplot_aligned_state_quality_20260626"

SCENE_ORDER = ["sce-1", "sce-1m", "sce-2", "sce-2m", "sce-3", "sce-3m"]
SCENE_LABEL = {
    "sce-1": "sce1",
    "sce-1m": "sce1m",
    "sce-2": "sce2",
    "sce-2m": "sce2m",
    "sce-3": "sce3",
    "sce-3m": "sce3m",
}
RELATION_ORDER = [
    "source_etg_supported",
    "etg_selected_outside_source_etg",
    "shared_selected_outside_source_etg",
    "synergy_selected_outside_source_etg",
]
RELATION_LABEL = {
    "source_etg_supported": "Source ETG supported",
    "etg_selected_outside_source_etg": "etg-only outside ETG",
    "shared_selected_outside_source_etg": "Shared outside ETG",
    "synergy_selected_outside_source_etg": "Synergy outside ETG",
}
RELATION_SHORT = {
    "source_etg_supported": "Source",
    "etg_selected_outside_source_etg": "etg-only",
    "shared_selected_outside_source_etg": "Shared",
    "synergy_selected_outside_source_etg": "Synergy",
}
COLORS = {
    "source_etg_supported": "#94A3B8",
    "etg_selected_outside_source_etg": "#60A5FA",
    "shared_selected_outside_source_etg": "#F59E0B",
    "synergy_selected_outside_source_etg": "#EF4444",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def canonical_state(norm_state: Any) -> str:
    if not isinstance(norm_state, dict):
        return ""
    parts: List[Any] = []
    for side in ("red_army", "blue_army"):
        units = norm_state.get(side) or []
        clean = []
        for unit in units:
            try:
                clean.append([round(float(x), 5) for x in unit[:3]])
            except Exception:
                continue
        clean.sort()
        parts.append((side, clean))
    text = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, start=1):
            try:
                yield idx, json.loads(line)
            except Exception:
                continue


def episode_file(eval_dir: Path, repeat: str) -> Path:
    return eval_dir / "repeats" / str(repeat) / "episodes.jsonl"


def load_selected(selected_dir: Path) -> pd.DataFrame:
    cache_dir = selected_dir / "cache"
    syn = pd.read_csv(cache_dir / "selected_gate_positive_episodes.csv")
    etg = pd.read_csv(cache_dir / "selected_etg_counterpart_episodes.csv")
    syn["selected_source_file"] = "selected_gate_positive_episodes.csv"
    etg["selected_source_file"] = "selected_etg_counterpart_episodes.csv"
    data = pd.concat([syn, etg], ignore_index=True, sort=False)
    data["selection_key_exact"] = data.apply(
        lambda r: "|".join(
            [
                str(r.get("method", "")),
                str(r.get("scenario", "")),
                str(r.get("experiment_id", "")),
                str(r.get("eval_name", "")),
                str(r.get("repeat", "")),
                str(int(r.get("episode_id"))) if pd.notna(r.get("episode_id")) else "",
            ]
        ),
        axis=1,
    )
    return data


def load_exact_episodes(selected: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    wanted: Dict[Tuple[str, str], Dict[int, List[pd.Series]]] = {}
    for _, row in selected.iterrows():
        eval_dir = Path(str(row["eval_dir"]))
        repeat = str(row["repeat"])
        episode_id = int(row["episode_id"])
        wanted.setdefault((str(eval_dir), repeat), {}).setdefault(episode_id, []).append(row)

    episode_rows: List[Dict[str, Any]] = []
    frame_rows: List[Dict[str, Any]] = []
    for (eval_dir_str, repeat), by_episode in wanted.items():
        path = episode_file(Path(eval_dir_str), repeat)
        seen_episode_ids = set()
        for line_no, episode in iter_jsonl(path):
            episode_id = int(episode.get("episode_id", line_no))
            if episode_id not in by_episode:
                continue
            seen_episode_ids.add(episode_id)
            rows = by_episode[episode_id]
            for source_row in rows:
                score = to_float(episode.get("score", source_row.get("score")))
                result = str(episode.get("result", source_row.get("result", "")))
                frames = episode.get("frames") or []
                episode_rows.append(
                    {
                        "method": str(source_row["method"]),
                        "scenario": str(source_row["scenario"]),
                        "experiment_id": str(source_row["experiment_id"]),
                        "eval_name": str(source_row["eval_name"]),
                        "repeat": repeat,
                        "episode_id": episode_id,
                        "line_no_read": line_no,
                        "line_no_selected": int(source_row.get("line_no", line_no)),
                        "score": score,
                        "score_selected_csv": to_float(source_row.get("score")),
                        "result": result,
                        "win": 1.0 if result.lower() == "win" or bool(source_row.get("win", False)) else 0.0,
                        "frame_count": len(frames),
                        "selected_rank": to_float(source_row.get("selected_rank")),
                        "counterpart_rank": to_float(source_row.get("counterpart_rank")),
                        "selection_tier": str(source_row.get("selection_tier", "")),
                        "counterpart_rule": str(source_row.get("counterpart_rule", "")),
                        "selected_source_file": str(source_row.get("selected_source_file", "")),
                    }
                )
                for step, frame in enumerate(frames):
                    digest = canonical_state(frame.get("eval_bktree_norm_state"))
                    if not digest:
                        continue
                    denom = max(len(frames) - 1, 1)
                    game_progress = float(step / denom)
                    hp_my = to_float(frame.get("hp_my"))
                    hp_enemy = to_float(frame.get("hp_enemy"))
                    hp_margin = hp_my - hp_enemy if math.isfinite(hp_my) and math.isfinite(hp_enemy) else float("nan")
                    hp_total = hp_my + hp_enemy if math.isfinite(hp_my) and math.isfinite(hp_enemy) else float("nan")
                    shadow = frame.get("mechanism_shadow")
                    tuning = shadow.get("tuning") if isinstance(shadow, dict) and isinstance(shadow.get("tuning"), dict) else {}
                    changed = bool(frame.get("shadow_mechanism_changed_action"))
                    if isinstance(shadow, dict):
                        changed = changed or bool(shadow.get("mechanism_changed_action"))
                    status = str(frame.get("nid_status", "")).lower()
                    is_ood = bool(frame.get("nid_is_ood")) or status in {"ood", "bktree_rejected", "near_rejected", "missing"}
                    frame_rows.append(
                        {
                            "method": str(source_row["method"]),
                            "scenario": str(source_row["scenario"]),
                            "experiment_id": str(source_row["experiment_id"]),
                            "eval_name": str(source_row["eval_name"]),
                            "repeat": repeat,
                            "episode_id": episode_id,
                            "step": step,
                            "frame_count": len(frames),
                            "game_progress": game_progress,
                            "score": score,
                            "state_digest": digest,
                            "is_ood": bool(is_ood),
                            "nid_status": status,
                            "action_source": str(frame.get("action_source") or ""),
                            "hp_my": hp_my,
                            "hp_enemy": hp_enemy,
                            "hp_margin": hp_margin,
                            "hp_total": hp_total,
                            "hp_delta": to_float(frame.get("hp_delta")),
                            "gate_changed": bool(changed),
                            "gate_confidence": to_float(tuning.get("confidence")),
                            "gate_advantage": to_float(tuning.get("advantage")),
                            "gate_candidate_visits": to_float(tuning.get("candidate_visits")),
                            "gate_opportunity": bool(tuning.get("opportunity")),
                            "gate_validation": bool(tuning.get("validation")),
                            "tuning_reason": str(tuning.get("reason") or ""),
                        }
                    )
        missing = sorted(set(by_episode) - seen_episode_ids)
        if missing:
            raise RuntimeError(f"Missing selected episodes in {path}: {missing[:10]} ... total={len(missing)}")
    return pd.DataFrame(episode_rows), frame_rows


def classify_state_relation(methods: Sequence[str], ood_frames: int) -> str:
    method_set = set(methods)
    if int(ood_frames) == 0:
        return "source_etg_supported"
    if "Synergy" in method_set and "etg-only" in method_set:
        return "shared_selected_outside_source_etg"
    if "Synergy" in method_set:
        return "synergy_selected_outside_source_etg"
    return "etg_selected_outside_source_etg"


def aggregate_states(frame_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frames = pd.DataFrame(frame_rows)
    if frames.empty:
        return frames
    rows = []
    for (scenario, state_digest), group in frames.groupby(["scenario", "state_digest"], sort=False):
        methods = sorted(group["method"].dropna().astype(str).unique().tolist())
        ood_frames = int(group["is_ood"].sum())
        syn = group[group["method"].eq("Synergy")]
        rows.append(
            {
                "scenario": scenario,
                "state_digest": state_digest,
                "methods": ",".join(methods),
                "source_relation": classify_state_relation(methods, ood_frames),
                "frames": int(len(group)),
                "episodes": int(group[["method", "experiment_id", "eval_name", "repeat", "episode_id"]].drop_duplicates().shape[0]),
                "ood_frames": ood_frames,
                "ood_ratio": float(ood_frames / max(len(group), 1)),
                "mean_hp_my": group["hp_my"].mean(),
                "mean_hp_enemy": group["hp_enemy"].mean(),
                "mean_hp_margin": group["hp_margin"].mean(),
                "mean_hp_total": group["hp_total"].mean(),
                "mean_game_progress": group["game_progress"].mean(),
                "median_game_progress": group["game_progress"].median(),
                "mean_score_conditioned": group["score"].mean(),
                "synergy_frames": int(len(syn)),
                "synergy_ood_ratio": float(syn["is_ood"].mean()) if len(syn) else np.nan,
                "gate_changed_ratio": float(syn["gate_changed"].mean()) if len(syn) else np.nan,
                "gate_validation_ratio": float(syn["gate_validation"].mean()) if len(syn) else np.nan,
                "gate_opportunity_ratio": float(syn["gate_opportunity"].mean()) if len(syn) else np.nan,
                "mean_gate_confidence": syn["gate_confidence"].mean() if len(syn) else np.nan,
                "mean_gate_advantage": syn["gate_advantage"].mean() if len(syn) else np.nan,
                "mean_gate_candidate_visits": syn["gate_candidate_visits"].mean() if len(syn) else np.nan,
                "top_action_source": group["action_source"].mode().iloc[0] if not group["action_source"].mode().empty else "",
                "top_tuning_reason": syn["tuning_reason"].mode().iloc[0] if len(syn) and not syn["tuning_reason"].mode().empty else "",
            }
        )
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def state_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene in SCENE_ORDER:
        scene_part = states[states["scenario"].eq(scene)]
        total_frames = float(scene_part["frames"].sum())
        for relation in RELATION_ORDER:
            part = scene_part[scene_part["source_relation"].eq(relation)]
            rows.append(
                {
                    "scenario": scene,
                    "source_relation": relation,
                    "unique_states": int(len(part)),
                    "frames": int(part["frames"].sum()),
                    "frame_share": float(part["frames"].sum() / total_frames) if total_frames else 0.0,
                    "frame_weighted_hp_margin": weighted_mean(part["mean_hp_margin"], part["frames"]) if len(part) else np.nan,
                    "frame_weighted_hp_my": weighted_mean(part["mean_hp_my"], part["frames"]) if len(part) else np.nan,
                    "frame_weighted_hp_enemy": weighted_mean(part["mean_hp_enemy"], part["frames"]) if len(part) else np.nan,
                    "frame_weighted_game_progress": weighted_mean(part["mean_game_progress"], part["frames"]) if len(part) else np.nan,
                    "frame_weighted_score_conditioned": weighted_mean(part["mean_score_conditioned"], part["frames"]) if len(part) else np.nan,
                    "gate_changed_ratio": weighted_mean(part["gate_changed_ratio"], part["synergy_frames"]) if len(part) else np.nan,
                    "gate_validation_ratio": weighted_mean(part["gate_validation_ratio"], part["synergy_frames"]) if len(part) else np.nan,
                    "mean_gate_confidence": weighted_mean(part["mean_gate_confidence"], part["synergy_frames"]) if len(part) else np.nan,
                    "mean_gate_advantage": weighted_mean(part["mean_gate_advantage"], part["synergy_frames"]) if len(part) else np.nan,
                    "mean_gate_candidate_visits": weighted_mean(part["mean_gate_candidate_visits"], part["synergy_frames"]) if len(part) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def draw_score_boxplot(episodes: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.05))
    box_data, positions, colors = [], [], []
    for idx, scene in enumerate(SCENE_ORDER):
        part = episodes[episodes["scenario"].eq(scene)]
        etg = part[part["method"].eq("etg-only")]["score"].dropna().values
        syn = part[part["method"].eq("Synergy")]["score"].dropna().values
        box_data.extend([etg, syn])
        positions.extend([idx - 0.17, idx + 0.17])
        colors.extend(["#6F90B4", "#4FB477"])
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#334155",
            "markersize": 5.6,
            "markeredgewidth": 1.0,
        },
        medianprops={"color": "#263238", "linewidth": 1.35},
        whiskerprops={"color": "#4B5563", "linewidth": 1.15},
        capprops={"color": "#4B5563", "linewidth": 1.15},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_edgecolor("#3F4652")
        patch.set_linewidth(1.15)
    ax.set_xticks(np.arange(len(SCENE_ORDER)))
    ax.set_xticklabels([SCENE_LABEL[s] for s in SCENE_ORDER], fontsize=12)
    ax.set_xlabel("Scenario", fontsize=13)
    ax.set_ylabel("Episode score distribution", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#D8DEE7", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    handles = [
        mpl.patches.Patch(facecolor="#6F90B4", edgecolor="#3F4652", alpha=0.82, label="Graph-only planning"),
        mpl.patches.Patch(facecolor="#4FB477", edgecolor="#3F4652", alpha=0.82, label="SAGE (full)"),
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        fancybox=False,
        edgecolor="#CBD5E1",
        facecolor="white",
        framealpha=0.88,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.045),
        ncol=1,
        fontsize=11.5,
        borderpad=0.35,
        labelspacing=0.28,
        handlelength=1.25,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    save(fig, fig_dir, "fig1_reproduced_selected_score_boxplot")


def draw_relation_stack(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.25))
    x = np.arange(len(SCENE_ORDER))
    bottom = np.zeros(len(SCENE_ORDER))
    for relation in RELATION_ORDER:
        vals, margins = [], []
        for scene in SCENE_ORDER:
            row = summary[(summary["scenario"].eq(scene)) & (summary["source_relation"].eq(relation))]
            vals.append(float(row["frame_share"].iloc[0]) if len(row) else 0.0)
            margins.append(float(row["frame_weighted_hp_margin"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(x, vals, bottom=bottom, color=COLORS[relation], edgecolor="white", linewidth=0.6, label=RELATION_LABEL[relation])
        for idx, (bar, share, margin) in enumerate(zip(bars, vals, margins)):
            if share < 0.055 or not np.isfinite(margin):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[idx] + share / 2,
                f"{margin:+.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "boxstyle": "round,pad=0.12"},
            )
        bottom += np.asarray(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABEL[s] for s in SCENE_ORDER])
    ax.set_ylabel("Frame share in selected episodes")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    save(fig, fig_dir, "fig2_boxplot_aligned_state_relation_stack_with_hp_margin")


def draw_hp_margin_boxplot(states: pd.DataFrame, fig_dir: Path, min_frames: int) -> None:
    data = states[states["frames"].ge(min_frames)].copy()
    fig, axes = plt.subplots(1, 6, figsize=(11.4, 2.95), sharey=False)
    for ax, scene in zip(axes, SCENE_ORDER):
        part = data[data["scenario"].eq(scene)]
        values = [part[part["source_relation"].eq(rel)]["mean_hp_margin"].dropna().values for rel in RELATION_ORDER]
        if not any(len(v) for v in values):
            ax.set_axis_off()
            continue
        bp = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#111827", "linewidth": 1.1})
        for patch, relation in zip(bp["boxes"], RELATION_ORDER):
            patch.set_facecolor(COLORS[relation])
            patch.set_alpha(0.74)
            patch.set_edgecolor("#334155")
        ax.axhline(0, color="#64748B", linewidth=0.75, linestyle="--", alpha=0.55)
        ax.set_title(SCENE_LABEL[scene])
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.16)
        if ax is axes[0]:
            ax.set_ylabel("State-level HP margin")
    handles = [mpl.patches.Patch(facecolor=COLORS[r], edgecolor="#334155", label=RELATION_SHORT[r]) for r in RELATION_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, fig_dir, "fig3_boxplot_aligned_hp_margin_distribution")


def draw_hp_margin_with_progress(states: pd.DataFrame, fig_dir: Path, min_frames: int) -> None:
    data = states[states["frames"].ge(min_frames)].copy()
    rng = np.random.default_rng(20260626)
    fig, ax_hp = plt.subplots(figsize=(15.6, 3.55))
    hp_all = data["mean_hp_margin"].dropna().astype(float)
    if len(hp_all):
        hp_min = float(hp_all.min())
        hp_max = float(hp_all.max())
        hp_pad = max(4.0, (hp_max - hp_min) * 0.08)
        shared_ylim = (hp_min - hp_pad, hp_max + hp_pad)
    else:
        shared_ylim = (-8.0, 106.0)

    relation_gap = 1.12
    scene_gap = 1.25
    hp_offset = -0.24
    progress_offset = 0.34
    hp_positions_all: List[float] = []
    hp_values_all: List[np.ndarray] = []
    hp_relations_all: List[str] = []
    progress_payload: List[Tuple[float, str, np.ndarray]] = []
    scene_ranges: List[Tuple[float, float, float, str]] = []
    scene_progress_ranges: Dict[str, Tuple[float, float]] = {}

    for scene_idx, scene in enumerate(SCENE_ORDER):
        part = data[data["scenario"].eq(scene)].copy()
        scene_base = scene_idx * (len(RELATION_ORDER) * relation_gap + scene_gap)
        centers = [scene_base + rel_idx * relation_gap for rel_idx in range(len(RELATION_ORDER))]
        scene_start = min(centers) - 0.62
        scene_end = max(centers) + 0.72
        scene_center = (scene_start + scene_end) / 2.0
        scene_ranges.append((scene_start, scene_end, scene_center, SCENE_LABEL[scene]))
        scene_hp = part["mean_hp_margin"].dropna().astype(float)
        if len(scene_hp):
            scene_min = float(scene_hp.min())
            scene_max = float(scene_hp.max())
            scene_span = max(scene_max - scene_min, 1.0)
            scene_low = scene_min - max(3.0, scene_span * 0.08)
            scene_high = scene_max + max(3.0, scene_span * 0.08)
            if scene_high - scene_low < 18.0:
                scene_mid = (scene_high + scene_low) / 2.0
                scene_low = scene_mid - 9.0
                scene_high = scene_mid + 9.0
            scene_progress_ranges[scene] = (scene_low, scene_high)
        else:
            scene_progress_ranges[scene] = shared_ylim
        for center, relation in zip(centers, RELATION_ORDER):
            subset = part[part["source_relation"].eq(relation)].copy()
            hp_vals = subset["mean_hp_margin"].dropna().values
            progress_vals = subset["mean_game_progress"].dropna().values * 100.0
            if len(hp_vals):
                hp_positions_all.append(center + hp_offset)
                hp_values_all.append(hp_vals)
                hp_relations_all.append(relation)
            if len(progress_vals):
                scene_low, scene_high = scene_progress_ranges[scene]
                progress_y = scene_low + (np.clip(progress_vals, 0.0, 100.0) / 100.0) * (scene_high - scene_low)
                progress_payload.append((center + progress_offset, relation, progress_y))

    if scene_progress_ranges:
        progress_lows = [bounds[0] for bounds in scene_progress_ranges.values()]
        progress_highs = [bounds[1] for bounds in scene_progress_ranges.values()]
        shared_ylim = (min(shared_ylim[0], min(progress_lows)), max(shared_ylim[1], max(progress_highs)))

    if hp_values_all:
        bp = ax_hp.boxplot(
            hp_values_all,
            positions=hp_positions_all,
            widths=0.34,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.0},
            whiskerprops={"color": "#475569", "linewidth": 0.82},
            capprops={"color": "#475569", "linewidth": 0.82},
        )
        for patch, relation in zip(bp["boxes"], hp_relations_all):
            patch.set_facecolor(COLORS[relation])
            patch.set_alpha(0.62)
            patch.set_edgecolor("#334155")
            patch.set_linewidth(0.85)

    for scene_idx, scene in enumerate(SCENE_ORDER):
        part = data[data["scenario"].eq(scene)].copy()
        scene_base = scene_idx * (len(RELATION_ORDER) * relation_gap + scene_gap)
        centers = [scene_base + rel_idx * relation_gap for rel_idx in range(len(RELATION_ORDER))]
        scene_low, scene_high = scene_progress_ranges[scene]
        for pos, relation in enumerate(RELATION_ORDER, start=1):
            subset = part[part["source_relation"].eq(relation)].copy()
            if subset.empty:
                continue
            if len(subset) > 90:
                subset = subset.sample(n=90, random_state=20260626)
            hp_x = rng.normal(loc=centers[pos - 1] + hp_offset, scale=0.035, size=len(subset))
            progress_x = rng.normal(loc=centers[pos - 1] + progress_offset, scale=0.030, size=len(subset))
            hp_y = subset["mean_hp_margin"].astype(float).to_numpy()
            progress_pct = np.clip(subset["mean_game_progress"].astype(float).to_numpy() * 100.0, 0.0, 100.0)
            progress_y = scene_low + (progress_pct / 100.0) * (scene_high - scene_low)
            valid = np.isfinite(hp_y) & np.isfinite(progress_y)
            for x0, y0, x1, y1 in zip(hp_x[valid], hp_y[valid], progress_x[valid], progress_y[valid]):
                ax_hp.plot(
                    [x0, x1],
                    [y0, y1],
                    color=COLORS[relation],
                    alpha=0.07,
                    linewidth=0.42,
                    zorder=1,
                    rasterized=True,
                )
            sizes = np.clip(6 + np.sqrt(subset["frames"].astype(float)) * 2.0, 6, 22)
            ax_hp.scatter(
                hp_x,
                hp_y,
                s=sizes,
                color=COLORS[relation],
                alpha=0.34,
                edgecolors="none",
                zorder=3,
                rasterized=True,
            )
            ax_hp.scatter(
                progress_x,
                progress_y,
                s=np.clip(5 + np.sqrt(subset["frames"].astype(float)) * 1.6, 5, 18),
                color=COLORS[relation],
                alpha=0.20,
                edgecolors="none",
                zorder=2,
                rasterized=True,
            )

    if progress_payload:
        vp = ax_hp.violinplot(
            [vals for _, _, vals in progress_payload],
            positions=[pos for pos, _, _ in progress_payload],
            widths=0.30,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, (_, relation, _) in zip(vp["bodies"], progress_payload):
            body.set_facecolor(COLORS[relation])
            body.set_alpha(0.18)
            body.set_edgecolor(COLORS[relation])
            body.set_linewidth(0.65)
            body.set_zorder(1.4)
        if "cmedians" in vp:
            vp["cmedians"].set_color("#334155")
            vp["cmedians"].set_linewidth(0.75)
            vp["cmedians"].set_zorder(2.4)

    ax_hp.axhline(0, color="#64748B", linewidth=0.8, linestyle="--", alpha=0.55)
    ax_hp.set_ylim(*shared_ylim)
    ax_hp.set_ylabel("HP margin")
    ax_hp.grid(axis="y", alpha=0.16)
    ax_hp.set_axisbelow(True)
    ax_hp.set_xticks([])
    ax_hp.spines["bottom"].set_visible(False)
    if scene_ranges:
        x_min = scene_ranges[0][0] - 0.15
        x_max = scene_ranges[-1][1] + 0.55
        ax_hp.set_xlim(x_min, x_max)
        y_low, y_high = ax_hp.get_ylim()
        y_span = y_high - y_low
        lower_bracket_y = y_low + y_span * 0.085
        lower_tick_y = lower_bracket_y + y_span * 0.035
        lower_text_y = lower_bracket_y + y_span * 0.012
        upper_bracket_y = y_high - y_span * 0.060
        upper_tick_y = upper_bracket_y - y_span * 0.035
        upper_text_y = upper_bracket_y + y_span * 0.025
        for scene, (start, end, center, label) in zip(SCENE_ORDER, scene_ranges):
            use_upper = scene in {"sce-3", "sce-3m"}
            bracket_y = upper_bracket_y if use_upper else lower_bracket_y
            tick_y = upper_tick_y if use_upper else lower_tick_y
            text_y = (upper_tick_y - y_span * 0.012) if use_upper else lower_text_y
            text_va = "top" if use_upper else "bottom"
            ax_hp.plot(
                [start, end],
                [bracket_y, bracket_y],
                color="#334155",
                linewidth=0.9,
                clip_on=True,
            )
            ax_hp.plot(
                [start, start],
                [bracket_y, tick_y],
                color="#334155",
                linewidth=0.9,
                clip_on=True,
            )
            ax_hp.plot(
                [end, end],
                [bracket_y, tick_y],
                color="#334155",
                linewidth=0.9,
                clip_on=True,
            )
            ax_hp.text(
                center,
                text_y,
                label,
                ha="center",
                va=text_va,
                fontsize=10.5,
                clip_on=True,
            )
            axis_x = end - 0.08
            scene_low, scene_high = scene_progress_ranges[scene]
            ax_hp.plot(
                [axis_x, axis_x],
                [scene_low, scene_high],
                color="#64748B",
                linewidth=0.75,
                alpha=0.68,
                zorder=0,
            )
            for tick in (0, 50, 100):
                tick_y_data = scene_low + (tick / 100.0) * (scene_high - scene_low)
                ax_hp.plot(
                    [axis_x - 0.055, axis_x + 0.055],
                    [tick_y_data, tick_y_data],
                    color="#64748B",
                    linewidth=0.75,
                    alpha=0.68,
                    zorder=0,
                )
                ax_hp.text(
                    axis_x + 0.075,
                    tick_y_data,
                    f"{tick}",
                    color="#64748B",
                    ha="left",
                    va="center",
                    fontsize=7.2,
                )
        last_scene = SCENE_ORDER[-1]
        last_axis_x = scene_ranges[-1][1] + 0.33
        last_low, last_high = scene_progress_ranges[last_scene]
        ax_hp.text(
            last_axis_x,
            (last_low + last_high) / 2.0,
            "Progress (%)",
            color="#475569",
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
        )
    legend_relation_labels = {
        "source_etg_supported": "Source graph-supported states",
        "etg_selected_outside_source_etg": "Graph-only unseen states",
        "shared_selected_outside_source_etg": "Shared unseen states",
        "synergy_selected_outside_source_etg": "SAGE-only unseen states",
    }
    relation_handles = [
        mpl.patches.Patch(facecolor=COLORS[r], edgecolor="#334155", alpha=0.62, label=legend_relation_labels[r])
        for r in RELATION_ORDER
    ]
    style_handles = [
        mpl.lines.Line2D([0], [0], color="#334155", linewidth=4, label="HP margin box"),
        mpl.patches.Patch(facecolor="#CBD5E1", edgecolor="#334155", alpha=0.24, label="Episode-progress violin"),
        mpl.lines.Line2D([0], [0], color="#94A3B8", linewidth=0.8, alpha=0.55, label="Same-state link"),
    ]
    fig.legend(
        handles=relation_handles + style_handles,
        frameon=False,
        ncol=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=0.92,
        handlelength=1.8,
        handletextpad=0.45,
        fontsize=11.8,
    )
    ax_hp.tick_params(axis="both", labelsize=10.5)
    ax_hp.yaxis.label.set_size(12)
    fig.tight_layout(rect=[0, 0.015, 1, 0.952])
    save(fig, fig_dir, "fig3b_boxplot_aligned_hp_margin_with_game_progress")


def draw_hp_margin_progress_with_score(states: pd.DataFrame, fig_dir: Path, min_frames: int) -> None:
    data = states[states["frames"].ge(min_frames)].copy()
    rng = np.random.default_rng(20260626)
    fig, (ax_hp, ax_score) = plt.subplots(
        2,
        1,
        figsize=(15.6, 5.25),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.435], "hspace": 0.055},
    )
    hp_all = data["mean_hp_margin"].dropna().astype(float)
    if len(hp_all):
        hp_min = float(hp_all.min())
        hp_max = float(hp_all.max())
        hp_pad = max(4.0, (hp_max - hp_min) * 0.08)
        hp_ylim = (hp_min - hp_pad, hp_max + hp_pad)
    else:
        hp_ylim = (-8.0, 106.0)

    relation_gap = 1.12
    scene_gap = 1.25
    hp_offset = -0.24
    progress_offset = 0.34
    hp_positions_all: List[float] = []
    hp_values_all: List[np.ndarray] = []
    hp_relations_all: List[str] = []
    score_positions_all: List[float] = []
    score_values_all: List[np.ndarray] = []
    score_relations_all: List[str] = []
    progress_payload: List[Tuple[float, str, np.ndarray]] = []
    scene_ranges: List[Tuple[float, float, float, str]] = []
    scene_progress_ranges: Dict[str, Tuple[float, float]] = {}

    for scene_idx, scene in enumerate(SCENE_ORDER):
        part = data[data["scenario"].eq(scene)].copy()
        scene_base = scene_idx * (len(RELATION_ORDER) * relation_gap + scene_gap)
        centers = [scene_base + rel_idx * relation_gap for rel_idx in range(len(RELATION_ORDER))]
        scene_start = min(centers) - 0.62
        scene_end = max(centers) + 0.72
        scene_center = (scene_start + scene_end) / 2.0
        scene_ranges.append((scene_start, scene_end, scene_center, SCENE_LABEL[scene]))

        scene_hp = part["mean_hp_margin"].dropna().astype(float)
        if len(scene_hp):
            scene_min = float(scene_hp.min())
            scene_max = float(scene_hp.max())
            scene_span = max(scene_max - scene_min, 1.0)
            scene_low = scene_min - max(3.0, scene_span * 0.08)
            scene_high = scene_max + max(3.0, scene_span * 0.08)
            if scene_high - scene_low < 18.0:
                scene_mid = (scene_high + scene_low) / 2.0
                scene_low = scene_mid - 9.0
                scene_high = scene_mid + 9.0
            scene_progress_ranges[scene] = (scene_low, scene_high)
        else:
            scene_progress_ranges[scene] = hp_ylim

        for center, relation in zip(centers, RELATION_ORDER):
            subset = part[part["source_relation"].eq(relation)].copy()
            hp_vals = subset["mean_hp_margin"].dropna().values
            progress_vals = subset["mean_game_progress"].dropna().values * 100.0
            score_vals = subset["mean_score_conditioned"].dropna().values
            if len(hp_vals):
                hp_positions_all.append(center + hp_offset)
                hp_values_all.append(hp_vals)
                hp_relations_all.append(relation)
            if len(progress_vals):
                scene_low, scene_high = scene_progress_ranges[scene]
                progress_y = scene_low + (np.clip(progress_vals, 0.0, 100.0) / 100.0) * (scene_high - scene_low)
                progress_payload.append((center + progress_offset, relation, progress_y))
            if len(score_vals):
                score_positions_all.append(center)
                score_values_all.append(score_vals)
                score_relations_all.append(relation)

    if scene_progress_ranges:
        progress_lows = [bounds[0] for bounds in scene_progress_ranges.values()]
        progress_highs = [bounds[1] for bounds in scene_progress_ranges.values()]
        hp_ylim = (min(hp_ylim[0], min(progress_lows)), max(hp_ylim[1], max(progress_highs)))

    if hp_values_all:
        bp = ax_hp.boxplot(
            hp_values_all,
            positions=hp_positions_all,
            widths=0.34,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.0},
            whiskerprops={"color": "#475569", "linewidth": 0.82},
            capprops={"color": "#475569", "linewidth": 0.82},
        )
        for patch, relation in zip(bp["boxes"], hp_relations_all):
            patch.set_facecolor(COLORS[relation])
            patch.set_alpha(0.62)
            patch.set_edgecolor("#334155")
            patch.set_linewidth(0.85)

    for scene_idx, scene in enumerate(SCENE_ORDER):
        part = data[data["scenario"].eq(scene)].copy()
        scene_base = scene_idx * (len(RELATION_ORDER) * relation_gap + scene_gap)
        centers = [scene_base + rel_idx * relation_gap for rel_idx in range(len(RELATION_ORDER))]
        scene_low, scene_high = scene_progress_ranges[scene]
        for pos, relation in enumerate(RELATION_ORDER, start=1):
            subset = part[part["source_relation"].eq(relation)].copy()
            if subset.empty:
                continue
            if len(subset) > 90:
                subset = subset.sample(n=90, random_state=20260626)
            hp_x = rng.normal(loc=centers[pos - 1] + hp_offset, scale=0.035, size=len(subset))
            progress_x = rng.normal(loc=centers[pos - 1] + progress_offset, scale=0.030, size=len(subset))
            hp_y = subset["mean_hp_margin"].astype(float).to_numpy()
            progress_pct = np.clip(subset["mean_game_progress"].astype(float).to_numpy() * 100.0, 0.0, 100.0)
            progress_y = scene_low + (progress_pct / 100.0) * (scene_high - scene_low)
            valid = np.isfinite(hp_y) & np.isfinite(progress_y)
            for x0, y0, x1, y1 in zip(hp_x[valid], hp_y[valid], progress_x[valid], progress_y[valid]):
                ax_hp.plot(
                    [x0, x1],
                    [y0, y1],
                    color=COLORS[relation],
                    alpha=0.07,
                    linewidth=0.42,
                    zorder=1,
                    rasterized=True,
                )
            ax_hp.scatter(
                hp_x,
                hp_y,
                s=np.clip(6 + np.sqrt(subset["frames"].astype(float)) * 2.0, 6, 22),
                color=COLORS[relation],
                alpha=0.34,
                edgecolors="none",
                zorder=3,
                rasterized=True,
            )
            ax_hp.scatter(
                progress_x,
                progress_y,
                s=np.clip(5 + np.sqrt(subset["frames"].astype(float)) * 1.6, 5, 18),
                color=COLORS[relation],
                alpha=0.20,
                edgecolors="none",
                zorder=2,
                rasterized=True,
            )

    if progress_payload:
        vp = ax_hp.violinplot(
            [vals for _, _, vals in progress_payload],
            positions=[pos for pos, _, _ in progress_payload],
            widths=0.30,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, (_, relation, _) in zip(vp["bodies"], progress_payload):
            body.set_facecolor(COLORS[relation])
            body.set_alpha(0.18)
            body.set_edgecolor(COLORS[relation])
            body.set_linewidth(0.65)
            body.set_zorder(1.4)
        if "cmedians" in vp:
            vp["cmedians"].set_color("#334155")
            vp["cmedians"].set_linewidth(0.75)
            vp["cmedians"].set_zorder(2.4)

    if score_values_all:
        bp_score = ax_score.boxplot(
            score_values_all,
            positions=score_positions_all,
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#334155",
                "markersize": 3.8,
                "markeredgewidth": 0.75,
            },
            medianprops={"color": "#111827", "linewidth": 1.0},
            whiskerprops={"color": "#475569", "linewidth": 0.82},
            capprops={"color": "#475569", "linewidth": 0.82},
        )
        for patch, relation in zip(bp_score["boxes"], score_relations_all):
            patch.set_facecolor(COLORS[relation])
            patch.set_alpha(0.62)
            patch.set_edgecolor("#334155")
            patch.set_linewidth(0.85)

    ax_hp.axhline(0, color="#64748B", linewidth=0.8, linestyle="--", alpha=0.55)
    ax_hp.set_ylim(*hp_ylim)
    ax_hp.set_ylabel("HP margin", labelpad=7)
    ax_hp.grid(axis="y", alpha=0.16)
    ax_hp.set_axisbelow(True)
    ax_hp.set_xticks([])
    ax_hp.spines["bottom"].set_visible(False)

    ax_score.grid(axis="y", alpha=0.16)
    ax_score.set_axisbelow(True)
    ax_score.set_xticks([])
    ax_score.spines["bottom"].set_visible(False)
    ax_score.patch.set_alpha(0.0)
    ax_score.set_ylabel("Episode score", labelpad=9)
    score_y_low, score_y_high = ax_score.get_ylim()
    score_y_span = score_y_high - score_y_low
    ax_score.set_ylim(score_y_low - score_y_span * 0.06, score_y_high + score_y_span * 0.13)
    score_y_low, score_y_high = ax_score.get_ylim()
    score_y_span = score_y_high - score_y_low
    lower_row_bracket_y = score_y_high - score_y_span * 0.075
    lower_row_tick_y = lower_row_bracket_y - score_y_span * 0.045
    lower_row_label_y = lower_row_bracket_y + score_y_span * 0.018
    lower_row_bottom_bracket_y = score_y_low + score_y_span * 0.130
    lower_row_bottom_tick_y = lower_row_bottom_bracket_y + score_y_span * 0.045
    lower_row_bottom_label_y = lower_row_bottom_bracket_y + score_y_span * 0.040

    if scene_ranges:
        x_min = scene_ranges[0][0] - 0.15
        x_max = scene_ranges[-1][1] + 0.55
        ax_hp.set_xlim(x_min, x_max)
        ax_score.set_xlim(x_min, x_max)
        y_low, y_high = ax_hp.get_ylim()
        y_span = y_high - y_low
        lower_bracket_y = y_low + y_span * 0.085
        lower_tick_y = lower_bracket_y + y_span * 0.035
        lower_text_y = lower_bracket_y + y_span * 0.012
        upper_bracket_y = y_high - y_span * 0.060
        upper_tick_y = upper_bracket_y - y_span * 0.035
        upper_text_y = upper_bracket_y + y_span * 0.025
        for scene, (start, end, center, label) in zip(SCENE_ORDER, scene_ranges):
            use_upper = scene in {"sce-3", "sce-3m"}
            bracket_y = upper_bracket_y if use_upper else lower_bracket_y
            tick_y = upper_tick_y if use_upper else lower_tick_y
            text_y = upper_text_y if use_upper else lower_text_y
            ax_hp.plot([start, end], [bracket_y, bracket_y], color="#334155", linewidth=0.9, clip_on=True)
            ax_hp.plot([start, start], [bracket_y, tick_y], color="#334155", linewidth=0.9, clip_on=True)
            ax_hp.plot([end, end], [bracket_y, tick_y], color="#334155", linewidth=0.9, clip_on=True)
            if use_upper:
                ax_hp.text(
                    center,
                    upper_tick_y - y_span * 0.012,
                    label,
                    ha="center",
                    va="top",
                    fontsize=10.5,
                    clip_on=True,
                )
                ax_score.text(
                    center,
                    lower_row_bottom_label_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    clip_on=False,
                    zorder=10,
                )
            else:
                ax_hp.text(center, text_y, label, ha="center", va="bottom", fontsize=10.5, clip_on=True)
                ax_score.text(
                    center,
                    lower_row_label_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10.5,
                    clip_on=False,
                    zorder=10,
                )

            score_bracket_y = lower_row_bottom_bracket_y if use_upper else lower_row_bracket_y
            score_tick_y = lower_row_bottom_tick_y if use_upper else lower_row_tick_y
            ax_score.plot([start, end], [score_bracket_y, score_bracket_y], color="#334155", linewidth=0.9, clip_on=True)
            ax_score.plot([start, start], [score_bracket_y, score_tick_y], color="#334155", linewidth=0.9, clip_on=True)
            ax_score.plot([end, end], [score_bracket_y, score_tick_y], color="#334155", linewidth=0.9, clip_on=True)

            axis_x = end - 0.08
            scene_low, scene_high = scene_progress_ranges[scene]
            ax_hp.plot([axis_x, axis_x], [scene_low, scene_high], color="#64748B", linewidth=0.75, alpha=0.68, zorder=0)
            for tick in (0, 50, 100):
                tick_y_data = scene_low + (tick / 100.0) * (scene_high - scene_low)
                ax_hp.plot(
                    [axis_x - 0.055, axis_x + 0.055],
                    [tick_y_data, tick_y_data],
                    color="#64748B",
                    linewidth=0.75,
                    alpha=0.68,
                    zorder=0,
                )
                ax_hp.text(axis_x + 0.075, tick_y_data, f"{tick}", color="#64748B", ha="left", va="center", fontsize=7.2)
        last_scene = SCENE_ORDER[-1]
        last_axis_x = scene_ranges[-1][1] + 0.62
        last_low, last_high = scene_progress_ranges[last_scene]
        ax_hp.text(
            last_axis_x,
            last_low + 0.82 * (last_high - last_low),
            "Episode progress (%)",
            color="#475569",
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
        )

    legend_relation_labels = {
        "source_etg_supported": "Source graph-supported states",
        "etg_selected_outside_source_etg": "Graph-only unseen states",
        "shared_selected_outside_source_etg": "Shared unseen states",
        "synergy_selected_outside_source_etg": "SAGE-only unseen states",
    }
    relation_handles = [
        mpl.patches.Patch(facecolor=COLORS[r], edgecolor="#334155", alpha=0.62, label=legend_relation_labels[r])
        for r in RELATION_ORDER
    ]
    style_handles = [
        mpl.lines.Line2D([0], [0], color="#334155", linewidth=4, label="HP margin box"),
        mpl.patches.Patch(facecolor="#CBD5E1", edgecolor="#334155", alpha=0.24, label="Episode-progress violin"),
        mpl.lines.Line2D([0], [0], color="#94A3B8", linewidth=0.8, alpha=0.55, label="Same-state link"),
        mpl.lines.Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="#334155", markersize=4, label="Score mean"),
    ]
    fig.legend(
        handles=relation_handles + style_handles,
        frameon=False,
        ncol=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=0.78,
        handlelength=1.6,
        handletextpad=0.40,
        fontsize=11.2,
    )
    for axis in (ax_hp, ax_score):
        axis.tick_params(axis="both", labelsize=10.5)
        axis.yaxis.label.set_size(12)
    fig.subplots_adjust(left=0.060, right=0.985, bottom=0.075, top=0.925, hspace=0.045)
    save(fig, fig_dir, "fig3c_hp_progress_and_episode_score_by_state_source")


def draw_state_conditioned_score_boxplot(states: pd.DataFrame, fig_dir: Path, min_frames: int) -> None:
    data = states[states["frames"].ge(min_frames)].copy()
    fig, axes = plt.subplots(1, 6, figsize=(11.4, 2.95), sharey=False)
    for ax, scene in zip(axes, SCENE_ORDER):
        part = data[data["scenario"].eq(scene)]
        values = [part[part["source_relation"].eq(rel)]["mean_score_conditioned"].dropna().values for rel in RELATION_ORDER]
        if not any(len(v) for v in values):
            ax.set_axis_off()
            continue
        bp = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#111827", "linewidth": 1.1})
        for patch, relation in zip(bp["boxes"], RELATION_ORDER):
            patch.set_facecolor(COLORS[relation])
            patch.set_alpha(0.74)
            patch.set_edgecolor("#334155")
        ax.set_title(SCENE_LABEL[scene])
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.16)
        if ax is axes[0]:
            ax.set_ylabel("Episode score conditioned on state")
    handles = [mpl.patches.Patch(facecolor=COLORS[r], edgecolor="#334155", label=RELATION_SHORT[r]) for r in RELATION_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, fig_dir, "fig5_boxplot_aligned_state_conditioned_terminal_score")


def draw_gate_heatmap(summary: pd.DataFrame, fig_dir: Path) -> None:
    metrics = [
        "frame_share",
        "frame_weighted_game_progress",
        "frame_weighted_hp_margin",
        "frame_weighted_score_conditioned",
        "gate_validation_ratio",
        "gate_changed_ratio",
        "mean_gate_confidence",
        "mean_gate_advantage",
        "mean_gate_candidate_visits",
    ]
    labels = ["Share", "Progress", "HP margin", "Score", "Gate valid.", "Changed", "C", "A", "v"]
    data = summary[summary["source_relation"].eq("synergy_selected_outside_source_etg")].set_index("scenario").reindex(SCENE_ORDER)
    mat = data[metrics].astype(float)
    display = mat.copy()
    for col in metrics:
        vals = display[col]
        lo, hi = vals.quantile(0.05), vals.quantile(0.95)
        display[col] = 0.5 if pd.isna(hi) or hi <= lo else ((vals - lo) / (hi - lo)).clip(0, 1)
    fig, ax = plt.subplots(figsize=(7.8, 2.8))
    im = ax.imshow(display.fillna(0).values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(SCENE_ORDER)))
    ax.set_yticklabels([SCENE_LABEL[s] for s in SCENE_ORDER])
    for i, scene in enumerate(SCENE_ORDER):
        for j, metric in enumerate(metrics):
            val = mat.loc[scene, metric] if scene in mat.index else np.nan
            if not np.isfinite(val):
                continue
            if metric in {"frame_share", "gate_validation_ratio", "gate_changed_ratio", "mean_gate_confidence", "mean_gate_advantage"}:
                text = f"{val:.2f}"
            elif metric == "frame_weighted_game_progress":
                text = f"{val:.0%}"
            elif metric in {"mean_gate_candidate_visits", "frame_weighted_score_conditioned"}:
                text = f"{val:.0f}"
            else:
                text = f"{val:+.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Normalized")
    fig.tight_layout()
    save(fig, fig_dir, "fig4_boxplot_aligned_synergy_outside_gate_evidence_heatmap")


def save(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_report(out_dir: Path, selected_dir: Path, episodes: pd.DataFrame, summary: pd.DataFrame, selected_summary: pd.DataFrame, min_frames: int) -> None:
    score_check = (
        episodes.groupby(["scenario", "method"], as_index=False)
        .agg(n=("episode_id", "count"), score_mean=("score", "mean"), score_std=("score", "std"), win_rate=("win", "mean"))
    )
    lines = [
        "# Boxplot-Aligned Gate State Quality Analysis",
        "",
        "## 样本口径",
        "",
        f"- 选样目录：`{selected_dir}`。",
        "- Synergy 样本：`cache/selected_gate_positive_episodes.csv`。",
        "- etg-only 样本：`cache/selected_etg_counterpart_episodes.csv`。",
        "- 本脚本不加入 trials、不加入全量 final eval、不重新 top-k；所有状态和 gate 统计只来自箱线图使用的同一批 episode。",
        "- 状态仍由 `eval_bktree_norm_state` 规范化 digest 聚合；outside/source 关系在该 selected 样本内部根据日志 `nid_is_ood/nid_status` 和方法出现集合重算。",
        "",
        "## 与箱线图分数一致性核查",
        "",
        score_check.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 原箱线图 summary 对照",
        "",
        selected_summary.to_markdown(index=False, floatfmt=".3f") if not selected_summary.empty else "未找到 `selected_vs_etg_summary.csv`。",
        "",
        "## 状态来源与质量汇总",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 输出图",
        "",
        "- `fig1_reproduced_selected_score_boxplot`：用同源 episode 复刻箱线图右侧分数分布，作为口径校验。",
        "- `fig2_boxplot_aligned_state_relation_stack_with_hp_margin`：同源 episode 内的状态来源占比 + HP margin。",
        f"- `fig3_boxplot_aligned_hp_margin_distribution`：同源 episode 内的状态 HP margin 分布，过滤 `frames < {min_frames}`。",
        f"- `fig3b_boxplot_aligned_hp_margin_with_game_progress`：上半部分为 HP margin 箱线图+散点，下半部分为相同状态类别的出现进度分布；过滤 `frames < {min_frames}`。",
        "- `fig4_boxplot_aligned_synergy_outside_gate_evidence_heatmap`：只看 Synergy selected outside ETG 状态的 gate 证据，并加入状态出现进度与终局 score。",
        f"- `fig5_boxplot_aligned_state_conditioned_terminal_score`：状态来源类别对应的 episode 终局 score 分布；过滤 `frames < {min_frames}`。",
    ]
    (out_dir / "BOXPLOT_ALIGNED_GATE_STATE_QUALITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-frames", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    out_dir = args.output_dir
    cache_dir = out_dir / "cache"
    fig_dir = out_dir / "figures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    selected = load_selected(args.selected_dir)
    episodes, frame_rows = load_exact_episodes(selected)
    frames = pd.DataFrame(frame_rows)
    states = aggregate_states(frame_rows)
    summary = state_summary(states)
    selected_summary_path = args.selected_dir / "cache" / "selected_vs_etg_summary.csv"
    selected_summary = pd.read_csv(selected_summary_path) if selected_summary_path.exists() else pd.DataFrame()

    selected.to_csv(cache_dir / "boxplot_selected_episode_rows.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(cache_dir / "boxplot_aligned_episode_table.csv", index=False, encoding="utf-8-sig")
    frames.to_csv(cache_dir / "boxplot_aligned_frame_table.csv", index=False, encoding="utf-8-sig")
    states.to_csv(cache_dir / "boxplot_aligned_state_table.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(cache_dir / "boxplot_aligned_state_relation_summary.csv", index=False, encoding="utf-8-sig")

    draw_score_boxplot(episodes, fig_dir)
    draw_relation_stack(summary, fig_dir)
    draw_hp_margin_boxplot(states, fig_dir, args.min_frames)
    draw_hp_margin_with_progress(states, fig_dir, args.min_frames)
    draw_hp_margin_progress_with_score(states, fig_dir, args.min_frames)
    draw_gate_heatmap(summary, fig_dir)
    draw_state_conditioned_score_boxplot(states, fig_dir, args.min_frames)
    write_report(out_dir, args.selected_dir, episodes, summary, selected_summary, args.min_frames)
    print(out_dir)


if __name__ == "__main__":
    main()
