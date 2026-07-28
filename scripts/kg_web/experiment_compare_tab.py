import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from src import ROOT_DIR
from kg_web.teacher_guided_tab import render_teacher_guided_etg_panel


_ALL_DATA_ROOT = ROOT_DIR / "output" / "learner_results" / "all_data"
_KG_CATALOG_PATH = ROOT_DIR / "configs" / "kg_catalog.yaml"
_REPRO_GUIDE_PATH = ROOT_DIR / "docs" / "batch_experiment_reproduction_guide.md"
_CONFIG_AUDIT_PATH = ROOT_DIR / "docs" / "batch_experiment_hardcoded_config_audit.md"
_MANIFEST_NAME = "experiment_manifest.json"
_FINAL_EVAL_SCRIPT = ROOT_DIR / "scripts" / "evaluate_archived_experiment.py"
_BATCH_FINAL_EVAL_SCRIPT = ROOT_DIR / "scripts" / "batch_final_eval_archived_experiments.py"
_SWITCH_GRID_EVAL_SCRIPT = ROOT_DIR / "scripts" / "run_switch_mechanism_eval_grid.py"
_FIXED_POOL_SWITCH_ABLATION_SCRIPT = ROOT_DIR / "scripts" / "run_fixed_pool_switch_ablation_eval.py"
_MULTISTEP_SENSITIVITY_SCRIPT = ROOT_DIR / "scripts" / "run_multistep_sensitivity_eval_grid.py"
_MULTISTEP_PARAMETER_SEARCH_SCRIPT = ROOT_DIR / "scripts" / "run_multistep_parameter_search.py"
_FIXED_POOL_SYNERGY_SCRIPT = ROOT_DIR / "scripts" / "run_fixed_pool_synergy_from_sensitivity.py"
_FIXED_POOL_VARIANT_FINAL_EVAL_SCRIPT = ROOT_DIR / "scripts" / "batch_final_eval_fixed_pool_variants.py"
_REPLAY_EVAL_SCRIPT = ROOT_DIR / "scripts" / "evaluate_replay_baseline.py"

_PHASE_LABELS = {
    "etg_only": "ETG-only",
    "exploration_only": "Exploration-only",
    "synergy": "ETG+微调协同",
    "unknown": "未知阶段",
}
_PHASE_ORDER = ["etg_only", "exploration_only", "synergy", "unknown"]

_SCENARIOS = [
    {
        "scenario": "sce1",
        "map_key": "sce-1",
        "map_id": "MarineMicro_MvsM_4",
        "preferred_kg": "MvsM4 - Augmented",
    },
    {
        "scenario": "sce1_mirror",
        "map_key": "sce-1m",
        "map_id": "MarineMicro_MvsM_4_mirror",
        "preferred_kg": "MvsM4-mirror - Augmented",
    },
    {
        "scenario": "sce2_dist",
        "map_key": "sce-2",
        "map_id": "MarineMicro_MvsM_4_dist",
        "preferred_kg": "MvsM4-dist - Augmented",
    },
    {
        "scenario": "sce2_dist_mirror",
        "map_key": "sce-2m",
        "map_id": "MarineMicro_MvsM_4_dist_mirror",
        "preferred_kg": "MvsM4-dist-mirror - Augmented",
    },
    {
        "scenario": "sce3_mvs8",
        "map_key": "sce-3",
        "map_id": "MarineMicro_MvsM_8",
        "preferred_kg": "MvsM8 - Augmented",
    },
    {
        "scenario": "sce3_mvs8_mirror",
        "map_key": "sce-3m",
        "map_id": "MarineMicro_MvsM_8_mirror",
        "preferred_kg": "MvsM8-mirror - Augmented",
    },
]

_MAP_ORDER = {item["map_id"]: idx for idx, item in enumerate(_SCENARIOS)}
_MAP_KEY_ORDER = {item["map_key"]: idx for idx, item in enumerate(_SCENARIOS)}
_REPLAY_SOURCE_DIRS = {
    "MarineMicro_MvsM_4": "data/MarineMicro_MvsM_4/6",
    "MarineMicro_MvsM_4_mirror": "data/MarineMicro_MvsM_4_mirror/3",
    "MarineMicro_MvsM_4_dist": "data/MarineMicro_MvsM_4_dist/1",
    "MarineMicro_MvsM_4_dist_mirror": "data/MarineMicro_MvsM_4_dist_mirror/3",
    "MarineMicro_MvsM_8": "data/MarineMicro_MvsM_8/1",
    "MarineMicro_MvsM_8_mirror": "data/MarineMicro_MvsM_8_mirror/1",
}
_METHOD_ORDER = {
    "synergy": 0,
    "ETG-only": 1,
    "MC-only": 2,
    "Replay-baseline": 3,
    "Teacher-guided-ETG": 4,
}

_REQUIRED_ARTIFACTS = [
    "manifest",
    "study_summary",
    "study_db",
    "runs",
    "trials",
    "action_tuning_model",
    "learner_log",
]

_REPLAY_REQUIRED_ARTIFACTS = [
    "manifest",
    "selected_sequences",
    "replay_eval_summary",
]

_TEACHER_REQUIRED_ARTIFACTS = [
    "manifest",
    "teacher_build_summary",
]


def _is_etg_only_manifest(manifest: Dict[str, Any]) -> bool:
    experiment_type = str(manifest.get("experiment_type", "")).lower()
    method = str(manifest.get("method", "")).lower()
    return (
        "etg_only" in experiment_type
        or "etg-only" in experiment_type
        or "etg-only" in method
        or "etg only" in method
    )


def _is_replay_baseline_manifest(manifest: Dict[str, Any]) -> bool:
    experiment_type = str(manifest.get("experiment_type", "")).lower()
    method = str(manifest.get("method", "")).lower()
    return (
        "historical_replay" in experiment_type
        or "replay_baseline" in experiment_type
        or "action replay" in method
        or "historical replay" in method
    )


def _is_teacher_guided_manifest(manifest: Dict[str, Any]) -> bool:
    experiment_type = str(manifest.get("experiment_type", "")).lower()
    method = str(manifest.get("method", "")).lower()
    dataset_type = str(manifest.get("dataset_type", "")).lower()
    return (
        "teacher_guided" in experiment_type
        or "teacher-guided" in method
        or "teacher_guided" in dataset_type
    )


def _required_artifacts_for_manifest(manifest: Dict[str, Any]) -> List[str]:
    if _is_teacher_guided_manifest(manifest):
        return list(_TEACHER_REQUIRED_ARTIFACTS)
    if _is_replay_baseline_manifest(manifest):
        return list(_REPLAY_REQUIRED_ARTIFACTS)
    required = list(_REQUIRED_ARTIFACTS)
    if _is_etg_only_manifest(manifest):
        required.remove("action_tuning_model")
    return required


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _nested_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _fmt(value: Any, digits: int = 3) -> str:
    num = _as_float(value)
    if num is None:
        return "-"
    return f"{num:.{digits}f}"


def _best_trial_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    best_trial = summary.get("best_trial")
    for trial in summary.get("trials", []):
        if isinstance(trial, dict) and trial.get("number") == best_trial:
            return trial
    return {}


def _best_trial_metric(summary: Dict[str, Any], key: str) -> Any:
    trial = _best_trial_record(summary)
    attrs = trial.get("user_attrs", {}) if isinstance(trial, dict) else {}
    return attrs.get(key) if isinstance(attrs, dict) else None


def _load_manifest(exp_dir: Path) -> Dict[str, Any]:
    manifest = _read_json(exp_dir / _MANIFEST_NAME) or {}
    manifest.setdefault("experiment_id", exp_dir.name)
    return manifest


@st.cache_data(ttl=60, show_spinner=False)
def _load_kg_catalog_entries() -> List[Dict[str, Any]]:
    if not _KG_CATALOG_PATH.exists():
        return []
    try:
        with open(str(_KG_CATALOG_PATH), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        entries = data.get("knowledge_graphs", [])
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def _artifact_status(exp_dir: Path) -> Dict[str, bool]:
    return {
        "manifest": (exp_dir / _MANIFEST_NAME).exists(),
        "study_summary": (exp_dir / "study_summary.json").exists(),
        "study_db": (exp_dir / "study.db").exists(),
        "runs": (exp_dir / "runs").is_dir(),
        "trials": (exp_dir / "trials").is_dir(),
        "action_tuning_model": (exp_dir / "action_tuning_model.pkl").exists(),
        "learner_log": (exp_dir / "learner.log").exists(),
        "selected_sequences": (exp_dir / "selected_sequences.json").exists(),
        "replay_eval_summary": (exp_dir / "replay_eval_summary.json").exists(),
        "teacher_build_summary": (exp_dir / "teacher_build_summary.json").exists(),
    }


def _has_complete_manifest(manifest: Dict[str, Any]) -> bool:
    if _is_replay_baseline_manifest(manifest):
        required = [
            "experiment_id",
            "map_key",
            "map_id",
            "experiment_type",
            "method",
            "data_dir",
            "dataset_type",
        ]
        if not all(manifest.get(key) not in (None, "") for key in required):
            return False
        source_sequences = manifest.get("source_sequences")
        return isinstance(source_sequences, dict) and bool(source_sequences)

    required = [
        "experiment_id",
        "map_key",
        "map_id",
        "experiment_type",
        "method",
        "kg_name",
        "kg_file",
        "transitions",
        "data_dir",
        "dataset_type",
        "replay_dataset_expansion",
        "source_run",
    ]
    if not all(manifest.get(key) not in (None, "") for key in required):
        return False
    bktree = manifest.get("bktree")
    if not isinstance(bktree, dict):
        return False
    return all(
        bktree.get(key) not in (None, "")
        for key in ("primary_threshold", "secondary_threshold", "path")
    )


def _artifact_summary(artifacts: Dict[str, bool], manifest: Dict[str, Any]) -> Dict[str, bool]:
    summary = dict(artifacts)
    summary["manifest_fields"] = artifacts.get("manifest", False) and _has_complete_manifest(manifest)
    required = _required_artifacts_for_manifest(manifest)
    summary["required_artifacts"] = required
    summary["archived_run"] = all(artifacts.get(key, False) for key in required)
    return summary


def _trial_phase(trial: Dict[str, Any]) -> str:
    attrs = trial.get("user_attrs", {}) if isinstance(trial.get("user_attrs"), dict) else {}
    phase = attrs.get("phase") or trial.get("phase") or "unknown"
    return str(phase)


def _trial_attrs(trial: Dict[str, Any]) -> Dict[str, Any]:
    attrs = trial.get("user_attrs", {})
    return attrs if isinstance(attrs, dict) else {}


def _phase_stats(summary: Dict[str, Any]) -> pd.DataFrame:
    buckets: Dict[str, Dict[str, Any]] = {}
    for trial in summary.get("trials", []):
        if not isinstance(trial, dict):
            continue
        phase = _trial_phase(trial)
        attrs = _trial_attrs(trial)
        bucket = buckets.setdefault(
            phase,
            {
                "phase": phase,
                "phase_label": _PHASE_LABELS.get(phase, phase),
                "trials": 0,
                "complete": 0,
                "pruned_or_probe": 0,
                "objective_values": [],
                "probe_values": [],
                "avg_scores": [],
                "win_rates": [],
                "episodes": 0,
            },
        )
        bucket["trials"] += 1
        state = str(trial.get("state", ""))
        if state.endswith("COMPLETE"):
            bucket["complete"] += 1
        if state.endswith("PRUNED") or attrs.get("exclude_from_parameter_optimization"):
            bucket["pruned_or_probe"] += 1

        value = _as_float(trial.get("value"))
        if value is not None:
            bucket["objective_values"].append(value)
        probe_value = _as_float(attrs.get("probe_objective"))
        if probe_value is not None:
            bucket["probe_values"].append(probe_value)
        avg_score = _as_float(attrs.get("avg_score"))
        if avg_score is not None:
            bucket["avg_scores"].append(avg_score)
        win_rate = _as_float(attrs.get("win_rate"))
        if win_rate is not None:
            bucket["win_rates"].append(win_rate)
        episodes = _as_float(attrs.get("num_episodes"))
        if episodes is not None:
            bucket["episodes"] += int(episodes)

    rows = []
    for phase, bucket in buckets.items():
        objective_values = bucket.pop("objective_values")
        probe_values = bucket.pop("probe_values")
        avg_scores = bucket.pop("avg_scores")
        win_rates = bucket.pop("win_rates")
        bucket.update(
            {
                "objective_mean": _mean(objective_values),
                "objective_max": max(objective_values) if objective_values else None,
                "probe_mean": _mean(probe_values),
                "probe_max": max(probe_values) if probe_values else None,
                "avg_score_mean": _mean(avg_scores),
                "win_rate_mean": _mean(win_rates),
            }
        )
        rows.append(bucket)

    rows.sort(
        key=lambda r: (
            _PHASE_ORDER.index(r["phase"]) if r["phase"] in _PHASE_ORDER else len(_PHASE_ORDER),
            r["phase"],
        )
    )
    return pd.DataFrame(rows)


@st.cache_data(ttl=30, show_spinner=False)
def _scan_experiments() -> List[Dict[str, Any]]:
    if not _ALL_DATA_ROOT.exists():
        return []

    exp_dirs: List[Path] = []
    for group_dir in sorted(_ALL_DATA_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not group_dir.is_dir():
            continue
        child_exps = [
            child
            for child in group_dir.iterdir()
            if child.is_dir()
            and ((child / "study_summary.json").exists() or (child / _MANIFEST_NAME).exists())
        ]
        exp_dirs.extend(child_exps)

    experiments: List[Dict[str, Any]] = []
    for exp_dir in sorted(exp_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        method_group = exp_dir.parent.name
        manifest = _load_manifest(exp_dir)
        summary = _read_json(exp_dir / "study_summary.json") or {}
        replay_summary = _read_json(exp_dir / "replay_eval_summary.json") or {}
        artifacts = _artifact_status(exp_dir)
        artifact_summary = _artifact_summary(artifacts, manifest)

        phase_df = _phase_stats(summary) if summary else pd.DataFrame()
        if _is_etg_only_manifest(manifest) and not phase_df.empty:
            phase_df = phase_df.copy()
            phase_df["phase"] = "etg_only"
            phase_df["phase_label"] = "ETG-only"
        if _is_replay_baseline_manifest(manifest):
            phase_df = pd.DataFrame()
        phase_labels = (
            " / ".join(phase_df["phase_label"].tolist()) if not phase_df.empty else "-"
        )
        if _is_replay_baseline_manifest(manifest):
            phase_labels = "Historical replay"

        replay_objective = _nested_get(replay_summary, "aggregate", "objective", "mean", default="-")
        replay_win_rate = _nested_get(replay_summary, "aggregate", "win_rate", "mean")
        replay_avg_score = _nested_get(replay_summary, "aggregate", "avg_score", "mean")
        replay_stability = _nested_get(replay_summary, "aggregate", "stability", "mean")
        replay_penalty = _nested_get(replay_summary, "aggregate", "penalty_factor", "mean")
        replay_episodes = _nested_get(replay_summary, "aggregate", "total_episodes", default="-")

        experiments.append(
            {
                "path": str(exp_dir),
                "method_group": method_group,
                "experiment_id": manifest.get("experiment_id", exp_dir.name),
                "display_name": manifest.get("display_name") or manifest.get("experiment_id", exp_dir.name),
                "modified_at": exp_dir.stat().st_mtime,
                "map_key": manifest.get("map_key", "-"),
                "map_id": manifest.get("map_id", "-"),
                "experiment_type": manifest.get("experiment_type", manifest.get("method", "-")),
                "kg_name": manifest.get("kg_name", "-"),
                "kg_file": manifest.get("kg_file", "-"),
                "data_dir": manifest.get("data_dir", "-"),
                "dataset_type": manifest.get("dataset_type", "-"),
                "replay_dataset_expansion": manifest.get("replay_dataset_expansion", "-"),
                "bktree_primary": _nested_get(manifest, "bktree", "primary_threshold", default="-"),
                "bktree_secondary": _nested_get(manifest, "bktree", "secondary_threshold", default="-"),
                "bktree_path": _nested_get(manifest, "bktree", "path", default="-"),
                "source_run": manifest.get("source_run", "-"),
                "total_trials": summary.get("total_trials", "-"),
                "completed_trials": summary.get("completed_trials", "-"),
                "best_value": replay_objective if _is_replay_baseline_manifest(manifest) else summary.get("best_value", "-"),
                "best_trial": summary.get("best_trial", "-"),
                "best_win_rate": replay_win_rate if _is_replay_baseline_manifest(manifest) else _best_trial_metric(summary, "win_rate"),
                "best_avg_score": replay_avg_score if _is_replay_baseline_manifest(manifest) else _best_trial_metric(summary, "avg_score"),
                "best_stability": replay_stability if _is_replay_baseline_manifest(manifest) else _best_trial_metric(summary, "stability"),
                "best_penalty_factor": replay_penalty if _is_replay_baseline_manifest(manifest) else _best_trial_metric(summary, "penalty_factor"),
                "best_num_episodes": replay_episodes if _is_replay_baseline_manifest(manifest) else _best_trial_metric(summary, "num_episodes"),
                "saved_at": summary.get("saved_at", "-"),
                "phases": phase_labels,
                "manifest_status": "已登记" if artifacts["manifest"] else "缺失",
                "artifacts": artifacts,
                "artifact_summary": artifact_summary,
                "manifest": manifest,
                "summary": summary,
                "replay_summary": replay_summary,
                "phase_df": phase_df,
            }
        )
    return experiments


def _find_catalog_entry(catalog: List[Dict[str, Any]], scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    preferred = scenario.get("preferred_kg")
    for entry in catalog:
        if entry.get("map_id") == scenario["map_id"] and entry.get("name") == preferred:
            return entry
    return None


def _is_augmented_catalog_entry(entry: Optional[Dict[str, Any]]) -> bool:
    if not entry:
        return False
    entry_type = str(entry.get("type", "")).lower()
    data_id = str(entry.get("data_id", "")).lower()
    data_dir = str(entry.get("data_dir", "")).lower()
    name = str(entry.get("name", "")).lower()
    return (
        entry_type == "augmented"
        or data_id.startswith("augmented")
        or "augmented" in data_dir
        or "augmented" in name
    )


def _find_scenario_experiment(
    experiments: List[Dict[str, Any]], scenario: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    candidates = [
        exp
        for exp in experiments
        if exp.get("map_key") == scenario["map_key"] or exp.get("map_id") == scenario["map_id"]
    ]
    if not candidates:
        return None
    synergy_candidates = [
        exp
        for exp in candidates
        if exp.get("method_group") == "synergy"
        or exp.get("experiment_type") == "three_phase_synergy"
    ]
    if synergy_candidates:
        candidates = synergy_candidates
    complete = [
        exp
        for exp in candidates
        if exp.get("artifact_summary", {}).get("archived_run")
        and exp.get("artifact_summary", {}).get("manifest_fields")
    ]
    return complete[0] if complete else candidates[0]


def _status_text(ok: bool, configured_label: str = "已配置") -> str:
    return f"✅ {configured_label}" if ok else "— 未配置"


def _scene_completion_rows(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    catalog = _load_kg_catalog_entries()
    rows = []
    for scenario in _SCENARIOS:
        entry = _find_catalog_entry(catalog, scenario)
        exp = _find_scenario_experiment(experiments, scenario)
        artifact_summary = exp.get("artifact_summary", {}) if exp else {}
        manifest = exp.get("manifest", {}) if exp else {}

        catalog_ok = entry is not None and _is_augmented_catalog_entry(entry)
        kg_file_ok = bool(catalog_ok and entry and entry.get("file"))
        transitions_ok = bool(catalog_ok and entry and entry.get("transitions"))
        data_dir_ok = bool(catalog_ok and entry and entry.get("data_dir"))
        bktree_ok = bool(
            manifest.get("bktree")
            and manifest["bktree"].get("primary_threshold") is not None
            and manifest["bktree"].get("secondary_threshold") is not None
        )
        replay_ok = bool(manifest.get("replay_dataset_expansion")) if exp else False
        archived_ok = bool(artifact_summary.get("archived_run"))
        manifest_ok = bool(artifact_summary.get("manifest_fields"))
        all_ok = all(
            [
                catalog_ok,
                kg_file_ok,
                transitions_ok,
                data_dir_ok,
                bktree_ok,
                replay_ok,
                archived_ok,
                manifest_ok,
            ]
        )

        rows.append(
            {
                "场景": scenario["scenario"],
                "map_key": scenario["map_key"],
                "map_id": scenario["map_id"],
                "目标ETG": scenario["preferred_kg"],
                "归档实验": exp["experiment_id"] if exp else "—",
                "总体状态": "✅ 复现配置完备" if all_ok else "— 未按 sce1 同款配置",
                "ETG目录": _status_text(catalog_ok, "已登记"),
                "ETG文件": _status_text(kg_file_ok),
                "Transitions": _status_text(transitions_ok),
                "数据目录": _status_text(data_dir_ok),
                "重演扩张": _status_text(replay_ok),
                "BKTree阈值": _status_text(bktree_ok),
                "Manifest字段": _status_text(manifest_ok, "完备"),
                "归档产物": _status_text(archived_ok, "完备"),
                "最优值": _fmt(exp.get("best_value")) if exp else "-",
                "总trial": exp.get("total_trials", "-") if exp else "-",
            }
        )
    return rows


def _render_manifest_template(selected: Dict[str, Any]) -> None:
    exp_dir = Path(selected["path"])
    template = {
        "experiment_id": exp_dir.name,
        "display_name": exp_dir.name,
        "map_key": "sce-1",
        "map_id": "MarineMicro_MvsM_4",
        "experiment_type": "three_phase_synergy",
        "method": "ETG+MCTS ActionTuning",
        "kg_name": "MvsM4 - Augmented",
        "kg_file": "MarineMicro_MvsM_4_augmented/kg_simple.pkl",
        "transitions": "MarineMicro_MvsM_4_augmented/kg_simple_transitions.pkl",
        "data_dir": "data/MarineMicro_MvsM_4/augmented_1",
        "dataset_type": "augmented",
        "replay_dataset_expansion": True,
        "bktree": {
            "primary_threshold": 0.7,
            "secondary_threshold": 0.5,
            "path": "data/MarineMicro_MvsM_4/augmented_1/bktree",
        },
        "source_run": "",
        "notes": "",
    }
    st.download_button(
        "下载 manifest 模板",
        data=json.dumps(template, ensure_ascii=False, indent=2),
        file_name=_MANIFEST_NAME,
        mime="application/json",
        use_container_width=True,
    )


def _plot_phase_stats(phase_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if phase_df.empty:
        return fig

    labels = phase_df["phase_label"].tolist()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=phase_df["trials"].tolist(),
            name="trial 数",
            marker_color="#9aa6b2",
            yaxis="y2",
            opacity=0.45,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=phase_df["objective_mean"].tolist(),
            mode="lines+markers",
            name="优化目标均值",
            line=dict(color="#2563eb", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=phase_df["probe_mean"].tolist(),
            mode="lines+markers",
            name="探索 probe 均值",
            line=dict(color="#7c3aed", width=3, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=phase_df["avg_score_mean"].tolist(),
            mode="lines+markers",
            name="episode 得分均值",
            line=dict(color="#16a34a", width=3),
        )
    )
    fig.update_layout(
        height=380,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=40, b=30),
        yaxis=dict(title="目标/得分"),
        yaxis2=dict(title="trial 数", overlaying="y", side="right", showgrid=False),
    )
    return fig


def _metric_mean(summary: Dict[str, Any], key: str) -> Any:
    value = _nested_get(summary, "aggregate", key, "mean")
    return value if value is not None else "-"


def _metric_ci(summary: Dict[str, Any], key: str) -> str:
    mean = _as_float(_nested_get(summary, "aggregate", key, "mean"))
    ci95 = _as_float(_nested_get(summary, "aggregate", key, "ci95"))
    if mean is None:
        return "-"
    if ci95 is None:
        return _fmt(mean)
    return f"{mean:.3f} ± {ci95:.3f}"


def _quote_cli_arg(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in ['"', "'", "&", "(", ")", "[", "]"]):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _build_final_eval_command(
    exp_dir: Path,
    episodes: int = 100,
    repeats: int = 1,
    timeout_minutes: int = 90,
    output_dir: Optional[Path] = None,
    action_tuning_mode: str = "自动",
) -> str:
    try:
        exp_arg = exp_dir.relative_to(ROOT_DIR)
    except ValueError:
        exp_arg = exp_dir

    cmd = [
        "python",
        str(Path("scripts") / "evaluate_archived_experiment.py"),
        "--experiment-dir",
        str(exp_arg),
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--eval-bktree-normalization",
        "decision",
        "--enable-mechanism-shadow-logging",
    ]
    if output_dir is not None:
        try:
            out_arg = output_dir.relative_to(ROOT_DIR)
        except ValueError:
            out_arg = output_dir
        cmd.extend(["--output-dir", str(out_arg)])
    if action_tuning_mode == "强制启用":
        cmd.append("--enable-action-tuning")
    elif action_tuning_mode == "强制关闭":
        cmd.append("--disable-action-tuning")
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _action_tuning_cli_value(action_tuning_mode: str) -> str:
    if action_tuning_mode == "强制启用":
        return "enable"
    if action_tuning_mode == "强制关闭":
        return "disable"
    return "auto"


def _has_complete_final_eval(exp_dir: Path, episodes: Optional[int] = None, repeats: Optional[int] = None) -> bool:
    for summary in _load_final_eval_summaries(exp_dir):
        if summary.get("status") != "completed":
            continue
        if episodes is not None and int(summary.get("episodes_per_repeat") or 0) != int(episodes):
            continue
        if repeats is not None and int(summary.get("requested_repeats") or 0) != int(repeats):
            continue
        status = _final_eval_bktree_artifacts(summary.get("_eval_dir", ""))
        total = int(status.get("total") or 0)
        complete = int(status.get("complete") or 0)
        eval_dir = Path(str(summary.get("_eval_dir", "")))
        repeat_dirs = sorted(p for p in (eval_dir / "repeats").glob("repeat_*") if p.is_dir()) if eval_dir.exists() else []
        if not repeat_dirs and eval_dir.exists():
            repeat_dirs = [eval_dir]
        local_compact = 0
        fresh_eval = 0
        valid_spawn_mode = 0
        strict_spawn = 0
        compatible_norm = 0
        start_validation = 0
        for repeat_dir in repeat_dirs:
            cfg = _read_json(repeat_dir / "bktree" / "bktree_config.json")
            if not isinstance(cfg, dict):
                cfg = {}
            if str(cfg.get("state_id_mode") or "").lower() == "local_compact":
                local_compact += 1
            if str(cfg.get("source_mode") or "").lower() == "fresh" or str(cfg.get("mode") or "").lower() == "fresh_eval":
                fresh_eval += 1
            if str(cfg.get("initial_spawn_mode") or "").lower() in {
                "debug_spawn",
                "map_reset",
                "manual_kill_spawn",
            }:
                valid_spawn_mode += 1
            if str(cfg.get("spawn_validation") or "").lower() == "strict_count_hp":
                strict_spawn += 1
            if str(cfg.get("normalization") or "").lower() in {
                "pymarl_compatible",
                "pymarl",
                "onpolicy",
                "decision",
                "live",
                "predictionrts",
            }:
                compatible_norm += 1
            if (repeat_dir / "start_state_validation.json").exists():
                start_validation += 1
        if (
            total > 0
            and complete >= total
            and local_compact >= total
            and fresh_eval >= total
            and valid_spawn_mode >= total
            and strict_spawn >= total
            and compatible_norm >= total
            and start_validation >= total
        ):
            return True
    return False


def _is_run_suffix_experiment(item: Dict[str, Any]) -> bool:
    exp_id = str(item.get("experiment_id") or "")
    path_name = Path(str(item.get("path") or "")).name
    return bool(
        re.search(r"(?:^|_)run_\d{4,}$", exp_id)
        or re.search(r"(?:^|_)run_\d{4,}$", path_name)
    )


def _build_batch_final_eval_command(
    experiment_ids: List[str],
    episodes: int = 100,
    repeats: int = 1,
    timeout_minutes: int = 90,
    action_tuning_mode: str = "自动",
    run_suffix_only: bool = True,
    skip_complete: bool = True,
    batch_tag: Optional[str] = None,
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "batch_final_eval_archived_experiments.py"),
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--enable-mechanism-shadow-logging",
    ]
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    if skip_complete:
        cmd.append("--skip-complete")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _fixed_pool_variant_count(exp_dir: Path) -> int:
    data = _read_json(exp_dir / "fixed_pool_source_variants.json") or {}
    variants = data.get("variants") if isinstance(data.get("variants"), list) else []
    return len(variants)


def _build_fixed_pool_variant_final_eval_command(
    experiment_ids: List[str],
    episodes_per_variant: int = 100,
    repeats: int = 1,
    timeout_minutes: int = 90,
    variants_per_exp: int = 8,
    skip_complete: bool = True,
    batch_tag: Optional[str] = None,
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "batch_final_eval_fixed_pool_variants.py"),
        "--episodes-per-variant",
        str(int(episodes_per_variant)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--variants-per-exp",
        str(int(variants_per_exp)),
        "--enable-mechanism-shadow-logging",
        "--enable-planning-switch-logging",
    ]
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    if skip_complete:
        cmd.append("--skip-complete")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _build_switch_grid_eval_command(
    experiment_ids: List[str],
    method_groups: Optional[List[str]] = None,
    episodes: int = 100,
    repeats: int = 1,
    timeout_minutes: int = 120,
    action_tuning_mode: str = "自动",
    run_suffix_only: bool = True,
    batch_tag: Optional[str] = None,
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "run_switch_mechanism_eval_grid.py"),
        "--methods",
        *[str(method) for method in (method_groups or ["synergy"])],
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--enable-mechanism-shadow-logging",
        "--enable-planning-switch-logging",
    ]
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _build_fixed_pool_switch_ablation_command(
    experiment_ids: List[str],
    method_groups: Optional[List[str]] = None,
    episodes: int = 50,
    repeats: int = 1,
    timeout_minutes: int = 120,
    action_tuning_mode: str = "å¼ºåˆ¶å…³é—­",
    run_suffix_only: bool = True,
    variants_per_exp: int = 8,
    batch_tag: Optional[str] = None,
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "run_fixed_pool_switch_ablation_eval.py"),
        "--methods",
        *[str(method) for method in (method_groups or ["ETG-only"])],
        "--variants-per-exp",
        str(int(variants_per_exp)),
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--enable-mechanism-shadow-logging",
        "--enable-planning-switch-logging",
    ]
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _build_multistep_sensitivity_command(
    experiment_ids: List[str],
    method_groups: Optional[List[str]] = None,
    episodes: int = 100,
    repeats: int = 1,
    timeout_minutes: int = 120,
    action_tuning_mode: str = "è‡ªåŠ¨",
    run_suffix_only: bool = True,
    batch_tag: Optional[str] = None,
    beam_widths: str = "2,4,6",
    lookaheads: str = "2,3",
    score_modes: str = "quality,future_reward",
    action_strategies: str = "highest_transition_prob,best_subtree_quality",
    backup_distances: str = "none,0,0.2,0.5",
    backup_score_thresholds: str = "0",
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "run_multistep_sensitivity_eval_grid.py"),
        "--methods",
        *[str(method) for method in (method_groups or ["synergy"])],
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--beam-widths",
        str(beam_widths),
        "--lookaheads",
        str(lookaheads),
        "--score-modes",
        str(score_modes),
        "--action-strategies",
        str(action_strategies),
        "--backup-distances",
        str(backup_distances),
        "--backup-score-thresholds",
        str(backup_score_thresholds),
    ]
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _build_multistep_parameter_search_command(
    experiment_ids: List[str],
    scenarios: Optional[List[str]] = None,
    method_groups: Optional[List[str]] = None,
    trials: int = 60,
    episodes_per_trial: int = 100,
    timeout_minutes: int = 120,
    run_suffix_only: bool = True,
    batch_tag: Optional[str] = None,
    beam_width_range: str = "2,10",
    lookahead_range: str = "2,15",
    score_modes: str = "quality,future_reward,win_rate",
    action_strategies: str = "highest_transition_prob,best_subtree_quality,best_subtree_winrate,random_beam",
    backup_mode: str = "both",
    backup_distance_range: str = "0,1",
    backup_score_range: str = "0,1",
    high_quality_top_k: int = 8,
    high_quality_ratio: float = 0.85,
    diversity_weight: float = 0.35,
    kg_type: str = "augmented",
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "run_multistep_parameter_search.py"),
        "--trials",
        str(int(trials)),
        "--episodes-per-trial",
        str(int(episodes_per_trial)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--beam-width-range",
        str(beam_width_range),
        "--lookahead-range",
        str(lookahead_range),
        "--score-modes",
        str(score_modes),
        "--action-strategies",
        str(action_strategies),
        "--backup-mode",
        str(backup_mode),
        "--backup-distance-range",
        str(backup_distance_range),
        "--backup-score-range",
        str(backup_score_range),
        "--high-quality-top-k",
        str(int(high_quality_top_k)),
        "--high-quality-ratio",
        str(float(high_quality_ratio)),
        "--diversity-weight",
        str(float(diversity_weight)),
    ]
    if scenarios:
        cmd.extend(["--kg-type", str(kg_type)])
        for scenario in scenarios:
            cmd.extend(["--scenario", str(scenario)])
    else:
        cmd.extend(["--methods", *[str(method) for method in (method_groups or ["ETG-only"])]])
        for exp_id in experiment_ids:
            cmd.extend(["--experiment-id", str(exp_id)])
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    if run_suffix_only and not scenarios:
        cmd.append("--run-suffix-only")
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _build_fixed_pool_synergy_command(
    variant_json: str,
    experiment_ids: Optional[List[str]] = None,
    episodes_per_trial: int = 300,
    cycle_count: int = 18,
    etg_trials_per_cycle: int = 20,
    exploration_trials_per_cycle: int = 20,
    synergy_trials_per_cycle: int = 20,
    variants_per_experiment: int = 8,
    selection: str = "round_robin",
    batch_tag: Optional[str] = None,
) -> str:
    cmd = [
        "python",
        str(Path("scripts") / "run_fixed_pool_synergy_from_sensitivity.py"),
        "--variant-json",
        str(variant_json),
        "--episodes-per-trial",
        str(int(episodes_per_trial)),
        "--cycle-count",
        str(int(cycle_count)),
        "--etg-trials-per-cycle",
        str(int(etg_trials_per_cycle)),
        "--exploration-trials-per-cycle",
        str(int(exploration_trials_per_cycle)),
        "--synergy-trials-per-cycle",
        str(int(synergy_trials_per_cycle)),
        "--variants-per-experiment",
        str(int(variants_per_experiment)),
        "--selection",
        str(selection),
    ]
    if batch_tag:
        cmd.extend(["--batch-tag", str(batch_tag)])
    for exp_id in experiment_ids or []:
        cmd.extend(["--experiment-id", str(exp_id)])
    return " ".join(_quote_cli_arg(part) for part in cmd)


def _render_copyable_command(command: str, key: str) -> None:
    safe_key = re.sub(r"[^0-9A-Za-z_]", "_", key)
    st.code(command, language="powershell")
    payload = json.dumps(command, ensure_ascii=False)
    html = f"""
    <button id="copy-{safe_key}" style="
        padding: 0.35rem 0.7rem;
        border: 1px solid #d1d5db;
        border-radius: 0.35rem;
        background: #f8fafc;
        cursor: pointer;
        font-size: 0.9rem;
    ">复制命令到剪贴板</button>
    <span id="msg-{safe_key}" style="margin-left: 0.6rem; color: #16a34a; font-size: 0.9rem;"></span>
    <script>
    const btn_{safe_key} = document.getElementById("copy-{safe_key}");
    const msg_{safe_key} = document.getElementById("msg-{safe_key}");
    btn_{safe_key}.onclick = async () => {{
        const text = {payload};
        try {{
            await navigator.clipboard.writeText(text);
            msg_{safe_key}.textContent = "已复制";
        }} catch (err) {{
            const ta = document.createElement("textarea");
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
            msg_{safe_key}.textContent = "已复制";
        }}
        setTimeout(() => msg_{safe_key}.textContent = "", 1800);
    }};
    </script>
    """
    components.html(html, height=42)


def _load_final_eval_summaries(exp_dir: Path) -> List[Dict[str, Any]]:
    root = exp_dir / "final_eval"
    if not root.exists():
        return []
    summaries: List[Dict[str, Any]] = []
    for path in root.glob("*/final_eval_summary.json"):
        data = _read_json(path)
        if not data:
            continue
        data["_summary_path"] = str(path)
        data["_eval_dir"] = str(path.parent)
        summaries.append(data)
    summaries.sort(
        key=lambda item: item.get("completed_at") or item.get("updated_at") or item.get("started_at") or "",
        reverse=True,
    )
    return summaries


def _latest_final_eval(exp_dir: Path) -> Optional[Dict[str, Any]]:
    summaries = _load_final_eval_summaries(exp_dir)
    return summaries[0] if summaries else None


def _final_eval_bktree_artifacts(eval_dir_raw: Any) -> Dict[str, Any]:
    if eval_dir_raw in (None, ""):
        return {"complete": 0, "total": 0, "node_log_rows": 0, "details": []}
    eval_dir = Path(str(eval_dir_raw or ""))
    if not eval_dir.exists():
        return {"complete": 0, "total": 0, "node_log_rows": 0, "details": []}
    repeat_dirs = sorted(p for p in (eval_dir / "repeats").glob("repeat_*") if p.is_dir())
    if not repeat_dirs:
        repeat_dirs = [eval_dir]

    complete = 0
    node_log_rows = 0
    compatible_norm = 0
    start_validation = 0
    details = []
    for repeat_dir in repeat_dirs:
        bktree_dir = repeat_dir / "bktree"
        node_log = bktree_dir / "node_log.txt"
        state_node = bktree_dir / "state_node.txt"
        primary = bktree_dir / "primary_bktree.json"
        secondary_count = len(list(bktree_dir.glob("secondary_bktree_*.json"))) if bktree_dir.exists() else 0
        cfg = _read_json(bktree_dir / "bktree_config.json")
        if not isinstance(cfg, dict):
            cfg = {}
        is_compatible_norm = str(cfg.get("normalization") or "").lower() in {
            "pymarl_compatible",
            "pymarl",
            "onpolicy",
            "decision",
            "live",
            "predictionrts",
        }
        has_start_validation = (repeat_dir / "start_state_validation.json").exists()
        is_complete = node_log.exists() and state_node.exists() and primary.exists() and secondary_count > 0
        if is_complete:
            complete += 1
        if is_complete and is_compatible_norm:
            compatible_norm += 1
        if is_complete and has_start_validation:
            start_validation += 1
        if node_log.exists():
            try:
                with open(str(node_log), "r", encoding="utf-8", errors="replace") as f:
                    rows = sum(1 for line in f if line.strip())
                node_log_rows += rows
            except Exception:
                rows = 0
        else:
            rows = 0
        details.append(
            {
                "repeat": repeat_dir.name,
                "complete": is_complete,
                "node_log_rows": rows,
                "state_node": state_node.exists(),
                "primary": primary.exists(),
                "secondary_count": secondary_count,
                "normalization": cfg.get("normalization", ""),
                "compatible_norm": is_compatible_norm,
                "start_validation": has_start_validation,
            }
        )

    total = len(repeat_dirs)
    return {
        "complete": complete,
        "total": total,
        "node_log_rows": node_log_rows,
        "compatible_norm": compatible_norm,
        "start_validation": start_validation,
        "details": details,
    }


def _format_eval_bktree_status(eval_dir_raw: Any) -> str:
    status = _final_eval_bktree_artifacts(eval_dir_raw)
    total = int(status.get("total") or 0)
    if total <= 0:
        return "未生成"
    rows = int(status.get("node_log_rows") or 0)
    norm = int(status.get("compatible_norm") or 0)
    validation = int(status.get("start_validation") or 0)
    return (
        f"{status.get('complete', 0)}/{total} 完整，{rows} 条序列；"
        f"兼容归一化 {norm}/{total}，起点校验 {validation}/{total}"
    )


def _final_eval_rows(exp_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for summary in _load_final_eval_summaries(exp_dir):
        eval_dir = summary.get("_eval_dir", "")
        rows.append(
            {
                "复评ID": Path(eval_dir).name,
                "状态": summary.get("status", "-"),
                "回合/重复": f"{summary.get('episodes_per_repeat', '-')} × {summary.get('requested_repeats', '-')}",
                "总回合": _nested_get(summary, "aggregate", "total_episodes", default="-"),
                "胜率": _metric_ci(summary, "win_rate"),
                "平均得分": _metric_ci(summary, "avg_score"),
                "稳定性": _metric_ci(summary, "stability"),
                "惩罚系数": _metric_ci(summary, "penalty_factor"),
                "目标值": _metric_ci(summary, "objective"),
                "动作微调": "启用" if summary.get("action_tuning_enabled") else "关闭",
                "Eval BKTree": _format_eval_bktree_status(eval_dir),
                "完成时间": summary.get("completed_at") or summary.get("updated_at") or "-",
                "目录": eval_dir or "-",
            }
        )
    return rows


def _pct(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def _fmt_pct(value: Any) -> str:
    num = _as_float(value)
    if num is None:
        return "-"
    return f"{num * 100:.1f}%"


def _entropy(counter: Counter) -> Optional[float]:
    total = sum(counter.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _final_eval_episode_jsonl_candidates(exp_dir: Path) -> List[Path]:
    latest_eval = _latest_final_eval(exp_dir)
    candidates: List[Path] = []
    if latest_eval and latest_eval.get("_eval_dir"):
        eval_dir = Path(latest_eval["_eval_dir"])
        direct = eval_dir / "episodes.jsonl"
        if direct.exists():
            candidates.append(direct)
        candidates.extend(sorted(eval_dir.glob("repeats/*/episodes.jsonl")))
    if not candidates:
        candidates.extend(sorted(exp_dir.glob("final_eval/*/episodes.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(exp_dir.glob("final_eval/*/repeats/*/episodes.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True))
    return candidates


def _best_trial_episode_jsonl_candidates(exp_dir: Path) -> List[Path]:
    summary = _read_json(exp_dir / "study_summary.json") or {}
    best_trial = summary.get("best_trial")
    if best_trial in (None, "-"):
        return []
    try:
        trial_number = int(best_trial)
    except (TypeError, ValueError):
        return []

    candidates = [
        exp_dir / "trials" / f"trial_{trial_number:04d}" / "episodes.jsonl",
        exp_dir / "trials" / f"trial_{trial_number}" / "episodes.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def _diagnostic_episode_jsonl_candidates(exp_dir: Path, method_group: str) -> Dict[str, Any]:
    if method_group != "Replay-baseline":
        best_trial_files = _best_trial_episode_jsonl_candidates(exp_dir)
        if best_trial_files:
            return {
                "files": best_trial_files,
                "scope": "best_trial",
                "scope_label": "最优 trial",
                "fallback": False,
            }
        final_eval_files = _final_eval_episode_jsonl_candidates(exp_dir)
        return {
            "files": final_eval_files,
            "scope": "final_eval_fallback",
            "scope_label": "最终复评（未找到最优 trial episodes）",
            "fallback": True,
        }

    replay_files = _final_eval_episode_jsonl_candidates(exp_dir)
    return {
        "files": replay_files,
        "scope": "replay_eval",
        "scope_label": "回放评估",
        "fallback": False,
    }


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


@st.cache_resource(show_spinner=False)
def _load_readonly_bktree_file(path: str, distance_index: int) -> Any:
    from src.structure.BKTree_sc2 import BKTree, get_max_cluster_id
    from src.structure.custom_distance_sc2 import CustomDistance

    with open(path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)
    tree = BKTree(CustomDistance(threshold=0.5).multi_distance, distance_index=distance_index)
    tree.root = _deserialize_bktree_node(tree_data)
    if tree.root is not None:
        tree.next_cluster_id = get_max_cluster_id(tree) + 1
    return tree


def _query_common_bktree_state(
    norm_state: Dict[str, Any],
    bktree_path: str,
    primary_threshold: float,
    secondary_threshold: float,
) -> Dict[str, Any]:
    bktree_dir = ROOT_DIR / bktree_path if not Path(bktree_path).is_absolute() else Path(bktree_path)
    primary_file = bktree_dir / "primary_bktree.json"
    if not primary_file.exists():
        return {"cluster_key": None, "rejected": True, "reason": "primary_bktree_missing"}

    primary_tree = _load_readonly_bktree_file(str(primary_file), 0)
    primary_id, primary_dist = primary_tree.query_nearest(norm_state)
    if primary_id is None:
        return {"cluster_key": None, "rejected": True, "reason": "primary_query_failed"}

    secondary_file = bktree_dir / f"secondary_bktree_{int(primary_id)}.json"
    if not secondary_file.exists():
        return {
            "cluster_key": f"{int(primary_id)}:1",
            "primary_id": int(primary_id),
            "secondary_id": 1,
            "primary_distance": float(primary_dist),
            "secondary_distance": None,
            "rejected": float(primary_dist) > float(primary_threshold),
            "reason": "secondary_bktree_missing",
        }

    secondary_tree = _load_readonly_bktree_file(str(secondary_file), 1)
    secondary_id, secondary_dist = secondary_tree.query_nearest(norm_state)
    secondary_id = int(secondary_id) if secondary_id is not None else 1
    rejected = float(primary_dist) > float(primary_threshold) or float(secondary_dist) > float(secondary_threshold)
    return {
        "cluster_key": f"{int(primary_id)}:{secondary_id}",
        "primary_id": int(primary_id),
        "secondary_id": secondary_id,
        "primary_distance": float(primary_dist),
        "secondary_distance": float(secondary_dist),
        "rejected": rejected,
        "reason": "distance_over_threshold" if rejected else "accepted",
    }


def _chosen_beam_path(plan: Dict[str, Any]) -> Dict[str, Any]:
    paths = plan.get("beam_paths")
    if not isinstance(paths, list):
        return {}
    for path in paths:
        if isinstance(path, dict) and path.get("chosen"):
            return path
    return paths[0] if paths and isinstance(paths[0], dict) else {}


@st.cache_data(ttl=60, show_spinner=False)
def _compute_mechanism_diagnostics(exp_path: str, method_group: str, modified_at: float) -> Dict[str, Any]:
    exp_dir = Path(exp_path)
    source_info = _diagnostic_episode_jsonl_candidates(exp_dir, method_group)
    episode_files = source_info["files"]
    if not episode_files:
        return {"available": False, "reason": "no episodes.jsonl found", "path": ""}

    result_counts: Counter = Counter()
    action_sources: Counter = Counter()
    nid_statuses: Counter = Counter()
    plan_triggers: Counter = Counter()
    tuning_sources: Counter = Counter()
    action_codes: Counter = Counter()
    state_clusters: Counter = Counter()
    nids: Counter = Counter()

    episode_count = 0
    step_count = 0
    score_sum = 0.0
    score_count = 0
    plan_count = 0
    exact_count = 0
    ood_count = 0
    non_primary_count = 0
    no_action_count = 0
    exploration_count = 0
    high_conf_path_count = 0
    chosen_path_count = 0
    opportunity_count = 0
    validation_count = 0
    replacement_count = 0
    candidate_eligible_count = 0
    confidence_sum = 0.0
    confidence_count = 0
    adverse_events = 0
    adverse_recovered = 0
    adverse_episodes = 0
    adverse_loss_or_dogfall = 0

    for episode_file in episode_files:
        with open(str(episode_file), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode_count += 1
                result = episode.get("result", "-")
                result_counts[result] += 1
                score = _as_float(episode.get("score", episode.get("final_score")))
                if score is not None:
                    score_sum += score
                    score_count += 1
                frames = episode.get("frames") or episode.get("steps") or []
                if not isinstance(frames, list):
                    continue

                episode_adverse = False
                advantages = []
                for frame in frames:
                    if not isinstance(frame, dict):
                        continue
                    hp_my = _as_float(frame.get("hp_my"))
                    hp_enemy = _as_float(frame.get("hp_enemy"))
                    advantages.append((hp_my - hp_enemy) if hp_my is not None and hp_enemy is not None else None)

                for idx, frame in enumerate(frames):
                    if not isinstance(frame, dict):
                        continue
                    step_count += 1
                    action_code = frame.get("action_code")
                    if action_code:
                        action_codes[str(action_code)] += 1
                    else:
                        no_action_count += 1

                    nid = frame.get("eval_state_id", frame.get("nid"))
                    if nid is not None:
                        nids[str(nid)] += 1
                    cluster = frame.get("state_cluster")
                    if cluster is not None:
                        state_clusters[str(cluster)] += 1

                    source = frame.get("action_source")
                    if source:
                        action_sources[str(source)] += 1
                        if source not in {"kg_plan", "kg_follow"}:
                            non_primary_count += 1
                    if frame.get("is_exploration"):
                        exploration_count += 1

                    nid_status = frame.get("nid_status")
                    if nid_status:
                        nid_statuses[str(nid_status)] += 1
                        if nid_status == "exact":
                            exact_count += 1
                    if frame.get("nid_is_ood") or nid_status in {"bktree_rejected", "ood", "missing"}:
                        ood_count += 1

                    plan = frame.get("plan")
                    if isinstance(plan, dict):
                        plan_count += 1
                        trigger = plan.get("trigger")
                        if trigger:
                            plan_triggers[str(trigger)] += 1
                        chosen_path = _chosen_beam_path(plan)
                        if chosen_path:
                            chosen_path_count += 1
                            steps = chosen_path.get("steps")
                            win_rates = [
                                _as_float(step.get("win_rate"))
                                for step in steps
                                if isinstance(step, dict) and _as_float(step.get("win_rate")) is not None
                            ] if isinstance(steps, list) else []
                            if win_rates and max(win_rates) >= 0.9:
                                high_conf_path_count += 1
                        tuning = plan.get("action_tuning")
                        if isinstance(tuning, dict):
                            if tuning.get("source"):
                                tuning_sources[str(tuning.get("source"))] += 1
                            if tuning.get("opportunity"):
                                opportunity_count += 1
                            if tuning.get("validation"):
                                validation_count += 1
                            if tuning.get("candidate_eligible"):
                                candidate_eligible_count += 1
                            confidence = _as_float(tuning.get("confidence"))
                            if confidence is not None:
                                confidence_sum += confidence
                                confidence_count += 1
                            if (
                                tuning.get("etg_action")
                                and tuning.get("action")
                                and tuning.get("etg_action") != tuning.get("action")
                            ):
                                replacement_count += 1

                    hp_delta = _as_float(frame.get("hp_delta"))
                    if hp_delta is not None and hp_delta < -12:
                        adverse_events += 1
                        episode_adverse = True
                        current_advantage = advantages[idx] if idx < len(advantages) else None
                        future_advantages = [
                            adv for adv in advantages[idx + 1 : idx + 4] if adv is not None
                        ]
                        if current_advantage is not None and future_advantages and max(future_advantages) >= current_advantage:
                            adverse_recovered += 1

                if episode_adverse:
                    adverse_episodes += 1
                    if result != "Win":
                        adverse_loss_or_dogfall += 1

    state_distribution = nids if nids else state_clusters
    state_entropy = _entropy(state_distribution)
    state_unique = len(state_distribution)
    state_entropy_norm = (
        state_entropy / math.log2(state_unique)
        if state_entropy is not None and state_unique > 1
        else None
    )

    return {
        "available": True,
        "path": str(episode_files[0]),
        "scope": source_info.get("scope"),
        "scope_label": source_info.get("scope_label"),
        "scope_fallback": bool(source_info.get("fallback")),
        "file_count": len(episode_files),
        "episodes": episode_count,
        "steps": step_count,
        "avg_score": (score_sum / score_count) if score_count else None,
        "win_rate": _pct(result_counts.get("Win", 0), episode_count),
        "dogfall_rate": _pct(result_counts.get("Dogfall", 0), episode_count),
        "loss_rate": _pct(result_counts.get("Loss", 0), episode_count),
        "unique_nids": len(nids),
        "unique_state_clusters": len(state_clusters),
        "state_visit_entropy": state_entropy,
        "state_visit_entropy_norm": state_entropy_norm,
        "action_entropy": _entropy(action_codes),
        "exact_ratio": _pct(exact_count, step_count),
        "ood_ratio": _pct(ood_count, step_count),
        "non_primary_action_ratio": _pct(non_primary_count, step_count),
        "exploration_ratio": _pct(exploration_count, step_count),
        "no_action_ratio": _pct(no_action_count, step_count),
        "plan_ratio": _pct(plan_count, step_count),
        "high_conf_path_ratio": _pct(high_conf_path_count, chosen_path_count),
        "tuning_opportunity_ratio": _pct(opportunity_count, plan_count),
        "tuning_validation_ratio": _pct(validation_count, plan_count),
        "tuning_replacement_ratio": _pct(replacement_count, plan_count),
        "candidate_eligible_ratio": _pct(candidate_eligible_count, plan_count),
        "avg_tuning_confidence": (confidence_sum / confidence_count) if confidence_count else None,
        "adverse_event_ratio": _pct(adverse_events, step_count),
        "adverse_recovery_ratio": _pct(adverse_recovered, adverse_events),
        "adverse_bad_outcome_ratio": _pct(adverse_loss_or_dogfall, adverse_episodes),
        "result_counts": dict(result_counts),
        "action_sources": dict(action_sources),
        "nid_statuses": dict(nid_statuses),
        "plan_triggers": dict(plan_triggers),
        "tuning_sources": dict(tuning_sources),
        "method_group": method_group,
        "modified_at": modified_at,
    }


def _map_sort_key(item: Dict[str, Any]) -> int:
    map_id = str(item.get("map_id", ""))
    map_key = str(item.get("map_key", ""))
    return _MAP_ORDER.get(map_id, _MAP_KEY_ORDER.get(map_key, 999))


def _method_sort_key(item: Dict[str, Any]) -> int:
    return _METHOD_ORDER.get(str(item.get("method_group", "")), 999)


def _sort_experiments(
    experiments: List[Dict[str, Any]], sort_mode: str
) -> List[Dict[str, Any]]:
    if sort_mode == "方法组 → 地图":
        return sorted(
            experiments,
            key=lambda item: (
                _method_sort_key(item),
                _map_sort_key(item),
                str(item.get("experiment_id", "")),
            ),
        )
    if sort_mode == "最近更新":
        return sorted(experiments, key=lambda item: float(item.get("modified_at", 0)), reverse=True)
    if sort_mode == "实验ID":
        return sorted(experiments, key=lambda item: str(item.get("experiment_id", "")).lower())
    return sorted(
        experiments,
        key=lambda item: (
            _map_sort_key(item),
            _method_sort_key(item),
            str(item.get("experiment_id", "")),
        ),
    )


def _scenario_option_value(map_key: Any, map_id: Any) -> str:
    return f"{map_key or '-'}||{map_id or '-'}"


def _scenario_filter_options(experiments: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    seen = set()
    for scenario in _SCENARIOS:
        value = _scenario_option_value(scenario["map_key"], scenario["map_id"])
        options.append(
            {
                "value": value,
                "label": _scenario_label(scenario),
                "map_key": scenario["map_key"],
                "map_id": scenario["map_id"],
            }
        )
        seen.add(value)

    for item in sorted(experiments, key=lambda exp: (_map_sort_key(exp), str(exp.get("experiment_id", "")))):
        map_key = str(item.get("map_key", "-"))
        map_id = str(item.get("map_id", "-"))
        value = _scenario_option_value(map_key, map_id)
        if value in seen:
            continue
        options.append(
            {
                "value": value,
                "label": f"未登记场景 | {map_key} | {map_id}",
                "map_key": map_key,
                "map_id": map_id,
            }
        )
        seen.add(value)
    return options


def _filter_experiments_by_scenarios(
    experiments: List[Dict[str, Any]],
    scenario_options: List[Dict[str, str]],
    selected_values: List[str],
) -> List[Dict[str, Any]]:
    if not selected_values or len(selected_values) == len(scenario_options):
        return list(experiments)

    selected = {
        (option["map_key"], option["map_id"])
        for option in scenario_options
        if option["value"] in selected_values
    }
    return [
        item
        for item in experiments
        if (str(item.get("map_key", "-")), str(item.get("map_id", "-"))) in selected
    ]


def _method_label(item: Dict[str, Any]) -> str:
    group = str(item.get("method_group", ""))
    if group == "synergy":
        return "ETG+MC Synergy"
    if group == "ETG-only":
        return "ETG-only"
    if group == "MC-only":
        return "MC-only"
    if group == "Replay-baseline":
        return "Historical Replay"
    return group or "-"


def _paper_table_rows(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in experiments:
        latest_eval = _latest_final_eval(Path(item["path"]))
        rows.append(
            {
                "Scenario": item.get("map_key", "-"),
                "Map": item.get("map_id", "-"),
                "Method": _method_label(item),
                "ETG": item.get("kg_name", "-"),
                "BKTree": f"{item['bktree_primary']} / {item['bktree_secondary']}",
                "Best Obj.": _fmt(item.get("best_value")),
                "Best WR": _fmt(item.get("best_win_rate")),
                "Best Score": _fmt(item.get("best_avg_score")),
                "Eval WR": _metric_ci(latest_eval, "win_rate") if latest_eval else "-",
                "Eval Score": _metric_ci(latest_eval, "avg_score") if latest_eval else "-",
                "Eval Stability": _metric_ci(latest_eval, "stability") if latest_eval else "-",
                "Eval Obj.": _metric_ci(latest_eval, "objective") if latest_eval else "-",
                "Eval Episodes": _nested_get(latest_eval, "aggregate", "total_episodes", default="-") if latest_eval else "-",
            }
        )
    return rows


def _start_final_eval_job(
    exp_dir: Path,
    episodes: int,
    repeats: int,
    timeout_minutes: int,
    action_tuning_mode: str,
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = exp_dir / "final_eval" / f"eval_{timestamp}"
    if output_dir.exists():
        for idx in range(1, 1000):
            candidate = output_dir.with_name(f"{output_dir.name}_{idx:03d}")
            if not candidate.exists():
                output_dir = candidate
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "final_eval.log"

    cmd = [
        sys.executable,
        str(_FINAL_EVAL_SCRIPT),
        "--experiment-dir",
        str(exp_dir),
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--output-dir",
        str(output_dir),
        "--eval-bktree-normalization",
        "decision",
        "--enable-mechanism-shadow-logging",
    ]
    if action_tuning_mode == "强制启用":
        cmd.append("--enable-action-tuning")
    elif action_tuning_mode == "强制关闭":
        cmd.append("--disable-action-tuning")

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_batch_final_eval_job(
    experiment_ids: List[str],
    episodes: int,
    repeats: int,
    timeout_minutes: int,
    action_tuning_mode: str,
    run_suffix_only: bool,
    skip_complete: bool,
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"web_batch_{timestamp}"
    launcher_dir = _ALL_DATA_ROOT / "_batch_final_eval_logs" / batch_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_path = launcher_dir / "web_batch_launcher.log"
    cmd = [
        sys.executable,
        str(_BATCH_FINAL_EVAL_SCRIPT),
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--batch-tag",
        batch_id,
        "--enable-mechanism-shadow-logging",
    ]
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    if skip_complete:
        cmd.append("--skip-complete")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "batch_id": batch_id,
        "count": len(experiment_ids),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_fixed_pool_variant_final_eval_job(
    experiment_ids: List[str],
    episodes_per_variant: int,
    repeats: int,
    timeout_minutes: int,
    variants_per_exp: int,
    skip_complete: bool,
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_tag = f"web_fpv_{timestamp}"
    batch_id = f"fixed_pool_variant_eval_{re.sub(r'[^0-9A-Za-z_.-]+', '_', batch_tag).strip('_')}"
    launcher_dir = _ALL_DATA_ROOT / "_batch_final_eval_logs" / batch_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_path = launcher_dir / "web_fixed_pool_variant_eval_launcher.log"
    cmd = [
        sys.executable,
        str(_FIXED_POOL_VARIANT_FINAL_EVAL_SCRIPT),
        "--episodes-per-variant",
        str(int(episodes_per_variant)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--variants-per-exp",
        str(int(variants_per_exp)),
        "--batch-tag",
        batch_tag,
        "--enable-mechanism-shadow-logging",
        "--enable-planning-switch-logging",
    ]
    if skip_complete:
        cmd.append("--skip-complete")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "batch_id": batch_id,
        "count": len(experiment_ids),
        "variant_count": int(variants_per_exp),
        "episodes_per_variant": int(episodes_per_variant),
        "job_count": len(experiment_ids) * int(variants_per_exp),
        "total_episodes": len(experiment_ids) * int(variants_per_exp) * int(episodes_per_variant) * int(repeats),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_switch_grid_eval_job(
    experiment_ids: List[str],
    method_groups: List[str],
    episodes: int,
    repeats: int,
    timeout_minutes: int,
    action_tuning_mode: str,
    run_suffix_only: bool,
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_tag = f"web_switch_grid_{timestamp}"
    batch_id = f"batch_switch_grid_{re.sub(r'[^0-9A-Za-z_.-]+', '_', batch_tag).strip('_')}"
    launcher_dir = _ALL_DATA_ROOT / "_batch_final_eval_logs" / batch_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_path = launcher_dir / "web_switch_grid_launcher.log"
    cmd = [
        sys.executable,
        str(_SWITCH_GRID_EVAL_SCRIPT),
        "--methods",
        *[str(method) for method in method_groups],
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--batch-tag",
        batch_tag,
        "--enable-mechanism-shadow-logging",
        "--enable-planning-switch-logging",
    ]
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "batch_id": batch_id,
        "count": len(experiment_ids),
        "variant_count": 5,
        "job_count": len(experiment_ids) * 5,
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_fixed_pool_switch_ablation_job(
    experiment_ids: List[str],
    method_groups: List[str],
    episodes: int,
    repeats: int,
    timeout_minutes: int,
    action_tuning_mode: str,
    run_suffix_only: bool,
    variants_per_exp: int,
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_tag = f"web_fixed_pool_switch_pair_{timestamp}"
    batch_id = f"batch_fixed_pool_switch_pair_{re.sub(r'[^0-9A-Za-z_.-]+', '_', batch_tag).strip('_')}"
    launcher_dir = _ALL_DATA_ROOT / "_batch_final_eval_logs" / batch_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_path = launcher_dir / "web_fixed_pool_switch_pair_launcher.log"
    cmd = [
        sys.executable,
        str(_FIXED_POOL_SWITCH_ABLATION_SCRIPT),
        "--methods",
        *[str(method) for method in method_groups],
        "--variants-per-exp",
        str(int(variants_per_exp)),
        "--episodes",
        str(int(episodes)),
        "--repeats",
        str(int(repeats)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--action-tuning",
        _action_tuning_cli_value(action_tuning_mode),
        "--batch-tag",
        batch_tag,
        "--enable-mechanism-shadow-logging",
        "--enable-planning-switch-logging",
    ]
    if run_suffix_only:
        cmd.append("--run-suffix-only")
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "batch_id": batch_id,
        "count": len(experiment_ids),
        "base_variant_count": int(variants_per_exp),
        "switch_variant_count": 4,
        "job_count": len(experiment_ids) * int(variants_per_exp) * 4,
        "total_episodes": len(experiment_ids) * int(variants_per_exp) * 4 * int(episodes) * int(repeats),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_multistep_sensitivity_job(
    experiment_ids: List[str],
    scenarios: List[str],
    method_groups: List[str],
    trials: int,
    episodes_per_trial: int,
    timeout_minutes: int,
    run_suffix_only: bool,
    beam_width_range: str,
    lookahead_range: str,
    score_modes: str,
    action_strategies: str,
    backup_mode: str,
    backup_distance_range: str,
    backup_score_range: str,
    high_quality_top_k: int,
    high_quality_ratio: float,
    diversity_weight: float,
    kg_type: str = "augmented",
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_tag = f"web_multistep_param_search_{timestamp}"
    batch_id = f"batch_multistep_param_search_{re.sub(r'[^0-9A-Za-z_.-]+', '_', batch_tag).strip('_')}"
    launcher_dir = _ALL_DATA_ROOT / "_batch_parameter_search_logs" / batch_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_path = launcher_dir / "web_multistep_parameter_search_launcher.log"
    cmd = [
        sys.executable,
        str(_MULTISTEP_PARAMETER_SEARCH_SCRIPT),
        "--trials",
        str(int(trials)),
        "--episodes-per-trial",
        str(int(episodes_per_trial)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--batch-tag",
        batch_tag,
        "--beam-width-range",
        str(beam_width_range),
        "--lookahead-range",
        str(lookahead_range),
        "--score-modes",
        str(score_modes),
        "--action-strategies",
        str(action_strategies),
        "--backup-mode",
        str(backup_mode),
        "--backup-distance-range",
        str(backup_distance_range),
        "--backup-score-range",
        str(backup_score_range),
        "--high-quality-top-k",
        str(int(high_quality_top_k)),
        "--high-quality-ratio",
        str(float(high_quality_ratio)),
        "--diversity-weight",
        str(float(diversity_weight)),
    ]
    if scenarios:
        cmd.extend(["--kg-type", str(kg_type)])
        for scenario in scenarios:
            cmd.extend(["--scenario", str(scenario)])
    else:
        cmd.extend(["--methods", *[str(method) for method in method_groups]])
        for exp_id in experiment_ids:
            cmd.extend(["--experiment-id", str(exp_id)])
    if run_suffix_only and not scenarios:
        cmd.append("--run-suffix-only")

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "batch_id": batch_id,
        "count": len(scenarios or experiment_ids),
        "trial_count": int(trials),
        "job_count": len(scenarios or experiment_ids),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_fixed_pool_synergy_job(
    variant_json: str,
    experiment_ids: List[str],
    episodes_per_trial: int,
    cycle_count: int,
    etg_trials_per_cycle: int,
    exploration_trials_per_cycle: int,
    synergy_trials_per_cycle: int,
    variants_per_experiment: int,
    selection: str,
) -> Dict[str, Any]:
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_tag = f"web_fixed_pool_synergy_{timestamp}"
    batch_id = f"fixed_pool_synergy_{re.sub(r'[^0-9A-Za-z_.-]+', '_', batch_tag).strip('_')}"
    launcher_dir = _ALL_DATA_ROOT / "_batch_final_eval_logs" / batch_id
    launcher_dir.mkdir(parents=True, exist_ok=True)
    log_path = launcher_dir / "web_fixed_pool_synergy_launcher.log"
    cmd = [
        sys.executable,
        str(_FIXED_POOL_SYNERGY_SCRIPT),
        "--variant-json",
        str(variant_json),
        "--episodes-per-trial",
        str(int(episodes_per_trial)),
        "--cycle-count",
        str(int(cycle_count)),
        "--etg-trials-per-cycle",
        str(int(etg_trials_per_cycle)),
        "--exploration-trials-per-cycle",
        str(int(exploration_trials_per_cycle)),
        "--synergy-trials-per-cycle",
        str(int(synergy_trials_per_cycle)),
        "--variants-per-experiment",
        str(int(variants_per_experiment)),
        "--selection",
        str(selection),
        "--batch-tag",
        batch_tag,
    ]
    for exp_id in experiment_ids:
        cmd.extend(["--experiment-id", str(exp_id)])

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
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "batch_id": batch_id,
        "count": len(experiment_ids),
        "total_trials": int(cycle_count)
        * (
            int(etg_trials_per_cycle)
            + int(exploration_trials_per_cycle)
            + int(synergy_trials_per_cycle)
        ),
        "log_path": str(log_path),
        "variant_json": str(variant_json),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _scenario_label(scenario: Dict[str, Any]) -> str:
    return f"{scenario['scenario']} | {scenario['map_key']} | {scenario['map_id']}"


def _default_replay_thresholds(map_id: str) -> Dict[str, float]:
    if "MvsM_8" in map_id:
        return {"primary": 1.0, "secondary": 0.5}
    return {"primary": 0.7, "secondary": 0.5}


def _build_replay_baseline_cmd_list(
    scenario: Dict[str, Any],
    top_k: int,
    episodes: int,
    timeout_minutes: int,
    experiment_id: str,
    overwrite: bool,
    dry_run: bool,
) -> List[str]:
    catalog = _load_kg_catalog_entries()
    entry = _find_catalog_entry(catalog, scenario) or {}
    source_dir = _REPLAY_SOURCE_DIRS.get(scenario["map_id"], "")
    thresholds = _default_replay_thresholds(scenario["map_id"])

    cmd = [
        "python",
        str(Path("scripts") / "evaluate_replay_baseline.py"),
        "--experiment-id",
        experiment_id,
        "--display-name",
        f"{scenario['scenario']} Historical Action Replay Baseline",
        "--map-key",
        scenario["map_key"],
        "--map-id",
        scenario["map_id"],
        "--action-log",
        str(Path(source_dir) / "action_log.csv"),
        "--result-log",
        str(Path(source_dir) / "game_result.txt"),
        "--data-dir",
        str(entry.get("data_dir") or f"data/{scenario['map_id']}/augmented_1"),
        "--kg-name",
        str(entry.get("name") or scenario.get("preferred_kg", "")),
        "--kg-file",
        str(entry.get("file") or ""),
        "--transitions",
        str(entry.get("transitions") or ""),
        "--dataset-type",
        str(entry.get("type") or "augmented"),
        "--replay-dataset-expansion",
        "--selection",
        "best_pool",
        "--top-k",
        str(int(top_k)),
        "--sequence-allocation",
        "random",
        "--episodes",
        str(int(episodes)),
        "--timeout-minutes",
        str(int(timeout_minutes)),
        "--primary-threshold",
        str(thresholds["primary"]),
        "--secondary-threshold",
        str(thresholds["secondary"]),
    ]
    if overwrite:
        cmd.append("--overwrite")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_replay_baseline_command(*args, **kwargs) -> str:
    return " ".join(_quote_cli_arg(part) for part in _build_replay_baseline_cmd_list(*args, **kwargs))


def _parse_positive_int_values(raw: str, default: List[int], min_value: int, max_value: int) -> List[int]:
    values: List[int] = []
    for part in re.split(r"[,，\s]+", str(raw or "").strip()):
        if not part:
            continue
        value = int(part)
        if value < min_value or value > max_value:
            raise ValueError(f"{value} is outside [{min_value}, {max_value}]")
        if value not in values:
            values.append(value)
    return values or list(default)


def _default_replay_experiment_id(
    scenario: Dict[str, Any],
    top_k: int,
    episodes: int,
    timeout_minutes: int,
    include_episodes: bool,
    include_timeout: bool,
    prefix: str,
) -> str:
    parts = [scenario["scenario"], "replay", "best", f"top{int(top_k)}"]
    if include_episodes:
        parts.append(f"n{int(episodes)}")
    if include_timeout:
        parts.append(f"t{int(timeout_minutes)}m")
    experiment_id = "_".join(parts)
    prefix = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(prefix or "").strip()).strip("_")
    return f"{prefix}_{experiment_id}" if prefix else experiment_id


def _build_replay_baseline_specs(
    scenarios: List[Dict[str, Any]],
    top_k_values: List[int],
    episode_values: List[int],
    timeout_values: List[int],
    overwrite: bool,
    dry_run: bool,
    experiment_prefix: str,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    include_episodes = len(episode_values) > 1 or episode_values[0] != 100
    include_timeout = len(timeout_values) > 1 or timeout_values[0] != 90
    for scenario in scenarios:
        for top_k in top_k_values:
            for episodes in episode_values:
                for timeout_minutes in timeout_values:
                    experiment_id = _default_replay_experiment_id(
                        scenario,
                        top_k,
                        episodes,
                        timeout_minutes,
                        include_episodes,
                        include_timeout,
                        experiment_prefix,
                    )
                    cmd = _build_replay_baseline_cmd_list(
                        scenario,
                        int(top_k),
                        int(episodes),
                        int(timeout_minutes),
                        experiment_id,
                        bool(overwrite),
                        bool(dry_run),
                    )
                    specs.append(
                        {
                            "scenario": scenario,
                            "top_k": int(top_k),
                            "episodes": int(episodes),
                            "timeout_minutes": int(timeout_minutes),
                            "experiment_id": experiment_id,
                            "cmd": cmd,
                            "command": " ".join(_quote_cli_arg(part) for part in cmd),
                        }
                    )
    return specs


def _start_replay_baseline_job(cmd: List[str], experiment_id: str) -> Dict[str, Any]:
    output_dir = _ALL_DATA_ROOT / "Replay-baseline" / experiment_id
    log_dir = _ALL_DATA_ROOT / "Replay-baseline" / "_launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{experiment_id}_{timestamp}.log"
    run_cmd = list(cmd)
    if run_cmd and run_cmd[0] == "python":
        run_cmd[0] = sys.executable
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log_file = open(str(log_path), "w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        run_cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        creationflags=flags,
        env=env,
    )
    return {
        "pid": proc.pid,
        "cmd": run_cmd,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _start_replay_baseline_batch_job(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    log_dir = _ALL_DATA_ROOT / "Replay-baseline" / "_launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"replay_batch_{timestamp}"
    payload_path = log_dir / f"{batch_id}.json"
    log_path = log_dir / f"{batch_id}.log"
    payload = [
        {
            "experiment_id": spec["experiment_id"],
            "cmd": spec["cmd"],
        }
        for spec in specs
    ]
    with open(str(payload_path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    launcher = (
        "import json, os, subprocess, sys\n"
        "payload_path, root_dir = sys.argv[1], sys.argv[2]\n"
        "with open(payload_path, 'r', encoding='utf-8') as f:\n"
        "    payload = json.load(f)\n"
        "env = dict(os.environ)\n"
        "env['PYTHONIOENCODING'] = 'utf-8'\n"
        "for idx, item in enumerate(payload, start=1):\n"
        "    cmd = list(item['cmd'])\n"
        "    if cmd and cmd[0] == 'python':\n"
        "        cmd[0] = sys.executable\n"
        "    print(f\"[BATCH] {idx}/{len(payload)} START {item['experiment_id']}\", flush=True)\n"
        "    rc = subprocess.run(cmd, cwd=root_dir, env=env).returncode\n"
        "    print(f\"[BATCH] {idx}/{len(payload)} DONE {item['experiment_id']} rc={rc}\", flush=True)\n"
    )

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log_file = open(str(log_path), "w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher, str(payload_path), str(ROOT_DIR)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        creationflags=flags,
        env=env,
    )
    return {
        "pid": proc.pid,
        "batch_id": batch_id,
        "count": len(specs),
        "payload_path": str(payload_path),
        "log_path": str(log_path),
        "experiments": [spec["experiment_id"] for spec in specs],
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _render_replay_baseline_panel_legacy() -> None:
    st.subheader("历史动作序列回放对照组")
    st.caption(
        "从各地图历史 `action_log.csv` 中选择最优/较优动作序列，直接按记录动作执行 N 局；结果自动归档到 `output/learner_results/all_data/Replay-baseline/<experiment_id>/`。"
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    scenario_idx = c1.selectbox(
        "选择地图",
        options=range(len(_SCENARIOS)),
        format_func=lambda idx: _scenario_label(_SCENARIOS[idx]),
        key="replay_baseline_scenario",
    )
    scenario = _SCENARIOS[int(scenario_idx)]
    top_k = c2.number_input(
        "最优序列池大小",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        key="replay_baseline_top_k",
        help="设为 1 时一直重复最高结果序列；大于 1 时从前 K 条较优序列中随机分配 N 局。",
    )
    episodes = c3.number_input(
        "执行局数 N",
        min_value=1,
        max_value=2000,
        value=100,
        step=10,
        key="replay_baseline_episodes",
    )

    d1, d2, d3 = st.columns([2, 1, 1])
    default_exp_id = f"{scenario['scenario']}_replay_best_top{int(top_k)}"
    experiment_id = d1.text_input(
        "实验ID",
        value=default_exp_id,
        key=f"replay_baseline_exp_id_{scenario['map_key']}_{int(top_k)}",
    )
    timeout_minutes = d2.number_input(
        "单序列超时分钟",
        min_value=5,
        max_value=240,
        value=90,
        step=5,
        key="replay_baseline_timeout",
    )
    overwrite = d3.toggle(
        "覆盖同名归档",
        value=False,
        key="replay_baseline_overwrite",
    )
    dry_run = st.toggle(
        "仅生成 manifest/候选序列，不启动 SC2",
        value=False,
        key="replay_baseline_dry_run",
    )

    cmd_list = _build_replay_baseline_cmd_list(
        scenario,
        int(top_k),
        int(episodes),
        int(timeout_minutes),
        experiment_id.strip() or default_exp_id,
        bool(overwrite),
        bool(dry_run),
    )
    command = " ".join(_quote_cli_arg(part) for part in cmd_list)

    source_dir = _REPLAY_SOURCE_DIRS.get(scenario["map_id"], "")
    action_log = ROOT_DIR / source_dir / "action_log.csv"
    result_log = ROOT_DIR / source_dir / "game_result.txt"
    expected_dir = _ALL_DATA_ROOT / "Replay-baseline" / (experiment_id.strip() or default_exp_id)
    status_rows = [
        {"项目": "action_log.csv", "路径": str(action_log), "状态": "存在" if action_log.exists() else "缺失"},
        {"项目": "game_result.txt", "路径": str(result_log), "状态": "存在" if result_log.exists() else "缺失"},
        {"项目": "归档目录", "路径": str(expected_dir), "状态": "已存在" if expected_dir.exists() else "将创建"},
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    with st.expander("历史回放基线命令（可直接复制到 PowerShell 执行）", expanded=True):
        _render_copyable_command(command, f"replay_baseline_{experiment_id}")

    if st.button("启动历史回放基线并自动归档", key="start_replay_baseline_job", use_container_width=True):
        try:
            job = _start_replay_baseline_job(cmd_list, experiment_id.strip() or default_exp_id)
            st.session_state.setdefault("replay_baseline_jobs", {})[experiment_id] = job
            st.success(f"已启动历史回放基线 PID={job['pid']}，日志：`{job['log_path']}`")
        except Exception as exc:
            st.error(f"启动历史回放基线失败：{exc}")

    job = st.session_state.get("replay_baseline_jobs", {}).get(experiment_id)
    if job:
        st.info(f"最近启动：PID={job['pid']}，输出目录 `{job['output_dir']}`")
        log_path = Path(job["log_path"])
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                with st.expander("历史回放日志尾部", expanded=False):
                    st.code("\n".join(lines[-80:]))
            except Exception:
                pass


def _render_replay_baseline_panel() -> None:
    st.subheader("历史动作序列回放对照组")
    st.caption(
        "从历史 `action_log.csv` 中选择 best_pool 动作序列，按组合规格批量回放；批量启动时按顺序执行，避免同时启动多个 SC2 客户端。"
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    scenario_indices = c1.multiselect(
        "选择地图（可多选）",
        options=range(len(_SCENARIOS)),
        default=[0],
        format_func=lambda idx: _scenario_label(_SCENARIOS[idx]),
        key="replay_baseline_scenarios_multi",
    )
    top_k_raw = c2.text_input(
        "最优序列池大小（可多个）",
        value="1",
        key="replay_baseline_top_k_values",
        help="逗号或空格分隔，例如 `1,3,5`。top1 表示重复最高分历史序列；topK 表示从前 K 条最高分序列池随机分配 N 局。",
    )
    episodes_raw = c3.text_input(
        "执行局数 N（可多个）",
        value="100",
        key="replay_baseline_episode_values",
        help="逗号或空格分隔，例如 `100,200`。",
    )

    d1, d2, d3 = st.columns([2, 1, 1])
    experiment_prefix = d1.text_input(
        "实验ID前缀（可选）",
        value="",
        key="replay_baseline_experiment_prefix",
        help="为空时自动生成 `<scenario>_replay_best_topK`；多 N 或多 timeout 时自动附加 `nN`、`tXm`。",
    )
    timeout_raw = d2.text_input(
        "超时分钟（可多个）",
        value="90",
        key="replay_baseline_timeout_values",
        help="逗号或空格分隔。批量启动时每个组合使用对应 timeout。",
    )
    overwrite = d3.toggle(
        "覆盖同名归档",
        value=False,
        key="replay_baseline_overwrite",
    )
    dry_run = st.toggle(
        "仅生成 manifest/候选序列，不启动 SC2",
        value=False,
        key="replay_baseline_dry_run",
    )

    try:
        selected_scenarios = [_SCENARIOS[int(idx)] for idx in scenario_indices]
        top_k_values = _parse_positive_int_values(top_k_raw, [1], 1, 100)
        episode_values = _parse_positive_int_values(episodes_raw, [100], 1, 2000)
        timeout_values = _parse_positive_int_values(timeout_raw, [90], 5, 240)
    except Exception as exc:
        st.error(f"批量参数解析失败：{exc}")
        return

    if not selected_scenarios:
        st.info("请至少选择一个地图。")
        return

    specs = _build_replay_baseline_specs(
        selected_scenarios,
        top_k_values,
        episode_values,
        timeout_values,
        bool(overwrite),
        bool(dry_run),
        experiment_prefix,
    )
    if not specs:
        st.info("当前组合为空。")
        return

    status_rows = []
    existing_specs = []
    for spec in specs:
        scenario = spec["scenario"]
        source_dir = _REPLAY_SOURCE_DIRS.get(scenario["map_id"], "")
        action_log = ROOT_DIR / source_dir / "action_log.csv"
        result_log = ROOT_DIR / source_dir / "game_result.txt"
        expected_dir = _ALL_DATA_ROOT / "Replay-baseline" / spec["experiment_id"]
        if expected_dir.exists():
            existing_specs.append(spec)
        status_rows.append(
            {
                "实验ID": spec["experiment_id"],
                "地图": scenario["map_id"],
                "topK": spec["top_k"],
                "N": spec["episodes"],
                "timeout": spec["timeout_minutes"],
                "action_log": "存在" if action_log.exists() else "缺失",
                "game_result": "存在" if result_log.exists() else "缺失",
                "归档目录": "已存在" if expected_dir.exists() else "将创建",
            }
        )
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    has_conflict = bool(existing_specs) and not overwrite
    if has_conflict:
        conflict_names = ", ".join(spec["experiment_id"] for spec in existing_specs[:8])
        if len(existing_specs) > 8:
            conflict_names += f" 等 {len(existing_specs)} 个"
        st.error(
            "以下归档目录已存在，且未勾选“覆盖同名归档”，因此启动后会在进入 SC2 前直接退出："
            f"{conflict_names}。请勾选覆盖、修改实验ID前缀，或手动删除/移动旧目录后再启动。"
        )

    command_text = "\n".join(spec["command"] for spec in specs)
    with st.expander("历史回放基线命令（按行顺序执行，可直接复制到 PowerShell）", expanded=True):
        _render_copyable_command(command_text, f"replay_baseline_batch_{len(specs)}")

    if len(specs) > 1:
        st.caption(f"当前将生成 {len(specs)} 个组合；点击启动后由一个后台批量启动器按顺序运行。")

    if st.button(
        "启动历史回放基线批量任务并自动归档",
        key="start_replay_baseline_batch_job",
        use_container_width=True,
        disabled=has_conflict,
    ):
        try:
            if len(specs) == 1:
                spec = specs[0]
                job = _start_replay_baseline_job(spec["cmd"], spec["experiment_id"])
                st.session_state.setdefault("replay_baseline_jobs", {})[spec["experiment_id"]] = job
                st.success(f"已启动历史回放基线 PID={job['pid']}，日志：`{job['log_path']}`")
            else:
                job = _start_replay_baseline_batch_job(specs)
                st.session_state.setdefault("replay_baseline_batch_jobs", {})[job["batch_id"]] = job
                st.success(f"已启动批量历史回放 PID={job['pid']}，组合数={job['count']}，日志：`{job['log_path']}`")
        except Exception as exc:
            st.error(f"启动历史回放基线失败：{exc}")

    jobs = st.session_state.get("replay_baseline_jobs", {})
    batch_jobs = st.session_state.get("replay_baseline_batch_jobs", {})
    if jobs or batch_jobs:
        with st.expander("历史回放后台任务", expanded=False):
            rows = []
            for exp_id, job in jobs.items():
                rows.append(
                    {
                        "类型": "single",
                        "ID": exp_id,
                        "PID": job.get("pid"),
                        "数量": 1,
                        "日志": job.get("log_path"),
                        "启动时间": job.get("started_at"),
                    }
                )
            for batch_id, job in batch_jobs.items():
                rows.append(
                    {
                        "类型": "batch",
                        "ID": batch_id,
                        "PID": job.get("pid"),
                        "数量": job.get("count"),
                        "日志": job.get("log_path"),
                        "启动时间": job.get("started_at"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_log = None
            if batch_jobs:
                latest_log = Path(list(batch_jobs.values())[-1]["log_path"])
            elif jobs:
                latest_log = Path(list(jobs.values())[-1]["log_path"])
            if latest_log and latest_log.exists():
                lines = latest_log.read_text(encoding="utf-8", errors="replace").splitlines()
                st.code("\n".join(lines[-80:]), language="text")


def _render_final_eval_panel(selected: Dict[str, Any], experiments: List[Dict[str, Any]]) -> None:
    exp_dir = Path(selected["path"])
    st.subheader("最终复评")
    st.caption(
        "用归档实验的 best params 重新跑固定 episode 数；复评会复制原始 BKTree 到本次输出目录，在副本上补建新状态，并保存每局 `state_id_sequence` 与 `bktree/node_log.txt`。Synergy 默认复用归档的 `action_tuning_model.pkl`。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    episodes = c1.number_input(
        "每次复评 episode 数",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        key=f"final_eval_episodes_{selected['experiment_id']}",
    )
    repeats = c2.number_input(
        "重复次数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key=f"final_eval_repeats_{selected['experiment_id']}",
    )
    timeout_minutes = c3.number_input(
        "单次超时分钟",
        min_value=5,
        max_value=240,
        value=90,
        step=5,
        key=f"final_eval_timeout_{selected['experiment_id']}",
    )
    action_tuning_mode = c4.selectbox(
        "动作微调开关",
        ["自动", "强制启用", "强制关闭"],
        key=f"final_eval_tuning_{selected['experiment_id']}",
    )

    command = _build_final_eval_command(
        exp_dir,
        episodes=int(episodes),
        repeats=int(repeats),
        timeout_minutes=int(timeout_minutes),
        action_tuning_mode=action_tuning_mode,
    )
    with st.expander("复评命令（可直接复制到 PowerShell 执行）", expanded=False):
        _render_copyable_command(command, f"detail_{selected['experiment_id']}")

    if st.button("启动该实验最终复评", key=f"start_final_eval_{selected['experiment_id']}", use_container_width=True):
        try:
            job = _start_final_eval_job(
                exp_dir,
                int(episodes),
                int(repeats),
                int(timeout_minutes),
                action_tuning_mode,
            )
            st.session_state.setdefault("final_eval_jobs", {})[selected["experiment_id"]] = job
            st.success(f"已启动复评进程 PID={job['pid']}，日志：`{job['log_path']}`")
        except Exception as exc:
            st.error(f"启动复评失败：{exc}")

    job = st.session_state.get("final_eval_jobs", {}).get(selected["experiment_id"])
    if job:
        st.info(f"最近启动：PID={job['pid']}，输出目录 `{job['output_dir']}`")
        log_path = Path(job["log_path"])
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                with st.expander("复评日志尾部", expanded=False):
                    st.code("\n".join(lines[-80:]))
            except Exception:
                pass

    rows = _final_eval_rows(exp_dir)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("该实验尚无最终复评结果。")

    same_map_rows = []
    for item in experiments:
        if item.get("map_id") != selected.get("map_id"):
            continue
        latest = _latest_final_eval(Path(item["path"]))
        if not latest:
            continue
        same_map_rows.append(
            {
                "实验ID": item["experiment_id"],
                "方法组": item["method_group"],
                "地图": item["map_id"],
                "复评ID": Path(latest.get("_eval_dir", "")).name,
                "总回合": _nested_get(latest, "aggregate", "total_episodes", default="-"),
                "胜率": _metric_ci(latest, "win_rate"),
                "平均得分": _metric_ci(latest, "avg_score"),
                "稳定性": _metric_ci(latest, "stability"),
                "目标值": _metric_ci(latest, "objective"),
                "Eval BKTree": _format_eval_bktree_status(latest.get("_eval_dir", "")),
                "完成时间": latest.get("completed_at") or latest.get("updated_at") or "-",
            }
        )
    if same_map_rows:
        st.markdown("**同地图最新复评对比**")
        st.dataframe(pd.DataFrame(same_map_rows), use_container_width=True, hide_index=True)


def _render_batch_final_eval_panel(experiments: List[Dict[str, Any]]) -> None:
    st.subheader("批量最终复评")
    st.caption(
        "对多个归档实验串行执行 final eval；每个实验会写入各自目录下的 `final_eval/eval_<run_id>_<batch_tag>`，同一批任务共享同一个 batch 后缀，不会互相覆盖，也不会同时启动多个 SC2 客户端。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    episodes = c1.number_input(
        "批量复评 episode 数",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        key="batch_final_eval_episodes",
    )
    repeats = c2.number_input(
        "批量重复次数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="batch_final_eval_repeats",
    )
    timeout_minutes = c3.number_input(
        "批量单项超时分钟",
        min_value=5,
        max_value=240,
        value=90,
        step=5,
        key="batch_final_eval_timeout",
    )
    action_tuning_mode = c4.selectbox(
        "批量动作微调",
        ["自动", "强制启用", "强制关闭"],
        key="batch_final_eval_tuning",
    )

    o1, o2 = st.columns([1, 1])
    run_suffix_only = o1.toggle(
        "只显示带 run_XXXX 后缀实验",
        value=True,
        key="batch_final_eval_run_suffix_only",
    )
    skip_complete = o2.toggle(
        "跳过同规格已完整复评",
        value=True,
        key="batch_final_eval_skip_complete",
    )

    candidates = [
        item
        for item in experiments
        if item.get("method_group") in {"ETG-only", "synergy"}
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
        and (not run_suffix_only or _is_run_suffix_experiment(item))
    ]
    all_batch_candidates = [
        item
        for item in experiments
        if item.get("method_group") in {"ETG-only", "synergy"}
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
    ]
    st.caption(
        f"当前批量复评候选：{len(candidates)} / {len(all_batch_candidates)}；"
        + ("已启用 run_XXXX 后缀筛选。" if run_suffix_only else "未启用 run_XXXX 后缀筛选。")
    )
    if not candidates:
        st.info("当前筛选范围内没有可批量复评的 ETG-only / Synergy 归档实验。")
        return

    complete_by_id = {
        item["experiment_id"]: _has_complete_final_eval(
            Path(item["path"]),
            episodes=int(episodes),
            repeats=int(repeats),
        )
        for item in candidates
    }
    default_ids = [
        item["experiment_id"]
        for item in candidates
        if not (skip_complete and complete_by_id.get(item["experiment_id"], False))
    ]
    option_ids = [item["experiment_id"] for item in candidates]
    label_by_id = {
        item["experiment_id"]: (
            f"{item['method_group']} / {item['experiment_id']} | {item.get('map_key', '')} | "
            f"{item.get('map_id', '')}"
            + (" | 已有完整复评" if complete_by_id.get(item["experiment_id"], False) else "")
        )
        for item in candidates
    }
    selected_ids = st.multiselect(
        "选择要复评的归档实验",
        options=option_ids,
        default=default_ids,
        format_func=lambda exp_id: label_by_id.get(exp_id, exp_id),
        key=(
            "batch_final_eval_selected_ids_"
            f"run{int(bool(run_suffix_only))}_"
            f"skip{int(bool(skip_complete))}_"
            f"ep{int(episodes)}_rep{int(repeats)}"
        ),
    )

    status_rows = []
    by_id = {item["experiment_id"]: item for item in candidates}
    for exp_id in selected_ids:
        item = by_id[exp_id]
        status_rows.append(
            {
                "实验ID": exp_id,
                "方法组": item["method_group"],
                "地图": item.get("map_id", ""),
                "source_run": _nested_get(item, "manifest", "source_run", default="-"),
                "同规格完整复评": "是" if complete_by_id.get(exp_id, False) else "否",
                "将执行": "否" if skip_complete and complete_by_id.get(exp_id, False) else "是",
            }
        )
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    suggested_batch_tag = f"batch_manual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_batch_final_eval_command(
        selected_ids,
        episodes=int(episodes),
        repeats=int(repeats),
        timeout_minutes=int(timeout_minutes),
        action_tuning_mode=action_tuning_mode,
        run_suffix_only=bool(run_suffix_only),
        skip_complete=bool(skip_complete),
        batch_tag=suggested_batch_tag,
    )
    with st.expander("批量复评命令（可直接复制到 PowerShell 执行）", expanded=False):
        st.caption(f"复制命令将使用批次后缀 `{suggested_batch_tag}`；同一命令启动的所有 final eval 目录会共享该后缀。")
        _render_copyable_command(command, f"batch_final_eval_{len(selected_ids)}")

    if st.button(
        "启动批量最终复评",
        key="start_batch_final_eval",
        use_container_width=True,
        disabled=not selected_ids,
    ):
        try:
            job = _start_batch_final_eval_job(
                selected_ids,
                int(episodes),
                int(repeats),
                int(timeout_minutes),
                action_tuning_mode,
                bool(run_suffix_only),
                bool(skip_complete),
            )
            st.session_state.setdefault("batch_final_eval_jobs", {})[job["batch_id"]] = job
            st.success(f"已启动批量复评 PID={job['pid']}，实验数={job['count']}，日志：`{job['log_path']}`")
        except Exception as exc:
            st.error(f"启动批量复评失败：{exc}")

    jobs = st.session_state.get("batch_final_eval_jobs", {})
    if jobs:
        with st.expander("批量最终复评后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "实验数": job.get("count"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass


def _render_fixed_pool_variant_final_eval_panel(experiments: List[Dict[str, Any]]) -> None:
    st.subheader("固定参数池多参数组最终复评")
    st.caption(
        "该入口专用于固定参数池 Synergy 存档。它不会使用 study_summary.json 的整体 best params，"
        "而是读取每个存档的 fixed_pool_source_variants.json，并按参数组分别执行 final eval。"
        "episode 数按每个参数组单独计算。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    episodes_per_variant = c1.number_input(
        "每参数组 episode 数",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        key="fixed_pool_variant_eval_episodes_per_variant",
        help="例如 6 个场景 × 8 个参数组 × 100 局，共 4800 局。",
    )
    repeats = c2.number_input(
        "每参数组重复次数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="fixed_pool_variant_eval_repeats",
    )
    timeout_minutes = c3.number_input(
        "单参数组超时分钟",
        min_value=5,
        max_value=300,
        value=90,
        step=5,
        key="fixed_pool_variant_eval_timeout",
    )
    variants_per_exp = c4.number_input(
        "每实验参数组数",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
        key="fixed_pool_variant_eval_variants_per_exp",
    )

    o1, o2 = st.columns([1, 1])
    run_suffix_only = o1.toggle(
        "只显示带 run_XXXX 后缀实验",
        value=True,
        key="fixed_pool_variant_eval_run_suffix_only",
    )
    skip_complete = o2.toggle(
        "跳过已完成参数组复评",
        value=True,
        key="fixed_pool_variant_eval_skip_complete",
    )

    candidates = [
        item
        for item in experiments
        if item.get("method_group") == "synergy"
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
        and (not run_suffix_only or _is_run_suffix_experiment(item))
        and (Path(item["path"]) / "fixed_pool_source_variants.json").exists()
    ]
    all_candidates = [
        item
        for item in experiments
        if item.get("method_group") == "synergy"
        and (Path(item["path"]) / "fixed_pool_source_variants.json").exists()
    ]
    st.caption(
        f"当前固定参数池 Synergy 候选：{len(candidates)} / {len(all_candidates)}；"
        f"预计 final eval 作业数为 {len(candidates) * int(variants_per_exp)}。"
    )
    if not candidates:
        st.info("当前筛选范围内没有包含 fixed_pool_source_variants.json 的 Synergy 存档。")
        return

    option_ids = [item["experiment_id"] for item in candidates]
    label_by_id = {}
    variant_count_by_id = {}
    for item in candidates:
        count = _fixed_pool_variant_count(Path(item["path"]))
        variant_count_by_id[item["experiment_id"]] = count
        label_by_id[item["experiment_id"]] = (
            f"synergy / {item['experiment_id']} | {item.get('map_key', '')} | "
            f"{item.get('map_id', '')} | 参数组 {count}"
        )
    selected_ids = st.multiselect(
        "选择要按参数组复评的 Synergy 存档",
        options=option_ids,
        default=option_ids,
        format_func=lambda exp_id: label_by_id.get(exp_id, exp_id),
        key=(
            "fixed_pool_variant_eval_selected_ids_"
            f"run{int(bool(run_suffix_only))}_"
            f"ep{int(episodes_per_variant)}_rep{int(repeats)}_var{int(variants_per_exp)}"
        ),
    )

    status_rows = []
    by_id = {item["experiment_id"]: item for item in candidates}
    for exp_id in selected_ids:
        item = by_id[exp_id]
        available_variants = int(variant_count_by_id.get(exp_id, 0))
        used_variants = min(int(variants_per_exp), available_variants) if int(variants_per_exp) > 0 else available_variants
        status_rows.append(
            {
                "实验ID": exp_id,
                "地图": item.get("map_key", ""),
                "可用参数组": available_variants,
                "将复评参数组": used_variants,
                "每参数组episode": int(episodes_per_variant),
                "预计总局数": used_variants * int(episodes_per_variant) * int(repeats),
            }
        )
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    suggested_batch_tag = f"web_fpv_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_fixed_pool_variant_final_eval_command(
        selected_ids,
        episodes_per_variant=int(episodes_per_variant),
        repeats=int(repeats),
        timeout_minutes=int(timeout_minutes),
        variants_per_exp=int(variants_per_exp),
        skip_complete=bool(skip_complete),
        batch_tag=suggested_batch_tag,
    )
    with st.expander("固定参数池多参数组复评命令（可直接复制到 PowerShell 执行）", expanded=False):
        st.caption(
            "输出目录形如 `<synergy_run>/final_eval/<batch_id>/variant_rankXX_<name>/`，"
            "每个参数组会生成独立 final eval。"
        )
        _render_copyable_command(command, f"fixed_pool_variant_eval_{len(selected_ids)}")

    if st.button(
        "启动固定参数池多参数组最终复评",
        key="start_fixed_pool_variant_final_eval",
        use_container_width=True,
        disabled=not selected_ids,
    ):
        try:
            job = _start_fixed_pool_variant_final_eval_job(
                selected_ids,
                int(episodes_per_variant),
                int(repeats),
                int(timeout_minutes),
                int(variants_per_exp),
                bool(skip_complete),
            )
            st.session_state.setdefault("fixed_pool_variant_final_eval_jobs", {})[job["batch_id"]] = job
            st.success(
                f"已启动固定参数池多参数组复评 PID={job['pid']}，实验数={job['count']}，"
                f"每实验参数组={job['variant_count']}，预计作业数={job['job_count']}，"
                f"预计总局数={job['total_episodes']}，日志：`{job['log_path']}`"
            )
        except Exception as exc:
            st.error(f"启动固定参数池多参数组复评失败：{exc}")

    jobs = st.session_state.get("fixed_pool_variant_final_eval_jobs", {})
    if jobs:
        with st.expander("固定参数池多参数组复评后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "实验数": job.get("count"),
                    "每实验参数组": job.get("variant_count"),
                    "预计作业数": job.get("job_count"),
                    "预计总局数": job.get("total_episodes"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass


def _render_switch_grid_eval_panel(experiments: List[Dict[str, Any]]) -> None:
    st.subheader("批量路径切换对照复评")
    st.caption(
        "用于证明“不开启路径切换更差”的成对机制实验。每个归档实验会串行运行默认 5 组 final eval："
        "`single_step_no_switch`、`multi_step_no_switch`、`multi_step_switch_exact`、"
        "`multi_step_switch_fuzzy020` 和 `multi_step_switch_fuzzy050`。"
        "输出目录会带有同一 switch-grid 批次后缀和 variant 名，便于后续做同批比较。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    episodes = c1.number_input(
        "切换对照 episode 数",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        key="switch_grid_eval_episodes",
    )
    repeats = c2.number_input(
        "切换对照重复次数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="switch_grid_eval_repeats",
    )
    timeout_minutes = c3.number_input(
        "切换对照单项超时分钟",
        min_value=5,
        max_value=300,
        value=120,
        step=5,
        key="switch_grid_eval_timeout",
    )
    action_tuning_mode = c4.selectbox(
        "切换对照动作微调",
        ["自动", "强制启用", "强制关闭"],
        key="switch_grid_eval_tuning",
    )

    o1, o2 = st.columns([1, 1])
    run_suffix_only = o1.toggle(
        "只显示带 run_XXXX 后缀实验",
        value=True,
        key="switch_grid_eval_run_suffix_only",
    )
    method_groups = o2.multiselect(
        "方法组",
        options=["synergy", "ETG-only"],
        default=["synergy"],
        key="switch_grid_eval_method_groups",
    )
    if not method_groups:
        st.warning("至少选择一个方法组后才能启动路径切换对照复评。")
        return

    candidates = [
        item
        for item in experiments
        if item.get("method_group") in set(method_groups)
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
        and (not run_suffix_only or _is_run_suffix_experiment(item))
    ]
    all_switch_candidates = [
        item
        for item in experiments
        if item.get("method_group") in {"ETG-only", "synergy"}
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
    ]
    st.caption(
        f"当前切换对照候选：{len(candidates)} / {len(all_switch_candidates)}；"
        f"每个实验默认会启动 5 个 variant，因此预计 final eval 作业数为 {len(candidates) * 5}。"
    )
    if not candidates:
        st.info("当前筛选范围内没有可运行路径切换对照的 ETG-only / Synergy 归档实验。")
        return

    option_ids = [item["experiment_id"] for item in candidates]
    label_by_id = {
        item["experiment_id"]: (
            f"{item['method_group']} / {item['experiment_id']} | {item.get('map_key', '')} | "
            f"{item.get('map_id', '')}"
        )
        for item in candidates
    }
    selected_ids = st.multiselect(
        "选择要运行切换对照的归档实验",
        options=option_ids,
        default=option_ids,
        format_func=lambda exp_id: label_by_id.get(exp_id, exp_id),
        key=(
            "switch_grid_eval_selected_ids_"
            f"run{int(bool(run_suffix_only))}_"
            f"methods{'_'.join(method_groups)}_"
            f"ep{int(episodes)}_rep{int(repeats)}"
        ),
    )

    by_id = {item["experiment_id"]: item for item in candidates}
    status_rows = []
    for exp_id in selected_ids:
        item = by_id[exp_id]
        status_rows.append(
            {
                "实验ID": exp_id,
                "方法组": item["method_group"],
                "地图": item.get("map_id", ""),
                "source_run": _nested_get(item, "manifest", "source_run", default="-"),
                "variant数": 5,
                "预计final eval目录数": 5,
                "预计总局数": 5 * int(episodes) * int(repeats),
            }
        )
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    suggested_batch_tag = f"web_switch_grid_manual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_switch_grid_eval_command(
        selected_ids,
        method_groups=method_groups,
        episodes=int(episodes),
        repeats=int(repeats),
        timeout_minutes=int(timeout_minutes),
        action_tuning_mode=action_tuning_mode,
        run_suffix_only=bool(run_suffix_only),
        batch_tag=suggested_batch_tag,
    )
    with st.expander("路径切换对照命令（可直接复制到 PowerShell 执行）", expanded=False):
        st.caption(
            f"复制命令将使用批次 `{suggested_batch_tag}`；"
            "每个实验会生成带 variant 后缀的独立 final eval 目录。"
        )
        _render_copyable_command(command, f"switch_grid_eval_{len(selected_ids)}")

    if st.button(
        "启动批量路径切换对照复评",
        key="start_switch_grid_eval",
        use_container_width=True,
        disabled=not selected_ids,
    ):
        try:
            job = _start_switch_grid_eval_job(
                selected_ids,
                method_groups,
                int(episodes),
                int(repeats),
                int(timeout_minutes),
                action_tuning_mode,
                bool(run_suffix_only),
            )
            st.session_state.setdefault("switch_grid_eval_jobs", {})[job["batch_id"]] = job
            st.success(
                f"已启动路径切换对照 PID={job['pid']}，实验数={job['count']}，"
                f"variant数={job['variant_count']}，预计作业数={job['job_count']}，日志：`{job['log_path']}`"
            )
        except Exception as exc:
            st.error(f"启动路径切换对照失败：{exc}")

    jobs = st.session_state.get("switch_grid_eval_jobs", {})
    if jobs:
        with st.expander("路径切换对照后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "实验数": job.get("count"),
                    "variant数": job.get("variant_count"),
                    "预计作业数": job.get("job_count"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass
            analysis_cmd = [
                "python",
                str(Path("scripts") / "analyze_switch_grid_eval.py"),
                "--batch-id",
                str(latest_job.get("batch_id", "")),
                "--export-raw-events",
            ]
            st.markdown("**路径切换对照统计命令**")
            st.caption("批次完成后运行该命令，会生成 variant 汇总、逐 episode 汇总、相对 no-switch 的对照表和原始事件表。")
            _render_copyable_command(
                " ".join(_quote_cli_arg(part) for part in analysis_cmd),
                f"switch_grid_analysis_{latest_job.get('batch_id', '')}",
            )


def _render_fixed_pool_switch_ablation_panel(experiments: List[Dict[str, Any]]) -> None:
    st.subheader("高质量参数池 switch-grid 成对复评")
    st.caption(
        "用于专门验证 `backup_enabled` / Switch-Aware backup 机制。"
        "该模块不会使用归档实验的单一 best params，而是读取同场景 `fixed_pool_source_variants.json` 中的高质量参数池；"
        "每个参数组固定 beam、lookahead、score mode、action strategy 等底座参数，只改变 backup 开关和阈值。"
        "输出按参数组分层保存为 `fixed_pool_switch_pair_* / rankXX_trialXXXX / ms_*`，不会覆盖现有 final eval。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    episodes = c1.number_input(
        "每个配置 episode 数",
        min_value=1,
        max_value=1000,
        value=50,
        step=10,
        key="fixed_pool_switch_eval_episodes",
    )
    repeats = c2.number_input(
        "重复次数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="fixed_pool_switch_eval_repeats",
    )
    variants_per_exp = c3.number_input(
        "每实验参数组数",
        min_value=1,
        max_value=32,
        value=8,
        step=1,
        key="fixed_pool_switch_eval_variants",
    )
    timeout_minutes = c4.number_input(
        "单项超时分钟",
        min_value=5,
        max_value=300,
        value=120,
        step=5,
        key="fixed_pool_switch_eval_timeout",
    )

    o1, o2, o3 = st.columns([1, 1, 1])
    run_suffix_only = o1.toggle(
        "只显示带 run_XXXX 后缀实验",
        value=True,
        key="fixed_pool_switch_eval_run_suffix_only",
    )
    method_groups = o2.multiselect(
        "方法组",
        options=["ETG-only", "synergy"],
        default=["ETG-only"],
        key="fixed_pool_switch_eval_method_groups",
    )
    action_tuning_mode = o3.selectbox(
        "动作微调",
        ["强制关闭", "自动", "强制启用"],
        key="fixed_pool_switch_eval_tuning",
    )
    if not method_groups:
        st.warning("至少选择一个方法组后才能启动高质量参数池 switch-grid。")
        return

    candidates = [
        item
        for item in experiments
        if item.get("method_group") in set(method_groups)
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
        and (not run_suffix_only or _is_run_suffix_experiment(item))
    ]
    if not candidates:
        st.info("当前筛选范围内没有可运行的 ETG-only / Synergy 归档实验。")
        return

    option_ids = [item["experiment_id"] for item in candidates]
    label_by_id = {
        item["experiment_id"]: (
            f"{item['method_group']} / {item['experiment_id']} | {item.get('map_key', '')} | "
            f"{item.get('map_id', '')}"
        )
        for item in candidates
    }
    selected_ids = st.multiselect(
        "选择要运行高质量参数池 switch-grid 的归档实验",
        options=option_ids,
        default=option_ids,
        format_func=lambda exp_id: label_by_id.get(exp_id, exp_id),
        key=(
            "fixed_pool_switch_eval_selected_ids_"
            f"run{int(bool(run_suffix_only))}_"
            f"methods{'_'.join(method_groups)}_"
            f"ep{int(episodes)}_rep{int(repeats)}_var{int(variants_per_exp)}"
        ),
    )

    by_id = {item["experiment_id"]: item for item in candidates}
    status_rows = []
    for exp_id in selected_ids:
        item = by_id[exp_id]
        status_rows.append(
            {
                "实验ID": exp_id,
                "方法组": item["method_group"],
                "地图": item.get("map_key", ""),
                "参数组数": int(variants_per_exp),
                "每参数组switch配置": 4,
                "预计final eval目录数": int(variants_per_exp) * 4,
                "预计总局数": int(variants_per_exp) * 4 * int(episodes) * int(repeats),
            }
        )
    if status_rows:
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    suggested_batch_tag = f"web_fixed_pool_switch_pair_manual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_fixed_pool_switch_ablation_command(
        selected_ids,
        method_groups=method_groups,
        episodes=int(episodes),
        repeats=int(repeats),
        timeout_minutes=int(timeout_minutes),
        action_tuning_mode=action_tuning_mode,
        run_suffix_only=bool(run_suffix_only),
        variants_per_exp=int(variants_per_exp),
        batch_tag=suggested_batch_tag,
    )
    with st.expander("高质量参数池 switch-grid 命令（可直接复制到 PowerShell 执行）", expanded=False):
        st.caption(
            f"复制命令将使用批次 `{suggested_batch_tag}`；"
            "每个参数组会生成 `ms_ns`、`ms_ex`、`ms_fz020`、`ms_fz050` 四个子目录。"
        )
        _render_copyable_command(command, f"fixed_pool_switch_eval_{len(selected_ids)}")

    if st.button(
        "启动高质量参数池 switch-grid 成对复评",
        key="start_fixed_pool_switch_eval",
        use_container_width=True,
        disabled=not selected_ids,
    ):
        try:
            job = _start_fixed_pool_switch_ablation_job(
                selected_ids,
                method_groups,
                int(episodes),
                int(repeats),
                int(timeout_minutes),
                action_tuning_mode,
                bool(run_suffix_only),
                int(variants_per_exp),
            )
            st.session_state.setdefault("fixed_pool_switch_eval_jobs", {})[job["batch_id"]] = job
            st.success(
                f"已启动高质量参数池 switch-grid PID={job['pid']}，实验数={job['count']}，"
                f"参数组/实验={job['base_variant_count']}，预计作业数={job['job_count']}，"
                f"预计总局数={job['total_episodes']}，日志：`{job['log_path']}`"
            )
        except Exception as exc:
            st.error(f"启动高质量参数池 switch-grid 失败：{exc}")

    jobs = st.session_state.get("fixed_pool_switch_eval_jobs", {})
    if jobs:
        with st.expander("高质量参数池 switch-grid 后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "实验数": job.get("count"),
                    "参数组数": job.get("base_variant_count"),
                    "预计作业数": job.get("job_count"),
                    "预计总局数": job.get("total_episodes"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass
            analysis_cmd = [
                "python",
                str(Path("scripts") / "analyze_switch_grid_eval.py"),
                "--batch-id",
                str(latest_job.get("batch_id", "")),
                "--export-raw-events",
            ]
            st.markdown("**高质量参数池 switch-grid 统计命令**")
            _render_copyable_command(
                " ".join(_quote_cli_arg(part) for part in analysis_cmd),
                f"fixed_pool_switch_analysis_{latest_job.get('batch_id', '')}",
            )


def _diagnostic_label(item: Dict[str, Any]) -> str:
    return f"{item['method_group']} / {item['experiment_id']} | {item['map_key']} | {item['map_id']}"


def _render_mechanism_diagnostics_panel(
    experiments: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]] = None,
) -> None:
    st.subheader("机制诊断")
    st.caption(
        "该面板离线统计机制指标：ETG-only/Synergy 默认读取归档实验的最优 trial（对应总览中的最优目标值、胜率和最终得分），Replay-baseline 读取回放评估结果。这里不把随机性理解为外部突变注入，而是观察随机状态转移下的状态解析、动作来源切换、规划置信度与不利转移后的恢复能力。"
    )
    if not experiments:
        st.info("当前筛选下没有可诊断的归档实验。")
        return

    options = list(range(len(experiments)))
    default_indices = []
    if selected:
        for idx, item in enumerate(experiments):
            if item.get("path") == selected.get("path"):
                default_indices = [idx]
                break
    if not default_indices:
        default_indices = options[: min(6, len(options))]

    selected_indices = st.multiselect(
        "诊断对象（可多选）",
        options=options,
        default=default_indices,
        format_func=lambda idx: _diagnostic_label(experiments[idx]),
        key="mechanism_diagnostics_selection",
    )
    if not selected_indices:
        st.info("请选择至少一个实验。")
        return

    with st.expander("Help：机制诊断指标说明", expanded=False):
        st.markdown(
            """
- `状态访问熵`：对一次评估中访问到的状态标识分布计算 Shannon entropy，ETG/Synergy 优先使用复评 BKTree 的 `eval_state_id`，没有时回退到 `nid` 或 `state_cluster`。数值越高，说明访问更分散；数值越低，说明轨迹集中在少数状态或循环上。
- `计算方式`：设访问到的状态集合为 $S$，状态 $s$ 被访问次数为 $n_s$，总步数为 $N$，则 $p_s=n_s/N$，`状态访问熵` 为 $H=-\\sum_{s\\in S}p_s\\log_2 p_s$。
- `状态熵占比`：$H/\\log_2|S|$，用于消除不同实验唯一状态数不同带来的尺度影响；越接近 100%，表示访问越接近均匀覆盖，越低表示少数状态被反复访问。
- `诊断数据源`：ETG-only/Synergy 使用 `study_summary.json` 中 `best_trial` 对应的 `trials/trial_xxxx/episodes.jsonl`；只有找不到该文件时才回退到最新 `final_eval`。
- `非主ETG动作`：动作来源不是 `kg_plan/kg_follow` 的比例，用来观察是否依赖 relaxed、OOD、fallback、tuning 或 diverge 通道。
- `不利转移暴露`：以 `hp_delta < -12` 作为自然随机性被决策放大的代理事件；`不利后恢复` 统计随后 3 步内血量优势是否恢复到事件发生时水平。
            """.strip()
        )

    rows = []
    distributions: List[Dict[str, Any]] = []
    for idx in selected_indices:
        item = experiments[int(idx)]
        diag = _compute_mechanism_diagnostics(
            item["path"],
            item["method_group"],
            float(item.get("modified_at", 0)),
        )
        if not diag.get("available"):
            rows.append(
                {
                    "实验ID": item["experiment_id"],
                    "方法组": item["method_group"],
                    "地图": item["map_id"],
                    "状态": diag.get("reason", "unavailable"),
                }
            )
            continue
        rows.append(
            {
                "实验ID": item["experiment_id"],
                "方法组": item["method_group"],
                "地图": item["map_id"],
                "诊断数据源": diag.get("scope_label", "-"),
                "episodes": diag["episodes"],
                "steps": diag["steps"],
                "胜率": _fmt_pct(diag["win_rate"]),
                "Dogfall": _fmt_pct(diag["dogfall_rate"]),
                "平均得分": _fmt(diag["avg_score"]),
                "唯一NID": diag["unique_nids"] or "-",
                "唯一状态簇": diag["unique_state_clusters"] or "-",
                "状态访问熵": _fmt(diag["state_visit_entropy"]),
                "状态熵占比": _fmt_pct(diag["state_visit_entropy_norm"]),
                "动作熵": _fmt(diag["action_entropy"]),
                "Exact解析": _fmt_pct(diag["exact_ratio"]),
                "OOD/拒绝解析": _fmt_pct(diag["ood_ratio"]),
                "非主ETG动作": _fmt_pct(diag["non_primary_action_ratio"]),
                "显式探索动作": _fmt_pct(diag["exploration_ratio"]),
                "无动作/序列耗尽": _fmt_pct(diag["no_action_ratio"]),
                "有规划记录": _fmt_pct(diag["plan_ratio"]),
                "高置信路径": _fmt_pct(diag["high_conf_path_ratio"]),
                "Tuning机会": _fmt_pct(diag["tuning_opportunity_ratio"]),
                "Tuning验证": _fmt_pct(diag["tuning_validation_ratio"]),
                "Tuning替换": _fmt_pct(diag["tuning_replacement_ratio"]),
                "候选动作达标": _fmt_pct(diag["candidate_eligible_ratio"]),
                "平均Tuning置信度": _fmt(diag["avg_tuning_confidence"]),
                "不利转移暴露": _fmt_pct(diag["adverse_event_ratio"]),
                "不利后恢复": _fmt_pct(diag["adverse_recovery_ratio"]),
                "不利局坏结果": _fmt_pct(diag["adverse_bad_outcome_ratio"]),
                "数据文件": diag["path"],
            }
        )
        distributions.append(
            {
                "实验ID": item["experiment_id"],
                "action_source": diag.get("action_sources", {}),
                "nid_status": diag.get("nid_statuses", {}),
                "plan_trigger": diag.get("plan_triggers", {}),
                "tuning_source": diag.get("tuning_sources", {}),
                "result": diag.get("result_counts", {}),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "`不利转移暴露` 使用 `hp_delta < -12` 作为自然随机性被放大的代理事件；Replay-baseline 重点看 `无动作/序列耗尽`、动作熵和状态簇覆盖，通常不具备 ETG 解析与规划字段。"
    )

    if distributions:
        with st.expander("机制分布明细", expanded=False):
            dist_rows = []
            for dist in distributions:
                for category in ("action_source", "nid_status", "plan_trigger", "tuning_source", "result"):
                    values = dist.get(category, {})
                    if not isinstance(values, dict):
                        continue
                    total = sum(int(v) for v in values.values()) if values else 0
                    for key, value in sorted(values.items(), key=lambda kv: int(kv[1]), reverse=True):
                        dist_rows.append(
                            {
                                "实验ID": dist["实验ID"],
                                "类别": category,
                                "取值": key,
                                "次数": value,
                                "占比": _fmt_pct(_pct(float(value), float(total))),
                            }
                        )
            if dist_rows:
                st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)
            else:
                st.info("所选实验没有可展开的机制分布字段。")


@st.cache_data(ttl=60, show_spinner=False)
def _collect_common_bktree_visits(
    exp_path: str,
    method_group: str,
    experiment_id: str,
    bktree_path: str,
    primary_threshold: float,
    secondary_threshold: float,
    modified_at: float,
    max_steps: int,
) -> Dict[str, Any]:
    exp_dir = Path(exp_path)
    source_info = _diagnostic_episode_jsonl_candidates(exp_dir, method_group)
    episode_files = source_info["files"]
    if not episode_files:
        return {"available": False, "reason": "no episodes.jsonl found"}

    visits: Counter = Counter()
    primary_visits: Counter = Counter()
    source_counter: Counter = Counter()
    result_counter: Counter = Counter()
    rejected_count = 0
    unresolved_count = 0
    steps = 0
    episodes = 0
    norm_query_count = 0
    recorded_cluster_count = 0
    query_cache: Dict[str, Dict[str, Any]] = {}

    for episode_file in episode_files:
        with open(str(episode_file), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if max_steps and steps >= max_steps:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episodes += 1
                result_counter[str(episode.get("result", "-"))] += 1
                frames = episode.get("frames") or episode.get("steps") or []
                if not isinstance(frames, list):
                    continue
                for frame in frames:
                    if max_steps and steps >= max_steps:
                        break
                    if not isinstance(frame, dict):
                        continue
                    steps += 1
                    if frame.get("action_source"):
                        source_counter[str(frame.get("action_source"))] += 1

                    cluster_key = None
                    primary_id = None
                    if isinstance(frame.get("norm_state"), dict):
                        norm_key = json.dumps(frame["norm_state"], sort_keys=True, separators=(",", ":"))
                        if norm_key not in query_cache:
                            query_cache[norm_key] = _query_common_bktree_state(
                                frame["norm_state"],
                                bktree_path,
                                primary_threshold,
                                secondary_threshold,
                            )
                        resolved = query_cache[norm_key]
                        norm_query_count += 1
                        if resolved.get("rejected"):
                            rejected_count += 1
                        cluster_key = resolved.get("cluster_key")
                        primary_id = resolved.get("primary_id")
                    else:
                        state_cluster = frame.get("state_cluster")
                        if isinstance(state_cluster, list) and len(state_cluster) >= 2:
                            try:
                                primary_id = int(state_cluster[0])
                                cluster_key = f"{primary_id}:{int(state_cluster[1])}"
                                recorded_cluster_count += 1
                            except (TypeError, ValueError):
                                cluster_key = None

                    if cluster_key is None:
                        unresolved_count += 1
                        continue
                    visits[cluster_key] += 1
                    if primary_id is not None:
                        primary_visits[str(primary_id)] += 1

    total = sum(visits.values())
    entropy = _entropy(visits)
    entropy_norm = entropy / math.log2(len(visits)) if entropy is not None and len(visits) > 1 else None
    top_count = visits.most_common(1)[0][1] if visits else 0
    return {
        "available": True,
        "experiment_id": experiment_id,
        "method_group": method_group,
        "scope": source_info.get("scope"),
        "scope_label": source_info.get("scope_label"),
        "bktree_path": bktree_path,
        "episodes": episodes,
        "steps": steps,
        "resolved_steps": total,
        "unique_clusters": len(visits),
        "unique_primary": len(primary_visits),
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "top_cluster_ratio": _pct(top_count, total),
        "rejected_ratio": _pct(rejected_count, norm_query_count),
        "unresolved_ratio": _pct(unresolved_count, steps),
        "norm_query_steps": norm_query_count,
        "recorded_cluster_steps": recorded_cluster_count,
        "visits": dict(visits),
        "primary_visits": dict(primary_visits),
        "action_sources": dict(source_counter),
        "result_counts": dict(result_counter),
        "modified_at": modified_at,
    }


def _weighted_overlap(left: Dict[str, int], right: Dict[str, int]) -> Optional[float]:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        return None
    keys = set(left) | set(right)
    return sum(min(left.get(key, 0) / left_total, right.get(key, 0) / right_total) for key in keys)


def _render_common_bktree_space_panel(
    experiments: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]] = None,
) -> None:
    st.subheader("共同BKTree状态空间分析")
    st.caption(
        "该模块只读加载各实验 manifest 中记录的 BKTree，并把同一地图下的 ETG-only、Synergy 与 Replay-baseline 映射到同一组 `(primary, secondary)` 聚类键上比较。Replay-baseline 若没有 `nid`，会使用逐帧 `norm_state` 查询同一 BKTree；不会改写任何 BKTree 文件。"
    )
    if not experiments:
        st.info("当前筛选下没有可分析的实验。")
        return

    map_ids = sorted({str(item.get("map_id", "-")) for item in experiments})
    default_map_index = 0
    if selected and selected.get("map_id") in map_ids:
        default_map_index = map_ids.index(selected["map_id"])
    chosen_map = st.selectbox(
        "选择共同BKTree地图",
        options=map_ids,
        index=default_map_index,
        key="common_bktree_map_select",
    )
    map_experiments = [item for item in experiments if str(item.get("map_id", "-")) == chosen_map]
    if not map_experiments:
        st.info("该地图下没有归档实验。")
        return

    options = list(range(len(map_experiments)))
    default_indices = options
    selected_indices = st.multiselect(
        "选择参与共同空间比较的实验",
        options=options,
        default=default_indices,
        format_func=lambda idx: _diagnostic_label(map_experiments[idx]),
        key=f"common_bktree_experiments_{chosen_map}",
    )
    max_steps = st.number_input(
        "每个实验最多解析步数",
        min_value=100,
        max_value=50000,
        value=10000,
        step=500,
        key=f"common_bktree_max_steps_{chosen_map}",
        help="Replay-baseline 需要逐帧查询 BKTree，大地图上建议先限制步数；该限制只影响分析速度，不修改原始数据。",
    )
    if not selected_indices:
        st.info("请选择至少一个实验。")
        return

    selected_items = [map_experiments[int(idx)] for idx in selected_indices]
    bktree_paths = {str(item.get("bktree_path", "-")) for item in selected_items}
    if len(bktree_paths) > 1:
        st.warning(
            "所选实验记录的 BKTree 路径不完全一致。仍可显示，但严格共同空间比较应优先选择同一 `data_dir/bktree` 下的实验。"
        )
    st.caption("BKTree路径：" + "；".join(sorted(bktree_paths)))

    analyses = []
    for item in selected_items:
        try:
            primary_threshold = float(item.get("bktree_primary", 1.0))
        except (TypeError, ValueError):
            primary_threshold = 1.0
        try:
            secondary_threshold = float(item.get("bktree_secondary", 0.5))
        except (TypeError, ValueError):
            secondary_threshold = 0.5
        analysis = _collect_common_bktree_visits(
            item["path"],
            item["method_group"],
            item["experiment_id"],
            str(item.get("bktree_path", "")),
            primary_threshold,
            secondary_threshold,
            float(item.get("modified_at", 0)),
            int(max_steps),
        )
        analysis["map_id"] = item.get("map_id", "-")
        analyses.append(analysis)

    summary_rows = []
    available = []
    for analysis in analyses:
        if not analysis.get("available"):
            summary_rows.append(
                {
                    "实验ID": analysis.get("experiment_id", "-"),
                    "状态": analysis.get("reason", "unavailable"),
                }
            )
            continue
        available.append(analysis)
        summary_rows.append(
            {
                "实验ID": analysis["experiment_id"],
                "方法组": analysis["method_group"],
                "诊断数据源": analysis["scope_label"],
                "episodes": analysis["episodes"],
                "steps": analysis["steps"],
                "解析步数": analysis["resolved_steps"],
                "唯一BKTree簇": analysis["unique_clusters"],
                "唯一Primary簇": analysis["unique_primary"],
                "状态访问熵": _fmt(analysis["entropy"]),
                "状态熵占比": _fmt_pct(analysis["entropy_norm"]),
                "Top簇占比": _fmt_pct(analysis["top_cluster_ratio"]),
                "Replay查询拒绝": _fmt_pct(analysis["rejected_ratio"]),
                "未解析": _fmt_pct(analysis["unresolved_ratio"]),
                "数据来源": "norm_state查询" if analysis["norm_query_steps"] else "记录state_cluster",
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if len(available) >= 2:
        ids = [item["experiment_id"] for item in available]
        jaccard_z = []
        overlap_z = []
        for left in available:
            j_row = []
            o_row = []
            left_set = set(left["visits"].keys())
            for right in available:
                right_set = set(right["visits"].keys())
                union = left_set | right_set
                j_row.append((len(left_set & right_set) / len(union)) if union else 0.0)
                o_row.append(_weighted_overlap(left["visits"], right["visits"]) or 0.0)
            jaccard_z.append(j_row)
            overlap_z.append(o_row)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(data=go.Heatmap(z=jaccard_z, x=ids, y=ids, colorscale="Blues", zmin=0, zmax=1))
            fig.update_layout(title="唯一BKTree簇 Jaccard 重叠", height=360)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(data=go.Heatmap(z=overlap_z, x=ids, y=ids, colorscale="Greens", zmin=0, zmax=1))
            fig.update_layout(title="访问分布重叠系数", height=360)
            st.plotly_chart(fig, use_container_width=True)

    if available:
        total_counter: Counter = Counter()
        for analysis in available:
            total_counter.update({key: int(value) for key, value in analysis["visits"].items()})
        top_keys = [key for key, _ in total_counter.most_common(30)]
        top_rows = []
        for key in top_keys:
            row = {"BKTree簇": key, "总访问": total_counter[key]}
            for analysis in available:
                row[analysis["experiment_id"]] = analysis["visits"].get(key, 0)
            top_rows.append(row)
        if top_rows:
            st.markdown("**Top BKTree簇访问对比**")
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

        scatter_rows = []
        for analysis in available:
            for key, count in Counter(analysis["visits"]).most_common(300):
                try:
                    primary, secondary = key.split(":", 1)
                    scatter_rows.append(
                        {
                            "experiment_id": analysis["experiment_id"],
                            "primary": int(primary),
                            "secondary": int(secondary),
                            "count": int(count),
                            "cluster": key,
                        }
                    )
                except ValueError:
                    continue
        if scatter_rows:
            scatter_df = pd.DataFrame(scatter_rows)
            fig = go.Figure()
            for exp_id, group in scatter_df.groupby("experiment_id"):
                fig.add_trace(
                    go.Scattergl(
                        x=group["primary"],
                        y=group["secondary"],
                        mode="markers",
                        name=exp_id,
                        marker={
                            "size": (group["count"].clip(upper=50) + 3),
                            "opacity": 0.55,
                        },
                        text=group["cluster"],
                        hovertemplate="cluster=%{text}<br>primary=%{x}<br>secondary=%{y}<extra>" + exp_id + "</extra>",
                    )
                )
            fig.update_layout(
                title="共同BKTree聚类访问散点（x=primary, y=secondary, size=访问次数）",
                xaxis_title="Primary cluster",
                yaxis_title="Secondary cluster",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_multistep_sensitivity_panel(experiments: List[Dict[str, Any]]) -> None:
    st.subheader("批量 multi-step 规划参数寻优")
    st.caption(
        "该阶段不是 final eval，也不是固定参数复评。它会为所选归档 ETG 实验启动新的 Optuna 参数寻优，"
        "强制 `mode=multi_step`，关闭动作微调和 Synergy 分阶段协同，只搜索纯图规划参数。"
        "寻优结束后会自动导出多目标高质量且参数多样的参数池，供后续固定参数池 Synergy 训练使用。"
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    trials = c1.number_input(
        "Optuna trials",
        min_value=1,
        max_value=2000,
        value=60,
        step=5,
        key="multistep_param_search_trials",
    )
    episodes_per_trial = c2.number_input(
        "每个 trial 对局数",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        key="multistep_param_search_episodes_per_trial",
    )
    timeout_minutes = c3.number_input(
        "单 trial 超时分钟",
        min_value=5,
        max_value=300,
        value=120,
        step=5,
        key="multistep_param_search_timeout",
    )

    p1, p2, p3 = st.columns([1, 1, 1])
    beam_width_range = p1.text_input(
        "beam_width 搜索范围",
        value="2,10",
        key="multistep_param_search_beam_range",
        help="输入最小值和最大值，例如 2,10。Optuna 会在区间内自动搜索，不再枚举固定网格。",
    )
    lookahead_range = p2.text_input(
        "lookahead_steps 搜索范围",
        value="2,15",
        key="multistep_param_search_lookahead_range",
    )
    backup_mode = p3.selectbox(
        "备选路径开关搜索",
        ["both", "on", "off"],
        index=0,
        key="multistep_param_search_backup_mode",
        help="both 表示让 Optuna 同时搜索启用和禁用备选路径。",
    )

    q1, q2 = st.columns([1, 2])
    score_modes = q1.text_input(
        "score modes",
        value="quality,future_reward,win_rate",
        key="multistep_param_search_score_modes",
    )
    action_strategies = q2.text_input(
        "action strategies",
        value="highest_transition_prob,best_subtree_quality,best_subtree_winrate,random_beam",
        key="multistep_param_search_action_strategies",
    )

    r1, r2, r3 = st.columns([1, 1, 1])
    backup_distance_range = r1.text_input(
        "backup distance 范围",
        value="0,1",
        key="multistep_param_search_backup_distance_range",
    )
    backup_score_range = r2.text_input(
        "backup score 范围",
        value="0,1",
        key="multistep_param_search_backup_score_range",
    )
    kg_type = r3.selectbox(
        "KG 数据类型",
        ["augmented", "simple", "any"],
        index=0,
        key="multistep_param_search_kg_type",
        help="默认使用论文主流程的 augmented KG，不再从旧归档实验文件夹读取参数。",
    )

    h1, h2, h3 = st.columns([1, 1, 1])
    hq_top_k = h1.number_input(
        "高质量参数组 top-k",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
        key="multistep_param_search_hq_top_k",
    )
    hq_ratio = h2.number_input(
        "高质量比例阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        format="%.2f",
        key="multistep_param_search_hq_ratio",
    )
    diversity_weight = h3.number_input(
        "参数多样性权重",
        min_value=0.0,
        max_value=2.0,
        value=0.35,
        step=0.05,
        format="%.2f",
        key="multistep_param_search_diversity_weight",
    )

    scenario_options = ["sce1", "sce2", "sce3", "sce1m", "sce2m", "sce3m"]
    selected_scenarios = st.multiselect(
        "选择要启动 multi-step 参数寻优的实验场景",
        options=scenario_options,
        default=["sce1"],
        key="multistep_param_search_scenarios",
        help="按场景从 configs/kg_catalog.yaml 读取 KG/data，新建 training run 并保存到新的 run_XXXX 文件夹。",
    )
    if not selected_scenarios:
        st.warning("至少选择一个实验场景后才能启动 multi-step 参数寻优。")
        return

    scenario_label = ", ".join(selected_scenarios)
    st.caption(
        f"当前选择场景：{scenario_label}；每个场景 Optuna trials={int(trials)}；"
        f"预计新建 training run 数={len(selected_scenarios)}。"
    )

    suggested_batch_tag = f"web_multistep_param_search_manual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_multistep_parameter_search_command(
        [],
        scenarios=selected_scenarios,
        trials=int(trials),
        episodes_per_trial=int(episodes_per_trial),
        timeout_minutes=int(timeout_minutes),
        run_suffix_only=False,
        batch_tag=suggested_batch_tag,
        beam_width_range=str(beam_width_range),
        lookahead_range=str(lookahead_range),
        score_modes=str(score_modes),
        action_strategies=str(action_strategies),
        backup_mode=str(backup_mode),
        backup_distance_range=str(backup_distance_range),
        backup_score_range=str(backup_score_range),
        high_quality_top_k=int(hq_top_k),
        high_quality_ratio=float(hq_ratio),
        diversity_weight=float(diversity_weight),
        kg_type=str(kg_type),
    )
    st.markdown("**multi-step 参数寻优命令**")
    _render_copyable_command(command, f"multistep_param_search_{len(selected_scenarios)}")

    if st.button(
        "启动批量 multi-step 参数寻优",
        key="start_multistep_param_search",
        use_container_width=True,
        disabled=not selected_scenarios,
    ):
        try:
            job = _start_multistep_sensitivity_job(
                [],
                selected_scenarios,
                ["ETG-only"],
                int(trials),
                int(episodes_per_trial),
                int(timeout_minutes),
                False,
                str(beam_width_range),
                str(lookahead_range),
                str(score_modes),
                str(action_strategies),
                str(backup_mode),
                str(backup_distance_range),
                str(backup_score_range),
                int(hq_top_k),
                float(hq_ratio),
                float(diversity_weight),
                str(kg_type),
            )
            st.session_state.setdefault("multistep_sensitivity_jobs", {})[job["batch_id"]] = job
            st.success(
                f"已启动 multi-step 参数寻优 PID={job['pid']}，源实验数={job['count']}，"
                f"每实验 trials={job['trial_count']}，日志：`{job['log_path']}`"
            )
        except Exception as exc:
            st.error(f"启动 multi-step 参数寻优失败：{exc}")

    jobs = st.session_state.get("multistep_sensitivity_jobs", {})
    if jobs:
        with st.expander("multi-step 参数寻优后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "源实验数": job.get("count"),
                    "每实验trials": job.get("trial_count"),
                    "training run数": job.get("job_count"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass
            analysis_cmd = [
                "python",
                str(Path("scripts") / "analyze_multistep_parameter_search.py"),
                "--batch-id",
                str(latest_job.get("batch_id", "")),
                "--high-quality-top-k",
                str(int(hq_top_k)),
                "--high-quality-ratio",
                str(float(hq_ratio)),
                "--diversity-weight",
                str(float(diversity_weight)),
            ]
            st.markdown("**multi-step 参数寻优统计命令**")
            _render_copyable_command(
                " ".join(_quote_cli_arg(part) for part in analysis_cmd),
                f"multistep_param_search_analysis_{latest_job.get('batch_id', '')}",
            )
            if st.button(
                "刷新运行中参数分析",
                key=f"refresh_multistep_param_analysis_{latest_job.get('batch_id', '')}",
                use_container_width=True,
            ):
                try:
                    env = dict(os.environ)
                    env["PYTHONIOENCODING"] = "utf-8"
                    result = subprocess.run(
                        analysis_cmd,
                        cwd=str(ROOT_DIR),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=120,
                        env=env,
                    )
                    if result.returncode == 0:
                        st.success("参数分析已刷新。")
                    else:
                        st.error(result.stdout[-2000:] + "\n" + result.stderr[-2000:])
                except Exception as exc:
                    st.error(f"刷新参数分析失败：{exc}")

            analysis_dir = (
                _ALL_DATA_ROOT
                / "_batch_parameter_search_logs"
                / str(latest_job.get("batch_id", ""))
                / "multistep_parameter_search_analysis"
            )
            best_csv = analysis_dir / "best_parameter_set_by_experiment.csv"
            corr_csv = analysis_dir / "parameter_numeric_correlations.csv"
            effect_csv = analysis_dir / "parameter_categorical_effects.csv"
            if best_csv.exists():
                st.markdown("**当前多目标最优参数组**")
                try:
                    st.dataframe(pd.read_csv(best_csv).head(20), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"读取最优参数表失败：{exc}")
            if corr_csv.exists():
                st.markdown("**数值参数关系 Top 20**")
                try:
                    corr_df = pd.read_csv(corr_csv)
                    st.dataframe(corr_df.head(20), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"读取参数相关性失败：{exc}")
            if effect_csv.exists():
                st.markdown("**类别参数效果摘要**")
                try:
                    effect_df = pd.read_csv(effect_csv)
                    st.dataframe(effect_df.head(40), use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"读取类别参数效果失败：{exc}")
    return
    st.subheader("批量 multi-step 参数敏感性复评")
    st.caption(
        "强制使用 multi-step 规划，扫描 beam、lookahead、评分方式、动作选择策略和备选路径阈值。"
        "每个 final eval 会保存 mechanism_shadow_frames.csv 和 planning_switch_candidates.csv。"
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    episodes = c1.number_input(
        "敏感性 episode 数",
        min_value=1,
        max_value=1000,
        value=100,
        step=10,
        key="multistep_sensitivity_episodes",
    )
    repeats = c2.number_input(
        "敏感性重复次数",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        key="multistep_sensitivity_repeats",
    )
    timeout_minutes = c3.number_input(
        "敏感性单项超时分钟",
        min_value=5,
        max_value=300,
        value=120,
        step=5,
        key="multistep_sensitivity_timeout",
    )
    action_tuning_mode = c4.selectbox(
        "动作微调",
        ["自动", "强制启用", "强制关闭"],
        index=2,
        key="multistep_sensitivity_tuning",
        help="建议先强制关闭得到纯规划敏感性，再用自动或强制启用验证 Synergy 执行阶段。",
    )

    p1, p2, p3 = st.columns([1, 1, 1])
    beam_widths = p1.text_input("beam_widths", value="2,4,6", key="multistep_sensitivity_beams")
    lookaheads = p2.text_input("lookahead_steps", value="2,3", key="multistep_sensitivity_lookaheads")
    backup_distances = p3.text_input(
        "backup distances",
        value="none,0,0.2,0.5",
        key="multistep_sensitivity_backup_distances",
        help="none 表示关闭备选路径，其余数值表示开启备选路径并设置距离阈值。",
    )

    q1, q2, q3 = st.columns([1, 1, 1])
    score_modes = q1.text_input("score modes", value="quality,future_reward", key="multistep_sensitivity_score_modes")
    action_strategies = q2.text_input(
        "action strategies",
        value="highest_transition_prob,best_subtree_quality",
        key="multistep_sensitivity_action_strategies",
    )
    backup_score_thresholds = q3.text_input(
        "backup score thresholds",
        value="0",
        key="multistep_sensitivity_backup_score_thresholds",
    )

    o1, o2 = st.columns([1, 1])
    run_suffix_only = o1.toggle(
        "只显示带 run_XXXX 后缀实验",
        value=True,
        key="multistep_sensitivity_run_suffix_only",
    )
    method_groups = o2.multiselect(
        "方法组",
        options=["ETG-only", "synergy"],
        default=["ETG-only"],
        key="multistep_sensitivity_method_groups",
    )
    if not method_groups:
        st.warning("至少选择一个方法组后才能启动 multi-step 参数敏感性复评。")
        return

    candidates = [
        item
        for item in experiments
        if item.get("method_group") in set(method_groups)
        and not _is_replay_baseline_manifest(item.get("manifest", {}))
        and (not run_suffix_only or _is_run_suffix_experiment(item))
    ]
    variant_count = (
        max(1, len([x for x in str(beam_widths).split(",") if x.strip()]))
        * max(1, len([x for x in str(lookaheads).split(",") if x.strip()]))
        * max(1, len([x for x in str(score_modes).split(",") if x.strip()]))
        * max(1, len([x for x in str(action_strategies).split(",") if x.strip()]))
        * max(1, len([x for x in str(backup_distances).split(",") if x.strip()]))
        * max(1, len([x for x in str(backup_score_thresholds).split(",") if x.strip()]))
    )
    st.caption(
        f"当前候选实验 {len(candidates)} 个；variant 数约 {variant_count}；"
        f"预计 final eval 作业数 {len(candidates) * variant_count}。"
    )
    if not candidates:
        st.info("当前筛选范围内没有可运行的 ETG-only / Synergy 归档实验。")
        return

    option_ids = [item["experiment_id"] for item in candidates]
    label_by_id = {
        item["experiment_id"]: (
            f"{item['method_group']} / {item['experiment_id']} | {item.get('map_key', '')} | "
            f"{item.get('map_id', '')}"
        )
        for item in candidates
    }
    selected_ids = st.multiselect(
        "选择要运行敏感性分析的归档实验",
        options=option_ids,
        default=option_ids[:1],
        format_func=lambda exp_id: label_by_id.get(exp_id, exp_id),
        key=(
            "multistep_sensitivity_selected_ids_"
            f"run{int(bool(run_suffix_only))}_"
            f"methods{'_'.join(method_groups)}"
        ),
    )

    suggested_batch_tag = f"web_multistep_sensitivity_manual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_multistep_sensitivity_command(
        selected_ids,
        method_groups=method_groups,
        episodes=int(episodes),
        repeats=int(repeats),
        timeout_minutes=int(timeout_minutes),
        action_tuning_mode=action_tuning_mode,
        run_suffix_only=bool(run_suffix_only),
        batch_tag=suggested_batch_tag,
        beam_widths=str(beam_widths),
        lookaheads=str(lookaheads),
        score_modes=str(score_modes),
        action_strategies=str(action_strategies),
        backup_distances=str(backup_distances),
        backup_score_thresholds=str(backup_score_thresholds),
    )
    st.markdown("**multi-step 参数敏感性命令**")
    _render_copyable_command(command, f"multistep_sensitivity_{len(selected_ids)}")

    if st.button(
        "启动批量 multi-step 参数敏感性复评",
        key="start_multistep_sensitivity",
        use_container_width=True,
        disabled=not selected_ids,
    ):
        try:
            job = _start_multistep_sensitivity_job(
                selected_ids,
                method_groups,
                int(episodes),
                int(repeats),
                int(timeout_minutes),
                action_tuning_mode,
                bool(run_suffix_only),
                str(beam_widths),
                str(lookaheads),
                str(score_modes),
                str(action_strategies),
                str(backup_distances),
                str(backup_score_thresholds),
            )
            st.session_state.setdefault("multistep_sensitivity_jobs", {})[job["batch_id"]] = job
            st.success(
                f"已启动 multi-step 敏感性复评 PID={job['pid']}，实验数={job['count']}，"
                f"variant数≈{job['variant_count']}，预计作业数≈{job['job_count']}，日志：`{job['log_path']}`"
            )
        except Exception as exc:
            st.error(f"启动 multi-step 敏感性复评失败：{exc}")

    jobs = st.session_state.get("multistep_sensitivity_jobs", {})
    if jobs:
        with st.expander("multi-step 敏感性后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "实验数": job.get("count"),
                    "variant数": job.get("variant_count"),
                    "预计作业数": job.get("job_count"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass
            a1, a2, a3 = st.columns([1, 1, 1])
            hq_top_k = a1.number_input(
                "高质量参数组 top-k",
                min_value=1,
                max_value=50,
                value=8,
                step=1,
                key="multistep_sensitivity_hq_top_k",
            )
            hq_ratio = a2.number_input(
                "高质量比例阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                format="%.2f",
                key="multistep_sensitivity_hq_ratio",
            )
            diversity_weight = a3.number_input(
                "参数多样性权重",
                min_value=0.0,
                max_value=2.0,
                value=0.35,
                step=0.05,
                format="%.2f",
                key="multistep_sensitivity_diversity_weight",
            )
            analysis_cmd = [
                "python",
                str(Path("scripts") / "analyze_multistep_sensitivity_eval.py"),
                "--batch-id",
                str(latest_job.get("batch_id", "")),
                "--high-quality-top-k",
                str(int(hq_top_k)),
                "--high-quality-ratio",
                str(float(hq_ratio)),
                "--diversity-weight",
                str(float(diversity_weight)),
            ]
            st.markdown("**multi-step 敏感性统计命令**")
            _render_copyable_command(
                " ".join(_quote_cli_arg(part) for part in analysis_cmd),
                f"multistep_sensitivity_analysis_{latest_job.get('batch_id', '')}",
            )


def _render_fixed_pool_synergy_panel() -> None:
    st.subheader("高质量参数池 Synergy 协同训练")
    st.caption(
        "读取 multi-step 参数寻优分析导出的 high_quality_diverse_variant_overrides.json，"
        "按这些固定参数组运行 ETG-only + exploration_only + synergy 三阶段循环协同训练，并自动归档到 all_data/synergy，后续可直接 final_eval。"
    )
    default_variant_json = ""
    jobs = st.session_state.get("multistep_sensitivity_jobs", {})
    if jobs:
        latest_job = list(jobs.values())[-1]
        batch_id = str(latest_job.get("batch_id", ""))
        candidate = (
            _ALL_DATA_ROOT
            / "_batch_parameter_search_logs"
            / batch_id
            / "multistep_parameter_search_analysis"
            / "high_quality_diverse_variant_overrides.json"
        )
        default_variant_json = str(candidate)
    variant_json = st.text_input(
        "高质量参数组 JSON",
        value=default_variant_json,
        key="fixed_pool_synergy_variant_json",
        help="通常是 analyze_multistep_parameter_search.py 生成的 high_quality_diverse_variant_overrides.json。",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    variants_per_exp = c1.number_input(
        "每实验参数组数",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
        key="fixed_pool_synergy_variants_per_exp",
    )
    episodes_per_trial = c2.number_input(
        "每trial局数",
        min_value=1,
        max_value=1000,
        value=300,
        step=50,
        key="fixed_pool_synergy_episodes",
        help="每个 trial 内实际执行的对局数。大规模机制分析建议至少 300。",
    )
    cycle_count = c3.number_input(
        "循环轮次",
        min_value=1,
        max_value=200,
        value=18,
        step=1,
        key="fixed_pool_synergy_cycle_count",
        help="总trial数 = 循环轮次 × 每轮三阶段trial数。默认 18 × 60 = 1080。",
    )
    c_stage1, c_stage2, c_stage3 = st.columns([1, 1, 1])
    etg_trials_per_cycle = c_stage1.number_input(
        "每轮 ETG-only trial数",
        min_value=0,
        max_value=1000,
        value=20,
        step=5,
        key="fixed_pool_synergy_etg_trials_per_cycle",
    )
    exploration_trials_per_cycle = c_stage2.number_input(
        "每轮 exploration_only trial数",
        min_value=0,
        max_value=1000,
        value=20,
        step=5,
        key="fixed_pool_synergy_exploration_trials_per_cycle",
    )
    synergy_trials_per_cycle = c_stage3.number_input(
        "每轮 Synergy trial数",
        min_value=1,
        max_value=1000,
        value=20,
        step=5,
        key="fixed_pool_synergy_synergy_trials_per_cycle",
    )
    total_trials = int(cycle_count) * (
        int(etg_trials_per_cycle)
        + int(exploration_trials_per_cycle)
        + int(synergy_trials_per_cycle)
    )
    st.info(
        f"当前设置总trial数：{total_trials} = {int(cycle_count)} × "
        f"({int(etg_trials_per_cycle)} + {int(exploration_trials_per_cycle)} + {int(synergy_trials_per_cycle)})；"
        f"预计每个源实验对局数：{total_trials * int(episodes_per_trial)}。"
    )
    selection = st.selectbox(
        "参数组调度方式",
        ["round_robin", "weighted", "best"],
        index=0,
        key="fixed_pool_synergy_selection",
        help="round_robin 更适合覆盖多种规划逻辑，weighted 更偏向高分参数组，best 只用最高分参数组。",
    )

    experiment_ids: List[str] = []
    variant_path = Path(variant_json) if variant_json else Path()
    if variant_json and variant_path.exists():
        try:
            data = json.loads(variant_path.read_text(encoding="utf-8"))
            variants = data.get("variants", data if isinstance(data, list) else [])
            experiment_ids = sorted(
                {
                    str(item.get("experiment_id"))
                    for item in variants
                    if isinstance(item, dict) and item.get("experiment_id")
                }
            )
        except Exception as exc:
            st.warning(f"读取参数组 JSON 失败：{exc}")
    selected_ids = st.multiselect(
        "选择要启动 Synergy 协同训练的源实验",
        options=experiment_ids,
        default=experiment_ids[:1],
        key="fixed_pool_synergy_selected_ids",
    )
    if variant_json and not variant_path.exists():
        st.info("请先运行上方敏感性统计命令，生成 high_quality_diverse_variant_overrides.json。")

    suggested_batch_tag = f"web_fixed_pool_synergy_manual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    command = _build_fixed_pool_synergy_command(
        variant_json,
        selected_ids,
        episodes_per_trial=int(episodes_per_trial),
        cycle_count=int(cycle_count),
        etg_trials_per_cycle=int(etg_trials_per_cycle),
        exploration_trials_per_cycle=int(exploration_trials_per_cycle),
        synergy_trials_per_cycle=int(synergy_trials_per_cycle),
        variants_per_experiment=int(variants_per_exp),
        selection=str(selection),
        batch_tag=suggested_batch_tag,
    )
    st.markdown("**固定参数池 Synergy 训练命令**")
    _render_copyable_command(command, f"fixed_pool_synergy_{len(selected_ids)}")

    disabled = not variant_json or not variant_path.exists() or not selected_ids
    if st.button(
        "启动固定参数池 Synergy 协同训练",
        key="start_fixed_pool_synergy",
        use_container_width=True,
        disabled=disabled,
    ):
        try:
            job = _start_fixed_pool_synergy_job(
                variant_json,
                selected_ids,
                int(episodes_per_trial),
                int(cycle_count),
                int(etg_trials_per_cycle),
                int(exploration_trials_per_cycle),
                int(synergy_trials_per_cycle),
                int(variants_per_exp),
                str(selection),
            )
            st.session_state.setdefault("fixed_pool_synergy_jobs", {})[job["batch_id"]] = job
            st.success(
                f"已启动固定参数池 Synergy PID={job['pid']}，源实验数={job['count']}，总trial={job.get('total_trials')}，日志：`{job['log_path']}`"
            )
        except Exception as exc:
            st.error(f"启动固定参数池 Synergy 失败：{exc}")

    fixed_jobs = st.session_state.get("fixed_pool_synergy_jobs", {})
    if fixed_jobs:
        with st.expander("固定参数池 Synergy 后台任务", expanded=False):
            rows = [
                {
                    "批次ID": batch_id,
                    "PID": job.get("pid"),
                    "源实验数": job.get("count"),
                    "总trial": job.get("total_trials"),
                    "参数组JSON": job.get("variant_json"),
                    "日志": job.get("log_path"),
                    "启动时间": job.get("started_at"),
                }
                for batch_id, job in fixed_jobs.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            latest_job = list(fixed_jobs.values())[-1]
            log_path = Path(latest_job.get("log_path", ""))
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-100:]), language="text")
                except Exception:
                    pass


def _render_experiment_compare_tab() -> None:
    st.caption(
        "读取 `output/learner_results/all_data/<method_group>/*` 下的归档实验，用于比较不同方法组、地图、ETG 来源、BKTree 阈值和关键产物。"
    )

    experiments = _scan_experiments()
    st.subheader("场景配置完备性总览（Synergy 组）")
    st.caption(
        "该表仅检查三阶段协同实验组是否已按统一标准完成各地图配置与归档；ETG-only 等基线实验在下方归档实验总览中比较。"
    )
    scene_rows = _scene_completion_rows(experiments)
    st.dataframe(pd.DataFrame(scene_rows), use_container_width=True, hide_index=True)

    with st.expander("其他地图复现 sce1 结果的执行步骤", expanded=False):
        if _REPRO_GUIDE_PATH.exists():
            st.markdown(_REPRO_GUIDE_PATH.read_text(encoding="utf-8"))
        else:
            st.info(f"复现说明文件不存在：`{_REPRO_GUIDE_PATH}`")

    with st.expander("批量实验复刻硬编码配置审计与修复记录", expanded=False):
        if _CONFIG_AUDIT_PATH.exists():
            st.markdown(_CONFIG_AUDIT_PATH.read_text(encoding="utf-8"))
        else:
            st.info(f"审计说明文件不存在：`{_CONFIG_AUDIT_PATH}`")

    render_teacher_guided_etg_panel()
    _render_replay_baseline_panel()

    if not experiments:
        st.info("尚未发现归档实验。请将实验目录放入 `output/learner_results/all_data/<method_group>/`。")
        return

    scenario_options = _scenario_filter_options(experiments)
    scenario_values = [option["value"] for option in scenario_options]
    selected_scenarios = st.multiselect(
        "场景筛选（可多选；空选表示全部场景）",
        options=scenario_values,
        default=scenario_values,
        format_func=lambda value: next(
            (option["label"] for option in scenario_options if option["value"] == value),
            str(value),
        ),
        key="archived_experiment_scenario_filter",
    )
    filtered_experiments = _filter_experiments_by_scenarios(
        experiments,
        scenario_options,
        selected_scenarios,
    )

    t1, t2, t3 = st.columns([2, 1, 1])
    overview_sort_mode = t1.selectbox(
        "归档实验排序",
        ["地图 → 方法组", "方法组 → 地图", "最近更新", "实验ID"],
        index=0,
        key="archived_experiment_sort_mode",
    )
    show_paper_table = t2.toggle(
        "论文表格视图",
        value=False,
        key="archived_experiment_paper_table_mode",
    )
    show_all_overview_fields = t3.toggle(
        "显示全部字段",
        value=False,
        key="archived_experiment_show_all_fields",
    )
    displayed_experiments = _sort_experiments(filtered_experiments, overview_sort_mode)
    st.caption(
        f"当前显示 {len(displayed_experiments)} / {len(experiments)} 个归档实验；可通过场景筛选切换全部、多个指定场景或单一场景。"
    )
    if not displayed_experiments:
        st.info("当前场景筛选下没有匹配的归档实验。")
        return

    _render_batch_final_eval_panel(displayed_experiments)
    _render_fixed_pool_variant_final_eval_panel(displayed_experiments)
    _render_fixed_pool_switch_ablation_panel(displayed_experiments)
    _render_switch_grid_eval_panel(displayed_experiments)
    _render_multistep_sensitivity_panel(displayed_experiments)
    _render_fixed_pool_synergy_panel()

    overview_rows = []
    for item in displayed_experiments:
        latest_eval = _latest_final_eval(Path(item["path"]))
        overview_rows.append(
            {
                "实验ID": item["experiment_id"],
                "方法组": item["method_group"],
                "地图": item["map_id"],
                "map_key": item["map_key"],
                "类型": item["experiment_type"],
                "ETG来源": item["kg_name"],
                "数据集": item["dataset_type"],
                "重演扩张": item["replay_dataset_expansion"],
                "BKTree阈值": f"{item['bktree_primary']} / {item['bktree_secondary']}",
                "总trial": item["total_trials"],
                "完成trial": item["completed_trials"],
                "最优目标值": _fmt(item["best_value"]),
                "最优胜率": _fmt(item["best_win_rate"]),
                "最优最终得分": _fmt(item["best_avg_score"]),
                "稳定性": _fmt(item["best_stability"]),
                "惩罚系数": _fmt(item["best_penalty_factor"]),
                "复评胜率": _metric_ci(latest_eval, "win_rate") if latest_eval else "-",
                "复评平均得分": _metric_ci(latest_eval, "avg_score") if latest_eval else "-",
                "复评稳定性": _metric_ci(latest_eval, "stability") if latest_eval else "-",
                "复评命令": "点击行后在下方生成",
                "阶段": item["phases"],
                "manifest": item["manifest_status"],
            }
        )

    st.subheader("归档实验总览")
    if show_paper_table:
        overview_df = pd.DataFrame(_paper_table_rows(displayed_experiments))
        st.caption("论文表格视图按当前排序组织，突出地图、方法与复评指标；隐藏命令、manifest 和过程字段。")
    else:
        overview_df = pd.DataFrame(overview_rows)
    if not show_all_overview_fields and not show_paper_table:
        compact_columns = [
            "实验ID",
            "方法组",
            "地图",
            "类型",
            "ETG来源",
            "BKTree阈值",
            "最优目标值",
            "最优胜率",
            "最优最终得分",
            "稳定性",
            "复评命令",
            "manifest",
        ]
        overview_df = overview_df[[c for c in compact_columns if c in overview_df.columns]]
        st.caption("当前为简略视图；`总trial`、`完成trial`、复评指标、数据集等字段已隐藏。打开右侧开关可显示全部字段。")
    selected_from_table: Optional[int] = None
    try:
        table_event = st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="archived_experiment_overview_table",
        )
        selected_rows = getattr(getattr(table_event, "selection", None), "rows", [])
        if selected_rows:
            selected_from_table = int(selected_rows[0])
    except TypeError:
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

    if selected_from_table is not None and 0 <= selected_from_table < len(displayed_experiments):
        clicked = displayed_experiments[selected_from_table]
        clicked_cmd = None if _is_replay_baseline_manifest(clicked.get("manifest", {})) else _build_final_eval_command(Path(clicked["path"]))
        with st.expander("当前选中行的复评命令", expanded=True):
            st.caption("该命令使用默认设置：100 episodes、1 次重复、90 分钟超时。详细设置可在下方实验详情中调整。")
            if clicked_cmd is None:
                st.caption("该行是历史动作序列回放基线；如需重跑，请使用上方“历史动作序列回放对照组”模块生成命令。")
            else:
                _render_copyable_command(clicked_cmd, f"overview_{clicked['experiment_id']}")

    labels = [
        f"{item['method_group']} / {item['experiment_id']} | {item['map_id']}"
        for item in displayed_experiments
    ]
    selected_idx = st.selectbox(
        "选择实验查看细节",
        options=range(len(displayed_experiments)),
        format_func=lambda idx: labels[idx],
        index=selected_from_table if selected_from_table is not None else 0,
    )
    selected = displayed_experiments[selected_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Value", _fmt(selected["best_value"]))
    c2.metric("Best Trial", selected["best_trial"])
    c3.metric("Total Trials", selected["total_trials"])
    c4.metric("Completed", selected["completed_trials"])

    st.subheader("阶段统计")
    phase_df = selected["phase_df"]
    if phase_df.empty:
        if _is_replay_baseline_manifest(selected.get("manifest", {})):
            st.info("历史动作序列回放基线没有 Optuna trial 阶段；请使用回放评估指标进行对比。")
        else:
            st.warning("该实验缺少 `study_summary.json` 或 trial 阶段信息。")
    else:
        display_df = phase_df[
            [
                "phase_label",
                "trials",
                "complete",
                "pruned_or_probe",
                "objective_mean",
                "objective_max",
                "probe_mean",
                "probe_max",
                "avg_score_mean",
                "win_rate_mean",
                "episodes",
            ]
        ].rename(
            columns={
                "phase_label": "阶段",
                "trials": "trial数",
                "complete": "参与寻优",
                "pruned_or_probe": "探索/剪枝",
                "objective_mean": "优化目标均值",
                "objective_max": "优化目标最大",
                "probe_mean": "probe均值",
                "probe_max": "probe最大",
                "avg_score_mean": "episode得分均值",
                "win_rate_mean": "胜率均值",
                "episodes": "episode数",
            }
        )
        st.plotly_chart(_plot_phase_stats(phase_df), use_container_width=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("配置与产物")
    artifact_rows = [
        {"产物": name, "状态": "存在" if exists else "缺失"}
        for name, exists in selected["artifacts"].items()
    ]
    a1, a2 = st.columns([1, 2])
    with a1:
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)
        if not selected["artifacts"].get("manifest"):
            _render_manifest_template(selected)
    with a2:
        st.json(
            {
                "experiment_id": selected["experiment_id"],
                "method_group": selected["method_group"],
                "map_key": selected["map_key"],
                "map_id": selected["map_id"],
                "kg_name": selected["kg_name"],
                "kg_file": selected["kg_file"],
                "data_dir": selected["data_dir"],
                "bktree": {
                    "primary_threshold": selected["bktree_primary"],
                    "secondary_threshold": selected["bktree_secondary"],
                    "path": selected["bktree_path"],
                },
                "source_run": selected["source_run"],
                "path": selected["path"],
            }
        )

    summary = selected["summary"]
    if summary.get("best_params"):
        with st.expander("最优参数"):
            st.json(summary["best_params"])
    with st.expander("Manifest 原文"):
        st.json(selected["manifest"])

    if _is_replay_baseline_manifest(selected.get("manifest", {})):
        rows = _final_eval_rows(Path(selected["path"]))
        if rows:
            st.subheader("历史回放评估结果")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        _render_final_eval_panel(selected, experiments)

    _render_mechanism_diagnostics_panel(displayed_experiments, selected)
    _render_common_bktree_space_panel(displayed_experiments, selected)
