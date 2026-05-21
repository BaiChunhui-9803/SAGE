import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from src import ROOT_DIR


_ALL_DATA_ROOT = ROOT_DIR / "output" / "learner_results" / "all_data"
_KG_CATALOG_PATH = ROOT_DIR / "configs" / "kg_catalog.yaml"
_REPRO_GUIDE_PATH = ROOT_DIR / "docs" / "batch_experiment_reproduction_guide.md"
_CONFIG_AUDIT_PATH = ROOT_DIR / "docs" / "batch_experiment_hardcoded_config_audit.md"
_MANIFEST_NAME = "experiment_manifest.json"
_FINAL_EVAL_SCRIPT = ROOT_DIR / "scripts" / "evaluate_archived_experiment.py"

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

_REQUIRED_ARTIFACTS = [
    "manifest",
    "study_summary",
    "study_db",
    "runs",
    "trials",
    "action_tuning_model",
    "learner_log",
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


def _required_artifacts_for_manifest(manifest: Dict[str, Any]) -> List[str]:
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
    }


def _has_complete_manifest(manifest: Dict[str, Any]) -> bool:
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
        artifacts = _artifact_status(exp_dir)
        artifact_summary = _artifact_summary(artifacts, manifest)

        phase_df = _phase_stats(summary) if summary else pd.DataFrame()
        if _is_etg_only_manifest(manifest) and not phase_df.empty:
            phase_df = phase_df.copy()
            phase_df["phase"] = "etg_only"
            phase_df["phase_label"] = "ETG-only"
        phase_labels = (
            " / ".join(phase_df["phase_label"].tolist()) if not phase_df.empty else "-"
        )

        experiments.append(
            {
                "path": str(exp_dir),
                "method_group": method_group,
                "experiment_id": manifest.get("experiment_id", exp_dir.name),
                "display_name": manifest.get("display_name") or manifest.get("experiment_id", exp_dir.name),
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
                "best_value": summary.get("best_value", "-"),
                "best_trial": summary.get("best_trial", "-"),
                "best_win_rate": _best_trial_metric(summary, "win_rate"),
                "best_avg_score": _best_trial_metric(summary, "avg_score"),
                "best_stability": _best_trial_metric(summary, "stability"),
                "best_penalty_factor": _best_trial_metric(summary, "penalty_factor"),
                "best_num_episodes": _best_trial_metric(summary, "num_episodes"),
                "saved_at": summary.get("saved_at", "-"),
                "phases": phase_labels,
                "manifest_status": "已登记" if artifacts["manifest"] else "缺失",
                "artifacts": artifacts,
                "artifact_summary": artifact_summary,
                "manifest": manifest,
                "summary": summary,
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


def _final_eval_rows(exp_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for summary in _load_final_eval_summaries(exp_dir):
        rows.append(
            {
                "复评ID": Path(summary.get("_eval_dir", "")).name,
                "状态": summary.get("status", "-"),
                "回合/重复": f"{summary.get('episodes_per_repeat', '-')} × {summary.get('requested_repeats', '-')}",
                "总回合": _nested_get(summary, "aggregate", "total_episodes", default="-"),
                "胜率": _metric_ci(summary, "win_rate"),
                "平均得分": _metric_ci(summary, "avg_score"),
                "稳定性": _metric_ci(summary, "stability"),
                "惩罚系数": _metric_ci(summary, "penalty_factor"),
                "目标值": _metric_ci(summary, "objective"),
                "动作微调": "启用" if summary.get("action_tuning_enabled") else "关闭",
                "完成时间": summary.get("completed_at") or summary.get("updated_at") or "-",
                "目录": summary.get("_eval_dir", "-"),
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
    ]
    if action_tuning_mode == "强制启用":
        cmd.append("--enable-action-tuning")
    elif action_tuning_mode == "强制关闭":
        cmd.append("--disable-action-tuning")

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log_file = open(str(log_path), "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        creationflags=flags,
    )
    return {
        "pid": proc.pid,
        "cmd": cmd,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _render_final_eval_panel(selected: Dict[str, Any], experiments: List[Dict[str, Any]]) -> None:
    exp_dir = Path(selected["path"])
    st.subheader("最终复评")
    st.caption(
        "用归档实验的 best params 重新跑固定 episode 数；Synergy 默认复用归档的 `action_tuning_model.pkl`，结果独立写入 `final_eval/`。"
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
                "完成时间": latest.get("completed_at") or latest.get("updated_at") or "-",
            }
        )
    if same_map_rows:
        st.markdown("**同地图最新复评对比**")
        st.dataframe(pd.DataFrame(same_map_rows), use_container_width=True, hide_index=True)


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

    if not experiments:
        st.info("尚未发现归档实验。请将实验目录放入 `output/learner_results/all_data/<method_group>/`。")
        return

    overview_rows = []
    for item in experiments:
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

    h1, h2 = st.columns([4, 1])
    h1.subheader("归档实验总览")
    show_all_overview_fields = h2.toggle(
        "显示全部字段",
        value=False,
        key="archived_experiment_show_all_fields",
    )
    overview_df = pd.DataFrame(overview_rows)
    if not show_all_overview_fields:
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

    if selected_from_table is not None and 0 <= selected_from_table < len(experiments):
        clicked = experiments[selected_from_table]
        clicked_cmd = _build_final_eval_command(Path(clicked["path"]))
        with st.expander("当前选中行的复评命令", expanded=True):
            st.caption("该命令使用默认设置：100 episodes、1 次重复、90 分钟超时。详细设置可在下方实验详情中调整。")
            _render_copyable_command(clicked_cmd, f"overview_{clicked['experiment_id']}")

    labels = [
        f"{item['method_group']} / {item['experiment_id']} | {item['map_id']}"
        for item in experiments
    ]
    selected_idx = st.selectbox(
        "选择实验查看细节",
        options=range(len(experiments)),
        format_func=lambda idx: labels[idx],
        index=selected_from_table if selected_from_table is not None else 0,
    )
    selected = experiments[selected_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Value", _fmt(selected["best_value"]))
    c2.metric("Best Trial", selected["best_trial"])
    c3.metric("Total Trials", selected["total_trials"])
    c4.metric("Completed", selected["completed_trials"])

    st.subheader("阶段统计")
    phase_df = selected["phase_df"]
    if phase_df.empty:
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

    _render_final_eval_panel(selected, experiments)
