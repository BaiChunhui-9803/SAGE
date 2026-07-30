"""Local-only manual episode filtering tab for archived final-eval data."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from etg_web.i18n import st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALL_DATA_ROOT = PROJECT_ROOT / "output" / "learner_results" / "all_data"
METHOD_DIRS = {
    "etg-only": "etg-only",
    "Synergy": "synergy",
}
SELECTION_DIR_NAME = "manual_episode_selections"
LOCAL_ONLY_NOTE = "本 tab 为本地人工数据筛选辅助功能，筛选文件与页面组件均标记为不需要上传到 git。"
CACHE_TTL_SECONDS = 30


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name.strip(), flags=re.UNICODE)
    return name.strip("._") or "manual_selection"


def _infer_scenario_from_name(experiment_id: str) -> str:
    match = re.match(r"(.+?)_(?:etg|synergy).*?_run_\d+", experiment_id, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.match(r"(.+?)_run_\d+", experiment_id, flags=re.IGNORECASE)
    return match.group(1) if match else experiment_id


def _infer_batch_tag(eval_name: str, summary: Dict[str, Any]) -> str:
    batch_tag = str(summary.get("batch_tag") or "").strip()
    if batch_tag:
        return batch_tag
    for pattern in [
        r"(web_batch_\d{8}_\d{6})",
        r"(web_[A-Za-z0-9_]+_\d{8}_\d{6})",
        r"(swg_\d{4}_\d{6})",
        r"(batch_[A-Za-z0-9_]+)",
    ]:
        match = re.search(pattern, eval_name)
        if match:
            return match.group(1)
    return "unknown_batch"


def _short_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _iter_final_eval_dirs(experiment_dir: Path) -> Iterable[Path]:
    final_eval_root = experiment_dir / "final_eval"
    if not final_eval_root.exists():
        return []
    return sorted(
        [p for p in final_eval_root.iterdir() if p.is_dir() and p.name.startswith("eval_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _has_episode_file(eval_dir: Path) -> bool:
    return any(eval_dir.glob("repeats/repeat_*/episodes.jsonl"))


def _is_eval_complete(eval_dir: Path) -> bool:
    summary = _safe_read_json(eval_dir / "final_eval_summary.json")
    if not summary:
        return False
    status = str(summary.get("status") or summary.get("state") or "").lower()
    if status and status not in {"completed", "complete", "success", "finished"}:
        return False
    return _has_episode_file(eval_dir)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def _scan_eval_catalog(include_unfinished: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method_label, method_dir in METHOD_DIRS.items():
        root = ALL_DATA_ROOT / method_dir
        if not root.exists():
            continue
        for experiment_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            manifest = _safe_read_json(experiment_dir / "experiment_manifest.json")
            experiment_id = manifest.get("experiment_id") or experiment_dir.name
            scenario = manifest.get("map_key") or _infer_scenario_from_name(experiment_id)
            map_id = manifest.get("map_id") or ""
            run_id = manifest.get("source_run") or ""
            if not run_id:
                run_match = re.search(r"run_(\d+)", experiment_id)
                run_id = f"run_{run_match.group(1)}" if run_match else ""
            for eval_dir in _iter_final_eval_dirs(experiment_dir):
                has_episode = _has_episode_file(eval_dir)
                complete = _is_eval_complete(eval_dir)
                if not has_episode:
                    continue
                if not include_unfinished and not complete:
                    continue
                summary = _safe_read_json(eval_dir / "final_eval_summary.json")
                rows.append(
                    {
                        "method": method_label,
                        "scenario": scenario,
                        "map_id": map_id,
                        "experiment_id": experiment_id,
                        "display_name": manifest.get("display_name") or experiment_id,
                        "run_id": run_id,
                        "eval_name": eval_dir.name,
                        "batch_tag": _infer_batch_tag(eval_dir.name, summary),
                        "eval_dir": str(eval_dir),
                        "complete": complete,
                        "summary_score": _safe_float(
                            summary.get("mean_score", summary.get("avg_score")), math.nan
                        ),
                        "summary_win_rate": _safe_float(
                            summary.get("win_rate", summary.get("winrate")), math.nan
                        ),
                        "mtime": datetime.fromtimestamp(eval_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
    return pd.DataFrame(rows)


def _frame_flag_count(frames: List[Dict[str, Any]], predicate) -> int:
    count = 0
    for frame in frames:
        try:
            if predicate(frame):
                count += 1
        except Exception:
            continue
    return count


def _extract_gate_tuning(frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    shadow = frame.get("mechanism_shadow") or {}
    tuning = shadow.get("tuning")
    if isinstance(tuning, dict):
        return tuning
    plan = frame.get("plan") or {}
    tuning = plan.get("action_tuning")
    if isinstance(tuning, dict):
        return tuning
    return None


def _finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        number = _safe_float(value, math.nan)
        if not math.isnan(number):
            out.append(number)
    return out


def _mean_or_zero(values: Iterable[Any]) -> float:
    finite = _finite_values(values)
    return float(sum(finite) / len(finite)) if finite else 0.0


def _max_or_zero(values: Iterable[Any]) -> float:
    finite = _finite_values(values)
    return float(max(finite)) if finite else 0.0


def _ratio_or_zero(flags: Iterable[bool]) -> float:
    flags = list(flags)
    return float(sum(1 for flag in flags if flag) / len(flags)) if flags else 0.0


def _bounded_positive_mean(values: Iterable[Any], scale: float = 10.0) -> float:
    finite = [max(0.0, value) for value in _finite_values(values)]
    if not finite:
        return 0.0
    bounded = [value / (value + scale) if value > 0 else 0.0 for value in finite]
    return float(sum(bounded) / len(bounded))


def _bounded_positive_max(values: Iterable[Any], scale: float = 10.0) -> float:
    finite = [max(0.0, value) for value in _finite_values(values)]
    if not finite:
        return 0.0
    bounded = [value / (value + scale) if value > 0 else 0.0 for value in finite]
    return float(max(bounded))


def _visit_support_values(visits: Iterable[Any], thresholds: Iterable[Any]) -> List[float]:
    values: List[float] = []
    for visit_raw, threshold_raw in zip(visits, thresholds):
        visit = _safe_float(visit_raw, math.nan)
        threshold = _safe_float(threshold_raw, math.nan)
        if math.isnan(visit) or visit <= 0:
            values.append(0.0)
        elif math.isnan(threshold) or threshold <= 0:
            values.append(visit / (visit + 10.0))
        else:
            values.append(min(visit / threshold, 1.0))
    return values


def _episode_record(
    item: Dict[str, Any],
    *,
    repeat_id: str,
    line_no: int,
    eval_dir: Path,
    catalog_row: Dict[str, Any],
) -> Dict[str, Any]:
    frames = item.get("frames") or []
    first_frame = frames[0] if frames else {}
    last_frame = frames[-1] if frames else {}
    state_seq = item.get("state_key_sequence") or item.get("state_id_sequence") or []
    score = _safe_float(item.get("score", item.get("final_score", last_frame.get("hp_delta"))))
    result = str(item.get("result") or item.get("win") or "")
    win = result.lower() in {"win", "true", "1", "success"} or item.get("win") is True
    episode_id = item.get("episode_id", item.get("trial_number", line_no))
    total_frames = len(frames)
    ood_frames = _frame_flag_count(
        frames,
        lambda f: bool(f.get("nid_is_ood"))
        or str(f.get("eval_bktree_status", "")).lower() == "ood"
        or str(f.get("nid_status", "")).lower() in {"ood", "bktree_rejected"},
    )
    changed_frames = _frame_flag_count(
        frames,
        lambda f: bool(f.get("shadow_mechanism_changed_action"))
        or bool((f.get("mechanism_shadow") or {}).get("mechanism_changed_action")),
    )
    switch_frames = _frame_flag_count(frames, lambda f: bool(f.get("switch_event")))
    tuning_frames = _frame_flag_count(
        frames,
        lambda f: "mc" in str(f.get("action_source") or "").lower()
        or bool(f.get("is_exploration")),
    )
    gate_tunings = [tuning for frame in frames if (tuning := _extract_gate_tuning(frame)) is not None]
    gate_advantages = [_safe_float(tuning.get("advantage"), math.nan) for tuning in gate_tunings]
    gate_confidences = [_safe_float(tuning.get("confidence"), math.nan) for tuning in gate_tunings]
    gate_visits = [_safe_float(tuning.get("candidate_visits"), math.nan) for tuning in gate_tunings]
    gate_threshold_advantages = [_safe_float(tuning.get("threshold_advantage"), math.nan) for tuning in gate_tunings]
    gate_threshold_confidences = [_safe_float(tuning.get("threshold_confidence"), math.nan) for tuning in gate_tunings]
    gate_threshold_visits = [_safe_float(tuning.get("threshold_visits"), math.nan) for tuning in gate_tunings]
    gate_advantage_flags = []
    for advantage, threshold in zip(gate_advantages, gate_threshold_advantages):
        if math.isnan(advantage):
            gate_advantage_flags.append(False)
        elif math.isnan(threshold):
            gate_advantage_flags.append(advantage > 0)
        else:
            gate_advantage_flags.append(advantage >= threshold)
    gate_confidence_flags = []
    for confidence, threshold in zip(gate_confidences, gate_threshold_confidences):
        if math.isnan(confidence):
            gate_confidence_flags.append(False)
        elif math.isnan(threshold):
            gate_confidence_flags.append(confidence > 0)
        else:
            gate_confidence_flags.append(confidence >= threshold)
    gate_supported_flags = []
    for visits, threshold in zip(gate_visits, gate_threshold_visits):
        if math.isnan(visits):
            gate_supported_flags.append(False)
        elif math.isnan(threshold):
            gate_supported_flags.append(visits > 0)
        else:
            gate_supported_flags.append(visits >= threshold)
    gate_validation_flags = [
        bool(tuning.get("validation")) or bool(tuning.get("candidate_eligible"))
        for tuning in gate_tunings
    ]
    gate_joint_pass_flags = [
        bool(advantage_ok and confidence_ok and visits_ok)
        for advantage_ok, confidence_ok, visits_ok in zip(
            gate_advantage_flags,
            gate_confidence_flags,
            gate_supported_flags,
        )
    ]
    gate_changed_flags = [
        bool((frame.get("mechanism_shadow") or {}).get("mechanism_changed_action"))
        or bool(frame.get("shadow_mechanism_changed_action"))
        for frame in frames
        if _extract_gate_tuning(frame) is not None
    ]
    plan_times = [
        _safe_float((frame.get("planning_timing") or {}).get("decision_elapsed_ms"), math.nan)
        for frame in frames
    ]
    plan_times = [v for v in plan_times if not math.isnan(v)]
    selection_key = f"{repeat_id}::ep_{episode_id}::line_{line_no}"
    gate_A_pass_ratio = _ratio_or_zero(gate_advantage_flags)
    gate_C_pass_ratio = _ratio_or_zero(gate_confidence_flags)
    gate_v_pass_ratio = _ratio_or_zero(gate_supported_flags)
    gate_validated_ratio = _ratio_or_zero(gate_validation_flags)
    gate_joint_pass_ratio = _ratio_or_zero(gate_joint_pass_flags)
    gate_changed_ratio = _ratio_or_zero(gate_changed_flags)
    gate_ratio = (len(gate_tunings) / total_frames) if total_frames else 0.0
    gate_visit_support_values = _visit_support_values(gate_visits, gate_threshold_visits)
    gate_visit_support_mean = float(sum(gate_visit_support_values) / len(gate_visit_support_values)) if gate_visit_support_values else 0.0
    gate_visit_support_max = float(max(gate_visit_support_values)) if gate_visit_support_values else 0.0
    gate_mean_evidence_score = (
        0.4 * _bounded_positive_mean(gate_advantages)
        + 0.3 * _mean_or_zero(gate_confidences)
        + 0.3 * gate_visit_support_mean
    )
    gate_peak_evidence_score = (
        0.4 * _bounded_positive_max(gate_advantages)
        + 0.3 * _max_or_zero(gate_confidences)
        + 0.3 * gate_visit_support_max
    )
    gate_mean_gate_pass_score = (
        0.35 * gate_A_pass_ratio
        + 0.25 * gate_C_pass_ratio
        + 0.25 * gate_v_pass_ratio
        + 0.15 * gate_validated_ratio
    )
    gate_peak_gate_pass_score = 1.0 if any(gate_joint_pass_flags) else 0.0
    gate_mean_rectification_score = gate_changed_ratio * gate_mean_gate_pass_score
    gate_peak_rectification_score = gate_changed_ratio * max(gate_peak_gate_pass_score, gate_joint_pass_ratio)
    gate_mean_support_score = gate_ratio * (
        0.45 * gate_mean_evidence_score
        + 0.35 * gate_mean_gate_pass_score
        + 0.20 * gate_changed_ratio
    )
    gate_peak_support_score = min(
        1.0,
        gate_ratio * 0.5
        + 0.35 * gate_peak_evidence_score
        + 0.15 * gate_peak_rectification_score,
    )
    return {
        "selected": False,
        "selection_key": selection_key,
        "method": catalog_row["method"],
        "scenario": catalog_row["scenario"],
        "experiment_id": catalog_row["experiment_id"],
        "eval_name": catalog_row["eval_name"],
        "repeat": repeat_id,
        "episode_id": episode_id,
        "line_no": line_no,
        "result": result or ("Win" if win else "Loss"),
        "win": bool(win),
        "score": score,
        "terminal_hp_margin": _safe_float(
            (
                last_frame.get("hp_my") - last_frame.get("hp_enemy")
                if last_frame.get("hp_my") is not None and last_frame.get("hp_enemy") is not None
                else score
            ),
            score,
        ),
        "final_hp_delta": _safe_float(last_frame.get("hp_delta"), score),
        "final_hp_my": _safe_float(last_frame.get("hp_my"), math.nan),
        "final_hp_enemy": _safe_float(last_frame.get("hp_enemy"), math.nan),
        "final_game_loop": _safe_int(last_frame.get("game_loop"), 0),
        "frame_count": total_frames,
        "state_seq_len": len(state_seq),
        "unique_state_count": len({str(x) for x in state_seq}),
        "ood_frames": ood_frames,
        "ood_ratio": (ood_frames / total_frames) if total_frames else 0.0,
        "mechanism_changed_frames": changed_frames,
        "mechanism_changed_ratio": (changed_frames / total_frames) if total_frames else 0.0,
        "switch_frames": switch_frames,
        "tuning_frames": tuning_frames,
        "tuning_ratio": (tuning_frames / total_frames) if total_frames else 0.0,
        "gate_frames": len(gate_tunings),
        "gate_ratio": gate_ratio,
        "gate_A_mean": _mean_or_zero(gate_advantages),
        "gate_A_max": _max_or_zero(gate_advantages),
        "gate_A_positive_ratio": _ratio_or_zero(
            value > 0 for value in _finite_values(gate_advantages)
        ),
        "gate_A_pass_ratio": gate_A_pass_ratio,
        "gate_C_mean": _mean_or_zero(gate_confidences),
        "gate_C_max": _max_or_zero(gate_confidences),
        "gate_C_pass_ratio": gate_C_pass_ratio,
        "gate_v_mean": _mean_or_zero(gate_visits),
        "gate_v_max": _max_or_zero(gate_visits),
        "gate_v_support_mean": gate_visit_support_mean,
        "gate_v_support_max": gate_visit_support_max,
        "gate_supported_frames": int(sum(1 for flag in gate_supported_flags if flag)),
        "gate_supported_ratio": gate_v_pass_ratio,
        "gate_validated_frames": int(sum(1 for flag in gate_validation_flags if flag)),
        "gate_validated_ratio": gate_validated_ratio,
        "gate_joint_pass_ratio": gate_joint_pass_ratio,
        "gate_changed_ratio": gate_changed_ratio,
        "gate_mean_evidence_score": gate_mean_evidence_score,
        "gate_peak_evidence_score": gate_peak_evidence_score,
        "gate_mean_gate_pass_score": gate_mean_gate_pass_score,
        "gate_peak_gate_pass_score": gate_peak_gate_pass_score,
        "gate_mean_rectification_score": gate_mean_rectification_score,
        "gate_peak_rectification_score": gate_peak_rectification_score,
        "gate_mean_support_score": gate_mean_support_score,
        "gate_peak_support_score": gate_peak_support_score,
        "avg_decision_ms": sum(plan_times) / len(plan_times) if plan_times else math.nan,
        "first_state_key": str(first_frame.get("state_key", "")),
        "last_state_key": str(last_frame.get("state_key", "")),
        "eval_dir": str(eval_dir),
    }


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def _load_episode_table(eval_dir_text: str, catalog_payload: Dict[str, Any]) -> pd.DataFrame:
    eval_dir = Path(eval_dir_text)
    rows: List[Dict[str, Any]] = []
    for episode_file in sorted(eval_dir.glob("repeats/repeat_*/episodes.jsonl")):
        repeat_id = episode_file.parent.name
        with episode_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    _episode_record(
                        item,
                        repeat_id=repeat_id,
                        line_no=line_no,
                        eval_dir=eval_dir,
                        catalog_row=catalog_payload,
                    )
                )
    return pd.DataFrame(rows)


def _numeric_metrics(df: pd.DataFrame) -> List[str]:
    excluded = {"selected", "win", "line_no", "final_hp_delta", "terminal_hp_margin"}
    return [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]


def _summarize_episodes(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "score_mean": math.nan,
            "score_std": math.nan,
            "score_median": math.nan,
            "score_min": math.nan,
            "score_max": math.nan,
            "win_rate": math.nan,
            "avg_frame_count": math.nan,
            "avg_ood_ratio": math.nan,
            "avg_changed_ratio": math.nan,
            "avg_gate_A_max": math.nan,
            "avg_gate_C_max": math.nan,
            "avg_gate_v_max": math.nan,
            "avg_gate_mean_evidence_score": math.nan,
            "avg_gate_peak_evidence_score": math.nan,
            "avg_gate_mean_gate_pass_score": math.nan,
            "avg_gate_peak_gate_pass_score": math.nan,
            "avg_gate_mean_rectification_score": math.nan,
            "avg_gate_peak_rectification_score": math.nan,
            "avg_gate_mean_support_score": math.nan,
            "avg_gate_peak_support_score": math.nan,
            "avg_decision_ms": math.nan,
        }
    def mean_col(name: str) -> float:
        return float(df[name].mean(skipna=True)) if name in df.columns else math.nan

    return {
        "n": int(len(df)),
        "score_mean": float(df["score"].mean()),
        "score_std": float(df["score"].std(ddof=0)) if len(df) > 1 else 0.0,
        "score_median": float(df["score"].median()),
        "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()),
        "win_rate": float(df["win"].mean()),
        "avg_frame_count": float(df["frame_count"].mean()),
        "avg_ood_ratio": float(df["ood_ratio"].mean()),
        "avg_changed_ratio": float(df["mechanism_changed_ratio"].mean()),
        "avg_gate_A_max": mean_col("gate_A_max"),
        "avg_gate_C_max": mean_col("gate_C_max"),
        "avg_gate_v_max": mean_col("gate_v_max"),
        "avg_gate_mean_evidence_score": mean_col("gate_mean_evidence_score"),
        "avg_gate_peak_evidence_score": mean_col("gate_peak_evidence_score"),
        "avg_gate_mean_gate_pass_score": mean_col("gate_mean_gate_pass_score"),
        "avg_gate_peak_gate_pass_score": mean_col("gate_peak_gate_pass_score"),
        "avg_gate_mean_rectification_score": mean_col("gate_mean_rectification_score"),
        "avg_gate_peak_rectification_score": mean_col("gate_peak_rectification_score"),
        "avg_gate_mean_support_score": mean_col("gate_mean_support_score"),
        "avg_gate_peak_support_score": mean_col("gate_peak_support_score"),
        "avg_decision_ms": float(df["avg_decision_ms"].mean(skipna=True)),
    }


def _ensure_selection_dir(eval_dir: Path) -> Path:
    selection_dir = eval_dir / SELECTION_DIR_NAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    note = selection_dir / "DO_NOT_COMMIT.md"
    if not note.exists():
        note.write_text(
            "# DO NOT COMMIT\n\n"
            "本目录由 Web 端“数据筛选”tab 生成，仅用于本地人工筛选复评 episode。\n"
            "这些筛选记录属于临时/本地分析产物，不需要上传到 git。\n",
            encoding="utf-8",
        )
    return selection_dir


def _save_selection(
    *,
    eval_dir: Path,
    selection_name: str,
    selected_df: pd.DataFrame,
    full_df: pd.DataFrame,
    catalog_row: Dict[str, Any],
) -> Tuple[Path, Path]:
    selection_dir = _ensure_selection_dir(eval_dir)
    slug = _sanitize_filename(selection_name)
    json_path = selection_dir / f"{slug}.json"
    csv_path = selection_dir / f"{slug}.csv"
    selected_keys = selected_df["selection_key"].astype(str).tolist()
    payload = {
        "local_only": True,
        "do_not_commit": True,
        "note": LOCAL_ONLY_NOTE,
        "selection_name": selection_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": catalog_row.get("method"),
        "scenario": catalog_row.get("scenario"),
        "experiment_id": catalog_row.get("experiment_id"),
        "eval_name": catalog_row.get("eval_name"),
        "eval_dir": str(eval_dir),
        "selected_count": int(len(selected_df)),
        "total_count": int(len(full_df)),
        "selected_episode_keys": selected_keys,
        "selected_episode_ids": selected_df[["repeat", "episode_id", "line_no", "selection_key"]].to_dict("records"),
        "summary": _summarize_episodes(selected_df),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected_df.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return json_path, csv_path


def _load_selection_keys(eval_dir: Path, selection_file: Optional[str]) -> List[str]:
    if not selection_file:
        return []
    payload = _safe_read_json(eval_dir / SELECTION_DIR_NAME / selection_file)
    return [str(x) for x in payload.get("selected_episode_keys", [])]


def _available_selection_files(eval_dir: Path) -> List[str]:
    selection_dir = eval_dir / SELECTION_DIR_NAME
    if not selection_dir.exists():
        return []
    return sorted([p.name for p in selection_dir.glob("*.json")])


def _render_episode_view(df: pd.DataFrame, metric_options: List[str]) -> pd.DataFrame:
    gate_contribution_metrics = [
        metric
        for metric in [
            "gate_mean_evidence_score",
            "gate_peak_evidence_score",
            "gate_mean_gate_pass_score",
            "gate_peak_gate_pass_score",
            "gate_mean_rectification_score",
            "gate_peak_rectification_score",
            "gate_mean_support_score",
            "gate_peak_support_score",
        ]
        if metric in df.columns
    ]
    contribution_metric = None
    if gate_contribution_metrics:
        contribution_metric = st.selectbox(
            "相关性贡献指标",
            gate_contribution_metrics,
            index=gate_contribution_metrics.index("gate_peak_support_score")
            if "gate_peak_support_score" in gate_contribution_metrics
            else 0,
            help=(
                "计算每个 episode 对 corr(该门控指标, score) 的贡献。"
                "正值支持“机制证据越强、得分越高”的论文结论，负值削弱该结论。"
            ),
        )
        score_z = pd.to_numeric(df["score"], errors="coerce")
        metric_z = pd.to_numeric(df[contribution_metric], errors="coerce")
        score_std = score_z.std(ddof=0)
        metric_std = metric_z.std(ddof=0)
        if pd.notna(score_std) and pd.notna(metric_std) and score_std > 1e-12 and metric_std > 1e-12:
            df = df.copy()
            df["corr_contribution"] = (
                ((score_z - score_z.mean()) / score_std)
                * ((metric_z - metric_z.mean()) / metric_std)
                / max(len(df), 1)
            )
        else:
            df = df.copy()
            df["corr_contribution"] = 0.0
        df["corr_contribution_label"] = df["corr_contribution"].map(
            lambda value: "有利" if value > 0 else ("不利" if value < 0 else "中性")
        )

    view_mode = st.radio("对局视图", ["表格勾选", "多维折线图"], horizontal=True, key="manual_filter_view")
    if view_mode == "多维折线图":
        st.caption(
            "`score` 是 episode 顶层最终分数，当前复评通常等于终局我方总血量减敌方总血量；"
            "`final_hp_delta` 是最后一步逐帧血量变化奖励，`terminal_hp_margin` 与 score 重叠，二者均不作为默认多维指标。"
        )
        st.caption(
            "Gated Exploration 指标中，`gate_A_*` 表示局部采样动作相对图推荐动作的经验优势，"
            "`gate_C_*` 表示置信度，`gate_v_*` 表示候选动作访问次数。"
            "`mean` 级评分反映整局持续支撑，`peak` 级评分反映是否出现强证据切换点；这些评分不包含对局得分。"
        )
        metrics = st.multiselect(
            "折线图指标",
            metric_options,
            default=[
                m
                for m in [
                    "score",
                    "gate_mean_support_score",
                    "gate_peak_support_score",
                    "gate_mean_rectification_score",
                    "gate_peak_rectification_score",
                    "ood_ratio",
                ]
                if m in metric_options
            ][:5],
        )
        if metrics:
            plot_df = df.copy()
            plot_df["episode_order"] = range(1, len(plot_df) + 1)
            scale_mode = st.radio(
                "折线图量纲",
                ["原始值双轴", "0-1归一化"],
                horizontal=True,
                help="比例类指标默认放右轴；归一化适合比较趋势，不适合读取原始数值。",
            )
            if scale_mode == "0-1归一化":
                norm_df = plot_df.copy()
                for metric in metrics:
                    values = pd.to_numeric(norm_df[metric], errors="coerce")
                    lo = values.min(skipna=True)
                    hi = values.max(skipna=True)
                    if pd.isna(lo) or pd.isna(hi):
                        norm_df[metric] = float("nan")
                    elif abs(float(hi) - float(lo)) < 1e-12:
                        norm_df[metric] = 0.5
                    else:
                        norm_df[metric] = (values - lo) / (hi - lo)
                long_df = norm_df.melt(
                    id_vars=["episode_order", "episode_id", "repeat", "selection_key", "selected"],
                    value_vars=metrics,
                    var_name="metric",
                    value_name="value",
                )
                fig = px.line(
                    long_df,
                    x="episode_order",
                    y="value",
                    color="metric",
                    markers=True,
                    hover_data=["episode_id", "repeat", "selection_key"],
                )
                fig.update_yaxes(title_text="Normalized value")
            else:
                secondary_defaults = [
                    metric
                    for metric in metrics
                    if any(token in metric.lower() for token in ["ratio", "rate", "win"])
                ]
                secondary_metrics = st.multiselect(
                    "右轴指标",
                    metrics,
                    default=secondary_defaults,
                    help="建议将 OOD ratio、机制触发比例等 0-1 指标放右轴。",
                )
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                for idx, metric in enumerate(metrics):
                    axis_name = "y2" if metric in secondary_metrics else "y"
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df["episode_order"],
                            y=pd.to_numeric(plot_df[metric], errors="coerce"),
                            mode="lines+markers",
                            name=metric,
                            yaxis=axis_name,
                            customdata=plot_df[["episode_id", "repeat", "selection_key"]].astype(str),
                            hovertemplate=(
                                "episode_order=%{x}<br>"
                                "value=%{y}<br>"
                                "episode_id=%{customdata[0]}<br>"
                                "repeat=%{customdata[1]}<br>"
                                "selection_key=%{customdata[2]}<extra></extra>"
                            ),
                            line=dict(color=colors[idx % len(colors)]),
                        )
                    )
                fig.update_layout(
                    xaxis=dict(title="Episode order"),
                    yaxis=dict(title="Score / HP / count metrics"),
                    yaxis2=dict(
                        title="Ratio / rate metrics",
                        overlaying="y",
                        side="right",
                        range=[0, 1],
                        showgrid=False,
                    ),
                )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("折线图用于观察趋势，实际筛选仍在下方表格中勾选并保存。")

    default_columns = [
        "selected",
        "episode_id",
        "repeat",
        "result",
        "score",
        "corr_contribution",
        "corr_contribution_label",
        "gate_mean_support_score",
        "gate_peak_support_score",
        "gate_mean_rectification_score",
        "gate_peak_rectification_score",
        "gate_mean_evidence_score",
        "gate_peak_evidence_score",
        "gate_A_max",
        "gate_C_max",
        "gate_v_max",
        "frame_count",
        "ood_ratio",
        "mechanism_changed_ratio",
        "switch_frames",
        "avg_decision_ms",
        "selection_key",
    ]
    columns = [c for c in default_columns if c in df.columns]
    edited = st.data_editor(
        df[columns].copy(),
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "selected": st.column_config.CheckboxColumn("选择", help="勾选后可保存为人工筛选集合"),
            "selection_key": st.column_config.TextColumn("selection_key", disabled=True),
        },
        disabled=[c for c in columns if c != "selected"],
        key="manual_episode_editor",
    )
    merged = df.copy()
    selection_map = edited.set_index("selection_key")["selected"].to_dict()
    merged["selected"] = merged["selection_key"].map(selection_map).fillna(False).astype(bool)
    if "corr_contribution" in merged.columns:
        preview_cols = [
            c
            for c in [
                "episode_id",
                "score",
                contribution_metric,
                "corr_contribution",
                "corr_contribution_label",
                "selected",
            ]
            if c and c in merged.columns
        ]
        st.caption("相关性贡献预览：绿色表示选择该 episode 有利于正相关结论，红色表示不利。")
        st.dataframe(
            merged[preview_cols]
            .sort_values("corr_contribution", ascending=False)
            .style.applymap(
                lambda value: (
                    "background-color: #d9f0d3; color: #006d2c"
                    if isinstance(value, (int, float)) and value > 0
                    else (
                        "background-color: #fee0d2; color: #a50f15"
                        if isinstance(value, (int, float)) and value < 0
                        else ""
                    )
                ),
                subset=["corr_contribution"],
            ),
            use_container_width=True,
            height=260,
        )
    return merged


def _load_selection_payload(selection_path: Path) -> Dict[str, Any]:
    payload = _safe_read_json(selection_path)
    payload["_selection_path"] = str(selection_path)
    payload["_selection_name"] = selection_path.stem
    return payload


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def _collect_dashboard_rows(include_unfinished: bool = False) -> pd.DataFrame:
    catalog = _scan_eval_catalog(include_unfinished=include_unfinished)
    rows: List[Dict[str, Any]] = []
    for _, eval_row in catalog.iterrows():
        row_payload = eval_row.to_dict()
        eval_dir = Path(row_payload["eval_dir"])
        episodes = _load_episode_table(str(eval_dir), row_payload)
        if episodes.empty:
            continue
        for rule_name, subset in [
            ("all", episodes),
            ("score_top_100", episodes.sort_values("score", ascending=False).head(100)),
        ]:
            summary = _summarize_episodes(subset)
            rows.append(
                {
                    **{k: row_payload.get(k) for k in ["method", "scenario", "experiment_id", "eval_name", "batch_tag", "eval_dir"]},
                    "rule": rule_name,
                    **summary,
                }
            )
        for selection_file in _available_selection_files(eval_dir):
            payload = _load_selection_payload(eval_dir / SELECTION_DIR_NAME / selection_file)
            keys = set(str(x) for x in payload.get("selected_episode_keys", []))
            subset = episodes[episodes["selection_key"].astype(str).isin(keys)]
            summary = _summarize_episodes(subset)
            rows.append(
                {
                    **{k: row_payload.get(k) for k in ["method", "scenario", "experiment_id", "eval_name", "batch_tag", "eval_dir"]},
                    "rule": f"manual:{payload.get('selection_name') or Path(selection_file).stem}",
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _render_dashboard(include_unfinished: bool) -> None:
    st.caption("展板汇总 all、score_top_100 与人工 manual 规则，用于快速比较各场景、方法和筛选规则。")
    dashboard = _collect_dashboard_rows(include_unfinished=include_unfinished)
    if dashboard.empty:
        st.info("尚未发现可汇总的复评数据或人工筛选记录。")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_filter = st.multiselect("展板场景", sorted(dashboard["scenario"].dropna().unique()))
    with c2:
        method_filter = st.multiselect("展板方法", sorted(dashboard["method"].dropna().unique()))
    with c3:
        rule_filter = st.multiselect("筛选规则", sorted(dashboard["rule"].dropna().unique()), default=["all", "score_top_100"])

    filtered = dashboard.copy()
    if scenario_filter:
        filtered = filtered[filtered["scenario"].isin(scenario_filter)]
    if method_filter:
        filtered = filtered[filtered["method"].isin(method_filter)]
    if rule_filter:
        filtered = filtered[filtered["rule"].isin(rule_filter)]

    metrics = [
        "n",
        "score_mean",
        "score_std",
        "score_median",
        "win_rate",
        "avg_frame_count",
        "avg_ood_ratio",
        "avg_changed_ratio",
        "avg_gate_A_max",
        "avg_gate_C_max",
        "avg_gate_v_max",
        "avg_gate_mean_evidence_score",
        "avg_gate_peak_evidence_score",
        "avg_gate_mean_gate_pass_score",
        "avg_gate_peak_gate_pass_score",
        "avg_gate_mean_rectification_score",
        "avg_gate_peak_rectification_score",
        "avg_gate_mean_support_score",
        "avg_gate_peak_support_score",
        "avg_decision_ms",
    ]
    metric = st.selectbox("展板指标", [m for m in metrics if m in filtered.columns], index=1)
    mode = st.radio("展板视图", ["统计图", "表格"], horizontal=True, key="manual_dashboard_view")
    if mode == "统计图":
        if filtered.empty:
            st.warning("当前筛选条件下没有数据。")
        else:
            plot_df = filtered.copy()
            plot_df["group"] = plot_df["method"] + " | " + plot_df["rule"]
            fig = px.bar(
                plot_df,
                x="scenario",
                y=metric,
                color="group",
                barmode="group",
                hover_data=["experiment_id", "eval_name", "n", "score_mean", "win_rate"],
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        display_cols = [
            "method",
            "scenario",
            "rule",
            "n",
            "score_mean",
            "score_std",
            "score_median",
            "win_rate",
            "avg_ood_ratio",
            "avg_changed_ratio",
            "avg_gate_A_max",
            "avg_gate_C_max",
            "avg_gate_v_max",
            "avg_gate_mean_evidence_score",
            "avg_gate_peak_evidence_score",
            "avg_gate_mean_gate_pass_score",
            "avg_gate_peak_gate_pass_score",
            "avg_gate_mean_rectification_score",
            "avg_gate_peak_rectification_score",
            "avg_gate_mean_support_score",
            "avg_gate_peak_support_score",
            "avg_decision_ms",
            "experiment_id",
            "eval_name",
            "batch_tag",
        ]
        st.dataframe(filtered[[c for c in display_cols if c in filtered.columns]], use_container_width=True)


def _render_data_filter_tab() -> None:
    st.warning(f"本地功能，不需要上传到 git。{LOCAL_ONLY_NOTE}")
    st.caption(
        "仅扫描归档复评输出并写入 manual_episode_selections，不启动游戏、不停止进程、"
        "不修改已有 episodes.jsonl 或 final_eval_summary.json。"
    )

    include_unfinished = st.toggle(
        "包含可能仍在写入的复评目录",
        value=False,
        help="默认关闭，以避免读取正在跑的批量复评中尚未完成的目录。",
    )
    if st.button("刷新复评目录缓存", use_container_width=False):
        _scan_eval_catalog.clear()
        _load_episode_table.clear()
        _collect_dashboard_rows.clear()
        st.rerun()

    catalog = _scan_eval_catalog(include_unfinished=include_unfinished)
    if catalog.empty:
        st.info(f"未在 {_short_path(ALL_DATA_ROOT)} 下找到可用 etg-only 或 Synergy 复评数据。")
        return

    st.subheader("人工筛选对局")
    batch_order = (
        catalog.groupby("batch_tag")["mtime"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    batch_options = ["全部批次"] + batch_order
    default_batch_index = 1 if batch_order else 0
    c0, c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.5, 1.7])
    with c0:
        batch_filter = st.selectbox(
            "批次过滤",
            batch_options,
            index=default_batch_index,
            help="默认显示最新复评批次。若找不到旧批次，请切换为“全部批次”。",
        )
    filtered_catalog = catalog if batch_filter == "全部批次" else catalog[catalog["batch_tag"] == batch_filter]
    with c1:
        methods = st.multiselect(
            "方法",
            sorted(filtered_catalog["method"].unique()),
            default=sorted(filtered_catalog["method"].unique()),
        )
    filtered_catalog = filtered_catalog[filtered_catalog["method"].isin(methods)] if methods else filtered_catalog
    if filtered_catalog.empty:
        st.warning("当前批次/方法过滤条件下没有复评目录。")
        return
    with c2:
        scenarios = sorted(filtered_catalog["scenario"].dropna().unique())
        scenario = st.selectbox("场景", scenarios)
    filtered_catalog = filtered_catalog[filtered_catalog["scenario"] == scenario]
    if filtered_catalog.empty:
        st.warning("当前场景过滤条件下没有复评目录。")
        return
    with c3:
        experiment_labels = (
            filtered_catalog.assign(label=lambda d: d["experiment_id"] + " | " + d["method"])
            .sort_values(["method", "experiment_id"])["label"]
            .unique()
            .tolist()
        )
        experiment_label = st.selectbox("归档实验", experiment_labels)
    experiment_id, method_label = [x.strip() for x in experiment_label.split("|", 1)]
    filtered_catalog = filtered_catalog[
        (filtered_catalog["experiment_id"] == experiment_id) & (filtered_catalog["method"] == method_label)
    ]
    with c4:
        eval_labels = (
            filtered_catalog.assign(label=lambda d: d["eval_name"] + " | " + d["batch_tag"] + " | " + d["mtime"])
            .sort_values("mtime", ascending=False)["label"]
            .tolist()
        )
        eval_label = st.selectbox("复评批次", eval_labels)
    eval_name = eval_label.split("|", 1)[0].strip()
    eval_row = filtered_catalog[filtered_catalog["eval_name"] == eval_name].iloc[0].to_dict()
    eval_dir = Path(eval_row["eval_dir"])

    st.info(f"当前复评目录：`{_short_path(eval_dir)}`")
    episode_df = _load_episode_table(str(eval_dir), eval_row)
    if episode_df.empty:
        st.warning("该复评目录下未读取到 episode 记录。")
        return

    selection_files = ["不加载"] + _available_selection_files(eval_dir)
    selected_file = st.selectbox("加载已有人工筛选", selection_files)
    loaded_keys = set(_load_selection_keys(eval_dir, None if selected_file == "不加载" else selected_file))
    if loaded_keys:
        episode_df["selected"] = episode_df["selection_key"].astype(str).isin(loaded_keys)

    metric_options = _numeric_metrics(episode_df)
    edited_df = _render_episode_view(episode_df, metric_options)
    selected_df = edited_df[edited_df["selected"]].copy()
    summary = _summarize_episodes(selected_df)
    all_summary = _summarize_episodes(edited_df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("已选 episode", f"{summary['n']} / {all_summary['n']}")
    m2.metric("已选均分", f"{summary['score_mean']:.3f}" if summary["n"] else "-")
    m3.metric("已选胜率", f"{summary['win_rate']:.2%}" if summary["n"] else "-")
    m4.metric("已选 OOD 比例", f"{summary['avg_ood_ratio']:.2%}" if summary["n"] else "-")

    save_col, note_col = st.columns([1, 2])
    with save_col:
        selection_name = st.text_input(
            "筛选记录名",
            value=f"{scenario}_{method_label}_{eval_name}_manual",
            help="将保存为 JSON 和 CSV，文件带 do_not_commit 标记。",
        )
        if st.button("保存当前勾选", type="primary", disabled=selected_df.empty):
            json_path, csv_path = _save_selection(
                eval_dir=eval_dir,
                selection_name=selection_name,
                selected_df=selected_df,
                full_df=edited_df,
                catalog_row=eval_row,
            )
            _collect_dashboard_rows.clear()
            st.success(f"已保存：`{_short_path(json_path)}` 和 `{_short_path(csv_path)}`")
    with note_col:
        st.markdown(
            "- 保存位置固定为当前复评目录下的 `manual_episode_selections/`。\n"
            "- 该目录会自动生成 `DO_NOT_COMMIT.md`，用于提示不要提交人工筛选产物。\n"
            "- 手动筛选不会覆盖 `all`、`score_top_100` 或原始复评数据。"
        )

    st.divider()
    st.subheader("筛选数据展板")
    load_dashboard = st.toggle(
        "加载全局展板",
        value=False,
        help="展板会读取各复评目录的 episode 文件。默认关闭，避免影响正在运行的批量复评。",
    )
    if load_dashboard:
        _render_dashboard(include_unfinished=include_unfinished)
    else:
        st.info("已就绪。打开“加载全局展板”后，可汇总 all、score_top_100 与人工筛选规则。")
