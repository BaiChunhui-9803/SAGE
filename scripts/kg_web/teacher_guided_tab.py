import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src import ROOT_DIR
from src.adapters.pymarl_action_projector import project_frame, summarize_projection_results


_ALL_DATA_ROOT = ROOT_DIR / "output" / "learner_results" / "all_data"
_TEACHER_SCRIPT = Path("scripts") / "build_teacher_guided_etg.py"
_TEACHER_WORK_ROOT = ROOT_DIR / "output" / "teacher_guided_etg"

_TEACHER_SCENARIOS = [
    {
        "scenario": "sce3_mvs8",
        "map_key": "sce-3",
        "map_id": "MarineMicro_MvsM_8",
    },
    {
        "scenario": "sce3_mvs8_mirror",
        "map_key": "sce-3m",
        "map_id": "MarineMicro_MvsM_8_mirror",
    },
]


def _quote_cli_arg(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if re.search(r"[\s`\"']", text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _scenario_label(item: Dict[str, Any]) -> str:
    return f"{item['map_key']} / {item['map_id']}"


def _sanitize(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value or "").strip()).strip("_")


def _default_teacher_jsonl(map_id: str, algorithm: str) -> str:
    return str(_TEACHER_WORK_ROOT / map_id / algorithm / "teacher_episodes.jsonl")


def _default_checkpoint_root(pymarl_root: str) -> str:
    return str(Path(pymarl_root) / "results" / "sacred")


def _scan_pymarl_runs(checkpoint_root: str, algorithm: str) -> List[Dict[str, Any]]:
    root = Path(checkpoint_root)
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        config_path = run_dir / "config.json"
        info_path = run_dir / "info.json"
        config = _read_json(config_path) or {}
        info = _read_json(info_path) or {}
        alg = str(config.get("config", {}).get("name") or config.get("name") or "").lower()
        raw_config = config.get("config", {}) if isinstance(config.get("config"), dict) else config
        if algorithm and alg and algorithm.lower() not in alg:
            continue
        map_name = (
            raw_config.get("env_args", {}).get("map_name")
            if isinstance(raw_config.get("env_args"), dict)
            else None
        )
        model_dir = run_dir / "models"
        model_steps = []
        if model_dir.exists():
            model_steps = [p.name for p in model_dir.iterdir() if p.is_dir()]
        rows.append(
            {
                "run_id": run_dir.name,
                "algorithm": alg or algorithm,
                "map_name": map_name or "-",
                "models": ", ".join(model_steps[-3:]) if model_steps else "-",
                "checkpoint_path": str(model_dir),
                "modified": pd.Timestamp.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "test_final_score": _last_metric(info, "test_final_score_mean"),
                "test_win": _last_metric(info, "test_battle_won_mean"),
            }
        )
    return rows


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _last_metric(info: Dict[str, Any], key: str) -> Any:
    value = info.get(key)
    if isinstance(value, list) and value:
        return value[-1]
    return value if value is not None else "-"


def _preview_teacher_jsonl(path: str, limit_episodes: int = 5, max_frames: int = 500) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {"available": False, "reason": f"文件不存在：{file_path}"}
    projections = []
    episodes = 0
    frames = 0
    score_values = []
    with open(str(file_path), "r", encoding="utf-8") as f:
        for line in f:
            if episodes >= limit_episodes or frames >= max_frames:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except Exception:
                continue
            episodes += 1
            if ep.get("final_score") is not None:
                try:
                    score_values.append(float(ep["final_score"]))
                except Exception:
                    pass
            for frame in ep.get("frames", []) or []:
                if frames >= max_frames:
                    break
                if not isinstance(frame, dict):
                    continue
                try:
                    projections.append(project_frame(frame))
                    frames += 1
                except Exception:
                    continue
    summary = summarize_projection_results(projections)
    summary.update(
        {
            "available": True,
            "preview_episodes": episodes,
            "preview_frames": frames,
            "final_score_mean_preview": sum(score_values) / len(score_values) if score_values else None,
        }
    )
    return summary


def _build_export_cmd(
    pymarl_root: str,
    algorithm: str,
    scenario: Dict[str, Any],
    checkpoint_path: str,
    episodes: int,
    teacher_jsonl: str,
) -> List[str]:
    return [
        "python",
        str(Path("src") / "main.py"),
        f"--config={algorithm}",
        "--env-config=sc2",
        "with",
        f"env_args.map_name={scenario['map_id']}",
        f"checkpoint_path={checkpoint_path}",
        "load_step=0",
        "evaluate=True",
        f"test_nepisode={int(episodes)}",
        "export_teacher_trajectory=True",
        f"teacher_trajectory_path={teacher_jsonl}",
        "enable_bktree_logging=True",
    ]


def _build_etg_cmd(
    scenario: Dict[str, Any],
    algorithm: str,
    teacher_jsonl: str,
    checkpoint_path: str,
    source_run_id: str,
    merge_mode: str,
    teacher_weight: int,
    projection_threshold: float,
    top_quantile: float,
    max_teacher_episodes: int,
    overwrite: bool,
    validate: bool,
    preview_only: bool,
) -> List[str]:
    map_id = scenario["map_id"]
    data_id = f"teacher_{_sanitize(algorithm)}_projected_1"
    exp_id = f"{scenario['scenario']}_teacher_{_sanitize(algorithm)}"
    cmd = [
        "python",
        str(_TEACHER_SCRIPT),
        "--teacher-jsonl",
        teacher_jsonl,
        "--map-id",
        map_id,
        "--map-key",
        scenario["map_key"],
        "--teacher-method",
        algorithm,
        "--source-run-id",
        source_run_id,
        "--checkpoint-path",
        checkpoint_path,
        "--bktree-dir",
        str(Path("data") / map_id / "augmented_1" / "bktree"),
        "--base-replay-input",
        str(Path("data") / map_id / "augmented_1_collected" / "ep0-3999_r10_p1_s0.5"),
        "--merge-mode",
        merge_mode,
        "--top-replay-quantile",
        str(float(top_quantile)),
        "--teacher-weight",
        str(int(teacher_weight)),
        "--projection-threshold",
        str(float(projection_threshold)),
        "--max-teacher-episodes",
        str(int(max_teacher_episodes)),
        "--data-id",
        data_id,
        "--output-dir",
        str(Path("cache") / "knowledge_graph" / f"{map_id}_teacher_{_sanitize(algorithm)}_projected"),
        "--kg-name",
        f"{map_id} - Teacher {algorithm.upper()} Projected",
        "--experiment-id",
        exp_id,
        "--primary-threshold",
        "1.0",
        "--secondary-threshold",
        "0.5",
        "--skip-distance-matrix",
        "--sparse-top-k",
        "32",
        "--sparse-max-source-states",
        "30000",
        "--sparse-max-candidates-per-primary",
        "512",
        "--register-catalog",
        "--archive-manifest",
    ]
    if overwrite:
        cmd.append("--overwrite")
    if validate:
        cmd.append("--validate")
    if preview_only:
        cmd.append("--preview-only")
    return cmd


def _cmd_text(cmd: List[str], cwd: Optional[str] = None) -> str:
    prefix = f"Set-Location -LiteralPath {_quote_cli_arg(cwd)}; " if cwd else ""
    return prefix + " ".join(_quote_cli_arg(part) for part in cmd)


def _start_teacher_batch_job(specs: List[Dict[str, Any]], job_kind: str) -> Dict[str, Any]:
    log_dir = _ALL_DATA_ROOT / "Teacher-guided-ETG" / "_launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"teacher_{job_kind}_{timestamp}"
    payload_path = log_dir / f"{batch_id}.json"
    log_path = log_dir / f"{batch_id}.log"
    with open(str(payload_path), "w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)

    launcher = (
        "import json, os, subprocess, sys\n"
        "payload_path = sys.argv[1]\n"
        "with open(payload_path, 'r', encoding='utf-8') as f:\n"
        "    specs = json.load(f)\n"
        "env = dict(os.environ)\n"
        "env['PYTHONIOENCODING'] = 'utf-8'\n"
        "for idx, spec in enumerate(specs, start=1):\n"
        "    cmd = list(spec['cmd'])\n"
        "    if cmd and cmd[0] == 'python':\n"
        "        cmd[0] = sys.executable\n"
        "    cwd = spec.get('cwd') or os.getcwd()\n"
        "    print(f\"[TEACHER] {idx}/{len(specs)} START {spec['name']}\", flush=True)\n"
        "    rc = subprocess.run(cmd, cwd=cwd, env=env).returncode\n"
        "    print(f\"[TEACHER] {idx}/{len(specs)} DONE {spec['name']} rc={rc}\", flush=True)\n"
    )
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log_file = open(str(log_path), "w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-c", launcher, str(payload_path)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR),
        creationflags=flags,
        env=env,
    )
    return {
        "pid": proc.pid,
        "batch_id": batch_id,
        "payload_path": str(payload_path),
        "log_path": str(log_path),
        "count": len(specs),
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def render_teacher_guided_etg_panel() -> None:
    st.subheader("Teacher-guided ETG 构建")
    st.caption(
        "用 PyMARL checkpoint 导出的逐步教师轨迹构建新的 projected-script ETG；默认只写入 teacher 专属目录，不覆盖已有 augmented ETG。"
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    pymarl_root = c1.text_input(
        "PyMARL 项目路径",
        value=r"D:\白春辉\实验平台\pymarl",
        key="teacher_pymarl_root",
        help="用于生成 checkpoint evaluation 导出命令；构建 ETG 本身只依赖 teacher_episodes.jsonl。",
    )
    algorithm = c2.selectbox("教师算法", ["qmix", "qtran", "vdn"], index=0, key="teacher_algorithm")
    episodes = c3.number_input("教师导出局数", min_value=1, max_value=5000, value=1000, step=20, key="teacher_export_episodes")

    checkpoint_root = st.text_input(
        "PyMARL sacred/checkpoint 根目录",
        value=_default_checkpoint_root(pymarl_root),
        key="teacher_checkpoint_root",
    )
    runs = _scan_pymarl_runs(checkpoint_root, algorithm)
    with st.expander("已发现的 PyMARL run/checkpoint", expanded=False):
        if runs:
            st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
        else:
            st.info("未扫描到匹配 run；仍可手动填写 checkpoint_path。")
    with st.expander("安装 PyMARL 教师轨迹导出钩子（可选）", expanded=False):
        install_cmd = [
            "python",
            str(Path("scripts") / "install_pymarl_teacher_export_hooks.py"),
            "--pymarl-root",
            pymarl_root,
            "--apply",
        ]
        st.caption("仅当外部 PyMARL 尚不支持 `export_teacher_trajectory` 时使用。该命令会修改 PyMARL 项目文件；建议先去掉 `--apply` 做 dry-run。")
        st.code(_cmd_text(install_cmd, str(ROOT_DIR)), language="powershell")

    d1, d2 = st.columns([2, 2])
    scenario_indices = d1.multiselect(
        "目标场景（可多选）",
        options=range(len(_TEACHER_SCENARIOS)),
        default=[0, 1],
        format_func=lambda idx: _scenario_label(_TEACHER_SCENARIOS[idx]),
        key="teacher_scenarios",
    )
    checkpoint_path = d2.text_input(
        "checkpoint_path（可手动指定）",
        value=r"D:\白春辉\实验平台\pymarl\results\sacred\CHANGE_RUN_ID\models",
        key="teacher_checkpoint_path",
        help="若批量场景使用不同 checkpoint，可先复制命令后逐条替换。",
    )
    source_run_id = st.text_input("source_run_id", value="CHANGE_RUN_ID", key="teacher_source_run_id")

    e1, e2, e3, e4 = st.columns(4)
    merge_mode = e1.selectbox(
        "构建数据模式",
        ["teacher_plus_top_replay", "teacher_plus_replay", "teacher_only"],
        index=0,
        key="teacher_merge_mode",
    )
    teacher_weight = e2.number_input("teacher weight", min_value=1, max_value=10, value=2, step=1, key="teacher_weight")
    top_quantile = e3.number_input("replay top比例", min_value=0.01, max_value=1.0, value=0.25, step=0.05, key="teacher_top_quantile")
    projection_threshold = e4.number_input("投影置信阈值", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="teacher_projection_threshold")

    f1, f2, f3 = st.columns(3)
    max_teacher_episodes = f1.number_input(
        "构建最多使用教师局数",
        min_value=0,
        max_value=5000,
        value=0,
        step=20,
        key="teacher_max_build_episodes",
        help="0 表示使用 teacher JSONL 中全部 episode。",
    )
    overwrite = f2.toggle("允许覆盖 teacher 产物", value=False, key="teacher_overwrite")
    validate = f3.toggle("构建后 validate KG", value=False, key="teacher_validate")

    if not scenario_indices:
        st.info("请选择至少一个目标场景。")
        return

    selected_scenarios = [_TEACHER_SCENARIOS[int(idx)] for idx in scenario_indices]
    export_specs = []
    build_specs = []
    preview_rows = []
    for scenario in selected_scenarios:
        teacher_jsonl = _default_teacher_jsonl(scenario["map_id"], algorithm)
        export_cmd = _build_export_cmd(
            pymarl_root,
            algorithm,
            scenario,
            checkpoint_path,
            int(episodes),
            teacher_jsonl,
        )
        build_cmd = _build_etg_cmd(
            scenario,
            algorithm,
            teacher_jsonl,
            checkpoint_path,
            source_run_id,
            merge_mode,
            int(teacher_weight),
            float(projection_threshold),
            float(top_quantile),
            int(max_teacher_episodes),
            bool(overwrite),
            bool(validate),
            preview_only=False,
        )
        preview_cmd = _build_etg_cmd(
            scenario,
            algorithm,
            teacher_jsonl,
            checkpoint_path,
            source_run_id,
            merge_mode,
            int(teacher_weight),
            float(projection_threshold),
            float(top_quantile),
            int(max_teacher_episodes),
            bool(overwrite),
            bool(validate),
            preview_only=True,
        )
        export_specs.append({"name": f"export_{scenario['scenario']}_{algorithm}", "cmd": export_cmd, "cwd": pymarl_root})
        build_specs.append({"name": f"build_{scenario['scenario']}_{algorithm}", "cmd": build_cmd, "cwd": str(ROOT_DIR)})
        preview = _preview_teacher_jsonl(teacher_jsonl)
        preview_rows.append(
            {
                "场景": scenario["map_key"],
                "teacher_jsonl": teacher_jsonl,
                "文件状态": "存在" if preview.get("available") else "缺失",
                "预览episode": preview.get("preview_episodes", "-"),
                "预览frame": preview.get("preview_frames", "-"),
                "平均置信": preview.get("mean_confidence", "-"),
                "低置信比例": preview.get("low_confidence_ratio", "-"),
                "预览命令": _cmd_text(preview_cmd, str(ROOT_DIR)),
            }
        )

    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    with st.expander("PyMARL 教师轨迹导出命令（需要 PyMARL 支持 export_teacher_trajectory 参数）", expanded=True):
        st.code("\n".join(_cmd_text(spec["cmd"], spec["cwd"]) for spec in export_specs), language="powershell")
    with st.expander("Teacher-guided ETG 构建命令", expanded=True):
        st.code("\n".join(_cmd_text(spec["cmd"], spec["cwd"]) for spec in build_specs), language="powershell")

    b1, b2 = st.columns(2)
    if b1.button("后台启动 PyMARL 教师轨迹导出（串行）", key="start_teacher_export_batch", use_container_width=True):
        try:
            job = _start_teacher_batch_job(export_specs, "export")
            st.session_state.setdefault("teacher_guided_jobs", {})[job["batch_id"]] = job
            st.success(f"已启动教师轨迹导出 PID={job['pid']}，日志：`{job['log_path']}`")
        except Exception as exc:
            st.error(f"启动教师轨迹导出失败：{exc}")
    if b2.button("后台构建 Teacher-guided ETG（串行）", key="start_teacher_build_batch", use_container_width=True):
        try:
            job = _start_teacher_batch_job(build_specs, "build")
            st.session_state.setdefault("teacher_guided_jobs", {})[job["batch_id"]] = job
            st.success(f"已启动 Teacher-guided ETG 构建 PID={job['pid']}，日志：`{job['log_path']}`")
        except Exception as exc:
            st.error(f"启动 Teacher-guided ETG 构建失败：{exc}")

    jobs = st.session_state.get("teacher_guided_jobs", {})
    if jobs:
        with st.expander("Teacher-guided 后台任务", expanded=False):
            st.dataframe(pd.DataFrame(jobs.values()), use_container_width=True, hide_index=True)
            latest = list(jobs.values())[-1]
            log_path = Path(latest["log_path"])
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    st.code("\n".join(lines[-80:]))
                except Exception:
                    pass
