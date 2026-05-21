#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Archive a completed parameter-learner run into output/learner_results/all_data.

The archive contains the original run artifacts plus an experiment_manifest.json
so the batch experiment tab can discover and compare it without manual copying.
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR


_ALL_DATA_ROOT = ROOT_DIR / "output" / "learner_results" / "all_data"
_KG_CATALOG_PATH = ROOT_DIR / "configs" / "kg_catalog.yaml"
_MANIFEST_NAME = "experiment_manifest.json"


_MAP_PREFIX = {
    "sce-1": "sce1",
    "sce-1m": "sce1m",
    "sce-2": "sce2",
    "sce-2m": "sce2m",
    "sce-3": "sce3",
    "sce-3m": "sce3m",
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _kg_catalog_entries() -> list:
    data = _read_yaml(_KG_CATALOG_PATH)
    entries = data.get("knowledge_graphs", [])
    return entries if isinstance(entries, list) else []


def _find_catalog_entry(kg_file: str = "", data_dir: str = "") -> Optional[Dict[str, Any]]:
    kg_file_norm = str(kg_file or "").replace("\\", "/")
    data_dir_norm = str(data_dir or "").replace("\\", "/")
    for entry in _kg_catalog_entries():
        entry_file = str(entry.get("file", "")).replace("\\", "/")
        entry_data = str(entry.get("data_dir", "")).replace("\\", "/")
        if kg_file_norm and entry_file == kg_file_norm:
            return entry
        if data_dir_norm and entry_data == data_dir_norm:
            return entry
    return None


def _parse_log_config(run_dir: Path) -> Dict[str, Any]:
    log_path = run_dir / "learner.log"
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    result: Dict[str, Any] = {}
    patterns = {
        "kg_file": r"KG file:\s*(.+)",
        "data_dir": r"data dir:\s*(.+)",
        "action_tuning": r"action_tuning:\s*(enabled|disabled)",
        "run_dir": r"run dir:\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = match.group(1).strip()
    return result


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    for name in ("learner_config_snapshot.yaml", "learner_config_snapshot.yml"):
        cfg = _read_yaml(run_dir / name)
        if cfg:
            return cfg
    return {}


def _detect_method_group(cfg: Dict[str, Any], log_cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    phased = bool((cfg.get("phased_optimization") or {}).get("enabled", False))
    tuning = bool((cfg.get("action_tuning") or {}).get("enabled", False))
    if str(log_cfg.get("action_tuning", "")).lower() == "enabled":
        tuning = True
    if phased:
        return (
            "synergy",
            "synergy",
            "three_phase_synergy",
            "ETG + MCTS ActionTuning",
        )
    if tuning:
        return (
            "MC-only",
            "mc",
            "mc_only_action_tuning",
            "MCTS/UCB ActionTuning",
        )
    return (
        "ETG-only",
        "etg",
        "etg_only_baseline",
        "ETG-only Beam Search",
    )


def _scenario_prefix(map_key: str, map_id: str) -> str:
    if map_key in _MAP_PREFIX:
        return _MAP_PREFIX[map_key]
    compact = str(map_key or map_id or "scenario").lower()
    compact = compact.replace("marine_micro_", "").replace("marinemicro_", "")
    compact = re.sub(r"[^0-9a-zA-Z]+", "_", compact).strip("_")
    return compact or "scenario"


def _unique_destination(base_dir: Path, source_run: str, overwrite: bool = False) -> Path:
    if overwrite or not base_dir.exists():
        return base_dir
    existing_manifest = _read_json(base_dir / _MANIFEST_NAME)
    if existing_manifest.get("source_run") == source_run:
        return base_dir
    candidate = base_dir.with_name(f"{base_dir.name}_{source_run}")
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{source_run}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def build_manifest(
    run_dir: Path,
    cfg: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = cfg or _load_run_config(run_dir)
    log_cfg = _parse_log_config(run_dir)
    game_cfg = cfg.get("game") or {}
    bktree_cfg = cfg.get("bktree") or {}

    kg_file = str(game_cfg.get("kg_file") or log_cfg.get("kg_file") or "")
    data_dir = str(game_cfg.get("data_dir") or log_cfg.get("data_dir") or "")
    catalog_entry = _find_catalog_entry(kg_file, data_dir) or {}
    map_key = str(game_cfg.get("map_key") or catalog_entry.get("map_key") or "")
    map_id = str(catalog_entry.get("map_id") or "")
    if not map_id and data_dir:
        parts = Path(data_dir.replace("\\", "/")).parts
        if len(parts) >= 2 and parts[0] == "data":
            map_id = parts[1]
    if not map_key:
        map_key = {
            "MarineMicro_MvsM_4": "sce-1",
            "MarineMicro_MvsM_4_mirror": "sce-1m",
            "MarineMicro_MvsM_4_dist": "sce-2",
            "MarineMicro_MvsM_4_dist_mirror": "sce-2m",
            "MarineMicro_MvsM_8": "sce-3",
            "MarineMicro_MvsM_8_mirror": "sce-3m",
        }.get(map_id, "")

    group, suffix, experiment_type, method = _detect_method_group(cfg, log_cfg)
    scenario = _scenario_prefix(map_key, map_id)
    exp_id = experiment_id or f"{scenario}_{suffix}"
    data_type = str(catalog_entry.get("type") or "")
    data_id = str(catalog_entry.get("data_id") or "")
    replay_expansion = (
        data_type.lower() == "augmented"
        or data_id.startswith("augmented")
        or "augmented" in data_dir.replace("\\", "/").lower()
        or "augmented" in kg_file.replace("\\", "/").lower()
    )

    primary_threshold = bktree_cfg.get("primary_threshold")
    secondary_threshold = bktree_cfg.get("secondary_threshold")
    if primary_threshold is None:
        primary_threshold = 0.7 if map_id and "MvsM_4" in map_id else 1.0
    if secondary_threshold is None:
        secondary_threshold = 0.5

    source_run = run_dir.name if run_dir.name.startswith("run_") else ""
    if not source_run and log_cfg.get("run_dir"):
        source_run = Path(str(log_cfg["run_dir"])).name

    bktree_path = str(Path(data_dir) / "bktree") if data_dir else ""
    bktree_path = bktree_path.replace("\\", "/")

    manifest = {
        "experiment_id": exp_id,
        "display_name": f"{scenario.upper()} {method}",
        "map_key": map_key,
        "map_id": map_id,
        "experiment_type": experiment_type,
        "method": method,
        "kg_name": catalog_entry.get("name", ""),
        "kg_file": kg_file,
        "transitions": catalog_entry.get("transitions", ""),
        "data_dir": data_dir,
        "dataset_type": data_type or ("augmented" if replay_expansion else "unknown"),
        "replay_dataset_expansion": bool(replay_expansion),
        "bktree": {
            "primary_threshold": float(primary_threshold),
            "secondary_threshold": float(secondary_threshold),
            "path": bktree_path,
        },
        "source_run": source_run,
        "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": f"Automatically archived from {run_dir.name}.",
    }
    return manifest


def archive_run(
    run_dir: Path,
    archive_root: Path = _ALL_DATA_ROOT,
    overwrite: bool = False,
    cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    if not (run_dir / "study_summary.json").exists():
        raise FileNotFoundError(f"study_summary.json not found in run directory: {run_dir}")

    cfg = cfg or _load_run_config(run_dir)
    manifest = build_manifest(run_dir, cfg=cfg)
    group, suffix, _, _ = _detect_method_group(cfg, _parse_log_config(run_dir))
    base_dest = archive_root / group / manifest["experiment_id"]
    dest = _unique_destination(base_dest, manifest.get("source_run", run_dir.name), overwrite)
    if dest.name != manifest["experiment_id"]:
        manifest["experiment_id"] = dest.name
        manifest["display_name"] = f"{manifest['display_name']} ({manifest.get('source_run', run_dir.name)})"

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    shutil.copytree(str(run_dir), str(dest), dirs_exist_ok=True)
    _write_json(dest / _MANIFEST_NAME, manifest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive a completed learner run")
    parser.add_argument("--run-dir", required=True, help="training_runs/run_xxxx directory")
    parser.add_argument("--archive-root", default=str(_ALL_DATA_ROOT), help="all_data root")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing archive destination")
    parser.add_argument("--manifest-only", action="store_true", help="write manifest into run-dir without copying")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT_DIR / run_dir

    if args.manifest_only:
        manifest = build_manifest(run_dir)
        _write_json(run_dir / _MANIFEST_NAME, manifest)
        print(run_dir)
        return

    dest = archive_run(
        run_dir,
        archive_root=Path(args.archive_root),
        overwrite=bool(args.overwrite),
    )
    print(dest)


if __name__ == "__main__":
    main()
