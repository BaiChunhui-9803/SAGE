"""Prepare or run a fresh Graph-only / Full SAGE evaluation with archived settings.

Preparation never starts SC2. Pass --run explicitly in the separate SC2 runtime.
The frozen evaluation-start model is copied before any online updates.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_assets.aiide26_sage.evidence import PACK, SCENARIOS, read_json


def check_runtime_inputs(scenario, model=None):
    """Reject missing/pointer inputs before the permissive historical game loader."""
    map_id = SCENARIOS[scenario]
    graph = ROOT / "cache/experience_transition_graph" / (map_id + "_augmented")
    data = ROOT / "data" / map_id / "augmented_1"
    map_names = {
        "sce-1": "local_enemy_test_1", "sce-1m": "local_enemy_test_1_mirror",
        "sce-2": "MarineMicro_MvsM_4_dist", "sce-2m": "MarineMicro_MvsM_4_dist_mirror",
        "sce-3": "MarineMicro_MvsM_8_far", "sce-3m": "MarineMicro_MvsM_8_far_mirror",
    }
    required = [graph / "etg_simple.pkl", graph / "etg_simple_transitions.pkl",
        graph / ("sparse_neighbors.pkl" if scenario.startswith("sce-3") else "state_distance_matrix.npz"),
        data / "graph/state_node.txt", data / "bktree/primary_bktree.json",
        ROOT / "assets/maps" / (map_names[scenario] + ".SC2Map")]
    if model is not None:
        required.append(model)

    def require_file(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required SAGE runtime input is missing or empty: {path}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Git LFS pointer instead of asset bytes: {path}. Run git lfs pull.")

    for path in required:
        require_file(path)
    primary = read_json(data / "bktree/primary_bktree.json")
    pending = [primary]
    while pending:
        node = pending.pop()
        require_file(data / "bktree" / f"secondary_bktree_{int(node['cluster_id'])}.json")
        pending.extend(node.get("children", {}).values())


def normalize_source_names(params):
    """The upstream KG→ETG rename changed source tags, not routing semantics."""
    aliases = {"kg_plan": "etg_plan", "kg_follow": "etg_follow", "kg_relaxed": "etg_relaxed"}
    result = copy.deepcopy(params)
    for key in ["tuning_etg_protected_sources", "tuning_validation_sources", "tuning_explore_sources"]:
        if key in result:
            result[key] = [aliases.get(value, value) for value in result[key]]
    if "tuning_validation_profiles" in result:
        result["tuning_validation_profiles"] = {
            aliases.get(key, key): value for key, value in result["tuning_validation_profiles"].items()}
    return result


def prepare(scenario, method, output, episodes=300):
    output = Path(output).resolve()
    # Restrict generated runs to the disposable output tree, even when called as a library.
    if ROOT / "output" not in output.parents:
        raise ValueError("Fresh evaluations must be written below this repository's output/ directory")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    candidates = []
    for path in (PACK / "raw/evaluations").glob("*/final_eval_summary.json"):
        meta = read_json(path)
        if meta["map_key"] == scenario and bool(meta["action_tuning_enabled"]) == (method == "full"):
            candidates.append((path, meta))
    if len(candidates) != 1:
        raise ValueError(f"Expected one frozen evaluation for {scenario}/{method}")
    path, meta = candidates[0]
    check_runtime_inputs(scenario, path.parent / "action_tuning_model.pkl" if method == "full" else None)
    params = normalize_source_names(meta["repeats"][0]["params"])
    output.mkdir(parents=True, exist_ok=False)
    params.update(local_result_dir=str(output), plan_log_path=str(output / "plan.log"), target_episodes=episodes)
    if method == "full":
        shutil.copy2(path.parent / "action_tuning_model.pkl", output / "action_tuning_model.pkl")
        params["action_tuning_model_path"] = str(output / "action_tuning_model.pkl")
    else:
        params.pop("action_tuning_model_path", None)
    (output / "startup_beam_params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(ROOT / "scripts/run_live_game.py"), "--mode", "game",
               "--map_key", scenario, "--etg_file", f"{SCENARIOS[scenario]}_augmented/etg_simple.pkl",
               "--data_dir", str(ROOT / "data" / SCENARIOS[scenario] / "augmented_1"),
               "--autopilot_mode", params["mode"], "--beam_params_file", str(output / "startup_beam_params.json"),
               "--max_episodes", str(episodes)]
    (output / "run_manifest.json").write_text(json.dumps({
        "source_evaluation": path.parent.name, "scenario": scenario, "method": method,
        "command": command, "changes": ["Relocated output/model paths", "Normalized kg_* source tags to etg_*",
            "Episode count set from CLI; all other recorded startup parameters preserved"],
        "scope": "Fresh SC2 evaluation; does not recreate the historical selected 100 episodes or seed sequence",
    }, indent=2) + "\n", encoding="utf-8")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="sce-1")
    parser.add_argument("--method", choices=["graph-only", "full"], default="graph-only")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true", help="Launch SC2 after preparing the run")
    args = parser.parse_args()
    output = args.output or ROOT / "output/paper_live" / f"{args.scenario}_{args.method}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    command = prepare(args.scenario, args.method, output, args.episodes)
    print(json.dumps({"output": str(output), "command": command, "started": args.run}, indent=2))
    if not args.run:
        return 0
    code = subprocess.call(command, cwd=ROOT)
    progress_path = output / "progress.json"
    progress = read_json(progress_path) if progress_path.is_file() else {}
    episode_path = output / "episodes.jsonl"
    completed = 0
    if episode_path.is_file():
        with episode_path.open(encoding="utf-8-sig") as stream:
            completed = sum(bool(line.strip()) for line in stream)
    passed = code == 0 and completed == args.episodes and progress.get("completed") == args.episodes
    (output / "completion.json").write_text(json.dumps({
        "passed": passed, "process_exit_code": code, "requested_episodes": args.episodes,
        "recorded_episodes": completed, "progress_completed": progress.get("completed"),
    }, indent=2) + "\n", encoding="utf-8")
    if not passed:
        print("Evaluation did not complete the requested episodes; inspect completion.json and the run log.", file=sys.stderr)
    return 0 if passed else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
