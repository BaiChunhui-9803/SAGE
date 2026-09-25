"""Check all six distributed SAGE graphs/models and both recorded planners on CPU."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_assets.aiide26_sage.evidence import PACK, SCENARIOS, read_json
from src.decision.action_tuning_model import ActionTuningModel
from src.decision.experience_transition_graph import DecisionExperienceTransitionGraph, load_compatible_pickle
from src.decision.etg_beam_search import plan_action
from src.decision.chain_rollout import chain_rollout
from src.utils.distance_assets import load_distances


def verify():
    manifest = read_json(PACK / "provenance.json")
    records = []
    for scenario, map_id in SCENARIOS.items():
        folder = ROOT / "cache/experience_transition_graph" / f"{map_id}_augmented"
        graph = DecisionExperienceTransitionGraph.load(str(folder / "etg_simple.pkl"))
        with (folder / "etg_simple_transitions.pkl").open("rb") as stream:
            transitions = load_compatible_pickle(stream)
        distances = load_distances(map_id, "augmented_1")
        assert distances is not None, scenario
        sparse = bool(getattr(distances, "is_sparse_distance_index", False))
        if not sparse:
            # Verify the entire decompressed matrix against the original NPY hash.
            archive = (folder / "state_distance_matrix.npz").relative_to(ROOT).as_posix()
            expected = next(row["source_sha256"] for row in manifest["files"] if row["path"] == archive)
            digest = hashlib.sha256()
            with Path(distances.filename).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == expected, f"Changed dense distances: {scenario}"
        plans = []
        # State 0 is not necessarily the live observation's cluster. Optimized
        # min-visits/probability filters may reject it; use the most-supported
        # nonterminal source to exercise each recorded planner configuration.
        supported_state = max((sid for sid in graph.state_action_map
            if sid in transitions and not transitions[sid].get("__terminal__")),
            key=lambda sid: sum(stat.visits for stat in graph.state_action_map[sid].values()))
        for path in sorted((PACK / "raw/evaluations").glob("*/final_eval_summary.json")):
            meta = read_json(path)
            if meta["map_key"] != scenario:
                continue
            params = read_json(path.parent / "fixed_params.json")
            plan = plan_action(graph, transitions, supported_state,
                beam_width=params["beam_width"], max_steps=params["lookahead_steps"],
                min_visits=params["min_visits"], min_cum_prob=params["min_cum_prob"],
                score_mode=params["score_mode"], max_state_revisits=params["max_state_revisits"],
                discount_factor=params["discount_factor"], action_strategy=params["action_strategy"],
                epsilon=params.get("epsilon", .1), rng_seed=42)
            assert plan.recommended_action is not None, path.parent.name
            plans.append({"run": path.parent.name, "state": supported_state, "action": plan.recommended_action,
                          "beam_nodes": len(plan.beam_results)})
            if meta["action_tuning_enabled"]:
                model = ActionTuningModel.load(str(path.parent / "action_tuning_model.pkl"))
                assert model.state_action_stats, scenario
                tuning_states = len(model.state_action_stats)
                del model
        # Exercise graph transition/switch code separately from live SC2.
        rollout = chain_rollout(graph, transitions, supported_state, beam_width=3, lookahead_steps=5,
            max_rollout_steps=8, rng_seed=42, rollout_mode="multi_step", enable_backup=True,
            next_state_mode="highest_prob", min_cum_prob=1e-6, dist_matrix=distances)
        assert len(rollout.chosen_path_ids) > 1, scenario
        row = {"scenario": scenario, "distance_shape": distances.shape, "sparse": sparse,
               "tuning_states": tuning_states, "plans": plans,
               "rollout_steps": len(rollout.chosen_path_ids)-1,
               "rollout_settings": {"next_state_mode": "highest_prob", "min_cum_prob": 1e-6,
                                    "scope": "deterministic graph smoke check, not paper evaluation"}}
        records.append(row)
        print(json.dumps(row), flush=True)
        del graph, transitions, distances, plan, rollout
        gc.collect()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/artifact_audit/runtime_verification.json")
    args = parser.parse_args()
    report = {"passed": True, "scope": "CPU graphs, distances, recorded beam-search settings, graph rollouts and tuning-model loading; final live action masking/router and SC2 are not exercised", "scenarios": verify()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
