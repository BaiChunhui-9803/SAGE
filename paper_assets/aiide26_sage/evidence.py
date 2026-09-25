"""Recompute SAGE evidence and display explicitly attributed paper references."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACK = Path(__file__).resolve().parent
ROOT = PACK.parents[1]
DATA = PACK / "data"
GATE = DATA / "figure_05_06_gate_exploration"
SCENARIOS = dict(zip(
    ["sce-1", "sce-1m", "sce-2", "sce-2m", "sce-3", "sce-3m"],
    ["MarineMicro_MvsM_4", "MarineMicro_MvsM_4_mirror", "MarineMicro_MvsM_4_dist",
     "MarineMicro_MvsM_4_dist_mirror", "MarineMicro_MvsM_8", "MarineMicro_MvsM_8_mirror"],
))
METHOD_NAMES = {"ETG-only": "Graph-only SAGE", "etg-only": "Graph-only SAGE", "Synergy": "Full SAGE"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def iter_episodes(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def legacy(name):
    """Load a frozen plotting routine, without executing its CLI entry point."""
    key = "sage_paper_legacy_" + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, PACK / "scripts/legacy" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def selected_rows():
    parts = []
    for name in ["selected_gate_positive_episodes.csv", "selected_etg_counterpart_episodes.csv"]:
        part = pd.read_csv(GATE / "selection" / name)
        part["selected_source_file"] = name
        # Membership in these frozen 100-row/scenario files defines the sample.
        # The historical `selected` column is False even for retained rows.
        part["eval_dir"] = part.experiment_id.map(lambda run: str(PACK / "raw/evaluations" / run))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def recompute_selected(output):
    """Use exact episode IDs, never top-k or a new selection based on outcomes."""
    module = legacy("gate_analysis")
    module.episode_file = lambda directory, repeat: directory / "episodes.jsonl.gz"
    module.iter_jsonl = iter_episodes
    # The source renamed ETG-only to etg-only after these CSVs were frozen.
    original_classify = module.classify_state_relation
    module.classify_state_relation = lambda methods, count: original_classify(
        ["etg-only" if m == "ETG-only" else m for m in methods], count)
    episodes, frames = module.load_exact_episodes(selected_rows())
    states = module.aggregate_states(frames)
    episodes.to_csv(output / "selected_episodes.csv", index=False)
    pd.DataFrame(frames).to_csv(output / "selected_frames.csv.gz", index=False)
    states.to_csv(output / "selected_states.csv", index=False)
    return episodes, states


def stats(scores, wins, ddof=1):
    scores = pd.Series(scores, dtype=float)
    outcomes = pd.Series(wins).astype(str).str.lower()
    if not outcomes.isin(["true", "win", "1", "1.0", "false", "loss", "lose", "0", "0.0", "tie", "draw", "dogfall"]).all():
        raise ValueError("Unknown episode outcome")
    wins = outcomes.isin(["true", "win", "1", "1.0"])
    if scores.empty or not np.isfinite(scores).all() or len(scores) != len(wins):
        raise ValueError("Missing or invalid episode scores/outcomes")
    return {"n": len(scores), "score_mean": scores.mean(), "score_std": scores.std(ddof=ddof),
            "win_rate_pct": 100 * wins.mean(), "std_ddof": ddof}


def table_measurements(episodes=None):
    if episodes is None:
        episodes = pd.read_csv(GATE / "analysis_ready/boxplot_aligned_episode_table.csv")
    rows = []
    for (scenario, method), group in episodes.groupby(["scenario", "method"]):
        rows.append(dict(table=1, scenario=scenario, method=METHOD_NAMES[method],
                         evidence="recomputed SAGE selected episodes", verification_scope="recomputed",
                         **stats(group.score, group.result)))
    # External methods are reference values transcribed from the immutable paper.
    # They are not measured, trained, or independently validated by this project.
    references = pd.read_csv(DATA / "external_methods/paper_reported_results.csv")
    references["verification_scope"] = "paper transcription only"
    references["evidence"] = "camera-ready reported value; external experiment"
    references["n"] = np.nan
    references["std_ddof"] = np.nan
    return pd.concat([pd.DataFrame(rows), references], ignore_index=True)


def compare_tables(actual):
    expected = pd.read_csv(PACK / "reference/camera_ready_tables.csv")
    merged = expected.merge(actual, on=["table", "scenario", "method"], suffixes=("_paper", ""),
                            validate="one_to_one", how="outer", indicator=True)
    if not merged._merge.eq("both").all() or len(merged) != 54:
        raise ValueError("Expected all 54 manuscript cells, with no missing or duplicated keys")
    for metric in ["score_mean", "score_std", "win_rate_pct"]:
        digits = 1 if metric == "win_rate_pct" else 2
        merged[metric + "_matches"] = np.isclose(merged[metric].round(digits), merged[metric + "_paper"], atol=1e-8, rtol=0)
    return merged.drop(columns="_merge")


def verify_aggregated_table(actual, expected, keys, numeric):
    """Compare identifiers and numeric values independently of row ordering."""
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError("Recomputed rows do not have the frozen identifiers")
    failures = []
    for column in numeric:
        if not np.allclose(a[column].astype(float), b[column].astype(float), equal_nan=True, atol=1e-8, rtol=1e-8):
            failures.append(column)
    return failures


def correlations():
    directory = DATA / "figure_03_parameter_correlation"
    trials = pd.read_csv(directory / "etg_only_parameter_search_trials.csv")
    frozen = pd.read_csv(directory / "parameter_numeric_correlations.csv")
    result = []
    for row in frozen.itertuples():
        part = trials[trials.map_key.eq(row.map_key)]
        x = pd.to_numeric(part[row.parameter], errors="coerce")
        y = pd.to_numeric(part[row.metric], errors="coerce")
        valid = x.notna() & y.notna()
        r = x[valid].corr(y[valid])
        result.append({"scenario": row.map_key, "parameter": row.parameter, "metric": row.metric,
                       "n": int(valid.sum()), "pearson": r, "frozen_pearson": row.pearson,
                       "matches": int(valid.sum()) == row.n and np.isclose(r, row.pearson, atol=1e-10)})
    # Check that the trial CSV still agrees with its six upstream studies.
    trial_failures = []
    for path in (directory / "study_summaries").glob("*.json"):
        run = path.name.split("__")[0]
        part = trials[trials.experiment_id.eq(run)].set_index("trial")
        study = read_json(path)
        if len(part) != len(study["trials"]):
            trial_failures.append(run + ": count")
        for trial in study["trials"]:
            idx = trial["number"]
            if idx not in part.index:
                trial_failures.append(run + ": missing trial " + str(idx)); continue
            for metric in ["avg_score", "win_rate", "score_std"]:
                value = trial.get("user_attrs", {}).get(metric)
                expected = float(value) if value is not None else np.nan
                if not np.isclose(float(part.loc[idx, metric]), expected, equal_nan=True):
                    trial_failures.append(f"{run}/{idx}/{metric}")
    return pd.DataFrame(result), trial_failures


def switch_groups():
    directory = DATA / "figure_04_backup_switch"
    comparison = pd.read_csv(directory / "switch_grid_analysis/switch_grid_comparison_vs_no_switch.csv")
    module = legacy("switch_analysis")
    module.load_override_params = lambda: {}  # rank/score columns are in the comparison CSV
    matrix = module.build_matrix(comparison)
    records = []
    for _, row in matrix[~matrix.all_negative].iterrows():
        for config in module.CONFIG_ORDER:
            records.append({"scene": row.scene, "map_key": row.map_key, "rank": row["rank"],
                            "config": config, "score": row.base_score if config == "ms_ns" else row[config + "_score"]})
    long = pd.DataFrame(records)
    kept = []
    for scene, part in long.groupby("scene", sort=False):
        pivot = part.pivot_table(index=["map_key", "rank"], columns="config", values="score")
        center = pivot[module.CONFIG_ORDER].mean(axis=1)
        q1, q3 = center.quantile([0.25, 0.75]); iqr = q3 - q1
        good = pivot.index if iqr <= 1e-9 else pivot.index[center.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
        kept.append(part.set_index(["map_key", "rank"]).loc[good].reset_index())
    return pd.concat(kept, ignore_index=True)


def file_checks():
    failures = []
    manifest = read_json(PACK / "provenance.json")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(row["path"] + ": missing"); continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            failures.append(row["path"] + ": checksum mismatch")
    return failures


def verify(output, from_raw=False, hashes=False):
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    episodes = None
    if from_raw:
        episodes, states = recompute_selected(output)
        bad = verify_aggregated_table(episodes, pd.read_csv(GATE / "analysis_ready/boxplot_aligned_episode_table.csv"),
            ["scenario", "method", "episode_id"], ["score", "win", "frame_count"])
        failures.extend("selected episodes: " + x for x in bad)
        expected_states = pd.read_csv(GATE / "analysis_ready/boxplot_aligned_state_table.csv")
        bad = verify_aggregated_table(states, expected_states, ["scenario", "state_digest"],
            ["frames", "episodes", "ood_frames", "mean_hp_margin", "mean_game_progress", "mean_score_conditioned"])
        failures.extend("selected states: " + x for x in bad)
        a = states.set_index(["scenario", "state_digest"]).source_relation.sort_index()
        b = expected_states.set_index(["scenario", "state_digest"]).source_relation.sort_index()
        if not a.equals(b): failures.append("state source classifications")
    comparison = compare_tables(table_measurements(episodes))
    for metric in ["score_mean", "score_std", "win_rate_pct"]:
        for row in comparison[~comparison[metric + "_matches"]].itertuples():
            failures.append(f"Table {row.table}/{row.scenario}/{row.method}/{metric}")
    comparison.to_csv(output / "table_verification.csv", index=False)
    for number in (1, 2):
        comparison[comparison.table.eq(number)].to_csv(output / f"table_{number:02d}.csv", index=False)
    corr, trial_failures = correlations()
    corr.to_csv(output / "figure03_correlations_recomputed.csv", index=False)
    failures.extend(trial_failures)
    if not corr.matches.all(): failures.append("Figure 3 correlation values")
    trimmed = switch_groups()
    frozen = pd.read_csv(DATA / "figure_04_backup_switch/fig4d_filtered_param_group_mean_scores_iqr_trimmed.csv")
    failures.extend("Figure 4: " + x for x in verify_aggregated_table(trimmed, frozen, ["map_key", "rank", "config"], ["score"]))
    trimmed.to_csv(output / "figure04_filtered_groups_recomputed.csv", index=False)
    if hashes: failures.extend(file_checks())
    report = {
        "passed": not failures, "from_raw": from_raw, "checksums_verified": hashes,
        "evidence_integrity_passed": not failures,
        "scope": "SAGE evidence recomputation and external paper-reference transcription checks",
        "sage_cells_recomputed": int(comparison.verification_scope.eq("recomputed").sum()),
        "external_cells_transcribed": int(comparison.verification_scope.eq("paper transcription only").sum()),
        "external_experiments_reproduced": False,
        "table_3_reproduced": False,
        "table_cells": len(comparison), "correlations": len(corr), "switch_group_rows": len(trimmed),
        "failures": failures,
        "limitations": [
            "External methods, including Replay, are paper-reported references, not locally recomputed measurements.",
            "SAGE score standard deviations use sample SD (ddof=1).",
            "SAGE paper results use a frozen selected 100 episodes per scenario/method from 300-episode runs.",
            "The `selected` flag in selection CSVs is stale; file membership and exact episode IDs define the sample.",
            "Figure 4 excludes groups with all-negative switching effects and applies an IQR filter.",
            "Figure 6 uses logged state-association status as a proxy for source coverage, not a new geometric membership test.",
            "Recomputing archived evidence does not reproduce live SC2 trajectories or retrain baseline checkpoints.",
            "Table 3 is transcribed for reference; complete historical callback records are unavailable.",
        ],
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
