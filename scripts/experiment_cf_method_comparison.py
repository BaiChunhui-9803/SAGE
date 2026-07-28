#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
experiment_cf_method_comparison -- Run all 6 candidate identification
methods on real data and produce a comparison report.

Usage:
    python scripts/experiment_cf_method_comparison.py \
        --run_dir output/learner_results/training_runs/run_0002 \
        --kg_file cache/knowledge_graph/MarineMicro_MvsM_4_augmented/kg_simple.pkl \
        --top_k 20
"""

from __future__ import annotations

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from scripts.cf_candidate_methods import (
    Candidate,
    QValueMethod,
    CausalMethod,
    PathDivergenceMethod,
    PageRankMethod,
    CFRMethod,
    StatisticalBaseline,
    load_kg_and_transitions,
    load_episodes,
    compute_jaccard_matrix,
    find_consensus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_all_methods(
    kg_path: str,
    run_dir: Path,
    top_k: int = 20,
    recent_trials: int = None,
) -> Dict[str, List[Candidate]]:
    logger.info("Loading ETG and transitions...")
    kg, transitions = load_kg_and_transitions(kg_path)
    logger.info(f"  ETG: {len(kg.unique_states)} states, {kg.total_visits} visits")

    logger.info("Loading episode data...")
    episodes = load_episodes(run_dir)
    if recent_trials is not None:
        trial_numbers = list(
            range(max(0, len(episodes) - recent_trials), len(episodes))
        )
    else:
        trial_numbers = None
    episodes = load_episodes(run_dir, trial_numbers)
    scores = [ep.get("score", 0) for ep in episodes]
    wins = sum(1 for ep in episodes if ep.get("result") == "Win")
    logger.info(
        f"  Episodes: {len(episodes)}, Win rate: {wins / len(episodes) * 100:.1f}%"
        if episodes
        else "  No episodes found"
    )
    if scores:
        logger.info(
            f"  Score: mean={np.mean(scores):.1f}, "
            f"p25={np.percentile(scores, 25):.1f}, "
            f"p75={np.percentile(scores, 75):.1f}"
        )

    all_results: Dict[str, List[Candidate]] = {}

    logger.info("\n" + "=" * 60)
    logger.info("Method 1/6: Q-value (Value Iteration)")
    logger.info("=" * 60)
    m1 = QValueMethod(kg, transitions, gamma=0.95, min_visits=5)
    all_results[m1.name()] = m1.run()

    logger.info("\n" + "=" * 60)
    logger.info("Method 2/6: Causal (do-calculus)")
    logger.info("=" * 60)
    m2 = CausalMethod(kg, transitions, min_visits=5, max_propagation_depth=10)
    all_results[m2.name()] = m2.run()

    logger.info("\n" + "=" * 60)
    logger.info("Method 3/6: Path Divergence (chi2)")
    logger.info("=" * 60)
    m3 = PathDivergenceMethod(episodes, min_count=5)
    all_results[m3.name()] = m3.run()

    logger.info("\n" + "=" * 60)
    logger.info("Method 4/6: PageRank (Hitting)")
    logger.info("=" * 60)
    m4 = PageRankMethod(kg, transitions, min_visits=5, max_iterations=200, damping=0.85)
    all_results[m4.name()] = m4.run()

    logger.info("\n" + "=" * 60)
    logger.info("Method 5/6: CFR (Regret)")
    logger.info("=" * 60)
    m5 = CFRMethod(episodes, kg, transitions, min_regret_samples=5)
    all_results[m5.name()] = m5.run()

    logger.info("\n" + "=" * 60)
    logger.info("Method 6/6: V2 Baseline (Stat)")
    logger.info("=" * 60)
    m6 = StatisticalBaseline(episodes, min_count=5)
    all_results[m6.name()] = m6.run()

    return all_results


def print_report(all_results: Dict[str, List[Candidate]], top_k: int = 20):
    methods = list(all_results.keys())

    print("\n" + "=" * 80)
    print("  CANDIDATE IDENTIFICATION METHODS COMPARISON REPORT")
    print("=" * 80)

    print(f"\n{'Method':<28} {'Candidates':>10} {'Top-1 Candidate':>30}")
    print("-" * 80)
    for m in methods:
        cands = all_results[m]
        top1 = (
            f"nid={cands[0].nid} {cands[0].bad_action}->{cands[0].recommended_action} (score={cands[0].score:.2f})"
            if cands
            else "N/A"
        )
        print(f"{m:<28} {len(cands):>10} {top1:>30}")

    print(f"\n{'=' * 80}")
    print(f"  TOP-{top_k} JACCARD SIMILARITY MATRIX")
    print(f"{'=' * 80}")

    jaccard = compute_jaccard_matrix(all_results, top_k=top_k)

    header = f"{'':>28}"
    for m in methods:
        short = m.split("(")[0].strip()[:12]
        header += f" {short:>12}"
    print(header)
    print("-" * (28 + 13 * len(methods)))

    for m1 in methods:
        row = f"{m1:>28}"
        for m2 in methods:
            val = jaccard.get(m1, {}).get(m2, 0.0)
            row += f" {val:>12.3f}"
        print(row)

    print(f"\n{'=' * 80}")
    print("  TOP-10 CANDIDATES PER METHOD")
    print(f"{'=' * 80}")

    for m in methods:
        cands = all_results[m][:10]
        if not cands:
            print(f"\n  [{m}] No candidates")
            continue
        print(f"\n  [{m}]")
        for i, c in enumerate(cands):
            meta_str = ""
            if "q_bad" in c.metadata:
                meta_str = (
                    f"Q_bad={c.metadata['q_bad']:.1f} Q_best={c.metadata['q_best']:.1f}"
                )
            elif "ate" in c.metadata:
                meta_str = f"ATE={c.metadata['ate']:.3f}"
            elif "chi2" in c.metadata:
                meta_str = f"chi2={c.metadata['chi2']:.1f} p={c.metadata['p_value']:.4f} V={c.metadata['cramers_v']:.3f}"
            elif "h_bad" in c.metadata:
                meta_str = (
                    f"h_bad={c.metadata['h_bad']:.3f} h_best={c.metadata['h_best']:.3f}"
                )
            elif "regret" in c.metadata:
                meta_str = (
                    f"regret={c.metadata['regret']:.1f} n={c.metadata['n_actual']}"
                )
            elif "potential" in c.metadata:
                meta_str = (
                    f"pot={c.metadata['potential']:.1f} gc={c.metadata['good_count']}"
                )
            print(
                f"    {i + 1:>2}. nid={c.nid:<5} {c.bad_action}->{c.recommended_action}  "
                f"score={c.score:.2f}  {meta_str}"
            )

    print(f"\n{'=' * 80}")
    print("  CONSENSUS CANDIDATES (endorsed by 2+ methods)")
    print(f"{'=' * 80}")

    consensus = find_consensus(all_results, top_k=50)
    if not consensus:
        print("  No consensus candidates found.")
    else:
        for i, (c, methods_list) in enumerate(consensus[:20]):
            print(
                f"  {i + 1:>2}. nid={c.nid:<5} {c.bad_action}->{c.recommended_action}  "
                f"score={c.score:.2f}  "
                f"endorsed_by={len(methods_list)} {methods_list}"
            )

    n_all_agree = sum(1 for _, ml in consensus if len(ml) >= 4)
    n_3plus = sum(1 for _, ml in consensus if len(ml) >= 3)
    n_2plus = len(consensus)
    print(
        f"\n  Summary: {n_all_agree} candidates with 4+ endorsements, "
        f"{n_3plus} with 3+, {n_2plus} with 2+"
    )

    print(f"\n{'=' * 80}")
    print("  METHOD-SPECIFIC CANDIDATES (only found by one method)")
    print(f"{'=' * 80}")

    for m in methods:
        other_top = set()
        for m2 in methods:
            if m2 == m:
                continue
            for c in all_results[m2][:top_k]:
                other_top.add(c.key())

        unique = [c for c in all_results[m][:top_k] if c.key() not in other_top]
        if unique:
            print(f"\n  [{m}] {len(unique)} unique in top-{top_k}:")
            for c in unique[:5]:
                print(
                    f"    nid={c.nid:<5} {c.bad_action}->{c.recommended_action}  score={c.score:.2f}"
                )
        else:
            print(f"\n  [{m}] No unique candidates (all overlap with others)")


def save_results_json(
    all_results: Dict[str, List[Candidate]], output_path: str, top_k: int = 50
):
    data = {}
    for method, cands in all_results.items():
        data[method] = [
            {
                "nid": c.nid,
                "bad_action": c.bad_action,
                "recommended_action": c.recommended_action,
                "score": round(c.score, 4),
                "method": c.method,
                "metadata": c.metadata,
            }
            for c in cands[:top_k]
        ]

    consensus = find_consensus(all_results, top_k=50)
    data["__consensus__"] = [
        {
            "nid": c.nid,
            "bad_action": c.bad_action,
            "recommended_action": c.recommended_action,
            "score": round(c.score, 4),
            "endorsed_by": methods,
            "n_endorsements": len(methods),
        }
        for c, methods in consensus[:50]
    ]

    jaccard = compute_jaccard_matrix(all_results, top_k=top_k)
    data["__jaccard__"] = jaccard

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare 6 candidate identification methods on real data"
    )
    parser.add_argument(
        "--run_dir",
        default=str(
            ROOT_DIR / "output" / "learner_results" / "training_runs" / "run_0002"
        ),
        help="Training run directory with episodes",
    )
    parser.add_argument(
        "--kg_file",
        default=str(
            ROOT_DIR
            / "cache"
            / "knowledge_graph"
            / "MarineMicro_MvsM_4_augmented"
            / "kg_simple.pkl"
        ),
        help="Path to kg_simple.pkl",
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Top-K for Jaccard comparison"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = run_all_methods(
        kg_path=args.kg_file,
        run_dir=Path(args.run_dir),
        top_k=args.top_k,
    )

    print_report(results, top_k=args.top_k)

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.run_dir) / "cf_method_comparison.json")
    save_results_json(results, output_path, top_k=50)
