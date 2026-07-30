#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cf_episode_agreement -- Per-episode multi-method agreement analysis.

For each scorable frame (nid, action) in every episode, 6 methods independently
judge whether the state has improvement potential. We then measure how well
the methods agree: Fleiss' Kappa, pairwise Cohen's Kappa, Spearman correlation,
per-episode agreement rates, and per-nid consensus.

Methods:
  0: Q-value       -- advantage = max Q(nid,a) - Q(nid, action);  threshold: > 0
  1: Causal        -- advantage = max wr(nid,a) - wr(nid, action); threshold: > 0.05
  2: PathDiv       -- nid-level Cramer's V + chi2 p; threshold: V>0.3 & p<0.05
  3: PageRank      -- advantage = max h(nid,a) - h(nid, action);  threshold: > 0.05
  4: CFR           -- advantage = max cf(nid,a) - cf(nid, action); threshold: > 0
  5: V2Baseline    -- nid-level stat score; threshold: > 2.0

Usage:
    python scripts/cf_episode_agreement.py
    python scripts/cf_episode_agreement.py --run_dir ... --etg_file ...
"""

from __future__ import annotations

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ROOT_DIR
from src.decision.experience_transition_graph import DecisionExperienceTransitionGraph
from scripts.cf_candidate_methods import load_etg_and_transitions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

METHODS = ["Q-value", "Causal", "PathDiv", "PageRank", "CFR", "V2Baseline"]
N_METHODS = 6


# ---------------------------------------------------------------------------
# Pre-computation: action-level advantages from ETG
# ---------------------------------------------------------------------------


def precompute_q_advantage(
    ETG, transitions, gamma=0.95, min_visits=5, max_iter=100, conv=1e-3
):
    states = set()
    for nid in transitions:
        states.add(nid)
        for a, info in transitions[nid].items():
            if not isinstance(info, dict):
                continue
            for ns in info.get("next_states", {}):
                states.add(ns)

    values = {s: 0.0 for s in states}

    for iteration in range(max_iter):
        new_values = dict(values)
        max_delta = 0.0
        for s in states:
            trans_s = transitions.get(s, {})
            if "__terminal__" in trans_s:
                continue
            best_q = float("-inf")
            for a, info in trans_s.items():
                if not isinstance(info, dict):
                    continue
                total = info.get("total", 0)
                if total < min_visits:
                    continue
                ns_dict = info.get("next_states", {})
                if not ns_dict:
                    continue
                quality = ETG.get_action_quality(s, a)
                reward = quality.get("avg_step_reward", 0.0) if quality else 0.0
                exp_future = sum(
                    (cnt / total) * values.get(ns, 0.0) for ns, cnt in ns_dict.items()
                )
                best_q = max(best_q, reward + gamma * exp_future)
            if best_q > float("-inf"):
                new_values[s] = best_q
                max_delta = max(max_delta, abs(new_values[s] - values[s]))
        values = new_values
        if max_delta < conv:
            logger.info(f"  Q-value VI converged at iter {iteration + 1}")
            break

    adv = {}
    for s in states:
        trans_s = transitions.get(s, {})
        q_vals = {}
        for a, info in trans_s.items():
            if not isinstance(info, dict):
                continue
            total = info.get("total", 0)
            if total < min_visits:
                continue
            ns_dict = info.get("next_states", {})
            if not ns_dict:
                continue
            quality = ETG.get_action_quality(s, a)
            reward = quality.get("avg_step_reward", 0.0) if quality else 0.0
            exp_future = sum(
                (cnt / total) * values.get(ns, 0.0) for ns, cnt in ns_dict.items()
            )
            q_vals[a] = reward + gamma * exp_future
        if len(q_vals) >= 2:
            q_best = max(q_vals.values())
            for a, q in q_vals.items():
                adv[(s, a)] = q_best - q
    return adv


def precompute_causal_advantage(transitions, min_visits=3):
    adv = {}
    for nid, actions in transitions.items():
        wr_map = {}
        for a, info in actions.items():
            if not isinstance(info, dict):
                continue
            if info.get("total", 0) < min_visits:
                continue
            wr_map[a] = info.get("win_rate", 0.0)
        if len(wr_map) >= 2:
            wr_best = max(wr_map.values())
            for a, wr in wr_map.items():
                adv[(nid, a)] = wr_best - wr
    return adv


def precompute_pagerank_advantage(
    ETG, transitions, min_visits=3, max_iter=200, damping=0.85
):
    states = set()
    for nid in transitions:
        states.add(nid)
        for a, info in transitions[nid].items():
            if not isinstance(info, dict):
                continue
            for ns in info.get("next_states", {}):
                states.add(ns)

    h = {}
    for s in states:
        trans_s = transitions.get(s, {})
        best_wr = 0.0
        for a, info in trans_s.items():
            if not isinstance(info, dict):
                continue
            if info.get("total", 0) >= 1:
                best_wr = max(best_wr, info.get("win_rate", 0.0))
        h[s] = best_wr

    for iteration in range(max_iter):
        max_delta = 0.0
        new_h = {}
        for s in states:
            trans_s = transitions.get(s, {})
            if "__terminal__" in trans_s:
                new_h[s] = h.get(s, 0.0)
                continue
            best_h = 0.0
            for a, info in trans_s.items():
                if not isinstance(info, dict):
                    continue
                total = info.get("total", 0)
                if total < 1:
                    continue
                ns_dict = info.get("next_states", {})
                if not ns_dict:
                    continue
                action_h = sum(
                    (cnt / total) * h.get(ns, 0.0) for ns, cnt in ns_dict.items()
                )
                best_h = max(best_h, action_h)
            quality = (
                ETG.get_action_quality(s, list(trans_s.keys())[0]) if trans_s else None
            )
            immediate_wr = quality.get("win_rate", 0.0) if quality else 0.0
            new_h[s] = damping * best_h + (1 - damping) * immediate_wr
            max_delta = max(max_delta, abs(new_h[s] - h.get(s, 0.0)))
        h = new_h
        if max_delta < 1e-4:
            logger.info(f"  PageRank converged at iter {iteration + 1}")
            break

    adv = {}
    for nid in transitions:
        trans_nid = transitions[nid]
        h_map = {}
        for a, info in trans_nid.items():
            if not isinstance(info, dict):
                continue
            total = info.get("total", 0)
            if total < min_visits:
                continue
            ns_dict = info.get("next_states", {})
            if not ns_dict:
                continue
            h_a = sum((cnt / total) * h.get(ns, 0.0) for ns, cnt in ns_dict.items())
            h_map[a] = h_a
        if len(h_map) >= 2:
            h_best = max(h_map.values())
            for a, hv in h_map.items():
                adv[(nid, a)] = h_best - hv
    return adv


def precompute_cfr_advantage(ETG, transitions, min_visits=3):
    v_etg = {}
    for nid in ETG.unique_states:
        quality = ETG.get_action_quality(nid, None)
        if quality is None:
            best_qs = 0.0
            for a_info in ETG.state_action_map.get(nid, {}).values():
                best_qs = max(best_qs, a_info.quality_score)
            v_etg[nid] = best_qs
        else:
            v_etg[nid] = quality.get("quality_score", 0.0)

    for nid in transitions:
        if nid not in v_etg:
            v_etg[nid] = 0.0

    adv = {}
    for nid, actions in transitions.items():
        cf_map = {}
        for a, info in actions.items():
            if not isinstance(info, dict):
                continue
            total = info.get("total", 0)
            if total < min_visits:
                continue
            ns_dict = info.get("next_states", {})
            if not ns_dict:
                continue
            cf_val = sum(
                (cnt / total) * v_etg.get(ns, 0.0) for ns, cnt in ns_dict.items()
            )
            cf_map[a] = cf_val
        if len(cf_map) >= 2:
            cf_best = max(cf_map.values())
            for a, cv in cf_map.items():
                adv[(nid, a)] = cf_best - cv
    return adv


# ---------------------------------------------------------------------------
# Pre-computation: nid-level statistics from episodes
# ---------------------------------------------------------------------------


def precompute_pathdiv_and_v2(summaries, transitions):
    scores = [s["score"] for s in summaries]
    p25 = float(np.percentile(scores, 25))
    p75 = float(np.percentile(scores, 75))

    nid_action_good = defaultdict(lambda: defaultdict(int))
    nid_action_bad = defaultdict(lambda: defaultdict(int))
    nid_action_good_hp = defaultdict(lambda: defaultdict(list))
    nid_action_bad_hp = defaultdict(lambda: defaultdict(list))

    valid_nids = set()
    for nid, actions in transitions.items():
        valid_acts = [
            a
            for a, info in actions.items()
            if isinstance(info, dict) and info.get("total", 0) >= 3
        ]
        if len(valid_acts) >= 2:
            valid_nids.add(nid)

    for ep in summaries:
        score = ep["score"]
        if score >= p75:
            group = "good"
        elif score <= p25:
            group = "bad"
        else:
            continue

        for nid, action, hp_delta in ep["frames"]:
            if nid is None or not action or nid not in valid_nids:
                continue
            if group == "good":
                nid_action_good[nid][action] += 1
                nid_action_good_hp[nid][action].append(hp_delta)
            else:
                nid_action_bad[nid][action] += 1
                nid_action_bad_hp[nid][action].append(hp_delta)

    cramers_v = {}
    chi2_p = {}
    v2_score = {}

    for nid in valid_nids:
        good_actions = nid_action_good[nid]
        bad_actions = nid_action_bad[nid]
        shared = set(good_actions.keys()) & set(bad_actions.keys())
        all_actions = set(good_actions.keys()) | set(bad_actions.keys())

        # --- V2 Baseline score ---
        best_v2 = 0.0
        for bad_act in shared:
            gc = good_actions.get(bad_act, 0)
            bc = bad_actions.get(bad_act, 0)
            if gc < 5 or bc < 5:
                continue
            bad_hp = np.mean(nid_action_bad_hp[nid].get(bad_act, [0]))
            for good_act in all_actions:
                if good_act == bad_act:
                    continue
                g2 = good_actions.get(good_act, 0)
                b2 = bad_actions.get(good_act, 0)
                if g2 < 5:
                    continue
                if g2 <= b2:
                    continue
                good_hp = np.mean(nid_action_good_hp[nid].get(good_act, [0]))
                pot = good_hp - bad_hp
                if pot > 0:
                    s = pot * min(g2 / 10.0, 3.0)
                    best_v2 = max(best_v2, s)
        v2_score[nid] = best_v2

        # --- PathDiv chi-squared ---
        action_list = [
            a
            for a in all_actions
            if good_actions.get(a, 0) >= 3 or bad_actions.get(a, 0) >= 3
        ]
        if len(action_list) < 2:
            cramers_v[nid] = 0.0
            chi2_p[nid] = 1.0
            continue

        contingency = np.zeros((len(action_list), 2))
        for i, a in enumerate(action_list):
            contingency[i, 0] = good_actions.get(a, 0)
            contingency[i, 1] = bad_actions.get(a, 0)

        row_sums = contingency.sum(axis=1)
        mask = row_sums > 0
        contingency = contingency[mask]
        if contingency.shape[0] < 2:
            cramers_v[nid] = 0.0
            chi2_p[nid] = 1.0
            continue

        if np.any(contingency == 0):
            contingency = contingency + 0.5

        try:
            chi2_val, p_val, dof, _ = chi2_contingency(contingency)
            n_total = contingency.sum()
            min_dim = min(contingency.shape)
            cv = (
                np.sqrt(chi2_val / (n_total * (min_dim - 1)))
                if min_dim > 1 and n_total > 0
                else 0.0
            )
            cramers_v[nid] = float(cv)
            chi2_p[nid] = float(p_val)
        except Exception:
            cramers_v[nid] = 0.0
            chi2_p[nid] = 1.0

    return cramers_v, chi2_p, v2_score


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------


def load_summaries(run_dir):
    summaries = []
    trials_dir = Path(run_dir) / "trials"
    if not trials_dir.exists():
        return summaries

    for td in sorted(trials_dir.iterdir()):
        if not td.is_dir() or not td.name.startswith("trial_"):
            continue
        ep_file = td / "episodes.jsonl"
        if not ep_file.exists():
            continue
        with open(str(ep_file), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                    frames = []
                    for fr in ep.get("frames", []):
                        nid = fr.get("nid")
                        action = fr.get("action_code", "")
                        hp_delta = fr.get("hp_delta", 0)
                        frames.append((nid, action, hp_delta))
                    summaries.append(
                        {
                            "score": ep.get("score", 0),
                            "result": ep.get("result", "Unknown"),
                            "frames": frames,
                        }
                    )
                except json.JSONDecodeError:
                    pass
    return summaries


# ---------------------------------------------------------------------------
# Frame scoring
# ---------------------------------------------------------------------------


def score_all_frames(
    summaries, q_adv, causal_adv, h_adv, cf_adv, cramers_v, chi2_p, v2_score, valid_nids
):
    all_scores = []
    all_labels = []
    ep_info = []
    nid_counts = defaultdict(lambda: {"total": 0, "improvable": np.zeros(N_METHODS)})

    scorable_nids = set(cramers_v.keys()) | set(v2_score.keys())
    for key in q_adv:
        scorable_nids.add(key[0])

    for ep_idx, ep in enumerate(summaries):
        ep_scorable = 0
        ep_improvable_votes = 0

        for nid, action, hp_delta in ep["frames"]:
            if nid is None or not action or nid not in valid_nids:
                continue

            scores = np.zeros(N_METHODS)
            labels = np.zeros(N_METHODS, dtype=bool)

            # 0: Q-value
            scores[0] = q_adv.get((nid, action), 0.0)
            labels[0] = scores[0] > 0.0

            # 1: Causal
            scores[1] = causal_adv.get((nid, action), 0.0)
            labels[1] = scores[1] > 0.05

            # 2: PathDiv (nid-level)
            scores[2] = cramers_v.get(nid, 0.0)
            labels[2] = cramers_v.get(nid, 0.0) > 0.3 and chi2_p.get(nid, 1.0) < 0.05

            # 3: PageRank
            scores[3] = h_adv.get((nid, action), 0.0)
            labels[3] = scores[3] > 0.05

            # 4: CFR
            scores[4] = cf_adv.get((nid, action), 0.0)
            labels[4] = scores[4] > 0.0

            # 5: V2Baseline (nid-level)
            scores[5] = v2_score.get(nid, 0.0)
            labels[5] = scores[5] > 2.0

            all_scores.append(scores)
            all_labels.append(labels)
            ep_scorable += 1
            ep_improvable_votes += labels.sum()

            nid_counts[nid]["total"] += 1
            nid_counts[nid]["improvable"] += labels

        ep_info.append(
            {
                "ep_idx": ep_idx,
                "score": ep["score"],
                "result": ep["result"],
                "n_scorable": ep_scorable,
                "n_improvable_votes": ep_improvable_votes,
            }
        )

    scores_arr = np.array(all_scores) if all_scores else np.zeros((0, N_METHODS))
    labels_arr = (
        np.array(all_labels) if all_labels else np.zeros((0, N_METHODS), dtype=bool)
    )
    return scores_arr, labels_arr, ep_info, dict(nid_counts)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def fleiss_kappa(labels):
    n, k = labels.shape
    n_ij = np.zeros((n, 2))
    n_ij[:, 1] = labels.sum(axis=1)
    n_ij[:, 0] = k - n_ij[:, 1]
    P_i = (np.sum(n_ij**2, axis=1) - k) / (k * (k - 1))
    P_bar = P_i.mean()
    p = n_ij.sum(axis=0) / (n * k)
    P_e = (p**2).sum()
    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cohens_kappa(l1, l2):
    n = len(l1)
    a = int(np.sum(l1 & l2))
    b = int(np.sum(l1 & ~l2))
    c = int(np.sum(~l1 & l2))
    d = int(np.sum(~l1 & ~l2))
    p_o = (a + d) / n
    p1t = (a + b) / n
    p1f = (c + d) / n
    p2t = (a + c) / n
    p2f = (b + d) / n
    p_e = p1t * p2t + p1f * p2f
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def kappa_interpretation(k):
    if k < 0:
        return "poor"
    elif k < 0.20:
        return "slight"
    elif k < 0.40:
        return "fair"
    elif k < 0.60:
        return "moderate"
    elif k < 0.80:
        return "substantial"
    else:
        return "almost perfect"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(scores, labels, ep_info, nid_counts, output_path=None):
    n_frames = labels.shape[0]
    if n_frames == 0:
        print("No scorable frames found.")
        return

    print("\n" + "=" * 80)
    print("  PER-EPISODE MULTI-METHOD AGREEMENT ANALYSIS")
    print("=" * 80)
    n_eps = len(ep_info)
    n_wins = sum(1 for e in ep_info if e["result"] == "Win")
    print(f"Episodes: {n_eps}  (Win: {n_wins}, Loss: {n_eps - n_wins})")
    print(f"Scorable frames: {n_frames}")
    print(f"Methods: {N_METHODS} ({', '.join(METHODS)})")

    # --- 1. Global agreement ---
    print(f"\n{'=' * 80}")
    print("  1. GLOBAL AGREEMENT")
    print(f"{'=' * 80}")

    fk = fleiss_kappa(labels)
    print(f"\nFleiss' Kappa (6 methods): {fk:.4f}  ({kappa_interpretation(fk)})")

    vote_counts = labels.sum(axis=1)
    print(f"\nVote distribution (how many methods say 'improvable'):")
    for v in range(N_METHODS + 1):
        cnt = int(np.sum(vote_counts == v))
        pct = cnt / n_frames * 100
        bar = "#" * int(pct / 2)
        print(f"  {v}/{N_METHODS} methods: {cnt:>8} frames ({pct:>5.1f}%) {bar}")

    unanimous = int(np.sum((vote_counts == 0) | (vote_counts == N_METHODS)))
    majority = int(np.sum((vote_counts <= 1) | (vote_counts >= N_METHODS - 1)))
    print(
        f"\nUnanimous agreement (0/6 or 6/6): {unanimous} ({unanimous / n_frames * 100:.1f}%)"
    )
    print(
        f"Strong agreement   (<=1 or >=5):  {majority} ({majority / n_frames * 100:.1f}%)"
    )

    # --- 2. Method improvable rates ---
    print(f"\n{'=' * 80}")
    print("  2. METHOD IMPROVABLE RATES")
    print(f"{'=' * 80}")

    method_rates = labels.mean(axis=0) * 100
    print(f"\n{'Method':<20} {'Impr. Rate':>12} {'Description'}")
    print("-" * 60)
    descs = [
        "action not Q-optimal at this nid",
        "action win_rate < best by > 5%",
        "nid has significant action-outcome dependence",
        "action hitting_prob < best by > 5%",
        "action has positive counterfactual regret",
        "nid has good/bad action distribution gap",
    ]
    for i, m in enumerate(METHODS):
        print(f"  {m:<18} {method_rates[i]:>10.1f}%   {descs[i]}")

    # --- 3. Pairwise Cohen's Kappa ---
    print(f"\n{'=' * 80}")
    print("  3. PAIRWISE COHEN'S KAPPA")
    print(f"{'=' * 80}")

    kappa_matrix = np.zeros((N_METHODS, N_METHODS))
    for i in range(N_METHODS):
        for j in range(N_METHODS):
            if i == j:
                kappa_matrix[i, j] = 1.0
            else:
                kappa_matrix[i, j] = cohens_kappa(labels[:, i], labels[:, j])

    short = [m[:8] for m in METHODS]
    header = f"{'':>12}" + "".join(f"{s:>12}" for s in short)
    print(f"\n{header}")
    print("-" * (12 + 12 * N_METHODS))
    for i, m in enumerate(METHODS):
        row = f"{m:>12}"
        for j in range(N_METHODS):
            row += f"{kappa_matrix[i, j]:>12.3f}"
        print(row)

    # --- 4. Pairwise Spearman ---
    print(f"\n{'=' * 80}")
    print("  4. PAIRWISE SPEARMAN RHO (continuous scores)")
    print(f"{'=' * 80}")

    rng = np.random.default_rng(42)
    sample_size = min(50000, n_frames)
    sample_idx = rng.choice(n_frames, size=sample_size, replace=False)
    sampled_scores = scores[sample_idx]

    def _rankdata(arr):
        ranks = np.empty_like(arr, dtype=float)
        order = np.argsort(arr)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    spearman_matrix = np.zeros((N_METHODS, N_METHODS))
    for i in range(N_METHODS):
        for j in range(N_METHODS):
            if i == j:
                spearman_matrix[i, j] = 1.0
            elif j < i:
                spearman_matrix[i, j] = spearman_matrix[j, i]
            else:
                s1 = sampled_scores[:, i]
                s2 = sampled_scores[:, j]
                if np.std(s1) < 1e-12 or np.std(s2) < 1e-12:
                    spearman_matrix[i, j] = 0.0
                else:
                    r1 = _rankdata(s1)
                    r2 = _rankdata(s2)
                    n_s = len(r1)
                    rho = 1.0 - 6.0 * np.sum((r1 - r2) ** 2) / (n_s * (n_s**2 - 1))
                    spearman_matrix[i, j] = float(rho)

    print(f"\n{header}")
    print("-" * (12 + 12 * N_METHODS))
    for i, m in enumerate(METHODS):
        row = f"{m:>12}"
        for j in range(N_METHODS):
            row += f"{spearman_matrix[i, j]:>12.3f}"
        print(row)

    # --- 5. Per-episode agreement ---
    print(f"\n{'=' * 80}")
    print("  5. PER-EPISODE AGREEMENT")
    print(f"{'=' * 80}")

    ep_agreement = []
    ep_agreement_win = []
    ep_agreement_loss = []

    frame_offset = 0
    for e in ep_info:
        n_sc = e["n_scorable"]
        if n_sc == 0:
            continue
        ep_labels = labels[frame_offset : frame_offset + n_sc]
        frame_offset += n_sc

        ep_votes = ep_labels.sum(axis=1)
        agree_count = int(np.sum((ep_votes <= 1) | (ep_votes >= N_METHODS - 1)))
        agree_rate = agree_count / n_sc
        ep_agreement.append(agree_rate)

        if e["result"] == "Win":
            ep_agreement_win.append(agree_rate)
        else:
            ep_agreement_loss.append(agree_rate)

    ep_agreement = np.array(ep_agreement)
    print(f"\nMean agreement rate: {ep_agreement.mean():.3f}")
    print(f"Median agreement rate: {np.median(ep_agreement):.3f}")
    n_high = int(np.sum(ep_agreement >= 0.8))
    n_low = int(np.sum(ep_agreement <= 0.2))
    print(
        f"High agreement episodes (>=80%): {n_high} ({n_high / len(ep_agreement) * 100:.1f}%)"
    )
    print(
        f"Low agreement episodes  (<=20%): {n_low} ({n_low / len(ep_agreement) * 100:.1f}%)"
    )

    if ep_agreement_win:
        print(
            f"\nWin  episodes ({len(ep_agreement_win)}): mean agreement = {np.mean(ep_agreement_win):.3f}"
        )
    if ep_agreement_loss:
        print(
            f"Loss episodes ({len(ep_agreement_loss)}): mean agreement = {np.mean(ep_agreement_loss):.3f}"
        )

    # --- 6. Top-30 nids ---
    print(f"\n{'=' * 80}")
    print("  6. TOP-30 MOST VISITED NIDS")
    print(f"{'=' * 80}")

    sorted_nids = sorted(nid_counts.items(), key=lambda x: -x[1]["total"])[:30]
    print(f"\n{'nid':>6} {'visits':>8} ", end="")
    for s in [m[:6] for m in METHODS]:
        print(f"{s:>8}", end="")
    print(f"  {'Consensus':>10}")
    print("-" * (6 + 8 + 8 * N_METHODS + 12))

    for nid, data in sorted_nids:
        total = data["total"]
        imp = data["improvable"]
        rates = imp / total * 100 if total > 0 else np.zeros(N_METHODS)
        n_yes = int(np.sum(imp >= total * 0.5))
        consensus = "YES" if n_yes >= 4 else ("partial" if n_yes >= 3 else "")
        print(f"{nid:>6} {total:>8} ", end="")
        for r in rates:
            print(f"{r:>7.1f}%", end="")
        print(f"  {consensus:>10}")

    # --- 7. Consensus analysis ---
    print(f"\n{'=' * 80}")
    print("  7. CONSENSUS NIDS")
    print(f"{'=' * 80}")

    consensus_yes = []
    consensus_no = []
    for nid, data in nid_counts.items():
        total = data["total"]
        imp = data["improvable"]
        n_yes = int(np.sum(imp >= total * 0.5))
        if n_yes >= 4:
            consensus_yes.append((nid, total, imp, n_yes))
        elif n_yes <= 1:
            consensus_no.append((nid, total, imp, n_yes))

    print(f"\nNids with >=4/6 methods agreeing 'improvable': {len(consensus_yes)}")
    print(f"Nids with <=1/6 methods agreeing 'improvable': {len(consensus_no)}")

    if consensus_yes:
        consensus_yes.sort(key=lambda x: -x[1])
        print(f"\nTop-15 consensus-improvable nids:")
        for nid, total, imp, n_yes in consensus_yes[:15]:
            pct = f"{n_yes}/{N_METHODS}"
            rates = " ".join(f"{int(r):>3}%" for r in (imp / total * 100))
            print(f"  nid={nid:<6} visits={total:<7} agree={pct}  rates=[{rates}]")

    # --- Save JSON ---
    if output_path:
        report_data = {
            "fleiss_kappa": float(fk),
            "n_frames": n_frames,
            "n_episodes": len(ep_info),
            "vote_distribution": {
                str(v): int(np.sum(vote_counts == v)) for v in range(N_METHODS + 1)
            },
            "method_improvable_rates": {
                m: float(method_rates[i]) for i, m in enumerate(METHODS)
            },
            "pairwise_kappa": {
                METHODS[i]: {
                    METHODS[j]: float(kappa_matrix[i, j]) for j in range(N_METHODS)
                }
                for i in range(N_METHODS)
            },
            "pairwise_spearman": {
                METHODS[i]: {
                    METHODS[j]: float(spearman_matrix[i, j]) for j in range(N_METHODS)
                }
                for i in range(N_METHODS)
            },
            "episode_agreement": {
                "mean": float(ep_agreement.mean()),
                "high_agreement_count": n_high,
                "low_agreement_count": n_low,
                "win_mean": float(np.mean(ep_agreement_win))
                if ep_agreement_win
                else 0.0,
                "loss_mean": float(np.mean(ep_agreement_loss))
                if ep_agreement_loss
                else 0.0,
            },
            "consensus_nids_yes": len(consensus_yes),
            "consensus_nids_no": len(consensus_no),
            "top_nids": [
                {
                    "nid": nid,
                    "visits": int(data["total"]),
                    "method_rates": {
                        METHODS[i]: float(data["improvable"][i] / data["total"] * 100)
                        for i in range(N_METHODS)
                    },
                }
                for nid, data in sorted_nids
            ],
        }
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Per-episode multi-method agreement analysis"
    )
    parser.add_argument(
        "--run_dir",
        default=str(
            ROOT_DIR / "output" / "learner_results" / "training_runs" / "run_0002"
        ),
    )
    parser.add_argument(
        "--etg_file",
        default=str(
            ROOT_DIR
            / "cache"
            / "experience_transition_graph"
            / "MarineMicro_MvsM_4_augmented"
            / "etg_simple.pkl"
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.run_dir) / "cf_episode_agreement.json")

    t_start = time.time()

    logger.info("Loading ETG and transitions...")
    ETG, transitions = load_etg_and_transitions(args.etg_file)
    logger.info(f"  ETG: {len(ETG.unique_states)} states, {ETG.total_visits} visits")

    valid_nids = set()
    for nid, actions in transitions.items():
        valid_acts = [
            a
            for a, info in actions.items()
            if isinstance(info, dict) and info.get("total", 0) >= 3
        ]
        if len(valid_acts) >= 2:
            valid_nids.add(nid)
    logger.info(f"  Valid nids (2+ actions with >=3 visits): {len(valid_nids)}")

    # Pre-compute from ETG
    logger.info("Pre-computing Q-value advantage table...")
    t0 = time.time()
    q_adv = precompute_q_advantage(ETG, transitions)
    logger.info(f"  Done: {len(q_adv)} entries in {time.time() - t0:.1f}s")

    logger.info("Pre-computing Causal advantage table...")
    t0 = time.time()
    causal_adv = precompute_causal_advantage(transitions)
    logger.info(f"  Done: {len(causal_adv)} entries in {time.time() - t0:.1f}s")

    logger.info("Pre-computing PageRank advantage table...")
    t0 = time.time()
    h_adv = precompute_pagerank_advantage(ETG, transitions)
    logger.info(f"  Done: {len(h_adv)} entries in {time.time() - t0:.1f}s")

    logger.info("Pre-computing CFR advantage table...")
    t0 = time.time()
    cf_adv = precompute_cfr_advantage(ETG, transitions)
    logger.info(f"  Done: {len(cf_adv)} entries in {time.time() - t0:.1f}s")

    # Load episodes
    logger.info("Loading episode data...")
    t0 = time.time()
    summaries = load_summaries(args.run_dir)
    scores_all = [s["score"] for s in summaries]
    wins = sum(1 for s in summaries if s["result"] == "Win")
    logger.info(
        f"  {len(summaries)} episodes loaded in {time.time() - t0:.1f}s "
        f"(win rate: {wins / len(summaries) * 100:.1f}%)"
    )

    # Pre-compute from episodes
    logger.info("Pre-computing PathDiv & V2Baseline from episodes...")
    t0 = time.time()
    cramers_v, chi2_p, v2_score = precompute_pathdiv_and_v2(summaries, transitions)
    logger.info(
        f"  Done in {time.time() - t0:.1f}s "
        f"(PathDiv nids: {len(cramers_v)}, V2 nids: {len(v2_score)})"
    )

    # Score all frames
    logger.info("Scoring all frames...")
    t0 = time.time()
    scores_arr, labels_arr, ep_info, nid_counts = score_all_frames(
        summaries,
        q_adv,
        causal_adv,
        h_adv,
        cf_adv,
        cramers_v,
        chi2_p,
        v2_score,
        valid_nids,
    )
    logger.info(f"  {labels_arr.shape[0]} frames scored in {time.time() - t0:.1f}s")

    # Report
    print_report(scores_arr, labels_arr, ep_info, nid_counts, output_path)

    logger.info(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
