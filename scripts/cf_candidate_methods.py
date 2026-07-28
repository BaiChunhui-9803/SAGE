#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cf_candidate_methods -- 6 ETG-based candidate identification methods
for counterfactual action override.

Methods:
  1. QValueMethod       - MDP Value Iteration -> Q(s,a) -> advantage
  2. CausalMethod       - Graph surgery (do-calculus approximation)
  3. PathDivergenceMethod - KL divergence / chi-squared on episode paths
  4. PageRankMethod     - Absorbing Markov chain hitting probability
  5. CFRMethod          - Counterfactual regret estimation
  6. StatisticalBaseline - V2 good/bad episode comparison

All methods output unified Candidate objects for fair comparison.

Usage:
    python scripts/experiment_cf_method_comparison.py --run_dir ...
"""

from __future__ import annotations

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decision.knowledge_graph import DecisionKnowledgeGraph, ActionStats

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    nid: int
    bad_action: str
    recommended_action: str
    score: float
    method: str = ""
    metadata: dict = field(default_factory=dict)

    def key(self) -> Tuple[int, str, str]:
        return (self.nid, self.bad_action, self.recommended_action)


def load_kg_and_transitions(kg_path: str, trans_path: str = None):
    if trans_path is None:
        candidate1 = kg_path.replace(".pkl", "_transitions.pkl")
        candidate2 = kg_path.replace(".pkl", ".transitions.pkl")
        if os.path.exists(candidate1):
            trans_path = candidate1
        elif os.path.exists(candidate2):
            trans_path = candidate2
        else:
            raise FileNotFoundError(
                f"Transitions file not found. Tried:\n  {candidate1}\n  {candidate2}"
            )
    kg = DecisionKnowledgeGraph.load(kg_path)
    import pickle

    with open(trans_path, "rb") as f:
        transitions = pickle.load(f)
    return kg, transitions


def load_episodes(run_dir: Path, trial_numbers: List[int] = None):
    trials_dir = run_dir / "trials"
    if not trials_dir.exists():
        return []
    episodes = []
    trial_dirs = sorted(trials_dir.iterdir()) if trials_dir.exists() else []
    for td in trial_dirs:
        if not td.is_dir() or not td.name.startswith("trial_"):
            continue
        try:
            tn = int(td.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        if trial_numbers is not None and tn not in trial_numbers:
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
                    episodes.append(ep)
                except json.JSONDecodeError:
                    pass
    return episodes


class QValueMethod:
    """MDP Value Iteration on ETG -> Q(s,a) -> advantage ranking."""

    def __init__(
        self,
        kg: DecisionKnowledgeGraph,
        transitions: dict,
        gamma: float = 0.95,
        min_visits: int = 5,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-3,
    ):
        self.kg = kg
        self.transitions = transitions
        self.gamma = gamma
        self.min_visits = min_visits
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._values: Dict[int, float] = {}
        self._q_values: Dict[int, Dict[str, float]] = {}

    def name(self) -> str:
        return "Q-value (VI)"

    def run(self) -> List[Candidate]:
        t0 = time.time()
        self._value_iteration()
        candidates = self._extract_candidates()
        elapsed = time.time() - t0
        logger.info(f"  [{self.name()}] {len(candidates)} candidates in {elapsed:.3f}s")
        return candidates

    def _value_iteration(self):
        states = set()
        for nid in self.transitions:
            states.add(nid)
            for a, info in self.transitions[nid].items():
                if not isinstance(info, dict):
                    continue
                for ns in info.get("next_states", {}):
                    states.add(ns)

        values = {s: 0.0 for s in states}

        for iteration in range(self.max_iterations):
            new_values = dict(values)
            max_delta = 0.0

            for s in states:
                trans_s = self.transitions.get(s, {})
                if "__terminal__" in trans_s:
                    continue

                best_q = float("-inf")
                for a, info in trans_s.items():
                    if not isinstance(info, dict):
                        continue
                    total = info.get("total", 0)
                    if total < self.min_visits:
                        continue
                    ns_dict = info.get("next_states", {})
                    if not ns_dict:
                        continue

                    quality = self.kg.get_action_quality(s, a)
                    reward = quality.get("avg_step_reward", 0.0) if quality else 0.0

                    expected_future = 0.0
                    for ns, cnt in ns_dict.items():
                        prob = cnt / total
                        expected_future += prob * values.get(ns, 0.0)

                    q = reward + self.gamma * expected_future
                    best_q = max(best_q, q)

                    if s not in self._q_values:
                        self._q_values[s] = {}
                    self._q_values[s][a] = q

                if best_q > float("-inf"):
                    new_values[s] = best_q
                    max_delta = max(max_delta, abs(new_values[s] - values[s]))

            values = new_values
            if max_delta < self.convergence_threshold:
                logger.info(f"  [{self.name()}] Converged at iteration {iteration + 1}")
                break

        self._values = values

    def _extract_candidates(self) -> List[Candidate]:
        candidates = []
        for nid, action_qs in self._q_values.items():
            if len(action_qs) < 2:
                continue

            sorted_actions = sorted(action_qs.items(), key=lambda x: -x[1])
            best_action, best_q = sorted_actions[0]

            for action, q in sorted_actions[1:]:
                delta_q = best_q - q
                if delta_q <= 0:
                    continue
                quality = self.kg.get_action_quality(nid, action)
                visits = quality.get("visits", 0) if quality else 0
                if visits < self.min_visits:
                    continue

                candidates.append(
                    Candidate(
                        nid=nid,
                        bad_action=action,
                        recommended_action=best_action,
                        score=delta_q,
                        method=self.name(),
                        metadata={
                            "q_bad": q,
                            "q_best": best_q,
                            "visits": visits,
                        },
                    )
                )

        candidates.sort(key=lambda c: -c.score)
        return candidates


class CausalMethod:
    """Graph surgery (do-calculus approximation) on ETG.

    Treats ETG as an approximate DAG (ignores self-loops, limits propagation depth).
    For each (nid, action), estimates P(win | do(action)) via forward propagation.
    """

    def __init__(
        self,
        kg: DecisionKnowledgeGraph,
        transitions: dict,
        min_visits: int = 5,
        max_propagation_depth: int = 10,
    ):
        self.kg = kg
        self.transitions = transitions
        self.min_visits = min_visits
        self.max_depth = max_propagation_depth

    def name(self) -> str:
        return "Causal (do-calculus)"

    def run(self) -> List[Candidate]:
        t0 = time.time()
        candidates = []
        for nid in self.transitions:
            trans_nid = self.transitions[nid]
            if "__terminal__" in trans_nid:
                continue

            valid_actions = {}
            for a, info in trans_nid.items():
                if not isinstance(info, dict):
                    continue
                if info.get("total", 0) < self.min_visits:
                    continue
                ns_dict = info.get("next_states", {})
                if ns_dict:
                    valid_actions[a] = info

            if len(valid_actions) < 2:
                continue

            action_win_probs = {}
            for a, info in valid_actions.items():
                ns_dict = info["next_states"]
                total = info["total"]
                win_prob = self._propagate_win_probability(nid, a, ns_dict, total)
                action_win_probs[a] = win_prob

            sorted_actions = sorted(action_win_probs.items(), key=lambda x: -x[1])
            best_action, best_prob = sorted_actions[0]

            for action, prob in sorted_actions[1:]:
                ate = best_prob - prob
                if ate <= 0:
                    continue
                quality = self.kg.get_action_quality(nid, action)
                visits = quality.get("visits", 0) if quality else 0
                candidates.append(
                    Candidate(
                        nid=nid,
                        bad_action=action,
                        recommended_action=best_action,
                        score=ate,
                        method=self.name(),
                        metadata={
                            "ate": ate,
                            "do_win_prob_best": best_prob,
                            "do_win_prob_bad": prob,
                            "visits": visits,
                        },
                    )
                )

        candidates.sort(key=lambda c: -c.score)
        elapsed = time.time() - t0
        logger.info(f"  [{self.name()}] {len(candidates)} candidates in {elapsed:.3f}s")
        return candidates

    def _propagate_win_probability(
        self, source_nid: int, action: str, ns_dict: dict, total: int
    ) -> float:
        visited = {source_nid}
        current = {}
        for ns, cnt in ns_dict.items():
            prob = cnt / total if total > 0 else 0.0
            if ns != source_nid:
                current[ns] = prob

        cum_win_prob = 0.0

        for depth in range(self.max_depth):
            if not current:
                break

            next_layer = defaultdict(float)
            for nid, reach_prob in current.items():
                if nid in visited:
                    continue

                trans_nid = self.transitions.get(nid, {})
                if "__terminal__" in trans_nid:
                    wr = self.kg.get_action_quality(nid, "0a")
                    terminal_wr = wr.get("win_rate", 0.0) if wr else 0.0
                    cum_win_prob += reach_prob * terminal_wr
                    visited.add(nid)
                    continue

                best_wr = 0.0
                best_action_info = None
                for a, info in trans_nid.items():
                    if not isinstance(info, dict):
                        continue
                    if info.get("total", 0) < 1:
                        continue
                    wr = info.get("win_rate", 0.0)
                    if wr > best_wr:
                        best_wr = wr
                        best_action_info = info

                if best_wr >= 0.95:
                    cum_win_prob += reach_prob * best_wr
                    visited.add(nid)
                    continue

                if best_action_info is None:
                    quality = (
                        self.kg.get_action_quality(nid, list(trans_nid.keys())[0])
                        if trans_nid
                        else None
                    )
                    fallback_wr = quality.get("win_rate", 0.0) if quality else 0.0
                    cum_win_prob += reach_prob * fallback_wr
                    visited.add(nid)
                    continue

                ns2 = best_action_info.get("next_states", {})
                t2 = best_action_info.get("total", 1)
                for ns, cnt2 in ns2.items():
                    if ns != nid and ns not in visited:
                        next_layer[ns] += reach_prob * (cnt2 / t2)

                visited.add(nid)

            current = dict(next_layer)

        return cum_win_prob


class PathDivergenceMethod:
    """KL divergence / chi-squared test on episode paths.

    For each nid, compare the action-outcome contingency table between
    good and bad episodes. Use chi-squared test for independence and
    Cramer's V for effect size.
    """

    def __init__(self, episodes: list, min_count: int = 5):
        self.episodes = episodes
        self.min_count = min_count

    def name(self) -> str:
        return "Path Divergence (chi2)"

    def run(self) -> List[Candidate]:
        t0 = time.time()

        scores = [ep.get("score", 0) for ep in self.episodes]
        if not scores:
            return []
        p25 = float(np.percentile(scores, 25))
        p75 = float(np.percentile(scores, 75))

        nid_action_stats = defaultdict(
            lambda: defaultdict(
                lambda: {"good": 0, "bad": 0, "neutral": 0, "good_hp": [], "bad_hp": []}
            )
        )

        for ep in self.episodes:
            score = ep.get("score", 0)
            if score >= p75:
                group = "good"
            elif score <= p25:
                group = "bad"
            else:
                continue

            frames = ep.get("frames", [])
            for fr in frames:
                nid = fr.get("nid")
                action_code = fr.get("action_code", "")
                hp_delta = fr.get("hp_delta", 0)
                if nid is None or not action_code:
                    continue
                nid_action_stats[nid][action_code][group] += 1
                if group == "good":
                    nid_action_stats[nid][action_code]["good_hp"].append(hp_delta)
                else:
                    nid_action_stats[nid][action_code]["bad_hp"].append(hp_delta)

        candidates = []
        for nid, actions in nid_action_stats.items():
            total_good = sum(a["good"] for a in actions.values())
            total_bad = sum(a["bad"] for a in actions.values())
            if total_good < self.min_count or total_bad < self.min_count:
                continue

            action_list = [
                a for a in actions if actions[a]["good"] >= 3 and actions[a]["bad"] >= 3
            ]
            if len(action_list) < 2:
                continue

            best_action = max(
                action_list,
                key=lambda a: (
                    np.mean(actions[a]["good_hp"]) if actions[a]["good_hp"] else 0
                ),
            )
            worst_action = min(
                action_list,
                key=lambda a: (
                    np.mean(actions[a]["bad_hp"]) if actions[a]["bad_hp"] else 0
                ),
            )
            if best_action == worst_action:
                continue

            good_hp_best = (
                np.mean(actions[best_action]["good_hp"])
                if actions[best_action]["good_hp"]
                else 0
            )
            bad_hp_worst = (
                np.mean(actions[worst_action]["bad_hp"])
                if actions[worst_action]["bad_hp"]
                else 0
            )
            potential = good_hp_best - bad_hp_worst
            if potential <= 0:
                continue

            n_actions = len(action_list)
            contingency = np.zeros((n_actions, 2))
            for i, a in enumerate(action_list):
                contingency[i, 0] = actions[a]["good"]
                contingency[i, 1] = actions[a]["bad"]

            if np.any(contingency == 0):
                contingency += 0.5

            from scipy.stats import chi2_contingency

            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
            except Exception:
                continue

            if p_value >= 0.05:
                continue

            n = contingency.sum()
            min_dim = min(contingency.shape)
            cramers_v = np.sqrt(chi2 / (n * (min_dim - 1))) if min_dim > 1 else 0.0

            score = potential * cramers_v * min(actions[best_action]["good"] / 10, 3.0)

            candidates.append(
                Candidate(
                    nid=nid,
                    bad_action=worst_action,
                    recommended_action=best_action,
                    score=score,
                    method=self.name(),
                    metadata={
                        "chi2": float(chi2),
                        "p_value": float(p_value),
                        "cramers_v": float(cramers_v),
                        "potential": float(potential),
                        "good_count": actions[best_action]["good"],
                        "bad_count": actions[worst_action]["bad"],
                    },
                )
            )

        candidates.sort(key=lambda c: -c.score)
        elapsed = time.time() - t0
        logger.info(f"  [{self.name()}] {len(candidates)} candidates in {elapsed:.3f}s")
        return candidates


class PageRankMethod:
    """Absorbing Markov chain hitting probability.

    Build transition matrix with Win/Loss as absorbing states.
    Compute h(s) = P(reach Win | start at s).
    Decompose to action level: h(nid, a) = sum P(s'|nid,a) * h(s').
    """

    def __init__(
        self,
        kg: DecisionKnowledgeGraph,
        transitions: dict,
        min_visits: int = 5,
        max_iterations: int = 200,
        damping: float = 0.85,
    ):
        self.kg = kg
        self.transitions = transitions
        self.min_visits = min_visits
        self.max_iterations = max_iterations
        self.damping = damping
        self._hitting_probs: Dict[int, float] = {}

    def name(self) -> str:
        return "PageRank (Hitting)"

    def run(self) -> List[Candidate]:
        t0 = time.time()
        self._compute_hitting_probabilities()
        candidates = self._extract_candidates()
        elapsed = time.time() - t0
        logger.info(f"  [{self.name()}] {len(candidates)} candidates in {elapsed:.3f}s")
        return candidates

    def _compute_hitting_probabilities(self):
        states = set()
        for nid in self.transitions:
            states.add(nid)
            for a, info in self.transitions[nid].items():
                if not isinstance(info, dict):
                    continue
                for ns in info.get("next_states", {}):
                    states.add(ns)

        h = {}
        for s in states:
            trans_s = self.transitions.get(s, {})
            best_wr = 0.0
            for a, info in trans_s.items():
                if not isinstance(info, dict):
                    continue
                if info.get("total", 0) >= 1:
                    best_wr = max(best_wr, info.get("win_rate", 0.0))
            h[s] = best_wr

        for iteration in range(self.max_iterations):
            max_delta = 0.0
            new_h = {}

            for s in states:
                trans_s = self.transitions.get(s, {})
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
                    self.kg.get_action_quality(s, list(trans_s.keys())[0])
                    if trans_s
                    else None
                )
                immediate_wr = quality.get("win_rate", 0.0) if quality else 0.0

                new_h[s] = self.damping * best_h + (1 - self.damping) * immediate_wr
                max_delta = max(max_delta, abs(new_h[s] - h.get(s, 0.0)))

            h = new_h
            if max_delta < 1e-4:
                logger.info(f"  [{self.name()}] Converged at iteration {iteration + 1}")
                break

        self._hitting_probs = h

    def _extract_candidates(self) -> List[Candidate]:
        candidates = []
        for nid in self.transitions:
            trans_nid = self.transitions[nid]
            if "__terminal__" in trans_nid:
                continue

            action_hitting = {}
            for a, info in trans_nid.items():
                if not isinstance(info, dict):
                    continue
                total = info.get("total", 0)
                if total < self.min_visits:
                    continue
                ns_dict = info.get("next_states", {})
                if not ns_dict:
                    continue

                h_a = sum(
                    (cnt / total) * self._hitting_probs.get(ns, 0.0)
                    for ns, cnt in ns_dict.items()
                )
                action_hitting[a] = h_a

            if len(action_hitting) < 2:
                continue

            sorted_actions = sorted(action_hitting.items(), key=lambda x: -x[1])
            best_action, best_h = sorted_actions[0]

            for action, h_val in sorted_actions[1:]:
                delta_h = best_h - h_val
                if delta_h <= 0:
                    continue
                quality = self.kg.get_action_quality(nid, action)
                visits = quality.get("visits", 0) if quality else 0

                candidates.append(
                    Candidate(
                        nid=nid,
                        bad_action=action,
                        recommended_action=best_action,
                        score=delta_h,
                        method=self.name(),
                        metadata={
                            "h_bad": h_val,
                            "h_best": best_h,
                            "visits": visits,
                        },
                    )
                )

        candidates.sort(key=lambda c: -c.score)
        return candidates


class CFRMethod:
    """Counterfactual Regret estimation from episode data + ETG.

    For each episode visit to (nid, action), estimate the counterfactual
    value of alternative actions using ETG downstream statistics.
    Accumulate regret per (nid, alt_action) pair.
    """

    def __init__(
        self,
        episodes: list,
        kg: DecisionKnowledgeGraph,
        transitions: dict,
        min_regret_samples: int = 5,
    ):
        self.episodes = episodes
        self.kg = kg
        self.transitions = transitions
        self.min_samples = min_regret_samples

    def name(self) -> str:
        return "CFR (Regret)"

    def run(self) -> List[Candidate]:
        t0 = time.time()

        nid_action_values = defaultdict(
            lambda: defaultdict(lambda: {"actual_values": [], "cf_values": []})
        )

        for ep in self.episodes:
            frames = ep.get("frames", [])
            outcome = ep.get("result", "Unknown")
            score = ep.get("score", 0)
            is_win = outcome == "Win"

            for i, fr in enumerate(frames):
                nid = fr.get("nid")
                action_code = fr.get("action_code", "")
                if nid is None or not action_code:
                    continue

                remaining_hp = sum(f.get("hp_delta", 0) for f in frames[i:])
                actual_value = remaining_hp + (50 if is_win else 0)

                nid_action_values[nid][action_code]["actual_values"].append(
                    actual_value
                )

                trans_nid = self.transitions.get(nid, {})
                for alt_action, info in trans_nid.items():
                    if not isinstance(info, dict):
                        continue
                    if alt_action == action_code:
                        continue
                    if info.get("total", 0) < 3:
                        continue

                    ns_dict = info.get("next_states", {})
                    if not ns_dict:
                        continue
                    total = info["total"]

                    cf_value = 0.0
                    for ns, cnt in ns_dict.items():
                        prob = cnt / total
                        ns_quality = self.kg.get_action_quality(ns, alt_action)
                        if ns_quality:
                            future_val = ns_quality.get("avg_future_reward", 0.0)
                            ns_wr = ns_quality.get("win_rate", 0.0)
                            cf_value += prob * (future_val + ns_wr * 50)
                        else:
                            cf_value += prob * 0.0

                    nid_action_values[nid][alt_action]["cf_values"].append(cf_value)

        candidates = []
        for nid, actions in nid_action_values.items():
            actual_actions = {a: v for a, v in actions.items() if v["actual_values"]}
            if not actual_actions:
                continue

            for actual_action, actual_data in actual_actions.items():
                avg_actual = float(np.mean(actual_data["actual_values"]))
                n_actual = len(actual_data["actual_values"])
                if n_actual < self.min_samples:
                    continue

                for alt_action, alt_data in actions.items():
                    if alt_action == actual_action:
                        continue
                    if not alt_data["cf_values"]:
                        continue

                    avg_cf = float(np.mean(alt_data["cf_values"]))
                    regret = avg_cf - avg_actual
                    if regret <= 0:
                        continue
                    if len(alt_data["cf_values"]) < self.min_samples:
                        continue

                    candidates.append(
                        Candidate(
                            nid=nid,
                            bad_action=actual_action,
                            recommended_action=alt_action,
                            score=regret,
                            method=self.name(),
                            metadata={
                                "avg_actual": avg_actual,
                                "avg_cf": avg_cf,
                                "regret": regret,
                                "n_actual": n_actual,
                                "n_cf_samples": len(alt_data["cf_values"]),
                            },
                        )
                    )

        candidates.sort(key=lambda c: -c.score)
        elapsed = time.time() - t0
        logger.info(f"  [{self.name()}] {len(candidates)} candidates in {elapsed:.3f}s")
        return candidates


class StatisticalBaseline:
    """V2 good/bad episode statistical comparison (baseline).

    For each nid, compare action distributions between good (>=p75 score)
    and bad (<=p25 score) episodes. Score = potential * count_factor.
    """

    def __init__(self, episodes: list, min_count: int = 5):
        self.episodes = episodes
        self.min_count = min_count

    def name(self) -> str:
        return "V2 Baseline (Stat)"

    def run(self) -> List[Candidate]:
        t0 = time.time()

        scores = [ep.get("score", 0) for ep in self.episodes]
        if not scores:
            return []
        p25 = float(np.percentile(scores, 25))
        p75 = float(np.percentile(scores, 75))

        nid_action_stats = defaultdict(
            lambda: defaultdict(
                lambda: {"good_count": 0, "bad_count": 0, "good_hp": [], "bad_hp": []}
            )
        )

        for ep in self.episodes:
            score = ep.get("score", 0)
            if score >= p75:
                group = "good"
            elif score <= p25:
                group = "bad"
            else:
                continue

            frames = ep.get("frames", [])
            for fr in frames:
                nid = fr.get("nid")
                action_code = fr.get("action_code", "")
                hp_delta = fr.get("hp_delta", 0)
                if nid is None or not action_code:
                    continue
                if group == "good":
                    nid_action_stats[nid][action_code]["good_count"] += 1
                    nid_action_stats[nid][action_code]["good_hp"].append(hp_delta)
                else:
                    nid_action_stats[nid][action_code]["bad_count"] += 1
                    nid_action_stats[nid][action_code]["bad_hp"].append(hp_delta)

        candidates = []
        for nid, actions in nid_action_stats.items():
            shared_actions = []
            for a, s in actions.items():
                if (
                    s["good_count"] >= self.min_count
                    and s["bad_count"] >= self.min_count
                ):
                    shared_actions.append(a)
            if len(shared_actions) < 1:
                continue

            total_good = sum(actions[a]["good_count"] for a in actions)
            total_bad = sum(actions[a]["bad_count"] for a in actions)

            for bad_action in shared_actions:
                bad_hp = (
                    np.mean(nid_action_stats[nid][bad_action]["bad_hp"])
                    if nid_action_stats[nid][bad_action]["bad_hp"]
                    else 0
                )

                for good_action in actions:
                    if good_action == bad_action:
                        continue
                    gc = actions[good_action]["good_count"]
                    bc_in_good = actions[good_action]["bad_count"]
                    if gc < self.min_count:
                        continue
                    if gc <= bc_in_good:
                        continue

                    good_hp = (
                        np.mean(actions[good_action]["good_hp"])
                        if actions[good_action]["good_hp"]
                        else 0
                    )
                    potential = good_hp - bad_hp
                    if potential <= 0:
                        continue

                    count_factor = min(gc / 10.0, 3.0)
                    score = potential * count_factor

                    candidates.append(
                        Candidate(
                            nid=nid,
                            bad_action=bad_action,
                            recommended_action=good_action,
                            score=score,
                            method=self.name(),
                            metadata={
                                "potential": float(potential),
                                "good_count": gc,
                                "bad_count": actions[bad_action]["bad_count"],
                                "good_avg_hp": float(good_hp),
                                "bad_avg_hp": float(bad_hp),
                            },
                        )
                    )

        candidates.sort(key=lambda c: -c.score)
        elapsed = time.time() - t0
        logger.info(f"  [{self.name()}] {len(candidates)} candidates in {elapsed:.3f}s")
        return candidates


def compute_jaccard_matrix(
    all_candidates: Dict[str, List[Candidate]], top_k: int = 20
) -> Dict[str, Dict[str, float]]:
    methods = list(all_candidates.keys())
    top_sets = {}
    for m in methods:
        cands = all_candidates[m][:top_k]
        top_sets[m] = set(c.key() for c in cands)

    matrix = {}
    for m1 in methods:
        matrix[m1] = {}
        for m2 in methods:
            s1 = top_sets.get(m1, set())
            s2 = top_sets.get(m2, set())
            if not s1 and not s2:
                matrix[m1][m2] = 1.0
            elif not s1 or not s2:
                matrix[m1][m2] = 0.0
            else:
                matrix[m1][m2] = len(s1 & s2) / len(s1 | s2)
    return matrix


def find_consensus(
    all_candidates: Dict[str, List[Candidate]], top_k: int = 50
) -> List[Tuple[Candidate, List[str]]]:
    cand_methods = defaultdict(list)
    cand_data = {}

    for method, cands in all_candidates.items():
        for c in cands[:top_k]:
            key = c.key()
            cand_methods[key].append(method)
            if key not in cand_data:
                cand_data[key] = c

    consensus = []
    for key, methods in cand_methods.items():
        if len(methods) >= 2:
            consensus.append((cand_data[key], methods))

    consensus.sort(key=lambda x: -len(x[1]))
    return consensus
