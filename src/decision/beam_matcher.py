"""
Beam Path Matcher — Match beam search paths against original game replays.

For each beam path (state sequence), searches all episodes for the most
similar subsequence using a state distance matrix (fuzzy matching) and
exact action matching, returning top-K candidates with similarity scores
and match outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.decision.etg_beam_search import BeamSearchResult


@dataclass
class MatchResult:
    episode_id: int
    start_pos: int
    end_pos: int
    state_similarity: float
    action_match_rate: float
    combined_score: float
    outcome: str
    episode_score: float
    matched_states: List[int]
    matched_actions: List[str]


def match_beam_paths(
    beam_paths: Dict[int, List[BeamSearchResult]],
    episode_states: List[List[int]],
    episode_actions: List[List[str]],
    episode_outcomes: List[str],
    episode_scores: List[float],
    distance_matrix: Optional[np.ndarray],
    top_k: int = 3,
    state_weight: float = 0.6,
    max_state_distance: float = 5.0,
) -> Dict[int, List[MatchResult]]:
    """
    For each beam, find the top-K most similar subsequence in original replays.

    Args:
        beam_paths: {beam_id: [step_0, step_1, ...]} from get_beam_paths().
        episode_states: List of per-episode state sequences.
        episode_actions: List of per-episode action sequences ('4d' format).
        episode_outcomes: List of per-episode outcomes ('Win'/'Loss').
        episode_scores: List of per-episode final scores (from game_result.txt).
        distance_matrix: NxN state distance matrix, or None for exact-only.
        top_k: Number of candidates per beam.
        state_weight: Weight for state similarity in combined score (0~1).
        max_state_distance: Maximum average distance to consider (pruning).

    Returns:
        {beam_id: [MatchResult, ...]} sorted by combined_score descending.
    """
    if distance_matrix is not None and getattr(
        distance_matrix, "is_sparse_distance_index", False
    ):
        all_distances = [
            d
            for entries in distance_matrix.neighbors.values()
            for _sid, d in entries
            if np.isfinite(d)
        ]
        max_dist = max(all_distances) if all_distances else 1.0
    elif distance_matrix is not None:
        max_dist = float(np.max(distance_matrix))
        if max_dist < 1e-9:
            max_dist = 1.0
    else:
        max_dist = 1.0

    results: Dict[int, List[MatchResult]] = {}

    for bid in sorted(beam_paths.keys()):
        path = beam_paths[bid]
        if not path or len(path) < 2:
            results[bid] = []
            continue

        query_states = [r.state for r in path if r.action]
        query_actions = [r.action for r in path if r.action]

        if not query_states:
            results[bid] = []
            continue

        L = len(query_states)
        candidates: List[MatchResult] = []

        for ep_id in range(len(episode_states)):
            ep_s = episode_states[ep_id]
            ep_a = episode_actions[ep_id]
            outcome = (
                episode_outcomes[ep_id] if ep_id < len(episode_outcomes) else "Unknown"
            )
            ep_score = episode_scores[ep_id] if ep_id < len(episode_scores) else 0.0

            if len(ep_s) < L:
                continue

            for t in range(len(ep_s) - L + 1):
                seg_s = ep_s[t : t + L]
                seg_a = ep_a[t : t + L] if len(ep_a) >= t + L else None

                if distance_matrix is not None:
                    total_dist = 0.0
                    valid = True
                    for i in range(L):
                        si, qi = seg_s[i], query_states[i]
                        if (
                            si < distance_matrix.shape[0]
                            and qi < distance_matrix.shape[1]
                        ):
                            d = float(distance_matrix[si][qi])
                            total_dist += d if np.isfinite(d) else max_dist
                        else:
                            total_dist += max_dist
                    avg_dist = total_dist / L
                    if avg_dist > max_state_distance:
                        continue
                    state_sim = max(0.0, 1.0 - avg_dist / max_dist)
                else:
                    exact = sum(1 for i in range(L) if seg_s[i] == query_states[i])
                    state_sim = exact / L
                    if state_sim < 0.1:
                        continue

                if seg_a is not None and len(seg_a) >= L:
                    action_match = (
                        sum(
                            1
                            for i in range(L)
                            if i < len(seg_a) and seg_a[i] == query_actions[i]
                        )
                        / L
                    )
                else:
                    action_match = 0.0

                score = state_weight * state_sim + (1 - state_weight) * action_match

                candidates.append(
                    MatchResult(
                        episode_id=ep_id,
                        start_pos=t,
                        end_pos=t + L - 1,
                        state_similarity=round(state_sim, 4),
                        action_match_rate=round(action_match, 4),
                        combined_score=round(score, 4),
                        outcome=outcome,
                        episode_score=ep_score,
                        matched_states=seg_s,
                        matched_actions=seg_a[:L] if seg_a else [],
                    )
                )

        candidates.sort(key=lambda m: m.combined_score, reverse=True)
        results[bid] = candidates[:top_k]

    return results
