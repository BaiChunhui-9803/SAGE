import json
from typing import Any, Dict, List, Tuple

import numpy as np

from kg_web.constants import _COMPOSITE_WEIGHTS
from src.decision.kg_beam_search import BeamSearchResult


def _compute_path_metrics(path: List[BeamSearchResult]) -> Dict[str, Any]:
    non_root = path[1:] if len(path) > 1 else path
    length = len(non_root)
    if length == 0:
        return {
            "length": 0,
            "first_action": "",
            "first_state": path[0].state if path else None,
            "last_state": path[0].state if path else None,
            "confidence": 1.0,
            "avg_quality": 0.0,
            "avg_win_rate": 0.0,
            "last_quality": path[0].quality_score if path else 0.0,
            "last_win_rate": path[0].win_rate if path else 0.0,
            "q_trend": "→",
            "wr_trend": "→",
            "avg_reward": 0.0,
        }
    cum_prob = path[-1].cumulative_probability
    avg_quality = float(np.mean([r.quality_score for r in non_root]))
    avg_win_rate = float(np.mean([r.win_rate for r in non_root]))
    avg_reward = float(np.mean([r.avg_future_reward for r in non_root]))
    last_quality = path[-1].quality_score
    last_win_rate = path[-1].win_rate
    mid = length // 2
    if mid > 0:
        q_first = np.mean([r.quality_score for r in non_root[:mid]])
        q_second = np.mean([r.quality_score for r in non_root[mid:]])
        wr_first = np.mean([r.win_rate for r in non_root[:mid]])
        wr_second = np.mean([r.win_rate for r in non_root[mid:]])
        q_trend = (
            "↑"
            if q_second > q_first * 1.05
            else ("↓" if q_second < q_first * 0.95 else "→")
        )
        wr_trend = (
            "↑"
            if wr_second > wr_first * 1.05
            else ("↓" if wr_second < wr_first * 0.95 else "→")
        )
    else:
        q_trend = "→"
        wr_trend = "→"
    return {
        "length": length,
        "first_action": path[1].action if len(path) > 1 else "",
        "first_state": path[1].state if len(path) > 1 else path[0].state,
        "last_state": path[-1].state,
        "confidence": cum_prob,
        "avg_quality": avg_quality,
        "avg_win_rate": avg_win_rate,
        "last_quality": last_quality,
        "last_win_rate": last_win_rate,
        "q_trend": q_trend,
        "wr_trend": wr_trend,
        "avg_reward": avg_reward,
    }


def _compute_composite_scores(
    beam_paths: List[List[BeamSearchResult]],
) -> Tuple[List[float], List[Dict[str, Any]]]:
    path_metrics = [_compute_path_metrics(p) for p in beam_paths]
    raw_confs = [m["confidence"] for m in path_metrics]
    raw_quals = [m["avg_quality"] for m in path_metrics]
    raw_wrs = [m["avg_win_rate"] for m in path_metrics]
    raw_rewards = [m["avg_reward"] for m in path_metrics]

    def _norm(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

    norm_confs = _norm(raw_confs)
    norm_quals = _norm(raw_quals)
    norm_wrs = _norm(raw_wrs)
    norm_rewards = _norm(raw_rewards)

    w_conf, w_qual, w_wr, w_rwd = _COMPOSITE_WEIGHTS
    composites = [
        w_conf * nc + w_qual * nq + w_wr * nw + w_rwd * nr
        for nc, nq, nw, nr in zip(norm_confs, norm_quals, norm_wrs, norm_rewards)
    ]
    return composites, path_metrics


def _build_rec_rows(
    sorted_indices: List[int],
    beam_paths: List[List[BeamSearchResult]],
    composites: List[float],
    path_metrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for display_order, orig_idx in enumerate(sorted_indices):
        m = path_metrics[orig_idx]
        rows.append(
            {
                "#": display_order + 1,
                "Path": orig_idx,
                "长度": m["length"],
                "首步动作": m["first_action"],
                "首步目标": m["first_state"],
                "末状态": m["last_state"],
                "置信度": f"{m['confidence']:.4f}",
                "平均Quality": f"{m['avg_quality']:.1f}",
                "平均胜率": f"{m['avg_win_rate']:.2%}",
                "末Quality": f"{m['last_quality']:.1f}",
                "末胜率": f"{m['last_win_rate']:.2%}",
                "Q趋势": m["q_trend"],
                "WR趋势": m["wr_trend"],
                "平均奖励": f"{m['avg_reward']:.1f}",
                "综合评分": f"{composites[orig_idx]:.3f}",
            }
        )
    return rows


def _build_path_detail_rows(
    beam_paths: List[List[BeamSearchResult]],
) -> List[Dict[str, Any]]:
    sorted_paths = sorted(
        enumerate(beam_paths),
        key=lambda x: tuple((r.state, r.action) for r in x[1]),
    )
    all_rows = []
    prev_path_states: List[int] = []
    for orig_idx, path in sorted_paths:
        cur_states = [r.state for r in path]
        fork_point = 0
        for i in range(min(len(cur_states), len(prev_path_states))):
            if cur_states[i] != prev_path_states[i]:
                break
            fork_point = i + 1
        for i, r in enumerate(path):
            if i < fork_point and orig_idx > 0:
                fork_mark = "↑"
            elif i == fork_point and orig_idx > 0:
                fork_mark = "➤"
            else:
                fork_mark = ""
            all_rows.append(
                {
                    "Path": orig_idx,
                    "分叉": fork_mark,
                    "Step": r.step,
                    "State": r.state,
                    "Action": r.action,
                    "Cum Prob": f"{r.cumulative_probability:.4f}",
                    "Quality": f"{r.quality_score:.2f}",
                    "Win Rate": f"{r.win_rate:.2%}",
                    "Step Reward": f"{r.avg_step_reward:.3f}",
                    "Future Reward": f"{r.avg_future_reward:.3f}",
                }
            )
        prev_path_states = cur_states
    return all_rows


def _results_to_json(
    results: list,
    beam_width: int,
    max_steps: int,
    min_cum_prob: float,
    score_mode: str,
    start_state: int,
) -> str:
    nodes = []
    for i, r in enumerate(results):
        nodes.append(
            {
                "index": i,
                "step": r.step,
                "state": r.state,
                "action": r.action,
                "cumulative_probability": r.cumulative_probability,
                "quality_score": r.quality_score,
                "win_rate": r.win_rate,
                "avg_step_reward": r.avg_step_reward,
                "avg_future_reward": r.avg_future_reward,
                "beam_id": r.beam_id,
                "parent_idx": r.parent_idx,
            }
        )
    edges = [
        {"source": r.parent_idx, "target": idx}
        for idx, r in enumerate(results)
        if r.parent_idx is not None
    ]
    return json.dumps(
        {
            "meta": {
                "start_state": start_state,
                "beam_width": beam_width,
                "max_steps": max_steps,
                "min_cum_prob": min_cum_prob,
                "score_mode": score_mode,
                "total_nodes": len(results),
            },
            "nodes": nodes,
            "edges": edges,
        },
        indent=2,
        ensure_ascii=False,
    )
