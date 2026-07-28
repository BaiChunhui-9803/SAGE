import json
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import networkx as nx
from pyvis.network import Network

from kg_web.constants import _BEAM_COLORS
from src.decision.kg_beam_search import BeamSearchResult


def render_pyvis_html(
    nodes: List[Dict],
    edges: List[Dict],
    state_stats: Dict[int, Dict],
    highlight_terminal: bool = True,
    edge_smooth_type: str = "continuous",
    edge_roundness: float = 0.25,
    layout_algorithm: str = "barnes_hut",
    freeze_layout: bool = False,
    canvas_height: int = 750,
) -> str:
    net = Network(
        height="100%",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
        select_menu=False,
        filter_menu=False,
    )

    _LAYOUT_DEFAULT_PARAMS = {
        "barnes_hut": {
            "gravity": -8000,
            "central_gravity": 0.3,
            "spring_length": 250,
            "spring_strength": 0.001,
            "damping": 0.09,
            "overlap": 0,
        },
        "force_atlas_2based": {
            "gravity": -50,
            "central_gravity": 0.01,
            "spring_length": 100,
            "spring_strength": 0.08,
            "damping": 0.4,
            "overlap": 0,
        },
        "repulsion": {
            "node_distance": 100,
            "central_gravity": 0.2,
            "spring_length": 200,
            "spring_strength": 0.05,
            "damping": 0.09,
        },
        "hrepulsion": {
            "node_distance": 120,
            "central_gravity": 0.0,
            "spring_length": 100,
            "spring_strength": 0.01,
            "damping": 0.09,
        },
    }
    net.set_options("""
    {
      "tooltip": {
        "style": "white-space: pre-line; font-family: monospace; background: #222; color: #fff; padding: 8px; border-radius: 4px;"
      }
    }
    """)

    from pyvis.physics import Physics

    physics = Physics()
    physics_fn = {
        "barnes_hut": physics.use_barnes_hut,
        "force_atlas_2based": physics.use_force_atlas_2based,
        "repulsion": physics.use_repulsion,
        "hrepulsion": physics.use_hrepulsion,
    }
    physics_fn[layout_algorithm](_LAYOUT_DEFAULT_PARAMS[layout_algorithm])
    net.options["physics"] = json.loads(physics.to_json())
    if freeze_layout:
        net.options["physics"]["enabled"] = False

    smooth_opt: dict = {"type": edge_smooth_type}
    if edge_smooth_type == "manual_arc":
        smooth_opt["roundness"] = edge_roundness
    net.options["edges"] = {"smooth": smooth_opt}

    max_visits = max((n["total_visits"] for n in nodes), default=1) or 1
    max_edge_visits = max((e["visits"] for e in edges), default=1) or 1

    edge_pair_count = defaultdict(int)
    edge_pair_index = defaultdict(int)
    for e in edges:
        edge_pair_count[(e["from"], e["to"])] += 1

    for n in nodes:
        sid = n["id"]
        visits = n["total_visits"]
        win_rate = n["best_win_rate"]
        quality = n["best_quality"]

        size = 10 + 30 * (visits / max_visits)

        is_terminal = n.get("is_terminal", False)
        node_shape = "dot"

        if is_terminal:
            color = "rgba(255, 80, 80, 0.9)"
            node_shape = "diamond"
        elif highlight_terminal and win_rate >= 0.5:
            color = f"rgba({int(255 * (1 - win_rate))}, {int(200 + 55 * win_rate)}, 80, 0.9)"
        elif highlight_terminal and win_rate <= 0.1 and visits > 5:
            color = f"rgba(255, {int(80 + 100 * win_rate)}, 80, 0.9)"
        else:
            r = int(100 + 155 * (1 - max(0, min(1, (win_rate + 0.2) / 1.2))))
            g = int(100 + 155 * max(0, min(1, (win_rate + 0.2) / 1.2)))
            color = f"rgba({r}, {g}, 180, 0.85)"

        title_parts = [
            f"<b>State {sid}</b>",
            f"Total Visits: {visits}",
            f"Available Actions: {n['n_actions']}",
            f"Best Win Rate: {win_rate * 100:.1f}%",
            f"Best Quality: {quality:.1f}",
        ]
        if is_terminal:
            title_parts.insert(1, "⚠️ Terminal State")
        title = "\n".join(title_parts)

        net.add_node(
            sid,
            label=str(sid),
            shape=node_shape if is_terminal else "dot",
            size=size if not is_terminal else size * 1.3,
            color=color,
            title=title,
            borderWidth=3 if is_terminal else 2,
            borderWidthSelected=4 if is_terminal else 4,
        )

    for e in edges:
        width = 1 + 4 * (e["visits"] / max_edge_visits)
        quality_norm = max(0, min(1, (e["quality_score"] + 40) / 60))
        r = int(255 * (1 - quality_norm))
        g = int(100 + 155 * quality_norm)
        b = int(150 + 105 * quality_norm)
        edge_color = f"rgba({r}, {g}, {b}, 0.7)"

        label = f"{e['action']} ({e['transition_prob'] * 100:.0f}%)"
        hover = "\n".join(
            [
                f"{e['from']} -> {e['to']}",
                f"Action: {e['action']}",
                f"Visits: {e['visits']}",
                f"Quality: {e['quality_score']:.1f}",
                f"Win Rate: {e['win_rate'] * 100:.1f}%",
                f"Avg Step Reward: {e['avg_step_reward']:.2f}",
                f"Avg Future Reward: {e['avg_future_reward']:.2f}",
                f"Transition Prob: {e['transition_prob'] * 100:.1f}%",
            ]
        )

        if edge_smooth_type == "continuous":
            smooth = {"type": "continuous"}
        else:
            key = (e["from"], e["to"])
            total = edge_pair_count[key]
            idx = edge_pair_index[key]
            edge_pair_index[key] += 1
            if total <= 1:
                smooth = {"type": "continuous"}
            else:
                roundness = (0.15 + 0.7 * idx / max(total - 1, 1)) * edge_roundness
                direction = "curvedCW" if idx % 2 == 0 else "curvedCCW"
                smooth = {"type": direction, "roundness": roundness}

        net.add_edge(
            e["from"],
            e["to"],
            label=label,
            title=hover,
            width=width,
            color=edge_color,
            arrows="to",
            font={"size": 8, "color": "#aaaaaa", "align": "middle"},
            smooth=smooth,
        )

    raw_html = net.generate_html()
    resize_js = """
<script>
(function() {
    document.querySelectorAll('center').forEach(function(el) { el.remove(); });
    document.documentElement.style.cssText = 'height:100%; margin:0; overflow:hidden;';
    document.body.style.cssText = 'height:100%; margin:0; padding:0; overflow:hidden;';
    var card = document.querySelector('.card');
    if (card) card.style.cssText = 'height:100%; width:100%; padding:0; margin:0;';
    var c = document.getElementById('mynetwork');
    if (!c) return;
    c.style.cssText = 'width:100%; height:100%; position:relative; float:none; border:none;';
    window.addEventListener('resize', function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    });
    setTimeout(function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    }, 1500);
})();
</script>"""
    return raw_html.replace("</body>", resize_js + "</body>")


def render_beam_tree(
    results: List[BeamSearchResult],
    highlight_indices: Optional[Set[int]] = None,
) -> str:
    if not results:
        return "<p>No results.</p>"

    g = nx.DiGraph()
    max_beam = max(r.beam_id for r in results)
    color_map = {i: _BEAM_COLORS[i % len(_BEAM_COLORS)] for i in range(max_beam + 1)}

    has_highlight = highlight_indices is not None and len(highlight_indices) > 0

    idx_map = {}
    for i, r in enumerate(results):
        idx_map[i] = i
        label = f"S{r.state}\nQ={r.quality_score:.1f}"

        if has_highlight:
            is_hl = i in highlight_indices
            if is_hl:
                node_color = "#FFD700"
                node_border = 4
                node_opacity = 1.0
                font_size = 16
            else:
                node_color = color_map[r.beam_id]
                node_border = 1
                node_opacity = 0.35
                font_size = 12
        else:
            node_color = color_map[r.beam_id]
            node_border = 3 if r.step == 0 else 1
            node_opacity = 1.0
            font_size = 14 if r.step == 0 else 13

        g.add_node(
            i,
            label=label,
            color=node_color,
            font={"size": font_size, "color": "white"},
            size=20 if r.step == 0 else 15,
            borderWidth=node_border,
            opacity=node_opacity,
            title=(
                f"State: {r.state}\n"
                f"Action: {r.action}\n"
                f"Step: {r.step}\n"
                f"Quality: {r.quality_score:.2f}\n"
                f"Win Rate: {r.win_rate:.2%}\n"
                f"Step Reward: {r.avg_step_reward:.3f}\n"
                f"Future Reward: {r.avg_future_reward:.3f}\n"
                f"Cum Prob: {r.cumulative_probability:.4f}"
            ),
        )

    for i, r in enumerate(results):
        if r.parent_idx is not None and r.parent_idx in idx_map:
            if has_highlight:
                edge_hl = i in highlight_indices and r.parent_idx in highlight_indices
                edge_color = "#FFD700" if edge_hl else "#666666"
                edge_width = 3 if edge_hl else 0.5
                edge_opacity = 1.0 if edge_hl else 0.3
            else:
                edge_color = color_map[r.beam_id]
                edge_width = 1
                edge_opacity = 1.0

            g.add_edge(
                idx_map[r.parent_idx],
                idx_map[i],
                title=f"{r.action} ({r.cumulative_probability:.3f})",
                label=f"{r.action} ({r.cumulative_probability:.1%})",
                color=edge_color,
                width=edge_width,
                font={"size": 10, "color": "#cccccc"},
                arrows="to",
                smooth={"type": "continuous"},
                opacity=edge_opacity,
            )

    net = Network(height="500px", width="100%", directed=True, notebook=False)

    for nid, ndata in g.nodes(data=True):
        net.add_node(nid, **ndata)
    for src, tgt, edata in g.edges(data=True):
        net.add_edge(src, tgt, **edata)

    options_dict = {
        "physics": {
            "enabled": False,
            "hierarchicalRepulsion": {
                "nodeDistance": 180,
                "centralGravity": 0.0,
                "springLength": 200,
                "springConstant": 0.01,
                "damping": 0.09,
            },
        },
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 250,
                "nodeSpacing": 180,
                "blockShifting": True,
                "edgeMinimization": True,
            },
        },
        "edges": {"smooth": {"type": "continuous", "roundness": 0.2}},
        "interaction": {"hover": True, "tooltipDelay": 50},
    }
    net.set_options(json.dumps(options_dict))

    raw_html = net.generate_html(notebook=False)

    resize_js = """<script>
(function() {
    var style = document.createElement('style');
    style.innerHTML = 'html, body { margin:0; padding:0; height:100%; overflow:hidden; } '
        + '.card { height:100%; padding:0; margin:0; } '
        + '#mynetwork { height:100%; width:100%; }';
    document.head.appendChild(style);
    var titles = document.querySelectorAll('center h1');
    titles.forEach(function(el) { el.remove(); });
    var centers = document.querySelectorAll('center');
    centers.forEach(function(el) { if(el.children.length === 0) el.remove(); });
    setTimeout(function() {
        if (window.network) {
            window.network.setSize('100%', '100%');
            window.network.redraw();
        }
    }, 500);
})();
</script>"""
    return raw_html.replace("</body>", resize_js + "</body>")
