from typing import Dict, Optional

import numpy as np
import streamlit as st
from scipy.interpolate import griddata
from sklearn.manifold import MDS

from kg_web.constants import NPY_DIR
from kg_web.loaders import load_state_hp_data


@st.cache_data(max_entries=10, show_spinner="正在计算 MDS 降维...")
def compute_mds(_map_id: str, _data_id: str):
    cache_path = NPY_DIR / f"mds_{_map_id}_{_data_id}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))

    dist_path = NPY_DIR / f"state_distance_matrix_{_map_id}_{_data_id}.npy"
    if not dist_path.exists():
        return None
    dist_mat = np.load(str(dist_path))
    n = dist_mat.shape[0]

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        normalized_stress=False,
        random_state=42,
        n_init=4,
        max_iter=300,
        eps=1e-6,
        n_jobs=-1,
    )
    coords = mds.fit_transform(dist_mat)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), coords)
    return coords


def _build_hp_diff_array(hp_data: Dict, n_states: int) -> np.ndarray:
    hp_diff = np.full(n_states, np.nan)
    for sid, info in hp_data["hp_lookup"].items():
        red = info.get("red_army", [])
        blue = info.get("blue_army", [])
        if red or blue:
            red_total = sum(u[2] * 100 for u in red)
            blue_total = sum(u[2] * 100 for u in blue)
            hp_diff[sid] = red_total - blue_total
    return hp_diff


def plot_mds_terrain(
    coords: np.ndarray,
    hp_diff: np.ndarray,
    state_stats: Optional[Dict[int, Dict]] = None,
):
    import plotly.graph_objects as go

    x, y = coords[:, 0], coords[:, 1]
    valid_mask = ~np.isnan(hp_diff)
    n_states = len(x)
    grid_n = int(min(200, np.sqrt(n_states) * 3))
    grid_n = max(grid_n, 50)

    traces = []

    if valid_mask.sum() > 10:
        x_v, y_v, z_v = x[valid_mask], y[valid_mask], hp_diff[valid_mask]

        x_margin = (x_v.max() - x_v.min()) * 0.05
        y_margin = (y_v.max() - y_v.min()) * 0.05
        xi = np.linspace(x_v.min() - x_margin, x_v.max() + x_margin, grid_n)
        yi = np.linspace(y_v.min() - y_margin, y_v.max() + y_margin, grid_n)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        zi = griddata((x_v, y_v), z_v, (xi_grid, yi_grid), method="linear")

        traces.append(
            go.Contour(
                z=zi,
                x=xi,
                y=yi,
                colorscale="RdBu",
                reversescale=False,
                opacity=0.65,
                contours=dict(start=-100, end=100, size=5),
                hovertemplate="MDS X: %{x:.1f}<br>MDS Y: %{y:.1f}<br>HP差: %{z:.1f}<extra></extra>",
                showscale=True,
                colorbar=dict(
                    title=dict(text="HP差值(红-蓝)", font=dict(size=12)), len=0.85
                ),
                line=dict(width=0),
            )
        )

    n_valid = int(valid_mask.sum())
    if n_valid > 0:
        traces.append(
            go.Scattergl(
                x=x[valid_mask],
                y=y[valid_mask],
                mode="markers",
                marker=dict(
                    size=4,
                    color=hp_diff[valid_mask],
                    colorscale="RdBu",
                    reversescale=False,
                    showscale=False,
                    opacity=0.7,
                    line=dict(width=0),
                ),
                customdata=np.column_stack(
                    [np.arange(n_states)[valid_mask], hp_diff[valid_mask]]
                ),
                hovertemplate="State: %{customdata[0]}<br>HP差: %{customdata[1]:.1f}<extra></extra>",
                name="有HP数据",
            )
        )

    n_missing = n_states - n_valid
    if n_missing > 0:
        traces.append(
            go.Scattergl(
                x=x[~valid_mask],
                y=y[~valid_mask],
                mode="markers",
                marker=dict(size=2, color="gray", opacity=0.25, line=dict(width=0)),
                customdata=np.arange(n_states)[~valid_mask],
                hovertemplate="State: %{customdata[0]} (无HP数据)<extra></extra>",
                name="无HP数据",
            )
        )

    title = f"状态地形图 ({n_states} 个状态)"
    if n_missing > 0:
        title += f"  [HP: {n_valid}, 缺失: {n_missing}]"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title="MDS X", gridcolor="#e0e0e0"),
        yaxis=dict(title="MDS Y", gridcolor="#e0e0e0", scaleanchor="x", scaleratio=1),
        font=dict(family="SimHei, Microsoft YaHei, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=50),
        dragmode="pan",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
