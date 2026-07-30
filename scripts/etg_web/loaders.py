import pickle
from typing import Dict, List, Tuple

from etg_web.i18n import st
import yaml
import numpy as np

from src import ROOT_DIR
from src.decision.experience_transition_graph import DecisionExperienceTransitionGraph

from etg_web.constants import etg_DIR, NPY_DIR, DATA_DIR


@st.cache_data
def load_etg_catalog() -> List[Dict]:
    path = ROOT_DIR / "configs" / "etg_catalog.yaml"
    if not path.exists():
        st.error(f"经验转移图目录不存在: {path}")
        st.stop()
    with open(path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    return catalog.get("experience_transition_graphs", [])


@st.cache_resource
def load_etg(etg_file: str) -> Tuple[Dict, float, float]:
    path = etg_DIR / etg_file

    if not path.exists():
        st.error(
            f"文件不存在: {path}\n请先运行 `python scripts/build_experience_transition_graph.py`"
        )
        st.stop()

    with open(path, "rb") as f:
        etg_data = pickle.load(f)

    quality_scores = [
        stats.quality_score
        for actions in etg_data["state_action_map"].values()
        for stats in actions.values()
    ]
    quality_min = min(quality_scores) if quality_scores else -60.0
    quality_max = max(quality_scores) if quality_scores else 80.0

    return etg_data, quality_min, quality_max


@st.cache_resource
def load_transitions(transitions_file: str) -> Dict:
    path = etg_DIR / transitions_file
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_etg_object(etg_file: str) -> DecisionExperienceTransitionGraph:
    path = etg_DIR / etg_file
    return DecisionExperienceTransitionGraph.load(str(path))


@st.cache_resource
def load_episode_data(map_id: str, data_id: str) -> Dict:
    import csv as _csv

    base = DATA_DIR / map_id / data_id
    states, actions, outcomes, scores = [], [], [], []

    node_log = base / "graph" / "node_log.txt"
    if node_log.exists():
        with open(node_log, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    states.append([int(p) for p in parts])

    action_log = base / "action_log.csv"
    if action_log.exists():
        with open(action_log, "r") as f:
            reader = _csv.reader(f)
            for row in reader:
                if row:
                    raw = row[0]
                    actions.append([raw[j : j + 2] for j in range(0, len(raw), 2)])

    result_file = base / "game_result.txt"
    if result_file.exists():
        with open(result_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                outcomes.append(parts[0] if parts else "Unknown")
                try:
                    sc = float(parts[2]) if len(parts) > 2 else 0.0
                    pn = float(parts[3]) if len(parts) > 3 else 0.0
                    scores.append(sc + pn)
                except (ValueError, IndexError):
                    scores.append(0.0)

    return {
        "states": states,
        "actions": actions,
        "outcomes": outcomes,
        "scores": scores,
        "n_episodes": len(states),
    }


@st.cache_resource
def load_distance_matrix_np(map_id: str, data_id: str):
    path = NPY_DIR / f"state_distance_matrix_{map_id}_{data_id}.npy"
    if path.exists():
        return np.load(str(path))
    return None


@st.cache_resource
def load_state_hp_data(map_id: str, data_id: str) -> Dict:
    import json as _json

    base = DATA_DIR / map_id / data_id
    sn_file = base / "graph" / "state_node.txt"

    reverse_dict: Dict = {}
    if sn_file.exists():
        with open(sn_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                key = eval(parts[0])
                sid = int(parts[1])
                score = float(parts[2])
                if sid not in reverse_dict:
                    reverse_dict[sid] = {"cluster": key, "score": score}

    primary_file = base / "bktree" / "primary_bktree.json"
    primary_cluster_ids = []
    if primary_file.exists():
        with open(primary_file, "r") as f:
            root = _json.load(f)
        _collect_cluster_ids(root, primary_cluster_ids)

    sub_node_map: Dict[Tuple[int, int], Dict] = {}

    def _search_sub(node):
        cid = node["cluster_id"]
        state = node.get("state", {})
        sub_node_map[(primary_id, cid)] = state
        for child in node.get("children", {}).values():
            _search_sub(child)

    for primary_id in primary_cluster_ids:
        sec_file = base / "bktree" / f"secondary_bktree_{primary_id}.json"
        if not sec_file.exists():
            continue
        try:
            with open(sec_file, "r") as f:
                sec_root = _json.load(f)
            _search_sub(sec_root)
        except Exception:
            pass

    hp_lookup: Dict[int, Dict] = {}
    for sid, info in reverse_dict.items():
        cluster = info["cluster"]
        if isinstance(cluster, tuple) and len(cluster) == 2:
            state_data = sub_node_map.get(cluster, {})
            if state_data:
                hp_lookup[sid] = {
                    "cluster": cluster,
                    "score": info["score"],
                    "red_army": state_data.get("red_army", []),
                    "blue_army": state_data.get("blue_army", []),
                }

    return {
        "reverse_dict": reverse_dict,
        "hp_lookup": hp_lookup,
        "n_states": len(reverse_dict),
    }


def _collect_cluster_ids(node, result_list):
    result_list.append(node["cluster_id"])
    for child in node.get("children", {}).values():
        _collect_cluster_ids(child, result_list)
