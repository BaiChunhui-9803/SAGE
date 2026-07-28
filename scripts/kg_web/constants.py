import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import ROOT_DIR

KG_DIR = ROOT_DIR / "cache" / "knowledge_graph"
NPY_DIR = ROOT_DIR / "cache" / "npy"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "output" / "live_results"

_BEAM_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]

_COMPOSITE_WEIGHTS = (0.30, 0.30, 0.30, 0.10)

_ACTION_STRATEGY_LABELS = {
    "best_beam": "Best Beam",
    "best_subtree_quality": "Best Subtree Quality",
    "best_subtree_winrate": "Best Subtree WinRate",
    "highest_transition_prob": "Highest Trans. Prob",
    "random_beam": "Random Beam",
    "epsilon_greedy": "Epsilon-Greedy",
}

_NEXT_STATE_MODE_LABELS = {
    "sample": "概率采样",
    "highest_prob": "最高概率",
}

BRIDGE_API_URL = "http://localhost:8000"

BKTREE_DEFAULT_THRESHOLDS = {
    "default": {"primary_threshold": 1.0, "secondary_threshold": 0.5},
    "MarineMicro_MvsM_4": {"primary_threshold": 0.7, "secondary_threshold": 0.5},
}

MAP_ID_TO_KEY = {
    "MarineMicro_MvsM_4": "sce-1",
    "MarineMicro_MvsM_4_mirror": "sce-1m",
    "MarineMicro_MvsM_4_dist": "sce-2",
    "MarineMicro_MvsM_4_dist_mirror": "sce-2m",
    "MarineMicro_MvsM_8": "sce-3",
    "MarineMicro_MvsM_8_mirror": "sce-3m",
}


def get_bktree_threshold_defaults(map_id: str = "") -> dict:
    defaults = dict(BKTREE_DEFAULT_THRESHOLDS["default"])
    if map_id and map_id in BKTREE_DEFAULT_THRESHOLDS:
        defaults.update(BKTREE_DEFAULT_THRESHOLDS[map_id])
    return defaults


def get_map_key_for_map_id(map_id: str = "", default: str = "sce-1") -> str:
    return MAP_ID_TO_KEY.get(map_id or "", default)
