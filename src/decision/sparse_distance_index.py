import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class _SparseDistanceRow:
    def __init__(self, index: "SparseDistanceIndex", row: int):
        self._index = index
        self._row = int(row)

    def __getitem__(self, col):
        return self._index[self._row, int(col)]


class SparseDistanceIndex:
    """
    Sparse state-distance provider used when a dense NxN matrix is infeasible.

    The class intentionally supports ``index[state_i, state_j]`` so existing
    code paths that used a dense numpy distance matrix can degrade gracefully.
    Unknown pairs return ``np.inf`` rather than raising.
    """

    is_sparse_distance_index = True

    def __init__(
        self,
        neighbors: Dict[int, List[Tuple[int, float]]],
        num_states: int,
        top_k: int,
        metadata: Optional[Dict] = None,
    ):
        self.neighbors = {
            int(state_id): [(int(nid), float(dist)) for nid, dist in entries]
            for state_id, entries in neighbors.items()
        }
        self.num_states = int(num_states)
        self.top_k = int(top_k)
        self.metadata = metadata or {}
        self.shape = (self.num_states, self.num_states)
        self._lookup: Dict[int, Dict[int, float]] = {
            sid: {nid: dist for nid, dist in entries}
            for sid, entries in self.neighbors.items()
        }

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return _SparseDistanceRow(self, int(key))
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("SparseDistanceIndex expects matrix-style [i, j] access")
        row, col = int(key[0]), int(key[1])
        if row == col:
            return 0.0
        dist = self._lookup.get(row, {}).get(col)
        if dist is not None:
            return dist
        dist = self._lookup.get(col, {}).get(row)
        if dist is not None:
            return dist
        return np.inf

    def get_neighbors(
        self,
        state_id: int,
        max_distance: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        entries = list(self.neighbors.get(int(state_id), []))
        if max_distance is not None:
            entries = [(sid, d) for sid, d in entries if d <= max_distance]
        if limit is not None and limit > 0:
            entries = entries[: int(limit)]
        return entries

    def has_state(self, state_id: int) -> bool:
        return int(state_id) in self.neighbors

    def finite_distance(self, state_i: int, state_j: int) -> Optional[float]:
        dist = float(self[state_i, state_j])
        if math.isfinite(dist):
            return dist
        return None

    def save(self, path: str) -> None:
        save_sparse_distance_index(self, path)


def save_sparse_distance_index(index: SparseDistanceIndex, path: str) -> None:
    payload = {
        "format": "PredictionRTS.SparseDistanceIndex.v1",
        "neighbors": index.neighbors,
        "num_states": index.num_states,
        "top_k": index.top_k,
        "metadata": index.metadata,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_sparse_distance_index(path: str) -> SparseDistanceIndex:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, SparseDistanceIndex):
        return payload
    return SparseDistanceIndex(
        neighbors=payload.get("neighbors", {}),
        num_states=int(payload.get("num_states", 0)),
        top_k=int(payload.get("top_k", 0)),
        metadata=payload.get("metadata", {}),
    )


def is_sparse_distance_index(obj) -> bool:
    return bool(getattr(obj, "is_sparse_distance_index", False))
