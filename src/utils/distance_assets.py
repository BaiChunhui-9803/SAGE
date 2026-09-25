"""Load the original dense/sparse distances shipped with an ETG.

Dense archives contain the original NPY bytes (no quantization). Extraction is
atomic so an interrupted first launch can be retried safely. Only the local
cache is written; the distributed evidence files remain unchanged.
"""
from pathlib import Path
import os
import shutil
import tempfile
import zipfile

import numpy as np

from src import ROOT_DIR


def load_distances(map_id, data_id, etg_file=None, root=ROOT_DIR):
    root = Path(root)
    cache = root / "cache/npy"
    matrix = cache / f"state_distance_matrix_{map_id}_{data_id}.npy"
    graph_root = root / "cache/experience_transition_graph"
    folders = []
    if etg_file:
        folders.append((graph_root / etg_file).parent)
    if data_id == "augmented_1":
        folders.append(graph_root / f"{map_id}_augmented")
    if matrix.exists():
        return np.load(matrix, mmap_mode="r", allow_pickle=False)
    for folder in dict.fromkeys(folders):
        archive = folder / "state_distance_matrix.npz"
        if not archive.exists():
            continue
        cache.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".npy.tmp", delete=False) as target:
                temporary = Path(target.name)
                with zipfile.ZipFile(archive) as packed, packed.open("distances.npy") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            check = np.load(temporary, mmap_mode="r", allow_pickle=False)
            if check.ndim != 2 or check.shape[0] != check.shape[1]:
                raise ValueError(f"Distance matrix is not square: {archive}")
            del check
            # Concurrent first launches may have completed the same extraction.
            try:
                os.link(temporary, matrix)
            except FileExistsError:
                pass
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return np.load(matrix, mmap_mode="r", allow_pickle=False)
    candidates = [cache / f"state_sparse_neighbors_{map_id}_{data_id}.pkl"]
    for folder in dict.fromkeys(folders):
        candidates.extend([folder / "sparse_neighbors.pkl", folder / "npy/sparse_neighbors.pkl"])
    for path in candidates:
        if path.exists():
            from src.decision.sparse_distance_index import load_sparse_distance_index
            return load_sparse_distance_index(str(path))
    return None
