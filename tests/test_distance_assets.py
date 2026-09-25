from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.distance_assets import load_distances


def test_interrupted_extraction_can_retry_and_preserves_original_bytes(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.array([[0, 0.23456789], [0.23456789, 0]], dtype=np.float64))
    graph = tmp_path / "cache/experience_transition_graph/test_augmented"
    graph.mkdir(parents=True)
    archive = graph / "state_distance_matrix.npz"
    archive.write_bytes(b"interrupted zip")
    with pytest.raises(zipfile.BadZipFile):
        load_distances("test", "augmented_1", root=tmp_path)
    target = tmp_path / "cache/npy/state_distance_matrix_test_augmented_1.npy"
    assert not target.exists()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as packed:
        packed.write(source, "distances.npy")
    matrix = load_distances("test", "augmented_1", root=tmp_path)
    assert isinstance(matrix, np.memmap)
    assert target.read_bytes() == source.read_bytes()
    assert matrix[0, 1] == np.load(source)[0, 1]
    assert not list(target.parent.glob("*.tmp"))
