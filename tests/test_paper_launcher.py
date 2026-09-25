from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_paper_sage as launcher


def test_gate_source_aliases_preserve_numeric_settings():
    params = {"tuning_etg_protected_sources": ["kg_plan", "kg_follow"],
              "tuning_validation_sources": ["kg_relaxed", "ood"],
              "tuning_validation_profiles": {"kg_relaxed": {"min_visits": 3}},
              "backup_score_threshold": .4704432873079989}
    actual = launcher.normalize_source_names(params)
    assert actual["tuning_etg_protected_sources"] == ["etg_plan", "etg_follow"]
    assert actual["tuning_validation_sources"] == ["etg_relaxed", "ood"]
    assert actual["tuning_validation_profiles"]["etg_relaxed"] == {"min_visits": 3}
    assert actual["backup_score_threshold"] == params["backup_score_threshold"]
    assert "kg_relaxed" in params["tuning_validation_profiles"]


def test_preparation_cannot_write_into_frozen_evidence():
    with pytest.raises(ValueError, match="output/"):
        launcher.prepare("sce-1", "full", launcher.PACK / "raw/new")


def test_missing_assets_cannot_silently_use_live_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Required SAGE runtime input"):
        launcher.prepare("sce-1", "full", tmp_path / "output/new")
    assert not (tmp_path / "output/new").exists()
