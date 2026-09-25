import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_assets.aiide26_sage import evidence


def test_sage_matches_and_external_references_are_not_claimed_as_reproduced(tmp_path):
    report = evidence.verify(tmp_path)
    assert report["passed"] is True
    assert report["evidence_integrity_passed"] is True
    assert report["failures"] == []
    assert report["sage_cells_recomputed"] == 12
    assert report["external_cells_transcribed"] == 42
    assert report["external_experiments_reproduced"] is False
    actual = pd.read_csv(tmp_path / "table_verification.csv")
    sage = actual[actual.method.isin(["Graph-only SAGE", "Full SAGE"])]
    assert len(sage) == 12
    assert sage.n.eq(100).all()
    assert sage[["score_mean_matches", "score_std_matches", "win_rate_pct_matches"]].all().all()
    references = actual[actual.verification_scope.eq("paper transcription only")]
    assert references.n.isna().all()
    assert references.std_ddof.isna().all()


def test_additional_mismatch_cannot_be_hidden_as_known(tmp_path, monkeypatch):
    measurements = evidence.table_measurements()
    row = measurements.method.eq("MAPPO") & measurements.scenario.eq("sce-2m")
    measurements.loc[row, "score_mean"] += 1
    monkeypatch.setattr(evidence, "table_measurements", lambda episodes=None: measurements)
    report = evidence.verify(tmp_path)
    assert not report["evidence_integrity_passed"]


def test_missing_and_changed_files_are_rejected(tmp_path, monkeypatch):
    data = tmp_path / "one.csv"
    data.write_text("original", encoding="utf-8")
    manifest = {"files": [{"path": "one.csv", "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}]}
    (tmp_path / "provenance.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(evidence, "PACK", tmp_path)
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    assert evidence.file_checks() == []
    data.write_text("changed")
    assert evidence.file_checks() == ["one.csv: checksum mismatch"]
    data.unlink()
    assert evidence.file_checks() == ["one.csv: missing"]


def test_selected_identifiers_cannot_be_replaced_by_same_scoring_rows():
    a = pd.DataFrame({"episode": [1, 2], "score": [10, 20]})
    b = pd.DataFrame({"episode": [1, 3], "score": [10, 20]})
    with pytest.raises(ValueError, match="identifiers"):
        evidence.verify_aggregated_table(a, b, ["episode"], ["score"])


def test_frozen_selection_is_balanced_and_unique():
    rows = evidence.selected_rows()
    assert not rows.duplicated(["experiment_id", "episode_id"]).any()
    assert len(rows.groupby(["scenario", "method"])) == 12
    assert rows.groupby(["scenario", "method"]).size().eq(100).all()


def test_historical_draw_label_counts_as_nonwin_but_invalid_outcome_fails():
    assert evidence.stats([1, 2], ["Win", "Dogfall"])["win_rate_pct"] == 50
    with pytest.raises(ValueError, match="outcome"):
        evidence.stats([1, 2], ["Win", "unreadable"])
