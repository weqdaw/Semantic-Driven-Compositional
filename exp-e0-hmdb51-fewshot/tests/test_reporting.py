import numpy as np
import pytest

from hmdb51_e0.reporting import summarize_batch
from hmdb51_e0.pipeline import run_development
from hmdb51_e0.provenance import read_json, write_json
from test_pipeline import prepared


def test_summary_uses_all_seeds_sample_std_and_unique_support_counts(prepared, config, tmp_path):
    batch = run_development(*prepared[:3], config, prepared[3])
    result = summarize_batch(batch, tmp_path / "report", config)
    assert len(result) == 3
    runs = read_json(batch / "batch.json")["runs"]
    values = [read_json(batch / r["directory"] / "metrics.json")["accuracy"] for r in runs if r["k"] == 1]
    assert result[0]["accuracy_mean"] == pytest.approx(np.mean(values))
    assert result[0]["accuracy_std"] == pytest.approx(np.std(values, ddof=1))
    assert result[0]["n_support_sets"] == 10
    assert result[0]["n_evaluated_videos"] == 50
    assert (tmp_path / "report/e0-results.md").is_file()
    assert "仅程序验证" in (tmp_path / "report/e0-results.md").read_text(encoding="utf-8")


def test_production_cli_rejects_nonofficial_features(prepared):
    from hmdb51_e0.cli import require_official_features
    with pytest.raises(ValueError, match="official|CLIP"):
        require_official_features(prepared[2])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mixed", "failed", "prediction"])
def test_summary_refuses_incomplete_or_mixed_or_corrupt_runs(prepared, config, tmp_path, mutation):
    batch = run_development(*prepared[:3], config, prepared[3])
    path = batch / "batch.json"
    data = read_json(path)
    if mutation == "missing":
        data["runs"].pop()
    elif mutation == "duplicate":
        data["runs"][-1] = data["runs"][0]
    elif mutation in ("mixed", "failed"):
        provenance_path = batch / data["runs"][0]["directory"] / "provenance.json"
        provenance = read_json(provenance_path)
        if mutation == "mixed":
            provenance["feature_identity"] = "wrong"
        else:
            provenance["status"] = "failed"
        write_json(provenance_path, provenance)
    else:
        prediction_path = batch / data["runs"][0]["directory"] / "predictions.csv"
        prediction_path.write_text("corrupt\n", encoding="utf-8")
    write_json(path, data)
    with pytest.raises(ValueError):
        summarize_batch(batch, tmp_path / "report", config)
