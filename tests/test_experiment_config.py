from pathlib import Path

import pandas as pd
import pytest

from msfem.experiment_config import expand_runs, load_config
from msfem.plotting.common import _time_groups


def test_smoke_config_pairs_modes_at_matching_final_times():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke-test.yaml"
    config = load_config(config_path)
    runs = list(expand_runs(config))
    modified = [row for row in runs if row["corrector_kind"] == "modified"]
    assert len(modified) == 64
    assert all(row["num_time_steps"] is None and row["time_step"] is None for row in modified)
    assert len({row["run_id"] for row in runs}) == len(runs)

    groups = list(_time_groups(pd.DataFrame(runs)))
    assert len(groups) == 2
    for final_time, steps, rows in groups:
        assert steps > 0
        assert set(rows.corrector_kind) == {"elliptic", "parabolic", "modified"}
        assert set(rows.final_time.dropna()) == {final_time}


def test_modified_only_time_groups():
    table = pd.DataFrame({"corrector_kind": ["modified"], "final_time": [0.01],
                          "num_time_steps": [None]})
    [(final_time, steps, rows)] = list(_time_groups(table))
    assert final_time == 0.01 and steps is None
    pd.testing.assert_frame_equal(rows, table)


@pytest.mark.parametrize("settings", [[], [{"final_time": 0}], [{"final_time": float("nan")}],
                                      [{"final_time": 0.01, "num_time_steps": 5}]])
def test_invalid_modified_settings(settings):
    config = {"experiment_name": "test", "problem": {"name": "constant"},
              "mesh": {"N_H": 2, "N_h": 4}, "delta": 0,
              "corrector_kind": "modified", "modified_settings": settings}
    with pytest.raises(ValueError, match="modified_settings"):
        list(expand_runs(config))
