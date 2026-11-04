"""Experiment paths, CSV schema, and atomic result updates."""

from pathlib import Path

import pandas as pd
import yaml

from .experiment_config import validate_snapshot


COUNT_COLUMNS = (
    "num_of_coarse_cells", "num_of_fine_cells", "num_of_coarse_dofs",
    "num_of_fine_dofs", "num_of_reconstruction_dofs",
)
ERROR_COLUMNS = tuple(
    f"{prefix}_{norm}_error"
    for prefix in ("msfem", "coarse_fem", "fine_fem", "corrector")
    for norm in ("l2", "h1", "h1_semi")
)
COLUMNS = (
    "experiment_name", "run_id", "status", "method", "problem", "epsilon",
    "N_H", "N_h", "H", "h", "delta", "assembly_type", "corrector_kind",
    "final_time", "num_time_steps", "time_step", *COUNT_COLUMNS,
    *ERROR_COLUMNS, "msfem_jump", "setup_seconds", "offline_seconds",
    "online_seconds", "coarse_fem_solve_seconds", "fine_fem_solve_seconds",
    "metrics_seconds", "error_message",
)


def results_csv_path(config: dict) -> Path:
    return Path("work") / config["experiment_name"] / "runs.csv"


def prepare_experiment(config: dict) -> Path:
    """Validate and update the experiment snapshot; return its CSV path."""
    csv_path = results_csv_path(config)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = csv_path.parent / "config.yaml"
    if snapshot.exists():
        validate_snapshot(yaml.safe_load(snapshot.read_text()), config)
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    return csv_path


def load_results(csv_path: Path) -> pd.DataFrame:
    return (pd.read_csv(csv_path, dtype={"run_id": str}) if csv_path.exists()
            else pd.DataFrame(columns=COLUMNS))


def save_result(csv_path: Path, table: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Replace a run row and atomically write the complete experiment table."""
    updated = table.loc[table["run_id"] != row["run_id"]]
    updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
    temp_path = csv_path.with_suffix(".tmp")
    updated.reindex(columns=COLUMNS).to_csv(temp_path, index=False)
    temp_path.replace(csv_path)
    return updated
