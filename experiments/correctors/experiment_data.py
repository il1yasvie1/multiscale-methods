"""Results and configuration snapshots for corrector runs."""

from pathlib import Path

import pandas as pd
import yaml

from experiments.solver_config import numerical_solver_settings


def results_path(config: dict) -> Path:
    return Path("results") / config["experiment"] / config["experiment_name"] / "runs.csv"


def prepare_experiment(config: dict) -> Path:
    csv_path = results_path(config)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = csv_path.parent / "config.yaml"
    if snapshot.exists():
        previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        # Snapshots created before the experiment discriminator belong to this runner.
        previous.setdefault("experiment", "correctors")
        # Compare numerical options below; diagnostic flags may change on resume.
        previous_without_diagnostics = {key: value for key, value in previous.items()
                                        if key != "solver_parameters"}
        current_without_diagnostics = {key: value for key, value in config.items()
                                       if key != "solver_parameters"}
        if previous_without_diagnostics != current_without_diagnostics:
            raise ValueError("Experiment config changed; use a new experiment_name")
        if numerical_solver_settings(previous, default_modified_mass_lumped=True,
                                     legacy_snapshot=True) != numerical_solver_settings(
                                         config, default_modified_mass_lumped=True
                                     ):
            raise ValueError("Numerical solver settings changed; use a new experiment_name")
    else:
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return csv_path


def load_results(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={"run_id": str}) if csv_path.exists() else pd.DataFrame()


def completed_run_ids(table: pd.DataFrame) -> set[str]:
    if table.empty:
        return set()
    required = ("total_l2", "total_h1_semi", "total_h1")
    complete = (table.status == "completed") & table.reindex(columns=required).notna().all(axis=1)
    return set(table.loc[complete, "run_id"])


def save_result(csv_path: Path, table: pd.DataFrame, row: dict) -> pd.DataFrame:
    if not table.empty:
        table = table.loc[table.run_id != row["run_id"]]
    updated = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    temp = csv_path.with_suffix(".tmp")
    updated.to_csv(temp, index=False)
    temp.replace(csv_path)
    return updated
