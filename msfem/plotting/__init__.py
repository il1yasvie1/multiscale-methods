"""Plot MsFEM sweeps from saved experiment tables."""

from pathlib import Path

import pandas as pd

from .files import prepare_figure_dirs
from .spatial import _colored_sweep, _fixed_h_delta_sweep, _overview_sweep
from .time import _time_sweep


def make_plots(csv_path: Path) -> int:
    table = pd.read_csv(csv_path, dtype={"run_id": str})
    table = table.loc[table.status == "completed"]
    figures = csv_path.parent / "figures"
    prepare_figure_dirs(figures)
    if table.empty:
        return 0
    return (_colored_sweep(table, "H", figures)
            + _colored_sweep(table, "delta", figures)
            + _fixed_h_delta_sweep(table, figures)
            + _time_sweep(table, figures)
            + _overview_sweep(table, "H", figures)
            + _overview_sweep(table, "delta", figures))


def plot_config(config_path: Path) -> None:
    """Redraw every completed run for this experiment, without a solver import."""
    from ..experiment_config import load_config
    from ..experiment_data import results_csv_path

    config = load_config(config_path)
    csv_path = results_csv_path(config)
    if not csv_path.exists():
        raise FileNotFoundError(f"No results found at {csv_path}")
    count = make_plots(csv_path)
    print(f"Created {count} figures in {csv_path.parent / 'figures'}")
