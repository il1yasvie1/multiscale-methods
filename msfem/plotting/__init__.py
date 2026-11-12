"""Plot MsFEM sweeps from saved experiment tables."""

from pathlib import Path

import pandas as pd

from ..experiment_config import expand_runs
from ..experiment_data import JUMP_COLUMNS
from .common import KINDS
from .notation import METRICS
from .files import prepare_figure_dirs
from .spatial import _colored_sweep, _fixed_parameter_sweep, _overview_sweep
from .time import _time_sweep


def make_plots(csv_path: Path, *, config: dict | None = None) -> int:
    table = pd.read_csv(csv_path, dtype={"run_id": str})
    # Missing diagnostics stay missing; columns outside METRICS are not plotted.
    table = table.reindex(columns=list(dict.fromkeys([*table.columns, *METRICS])))
    table = table.loc[table.status == "completed"]
    compare_all = False
    if config is not None:
        runs = list(expand_runs(config))
        table = table.loc[table.run_id.isin({run["run_id"] for run in runs})]
        compare_all = set(KINDS) <= {run["corrector_kind"] for run in runs}
    missing_jumps = table[list(JUMP_COLUMNS)].isna().any(axis=1).sum()
    if missing_jumps:
        print(f"{missing_jumps} completed rows lack unweighted jumps; rerun the config to compute them.")
    figures = csv_path.parent / "figures"
    prepare_figure_dirs(figures)
    if table.empty:
        return 0
    count = (_colored_sweep(table, "H", figures, compare_all=compare_all)
             + _colored_sweep(table, "delta", figures, compare_all=compare_all)
             + _fixed_parameter_sweep(table, "delta", figures, compare_all=compare_all)
             + _fixed_parameter_sweep(table, "H", figures, compare_all=compare_all)
             + _time_sweep(table, figures, compare_all=compare_all)
             + _overview_sweep(table, "H", figures, compare_all=compare_all)
             + _overview_sweep(table, "delta", figures, compare_all=compare_all))
    if compare_all and count == 0:
        print("No three-mode sweep has enough completed points at common settings.")
    return count


def plot_config(config_path: Path) -> None:
    """Redraw the completed runs selected by this YAML, without a solver import."""
    from ..experiment_config import load_config
    from ..experiment_data import results_csv_path

    config = load_config(config_path)
    csv_path = results_csv_path(config)
    if not csv_path.exists():
        raise FileNotFoundError(f"No results found at {csv_path}")
    count = make_plots(csv_path, config=config)
    print(f"Created {count} figures in {csv_path.parent / 'figures'}")
