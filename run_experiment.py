"""Run/resume or redraw an MsFEM or corrector experiment."""

import argparse
from pathlib import Path

import yaml


def experiment_method(path: Path) -> str:
    """Select the runner from the required experiment field."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment YAML must contain a mapping")
    experiment = config.get("experiment")
    if experiment not in ("msfem", "correctors"):
        raise ValueError("experiment must be 'msfem' or 'correctors'")
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command_or_config", help="run, plot, or a config path")
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command_or_config in ("run", "plot"):
        if args.config is None:
            parser.error("A config path is required")
        command, path = args.command_or_config, args.config
    else:
        if args.config is not None:
            parser.error("Use 'run' or 'plot' before a config path")
        command, path = "run", Path(args.command_or_config)
    if not path.exists() and not path.suffix:
        path = path.with_suffix(".yaml")
    method = experiment_method(path)
    if method == "msfem":
        if command == "plot":
            from experiments.msfem.plotting import plot_config
            plot_config(path)
        else:
            from experiments.msfem.experiment import run_config
            run_config(path)
        return

    from experiments.correctors.experiment_config import load_config
    from experiments.correctors.experiment_data import results_path
    if command == "plot":
        csv_path = results_path(load_config(path))
        if not csv_path.exists():
            raise FileNotFoundError(f"No results found at {csv_path}")
        from experiments.correctors.plotting import make_plots
        count = make_plots(csv_path)
        print(f"Created {count} figures in {csv_path.parent / 'figures'}")
    else:
        from experiments.correctors.experiment import run_config
        csv_path = run_config(path)
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank == 0:
            from experiments.correctors.plotting import make_plots
            count = make_plots(csv_path)
            print(f"Results: {csv_path}")
            print(f"Created {count} figures in {csv_path.parent / 'figures'}")


if __name__ == "__main__":
    main()
