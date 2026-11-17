"""Run or redraw whole-domain corrector experiments."""

import argparse

from .experiment_config import load_config
from .experiment_data import results_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "plot"))
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "run":
        from .experiment import run_config
        csv_path = run_config(args.config)
    else:
        csv_path = results_path(config)
        if not csv_path.exists():
            parser.error(f"No results at {csv_path}")
    if args.command == "plot":
        from .plotting import make_plots
        count = make_plots(csv_path)
        print(f"Results: {csv_path}; figures: {count}")
    else:
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank == 0:
            from .plotting import make_plots
            count = make_plots(csv_path)
            print(f"Results: {csv_path}; figures: {count}")


if __name__ == "__main__":
    main()
