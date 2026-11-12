"""Run/resume or redraw an experiment: python run.py [plot] configs/<name>."""

import argparse
from pathlib import Path


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
    if command == "plot":
        from msfem.plotting import plot_config
        plot_config(path)
    else:
        from msfem.experiment import run_config
        run_config(path)


if __name__ == "__main__":
    main()
