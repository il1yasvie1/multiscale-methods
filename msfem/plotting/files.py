"""Figure directory layout, generated-file cleanup, and image saving."""

from pathlib import Path

import matplotlib.pyplot as plt


SUBDIRECTORIES = ("H", "delta", "delta/fixed_H", "time")
GENERATED_PREFIXES = ("H_", "delta_", "time_", "h_", "overview_", "fixed_H_")


def prepare_figure_dirs(figures: Path) -> None:
    """Remove only generated PNGs, including those in the former flat layout."""
    folders = (figures, *(figures / subdirectory for subdirectory in SUBDIRECTORIES))
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for old_plot in folder.glob("*.png"):
            if old_plot.name.startswith(GENERATED_PREFIXES):
                old_plot.unlink()


def facet_id(values) -> str:
    return "_".join(str(value).replace(".", "p") for value in values)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def save_sweep_figure(fig, figures: Path, sweep: str, metric: str,
                      facet_values, T=None, Nt=None) -> None:
    time_id = "elliptic" if T is None else f"T{str(T).replace('.', 'p')}_Nt{Nt}"
    name = f"{sweep}_{metric}_{facet_id(facet_values)}_{time_id}.png"
    save_figure(fig, figures / sweep / name)
