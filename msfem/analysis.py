"""Plot MsFEM sweeps from the saved experiment table."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt
import pandas as pd


METRICS = (
    "msfem_l2_error", "msfem_h1_error", "msfem_h1_semi_error",
    "msfem_jump", "corrector_l2_error", "corrector_h1_error",
    "corrector_h1_semi_error",
)
OVERVIEW_METRICS = (
    "corrector_l2_error", "corrector_h1_semi_error", "msfem_jump",
    "msfem_l2_error", "msfem_h1_semi_error",
)
SWEEPS = ("H", "delta", "time")
KINDS = {"elliptic": (cm.Blues, "o"), "parabolic": (cm.Reds, "s")}
Y_LABELS = {
    "msfem_l2_error": r"$\|u^\varepsilon-u_{H,h}^{\mathrm{MsFEM}}\|_{L^2(\Omega)}$",
    "msfem_h1_error": r"$\|u^\varepsilon-u_{H,h}^{\mathrm{MsFEM}}\|_{H^1(\mathcal{T}_H)}$",
    "msfem_h1_semi_error": r"$|u^\varepsilon-u_{H,h}^{\mathrm{MsFEM}}|_{H^1(\mathcal{T}_H)}$",
    "msfem_jump": (
        r"$\left(\sum_{F\in\mathcal{F}_H^{\mathrm{int}}}"
        r"H^{-1}\|[u_{H,h}^{\mathrm{MsFEM}}]\|_{L^2(F)}^2\right)^{1/2}$"
    ),
    "corrector_l2_error": (
        r"$\left(\sum_{K\in\mathcal{T}_H}\sum_{j=1}^{2}"
        r"\|w_{K,j}-\chi_j^\varepsilon\|_{L^2(K)}^2\right)^{1/2}$"
    ),
    "corrector_h1_error": (
        r"$\left(\sum_{K\in\mathcal{T}_H}\sum_{j=1}^{2}"
        r"\|w_{K,j}-\chi_j^\varepsilon\|_{H^1(K)}^2\right)^{1/2}$"
    ),
    "corrector_h1_semi_error": (
        r"$\left(\sum_{K\in\mathcal{T}_H}\sum_{j=1}^{2}"
        r"|w_{K,j}-\chi_j^\varepsilon|_{H^1(K)}^2\right)^{1/2}$"
    ),
}
OVERVIEW_NAMES = (
    r"Corrector $L^2$ error norm", r"Corrector $H^1$ error semi-norm",
    "MsFEM jump norm", r"MsFEM $L^2$ error norm",
    r"MsFEM $H^1$ error semi-norm",
)
OVERVIEW_TITLES = {
    metric: name + "\n" + Y_LABELS[metric]
    for metric, name in zip(OVERVIEW_METRICS, OVERVIEW_NAMES)
}


def _parameter_title(columns, values, T=None, Nt=None):
    symbols = {
        "epsilon": r"\varepsilon", "N_H": "N_H", "N_h": "N_h",
        "delta": r"\delta", "H": "H",
    }
    parts = []
    for key, value in zip(columns, values):
        if key == "assembly_type":
            parts.append(r"\mathrm{PG}" if value == "petrov_galerkin" else r"\mathrm{G}")
        elif key in symbols and pd.notna(value):
            parts.append(f"{symbols[key]}={value:g}")
    if T is not None:
        parts.extend((f"T={T:g}", f"N_t={Nt}"))
    return "$" + r",\;".join(parts) + "$"


def _label(columns, values):
    return ", ".join(
        f"{key}={value:g}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in zip(columns, values) if pd.notna(value)
    )


def _groups(frame, columns):
    for values, rows in frame.groupby(list(columns), dropna=False):
        yield values if isinstance(values, tuple) else (values,), rows


def _time_groups(table):
    settings = table.loc[
        table.corrector_kind == "parabolic", ["final_time", "num_time_steps"]
    ].drop_duplicates()
    if settings.empty:
        yield None, None, table
    for T, Nt in settings.itertuples(index=False, name=None):
        rows = table.loc[
            (table.corrector_kind == "elliptic")
            | ((table.corrector_kind == "parabolic")
               & (table.final_time == T) & (table.num_time_steps == Nt))
        ]
        yield T, int(Nt), rows


def _power_two_label(value):
    exponent = round(math.log2(float(value)))
    return rf"$2^{{{exponent}}}$" if math.isclose(float(value), 2.0**exponent) else f"{value:g}"


def _color(kind, value, low, high, parameter):
    cmap, _ = KINDS[kind]
    strength = 0.65 if high == low else (value - low) / (high - low)
    if parameter == "H":
        strength = 1 - strength
    return cmap(0.35 + 0.6 * strength)


def _add_color_keys(fig, ax, kinds, values, parameter):
    """Place category legend and two parameter color scales outside the axes."""
    handles = [
        Line2D([0], [0], color=KINDS[kind][0](0.8), marker=KINDS[kind][1],
               linestyle="--" if kind == "elliptic" else "-",
               markerfacecolor="white" if kind == "elliptic" else None,
               label=kind if len(values) > 1 else f"{kind}, {parameter}={values[0]:g}")
        for kind in kinds
    ]
    old_handles, old_labels = ax.get_legend_handles_labels()
    handles.extend(old_handles)
    labels = [handle.get_label() for handle in handles[:len(kinds)]] + old_labels
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.02),
              frameon=False, fontsize=8)
    if len(values) == 1:
        return
    low, high = min(values), max(values)
    for index, kind in enumerate(kinds):
        original = KINDS[kind][0]
        shades = [original(0.35), original(0.95)]
        if parameter == "H":
            shades.reverse()
        palette = colors.LinearSegmentedColormap.from_list(f"{kind}_scale", shades)
        scale = cm.ScalarMappable(
            norm=colors.Normalize(vmin=low, vmax=high if high > low else low + 1),
            cmap=palette,
        )
        bar_ax = fig.add_axes([0.78 + 0.11 * index, 0.2, 0.022, 0.46])
        bar = fig.colorbar(scale, cax=bar_ax, ticks=values)
        if parameter == "H":
            bar.ax.set_yticklabels([_power_two_label(value) for value in values])
        bar.ax.tick_params(labelsize=7)
        bar.ax.set_title(
            f"{kind}\n" + (r"$\delta$" if parameter == "delta" else r"$H$"),
            fontsize=8, pad=5,
        )


def _save(fig, figures, sweep, metric, facet_values, T=None, Nt=None):
    facet_id = "_".join(str(v).replace(".", "p") for v in facet_values)
    time_id = "elliptic" if T is None else f"T{str(T).replace('.', 'p')}_Nt{Nt}"
    fig.savefig(
        figures / f"{sweep}_{metric}_{facet_id}_{time_id}.png",
        dpi=160, bbox_inches="tight", pad_inches=0.25,
    )
    plt.close(fig)


def _avoid_overzoom(ax):
    """Show meaningful scale when plotted values differ only by roundoff."""
    values = [float(value) for line in ax.lines for value in line.get_ydata()
              if pd.notna(value) and math.isfinite(float(value))]
    if values and min(values) > 0 and max(values) - min(values) < 1e-4 * max(values):
        center = (min(values) + max(values)) / 2
        ax.set_ylim(center * 0.9, center * 1.1)


def _overview_axes(rows, *, time=False):
    height = 6.0 if rows == 1 else 6.3 * rows + 0.4
    fig, axes = plt.subplots(rows, 5, figsize=(24, height),
                             squeeze=False, sharex="col")
    bottom = (0.26 if rows == 1 else 0.22) if time else 0.18
    fig.subplots_adjust(left=0.06, right=0.98, bottom=bottom, top=0.82,
                        wspace=0.28, hspace=0.55)
    for ax in axes.flat:
        ax.set_box_aspect(0.9)
    return fig, axes


def _overview_keys(fig, kinds, values, parameter):
    handles = [Line2D([0], [0], color=KINDS[kind][0](0.8),
                      marker=KINDS[kind][1],
                      linestyle="--" if kind == "elliptic" else "-",
                      markerfacecolor="white" if kind == "elliptic" else None,
                      label=kind) for kind in kinds]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               ncol=len(handles), frameon=False)
    if len(values) < 2:
        symbol = r"\delta" if parameter == "delta" else "H"
        fig.text(0.5, 0.05, f"${symbol}={values[0]:g}$", fontsize=10,
                 ha="center")
        return
    for index, kind in enumerate(kinds):
        shades = [KINDS[kind][0](0.35), KINDS[kind][0](0.95)]
        if parameter == "H":
            shades.reverse()
        scale = cm.ScalarMappable(
            norm=colors.Normalize(vmin=min(values), vmax=max(values)),
            cmap=colors.LinearSegmentedColormap.from_list(f"overview_{kind}", shades),
        )
        bar_ax = fig.add_axes([0.20 + 0.40 * index, 0.035, 0.28, 0.018])
        bar = fig.colorbar(scale, cax=bar_ax, ticks=values,
                           orientation="horizontal")
        if parameter == "H":
            bar.ax.set_xticklabels([_power_two_label(value) for value in values])
        bar.ax.tick_params(labelsize=8, pad=2)
        bar.ax.set_title(f"{kind}\n" + (r"$H$" if parameter == "H" else r"$\delta$"),
                         fontsize=9, pad=3)


def _overview_sweep(table, sweep, figures):
    """One row per parabolic setting, five columns of key metrics."""
    x = "H" if sweep == "H" else "delta"
    shade_by = "delta" if sweep == "H" else "H"
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for facet_values, frame in _groups(table, facets):
        settings = sorted(set(zip(
            frame.loc[frame.corrector_kind == "parabolic", "final_time"],
            frame.loc[frame.corrector_kind == "parabolic", "num_time_steps"],
        )))
        if not settings:
            continue
        shades = sorted(frame[shade_by].dropna().unique())
        kinds = [kind for kind in KINDS if kind in set(frame.corrector_kind)]
        fig, axes = _overview_axes(len(settings))
        fig.suptitle(_parameter_title(facets, facet_values), fontsize=14)
        for row_index, (T, Nt) in enumerate(settings):
            rows_at_time = frame.loc[
                (frame.corrector_kind == "elliptic")
                | ((frame.corrector_kind == "parabolic")
                   & (frame.final_time == T) & (frame.num_time_steps == Nt))
            ]
            for col_index, metric in enumerate(OVERVIEW_METRICS):
                ax = axes[row_index, col_index]
                for (kind, shade), rows in _groups(rows_at_time,
                                                   ("corrector_kind", shade_by)):
                    rows = rows.sort_values(x)
                    y = pd.to_numeric(rows[metric], errors="coerce")
                    if metric == "msfem_jump":
                        y = y.mask(y.abs() <= 1e-12, 0.0)
                    elif sweep == "H":
                        y = y.where(y > 0)
                    if y.notna().any():
                        ax.plot(rows[x], y, color=_color(kind, shade, shades[0],
                                shades[-1], shade_by), marker=KINDS[kind][1],
                                linestyle="--" if kind == "elliptic" else "-",
                                markerfacecolor="white" if kind == "elliptic" else None,
                                zorder=3 if kind == "elliptic" else 2,
                                linewidth=1.5, markersize=5)
                ax.set_title(OVERVIEW_TITLES[metric], fontsize=10, pad=10)
                if col_index == 0:
                    ax.set_ylabel(rf"$T={T:g},\;N_t={int(Nt)}$", fontsize=10)
                if row_index == len(settings) - 1:
                    ax.set_xlabel(r"$H$" if sweep == "H" else r"$\delta$")
                if sweep == "H":
                    ax.set_xscale("log", base=2)
                    if metric != "msfem_jump":
                        ax.set_yscale("log")
                    ticks = sorted(rows_at_time.H.unique())
                    ax.set_xticks(ticks, [_power_two_label(value) for value in ticks])
                    ax.set_xlim(max(ticks) * 1.05, min(ticks) / 1.05)
                if metric == "msfem_jump":
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
        _overview_keys(fig, kinds, shades, shade_by)
        facet_id = "_".join(str(value).replace(".", "p") for value in facet_values)
        fig.savefig(figures / f"overview_{sweep}_{facet_id}.png", dpi=160,
                    bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        count += 1
    return count


def _colored_sweep(table, sweep, figures):
    """Plot both corrector kinds together; shade by delta or coarse resolution."""
    x = "H" if sweep == "H" else "delta"
    shade_by = "delta" if sweep == "H" else "H"
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for T, Nt, time_rows in _time_groups(table):
        for facet_values, frame in _groups(time_rows, facets):
            if frame[x].nunique() < 2:
                continue
            values = sorted(frame[shade_by].dropna().unique())
            kinds = [kind for kind in KINDS if kind in set(frame.corrector_kind)]
            for metric in METRICS:
                if frame[metric].notna().sum() < 2:
                    continue
                fig, ax = plt.subplots(figsize=(12, 5.8))
                fig.subplots_adjust(left=0.20, right=0.70, bottom=0.15, top=0.86)
                for (kind, shade), rows in _groups(frame, ("corrector_kind", shade_by)):
                    rows = rows.sort_values(x)
                    y = pd.to_numeric(rows[metric], errors="coerce")
                    if metric == "msfem_jump":
                        y = y.mask(y.abs() <= 1e-12, 0.0)
                    elif sweep == "H":
                        y = y.where(y > 0)
                    if y.notna().sum() == 0:
                        continue
                    ax.plot(rows[x], y, color=_color(kind, shade, values[0], values[-1], shade_by),
                            marker=KINDS[kind][1], linewidth=1.8,
                            linestyle="--" if kind == "elliptic" else "-",
                            markerfacecolor="white" if kind == "elliptic" else None,
                            zorder=3 if kind == "elliptic" else 2,
                            label="_nolegend_")

                if sweep == "H" and metric.startswith("msfem_") and metric != "msfem_jump":
                    for baseline, style in (("coarse_fem", "--"), ("fine_fem", ":")):
                        column = metric.replace("msfem", baseline, 1)
                        rows = frame[[x, column]].drop_duplicates(x).sort_values(x)
                        ax.plot(rows[x], rows[column], color="0.45", linestyle=style,
                                linewidth=1.4, label=baseline.replace("_", " "))

                if not ax.lines:
                    plt.close(fig)
                    continue
                ax.set_xlabel(r"$H$" if sweep == "H" else r"$\delta$")
                ax.set_ylabel(Y_LABELS[metric], fontsize=10)
                ax.set_title(_parameter_title(facets, facet_values, T, Nt), fontsize=11)
                if sweep == "H":
                    ax.set_xscale("log", base=2)
                    if metric != "msfem_jump":
                        ax.set_yscale("log")
                    ticks = sorted(frame.H.unique())
                    ax.set_xticks(ticks, [_power_two_label(value) for value in ticks])
                    ax.invert_xaxis()
                if metric == "msfem_jump":
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
                _add_color_keys(fig, ax, kinds, values, shade_by)
                _save(fig, figures, sweep, metric, facet_values, T, Nt)
                count += 1
    return count


def _other_sweeps(table, figures):
    specs = {
        "time": ("final_time", ("problem", "epsilon", "N_H", "N_h", "delta", "assembly_type"), ()),
    }
    count = 0
    for sweep, (x, facets, curves) in specs.items():
        selected = table.loc[table.corrector_kind == "parabolic"] if sweep == "time" else table
        for facet_values, frame in _groups(selected, facets):
            if frame[x].nunique() < 2:
                continue
            for metric in METRICS:
                if frame[metric].notna().sum() < 2:
                    continue
                fig, ax = plt.subplots(figsize=(11.5, 5.8))
                fig.subplots_adjust(left=0.20, right=0.68, bottom=0.15, top=0.86)
                groups = _groups(frame, curves) if curves else [((), frame)]
                for values, rows in groups:
                    rows = rows.sort_values(x)
                    y = pd.to_numeric(rows[metric], errors="coerce")
                    if metric == "msfem_jump":
                        y = y.mask(y.abs() <= 1e-12, 0.0)
                    if y.notna().sum() == 0:
                        continue
                    ax.plot(rows[x], y, marker="o", label=_label(curves, values) or "parabolic")
                    if sweep == "time":
                        for time_value, metric_value, steps in zip(
                            rows[x], y, rows.num_time_steps
                        ):
                            if pd.notna(metric_value):
                                ax.annotate(rf"$N_t={int(steps)}$", (time_value, metric_value),
                                            xytext=(3, 4), textcoords="offset points", fontsize=7)

                if metric.startswith("msfem_") and metric != "msfem_jump":
                    for baseline in ("coarse_fem", "fine_fem"):
                        column = metric.replace("msfem", baseline, 1)
                        rows = frame[[x, column]].drop_duplicates(x).sort_values(x)
                        ax.plot(rows[x], rows[column], color="0.5", linestyle="--",
                                label=baseline.replace("_", " "))
                if sweep == "time":
                    reference = table
                    for column, value in zip(facets, facet_values):
                        reference = reference.loc[reference[column] == value]
                    reference = reference.loc[reference.corrector_kind == "elliptic"]
                    if not reference.empty and pd.notna(reference[metric].iloc[0]):
                        ax.axhline(reference[metric].iloc[0], color="black",
                                   linestyle=":", label="elliptic")

                if not ax.lines:
                    plt.close(fig)
                    continue
                ax.set_xlabel(r"$T$")
                ax.set_ylabel(Y_LABELS[metric], fontsize=10)
                ax.set_title(_parameter_title(facets, facet_values), fontsize=11)
                if metric == "msfem_jump":
                    ax.set_xscale("log")
                    ax.set_ylim(bottom=0)
                ax.grid(alpha=0.25)
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8,
                          frameon=False)
                _save(fig, figures, sweep, metric, facet_values)
                count += 1
    return count


def _bivariate_color(H, delta, H_values, deltas):
    """Hue varies with H; lightness varies with delta."""
    H_level = 0.5 if len(H_values) == 1 else (H - H_values[0]) / (H_values[-1] - H_values[0])
    delta_level = 0.5 if len(deltas) == 1 else (delta - deltas[0]) / (deltas[-1] - deltas[0])
    base = cm.plasma(0.12 + 0.72 * H_level)[:3]
    white_fraction = 0.55 * (1 - delta_level)
    return tuple(white_fraction + (1 - white_fraction) * channel for channel in base)


def _overview_time(table, figures):
    """One row per time-step count; encode H and delta in a two-dimensional key."""
    parabolic = table.loc[table.corrector_kind == "parabolic"]
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    count = 0
    for facet_values, frame in _groups(parabolic, facets):
        settings = frame[["final_time", "num_time_steps"]].drop_duplicates()
        time_steps = sorted(frame.num_time_steps.unique())
        tested_times = sorted(frame.final_time.unique())
        H_values = sorted(frame.H.unique())
        deltas = sorted(frame.delta.unique())
        delta_markers = {delta: markers[index % len(markers)]
                         for index, delta in enumerate(deltas)}
        fig, axes = _overview_axes(len(time_steps), time=True)
        fig.suptitle(_parameter_title(facets, facet_values), fontsize=14)
        for row_index, Nt in enumerate(time_steps):
            time_rows = frame.loc[frame.num_time_steps == Nt]
            for col_index, metric in enumerate(OVERVIEW_METRICS):
                ax = axes[row_index, col_index]
                for (H, delta), rows in _groups(time_rows, ("H", "delta")):
                    rows = rows.sort_values("final_time")
                    y = pd.to_numeric(rows[metric], errors="coerce")
                    if metric == "msfem_jump":
                        y = y.mask(y.abs() <= 1e-12, 0.0)
                    if y.notna().any():
                        ax.plot(rows.final_time, y,
                                color=_bivariate_color(H, delta, H_values, deltas),
                                marker=delta_markers[delta], linewidth=1.5,
                                markersize=6)
                ax.set_title(OVERVIEW_TITLES[metric], fontsize=10, pad=10)
                if col_index == 0:
                    ax.set_ylabel(rf"$N_t={int(Nt)}$", fontsize=10)
                if row_index == len(time_steps) - 1:
                    ax.set_xlabel(r"$T$")
                ax.set_xscale("log")
                ax.set_xlim(min(tested_times) / 1.2, max(tested_times) * 1.2)
                ax.set_xticks(tested_times, [f"{T:g}" for T in tested_times])
                ax.xaxis.set_minor_locator(NullLocator())
                if metric == "msfem_jump":
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
        key_ax = fig.add_axes([0.70, 0.02, 0.20,
                               0.14 if len(time_steps) == 1 else 0.09])
        swatches = [[_bivariate_color(H, delta, H_values, deltas)
                     for H in H_values] for delta in deltas]
        key_ax.imshow(swatches, origin="lower", aspect="auto", interpolation="nearest")
        key_ax.set_xticks(range(len(H_values)),
                          [_power_two_label(H) for H in H_values])
        key_ax.set_yticks(range(len(deltas)), [f"{delta:g}" for delta in deltas])
        key_ax.set_xlabel(r"$H$")
        key_ax.set_ylabel(r"$\delta$")
        key_ax.set_title("Color key", fontsize=10)
        facet_id = "_".join(str(value).replace(".", "p") for value in facet_values)
        fig.savefig(figures / f"overview_time_{facet_id}.png", dpi=160,
                    bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        count += 1
    return count


def make_plots(csv_path: Path) -> int:
    table = pd.read_csv(csv_path, dtype={"run_id": str})
    table = table.loc[table.status == "completed"]
    figures = csv_path.parent / "figures"
    figures.mkdir(exist_ok=True)
    for sweep in (*SWEEPS, "h", "overview"):
        for old_plot in figures.glob(f"{sweep}_*.png"):
            old_plot.unlink()
    if table.empty:
        return 0
    return (_colored_sweep(table, "H", figures)
            + _colored_sweep(table, "delta", figures)
            + _other_sweeps(table, figures)
            + _overview_sweep(table, "H", figures)
            + _overview_sweep(table, "delta", figures)
            + _overview_time(table, figures))


def plot_config(config_path: Path) -> None:
    """Redraw every completed run for this experiment, without a solver import."""
    from .experiment_config import load_config

    config = load_config(config_path)
    csv_path = Path("work") / config["experiment_name"] / "runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results found at {csv_path}")
    count = make_plots(csv_path)
    print(f"Created {count} figures in {csv_path.parent / 'figures'}")
