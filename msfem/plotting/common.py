"""Shared plot labels, styling, and grouping helpers."""

import math

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors
from matplotlib.lines import Line2D
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
FIXED_H_METRICS = (
    "msfem_l2_error", "msfem_h1_error", "msfem_h1_semi_error",
)
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


def _avoid_overzoom(ax):
    """Show meaningful scale when plotted values differ only by roundoff."""
    values = [float(value) for line in ax.lines for value in line.get_ydata()
              if pd.notna(value) and math.isfinite(float(value))]
    if values and min(values) > 0 and max(values) - min(values) < 1e-4 * max(values):
        center = (min(values) + max(values)) / 2
        ax.set_ylim(center * 0.9, center * 1.1)


def _overview_axes(rows):
    height = 6.0 if rows == 1 else 6.3 * rows + 0.4
    fig, axes = plt.subplots(rows, 5, figsize=(24, height),
                             squeeze=False, sharex="col")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.82,
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


