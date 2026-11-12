"""Shared plot styling, legends, and grouping helpers."""

import math

import matplotlib
matplotlib.use("Agg")
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd


from .notation import (
    AXIS_LABELS, FEM_REFERENCE_STYLES, KIND_LABELS, OVERVIEW_METRICS,
    PARAMETER_SYMBOLS, parameter_expression, parameter_label, power_two_label,
)


KINDS = {"elliptic": (cm.Blues, "o"), "parabolic": (cm.Reds, "s"),
         "modified": (cm.Greens, "^")}
MODE_STYLES = {
    "elliptic": dict(linestyle="--", markersize=8, zorder=3),
    "parabolic": dict(linestyle="-", markersize=4, zorder=4),
    "modified": dict(linestyle="-.", markersize=9, markerfacecolor="none", zorder=5),
}


def _parameter_title(columns, values, T=None, Nt=None):
    parts = [parameter_expression(key, value) for key, value in zip(columns, values)
             if (key in PARAMETER_SYMBOLS or key in ("assembly_type", "N_H", "N_h"))
             and pd.notna(value)]
    if T is not None:
        parts.append(parameter_expression("final_time", T))
        if Nt is not None:
            parts.append(parameter_expression("num_time_steps", Nt))
    return "$" + r",\;".join(parts) + "$"


def _groups(frame, columns):
    for values, rows in frame.groupby(list(columns), dropna=False):
        yield values if isinstance(values, tuple) else (values,), rows


def _time_groups(table, *, compare_all=False):
    timed = table.loc[table.final_time.notna()]
    if timed.empty and not compare_all:
        yield None, None, table
    for T, at_time in timed.groupby("final_time"):
        shared = ((table.corrector_kind == "elliptic")
                  | ((table.corrector_kind == "modified") & (table.final_time == T)))
        steps = sorted(at_time.num_time_steps.dropna().unique())
        for Nt in steps or [None]:
            selected = shared | ((table.final_time == T) & (table.num_time_steps == Nt))
            rows = table.loc[selected]
            if compare_all:
                coverage = rows.groupby(["H", "delta"]).corrector_kind.transform("nunique")
                rows = rows.loc[coverage == len(KINDS)]
            if not rows.empty:
                yield T, None if Nt is None else int(Nt), rows


def _color(kind, value, low, high, parameter):
    cmap, _ = KINDS[kind]
    strength = 0.65 if high == low else (value - low) / (high - low)
    if parameter == "H":
        strength = 1 - strength
    return cmap(0.35 + 0.6 * strength)


def _add_color_keys(fig, ax, kinds, values, parameter):
    """Place the category legend and parameter color scales outside the axes."""
    handles = [
        Line2D([0], [0], color=KINDS[kind][0](0.8), marker=KINDS[kind][1],
               **MODE_STYLES[kind],
               label=KIND_LABELS[kind] if len(values) > 1 else f"{KIND_LABELS[kind]}, {parameter_label(parameter, values[0])}")
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
        bar_ax = fig.add_axes([0.76 + 0.22 * index / len(kinds), 0.2, 0.019, 0.46])
        bar = fig.colorbar(scale, cax=bar_ax, ticks=values)
        if parameter == "H":
            bar.ax.set_yticklabels([power_two_label(value) for value in values])
        bar.ax.tick_params(labelsize=7)
        bar.ax.set_title(
            f"{KIND_LABELS[kind]}\n" + AXIS_LABELS[parameter],
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
    fig, axes = plt.subplots(rows, len(OVERVIEW_METRICS), figsize=(4.8 * len(OVERVIEW_METRICS), height),
                             squeeze=False, sharex="col")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.82,
                        wspace=0.28, hspace=0.55)
    for ax in axes.flat:
        ax.set_box_aspect(0.9)
    return fig, axes


def _overview_keys(fig, kinds, values, parameter, *, fem_references=False):
    handles = [Line2D([0], [0], color=KINDS[kind][0](0.8),
                      marker=KINDS[kind][1],
                      **MODE_STYLES[kind],
                      label=KIND_LABELS[kind]) for kind in kinds]
    if fem_references:
        handles.extend(Line2D([0], [0], **style)
                       for style in FEM_REFERENCE_STYLES.values())
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.99),
               ncol=len(handles), frameon=False)
    if len(values) < 2:
        fig.text(0.5, 0.05, parameter_label(parameter, values[0]), fontsize=10,
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
        slot = 0.8 / len(kinds)
        bar_ax = fig.add_axes([0.1 + slot * (index + 0.15), 0.035, 0.7 * slot, 0.018])
        bar = fig.colorbar(scale, cax=bar_ax, ticks=values,
                           orientation="horizontal")
        if parameter == "H":
            bar.ax.set_xticklabels([power_two_label(value) for value in values])
        bar.ax.tick_params(labelsize=8, pad=2)
        bar.ax.set_title(f"{KIND_LABELS[kind]}\n" + AXIS_LABELS[parameter],
                         fontsize=9, pad=3)


