"""H and delta sweeps, including panels at fixed delta or H."""

import math

import matplotlib.pyplot as plt
import pandas as pd

from ..experiment_data import JUMP_COLUMNS
from .common import (
    KINDS, MODE_STYLES, _add_color_keys, _avoid_overzoom, _color, _groups,
    _overview_axes, _overview_keys, _parameter_title, _time_groups,
)
from .notation import (
    AXIS_LABELS, FEM_REFERENCE_STYLES, KIND_LABELS, METRICS, OVERVIEW_METRICS,
    OVERVIEW_TITLES, SOLUTION_ERROR_METRICS, Y_LABELS, parameter_label, power_two_label,
)
from .files import facet_id, save_figure, save_sweep_figure


def _add_fem_references(ax, frame, metric, sweep):
    """Coarse FEM references for MsFEM L2 and H1 seminorm errors versus H."""
    if sweep != "H" or metric not in ("msfem_l2_error", "msfem_h1_semi_error"):
        return
    for baseline, style in FEM_REFERENCE_STYLES.items():
        column = metric.replace("msfem", baseline, 1)
        if column not in frame:
            continue
        rows = frame[["H", column]].dropna().drop_duplicates("H").sort_values("H")
        if rows.empty:
            continue
        ax.plot(rows.H, rows[column], linewidth=1.4, **style)


def _overview_sweep(table, sweep, figures, *, compare_all=False):
    """One row per time setting; delta overviews use a fixed H per figure."""
    x = "H" if sweep == "H" else "delta"
    shade_by = "delta" if sweep == "H" else "H"
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    if sweep == "delta":
        facets += ("H",)
    count = 0
    for facet_values, frame in _groups(table, facets):
        if frame[x].nunique() < 2:
            continue
        settings = list(_time_groups(frame, compare_all=compare_all))
        if not settings:
            continue
        frame = pd.concat([rows for _, _, rows in settings])
        shades = sorted(frame[shade_by].dropna().unique())
        kinds = [kind for kind in KINDS if kind in set(frame.corrector_kind)]
        fig, axes = _overview_axes(len(settings))
        fig.suptitle(_parameter_title(facets, facet_values), fontsize=14)
        for row_index, (T, Nt, rows_at_time) in enumerate(settings):
            for col_index, metric in enumerate(OVERVIEW_METRICS):
                ax = axes[row_index, col_index]
                has_values = False
                for (kind, shade), rows in _groups(rows_at_time,
                                                   ("corrector_kind", shade_by)):
                    rows = rows.sort_values(x)
                    y = pd.to_numeric(rows[metric], errors="coerce")
                    if metric not in JUMP_COLUMNS and sweep == "H":
                        y = y.where(y > 0)
                    if y.notna().any():
                        has_values = True
                        ax.plot(rows[x], y, color=_color(kind, shade, shades[0],
                                shades[-1], shade_by), marker=KINDS[kind][1],
                                linewidth=1.5, **MODE_STYLES[kind])
                if not has_values:
                    ax.text(0.5, 0.5, "Not available: rerun metrics", ha="center",
                            va="center", transform=ax.transAxes, fontsize=9)
                _add_fem_references(ax, rows_at_time, metric, sweep)
                ax.set_title(OVERVIEW_TITLES[metric], fontsize=10, pad=10)
                if col_index == 0:
                    ax.set_ylabel(_parameter_title((), (), T, Nt), fontsize=10)
                if row_index == len(settings) - 1:
                    ax.set_xlabel(AXIS_LABELS[sweep])
                if sweep == "H":
                    ax.set_xscale("log", base=2)
                    if metric not in JUMP_COLUMNS and has_values:
                        ax.set_yscale("log")
                    ticks = sorted(frame.H.unique())
                    ax.set_xticks(ticks, [power_two_label(value) for value in ticks])
                    ax.set_xlim(max(ticks) * 1.05, min(ticks) / 1.05)
                if metric in JUMP_COLUMNS:
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
        _overview_keys(fig, kinds, shades, shade_by,
                       fem_references=(sweep == "H" and any(
                           name in frame and frame[name].notna().any()
                           for name in ("coarse_fem_l2_error", "coarse_fem_h1_semi_error"))))
        save_figure(fig, figures / sweep / f"overview_{sweep}_{facet_id(facet_values)}.png")
        count += 1
    return count


def _colored_sweep(table, sweep, figures, *, compare_all=False):
    """Plot corrector kinds together, shaded by delta or coarse resolution."""
    x = "H" if sweep == "H" else "delta"
    shade_by = "delta" if sweep == "H" else "H"
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for facet_values, facet_rows in _groups(table, facets):
        for T, Nt, frame in _time_groups(facet_rows, compare_all=compare_all):
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
                    if metric not in JUMP_COLUMNS and sweep == "H":
                        y = y.where(y > 0)
                    if y.notna().sum() == 0:
                        continue
                    ax.plot(rows[x], y, color=_color(kind, shade, values[0], values[-1], shade_by),
                            marker=KINDS[kind][1], linewidth=1.8,
                            label="_nolegend_", **MODE_STYLES[kind])

                _add_fem_references(ax, frame, metric, sweep)

                if not ax.lines:
                    plt.close(fig)
                    continue
                ax.set_xlabel(AXIS_LABELS[sweep])
                ax.set_ylabel(Y_LABELS[metric], fontsize=10)
                ax.set_title(_parameter_title(facets, facet_values, T, Nt), fontsize=11)
                if sweep == "H":
                    ax.set_xscale("log", base=2)
                    if metric not in JUMP_COLUMNS:
                        ax.set_yscale("log")
                    ticks = sorted(frame.H.unique())
                    ax.set_xticks(ticks, [power_two_label(value) for value in ticks])
                    ax.invert_xaxis()
                if metric in JUMP_COLUMNS:
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
                _add_color_keys(fig, ax, kinds, values, shade_by)
                save_sweep_figure(fig, figures, sweep, metric, facet_values, T, Nt)
                count += 1
    return count


def _fixed_parameter_sweep(table, sweep, figures, *, compare_all=False):
    """Compare correctors in separate panels for the fixed spatial parameter."""
    x = sweep
    fixed = "delta" if sweep == "H" else "H"
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for facet_values, facet_rows in _groups(table, facets):
        for T, Nt, frame in _time_groups(facet_rows, compare_all=compare_all):
            fixed_values = sorted(frame[fixed].dropna().unique(), reverse=fixed == "H")
            if frame[x].nunique() < 2 or not fixed_values:
                continue
            kinds = [kind for kind in KINDS if kind in set(frame.corrector_kind)]
            for metric in SOLUTION_ERROR_METRICS:
                if frame[metric].notna().sum() < 2:
                    continue
                ncols = len(fixed_values) if len(fixed_values) <= 3 else 2
                nrows = math.ceil(len(fixed_values) / ncols)
                fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                                         figsize=(6.2 * ncols, 3.8 * nrows + 0.8))
                fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12,
                                    top=0.80, wspace=0.28, hspace=0.35)
                fig.suptitle(_parameter_title(facets, facet_values, T, Nt),
                             fontsize=12, y=0.99)
                fig.supylabel(Y_LABELS[metric], fontsize=10, x=0.01)
                handles = []
                for ax, value in zip(axes.flat, fixed_values):
                    fixed_rows = frame.loc[frame[fixed] == value]
                    for kind in kinds:
                        rows = fixed_rows.loc[fixed_rows.corrector_kind == kind].sort_values(x)
                        y = pd.to_numeric(rows[metric], errors="coerce")
                        if sweep == "H":
                            y = y.where(y > 0)
                        if not y.notna().any():
                            continue
                        line, = ax.plot(rows[x], y, color=KINDS[kind][0](0.8),
                                        marker=KINDS[kind][1], linewidth=1.6,
                                        label=KIND_LABELS[kind], **MODE_STYLES[kind])
                        if KIND_LABELS[kind] not in [handle.get_label() for handle in handles]:
                            handles.append(line)
                    _add_fem_references(ax, fixed_rows, metric, sweep)
                    existing = {handle.get_label() for handle in handles}
                    handles.extend(line for line in ax.lines
                                   if not line.get_label().startswith("_")
                                   and line.get_label() not in existing)
                    if sweep == "H":
                        ax.set_title(parameter_label("delta", value), fontsize=11)
                        ax.set_xscale("log", base=2)
                        ax.set_yscale("log")
                        ticks = sorted(frame.H.unique())
                        ax.set_xticks(ticks, [power_two_label(tick) for tick in ticks])
                        ax.set_xlim(max(ticks) * 1.05, min(ticks) / 1.05)
                        _avoid_overzoom(ax)
                    else:
                        ax.set_title(parameter_label("H", value, power_two=True), fontsize=11)
                    ax.set_xlabel(AXIS_LABELS[sweep])
                    ax.grid(alpha=0.25)
                for ax in list(axes.flat)[len(fixed_values):]:
                    ax.set_visible(False)
                if not handles:
                    plt.close(fig)
                    continue
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.91),
                           ncol=len(handles), frameon=False)
                save_sweep_figure(fig, figures / sweep, f"fixed_{fixed}", metric,
                                  facet_values, T, Nt)
                count += 1
    return count


