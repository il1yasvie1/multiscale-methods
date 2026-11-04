"""H and delta sweeps, including fixed-H delta panels."""

import math

import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    FIXED_H_METRICS, KINDS, METRICS, OVERVIEW_METRICS, OVERVIEW_TITLES,
    Y_LABELS, _add_color_keys, _avoid_overzoom, _color, _groups,
    _overview_axes, _overview_keys, _parameter_title, _power_two_label,
    _time_groups,
)
from .files import facet_id, save_figure, save_sweep_figure


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
        save_figure(fig, figures / sweep / f"overview_{sweep}_{facet_id(facet_values)}.png")
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
                save_sweep_figure(fig, figures, sweep, metric, facet_values, T, Nt)
                count += 1
    return count


def _fixed_h_delta_sweep(table, figures):
    """Show each coarse resolution on its own y scale in a delta sweep."""
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for T, Nt, time_rows in _time_groups(table):
        for facet_values, frame in _groups(time_rows, facets):
            H_values = sorted(frame.H.dropna().unique(), reverse=True)
            if frame.delta.nunique() < 2 or not H_values:
                continue
            kinds = [kind for kind in KINDS if kind in set(frame.corrector_kind)]
            for metric in FIXED_H_METRICS:
                if frame[metric].notna().sum() < 2:
                    continue
                ncols = len(H_values) if len(H_values) <= 3 else 2
                nrows = math.ceil(len(H_values) / ncols)
                fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                                         figsize=(6.2 * ncols, 3.8 * nrows + 0.8))
                fig.subplots_adjust(left=0.14, right=0.97, bottom=0.12,
                                    top=0.80, wspace=0.28, hspace=0.35)
                fig.suptitle(_parameter_title(facets, facet_values, T, Nt),
                             fontsize=12, y=0.99)
                fig.supylabel(Y_LABELS[metric], fontsize=10, x=0.01)
                handles = []
                for ax, H in zip(axes.flat, H_values):
                    H_rows = frame.loc[frame.H == H]
                    for kind in kinds:
                        rows = H_rows.loc[H_rows.corrector_kind == kind].sort_values("delta")
                        y = pd.to_numeric(rows[metric], errors="coerce")
                        if y.notna().sum() < 2:
                            continue
                        line, = ax.plot(rows.delta, y, color=KINDS[kind][0](0.8),
                                        marker=KINDS[kind][1], linewidth=1.6,
                                        linestyle="--" if kind == "elliptic" else "-",
                                        markerfacecolor="white" if kind == "elliptic" else None,
                                        label=kind)
                        if kind not in [handle.get_label() for handle in handles]:
                            handles.append(line)
                    ax.set_title(rf"$H={_power_two_label(H).strip('$')}$", fontsize=11)
                    ax.set_xlabel(r"$\delta$")
                    ax.grid(alpha=0.25)
                for ax in list(axes.flat)[len(H_values):]:
                    ax.set_visible(False)
                if not handles:
                    plt.close(fig)
                    continue
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.91),
                           ncol=len(handles), frameon=False)
                save_sweep_figure(fig, figures / "delta", "fixed_H", metric,
                                  facet_values, T, Nt)
                count += 1
    return count


