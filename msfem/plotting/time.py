"""Final-time sweep plots for parabolic and modified correctors."""

import math

from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

from ..experiment_data import JUMP_COLUMNS
from .common import KINDS, MODE_STYLES, _avoid_overzoom, _groups, _parameter_title, _time_groups
from .notation import AXIS_LABELS, KIND_LABELS, OVERVIEW_METRICS, Y_LABELS, parameter_label
from .files import facet_id, save_figure


def _time_sweep(table, figures, *, compare_all=False):
    """Compare time dependence at fixed H, with delta as color."""
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for facet_values, frame in _groups(table, facets):
        if compare_all:
            matching = [rows for _, _, rows in _time_groups(frame, compare_all=True)]
            if not matching:
                continue
            frame = pd.concat(matching).drop_duplicates("run_id")
        timed = frame.loc[frame.final_time.notna()]
        tested_times = sorted(timed.final_time.unique())
        if len(tested_times) < 2:
            continue
        H_values = sorted(timed.H.dropna().unique(), reverse=True)
        deltas = sorted(timed.delta.dropna().unique())
        kinds = [kind for kind in ("parabolic", "modified") if kind in set(timed.corrector_kind)]
        palette = {delta: cm.viridis(0.15 + 0.75 * index / max(1, len(deltas) - 1))
                   for index, delta in enumerate(deltas)}
        settings = timed[["final_time", "num_time_steps"]].dropna().drop_duplicates()
        tick_labels = []
        for T in tested_times:
            steps = settings.loc[settings.final_time == T, "num_time_steps"]
            tick_labels.append(f"{T:g}\n" + parameter_label("num_time_steps", int(steps.iloc[0])) if len(steps) == 1
                               else f"{T:g}")
        ncols = len(H_values) if len(H_values) <= 3 else 2
        nrows = math.ceil(len(H_values) / ncols)
        for metric in OVERVIEW_METRICS:
            if timed[metric].notna().sum() < 2:
                continue
            fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                                     figsize=(5.8 * ncols, 3.7 * nrows + 0.8))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19 if nrows == 1 else 0.12,
                                top=0.79, wspace=0.28, hspace=0.42)
            fig.suptitle(_parameter_title(facets, facet_values), fontsize=12, y=0.99)
            fig.supylabel(Y_LABELS[metric], fontsize=10, x=0.01)
            for ax, H in zip(axes.flat, H_values):
                H_rows = frame.loc[frame.H == H]
                for delta in deltas:
                    at_delta = H_rows.loc[H_rows.delta == delta]
                    for kind in kinds:
                        rows = at_delta.loc[at_delta.corrector_kind == kind].sort_values("final_time")
                        y = pd.to_numeric(rows[metric], errors="coerce")
                        if y.notna().any():
                            ax.plot(rows.final_time, y, color=palette[delta], marker=KINDS[kind][1],
                                    linewidth=1.6, **MODE_STYLES[kind])
                    reference = at_delta.loc[at_delta.corrector_kind == "elliptic", metric]
                    if not reference.empty and pd.notna(reference.iloc[0]):
                        reference_value = float(reference.iloc[0])
                        ax.axhline(reference_value, color=palette[delta],
                                   linestyle=":", linewidth=1.2)
                ax.set_title(parameter_label("H", H, power_two=True), fontsize=11)
                ax.set_xticks(tested_times, tick_labels)
                ax.set_xlabel(AXIS_LABELS["final_time"])
                if metric in JUMP_COLUMNS:
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
            for ax in list(axes.flat)[len(H_values):]:
                ax.set_visible(False)
            handles = [Line2D([0], [0], color=palette[delta],
                              label=parameter_label("delta", delta)) for delta in deltas]
            handles.extend(Line2D([0], [0], color="0.35", marker=KINDS[kind][1],
                                  label=KIND_LABELS[kind], **MODE_STYLES[kind]) for kind in kinds)
            if "elliptic" in set(frame.corrector_kind):
                handles.append(Line2D([0], [0], color="0.35", linestyle=":",
                                      label=KIND_LABELS["elliptic"] + " reference"))
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.91),
                       ncol=min(5, len(handles)), frameon=False, fontsize=9)
            save_figure(fig, figures / "time" / f"time_{metric}_{facet_id(facet_values)}.png")
            count += 1
    return count


