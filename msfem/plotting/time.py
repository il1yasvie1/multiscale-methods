"""Final-time sweep plots for parabolic correctors."""

import math

from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    OVERVIEW_METRICS, Y_LABELS, _avoid_overzoom, _groups,
    _parameter_title, _power_two_label,
)
from .files import facet_id, save_figure


def _time_sweep(table, figures):
    """Compare time dependence at fixed H, with delta as color."""
    facets = ("problem", "epsilon", "N_h", "assembly_type")
    count = 0
    for facet_values, frame in _groups(table, facets):
        parabolic = frame.loc[frame.corrector_kind == "parabolic"]
        tested_times = sorted(parabolic.final_time.dropna().unique())
        if len(tested_times) < 2:
            continue
        H_values = sorted(parabolic.H.dropna().unique(), reverse=True)
        deltas = sorted(parabolic.delta.dropna().unique())
        palette = {delta: cm.viridis(0.15 + 0.75 * index / max(1, len(deltas) - 1))
                   for index, delta in enumerate(deltas)}
        settings = parabolic[["final_time", "num_time_steps"]].drop_duplicates()
        tick_labels = []
        for T in tested_times:
            steps = settings.loc[settings.final_time == T, "num_time_steps"]
            tick_labels.append(f"{T:g}\n$N_t={int(steps.iloc[0])}$" if len(steps) == 1
                               else f"{T:g}")
        ncols = len(H_values) if len(H_values) <= 3 else 2
        nrows = math.ceil(len(H_values) / ncols)
        for metric in OVERVIEW_METRICS:
            if parabolic[metric].notna().sum() < 2:
                continue
            fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                                     figsize=(5.8 * ncols, 3.7 * nrows + 0.8))
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19 if nrows == 1 else 0.12,
                                top=0.79, wspace=0.28, hspace=0.42)
            fig.suptitle(_parameter_title(facets, facet_values), fontsize=12, y=0.99)
            fig.supylabel(Y_LABELS[metric], fontsize=10, x=0.01)
            for ax, H in zip(axes.flat, H_values):
                H_rows = frame.loc[frame.H == H]
                for delta_index, delta in enumerate(deltas):
                    rows = H_rows.loc[(H_rows.delta == delta)
                                      & (H_rows.corrector_kind == "parabolic")].sort_values("final_time")
                    y = pd.to_numeric(rows[metric], errors="coerce")
                    if metric == "msfem_jump":
                        y = y.mask(y.abs() <= 1e-12, 0.0)
                    if y.notna().any():
                        ax.plot(rows.final_time, y, color=palette[delta], marker="o",
                                linewidth=1.6, markersize=max(4, 9 - delta_index),
                                markerfacecolor="white" if delta_index == 0 else palette[delta])
                    reference = H_rows.loc[(H_rows.delta == delta)
                                            & (H_rows.corrector_kind == "elliptic"), metric]
                    if not reference.empty and pd.notna(reference.iloc[0]):
                        reference_value = float(reference.iloc[0])
                        if metric == "msfem_jump" and abs(reference_value) <= 1e-12:
                            reference_value = 0.0
                        ax.axhline(reference_value, color=palette[delta],
                                   linestyle=":", linewidth=1.2)
                ax.set_title(rf"$H={_power_two_label(H).strip('$')}$", fontsize=11)
                ax.set_xticks(tested_times, tick_labels)
                ax.set_xlabel(r"$T$")
                if metric == "msfem_jump":
                    ax.set_ylim(bottom=0)
                _avoid_overzoom(ax)
                ax.grid(alpha=0.25)
            for ax in list(axes.flat)[len(H_values):]:
                ax.set_visible(False)
            handles = [Line2D([0], [0], color=palette[delta], marker="o",
                              label=rf"$\delta={delta:g}$") for delta in deltas]
            if "elliptic" in set(frame.corrector_kind):
                handles.append(Line2D([0], [0], color="0.35", linestyle=":",
                                      label="elliptic reference"))
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.91),
                       ncol=min(5, len(handles)), frameon=False, fontsize=9)
            save_figure(fig, figures / "time" / f"time_{metric}_{facet_id(facet_values)}.png")
            count += 1
    return count


