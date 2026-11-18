"""Plot total corrector errors using the R/T layout of the MsFEM experiments."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NORMS = ("l2", "h1_semi", "h1")
KINDS = ("elliptic", "parabolic", "modified")
STYLES = {
    "elliptic": dict(color="tab:blue", linestyle="--"),
    "parabolic": dict(color="tab:red", linestyle="-"),
    "modified": dict(color="tab:green", linestyle="-."),
}
KIND_LABELS = {
    "elliptic": r"$\alpha=\mathrm{ell}$",
    "parabolic": r"$\alpha=\mathrm{par}$",
    "modified": r"$\alpha=\mathrm{mod}$",
}
# Matplotlib mathtext renders the canonical \lVert and \rVert as \|.
Y_LABELS = {
    "l2": r"$E^\alpha_{\chi,\|\cdot\|_{L^2}}$",
    "h1_semi": r"$E^\alpha_{\chi,|\cdot|_{H^1}}$",
    "h1": r"$E^\alpha_{\chi,\|\cdot\|_{H^1}}$",
}


def _id(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _facet(table) -> str:
    first = table.iloc[0]
    return f"{first['problem']}_{_id(first['epsilon'])}_{_id(first['points_per_unit'])}"


def _prepare_dirs(figures: Path) -> None:
    directories = ("R", "R/fixed_T", "R/fixed_default_T", "T", "T/fixed_R",
                   "overview")
    figures.mkdir(parents=True, exist_ok=True)
    for directory in directories:
        (figures / directory).mkdir(parents=True, exist_ok=True)
    # Remove figures produced by earlier versions of this experiment on redraw.
    for image in figures.rglob("*.png"):
        if image.name.startswith(("total_", "boundary_", "modeling_", "R_", "T_",
                                  "fixed_T_", "fixed_R_", "fixed_default_T_", "overview_")):
            image.unlink()


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _format(ax, x: str, norm: str, title: str = "") -> None:
    ax.set(xlabel=f"${x}$", ylabel=Y_LABELS[norm], title=title)
    ax.grid(True, which="both", alpha=0.25)


def _elliptic_R_axes(ax, table, norm: str) -> bool:
    rows = table.loc[table.corrector_kind == "elliptic"].sort_values("R")
    if len(rows) < 2:
        return False
    ax.semilogy(rows.R, rows[f"total_{norm}"], label=KIND_LABELS["elliptic"],
                **STYLES["elliptic"])
    _format(ax, "R", norm)
    ax.legend(fontsize=9)
    return True


def _default_axes(ax, table, norm: str) -> bool:
    plotted = False
    for kind in KINDS:
        source = ["none"] if kind == "elliptic" else ["default", "grid_default"]
        rows = table.loc[(table.corrector_kind == kind) &
                         table.time_source.isin(source)].sort_values("R")
        if len(rows) < 2:
            continue
        ax.semilogy(rows.R, rows[f"total_{norm}"], label=KIND_LABELS[kind], **STYLES[kind])
        plotted = True
    _format(ax, "R", norm, r"$T=T_{\mathrm{default}}(R)$")
    if plotted:
        ax.legend(fontsize=9)
    return plotted


def _fixed_T_axes(ax, table, grid, T: float, norm: str) -> bool:
    rows_at_time = grid.loc[np.isclose(grid["T"], T)]
    plotted = False
    for kind in KINDS:
        rows = (table if kind == "elliptic" else rows_at_time)
        rows = rows.loc[rows.corrector_kind == kind].sort_values("R")
        if len(rows) < 2:
            continue
        ax.semilogy(rows.R, rows[f"total_{norm}"], label=KIND_LABELS[kind], **STYLES[kind])
        plotted = True
    _format(ax, "R", norm, rf"$T={T:g}$")
    if plotted:
        ax.legend(fontsize=9)
    return plotted


def _fixed_R_axes(ax, table, grid, R: float, norm: str) -> bool:
    rows_at_R = grid.loc[np.isclose(grid.R, R)]
    plotted = False
    for kind in ("parabolic", "modified"):
        rows = rows_at_R.loc[rows_at_R.corrector_kind == kind].sort_values("T")
        if len(rows) < 2:
            continue
        ax.semilogy(rows["T"], rows[f"total_{norm}"], label=KIND_LABELS[kind],
                    **STYLES[kind])
        plotted = True
    baseline = table.loc[np.isclose(table.R, R) & (table.corrector_kind == "elliptic")]
    if plotted and len(baseline) == 1:
        ax.axhline(float(baseline.iloc[0][f"total_{norm}"]), color="tab:blue",
                   linestyle="--", label=KIND_LABELS["elliptic"])
    _format(ax, "T", norm, rf"$R={R:g}$")
    if plotted:
        ax.legend(fontsize=9)
    return plotted


def _single_and_overview(table, grid, figures: Path) -> int:
    count = 0
    facet = _facet(table)
    times = np.sort(grid["T"].unique())
    radii = np.sort(grid.R.unique())
    if table.corrector_kind.eq("elliptic").all():
        families = [("R", None, _elliptic_R_axes, figures / "R")]
    else:
        families = [("fixed_default_T", None, _default_axes, figures / "R/fixed_default_T")]
    families += [("fixed_T", T, _fixed_T_axes, figures / "R/fixed_T") for T in times]
    families += [("fixed_R", R, _fixed_R_axes, figures / "T/fixed_R") for R in radii]
    for family, value, draw, directory in families:
        suffix = "" if value is None else f"_{'T' if family == 'fixed_T' else 'R'}{_id(value)}"
        panels = []
        for norm in NORMS:
            fig, ax = plt.subplots(figsize=(7, 4.4))
            args = (table, norm) if value is None else (table, grid, value, norm)
            if draw(ax, *args):
                _save(fig, directory / f"{family}_corrector_{norm}_error_{facet}{suffix}.png")
                count += 1
                panels.append(norm)
            else:
                plt.close(fig)
        if len(panels) < 2:
            continue
        upscaling_rows = _upscaling_rows(table)
        upscaling_panel = (family in ("R", "fixed_default_T")
                           and any((upscaling_rows.corrector_kind == kind).sum() >= 2
                                   for kind in KINDS))
        panel_count = len(panels) + int(upscaling_panel)
        fig, axes = plt.subplots(1, panel_count, figsize=(5.3 * panel_count, 4.5))
        for ax, norm in zip(np.atleast_1d(axes), panels):
            args = (table, norm) if value is None else (table, grid, value, norm)
            draw(ax, *args)
        if upscaling_panel:
            _upscaling_axes(np.atleast_1d(axes)[-1], table)
        _save(fig, figures / "overview" / f"overview_{family}_{facet}{suffix}.png")
        count += 1
    return count


def _colored_sweep(table, grid, figures: Path, norm: str, axis: str) -> int:
    colored = "R" if axis == "T" else "T"
    values = np.sort(grid[colored].unique())
    if len(values) < 2 or grid[axis].nunique() < 2:
        return 0
    fig, ax = plt.subplots(figsize=(8.5, 5))
    palette = plt.cm.viridis if colored == "R" else plt.cm.plasma
    colors = palette(np.linspace(0.18, 0.88, len(values)))
    for value, color in zip(values, colors):
        for kind in ("parabolic", "modified"):
            rows = grid.loc[np.isclose(grid[colored], value) &
                            (grid.corrector_kind == kind)].sort_values(axis)
            if rows.empty:
                continue
            style = {**STYLES[kind], "color": color}
            ax.semilogy(rows[axis], rows[f"total_{norm}"],
                        label=rf"${colored}={value:g}$, {KIND_LABELS[kind]}", **style)
        if axis == "T":
            baseline = table.loc[np.isclose(table.R, value) &
                                 (table.corrector_kind == "elliptic")]
            if len(baseline) == 1:
                ax.axhline(float(baseline.iloc[0][f"total_{norm}"]), color=color,
                           linestyle=":", label=rf"$R={value:g}$, {KIND_LABELS['elliptic']}")
    if axis == "R":
        baseline = table.loc[table.corrector_kind == "elliptic"].sort_values("R")
        if len(baseline) >= 2:
            ax.semilogy(baseline.R, baseline[f"total_{norm}"], color="black",
                        linestyle=":", label=KIND_LABELS["elliptic"])
    _format(ax, axis, norm)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, figures / axis /
          f"{axis}_corrector_{norm}_error_{_facet(table)}_colored_by_{colored}.png")
    return 1


def _heatmap(grid, figures: Path, norm: str) -> int:
    radii = np.sort(grid.R.unique())
    times = np.sort(grid["T"].unique())
    if len(radii) < 2 or len(times) < 2:
        return 0
    panels = []
    for kind in ("parabolic", "modified"):
        pivot = grid.loc[grid.corrector_kind == kind].pivot_table(
            index="R", columns="T", values=f"total_{norm}", aggfunc="first"
        ).reindex(index=radii, columns=times)
        if pivot.notna().all().all():
            panels.append((kind, np.log10(np.maximum(pivot.to_numpy(), 1e-18))))
    if not panels:
        return 0
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)
    limits = (min(values.min() for _, values in panels),
              max(values.max() for _, values in panels))
    for ax, (kind, values) in zip(axes.flat, panels):
        image = ax.imshow(values, origin="lower", aspect="auto", vmin=limits[0], vmax=limits[1],
                          extent=(times[0] - (times[1] - times[0]) / 2,
                                  times[-1] + (times[-1] - times[-2]) / 2,
                                  radii[0] - (radii[1] - radii[0]) / 2,
                                  radii[-1] + (radii[-1] - radii[-2]) / 2))
        ax.set(xlabel="$T$", ylabel="$R$",
               title=f"{KIND_LABELS[kind]}: " + r"$\log_{10}$" + Y_LABELS[norm])
        fig.colorbar(image, ax=ax)
    _save(fig, figures / "overview" /
          f"overview_corrector_{norm}_error_{_facet(grid)}_heatmap.png")
    return 1


def _upscaling_rows(table):
    if not {"a0_error_frobenius", "upscaling_domain"} <= set(table.columns):
        return table.iloc[:0]
    valid = (table.upscaling_domain.eq("K_L")
             & table.a0_error_frobenius.gt(0)
             & np.isfinite(table.a0_error_frobenius))
    # A time sweep can have multiple values at each R; this R plot uses
    # the default T(R) for timed correctors.
    selected = table.corrector_kind.eq("elliptic") | table.time_source.isin(
        ["default", "grid_default"]
    )
    return table.loc[valid & selected].sort_values("R")


def _upscaling_axes(ax, table) -> bool:
    rows = _upscaling_rows(table)
    plotted = False
    guide = None
    for kind in KINDS:
        method = rows.loc[rows.corrector_kind == kind]
        if len(method) < 2:
            continue
        R = method.R.to_numpy(dtype=float)
        errors = method.a0_error_frobenius.to_numpy(dtype=float)
        ax.loglog(R, errors, label=KIND_LABELS[kind], **STYLES[kind])
        if guide is None:
            guide = (R, errors[0] * R[0] / R)
        plotted = True
    if not plotted:
        return False
    ax.loglog(*guide, color="0.4", linestyle=":", label=r"$O(R^{-1})$")
    ax.set(xlabel=r"$R$",
           ylabel=r"$\|a_{R,L}^{0,\alpha,h}-a^0\|_F$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    return True


def _upscaling_plot(table, figures: Path) -> int:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    if not _upscaling_axes(ax, table):
        plt.close(fig)
        return 0
    _save(fig, figures / "R" / f"R_upscaling_error_{_facet(table)}.png")
    return 1


def make_plots(csv_path: Path) -> int:
    table = pd.read_csv(csv_path, dtype={"run_id": str})
    table = table.loc[table.status == "completed"].copy()
    snapshot = csv_path.parent / "config.yaml"
    if snapshot.exists():
        from .experiment_config import expand_runs
        from experiments.solver_config import KSP_DIAGNOSTICS, MFN_DIAGNOSTICS
        import yaml
        config = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
        config.setdefault("experiment", "correctors")
        if config["experiment"] != "correctors":
            raise ValueError("Saved experiment is not a corrector experiment")
        old_options = config.get("solver_parameters", {})
        if isinstance(old_options, dict) and old_options.keys() <= (
            KSP_DIAGNOSTICS | MFN_DIAGNOSTICS
        ):
            config.pop("solver_parameters", None)
        selected = {run["run_id"] for run in expand_runs(config)}
        table = table.loc[table.run_id.isin(selected)]
    if table.empty:
        return 0
    figures = csv_path.parent / "figures"
    _prepare_dirs(figures)
    grid = table.loc[table.time_source.isin(["grid", "grid_default"])].copy()
    count = _single_and_overview(table, grid, figures)
    count += _upscaling_plot(table, figures)
    for norm in NORMS:
        count += _colored_sweep(table, grid, figures, norm, "R")
        count += _colored_sweep(table, grid, figures, norm, "T")
        count += _heatmap(grid, figures, norm)
    return count
