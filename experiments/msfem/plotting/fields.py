"""Periodic field figures: python -m experiments.msfem.plotting.fields --help.

Use the existing MsFEM solver, retain the complete selected patch, and save
per-mode NPZ archives for redraws without DOLFINx (--plot-only).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from .common import KINDS
from .notation import KIND_LABELS


def exact_fields(points, epsilon):
    """NumPy evaluations of PeriodicCoefficientProblem's analytic expressions.

    Correctors are epsilon*chi_j(x/epsilon), with zero periodic mean;
    do not align their mean with the local Dirichlet correctors.
    """
    x, y = np.asarray(points).T
    theta = 2 * np.pi * x / epsilon
    coefficient = np.array([2 / (2 + np.cos(theta)), 1 + 0.5 * np.cos(theta)]) / (8 * np.pi**2)
    correctors = np.array([epsilon / (4 * np.pi) * np.sin(theta), np.zeros_like(x)])
    solution = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
                + epsilon / 2 * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(theta))
    return coefficient, correctors, solution


def _mesh_arrays(domain):
    return domain.geometry.x[:, :2].copy(), np.asarray(domain.geometry.dofmaps[0]).copy()


def _function_arrays(function):
    """Preserve DG nodes and cell connectivity; never merge coincident nodes."""
    V = function.function_space
    n = V.mesh.topology.index_map(V.mesh.topology.dim).size_local
    cells = np.array([V.dofmap.cell_dofs(i) for i in range(n)])
    return V.tabulate_dof_coordinates()[:, :2].copy(), cells, function.x.array.real.copy()


def _edges(points, cells, boundary=False):
    edges = np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return points[edges[counts == 1] if boundary else edges]


def solve_fields(settings, output):
    from dolfinx import mesh
    from mpi4py import MPI
    from multiscale.msfem import MsFEM_Oversampling
    from multiscale.problems import PeriodicCoefficientProblem

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Run this field generator in one process")
    N_H, N_h = settings["N_H"], settings["N_h"]
    problem = PeriodicCoefficientProblem(settings["epsilon"])
    coarse = mesh.create_unit_square(MPI.COMM_SELF, N_H, N_H)
    cp, cc = _mesh_arrays(coarse)
    centers = cp[cc].mean(axis=1)
    i = N_H // 2 - 1
    target = np.array([i + 2 / 3, i + 1 / 3]) / N_H
    cell = settings["coarse_cell"]
    cell = int(np.argmin(np.sum((centers - target)**2, axis=1))) if cell is None else cell
    if not 0 <= cell < len(cc):
        raise ValueError("Invalid coarse-cell index")
    core = cp[cc[cell]]
    if np.any(core <= 0) or np.any(core >= 1):
        raise ValueError("Select an interior coarse element")
    method = None
    try:
        for kind in KINDS:
            path = output / f"{kind}.npz"
            if path.exists():
                with np.load(path) as archive:
                    if json.loads(str(archive["settings"])) != settings:
                        raise ValueError(f"Different settings in {path}; choose another --output directory")
                print(f"Reuse {path}", flush=True)
                continue
            if method is None:
                method = MsFEM_Oversampling(problem, coarse, (N_h // N_H).bit_length() - 1)
            captured = {}
            processed = 0

            def observe(patch, core_cells, correctors):
                nonlocal processed
                processed += 1
                indices = np.asarray(patch.geometry.dofmaps[0])[core_cells]
                center = patch.geometry.x[indices, :2].mean(axis=(0, 1))
                if np.allclose(center, centers[cell], rtol=0, atol=1e-12):
                    xy, triangles, first = _function_arrays(correctors[0])
                    pp, pc = _mesh_arrays(patch)
                    captured.update(patch_xy=xy, patch_triangles=triangles,
                                    correctors=np.array([first, correctors[1].x.array.real.copy()]),
                                    patch_boundary=_edges(pp, pc, boundary=True))
                if processed % 64 == 0:
                    print(f"{kind}: {processed}/{len(cc)} patches", flush=True)

            options = {} if kind == "elliptic" else {"final_time": settings["T"]}
            if kind == "parabolic":
                options["num_time_steps"] = settings["N_t"]
            method.offline(settings["delta"], corrector_kind=kind,
                           assembly_type=settings["assembly_type"],
                           corrector_observer=observe, **options)
            if not captured:
                raise RuntimeError("The selected local corrector was not captured")
            solution = method.online()
            xy, triangles, values = _function_arrays(solution)
            path_tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(path_tmp, settings=json.dumps(settings),
                                xy=xy, triangles=triangles, solution=values,
                                coarse_xy=cp, coarse_triangles=cc, core=core,
                                coarse_cell=cell, **captured)
            path_tmp.replace(path)
            print(f"Saved {path}", flush=True)

    finally:
        if method is not None:
            method.close()


def _save(fig, output, name):
    path = output / f"{name}.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print(path, flush=True)


def _axes3d(fig, rows, cols, index, title):
    ax = fig.add_subplot(rows, cols, index, projection="3d")
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("$x_1$", labelpad=5)
    ax.set_ylabel("$x_2$", labelpad=5)
    ax.view_init(elev=28, azim=-125)
    ax.set_box_aspect((1, 1, 0.6))
    ax.tick_params(labelsize=8)
    for axis in ("x", "y", "z"):
        ax.locator_params(axis=axis, nbins=4)
    return ax


def plot_analytic(settings, output):
    # Dense analytic sampling resolves the coefficient independently of h.
    n = max(241, int(np.ceil(20 / settings["epsilon"])) + 1)
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, 41))
    coefficient, correctors, _ = exact_fields(np.column_stack([X.ravel(), Y.ravel()]), settings["epsilon"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    norm = Normalize(coefficient.min(), coefficient.max())
    for j, ax in enumerate(axes):
        artist = ax.imshow(coefficient[j].reshape(X.shape), origin="lower", extent=(0, 1, 0, 1),
                           cmap="viridis", norm=norm, interpolation="nearest")
        ax.set(title=rf"$a^\varepsilon_{{{j+1}{j+1}}}(x)$", xlabel="$x_1$", ylabel="$x_2$")
    fig.colorbar(artist, ax=axes, label="Coefficient value", shrink=0.8)
    fig.suptitle(rf"Periodic coefficient, $\varepsilon={settings['epsilon']:g}$")
    _save(fig, output, "01_periodic_coefficient")

    fig = plt.figure(figsize=(13, 6), layout="constrained")
    amplitude = settings["epsilon"] / (4 * np.pi)
    for j in range(2):
        ax = _axes3d(fig, 1, 2, j + 1,
                     rf"Exact $\varepsilon\chi_{j+1}(x/\varepsilon)$" + (" = 0" if j == 1 else ""))
        ax.plot_surface(X, Y, correctors[j].reshape(X.shape), cmap="coolwarm",
                        norm=Normalize(-amplitude, amplitude), rcount=len(Y), ccount=n,
                        linewidth=0, antialiased=True)
        ax.set_zlim(-1.1 * amplitude, 1.1 * amplitude)
        ax.set_zlabel("Corrector value", labelpad=9)
    fig.suptitle(rf"Exact scaled correctors, $\varepsilon={settings['epsilon']:g}$")
    _save(fig, output, "02_exact_correctors")


def _patch_geometry(data, output, title):
    cp, cc, core = data["coarse_xy"], data["coarse_triangles"], data["core"]
    xy, triangles = data["patch_xy"], data["patch_triangles"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax in axes:
        ax.add_collection(PolyCollection(xy[triangles], facecolor="#d7eaf5", edgecolor="none"))
        ax.add_collection(LineCollection(_edges(cp, cc), colors="0.5", linewidths=0.6))
        ax.add_collection(LineCollection(data["patch_boundary"], colors="#176a98", linewidths=2))
        ax.add_patch(Polygon(core, facecolor="#f6b558", edgecolor="#b45d00", linewidth=2))
        ax.text(*core.mean(axis=0), "$K$", ha="center", va="center")
        ax.set_aspect("equal")
        ax.set(xlabel="$x_1$", ylabel="$x_2$")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), title="Selected interior coarse element")
    pad = 0.1 * np.ptp(xy, axis=0).max()
    axes[1].set(xlim=(xy[:, 0].min()-pad, xy[:, 0].max()+pad),
                ylim=(xy[:, 1].min()-pad, xy[:, 1].max()+pad), title="Full oversampling domain")
    fig.suptitle(title)
    fig.legend(handles=[Patch(facecolor="#f6b558", label="Core coarse element K"),
                        Patch(facecolor="#d7eaf5", label="Oversampling region"),
                        Line2D([], [], color="0.5", label="Coarse-element edges")],
               loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output, "03_patch_geometry")


def _patch_coarse_edges(data):
    """Clip coarse lines to the actual triangular/staircase patch footprint."""
    xy, triangles = data["patch_xy"], data["patch_triangles"]
    finder = mtri.Triangulation(*xy.T, triangles=triangles).get_trifinder()
    edges = _edges(data["coarse_xy"], data["coarse_triangles"])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    edges = edges[np.all(edges.max(axis=1) >= lo, axis=1) & np.all(edges.min(axis=1) <= hi, axis=1)]
    points = edges[:, :1] + np.linspace(0, 1, 65)[None, :, None] * (edges[:, 1:] - edges[:, :1])
    segments = np.stack([points[:, :-1], points[:, 1:]], axis=2).reshape(-1, 2, 2)
    inside = np.all(finder(segments[:, :, 0], segments[:, :, 1]) >= 0, axis=1)
    return segments[inside]


def plot_comparisons(settings, output):
    data = {}
    for kind in KINDS:
        with np.load(output / f"{kind}.npz") as archive:
            if json.loads(str(archive["settings"])) != settings:
                raise ValueError(f"Settings differ for {kind}")
            data[kind] = {key: archive[key] for key in archive.files if key != "settings"}
    first = data["elliptic"]
    title = (rf"$H=1/{settings['N_H']},\ h=1/{settings['N_h']},\ "
             rf"\varepsilon={settings['epsilon']:g},\ \delta={settings['delta']:g},\ "
             rf"T={settings['T']:g},\ N_t={settings['N_t']},\ "
             + (r"\mathrm{PG}$" if settings["assembly_type"] == "petrov_galerkin" else r"\mathrm{G}$"))
    _patch_geometry(first, output, title)
    clipped_edges = _patch_coarse_edges(first)
    fig = plt.figure(figsize=(19, 11), layout="constrained")
    fig.suptitle("Local versus exact scaled correctors on the full oversampling domain\n" + title)
    # Same vertical scale for both directions: avoid magnifying roundoff in chi_2=0.
    amplitude = max(settings["epsilon"] / (4 * np.pi),
                    *(np.max(np.abs(d["correctors"])) for d in data.values()))
    floor = -1.3 * amplitude
    for j in range(2):
        for col, (kind, d) in enumerate(data.items()):
            ax = _axes3d(fig, 2, 3, j * 3 + col + 1, KIND_LABELS[kind] + rf", direction $j={j+1}$")
            xy, triangles = d["patch_xy"], d["patch_triangles"]
            ax.plot_trisurf(*xy.T, d["correctors"][j], triangles=triangles,
                            color=KINDS[kind][0](0.7), alpha=0.78, linewidth=0, shade=True)
            edges = _edges(xy, triangles)
            points = edges[:, :1] + np.linspace(0, 1, 5)[None, :, None] * (edges[:, 1:] - edges[:, :1])
            z = exact_fields(points.reshape(-1, 2), settings["epsilon"])[1][j].reshape(points.shape[:2])
            ax.add_collection3d(Line3DCollection(np.dstack([points, z]), colors="0.15", linewidths=0.35))
            for lines, color, width in ((clipped_edges, "0.55", 0.65),
                                        (d["patch_boundary"], "#176a98", 1.5)):
                ax.add_collection3d(Line3DCollection(
                    np.dstack([lines, np.full(lines.shape[:2], floor)]), colors=color, linewidths=width))
            core3d = np.column_stack([d["core"], np.full(3, floor)])
            ax.add_collection3d(Poly3DCollection([core3d], facecolor="#f6b558", edgecolor="#b45d00", alpha=0.9))
            ax.text(*d["core"].mean(axis=0), floor, "$K$", ha="center")
            ax.set_zlim(floor, 1.3 * amplitude)
            ax.set_zlabel("Corrector value", labelpad=9)
    fig.legend(handles=[Patch(facecolor=KINDS[kind][0](0.7), label=KIND_LABELS[kind]) for kind in KINDS]
               + [Line2D([], [], color="0.15", label="Exact scaled corrector (wireframe)"),
                  Patch(facecolor="#f6b558", label="Core K on base plane"),
                  Line2D([], [], color="0.55", label="Coarse-element edges on base plane")],
               loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output, "04_local_correctors_comparison")

    fig = plt.figure(figsize=(14, 12), layout="constrained")
    fig.suptitle("Exact and MsFEM solutions\n" + title)
    fields = [(r"Exact $u^\varepsilon$", first, exact_fields(first["xy"], settings["epsilon"])[2])]
    fields += [("MsFEM " + KIND_LABELS[kind], d, d["solution"]) for kind, d in data.items()]
    bound = max(float(np.max(np.abs(values))) for _, _, values in fields)
    norm = Normalize(-bound, bound)
    for index, (label, d, values) in enumerate(fields, start=1):
        ax = _axes3d(fig, 2, 2, index, label)
        artist = ax.plot_trisurf(*d["xy"].T, values, triangles=d["triangles"],
                                cmap="coolwarm", norm=norm, linewidth=0, antialiased=False)
        ax.set_zlim(-1.1 * bound, 1.1 * bound)
        ax.set_zlabel("Solution value", labelpad=8)
    fig.colorbar(artist, ax=fig.axes, shrink=0.55, label="Solution value", pad=0.07)
    _save(fig, output, "05_exact_vs_msfem")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-size", type=int, default=16)
    parser.add_argument("--fine-size", type=int, default=128, help="N_h, so h=1/N_h")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=0.025)
    parser.add_argument("--num-time-steps", type=int, default=40)
    parser.add_argument("--assembly-type", choices=["petrov_galerkin", "galerkin"], default="petrov_galerkin")
    parser.add_argument("--coarse-cell", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/msfem/periodic-fields"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plot-only", action="store_true", help="Redraw saved fields without importing DOLFINx")
    group.add_argument("--analytic-only", action="store_true", help="Draw the coefficient and exact correctors only")
    args = parser.parse_args()
    ratio = args.fine_size // args.coarse_size if args.coarse_size > 0 else 0
    if args.coarse_size < 3 or args.fine_size % args.coarse_size or ratio < 1 or ratio & (ratio - 1):
        parser.error("Need N_H>=3 and N_h/N_H a positive power of two")
    if not all(np.isfinite(v) and v > 0 for v in (args.epsilon, args.final_time)):
        parser.error("epsilon and final-time must be finite and positive")
    if not np.isfinite(args.delta) or args.delta < 0 or args.num_time_steps < 1:
        parser.error("Need delta>=0 and num-time-steps>=1")
    settings = dict(N_H=args.coarse_size, N_h=args.fine_size, epsilon=args.epsilon,
                    delta=args.delta, T=args.final_time, N_t=args.num_time_steps,
                    assembly_type=args.assembly_type, coarse_cell=args.coarse_cell)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_analytic(settings, args.output)
    if not args.analytic_only:
        if not args.plot_only:
            solve_fields(settings, args.output)
        plot_comparisons(settings, args.output)


if __name__ == "__main__":
    main()
