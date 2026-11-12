"""Plot the actual DOLFINx patch returned by src.oversampling.oversampling."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "assets" / "oversampling"

def mesh_arrays(domain):
    """Extract the linear triangular geometry for Matplotlib."""
    return (domain.geometry.x[:, :2].copy(),
            np.asarray(domain.geometry.dofmaps[0]).copy())


def mesh_edges(points, cells, *, boundary_only=False):
    edges = np.concatenate((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return points[edges[counts == 1] if boundary_only else edges]


def oversampling_geometry(N_H, N_h, delta, coarse_cell=None):
    if N_H < 1 or N_h < N_H or N_h % N_H:
        raise ValueError("N_h must be a positive integer multiple of N_H")
    ratio = N_h // N_H
    if ratio & (ratio - 1):
        raise ValueError("N_h/N_H must be a power of two")
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    from dolfinx import mesh
    from mpi4py import MPI
    from src.oversampling import oversampling
    from src.two_level_mesh import refine_with_meshtags

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Run the figure generator in a single process")
    coarse_mesh = mesh.create_unit_square(MPI.COMM_SELF, N_H, N_H)
    fine_mesh, _ = refine_with_meshtags(coarse_mesh, ratio.bit_length() - 1)
    cp, cc = mesh_arrays(coarse_mesh)
    fp, fc = mesh_arrays(fine_mesh)
    if coarse_cell is None:
        i = max(0, N_H // 2 - 1)
        target = np.array([i + 2 / 3, i + 1 / 3]) / N_H
        coarse_cell = int(np.argmin(np.sum((cp[cc].mean(axis=1) - target) ** 2, axis=1)))
    if not 0 <= coarse_cell < len(cc):
        raise ValueError("coarse_cell is outside the DOLFINx coarse mesh")
    vertices = cp[cc[coarse_cell]]
    patch_mesh, _ = oversampling(coarse_cell, delta, coarse_mesh, fine_mesh)
    pp, pc = mesh_arrays(patch_mesh)
    if not len(pc):
        raise ValueError("Oversampling returned an empty patch")
    return cp, cc, fp, fc, vertices, pp, pc


def plot_oversampling(N_H=8, N_h=64, delta=0.5, coarse_cell=None,
                      output=DEFAULT_OUTPUT):
    cp, cc, fp, fc, vertices, pp, pc = oversampling_geometry(N_H, N_h, delta, coarse_cell)
    H, h = 1 / N_H, 1 / N_h
    fine_edges, coarse_edges = mesh_edges(fp, fc), mesh_edges(cp, cc)
    patch_edges = mesh_edges(pp, pc, boundary_only=True)
    colors = dict(fine="#b8bec6", coarse="#45536b", patch="#cce6f3",
                  patch_edge="#12658c", core="#f5b45b", core_edge="#9f4f0a")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8))
    for ax in axes:
        ax.add_collection(PolyCollection(pp[pc], facecolors=colors["patch"],
                                         edgecolors="none", zorder=1))
        ax.add_collection(LineCollection(fine_edges, colors=colors["fine"], linewidths=0.28, zorder=2))
        ax.add_collection(LineCollection(coarse_edges, colors=colors["coarse"], linewidths=0.9, zorder=3))
        ax.add_patch(Polygon(vertices, facecolor=colors["core"], edgecolor=colors["core_edge"],
                             alpha=0.88, linewidth=2.1, zorder=4))
        ax.add_collection(LineCollection(patch_edges, colors=colors["patch_edge"], linewidths=2, zorder=5))
        ax.text(*vertices.mean(axis=0), "$K$", ha="center", va="center", fontsize=16,
                color=colors["core_edge"], zorder=6)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.tick_params(labelsize=9)

    axes[0].set(xlim=(0, 1), ylim=(0, 1), title=r"(a) $\Omega=(0,1)^2$")
    axes[0].set_xticks(np.linspace(0, 1, 5))
    axes[0].set_yticks(np.linspace(0, 1, 5))
    x0, y0 = vertices.min(axis=0)
    # The vertical leg is the right-hand leg for the default coarse element.
    leg_x, counts = np.unique(vertices[:, 0], return_counts=True)
    leg_x = leg_x[counts == 2].item()
    leg_y = vertices[np.isclose(vertices[:, 0], leg_x), 1].mean()
    patch_x = pp[:, 0].max() if leg_x > vertices[:, 0].mean() else pp[:, 0].min()
    gap = abs(patch_x - leg_x)
    gap_label = r"$\delta H$" if np.isclose(gap, delta * H) else rf"$d_x={gap:.4g}$"
    detail_points = np.vstack((vertices, [patch_x, leg_y],
                               [x0, y0 - 0.2 * H], [x0 + H, y0 + 1.4 * H]))
    center = (detail_points.min(axis=0) + detail_points.max(axis=0)) / 2
    half_width = np.ptp(detail_points, axis=0).max() / 2 + 0.18 * H
    axes[1].set(xlim=(center[0] - half_width, center[0] + half_width),
                ylim=(center[1] - half_width, center[1] + half_width),
                title=r"(b) $K$ and $S^\delta_K$")
    dimensions = [
        (x0, x0 + H, y0 - 0.16 * H, rf"$H=1/{N_H}$"),
        (x0 + 0.85 * H, x0 + 0.85 * H + h, y0 + 1.23 * H, rf"$h=1/{N_h}$"),
    ]
    if gap > 1e-12:
        dimensions.append((leg_x, patch_x, leg_y, gap_label))
    for left, right, y, label in dimensions:
        axes[1].annotate("", (left, y), (right, y),
                         arrowprops=dict(arrowstyle="<->", color="#1b263b"), zorder=7)
        axes[1].text((left + right) / 2, y + 0.035 * H, label, ha="center", va="bottom",
                     fontsize=11, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9), zorder=8)

    fig.suptitle(rf"Barycentric oversampling: $H=1/{N_H}$, $h=1/{N_h}$, $\delta={delta:g}$",
                 fontsize=16, y=0.97)
    handles = [
        Patch(facecolor=colors["core"], edgecolor=colors["core_edge"], label=r"Coarse element $K$"),
        Patch(facecolor=colors["patch"], edgecolor=colors["patch_edge"], label=r"Oversampling domain $S^\delta_K$"),
        Line2D([], [], color=colors["coarse"], lw=1.2, label=rf"Coarse Mesh $\mathcal{{T}}^H$ ($N_H={N_H}$)"),
        Line2D([], [], color=colors["fine"], lw=0.8, label=rf"Fine Mesh $\mathcal{{T}}^h$ ($N_h={N_h}$)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, fontsize=9, frameon=False, columnspacing=1.8)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.24, top=0.85, wspace=0.23)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=220, facecolor="white")
    plt.close(fig)
    return output.with_suffix(".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-size", type=int, default=8)
    parser.add_argument("--fine-size", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--coarse-cell", type=int, help="DOLFINx coarse cell index; default: an interior cell")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(plot_oversampling(args.coarse_size, args.fine_size, args.delta,
                            args.coarse_cell, args.output))
