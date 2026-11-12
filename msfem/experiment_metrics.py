"""Diagnostics evaluated while the mesh and local correctors are available."""

from __future__ import annotations

import math
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dolfinx import fem


def solution_errors(solution: fem.Function, exact_solution) -> dict[str, float]:
    from src.utils import error_norm

    return {
        "l2_error": error_norm(solution, exact_solution, "L2"),
        "h1_error": error_norm(solution, exact_solution, "H1"),
        "h1_semi_error": error_norm(solution, exact_solution, "H1-semi"),
    }


def _facet_l2_squared(points, values) -> float:
    """Exact simplex-facet integral of the squared P1 trace, summed by column."""
    points, values = np.asarray(points), np.asarray(values)
    n = len(points)
    edges = points[1:] - points[0]
    measure = math.sqrt(max(0.0, float(np.linalg.det(edges @ edges.T)))) / math.factorial(n - 1)
    # The P1 face mass matrix is |F|/(n(n+1)) * (I + 11^T).
    return float(measure / (n * (n + 1)) * (
        np.sum(np.abs(values)**2) + np.sum(np.abs(values.sum(axis=0))**2)
    ))


def interface_jumps(method, solution) -> dict[str, float]:
    """Collective, unweighted jumps on internal coarse facets only.

    Every rank holds the same COMM_SELF mesh; patch data are gathered to rank 0.
    P1 traces are integrated exactly on the fine facets subdividing coarse facets.
    """
    local_correctors = {
        cell: (data.fine_dofs, data.corrector_values)
        for cell, data in method.cell_basis_data.items()
    }
    rank_data = method.comm.gather(local_correctors, root=0)
    if method.comm.rank != 0:
        return {}
    domain, V = method.fine_mesh, method.V_multiscale
    tdim = domain.topology.dim
    parent = np.full(domain.topology.index_map(tdim).size_local, -1, dtype=np.int32)
    parent[method.coarse_to_fine_meshtag.indices] = method.coarse_to_fine_meshtag.values
    if np.any(parent < 0):
        raise ValueError("Every fine cell must have a coarse parent")
    count = V.dofmap.index_map.size_local
    correctors = np.empty((count, domain.geometry.dim), dtype=solution.x.array.dtype)
    for data in rank_data:
        for fine_dofs, values in data.values():
            correctors[fine_dofs] = values
    coordinates = V.tabulate_dof_coordinates()
    domain.topology.create_connectivity(tdim - 1, tdim)
    connectivity = domain.topology.connectivity(tdim - 1, tdim)
    squared = dict(msfem=0.0, corrector=0.0)
    for facet in range(domain.topology.index_map(tdim - 1).size_local):
        cells = connectivity.links(facet)
        if len(cells) != 2 or parent[cells[0]] == parent[cells[1]]:
            continue
        left = V.dofmap.cell_dofs(int(cells[0]))
        right = V.dofmap.cell_dofs(int(cells[1]))
        matches = np.all(np.isclose(coordinates[left, None, :],
                                   coordinates[right][None, :, :],
                                   rtol=0, atol=1e-12), axis=2)
        il, ir = np.nonzero(matches)
        if len(il) != tdim:
            raise ValueError("Expected matching P1 nodes on a simplex facet")
        left, right = left[il], right[ir]
        points = coordinates[left]
        squared["msfem"] += _facet_l2_squared(points, solution.x.array[left] - solution.x.array[right])
        squared["corrector"] += _facet_l2_squared(points, correctors[left] - correctors[right])
    return {f"{name}_jump_l2": math.sqrt(max(0.0, value)) for name, value in squared.items()}


class CorrectorError:
    """Accumulate directional L2 and H1-seminorm errors over coarse cells."""

    def __init__(self, problem):
        self.problem = problem
        self.l2_squared = 0.0
        self.h1_semi_squared = 0.0
        self.seconds = 0.0

    def __call__(self, patch_mesh, coarse_cell_patch_cells, correctors) -> None:
        from dolfinx import fem, mesh
        import ufl

        started = perf_counter()
        cells = np.sort(np.asarray(coarse_cell_patch_cells, dtype=np.int32))
        tags = mesh.meshtags(
            patch_mesh, patch_mesh.topology.dim, cells,
            np.ones(len(cells), dtype=np.int32),
        )
        dx_K = ufl.Measure("dx", domain=patch_mesh, subdomain_data=tags)(
            1, metadata={"quadrature_degree": 12}
        )
        references = self.problem.periodic_cell_correctors(patch_mesh)
        if len(references) != len(correctors):
            raise ValueError("Reference correctors and computed correctors differ in count")
        for computed, reference in zip(correctors, references):
            difference = computed - reference
            self.l2_squared += float(np.real(fem.assemble_scalar(
                fem.form(ufl.inner(difference, difference) * dx_K)
            )))
            self.h1_semi_squared += float(np.real(fem.assemble_scalar(
                fem.form(ufl.inner(ufl.grad(difference), ufl.grad(difference)) * dx_K)
            )))
        self.seconds += perf_counter() - started

    def results(self) -> dict[str, float]:
        l2 = math.sqrt(max(0.0, self.l2_squared))
        semi = math.sqrt(max(0.0, self.h1_semi_squared))
        return {
            "corrector_l2_error": l2,
            "corrector_h1_error": math.hypot(l2, semi),
            "corrector_h1_semi_error": semi,
        }
