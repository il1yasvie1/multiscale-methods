"""Diagnostics evaluated while the mesh and local correctors are available."""

import math
from time import perf_counter

from dolfinx import fem
import numpy as np
import ufl

from src.utils import error_norm


def solution_errors(solution: fem.Function, exact_solution) -> dict[str, float]:
    return {
        "l2_error": error_norm(solution, exact_solution, "L2"),
        "h1_error": error_norm(solution, exact_solution, "H1"),
        "h1_semi_error": error_norm(solution, exact_solution, "H1-semi"),
    }


def jump_norm(solution: fem.Function, H: float) -> float:
    """H^(-1/2)-weighted jump on interior fine facets (zero inside each K)."""
    domain = solution.function_space.mesh
    squared = fem.assemble_scalar(fem.form(
        (1 / H) * ufl.inner(ufl.jump(solution), ufl.jump(solution))
        * ufl.dS(domain=domain)
    ))
    return math.sqrt(max(0.0, float(np.real(squared))))


class CorrectorError:
    """Accumulate directional L2 and H1-seminorm errors over coarse cells."""

    def __init__(self, problem):
        self.problem = problem
        self.l2_squared = 0.0
        self.h1_semi_squared = 0.0
        self.seconds = 0.0

    def __call__(self, patch_mesh, coarse_cell_patch_cells, correctors) -> None:
        from dolfinx import mesh

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
