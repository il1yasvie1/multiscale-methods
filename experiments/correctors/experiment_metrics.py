"""Corrector and effective-tensor errors on the exact inner square."""

import math

from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import ufl


def inner_measure(domain, L: float):
    tdim = domain.topology.dim
    owned = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
    midpoints = mesh.compute_midpoints(domain, tdim, owned)
    selected = owned[np.all(np.abs(midpoints[:, :2]) < L / 2 + 1e-12, axis=1)]
    tags = mesh.meshtags(domain, tdim, selected, np.ones(len(selected), dtype=np.int32))
    return ufl.Measure("dx", domain=domain, subdomain_data=tags)(
        1, metadata={"quadrature_degree": 12}
    )


def upscaling_error(correctors, coefficient, dx_inner,
                    exact: np.ndarray) -> dict[str, float | str]:
    """Uniform flux average over K_L for each corrector method."""
    domain = dx_inner.ufl_domain().ufl_cargo()
    local_area = fem.assemble_scalar(fem.form(1.0 * dx_inner))
    inner_area = domain.comm.allreduce(float(np.real(local_area)), op=MPI.SUM)
    if inner_area <= 0:
        raise ValueError("The mesh contains no cells in K_L")
    tensor = np.empty((2, 2))
    for j, corrector in enumerate(correctors):
        ej = ufl.as_vector([int(i == j) for i in range(2)])
        flux = coefficient * (ej + ufl.grad(corrector))
        for i in range(2):
            local = fem.assemble_scalar(fem.form(flux[i] * dx_inner))
            tensor[i, j] = domain.comm.allreduce(float(np.real(local)), op=MPI.SUM) / inner_area
    error = float(np.linalg.norm(tensor - exact, ord="fro"))
    return {"upscaling_domain": "K_L", "a0_error_frobenius": error,
            **{f"a0_{i + 1}{j + 1}": float(tensor[i, j])
               for i in range(2) for j in range(2)}}


def compare(correctors, references, dx) -> dict[str, float]:
    if len(correctors) != 2 or len(references) != 2:
        raise ValueError("Expected both corrector directions")
    domain = dx.ufl_domain().ufl_cargo()
    l2_form = 0
    semi_form = 0
    for computed, reference in zip(correctors, references):
        difference = reference - computed
        l2_form += ufl.inner(difference, difference) * dx
        semi_form += ufl.inner(ufl.grad(difference), ufl.grad(difference)) * dx
    l2_sq = domain.comm.allreduce(float(np.real(fem.assemble_scalar(fem.form(l2_form)))), op=MPI.SUM)
    semi_sq = domain.comm.allreduce(float(np.real(fem.assemble_scalar(fem.form(semi_form)))), op=MPI.SUM)
    l2 = math.sqrt(max(0.0, l2_sq))
    semi = math.sqrt(max(0.0, semi_sq))
    return {"l2": l2, "h1_semi": semi, "h1": math.hypot(l2, semi)}
