"""L2, H1-seminorm, and H1 corrector errors on the exact inner square."""

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
