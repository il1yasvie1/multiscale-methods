from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
import numpy as np


def boundary_dofs(V):
    tdim = V.mesh.topology.dim
    fdim = tdim - 1
    V.mesh.topology.create_connectivity(fdim, tdim)
    exterior_facets = mesh.exterior_facet_indices(V.mesh.topology)
    return fem.locate_dofs_topological(V, fdim, exterior_facets)

def error_norm(
    uh: fem.Function,
    u,
    norm_type: str,
) -> float:

    domain = uh.function_space.mesh
    error = uh - u(domain)
    l2_term = ufl.inner(error, error)
    if norm_type == "L2":
        integrand = l2_term
    elif norm_type == "H1-semi":
        integrand = ufl.inner(ufl.grad(error), ufl.grad(error))
    elif norm_type == "H1":
        integrand = l2_term + ufl.inner(ufl.grad(error), ufl.grad(error))
    else:
        raise ValueError("norm_type must be 'L2', 'H1', or 'H1-semi'")

    dx = ufl.dx(domain=domain, metadata={"quadrature_degree": 12})
    local_squared = fem.assemble_scalar(fem.form(integrand * dx))
    global_squared = domain.comm.allreduce(local_squared, op=MPI.SUM)
    return np.sqrt(global_squared)
