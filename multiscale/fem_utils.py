from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
import numpy as np


def boundary_facets(domain):
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    return mesh.exterior_facet_indices(domain.topology)


def boundary_dofs(V):
    return fem.locate_dofs_topological(
        V, V.mesh.topology.dim - 1, boundary_facets(V.mesh)
    )


def zero_dirichlet_bc(V):
    return fem.dirichletbc(fem.Function(V), boundary_dofs(V))


def error_norm(
    uh: fem.Function,
    u,
    norm_type: str,
) -> float:

    domain = uh.function_space.mesh
    error = uh - u
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
