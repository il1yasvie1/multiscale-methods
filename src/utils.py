from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
import numpy as np
import pyvista
import dolfinx


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


def plot_function(u, filename):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data[u.name] = u.x.array
    u_grid.set_active_scalars(u.name)
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid,
                       show_edges=False,
                       edge_color="gray",
                       line_width=0.1)
    u_plotter.view_xy()
    if pyvista.OFF_SCREEN:
        u_plotter.screenshot(filename)
    else:
        u_plotter.show()


def plot_function_warped(u, filename):
    u_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    u_grid.point_data[u.name] = u.x.array
    u_grid.set_active_scalars(u.name)
    warped = u_grid.warp_by_scalar()
    plotter = pyvista.Plotter()
    plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    if pyvista.OFF_SCREEN:
        plotter.screenshot(filename)
    else:
        plotter.show()


def plot_oversampling(coarse_mesh, fine_mesh, patch_mesh, filename):
    plotter = pyvista.Plotter()
    tdim = coarse_mesh.topology.dim

    fine_mesh.topology.create_connectivity(tdim, tdim)
    fine_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(fine_mesh, tdim))
    plotter.add_mesh(fine_grid,
            show_edges=True,
            color="w",
            edge_color="gray",
            line_width=1,
            label="Fine mesh")

    coarse_mesh.topology.create_connectivity(tdim, tdim)
    coarse_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(coarse_mesh, tdim))
    plotter.add_mesh(coarse_grid,
            show_edges=True,
            color="w",
            edge_color="red",
            line_width=3,
            opacity=0.5,
            label="Coarse mesh")

    patch_mesh.topology.create_connectivity(tdim, tdim)
    local_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(patch_mesh, tdim))
    plotter.add_mesh(local_grid,
            show_edges=True,
            color="lightskyblue",
            edge_color="blue",
            line_width=2,
            opacity=1,
            label="Local mesh")

    plotter.view_xy()
    plotter.show_axes()
    if pyvista.OFF_SCREEN:
        plotter.screenshot(filename)
    else:
        plotter.show()
