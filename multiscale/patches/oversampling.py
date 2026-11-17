import numpy as np
from dolfinx import mesh


def oversampling(
        coarse_cell:int,
        delta:float,
        coarse_mesh: mesh.Mesh,
        fine_mesh: mesh.Mesh,
) -> tuple[mesh.Mesh, mesh.EntityMap]:

    v0, v1, v2 = coarse_mesh.geometry.x[
        coarse_mesh.geometry.dofmaps[0][coarse_cell], :2]
    B = np.linalg.inv(np.column_stack((v1 - v0, v2 - v0)))

    def marker(x):
        lam1, lam2 = B @ (x[:2] - v0[:, None])
        lam0 = 1.0 - lam1 - lam2
        tol = 1e-12
        return ((lam0 >= -delta - tol)
                & (lam1 >= -delta - tol)
                & (lam2 >= -delta - tol))

    marked_indices = mesh.locate_entities(
        fine_mesh, fine_mesh.topology.dim, marker)

    patch_mesh, patch_to_fine_entity_map, _, _ = mesh.create_submesh(
        fine_mesh, fine_mesh.topology.dim, marked_indices)

    return patch_mesh, patch_to_fine_entity_map
