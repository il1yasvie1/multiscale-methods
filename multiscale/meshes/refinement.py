import numpy as np
from dolfinx import mesh


def refine_with_meshtags(
        coarse_mesh:mesh.Mesh, num_refinements:int
) -> tuple[mesh.Mesh, mesh.MeshTags]:

    tdim = coarse_mesh.topology.dim
    cell_map = coarse_mesh.topology.index_map(tdim)
    coarse_cells = np.arange(cell_map.size_local, dtype=np.int32)
    coarse_to_fine_meshtag = mesh.meshtags(
        coarse_mesh,
        tdim,
        coarse_cells,
        cell_map.local_to_global(coarse_cells).astype(np.int32),
    )

    fine_mesh = coarse_mesh
    for _ in range(num_refinements):
        fine_mesh.topology.create_entities(1)
        fine_mesh, parent_cells, _ = mesh.refine(
            fine_mesh,
            partitioner=None,
            option=mesh.RefinementOption.parent_cell,
        )
        coarse_to_fine_meshtag = mesh.transfer_meshtag(coarse_to_fine_meshtag, fine_mesh, parent_cells)
    return fine_mesh, coarse_to_fine_meshtag
