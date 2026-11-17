from dolfinx import fem
import numpy as np


def get_local_affine_basis_extensions(
    V_coarse: fem.FunctionSpace,
    V_patch: fem.FunctionSpace,
    coarse_cell_index: int,
) -> tuple[list[fem.Function], np.ndarray]:

    coarse_dofs = V_coarse.dofmap.cell_dofs(coarse_cell_index)
    coarse_points = V_coarse.tabulate_dof_coordinates()[coarse_dofs]
    coarse_matrix = np.array([
        [1, p[0], p[1]] for p in coarse_points
    ])

    basis_coefficients = np.linalg.inv(coarse_matrix).T

    patch_points = V_patch.tabulate_dof_coordinates()
    affine_basis = []
    for i in range(3):
        a, b, c = basis_coefficients[i]
        phi = fem.Function(V_patch)
        phi.x.array[:] = (
            a + b * patch_points[:, 0] + c * patch_points[:, 1]
        )
        phi.x.scatter_forward()
        affine_basis.append(phi)

    return affine_basis, basis_coefficients[:, 1:]


def get_local_multiscale_basis(
    affine_basis: list[fem.Function],
    basis_gradients: np.ndarray,
    correctors: list[fem.Function]
) -> list[fem.Function]:

    corrector_values = np.column_stack([corrector.x.array for corrector in correctors])
    multiscale_basis = []
    for i, phi in enumerate(affine_basis):
        psi = fem.Function(phi.function_space)
        psi.x.array[:] = phi.x.array + corrector_values @ basis_gradients[i]
        psi.x.scatter_forward()
        multiscale_basis.append(psi)
    return multiscale_basis
