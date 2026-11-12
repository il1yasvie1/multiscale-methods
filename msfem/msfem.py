from dataclasses import dataclass
from typing import Callable, Literal

import dolfinx
from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc

from src.elliptic_problem_data import EllipticProblemData
from src.oversampling import oversampling
from src.two_level_mesh import refine_with_meshtags
from src.local_correctors import (
    get_local_elliptic_correctors,
    get_local_modified_correctors,
    get_local_parabolic_correctors,
)
from src.local_basis import get_local_affine_basis_extensions, get_local_multiscale_basis
from src.utils import boundary_dofs


@dataclass
class CellBasisData:
    fine_dofs: np.ndarray
    basis_values: np.ndarray
    corrector_values: np.ndarray


class MsFEM_Oversampling:
    def __init__(
        self,
        problem: EllipticProblemData,
        coarse_mesh: mesh.Mesh,
        num_refinements: int):

        self.problem = problem
        self.comm = MPI.COMM_WORLD
        self.coarse_mesh = coarse_mesh
        if coarse_mesh.comm.size != 1:
            raise ValueError("Each MPI rank must create its own coarse mesh on COMM_SELF")
        self.tdim = coarse_mesh.topology.dim
        self.fine_mesh, self.coarse_to_fine_meshtag = refine_with_meshtags(coarse_mesh, num_refinements)
        self.V_coarse = fem.functionspace(coarse_mesh, ("P", 1))
        self.V_multiscale = fem.functionspace(self.fine_mesh, ("DP", 1))
        self.num_coarse_cells = coarse_mesh.topology.index_map(self.tdim).size_local
        self.num_coarse_dofs = self.V_coarse.dofmap.index_map.size_local
        self.cell_basis_data: dict[int, CellBasisData] = {}
        self.A_H: PETSc.Mat | None = None
        self.b_H: PETSc.Vec | None = None

    def _assemble_locally(
        self,
        patch_mesh: mesh.Mesh,
        coarse_cell_on_patch_list: np.ndarray,
        test_basis: list[fem.Function],
        trial_basis: list[fem.Function],
        quadrature_degree: int = 12,
    ) -> tuple[np.ndarray, np.ndarray]:
        cells = np.unique(np.asarray(coarse_cell_on_patch_list, dtype=np.int32))
        coarse_cell_meshtag = mesh.meshtags(
            patch_mesh,
            patch_mesh.topology.dim,
            cells,
            np.ones(cells.size, dtype=np.int32),
        )
        dx_coarse_cell = ufl.Measure(
            "dx", domain=patch_mesh, subdomain_data=coarse_cell_meshtag
        )(1, metadata={"quadrature_degree": quadrature_degree})
        coefficient = self.problem.coefficient(patch_mesh)
        source = self.problem.source(patch_mesh)

        local_lhs = np.empty(
            (len(test_basis), len(trial_basis)), dtype=dolfinx.default_scalar_type
        )
        local_rhs = np.empty(len(test_basis), dtype=dolfinx.default_scalar_type)
        for i, phi in enumerate(test_basis):
            local_rhs[i] = fem.assemble_scalar(fem.form(source * phi * dx_coarse_cell))
            for j, psi in enumerate(trial_basis):
                local_lhs[i, j] = fem.assemble_scalar(
                    fem.form(
                        ufl.inner(coefficient * ufl.grad(psi), ufl.grad(phi))
                        * dx_coarse_cell
                    )
                )
        return local_lhs, local_rhs

    def _tabulate_cell_multiscale_basis(
        self,
        coarse_cell_fine_cells: np.ndarray,
        coarse_cell_patch_cells: np.ndarray,
        multiscale_basis: list[fem.Function],
        correctors: list[fem.Function],
    ) -> CellBasisData:
        fine_cells = np.asarray(coarse_cell_fine_cells, dtype=np.int32)
        patch_cells = np.asarray(coarse_cell_patch_cells, dtype=np.int32)

        fine_dofs = np.concatenate([
            self.V_multiscale.dofmap.cell_dofs(int(cell)) for cell in fine_cells
        ]).astype(np.int32)
        values = np.empty(
            (fine_dofs.size, len(multiscale_basis) + len(correctors)),
            dtype=dolfinx.default_scalar_type,
        )
        for j, psi in enumerate([*multiscale_basis, *correctors]):
            psi_fine = fem.Function(self.V_multiscale)
            psi_fine.interpolate(psi, patch_cells, fine_cells)
            values[:, j] = psi_fine.x.array[fine_dofs]

        return CellBasisData(
            fine_dofs=fine_dofs,
            basis_values=values[:, :len(multiscale_basis)].copy(),
            corrector_values=values[:, len(multiscale_basis):].copy(),
        )

    def offline(
        self,
        delta: float,
        *,
        corrector_kind: Literal["elliptic", "parabolic", "modified"] = "elliptic",
        final_time: float | None = None,
        num_time_steps: int | None = None,
        assembly_type: str = "petrov_galerkin",
        corrector_observer: Callable | None = None,
    ) -> None:
        if corrector_kind not in ("elliptic", "parabolic", "modified"):
            raise ValueError("corrector_kind must be 'elliptic', 'parabolic' or 'modified'")
        if corrector_kind == "parabolic" and (final_time is None or num_time_steps is None):
            raise ValueError("Parabolic correctors require final_time and num_time_steps")
        if corrector_kind == "modified":
            if final_time is None or not np.isfinite(final_time) or final_time <= 0:
                raise ValueError("Modified correctors require a finite positive final_time")
            if num_time_steps is not None:
                raise ValueError("num_time_steps is only used by parabolic correctors")
        if assembly_type not in ("petrov_galerkin", "galerkin"):
            raise ValueError("assembly_type must be 'petrov_galerkin' or 'galerkin'")

        self.cell_basis_data.clear()
        local_contributions = []
        local_error = None
        try:
            for coarse_cell_index in range(self.comm.rank, self.num_coarse_cells, self.comm.size):
                patch_mesh, patch_to_fine_entity_map = oversampling(coarse_cell_index, delta, self.coarse_mesh, self.fine_mesh)
                V_patch = fem.functionspace(patch_mesh, ("P", 1))

                if corrector_kind == "elliptic":
                    correctors = get_local_elliptic_correctors(V_patch, self.problem)
                elif corrector_kind == "modified":
                    correctors = get_local_modified_correctors(V_patch, self.problem, final_time)
                else:
                    correctors = get_local_parabolic_correctors(
                        V_patch, self.problem,
                        final_time=final_time,
                        num_time_steps=num_time_steps,
                    )
                affine_basis, basis_gradients = get_local_affine_basis_extensions(self.V_coarse, V_patch, coarse_cell_index)
                multiscale_basis = get_local_multiscale_basis(affine_basis, basis_gradients, correctors)

                coarse_cell_fine_cells = self.coarse_to_fine_meshtag.find(coarse_cell_index)
                coarse_cell_patch_cells = patch_to_fine_entity_map.sub_topology_to_topology(
                    coarse_cell_fine_cells, inverse=True
                )

                if corrector_observer is not None:
                    corrector_observer(patch_mesh, coarse_cell_patch_cells, correctors)

                local_lhs, local_rhs = self._assemble_locally(
                    patch_mesh,
                    coarse_cell_patch_cells,
                    affine_basis if assembly_type == "petrov_galerkin" else multiscale_basis,
                    multiscale_basis,
                )
                coarse_dofs = self.V_coarse.dofmap.cell_dofs(coarse_cell_index)
                local_contributions.append((coarse_dofs, local_lhs, local_rhs))

                self.cell_basis_data[coarse_cell_index] = (
                    self._tabulate_cell_multiscale_basis(
                        coarse_cell_fine_cells,
                        coarse_cell_patch_cells,
                        multiscale_basis,
                        correctors,
                    )
                )
                del multiscale_basis, affine_basis, correctors, V_patch
                del patch_mesh, patch_to_fine_entity_map
        except Exception as error:
            local_error = f"rank {self.comm.rank}: {type(error).__name__}: {error}"

        errors = self.comm.allgather(local_error)
        if any(errors):
            raise RuntimeError("Offline patch failure: " + "; ".join(e for e in errors if e))

        contributions = self.comm.gather(local_contributions, root=0)
        assembly_error = None
        if self.comm.rank == 0:
            try:
                A_H = PETSc.Mat().createAIJ(
                    (self.num_coarse_dofs, self.num_coarse_dofs),
                    nnz=min(12, self.num_coarse_dofs),
                    comm=PETSc.COMM_SELF,
                )
                A_H.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                b_H = A_H.createVecRight()
                b_H.set(0)
                for rank_contributions in contributions:
                    for coarse_dofs, local_lhs, local_rhs in rank_contributions:
                        A_H.setValues(coarse_dofs, coarse_dofs, local_lhs, addv=PETSc.InsertMode.ADD_VALUES)
                        b_H.setValues(coarse_dofs, local_rhs, addv=PETSc.InsertMode.ADD_VALUES)
                A_H.assemble()
                b_H.assemble()
                coarse_boundary_dofs = boundary_dofs(self.V_coarse)
                A_H.zeroRowsColumns(coarse_boundary_dofs, diag=1.0)
                b_H.setValues(
                    coarse_boundary_dofs,
                    np.zeros(len(coarse_boundary_dofs), dtype=PETSc.ScalarType),
                )
                b_H.assemble()
                self.A_H, self.b_H = A_H, b_H
            except Exception as error:
                assembly_error = f"{type(error).__name__}: {error}"
        assembly_error = self.comm.bcast(assembly_error, root=0)
        if assembly_error:
            raise RuntimeError("Coarse assembly failed: " + assembly_error)

    def _reconstruct_multiscale_solution(self, u_H: fem.Function
    )-> fem.Function:
        u_ms = fem.Function(self.V_multiscale, name="Multiscale_Solution")
        for coarse_cell_index, data in self.cell_basis_data.items():
            coarse_dofs = self.V_coarse.dofmap.cell_dofs(coarse_cell_index)
            u_ms.x.array[data.fine_dofs] = (
                data.basis_values @ u_H.x.array[coarse_dofs]
            )

        # Every rank built the same COMM_SELF fine mesh, so DG dof numbers agree.
        # Each coarse cell is computed once, making its fine-cell dofs disjoint.
        self.comm.Allreduce(MPI.IN_PLACE, u_ms.x.array, op=MPI.SUM)
        u_ms.x.scatter_forward()
        return u_ms

    def online(self) -> fem.Function:
        coarse_values, solve_error = None, None
        if self.comm.rank == 0:
            try:
                solver = PETSc.KSP().create(self.A_H.comm)
                coarse_solution = self.A_H.createVecRight()
                try:
                    solver.setOptionsPrefix("coarse_problem_")
                    solver.setOperators(self.A_H)
                    solver.setType(PETSc.KSP.Type.PREONLY)
                    solver.getPC().setType(PETSc.PC.Type.LU)
                    solver.setErrorIfNotConverged(True)
                    solver.solve(self.b_H, coarse_solution)
                    coarse_values = coarse_solution.getArray(readonly=True).copy()
                finally:
                    coarse_solution.destroy()
                    solver.destroy()
            except Exception as error:
                solve_error = f"{type(error).__name__}: {error}"
        coarse_values, solve_error = self.comm.bcast((coarse_values, solve_error), root=0)
        if solve_error:
            raise RuntimeError("Coarse solve failed: " + solve_error)
        u_H = fem.Function(self.V_coarse)
        u_H.x.array[:] = coarse_values
        return self._reconstruct_multiscale_solution(u_H)
