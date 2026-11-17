from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from multiscale.fem_utils import boundary_dofs
from .LocalCorrectors import LocalCorrectors


class ModifiedEllipticCorrectors(LocalCorrectors):
    def solve(self) -> list[fem.Function]:
        from slepc4py import SLEPc
        mass_options = self.solver_parameters.get("mass", {})
        mass_petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
        mass_petsc_options.update({key: value for key, value in mass_options.items()
                                   if key not in ("ksp_view", "ksp_monitor")})
        mfn_options = self.solver_parameters.get("mfn", {})
        mass_lumped = self.solver_parameters.get("mass_lumped", False)
        show_mass_view = "ksp_view" in mass_options and mass_options["ksp_view"] is not False
        monitor_mass = "ksp_monitor" in mass_options and mass_options["ksp_monitor"] is not False

        V = self.function_space
        A = self.coefficient
        correctors = [fem.Function(V) for _ in range(self.gdim)]
        index_map = V.dofmap.index_map
        free = np.setdiff1d(
            np.arange(index_map.size_local, dtype=np.int32), boundary_dofs(V)
        )
        free_global = index_map.local_to_global(free).astype(PETSc.IntType)

        uj, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(
            domain=self.mesh,
            metadata={"quadrature_degree": self.quadrature_degree},
        )

        def interior_matrix(form):
            matrix = fem_petsc.assemble_matrix(fem.form(form))
            matrix.assemble()
            interior = matrix.createSubMatrix(free_is, free_is)
            return interior

        comm = self.mesh.comm
        free_is = PETSc.IS().createGeneral(free_global, comm=comm)
        stiffness = interior_matrix(
            ufl.inner(A * ufl.grad(uj), ufl.grad(v)) * dx
        )

        options = PETSc.Options()
        monitor_state = {"solves": 0, "max_iterations": 0}
        if mass_lumped:
            lumped = fem_petsc.assemble_vector(fem.form(v * dx))
            lumped.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                               mode=PETSc.ScatterMode.REVERSE)
            diagonal = lumped.array_r[free].copy()
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
                raise RuntimeError("Mass-lumped diagonal must be positive and finite")
            inverse_diagonal = 1 / diagonal
            if show_mass_view or monitor_mass:
                PETSc.Sys.Print("modified mass: lumped diagonal inverse (no KSP)")
        else:
            mass = interior_matrix(ufl.inner(uj, v) * dx)
            mass_solver = PETSc.KSP().create(comm)
            mass_solver.setOperators(mass)
            mass_solver.setOptionsPrefix("corrector_mass_")
            options.prefixPush(mass_solver.getOptionsPrefix())
            for key, value in mass_petsc_options.items():
                options[key] = value
            mass_solver.setFromOptions()
            for key in mass_petsc_options:
                del options[key]
            options.prefixPop()
            mass_solver.setErrorIfNotConverged(True)
            mass_solver.setUp()
            if show_mass_view:
                mass_solver.view()
            if monitor_mass:
                def first_mass_solve_monitor(_ksp, iteration, residual):
                    if iteration == 0:
                        monitor_state["solves"] += 1
                    monitor_state["max_iterations"] = max(
                        monitor_state["max_iterations"], iteration)
                    if monitor_state["solves"] == 1:
                        PETSc.Sys.Print(
                            f"  modified mass KSP {iteration} residual {residual:.12e}")

                mass_solver.setMonitor(first_mass_solve_monitor)

        work = stiffness.createVecLeft()
        initial = stiffness.createVecRight()
        solution = stiffness.createVecRight()

        class DiffusionOperator:
            def mult(self, _mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
                stiffness.mult(x, work)
                if mass_lumped:
                    y.array[:] = inverse_diagonal * work.array_r
                else:
                    mass_solver.solve(work, y)

        operator = PETSc.Mat().createPython(stiffness.getSizes(), DiffusionOperator(), comm=comm)
        operator.setUp()

        mfn = SLEPc.MFN().create(comm)
        mfn.setOperator(operator)
        mfn.setOptionsPrefix("corrector_mfn_")
        mfn.setType(mfn_options.get("type", SLEPc.MFN.Type.KRYLOV))
        mfn.setTolerances(
            tol=mfn_options.get("tol", max(1e-12, 10 * np.finfo(PETSc.RealType).eps))
        )
        mfn_diagnostics = {key: value for key, value in mfn_options.items()
                           if key in ("mfn_view", "mfn_monitor") and value is not False}
        options.prefixPush(mfn.getOptionsPrefix())
        for key, value in mfn_diagnostics.items():
            options[key] = value
        mfn.setFromOptions()
        for key in mfn_diagnostics:
            del options[key]
        options.prefixPop()
        mfn.setErrorIfNotConverged(True)
        function = mfn.getFN()
        function.setType(SLEPc.FN.Type.PHI)
        function.setPhiIndex(1)
        function.setScale(-self.corrector_parameters["final_time"],
                          self.corrector_parameters["final_time"])

        largest_rhs_norm = 0.0
        for j, uj in enumerate(correctors):
            ej = ufl.as_vector([int(i == j) for i in range(self.gdim)])
            rhs = fem_petsc.assemble_vector(
                fem.form(-ufl.inner(A * ej, ufl.grad(v)) * dx)
            )
            rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                            mode=PETSc.ScatterMode.REVERSE)
            free_rhs = rhs.array_r[free]
            rhs_norm = np.sqrt(comm.allreduce(
                float(np.vdot(free_rhs, free_rhs).real), op=MPI.SUM))
            largest_rhs_norm = max(largest_rhs_norm, rhs_norm)
            if rhs_norm == 0 or rhs_norm < 1e-12 * largest_rhs_norm:
                continue
            if mass_lumped:
                initial.array[:] = inverse_diagonal * rhs.array_r[free]
            else:
                work.array[:] = rhs.array_r[free]
                mass_solver.solve(work, initial)
            if initial.norm() == 0.0:
                continue
            mfn.solve(initial, solution)
            uj.x.array[free] = solution.array_r
            uj.x.scatter_forward()

        if monitor_mass and not mass_lumped:
            PETSc.Sys.Print(
                f"modified mass KSP: {monitor_state['solves']} solves, "
                f"maximum {monitor_state['max_iterations']} iterations")
        return correctors
