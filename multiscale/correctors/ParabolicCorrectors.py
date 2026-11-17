from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from petsc4py import PETSc
import ufl

from multiscale.fem_utils import zero_dirichlet_bc
from .LocalCorrectors import LocalCorrectors


class ParabolicCorrectors(LocalCorrectors):
    def solve(self) -> list[fem.Function]:
        V = self.function_space
        A = self.coefficient
        wj_trial, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(
            domain=self.mesh,
            metadata={"quadrature_degree": self.quadrature_degree},
        )
        mass_form = fem.form(ufl.inner(wj_trial, v) * dx)
        stiffness = ufl.inner(A * ufl.grad(wj_trial), ufl.grad(v)) * dx
        num_time_steps = self.corrector_parameters["num_time_steps"]
        time_step = self.corrector_parameters["final_time"] / num_time_steps
        step_form = fem.form(ufl.inner(wj_trial, v) * dx + time_step * stiffness)
        bcs = [zero_dirichlet_bc(V)]

        unknown = self.solver_parameters.keys() - {"initial", "step"}
        if unknown:
            raise ValueError(f"Unknown parabolic solver groups: {sorted(unknown)}")
        if any(not isinstance(options, dict) for options in self.solver_parameters.values()):
            raise ValueError("Parabolic initial and step solver parameters must be mappings")
        default_options = {"ksp_type": "preonly", "pc_type": "lu"}

        def create_solver(matrix: PETSc.Mat, prefix: str, group: str) -> PETSc.KSP:
            petsc_options = {**default_options, **self.solver_parameters.get(group, {})}
            solver = PETSc.KSP().create(matrix.comm)
            solver.setOptionsPrefix(prefix)
            solver.setOperators(matrix)
            options = PETSc.Options()
            for key, value in petsc_options.items():
                options[f"{prefix}{key}"] = value
            solver.setFromOptions()
            for key in petsc_options:
                del options[f"{prefix}{key}"]
            solver.setErrorIfNotConverged(True)
            solver.setUp()
            return solver

        mass = fem_petsc.assemble_matrix(mass_form)
        mass.assemble()
        constrained_mass = fem_petsc.assemble_matrix(mass_form, bcs=bcs)
        constrained_mass.assemble()
        step_matrix = fem_petsc.assemble_matrix(step_form, bcs=bcs)
        step_matrix.assemble()
        initial_solver = create_solver(constrained_mass, "parabolic_initial_", "initial")
        step_solver = create_solver(step_matrix, "parabolic_step_", "step")
        step_rhs = mass.createVecLeft()
        correctors = []
        for j in range(self.gdim):
            ej = ufl.as_vector([int(i == j) for i in range(self.gdim)])
            initial_rhs = fem_petsc.assemble_vector(
                fem.form(-ufl.inner(A * ej, ufl.grad(v)) * dx)
            )
            initial_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                    mode=PETSc.ScatterMode.REVERSE)
            fem_petsc.set_bc(initial_rhs, bcs)
            wj = fem.Function(V)
            wj_next = fem.Function(V)
            initial_solver.solve(initial_rhs, wj.x.petsc_vec)
            wj.x.scatter_forward()

            uj = fem.Function(V)
            uj.x.array[:] = 0.5 * time_step * wj.x.array
            for step in range(num_time_steps):
                mass.mult(wj.x.petsc_vec, step_rhs)
                fem_petsc.set_bc(step_rhs, bcs)
                step_solver.solve(step_rhs, wj_next.x.petsc_vec)
                wj_next.x.scatter_forward()
                weight = 0.5 if step == num_time_steps - 1 else 1.0
                uj.x.array[:] += weight * time_step * wj_next.x.array
                wj, wj_next = wj_next, wj

            uj.x.scatter_forward()
            correctors.append(uj)
            initial_rhs.destroy()

        step_rhs.destroy()
        initial_solver.destroy()
        step_solver.destroy()
        mass.destroy()
        constrained_mass.destroy()
        step_matrix.destroy()

        return correctors
