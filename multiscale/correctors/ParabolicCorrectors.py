from contextlib import ExitStack

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
            try:
                solver.setOptionsPrefix(prefix)
                solver.setOperators(matrix)
                options = PETSc.Options()
                inserted = []
                try:
                    for key, value in petsc_options.items():
                        options[f"{prefix}{key}"] = value
                        inserted.append(key)
                    solver.setFromOptions()
                finally:
                    for key in inserted:
                        del options[f"{prefix}{key}"]
                solver.setErrorIfNotConverged(True)
                solver.setUp()
                return solver
            except BaseException:
                solver.destroy()
                raise

        with ExitStack() as resources:
            mass = fem_petsc.assemble_matrix(mass_form)
            resources.callback(mass.destroy)
            mass.assemble()
            constrained_mass = fem_petsc.assemble_matrix(mass_form, bcs=bcs)
            resources.callback(constrained_mass.destroy)
            constrained_mass.assemble()
            step_matrix = fem_petsc.assemble_matrix(step_form, bcs=bcs)
            resources.callback(step_matrix.destroy)
            step_matrix.assemble()
            stiffness_matrix = fem_petsc.assemble_matrix(fem.form(stiffness), bcs=bcs)
            resources.callback(stiffness_matrix.destroy)
            stiffness_matrix.assemble()
            initial_solver = create_solver(constrained_mass, "parabolic_initial_", "initial")
            resources.callback(initial_solver.destroy)
            step_solver = create_solver(step_matrix, "parabolic_step_", "step")
            resources.callback(step_solver.destroy)
            final_solver = create_solver(stiffness_matrix, "parabolic_final_", "step")
            resources.callback(final_solver.destroy)
            step_rhs = mass.createVecLeft()
            resources.callback(step_rhs.destroy)
            correctors = []
            for j in range(self.gdim):
                with ExitStack() as direction_resources:
                    ej = ufl.as_vector([int(i == j) for i in range(self.gdim)])
                    initial_rhs = fem_petsc.assemble_vector(
                        fem.form(-ufl.inner(A * ej, ufl.grad(v)) * dx)
                    )
                    direction_resources.callback(initial_rhs.destroy)
                    initial_rhs.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                                            mode=PETSc.ScatterMode.REVERSE)
                    fem_petsc.set_bc(initial_rhs, bcs)
                    wj = fem.Function(V)
                    wj_next = fem.Function(V)
                    initial_solver.solve(initial_rhs, wj.x.petsc_vec)
                    wj.x.scatter_forward()

                    for _ in range(num_time_steps):
                        mass.mult(wj.x.petsc_vec, step_rhs)
                        fem_petsc.set_bc(step_rhs, bcs)
                        step_solver.solve(step_rhs, wj_next.x.petsc_vec)
                        wj_next.x.scatter_forward()
                        wj, wj_next = wj_next, wj

                    mass.mult(wj.x.petsc_vec, step_rhs)
                    initial_rhs.axpy(-1.0, step_rhs)
                    fem_petsc.set_bc(initial_rhs, bcs)
                    uj = fem.Function(V)
                    final_solver.solve(initial_rhs, uj.x.petsc_vec)
                    uj.x.scatter_forward()
                    correctors.append(uj)

            return correctors
