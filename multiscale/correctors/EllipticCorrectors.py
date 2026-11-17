from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl

from multiscale.fem_utils import zero_dirichlet_bc
from .LocalCorrectors import LocalCorrectors


class EllipticCorrectors(LocalCorrectors):
    def solve(self) -> list[fem.Function]:
        V = self.function_space
        A = self.coefficient
        uj, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(
            domain=self.mesh,
            metadata={"quadrature_degree": self.quadrature_degree},
        )
        lhs = ufl.inner(A * ufl.grad(uj), ufl.grad(v)) * dx
        zero_bc = zero_dirichlet_bc(V)

        petsc_options = {"ksp_type": "preonly", "pc_type": "lu",
                         "ksp_error_if_not_converged": True}
        petsc_options.update(self.solver_parameters)

        correctors = []
        for j in range(self.gdim):
            ej = ufl.as_vector([int(i == j) for i in range(self.gdim)])
            rhs = -ufl.inner(A * ej, ufl.grad(v)) * dx
            problem = LinearProblem(
                lhs,
                rhs,
                bcs=[zero_bc],
                petsc_options_prefix="local_corrector_problem_",
                petsc_options=petsc_options,
            )
            uj = problem.solve()
            correctors.append(uj)
        return correctors
