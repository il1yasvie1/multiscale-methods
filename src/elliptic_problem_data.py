from abc import ABC, abstractmethod
from dolfinx.mesh import Mesh 
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import dolfinx

import ufl

from src.utils import boundary_dofs

class EllipticProblemData(ABC):
    @abstractmethod
    def coefficient(self, domain: Mesh):
        raise NotImplementedError

    @abstractmethod
    def source(self, domain: Mesh):
        raise NotImplementedError

    def reference_solution(self, domain: Mesh) -> fem.Function:
        V = fem.functionspace(domain, ("P", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(domain=domain, metadata={"quadrature_degree": 12})
        a = ufl.inner(self.coefficient(domain) * ufl.grad(u), ufl.grad(v)) * dx
        L = self.source(domain) * v * dx
        bc = fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs(V), V)
        problem = LinearProblem(
            a, L, bcs=[bc], petsc_options_prefix="elliptic_problem_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        return problem.solve()


class ConstantCoefficientProblem(EllipticProblemData):
    def coefficient(self, domain):
        return 1.0

    def exact_solution(self, domain):
        x = ufl.SpatialCoordinate(domain)
        u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) 
        return u_exact

    def source(self, domain: Mesh):
        f = 2 * ufl.pi**2 * self.exact_solution(domain) 
        return f


class NonConstantCoefficientProblem(EllipticProblemData):
    def coefficient(self, domain):
        x = ufl.SpatialCoordinate(domain)
        return 1.0 + x[0]

    def exact_solution(self, domain):
        x = ufl.SpatialCoordinate(domain)
        u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) 
        return u_exact

    def source(self, domain: Mesh):
        x = ufl.SpatialCoordinate(domain)
        return (
            2 * ufl.pi**2 * (1.0 + x[0]) * self.exact_solution(domain)
            - ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        )


class PeriodicCoefficientProblem(EllipticProblemData):
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def coefficient(self, domain: Mesh):
        x = ufl.SpatialCoordinate(domain)[0]
        theta = 2 * ufl.pi * x / self.epsilon
        return (1 / (8 * ufl.pi**2)) * ufl.as_matrix(
            [
                [2 / (2 + ufl.cos(theta)), 0],
                [0, 1 + 0.5 * ufl.cos(theta)],
            ]
        )

    def exact_solution(self, domain: Mesh):
        x, y = ufl.SpatialCoordinate(domain)
        return (
            ufl.sin(2 * ufl.pi * x) * ufl.sin(2 * ufl.pi * y)
            + self.epsilon / 2
            * ufl.cos(2 * ufl.pi * x) * ufl.sin(2 * ufl.pi * y)
            * (ufl.sin(2 * ufl.pi * x / self.epsilon)
               - x * ufl.sin(2 * ufl.pi / self.epsilon))
        )

    def source(self, domain: Mesh):
        return -ufl.div(
            self.coefficient(domain) * ufl.grad(self.exact_solution(domain))
        )

    def periodic_cell_correctors(self, domain: Mesh):
        x = ufl.SpatialCoordinate(domain)
        chi = self.epsilon / (4 * ufl.pi) * ufl.sin(
            2 * ufl.pi * x[0] / self.epsilon
        )
        return [chi, 0.0]
