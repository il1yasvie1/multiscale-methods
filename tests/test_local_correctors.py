import numpy as np
import pytest
import ufl

from dolfinx import fem, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI

from msfem import MsFEM_Oversampling
from src.elliptic_problem_data import (
    ConstantCoefficientProblem,
    NonConstantCoefficientProblem,
    PeriodicCoefficientProblem,
    RandomCoefficientProblem,
)
from src.local_correctors import get_local_modified_correctors
from src.utils import boundary_dofs
from msfem.experiment_metrics import CorrectorError, interface_jumps, solution_errors


@pytest.mark.parametrize("N, problem", [
    (2, NonConstantCoefficientProblem()),
    (8, NonConstantCoefficientProblem()),
    (4, RandomCoefficientProblem(model="checkerboard", micro_size=4)),
])
def test_modified_corrector_matches_generalized_eigenbasis(N, problem):
    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)
    V = fem.functionspace(domain, ("P", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    A = problem.coefficient(domain)
    dx = ufl.dx(domain=domain, metadata={"quadrature_degree": 12})
    boundary = boundary_dofs(V)
    free = np.setdiff1d(np.arange(V.dofmap.index_map.size_local), boundary).astype(np.int32)

    def dense_interior(form):
        matrix = fem_petsc.assemble_matrix(fem.form(form))
        try:
            matrix.assemble()
            return matrix.getValues(free, free)
        finally:
            matrix.destroy()

    M = dense_interior(ufl.inner(u, v) * dx)
    K = dense_interior(ufl.inner(A * ufl.grad(u), ufl.grad(v)) * dx)
    # Reduce the generalized eigenproblem with the Cholesky factor of M.
    L = np.linalg.cholesky(M)
    scaled_stiffness = np.linalg.solve(L, K)
    symmetric_operator = np.linalg.solve(L, scaled_stiffness.conj().T).conj().T
    eigenvalues, orthogonal_vectors = np.linalg.eigh(symmetric_operator)
    eigenvectors = np.linalg.solve(L.conj().T, orthogonal_vectors)
    T = 0.015
    weights = -np.expm1(-T * eigenvalues) / eigenvalues
    correctors = get_local_modified_correctors(V, problem, T)
    for j, corrector in enumerate(correctors):
        direction = ufl.as_vector([int(i == j) for i in range(2)])
        rhs = fem_petsc.assemble_vector(fem.form(-ufl.inner(A * direction, ufl.grad(v)) * dx))
        try:
            expected = eigenvectors @ (weights * (eigenvectors.conj().T @ rhs.array_r[free]))
        finally:
            rhs.destroy()
        np.testing.assert_allclose(corrector.x.array[free], expected, atol=1e-12, rtol=1e-10)
        np.testing.assert_array_equal(corrector.x.array[boundary], 0)


@pytest.mark.parametrize("N", [1, 4])
def test_modified_correctors_vanish_for_constant_coefficient(N):
    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)
    V = fem.functionspace(domain, ("P", 1))
    correctors = get_local_modified_correctors(V, ConstantCoefficientProblem(), 0.02)
    assert len(correctors) == 2
    for corrector in correctors:
        np.testing.assert_allclose(corrector.x.array, 0, atol=1e-13)


@pytest.mark.parametrize("kind, time_options", [
    ("elliptic", {}),
    ("parabolic", {"final_time": 0.02, "num_time_steps": 8}),
    ("modified", {"final_time": 0.02}),
])
def test_msfem_offline_and_online_for_each_mode(kind, time_options):
    coarse_mesh = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    method = MsFEM_Oversampling(NonConstantCoefficientProblem(), coarse_mesh, 2)
    method.offline(0.25, corrector_kind=kind, **time_options)
    solution = method.online()
    assert np.isfinite(solution.x.array).all()
    assert np.linalg.norm(solution.x.array) > 0
    if MPI.COMM_WORLD.rank == 0:
        method.A_H.destroy()
        method.b_H.destroy()


@pytest.mark.skipif(MPI.COMM_WORLD.size != 1, reason="Independent global-field comparison is serial")
@pytest.mark.parametrize("delta", [0., 1.])
@pytest.mark.parametrize("kind, time_options", [
    ("elliptic", {}),
    ("parabolic", {"final_time": 0.025, "num_time_steps": 40}),
    ("modified", {"final_time": 0.025}),
])
def test_periodic_metrics_against_global_ufl_integrals(delta, kind, time_options):
    problem = PeriodicCoefficientProblem(0.05)
    coarse = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    method = MsFEM_Oversampling(problem, coarse, 2)
    observer = CorrectorError(problem)
    method.offline(delta, corrector_kind=kind, corrector_observer=observer, **time_options)
    try:
        solution = method.online()
        measured = interface_jumps(method, solution)
        assert set(measured) == {"corrector_jump_l2", "msfem_jump_l2"}
        dS = ufl.dS(domain=method.fine_mesh)

        def jump_squared(function):
            return float(np.real(fem.assemble_scalar(fem.form(
                ufl.inner(ufl.jump(function), ufl.jump(function)) * dS
            ))))

        # Independent UFL integration on all internal fine facets. Within each
        # coarse cell the reconstructed P1 fields are continuous, giving zero.
        expected = {"msfem_jump_l2": np.sqrt(max(0., jump_squared(solution)))}
        corrector_squared = 0.
        errors_squared = dict(l2_error=0., h1_error=0., h1_semi_error=0.)
        for j in range(2):
            computed = fem.Function(method.V_multiscale)
            for data in method.cell_basis_data.values():
                computed.x.array[data.fine_dofs] = data.corrector_values[:, j]
            corrector_squared += jump_squared(computed)
            errors = solution_errors(computed, lambda domain: problem.periodic_cell_correctors(domain)[j])
            for name in errors_squared:
                errors_squared[name] += errors[name]**2
        for name, value in errors_squared.items():
            assert observer.results()[f"corrector_{name}"] == pytest.approx(np.sqrt(value), rel=1e-10, abs=1e-12)
        expected["corrector_jump_l2"] = np.sqrt(max(0., corrector_squared))
        for name, reference in expected.items():
            assert measured[name] == pytest.approx(reference, rel=1e-9, abs=1e-12)
    finally:
        method.A_H.destroy()
        method.b_H.destroy()
