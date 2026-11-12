import dolfinx
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from petsc4py import PETSc
import ufl

from src.utils import boundary_dofs
from src.elliptic_problem_data import EllipticProblemData


def get_local_elliptic_correctors(V: fem.FunctionSpace, problem: EllipticProblemData
) -> list[fem.Function]:

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    A = problem.coefficient(V.mesh)
    dx = ufl.dx(domain=V.mesh, metadata={"quadrature_degree": 12})
    lhs =  ufl.inner(A * ufl.grad(u), ufl.grad(v)) * dx
    correctors = []
    for direction in range(V.mesh.geometry.dim):
        ej = ufl.as_vector([int(i == direction) for i in range(V.mesh.geometry.dim)])
        rhs = ufl.inner(-A * ej, ufl.grad(v)) * dx
        linear_problem = LinearProblem(
            lhs,
            rhs,
            bcs=[fem.dirichletbc(dolfinx.default_real_type(0), boundary_dofs(V), V)],
            petsc_options_prefix="local_corrector_problem_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            })
        corrector = linear_problem.solve()
        correctors.append(corrector)
        del linear_problem
    return correctors


def get_local_modified_correctors(
    V: fem.FunctionSpace,
    problem: EllipticProblemData,
    final_time: float,
) -> list[fem.Function]:
    from slepc4py import SLEPc

    gdim = V.mesh.geometry.dim
    correctors = [fem.Function(V) for _ in range(gdim)]
    free = np.setdiff1d(
        np.arange(len(correctors[0].x.array), dtype=PETSc.IntType), boundary_dofs(V)
    )
    if not free.size:
        return correctors

    A = problem.coefficient(V.mesh)
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.dx(domain=V.mesh, metadata={"quadrature_degree": 12})

    class DiffusionOperator:
        def mult(self, _mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec) -> None:
            stiffness.mult(x, work)
            mass_solver.solve(work, y)

    def interior_matrix(form):
        matrix = fem_petsc.assemble_matrix(fem.form(form))
        matrix.assemble()
        interior = matrix.createSubMatrix(free_is, free_is)
        matrix.destroy()
        return interior

    comm = V.mesh.comm
    free_is = PETSc.IS().createGeneral(free, comm=comm)
    mass = interior_matrix(ufl.inner(trial, test) * dx)
    stiffness = interior_matrix(ufl.inner(A * ufl.grad(trial), ufl.grad(test)) * dx)

    mass_solver = PETSc.KSP().create(comm)
    mass_solver.setOperators(mass)
    mass_solver.setType(PETSc.KSP.Type.PREONLY)
    mass_solver.getPC().setType(PETSc.PC.Type.LU)
    mass_solver.setErrorIfNotConverged(True)
    mass_solver.setUp()

    work = mass.createVecLeft()
    initial = mass.createVecRight()
    solution = mass.createVecRight()

    # A_h = M^{-1} L: apply stiffness, then solve with mass.
    operator = PETSc.Mat().createPython(mass.getSizes(), DiffusionOperator(), comm=comm)
    operator.setUp()

    mfn = SLEPc.MFN().create(comm)
    mfn.setOperator(operator)
    mfn.setType(SLEPc.MFN.Type.KRYLOV)
    mfn.setTolerances(tol=max(1e-12, 10 * np.finfo(PETSc.RealType).eps))
    mfn.setErrorIfNotConverged(True)
    function = mfn.getFN()
    function.setType(SLEPc.FN.Type.PHI)
    function.setPhiIndex(1)
    # chi^{h,2}_{j,K}(T) = T phi_1(-T A_h) v^h_{j,0}.
    function.setScale(-final_time, final_time)

    for direction, corrector in enumerate(correctors):
        unit_vector = ufl.as_vector([int(i == direction) for i in range(gdim)])
        rhs = fem_petsc.assemble_vector(
            fem.form(-ufl.inner(A * unit_vector, ufl.grad(test)) * dx)
        )
        work.array[:] = rhs.array_r[free]
        rhs.destroy()
        mass_solver.solve(work, initial)

        if initial.norm() == 0.0:
            continue
        mfn.solve(initial, solution)
        corrector.x.array[free] = solution.array_r
        corrector.x.scatter_forward()

    function.destroy()
    mfn.destroy()
    operator.destroy()
    solution.destroy()
    initial.destroy()
    work.destroy()
    mass_solver.destroy()
    stiffness.destroy()
    mass.destroy()
    free_is.destroy()
    return correctors


def get_local_parabolic_correctors(
    V: fem.FunctionSpace,
    problem: EllipticProblemData,
    final_time: float,
    num_time_steps: int,
) -> list[fem.Function]:

    A = problem.coefficient(V.mesh)
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.dx(domain=V.mesh, metadata={"quadrature_degree": 12})

    mass_form = fem.form(trial * test * dx)
    stiffness = ufl.inner(A * ufl.grad(trial), ufl.grad(test)) * dx
    time_step = final_time / num_time_steps
    step_form = fem.form(trial * test * dx + time_step * stiffness)
    zero_bc = fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs(V), V)
    bcs = [zero_bc]

    mass = fem_petsc.assemble_matrix(mass_form)
    mass.assemble()
    constrained_mass = fem_petsc.assemble_matrix(mass_form, bcs=bcs)
    constrained_mass.assemble()
    step_matrix = fem_petsc.assemble_matrix(step_form, bcs=bcs)
    step_matrix.assemble()

    def create_lu_solver(matrix: PETSc.Mat, prefix: str) -> PETSc.KSP:
        solver = PETSc.KSP().create(matrix.comm)
        solver.setOptionsPrefix(prefix)
        solver.setOperators(matrix)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
        solver.setErrorIfNotConverged(True)
        solver.setUp()
        return solver

    initial_solver = create_lu_solver(constrained_mass, "parabolic_initial_")
    step_solver = create_lu_solver(step_matrix, "parabolic_step_")
    step_rhs = mass.createVecRight()
    correctors: list[fem.Function] = []
    try:
        gdim = V.mesh.geometry.dim
        for direction in range(gdim):
            unit_vector = ufl.as_vector(
                [int(i == direction) for i in range(gdim)]
            )
            initial_form = fem.form(
                -ufl.inner(A * unit_vector, ufl.grad(test)) * dx
            )
            initial_rhs = fem_petsc.assemble_vector(initial_form)
            try:
                fem_petsc.set_bc(initial_rhs, bcs)
                previous = fem.Function(V)
                current = fem.Function(V)
                initial_solver.solve(initial_rhs, previous.x.petsc_vec)
                previous.x.scatter_forward()

                corrector = fem.Function(V)
                corrector.x.array[:] = 0.5 * time_step * previous.x.array
                for step in range(num_time_steps):
                    mass.mult(previous.x.petsc_vec, step_rhs)
                    fem_petsc.set_bc(step_rhs, bcs)
                    step_solver.solve(step_rhs, current.x.petsc_vec)
                    current.x.scatter_forward()
                    weight = 0.5 if step == num_time_steps - 1 else 1.0
                    corrector.x.array[:] += weight * time_step * current.x.array
                    previous, current = current, previous

                corrector.x.scatter_forward()
                correctors.append(corrector)
            finally:
                initial_rhs.destroy()
    finally:
        step_rhs.destroy()
        initial_solver.destroy()
        step_solver.destroy()
        mass.destroy()
        constrained_mass.destroy()
        step_matrix.destroy()

    return correctors
