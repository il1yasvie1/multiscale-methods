from dolfinx import mesh
from mpi4py import MPI

from msfem import MsFEM_Oversampling
from src.elliptic_problem_data import PeriodicCoefficientProblem
from src.utils import error_norm, plot_function


coarse_mesh = mesh.create_unit_square(MPI.COMM_SELF, 8, 8)
num_refinements = 4

msfem = MsFEM_Oversampling(PeriodicCoefficientProblem(), coarse_mesh, num_refinements)

msfem.offline(0.0, corrector_kind="parabolic", final_time=0.005, num_time_steps=8)
# msfem.offline(2.0, corrector_kind="elliptic")
u_ms = msfem.online()
if MPI.COMM_WORLD.rank == 0:
    print(error_norm(u_ms, msfem.problem.exact_solution, "H1"))


if MPI.COMM_WORLD.rank == 0:
    u_H = PeriodicCoefficientProblem().reference_solution(msfem.coarse_mesh)
    print(error_norm(u_H, msfem.problem.exact_solution, "H1"))

    plot_function(u_ms, "")

