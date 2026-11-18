"""Run elliptic, parabolic, and modified correctors on centered K_R."""

from pathlib import Path
from time import perf_counter
import traceback

from mpi4py import MPI
import numpy as np

from experiments.solver_config import options_for_kind
from .experiment_config import expand_runs, load_config, solver_parameters
from .experiment_data import completed_run_ids, load_results, prepare_experiment, save_result


def run_one(params: dict, grouped_solver_options: dict | None = None) -> dict:
    from dolfinx import mesh
    from multiscale.correctors import create_local_correctors
    from multiscale.problems import PeriodicCoefficientProblem
    from .experiment_metrics import compare, inner_measure, upscaling_error

    comm = MPI.COMM_WORLD
    started = perf_counter()
    R, L, n = params["R"], params["L"], params["n"]
    domain = mesh.create_unit_square(comm, n, n)
    domain.geometry.x[:, :2] = R * (domain.geometry.x[:, :2] - 0.5)
    problem = PeriodicCoefficientProblem(epsilon=1.0)
    coefficient = problem.coefficient(domain)
    exact = problem.periodic_cell_correctors(domain)
    dx_inner = inner_measure(domain, L)
    setup_seconds = comm.allreduce(perf_counter() - started, op=MPI.MAX)

    settings = {"kind": params["corrector_kind"]}
    if params["T"] is not None:
        settings["final_time"] = params["T"]
    if params["num_time_steps"] is not None:
        settings["num_time_steps"] = params["num_time_steps"]
    solver_options = options_for_kind(
        grouped_solver_options or {}, params["corrector_kind"],
        default_modified_mass_lumped=True,
    )
    solver = create_local_correctors(domain, ("P", 1), coefficient, settings,
                                     solver_parameters=solver_options)
    started = perf_counter()
    correctors = solver.solve()
    solve_seconds = comm.allreduce(perf_counter() - started, op=MPI.MAX)
    j2_max = comm.allreduce(float(np.max(np.abs(correctors[1].x.array))), op=MPI.MAX)
    if j2_max > 1e-8:
        raise RuntimeError(f"Expected the second corrector to vanish; max |chi_2|={j2_max:g}")

    started = perf_counter()
    total = compare(correctors, exact, dx_inner)
    upscaling = upscaling_error(correctors, coefficient, dx_inner,
                                problem.homogenized_tensor())
    metrics_seconds = comm.allreduce(perf_counter() - started, op=MPI.MAX)
    count = solver.function_space.dofmap.index_map.size_local
    num_dofs = comm.allreduce(count, op=MPI.SUM)
    return {
        "num_dofs": num_dofs, "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds, "metrics_seconds": metrics_seconds,
        **{f"total_{key}": value for key, value in total.items()},
        **upscaling,
    }


def run_config(config_path: str | Path) -> Path:
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        try:
            config = load_config(config_path)
            runs = expand_runs(config)
            csv_path = prepare_experiment(config)
            setup_error = None
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
            runs = None
            csv_path = None
    else:
        setup_error = None
        runs = None
        csv_path = None
    setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        raise RuntimeError("Corrector experiment setup failed: " + setup_error)
    runs = comm.bcast(runs, root=0)
    csv_path = comm.bcast(csv_path, root=0)
    grouped_solver_options = comm.bcast(
        solver_parameters(config) if comm.rank == 0 else None, root=0
    )
    table = load_results(csv_path) if comm.rank == 0 else None
    completed = comm.bcast(completed_run_ids(table) if comm.rank == 0 else None, root=0)
    failures = 0
    for params in runs:
        if params["run_id"] in completed:
            continue
        if comm.rank == 0:
            time_label = f" T={params['T']}" if params["T"] is not None else ""
            print(f"R={params['R']:g}{time_label} {params['corrector_kind']}", flush=True)
        local_error = None
        try:
            values = run_one(params, grouped_solver_options)
        except Exception:
            local_error = traceback.format_exc()
            values = {}
        errors = comm.allgather(local_error)
        if comm.rank == 0:
            row = {**params, "status": "failed" if any(errors) else "completed",
                   "error_message": "\n".join(error for error in errors if error)}
            failures += bool(any(errors))
            if not any(errors):
                row.update(values)
            table = save_result(csv_path, table, row)
            print(f"  {row['status']}", flush=True)
    failures = comm.bcast(failures if comm.rank == 0 else None, root=0)
    if failures:
        raise RuntimeError(f"{failures} corrector runs failed; see {csv_path}")
    return csv_path
