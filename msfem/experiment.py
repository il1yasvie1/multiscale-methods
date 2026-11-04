"""MsFEM batch runner used by the repository-level run.py."""

from pathlib import Path
from time import perf_counter
import traceback

from dolfinx import mesh
from mpi4py import MPI

from .analysis import make_plots
from .experiment_config import expand_runs, load_config
from .experiment_data import load_results, prepare_experiment, save_result
from .experiment_metrics import CorrectorError, jump_norm, solution_errors
from .msfem import MsFEM_Oversampling
from src.elliptic_problem_data import (
    ConstantCoefficientProblem,
    NonConstantCoefficientProblem,
    PeriodicCoefficientProblem,
)


PROBLEMS = {
    "constant": ConstantCoefficientProblem,
    "nonconstant": NonConstantCoefficientProblem,
    "periodic": PeriodicCoefficientProblem,
}

def _problem(params: dict):
    cls = PROBLEMS[params["problem"]]
    return cls(params["epsilon"]) if params["problem"] == "periodic" else cls()


def _baseline(method, problem) -> dict:
    values = {}
    for name, domain in (("coarse_fem", method.coarse_mesh),
                         ("fine_fem", method.fine_mesh)):
        started = perf_counter()
        solution = problem.reference_solution(domain)
        values[f"{name}_solve_seconds"] = perf_counter() - started
        values.update({f"{name}_{key}": value for key, value in
                       solution_errors(solution, problem.exact_solution).items()})
        if name == "fine_fem":
            values["num_of_fine_dofs"] = solution.function_space.dofmap.index_map.size_local
    return values


def run_one(params: dict, baseline_cache: dict) -> dict:
    comm = MPI.COMM_WORLD
    problem = _problem(params)
    started = perf_counter()
    coarse_mesh = mesh.create_unit_square(
        MPI.COMM_SELF, params["N_H"], params["N_H"]
    )
    refinements = (params["N_h"] // params["N_H"]).bit_length() - 1
    method = MsFEM_Oversampling(problem, coarse_mesh, refinements)
    setup_seconds = comm.allreduce(perf_counter() - started, op=MPI.MAX)
    result = {
        "setup_seconds": setup_seconds,
        "num_of_coarse_cells": method.num_coarse_cells,
        "num_of_fine_cells": method.fine_mesh.topology.index_map(method.tdim).size_local,
        "num_of_coarse_dofs": method.num_coarse_dofs,
        "num_of_reconstruction_dofs": method.V_multiscale.dofmap.index_map.size_local,
    }

    corrector_error = CorrectorError(problem) if hasattr(
        problem, "periodic_cell_correctors"
    ) else None
    started = perf_counter()
    method.offline(
        params["delta"], corrector_kind=params["corrector_kind"],
        final_time=params["final_time"], num_time_steps=params["num_time_steps"],
        assembly_type=params["assembly_type"],
        corrector_observer=corrector_error,
    )
    offline_seconds = perf_counter() - started - (
        corrector_error.seconds if corrector_error else 0.0)
    result["offline_seconds"] = comm.allreduce(offline_seconds, op=MPI.MAX)
    if corrector_error is not None:
        corrector_error.l2_squared = comm.allreduce(corrector_error.l2_squared, op=MPI.SUM)
        corrector_error.h1_semi_squared = comm.allreduce(corrector_error.h1_semi_squared, op=MPI.SUM)
        corrector_error.seconds = comm.allreduce(corrector_error.seconds, op=MPI.MAX)

    started = perf_counter()
    solution = method.online()
    result["online_seconds"] = comm.allreduce(perf_counter() - started, op=MPI.MAX)

    if comm.rank == 0:
        key = (params["problem"], params["epsilon"], params["N_H"], params["N_h"])
        if key not in baseline_cache:
            baseline_cache[key] = _baseline(method, problem)
        result.update(baseline_cache[key])
        started = perf_counter()
        result.update({f"msfem_{key}": value for key, value in
                       solution_errors(solution, problem.exact_solution).items()})
        result["msfem_jump"] = jump_norm(solution, params["H"])
        if corrector_error is not None:
            result.update(corrector_error.results())
        result["metrics_seconds"] = perf_counter() - started + (
            corrector_error.seconds if corrector_error else 0.0
        )
        method.A_H.destroy()
        method.b_H.destroy()
    return result


def run_config(config_path: Path) -> None:
    comm = MPI.COMM_WORLD
    setup_error = None
    if comm.rank == 0:
        try:
            config = load_config(config_path)
            csv_path = prepare_experiment(config)
            output = csv_path.parent
            table = load_results(csv_path)
            completed = set(table.loc[table["status"] == "completed", "run_id"])
            runs = [params for params in expand_runs(config)
                    if params["run_id"] not in completed]
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
    else:
        runs = None
    setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        raise RuntimeError("Experiment setup failed: " + setup_error)
    runs = comm.bcast(runs, root=0)
    baseline_cache = {}
    failures = 0
    for params in runs:
        if comm.rank == 0:
            print(f"{params['run_id']} {params['problem']} H={params['H']:g} "
                  f"h={params['h']:g} delta={params['delta']:g} "
                  f"{params['corrector_kind']} {params['assembly_type']}", flush=True)
        local_error = None
        try:
            result = run_one(params, baseline_cache)
        except Exception as error:
            local_error = f"rank {comm.rank}: {type(error).__name__}: {error}\n{traceback.format_exc()}"
        errors = comm.allgather(local_error)
        write_error = None
        if comm.rank == 0:
            try:
                row = {**params, "status": "completed"}
                if any(errors):
                    failures += 1
                    row["status"] = "failed"
                    row["error_message"] = "; ".join(e.splitlines()[0] for e in errors if e)
                    run_dir = output / params["run_id"]
                    run_dir.mkdir(exist_ok=True)
                    (run_dir / "run.log").write_text("\n".join(e for e in errors if e))
                else:
                    row.update(result)
                    row["error_message"] = ""
                table = save_result(csv_path, table, row)
            except Exception as error:
                write_error = f"{type(error).__name__}: {error}"
        write_error = comm.bcast(write_error, root=0)
        if write_error:
            raise RuntimeError("Experiment result write failed: " + write_error)

    if comm.rank == 0:
        if failures:
            raise SystemExit(f"{failures} runs failed; see {csv_path} and per-run logs")
        print(f"Results: {csv_path}")
        print(f"Created {make_plots(csv_path)} figures in {output / 'figures'}")
