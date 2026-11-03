"""MsFEM batch runner used by the repository-level run.py."""

from pathlib import Path
from time import perf_counter
import traceback

from dolfinx import mesh
from mpi4py import MPI
import pandas as pd
import yaml

from .analysis import make_plots
from .experiment_config import expand_runs, load_config, validate_snapshot
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

COUNT_COLUMNS = (
    "num_of_coarse_cells", "num_of_fine_cells", "num_of_coarse_dofs",
    "num_of_fine_dofs", "num_of_reconstruction_dofs",
)
ERROR_COLUMNS = tuple(
    f"{prefix}_{norm}_error"
    for prefix in ("msfem", "coarse_fem", "fine_fem", "corrector")
    for norm in ("l2", "h1", "h1_semi")
)
COLUMNS = (
    "experiment_name", "run_id", "status", "method", "problem", "epsilon",
    "N_H", "N_h", "H", "h", "delta", "assembly_type", "corrector_kind",
    "final_time", "num_time_steps", "time_step", *COUNT_COLUMNS,
    *ERROR_COLUMNS, "msfem_jump", "setup_seconds", "offline_seconds",
    "online_seconds", "coarse_fem_solve_seconds", "fine_fem_solve_seconds",
    "metrics_seconds", "error_message",
)


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
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Batch experiments currently support one MPI rank")

    problem = _problem(params)
    started = perf_counter()
    coarse_mesh = mesh.create_unit_square(
        MPI.COMM_SELF, params["N_H"], params["N_H"]
    )
    refinements = (params["N_h"] // params["N_H"]).bit_length() - 1
    method = MsFEM_Oversampling(problem, coarse_mesh, refinements)
    result = {
        "setup_seconds": perf_counter() - started,
        "num_of_coarse_cells": method.num_coarse_cells,
        "num_of_fine_cells": method.fine_mesh.topology.index_map(method.tdim).size_local,
        "num_of_coarse_dofs": method.num_coarse_dofs,
        "num_of_reconstruction_dofs": method.V_multiscale.dofmap.index_map.size_local,
    }

    key = (params["problem"], params["epsilon"], params["N_H"], params["N_h"])
    if key not in baseline_cache:
        baseline_cache[key] = _baseline(method, problem)
    result.update(baseline_cache[key])

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
    result["offline_seconds"] = perf_counter() - started - (
        corrector_error.seconds if corrector_error else 0.0
    )

    started = perf_counter()
    solution = method.online()
    result["online_seconds"] = perf_counter() - started

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
    config = load_config(config_path)
    runs = list(expand_runs(config))
    output = Path("work") / config["experiment_name"]
    csv_path = output / "runs.csv"
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "config.yaml"
    if snapshot.exists():
        validate_snapshot(yaml.safe_load(snapshot.read_text()), config)
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))

    table = pd.read_csv(csv_path, dtype={"run_id": str}) if csv_path.exists() else pd.DataFrame(columns=COLUMNS)
    completed = set(table.loc[table["status"] == "completed", "run_id"])
    baseline_cache = {}
    failures = 0
    for params in runs:
        if params["run_id"] in completed:
            continue
        print(f"{params['run_id']} {params['problem']} H={params['H']:g} "
              f"h={params['h']:g} delta={params['delta']:g} "
              f"{params['corrector_kind']} {params['assembly_type']}", flush=True)
        row = {**params, "status": "completed"}
        try:
            row.update(run_one(params, baseline_cache))
            row["error_message"] = ""
        except Exception as error:
            failures += 1
            row["status"] = "failed"
            row["error_message"] = f"{type(error).__name__}: {error}"
            run_dir = output / params["run_id"]
            run_dir.mkdir(exist_ok=True)
            (run_dir / "run.log").write_text(traceback.format_exc())
        table = table.loc[table["run_id"] != params["run_id"]]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        temp_path = csv_path.with_suffix(".tmp")
        table.reindex(columns=COLUMNS).to_csv(temp_path, index=False)
        temp_path.replace(csv_path)

    if failures:
        raise SystemExit(f"{failures} runs failed; see {csv_path} and per-run logs")
    print(f"Results: {csv_path}")
    print(f"Created {make_plots(csv_path)} figures in {output / 'figures'}")
