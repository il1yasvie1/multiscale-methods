# multiscale-methods

The repository separates reusable numerical components from individual multiscale algorithms.

```text
src/                  Shared problem data, mesh, patch, corrector, and basis utilities
  elliptic_problem_data.py
  two_level_mesh.py
  oversampling.py
  local_correctors.py
  local_basis.py
  utils.py
msfem/                MsFEM solver and experiment workflow
  msfem.py            Solver
  experiment.py       Batch runner
  experiment_config.py  YAML validation and run expansion
  experiment_data.py    Output paths, CSV schema, and atomic writes
  experiment_metrics.py Numerical diagnostics
  plotting/           Plot generation from saved CSV data
    common.py         Labels, style, and grouping helpers
    files.py          Figure names, directories, and cleanup
    spatial.py        H and delta sweeps
    time.py           Final-time sweeps
  analysis.py         Compatibility import for plotting entry points
  example.py          Small MsFEM example
configs/              Batch experiment YAML
run.py                Command-line entry point
work/                 Generated data (ignored by Git)
```

Import the solver from the repository root with `from msfem import MsFEM_Oversampling`. The
`src` modules contain reusable numerical components. Algorithm classes belong
in their own packages.

Batch experiments replicate the unit-square triangular coarse and fine meshes on
each MPI rank. Ranks compute disjoint coarse-cell patches; rank 0 assembles and
solves the coarse system. This first MPI implementation trades memory for a
simple independent-patch workflow.
They require DOLFINx, PETSc, mpi4py, PyYAML, pandas and Matplotlib.

```sh
python run.py configs/periodic_msfem.yaml
mpiexec -n 2 python run.py configs/smoke-test.yaml
python run.py plot configs/periodic_msfem.yaml
python run.py --help
```

`experiment_name` selects a directory under `work/`. Each valid parameter
combination creates one row in `runs.csv` with a deterministic `run_id`.
Completed rows are skipped on rerun; failed rows are retried. `N_h/N_H` must
be a power of two. Each `parabolic_settings` entry pairs an explicit
`final_time` with `num_time_steps`; elliptic runs leave time fields empty.
You can extend or change parameter lists under the same `experiment_name`:
existing rows remain in `runs.csv`, only new combinations run, and plots use
all completed rows for that experiment. `work/<experiment_name>/config.yaml`
records the latest YAML. Changing the method or problem family requires a new
`experiment_name`.

The runner writes one row at a time to `runs.csv` through a temporary file,
then replaces the CSV. `python run.py plot <config>` imports only the plotting
and data modules, so it can redraw figures without the DOLFINx solver.

`problem.name` accepts `constant`, `nonconstant`, or `periodic`; the periodic
problem accepts an `epsilon` list. `assembly_type` accepts `petrov_galerkin`
or `galerkin`. Lists form a Cartesian product, except the parabolic time pairs.

All solution errors use the analytical solution. MsFEM H1 errors are broken
norms because reconstruction may jump across coarse-cell interfaces; the jump
is separately reported. Periodic corrector errors are integrated over each
coarse cell against the zero-mean periodic cell correctors, without mean
alignment. No checkpoints are written. The runner also measures coarse and
fine P1 FEM errors and solve times. Plots read only the saved CSV.
Plots are generated only when a sweep has at least two distinct x values.
The runner generates plots after the batch completes. The configuration path
may be written with or without the `.yaml` suffix.
`plot` redraws figures from all completed CSV rows for the experiment without
running any solver. H and delta sweeps use blue for elliptic and red for
parabolic correctors. The H sweep uses log-log axes with power-of-two H ticks;
darker shades mean larger delta. In the delta sweep, darker shades mean smaller
H, and the color scales are labeled with powers of two. The legend and color
scales are placed outside the plotting area.
Figures are grouped under `work/<experiment_name>/figures/H/`, `delta/`, and
`time/`; the H and delta directories also contain their sweep overviews. The
`delta/fixed_H/` directory has additional MsFEM error plots with one panel per
coarse resolution. Each panel has its own vertical scale so variation with
delta at fixed H remains visible. Redrawing removes old generated plots from
the former flat layout.
Time plots are generated only when at least two final times were tested. They
show one panel per H, color by delta, and use dotted horizontal lines for the
corresponding elliptic references. Tick labels show both T and N_t; comparisons
should account for any change in time-step size T/N_t. The jump vertical axis
is linear so numerical zero values remain visible. Delta plots keep a linear
horizontal axis because the sweep may include delta=0. Obsolete h-sweep plots
are removed when plots are redrawn.
