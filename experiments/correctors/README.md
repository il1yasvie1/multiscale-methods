# Whole-domain corrector experiments

This experiment solves the two corrector directions on the centered square
`K_R = (-R/2, R/2)^2` using `PeriodicCoefficientProblem(epsilon=1.0)`. It
measures total error against the analytic exact periodic corrector on `K_L`,
where `L = R/delta` and `delta = 1.5`. The second corrector is analytically
zero and is checked during each run; reported errors sum both directions
without separate columns.

Run the small functional check first:

```sh
python run_experiment.py configs/correctors/periodic-corrector-smoke.yaml
python run_experiment.py plot configs/correctors/periodic-corrector-smoke.yaml
```

The full default-time sweep is:

```sh
python run_experiment.py configs/correctors/periodic-corrector-default-t.yaml
```

The full sweep has up to 2034 grid intervals per direction and is expensive.
The smoke configuration uses only 16 nominal intervals per unit length; its
errors are not spatial-convergence evidence. Runs are saved in
`results/correctors/<experiment_name>/runs.csv`, alongside a configuration
snapshot and figures. Completed runs are skipped on restart.

## Config interpretation

Set `experiment: correctors` in every corrector experiment YAML.
The YAML filename supplies `experiment_name`; for example,
`periodic-corrector-smoke.yaml` writes to
`results/correctors/periodic-corrector-smoke/`.

`R` and optional `T` are linear grids. With `T: {default_setting: true}` only,
each `R` gets one time, `T = (R-L)/(2*pi*sqrt(alpha*beta)) = 4*pi*R/3` for this
coefficient. If `T` also has `start`, `stop`, and `num`, its Cartesian product
with the `R` grid is run, and the default time for each `R` is added. The
elliptic corrector is solved only once per `R`. Parabolic steps use
`num_time_steps = ceil(T/max_time_step)`; modified correctors do not use time
steps.

The number of mesh intervals per direction is rounded **up** to a multiple of
six from `points_per_unit*R`. This places both sides of `K_L` on mesh lines so
the inner-region integrals are evaluated on the specified domain. The actual
`n` and `h = R/n` are recorded in every result row.

For each method, the results report the L2, H1-seminorm, and H1 total errors
summed over both corrector directions. The difference is exact minus computed
corrector, so the L2 error is

`E^alpha_{chi,||.||_{L2}} = (sum_j integral_{K_L} |chi_j - chi^h_{j,R,T}|^2 dx)^(1/2)`.

The H1-seminorm uses the gradient of the same difference. The H1 error is the
square root of the sum of the squared L2 and H1-seminorm errors. Figure axes use
the corresponding `E^alpha_{chi,||.||}` notation.

Plots are generated from completed CSV rows under `figures/R/`, `figures/T/`,
and `figures/overview/`. The fixed-setting folders are `R/fixed_T/`,
`R/fixed_default_T/`, and `T/fixed_R/`. A complete R-by-T grid also produces
colored sweeps and heatmaps. A default-time-only sweep produces R curves and
an overview without inventing a rectangular grid. Filenames follow the MsFEM
convention, for example
`R/fixed_T/fixed_T_corrector_l2_error_periodic_1_16_T1.png`. The CSV columns
are named `total_l2`, `total_h1_semi`, and `total_h1`.

For this experiment, `modified` defaults to SLEPc's phi function with a lumped mass
matrix. The `mass_lumped` field in each modified result row distinguishes these
runs from earlier consistent-mass results. This matches the paper's choice to
lump the mass matrix, though the matrix-function action is computed by SLEPc.

## Solver diagnostics

Solver settings are grouped by corrector kind. Parabolic settings are further
split because the initial mass solve and repeated time-step solve use different
matrices:

```yaml
solver_parameters:
  elliptic: {ksp_view: true}
  parabolic:
    initial: {ksp_type: preonly, pc_type: lu}
    step: {ksp_type: preonly, pc_type: lu, ksp_view: true}
  modified:
    mfn: {mfn_view: true}
```

Both parabolic groups default to `ksp_type: preonly` and `pc_type: lu` when
omitted. The default-time configuration enables `ksp_view` and `ksp_monitor`
for elliptic and both parabolic solves. These display the KSP/PC setup and
residuals.
For a short diagnostic run of all three methods, use
`python run_experiment.py configs/correctors/periodic-corrector-solver-debug.yaml`.
PETSc's option is `ksp_view` (singular). Put `mfn_view` and `mfn_monitor` under
`modified.mfn` to display the SLEPc matrix-function solver. Set a flag to
`false` or remove it to disable that output. Diagnostic flags may change
while resuming the same experiment; numerical solver settings require a new
YAML filename and result directory.

Mass lumping makes the modified solver's mass inverse diagonal, so it has no
inner KSP to monitor; `mfn_view` and `mfn_monitor` show its matrix-function
iterations. To use the consistent-mass path, set
`solver_parameters.modified.mass_lumped: false`; only then does
`modified.mass` configure its KSP. This path defaults to a direct LU mass
solve and prints one KSP setup, the first mass solve's residuals,
and a count of its inner solves when diagnostics are enabled. The elliptic and
parabolic solvers default to direct LU. Parabolic assembles its mass and time-step
matrices once and reuses the same two KSP objects for all directions and time
steps. Its KSP monitor can still print once per solve, so a run with many time
steps can produce a long terminal log and distort wall-time comparisons.
