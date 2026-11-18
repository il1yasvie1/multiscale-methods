# Whole-domain corrector experiments

This experiment solves the two corrector directions on the centered square
`K_R = (-R/2, R/2)^2` using `PeriodicCoefficientProblem(epsilon=1.0)`. It
measures total error against the analytic exact periodic corrector on `K_L`,
where `R = (1 + 2*delta)L`: `delta` is the extension on each side measured in
units of `L`. The second corrector is analytically
zero and is checked during each run; reported errors sum both directions
without separate columns.

Run or resume the default-time sweep, then redraw it without solving:

```sh
python run_experiment.py configs/correctors/default-t.yaml
python run_experiment.py plot configs/correctors/default-t.yaml
```

The elliptic-only sweep is:

```sh
python run_experiment.py configs/correctors/only-elliptic.yaml
```

Runs are saved in
`results/correctors/<experiment_name>/runs.csv`, alongside a configuration
snapshot and figures. Completed runs are skipped on restart.

## Config interpretation

Set `experiment: correctors` in every corrector experiment YAML.
The YAML filename supplies `experiment_name`; for example, `default-t.yaml`
writes to `results/correctors/default-t/`.

`R` is a linear grid. `T` is required when `corrector_kind` includes
`parabolic` or `modified`; an elliptic-only experiment may omit it and produces
ordinary `R` plots without a time label. With `T: {default_setting: true}` only,
each `R` gets one time. For this coefficient,
`T = (R-L)/(2*pi*sqrt(alpha*beta)) = 8*pi*delta*R/(1 + 2*delta)`; at
`delta = 0.25`, this reduces to `4*pi*R/3`. If `T` also has `start`, `stop`,
and `num`, its Cartesian product
with the `R` grid is run, and the default time for each `R` is added. The
elliptic corrector is solved only once per `R`. Parabolic steps use
`num_time_steps = ceil(T/max_time_step)`; modified correctors do not use time
steps.

The number of mesh intervals per direction is rounded up from
`points_per_unit*R`. Common rational values of `delta` also use a compatible
multiple that places both sides of `K_L` on mesh lines: `delta = 0.25`, `0.5`,
and `1` use multiples of 6, 4, and 3. The actual `n` and `h = R/n` are recorded
in every result row. For other finite nonnegative values, the selected cells
give the mesh representation of `K_L`, and the upscaling average uses their
assembled area.

For each method, the results report the L2, H1-seminorm, and H1 total errors
summed over both corrector directions. The difference is exact minus computed
corrector, so the L2 error is

`E^alpha_{chi,||.||_{L2}} = (sum_j integral_{K_L} |chi_j - chi^h_{j,R,T}|^2 dx)^(1/2)`.

For every corrector method, the CSV records the uniform average of
`a(e_j + grad(u_j))` over `K_L` as `a0_ij` and its Frobenius error relative
to the exact homogenized tensor `I/(8*pi**2)`. Rows carry
`upscaling_domain: K_L`, so older full-box averages are recomputed on resume.
The `R_upscaling_error_*.png` log-log plot compares the elliptic, parabolic,
and modified methods at the default time with an `O(R^-1)` guide. The R
overview adds this plot as a fourth panel. The elliptic method uses the existing
zero Dirichlet corrector; the standard cell problem in
Abdulle--Arjmand--Paganoni uses periodic boundary conditions, so this is an
order check for the Dirichlet variant rather than a literal reproduction of
that theorem.

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
