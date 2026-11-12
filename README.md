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
    notation.py       Mathematical labels matching notes/notes.qmd
    common.py         Style, legends, and grouping helpers
    files.py          Figure names, directories, and cleanup
    spatial.py        H and delta sweeps
    time.py           Final-time sweeps
configs/              Batch experiment YAML
notes/                Mathematical notes, figures, and rendered documentation
  notes.qmd           Quarto source
  notes.css           Note styling
  _math-macros.qmd    Shared math macro definitions
  notes.html          Rendered notes
  assets/             Source figures (PNG/SVG)
  notes_files/        HTML runtime dependencies (commit for GitHub Pages)
  plot_oversampling.py  Oversampling figure generator
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
They require DOLFINx, PETSc, mpi4py, SciPy, PyYAML, pandas and Matplotlib.
The modified mode also requires SLEPc and `slepc4py`.

```sh
python run.py configs/periodic-128.yaml
mpiexec -n 2 python run.py configs/smoke-test.yaml
python run.py plot configs/periodic-128.yaml
python run.py --help
```

`experiment_name` selects a directory under `work/`. Each valid parameter
combination creates one row in `runs.csv` with a deterministic `run_id`.
Completed rows are skipped on rerun; failed rows are retried. `N_h/N_H` must
be a power of two. `corrector_kind` accepts `elliptic`, `parabolic`, and
`modified`. Each `parabolic_settings` entry pairs `final_time` with
`num_time_steps`. Each `modified_settings` entry takes `final_time` only.
Elliptic runs leave the time fields empty. The smoke-test config runs all
three modes at matched final times for the two time-dependent formulations.

The modified mode uses the local equation in [Abdulle, Arjmand and Paganoni,
Eq. (8)](https://arxiv.org/abs/2001.06315) with the matrix function from their
Approach 3. With patch-interior mass and stiffness matrices $M$ and $K$, and
the assembled weak load $b_j$, it computes

$$\chi_j=T\varphi_1(-T M^{-1}K)M^{-1}b_j,\qquad \varphi_1(z)=(e^z-1)/z.$$

SLEPc MFN applies $\varphi_1$ using restarted Arnoldi. PETSc applies $M^{-1}K$
through a stiffness multiplication and a solve with the factored mass matrix.
Patch boundary degrees of freedom are held at zero.
You can extend or change parameter lists under the same `experiment_name`:
existing rows remain in `runs.csv`, only new combinations run, and plots use
completed rows selected by the current YAML. `work/<experiment_name>/config.yaml`
records the latest YAML. Changing the method or problem family requires a new
`experiment_name`.

The runner writes one row at a time to `runs.csv` through a temporary file,
then replaces the CSV. `python run.py plot <config>` imports only the plotting
and data modules, so it can redraw figures without the DOLFINx solver.

`problem.name` accepts `constant`, `nonconstant`, `periodic`, or `random`; the periodic
problem accepts an `epsilon` list. `assembly_type` accepts `petrov_galerkin`
or `galerkin`. Lists form a Cartesian product using the time settings for each mode.

For problems with a known exact solution, errors use the analytical solution;
random problems use the fine FEM solution as their reference. MsFEM H1 errors are broken
norms because reconstruction may jump across coarse-cell interfaces; the jump
is separately reported. Periodic corrector errors are integrated over each
coarse cell against the zero-mean periodic cell correctors, without mean
alignment. The unweighted internal-coarse-interface jumps are saved separately as
`msfem_jump_l2` and `corrector_jump_l2`.
CSV output contains only the current metrics; columns outside the schema are
ignored when reading older files. Completed rows missing either jump are
recomputed on rerun; plotting alone leaves their jump panels marked as unavailable.
No checkpoints are written. The runner also measures coarse and
fine P1 FEM errors and solve times. Plots read only the saved CSV.
Plots are generated only when a sweep has at least two distinct x values.
The runner generates plots after the batch completes. The configuration path
may be written with or without the `.yaml` suffix.
`plot` redraws figures from the completed CSV rows selected by its YAML without
running any solver. H and delta sweeps use blue for elliptic, red for
parabolic, and green for modified correctors. When the YAML selects all three,
the plots use only shared spatial settings and match parabolic and modified
runs at the same final time. The H sweep uses log-log axes with power-of-two H ticks;
darker shades mean larger delta. In the delta sweep, darker shades mean smaller
H, and the color scales are labeled with powers of two. The legend and color
scales are placed outside the plotting area.
Figures are grouped under `work/<experiment_name>/figures/H/`, `delta/`, and
`time/`; the H and delta directories also contain their sweep overviews. The
overviews group the corrector errors and jump before the solution errors and jump. Solution-error panels in H
overviews include a coarse FEM curve and a horizontal fine FEM reference;
delta overviews are separated by fixed H and show both FEM errors as horizontal
references at that H and h. The full metric definitions and periodic benchmark
derivation are in [notes/notes.qmd](notes/notes.qmd) (render from the repository
root with `quarto render notes/notes.qmd`). Open [notes/notes.html](notes/notes.html)
to read the rendered notes. Regenerate the oversampling illustration with
`python -m notes.plot_oversampling`; its default output is `notes/assets/oversampling`.
The `delta/fixed_H/` directory has additional MsFEM error plots with one panel per
coarse resolution. Each panel has its own vertical scale so variation with
delta at fixed H remains visible. Similarly, `H/fixed_delta/` contains MsFEM
error plots with one panel per delta and independent vertical scales, using
log-log axes and power-of-two H ticks. Redrawing removes old generated plots from
the former flat layout.
Time plots are generated only when at least two final times were tested. They
show one panel per H, color by delta, and use dotted horizontal lines for the
corresponding elliptic references. Line style distinguishes parabolic from
modified, and tick labels include N_t where it is specified uniquely for a
parabolic final time. The jump vertical axis
is linear so numerical zero values remain visible. Delta plots keep a linear
horizontal axis because the sweep may include delta=0. Obsolete h-sweep plots
are removed when plots are redrawn.


To publish the notes with GitHub Pages, commit `index.html`, `.nojekyll`,
`notes/notes.html`, `notes/notes.css`, `notes/assets/oversampling.png`, and
`notes/notes_files/`. The `notes_files/` directory is required by the current
HTML output, even though Quarto generates it. Keep `notes/notes.qmd`,
`notes/_math-macros.qmd`, and `notes/plot_oversampling.py` in the repository to
make the notes editable and reproducible. Build caches, intermediate exports,
and the unused SVG figure export are excluded by `.gitignore`.

After pushing, enable **Settings → Pages → Deploy from a branch → main → /(root)**.
The root `index.html` redirects visitors to the notes. Once Pages deployment
succeeds, the site will be available at
[il1yasvie1.github.io/multiscale-methods](https://il1yasvie1.github.io/multiscale-methods/).
For updates, run `quarto render notes/notes.qmd` and commit the updated HTML,
stylesheet, image, and `notes_files/` resources together.
