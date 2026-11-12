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

[il1yasvie1.github.io/multiscale-methods](https://il1yasvie1.github.io/multiscale-methods/).
