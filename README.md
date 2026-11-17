# multiscale-methods

Install this package in a Python environment that already provides DOLFINx and
PETSc. The modified corrector also requires SLEPc and `slepc4py`.

```sh
python -m pip install -e ".[experiments]"
```

For the modified corrector, install `.[experiments,modified]` in an environment
with SLEPc as well.

The package imports as `multiscale`; experiments remain outside the installed
package. The current implementation is a two-dimensional oversampling MsFEM.
After a release is published, the distribution can be installed as
`pip install multiscale-methods` in the same DOLFINx environment.

```text
multiscale/             Installable library
  meshes/              Coarse-to-fine mesh relationships
  patches/             Local region selection and submesh construction
  correctors/          Current local corrector solvers
  msfem/               Current oversampling MsFEM and its basis functions
  problems/            Existing elliptic problem data
experiments/msfem/      Batch runs, diagnostics, and plotting
experiments/correctors/ Whole-domain corrector experiments
configs/msfem/          MsFEM experiment YAML files
configs/correctors/     Corrector experiment YAML files
results/                Generated runs and figures, grouped by experiment (ignored by Git)
tests/                  Correctness checks
docs/msfem_docs/        MsFEM mathematical notes and figures
docs/correctors_docs/   Corrector notes
```

The existing MsFEM solver is available as
`from multiscale.msfem import MsFEM_Oversampling`. Local correctors can be
selected independently for each patch:

```python
from multiscale.correctors import create_local_correctors

solver = create_local_correctors(
    patch_mesh,
    ("P", 1),
    problem.coefficient(patch_mesh),
    {"kind": "modified", "final_time": 0.025},
    {"mass": {"ksp_type": "preonly", "pc_type": "lu"}},
)
correctors = solver.solve()
```

`LocalCorrectors` stores the common patch data and selects a concrete class.
Each of `EllipticCorrectors`, `ParabolicCorrectors`, and
`ModifiedEllipticCorrectors` implements its own `solve()`. The current forms
are scalar diffusion forms and return one function per spatial direction; the
caller supplies the scalar element. Solver parameters are method-specific:
elliptic accepts PETSc options directly, parabolic uses
`initial` and `step` KSP option groups, and modified uses `mass` KSP and `mfn`
option groups. `MeshHierarchy`, LOD, and HMM classes remain future work.

Current batch runs replicate the mesh on each MPI rank. Ranks solve different
coarse-cell patches, and rank 0 assembles and solves the coarse system. This
parallel approach still needs redesign before general 1D, 2D, or 3D support.

Every experiment YAML requires `experiment: msfem` or
`experiment: correctors` to select its runner. MsFEM YAML also retains
`method: msfem` for its numerical method setting. The YAML filename (without
`.yaml`) supplies `experiment_name`, which becomes the result directory name
under `results/<experiment>/`.

`solver_parameters` is grouped by corrector kind. Parabolic correctors use
separate `initial` and `step` KSP option mappings; modified correctors use
`mass` and `mfn` mappings. See the
[corrector experiment configuration](experiments/correctors/README.md#solver-diagnostics).

```sh
# Run experiments
python run_experiment.py configs/msfem/periodic-128.yaml
mpiexec -n 2 python run_experiment.py configs/msfem/smoke-test.yaml
python run_experiment.py configs/correctors/periodic-corrector-default-t.yaml

# Redraw experiment plots
python run_experiment.py plot configs/msfem/periodic-128.yaml
python run_experiment.py plot configs/correctors/periodic-corrector-default-t.yaml

# Generate field plots / redraw saved fields
python -m experiments.msfem.plotting.fields
python -m experiments.msfem.plotting.fields --plot-only

# Command-line options
python run_experiment.py --help
python -m experiments.msfem.plotting.fields --help
```

[MsFEM notes](docs/msfem_docs/msfem_docs.html)
