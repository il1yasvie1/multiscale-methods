# Codex Project Notebooks

## Before every commit

- [ ] Inspect the working tree. Confirm that additions, moves, and deletions are intentional, especially while the old `src/`, `msfem/`, `run.py`, and `tests/` paths are being reorganized.
- [ ] **Regenerate the repository tree below from the actual working tree before committing.** Include every project file, including new untracked files. Update directory and file names, provenance labels, progress, last local edit dates, and the next task. Keep ignored caches, build products, and experiment results in the separate generated-artifacts note.
- [ ] Preserve the provenance boundary: `multiscale/` is **human-verified** by the project owner; `experiments/` is **AI-generated**. Other paths stay unclassified until the owner assigns them. Do not silently promote an AI-generated file. Record the human review that justifies each promotion. The metrics code is a planned first promotion, not an accomplished one.
- [ ] Check implementation names and mathematical notation against [`docs/msfem/msfem.qmd`](msfem/msfem.qmd). Update the notes when an implementation decision changes their definitions.
- [ ] Check that the status and “Next” entries below describe the code being committed. Run checks relevant to the changed code and record any missing coverage. The current working tree does not contain a `tests/` directory.
- [ ] Review the final diff and commit only the intended project files.

## Status key and update rule

**Provenance:** `HV` = human-verified by the project owner; `AI` = AI-generated and awaiting human verification; `U` = not yet classified; `GEN` = generated artifact or third-party runtime. A directory's provenance applies to its listed descendants unless a file says otherwise. `HV` records review provenance; it is not a mathematical proof or a claim that every test passes.

**Progress:** `implemented` = present in the current working tree; `refactor` = present but scheduled for structural change; `next` = the immediate development focus; `planned` = an intended component, not yet present. Dates in the tree are **last local file-edit dates**, not verification dates. This inventory was reviewed on **2026-11-17**; refresh the dates and inventory before each commit.

The corrector interface and boundary helper changed on 2026-11-17 are awaiting owner review. Their presence under the previously HV `multiscale/` directory does not promote these new edits to HV provenance. `LocalCorrectors.py` is the exported implementation, combining the owner-specified constructor fields with AI-written methods and selection logic.

<details open>
<summary><strong>Repository tree</strong> — expand a directory for its complete project-file inventory</summary>

```text
multiscale-methods/
├── .gitignore                         [U]
├── README.md                          [U]
├── pyproject.toml                     [U]
├── run_experiment.py                  [U; experiment entry point]
├── configs/                           [U; experiment configuration]
├── experiments/                       [AI; experiment code]
├── multiscale/                        [HV; core library]
└── docs/                              [U; experiment and mathematical notes]
```

<details>
<summary><code>multiscale/</code> — HV · implemented, with refactors planned · latest local edit 2026-11-16</summary>

```text
multiscale/                              [HV]
├── __init__.py                         [HV; public exports]
├── fem_utils.py                        [HV baseline; AI boundary-helper edit pending review]
├── correctors/                        [HV; interface added; 2026-11-17]
│   ├── __init__.py
│   ├── LocalCorrectors.py              [U owner draft + AI interface; review pending]
│   ├── EllipticCorrectors.py           [AI; elliptic solve; review pending]
│   ├── ParabolicCorrectors.py          [AI; parabolic solve; review pending]
│   └── ModifiedEllipticCorrectors.py   [AI; modified solve; review pending]
├── meshes/                            [HV; refactor planned; 2026-11-16]
│   ├── __init__.py
│   └── refinement.py                  [current two-level refinement and tags]
├── msfem/                             [HV; refactor planned; 2026-11-16]
│   ├── __init__.py
│   ├── basis.py                       [affine extension and multiscale basis]
│   └── method.py                      [HV baseline; AI corrector-interface edit pending review]
├── patches/                           [HV; implemented; 2026-11-16]
│   ├── __init__.py
│   └── oversampling.py                [geometric, barycentric patch rule]
└── problems/                          [HV; implemented; 2026-11-16]
    ├── __init__.py
    └── elliptic.py                    [problem data and coefficients]
```

</details>

<details>
<summary><code>experiments/</code> — AI · implemented, human review pending · latest local edit 2026-11-17</summary>

```text
experiments/                            [AI]
├── __init__.py
└── msfem/
    ├── __init__.py
    ├── experiment.py
    ├── experiment_config.py
    ├── experiment_data.py
    ├── experiment_metrics.py          [AI; planned human review/promotion]
    └── plotting/
        ├── __init__.py
        ├── common.py
        ├── fields.py
        ├── files.py
        ├── notation.py
        ├── spatial.py
        └── time.py
```

</details>

<details>
<summary><code>configs/</code> — U · experiment inputs · latest local edit 2026-11-17</summary>

```text
configs/                               [U]
├── correctors/
│   ├── periodic-corrector-default-t.yaml
│   ├── periodic-corrector-smoke.yaml
│   └── periodic-corrector-solver-debug.yaml
└── msfem/
    ├── periodic-128.yaml
    ├── periodic-64.yaml
    ├── periodic-delta.yaml
    ├── random-checkerboard-time-sweep.yaml
    ├── random-checkerboard.yaml
    ├── random-lognormal-time-sweep.yaml
    ├── random-lognormal.yaml
    └── smoke-test.yaml
```

</details>

<details>
<summary><code>docs/</code> — U source · latest local edit 2026-11-18</summary>

```text
docs/
├── CODEX_PROJECT_NOTEBOOKS.md         [U; notebook]
└── msfem/
    ├── msfem.qmd                       [U; primary notation/algorithm source]
    ├── notes.css                       [U]
    ├── plot_oversampling.py           [U; figure generator]
    ├── assets/
    │   └── oversampling.png           [U; included figure]
```

</details>

</details>

Ignored local artifacts, outside the project-file tree: `.git/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `build/`, `multiscale_methods.egg-info/`, `results/`, and the ignored optional `docs/msfem/assets/oversampling.svg`. Recheck this list when `.gitignore` changes. The old `src/`, top-level `msfem/`, `run.py`, and `tests/` paths are currently absent from the working tree; their tracked deletions and replacements still need review before a commit.

## Conventions and notation

Use [`docs/msfem/msfem.qmd`](msfem/msfem.qmd) as the current MsFEM notation source. Keep this notebook focused on the core method; the detailed experimental metrics, plots, and CSV schema remain in the documentation and `experiments/`.

| Symbol or term | Meaning in this project |
| --- | --- |
| $\Omega$, $a^\varepsilon$, $u^\varepsilon$ | Domain, oscillatory coefficient, and elliptic solution. |
| $\mathcal T^H$, $\mathcal T^h$; $H$, $h$ | Coarse and refined fine meshes; mesh resolutions. |
| $K$, $S_K^\delta$, $\delta$ | Coarse cell, its oversampling patch, and barycentric expansion parameter. |
| $V^H$, $V^h$, $\mathring V^h(S_K^\delta)$ | Continuous coarse/fine $P_1$ spaces and the local zero-boundary subspace. |
| $\alpha\in\{\mathrm{ell},\mathrm{par},\mathrm{mod}\}$ | Elliptic, parabolic, or modified corrector choice. |
| $\chi_{j,K}^{h,\alpha}$ | Direction-$j$ local corrector on the patch. |
| $\widetilde\Phi^H_{l,K}$, $\Phi_{l,K}^{\mathrm{MsFEM},\alpha}$ | Affine extension of the coarse basis and corrected multiscale basis. |
| $T$, $N_t$ | Terminal time and time-step count where required by the corrector. |

Use **coarse cell** for $K$, **patch** for $S_K^\delta$, **trial** for corrected multiscale functions, and **test** for ordinary coarse $P_1$ functions in the Petrov–Galerkin path. The output is **nonconforming** because cellwise reconstructed trial functions can jump across coarse-cell boundaries; the reconstruction need not belong to $H^1(\Omega)$, and its gradient is interpreted elementwise. **Petrov–Galerkin** means the trial and test spaces differ: the local matrix uses a corrected trial basis and an affine coarse test basis, integrated over the core cell $K$. The patch is where the corrector is solved. In the notation of the notes, the global formulation is

$$
\sum_{K\in\mathcal T^H}\int_K a^\varepsilon\nabla u^{\mathrm{MsFEM},\alpha}\cdot\nabla v^H\,\mathrm dx
=\int_\Omega f v^H\,\mathrm dx\qquad\text{for every }v^H\in\mathring V^H.
$$

## Implementation notebook: one step at a time

1. **Two-level mesh manager — planned.** Today `refine_with_meshtags` and `MsFEM_Oversampling` hold the coarse/fine relationship and perform mapping operations directly. Introduce an initial two-level meshes manager that owns the coarse mesh, fine mesh, refinement tags, and queries such as “find fine cells in coarse cell $K$,” fine-to-coarse/patch cell maps, and relevant DOF lookups. A possible later name is `TwoLevelMeshHierarchy`; choose the final singular/plural form after the interface is clear. Move these responsibilities out of the MsFEM class incrementally.
2. **MsFEM method — implemented, later generalization.** `MsFEM_Oversampling` currently coordinates patch construction, correctors, basis construction, core-cell assembly, and reconstruction. A later, more general `MultiscaleFiniteElementMethod` should expose reusable stages and methods, with the **corrector formulation** remaining the organizing idea. Preserve the working Strategy 2 path while extracting common responsibilities.
3. **Oversampling — implemented.** `multiscale/patches/oversampling.py` currently selects fine cells by affine barycentric-coordinate inequalities, a **geometric** rule tied to the simplex and mesh geometry. A separate **topological** rule used elsewhere expands by fine-mesh adjacency layers; it is not this implementation. Keep patch selection in `patches/` so later methods can use other policies. Possible future consumers include localized LOD correctors, GMsFEM neighborhood construction, patch diagnostics, and Schwarz-type local solvers; each needs its own patch semantics and basis logic.
4. **Correctors — interface implemented.** `multiscale/correctors/LocalCorrectors.py` stores the shared patch inputs and selects one of three direct subclasses. `EllipticCorrectors.py`, `ParabolicCorrectors.py`, and `ModifiedEllipticCorrectors.py` each implement their own `solve()` and configure the solvers they need. They accept a patch mesh, element, coefficient, formulation parameters, solver parameters, and quadrature degree; `solve()` returns directional functions. `create_local_correctors` owns kind validation, while the concrete classes assume a compatible scalar element. `MsFEM_Oversampling` uses the selector. Keep method-specific parameters explicit: no terminal time for elliptic, $(T,N_t)$ for parabolic, and $T$ for modified. All three corrector kinds and all three MsFEM offline paths were smoke-tested; after replacing the explicit parabolic matrices with `LinearProblem`, the direct parabolic corrector and its MsFEM offline path were tested again. Full experiment coverage remains to be checked.
5. **Metrics promotion — planned.** `experiments/msfem/experiment_metrics.py` remains `AI`. Review its definitions and calculations against `docs/msfem/msfem.qmd`, then move or expose verified metric functionality in an appropriate human-verified location. Update both the tree and provenance record only after that review.

The current approach follows **Oversampling Strategy 2** in Henning and Peterseim, [*Oversampling for the Multiscale Finite Element Method*](https://doi.org/10.1137/120900332): solve directional patch correctors, combine them with coarse gradients, restrict the corrected basis to each coarse cell, and use a coarse test space in the PG formulation. This is an implementation reference, not a claim that the paper's separate constrained oversampling results apply to the present method.

## Next development focus

**Correctors.** The interface, per-kind `solve()` methods, and basic solver-option groups are in place. Reproduction of the Abdulle et al. effective-tensor experiments additionally requires filtered flux averaging; the Arjmand–Marttala whole-space experiments require a separate source-driven local problem. Keep the tree, provenance labels, and this “Next” entry current.
