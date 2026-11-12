"""Analytic trace integrals, interface selection, and legacy CSV regression."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from msfem.experiment_data import (
    COLUMNS, JUMP_COLUMNS, completed_run_ids, load_results, save_result,
)
from msfem.experiment_metrics import (
    _facet_l2_squared, interface_jumps,
)


def test_linear_trace_integral():
    # q(s)=1+2s on a length-three edge: integral = 3*13/3 = 13.
    points = np.array([[0., 0., 0.], [0., 3., 0.]])
    assert _facet_l2_squared(points, np.array([1., 3.])) == pytest.approx(13.)
    assert _facet_l2_squared(points, np.array([[1., 2.], [3., 2.]])) == pytest.approx(25.)
    # Complex traces and a triangular face are supported by the same mass formula.
    assert _facet_l2_squared(points, np.array([1j, 3j])) == pytest.approx(13.)
    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    assert _facet_l2_squared(triangle, np.ones(3)) == pytest.approx(0.5)


def test_interface_jumps_exclude_external_and_intra_coarse_facets():
    # Two coarse triangles, each bisected at the midpoint of their shared diagonal.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                         [0., 1., 0.], [0.5, 0.5, 0.]])
    cells = [[0, 1, 4], [1, 2, 4], [0, 4, 3], [4, 2, 3]]
    coordinates = vertices[np.array(cells)].reshape(-1, 3)
    facets = {}
    for cell, nodes in enumerate(cells):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            facets.setdefault(tuple(sorted((nodes[i], nodes[j]))), []).append(cell)
    neighbors = list(facets.values())
    solution_values = np.repeat([1., 1., 0., 0.], 3)
    correctors = np.vstack([np.tile([1., 2.], (6, 1)), np.zeros((6, 2))])
    data = {
        k: SimpleNamespace(fine_dofs=np.arange(6*k, 6*k+6),
                           corrector_values=correctors[6*k:6*k+6])
        for k in range(2)
    }
    topology = SimpleNamespace(
        dim=2, create_connectivity=lambda *_: None,
        connectivity=lambda *_: SimpleNamespace(links=lambda i: neighbors[i]),
        index_map=lambda dim: SimpleNamespace(size_local=4 if dim == 2 else len(neighbors)),
    )
    V = SimpleNamespace(
        dofmap=SimpleNamespace(index_map=SimpleNamespace(size_local=12),
                              cell_dofs=lambda c: np.arange(3*c, 3*c+3)),
        tabulate_dof_coordinates=lambda: coordinates,
    )
    method = SimpleNamespace(
        comm=SimpleNamespace(rank=0, gather=lambda local, root: [local]),
        fine_mesh=SimpleNamespace(topology=topology, geometry=SimpleNamespace(dim=2)),
        V_multiscale=V, cell_basis_data=data,
        coarse_to_fine_meshtag=SimpleNamespace(indices=np.arange(4), values=np.array([0, 0, 1, 1])),
    )
    solution = SimpleNamespace(x=SimpleNamespace(array=solution_values))
    result = interface_jumps(method, solution)
    assert result['msfem_jump_l2'] == pytest.approx(2**0.25)
    assert result['corrector_jump_l2'] == pytest.approx(np.sqrt(5)*2**0.25)
    assert set(result) == set(JUMP_COLUMNS)


def test_legacy_weighted_jump_is_not_reused_and_requires_recompute(tmp_path):
    path = tmp_path / 'runs.csv'
    pd.DataFrame([dict(run_id='abc', status='completed', msfem_jump=17.)]).to_csv(path, index=False)
    table = load_results(path)
    assert not completed_run_ids(table)
    assert table[list(JUMP_COLUMNS)].isna().all().all()
    row = dict(run_id='abc', status='completed', **dict.fromkeys(JUMP_COLUMNS, 0.5))
    updated = save_result(path, table, row)
    assert len(JUMP_COLUMNS) == 2
    assert completed_run_ids(updated) == {'abc'}
    assert "msfem_jump" not in updated.columns
    assert "msfem_jump" not in pd.read_csv(path).columns
    assert list(pd.read_csv(path).columns) == list(COLUMNS)
