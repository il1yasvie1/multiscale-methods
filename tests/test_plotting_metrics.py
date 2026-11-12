"""FEM reference lines and unweighted overview data, without a solver import."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from msfem.plotting import spatial
from msfem.experiment_data import JUMP_COLUMNS
from msfem.plotting.notation import METRICS, OVERVIEW_METRICS, Y_LABELS


def _table():
    rows = []
    for H in (0.5, 0.25):
        for delta in (0., 1.):
            for kind in ('elliptic', 'parabolic', 'modified'):
                rows.append(dict(
                    problem='periodic', epsilon=0.05, N_h=32,
                    assembly_type='petrov_galerkin', H=H, delta=delta,
                    corrector_kind=kind, final_time=None if kind == 'elliptic' else 0.025,
                    num_time_steps=40 if kind == 'parabolic' else None,
                    **{metric: 0.01 for metric in METRICS},
                    **{f'{base}_{norm}_error': H**2 if base == 'coarse_fem' else 0.001
                       for base in ('coarse_fem', 'fine_fem')
                       for norm in ('l2', 'h1', 'h1_semi')},
                ))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('metric', METRICS)
@pytest.mark.parametrize('sweep', ['H', 'delta'])
def test_fem_references_on_H_and_delta_axes(metric, sweep):
    table = _table()
    fig, ax = plt.subplots()
    frame = table if sweep == 'H' else table.loc[table.H == 0.5]
    spatial._add_fem_references(ax, frame, metric, sweep)
    if sweep == 'H' and metric in ('msfem_l2_error', 'msfem_h1_semi_error'):
        coarse, = ax.lines
        np.testing.assert_allclose(coarse.get_xdata(), [0.25, 0.5])
        np.testing.assert_allclose(coarse.get_ydata(), [0.0625, 0.25])
    else:
        assert not ax.lines
    plt.close(fig)


@pytest.mark.parametrize('missing_jumps', [False, True])
def test_overviews_use_both_jumps_and_only_H_references(monkeypatch, tmp_path, missing_jumps):
    table = _table()
    assert OVERVIEW_METRICS == (
        'corrector_l2_error', 'corrector_h1_semi_error', 'corrector_jump_l2',
        'msfem_l2_error', 'msfem_h1_semi_error', 'msfem_jump_l2',
    )
    if missing_jumps:
        table[list(JUMP_COLUMNS)] = np.nan
    else:
        # Tiny nonzero jumps must survive unchanged, without H scaling or rounding.
        table[list(JUMP_COLUMNS)] = 3e-14
    captured = []
    def inspect_figure(fig, path):
        fig.canvas.draw()  # Also validate every mathtext label and legend.
        axes = fig.axes[:len(OVERVIEW_METRICS)]
        for column in JUMP_COLUMNS:
            ax = axes[OVERVIEW_METRICS.index(column)]
            assert ax.get_yscale() == 'linear'
            if missing_jumps:
                assert not ax.lines
                assert any('Not available' in item.get_text() for item in ax.texts)
            else:
                assert len(ax.lines) == (6 if path.parent.name == 'H' else 3)
                for line in ax.lines:
                    np.testing.assert_allclose(line.get_ydata(), 3e-14, rtol=0, atol=1e-28)
        for column in ('msfem_l2_error', 'msfem_h1_semi_error'):
            ax = axes[OVERVIEW_METRICS.index(column)]
            refs = [line for line in ax.lines if line.get_label().startswith('FEM')]
            expected = [r'FEM $u^H$'] if path.parent.name == 'H' else []
            assert [line.get_label() for line in refs] == expected
        legend_labels = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
        assert (r'FEM $u^H$' in legend_labels) == (path.parent.name == 'H')
        captured.append(path)
        plt.close(fig)
    monkeypatch.setattr(spatial, 'save_figure', inspect_figure)
    assert spatial._overview_sweep(table, 'H', tmp_path, compare_all=True) == 1
    assert spatial._overview_sweep(table, 'delta', tmp_path, compare_all=True) == 2
    assert len(captured) == 3


def test_all_individual_math_labels_render():
    fig, ax = plt.subplots()
    for label in Y_LABELS.values():
        ax.set_ylabel(label)
        fig.canvas.draw()
    plt.close(fig)


def test_metric_symbols_match_notes_csv_mapping():
    """The independent, user-edited notes are the authority for metric notation."""
    import re
    from pathlib import Path
    from msfem.plotting.notation import METRIC_SYMBOLS

    notes = (Path(__file__).resolve().parents[1] / 'notes' / 'notes.qmd').read_text()
    mapping = {column: f'${symbol}$' for symbol, column in re.findall(
        r'^\| \$(.+)\$ \| `(\w+)` \|$', notes, re.MULTILINE)
        if column in METRIC_SYMBOLS}
    assert mapping == METRIC_SYMBOLS


def test_parameter_titles_and_legends_follow_notes():
    from msfem.plotting.common import _add_color_keys, _parameter_title
    from msfem.plotting.notation import FEM_REFERENCE_STYLES, KIND_LABELS

    title = _parameter_title(('epsilon', 'N_h', 'assembly_type'),
                             (0.05, 128, 'petrov_galerkin'), T=0.025, Nt=40)
    assert r'\varepsilon=0.05' in title
    assert 'h=1/128' in title and 'N_h' not in title
    assert 'T=0.025' in title and 'N_t=40' in title
    assert r'\mathrm{PG}' in title
    assert [style['label'] for style in FEM_REFERENCE_STYLES.values()] == [r'FEM $u^H$']
    fig, ax = plt.subplots()
    _add_color_keys(fig, ax, ['elliptic', 'parabolic', 'modified'], [0.5], 'delta')
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert all(r'\alpha=' in label and r'\delta=0.5' in label for label in labels)
    assert all(r'\mathrm{' in label for label in KIND_LABELS.values())
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize('kind,sweep,expected', [
    ('colored', 'H', 8), ('colored', 'delta', 8),
    ('fixed', 'H', 6), ('fixed', 'delta', 6), ('time', 'final_time', 6),
])
def test_sweep_types_share_metric_and_parameter_notation(monkeypatch, tmp_path, kind, sweep, expected):
    from msfem.plotting import time
    from msfem.plotting.notation import AXIS_LABELS

    frame = _table()
    frame.loc[frame.H == 0.25, 'corrector_jump_l2'] = 0.0
    frame.loc[frame.H == 0.5, 'corrector_jump_l2'] = 3e-14
    frame['run_id'] = [str(index) for index in range(len(frame))]
    second = frame.loc[frame.corrector_kind != 'elliptic'].copy()
    second['final_time'] = 0.0125
    second.loc[second.corrector_kind == 'parabolic', 'num_time_steps'] = 20
    second['run_id'] += '_second'
    if kind == 'time':
        frame = pd.concat([frame, second], ignore_index=True)
    figures = []
    def inspect_figure(fig, *args, **kwargs):
        assert all(line.get_label() != r'FEM $u^h$'
                   for ax in fig.axes for line in ax.lines)
        assert all(text.get_text() != r'FEM $u^h$'
                   for legend in fig.legends for text in legend.get_texts())
        title = fig._suptitle.get_text() if fig._suptitle else fig.axes[0].get_title()
        assert 'h=1/32' in title
        metric_label = fig._supylabel.get_text() if fig._supylabel else fig.axes[0].get_ylabel()
        assert metric_label in Y_LABELS.values()
        expect_coarse = sweep == 'H' and metric_label in (
            Y_LABELS['msfem_l2_error'], Y_LABELS['msfem_h1_semi_error'])
        for ax in fig.axes:
            if not ax.get_visible() or not ax.get_xlabel():
                continue
            refs = [line for line in ax.lines if line.get_label() == r'FEM $u^H$']
            assert len(refs) == int(expect_coarse)
            if kind == 'fixed' and metric_label == Y_LABELS['corrector_jump_l2']:
                assert ax.get_yscale() == 'linear'
                assert ax.get_ylim()[0] == 0
                for line in ax.lines:
                    H = line.get_xdata() if sweep == 'H' else (
                        0.5 if '2^{-1}' in ax.get_title() else 0.25)
                    np.testing.assert_allclose(line.get_ydata(),
                                               np.where(np.asarray(H) == 0.5, 3e-14, 0),
                                               rtol=0, atol=1e-28)
        assert all(ax.get_xlabel() == AXIS_LABELS[sweep] for ax in fig.axes if ax.get_xlabel())
        fig.canvas.draw()
        figures.append(metric_label)
        plt.close(fig)
    monkeypatch.setattr(spatial, 'save_sweep_figure', inspect_figure)
    monkeypatch.setattr(time, 'save_figure', inspect_figure)
    if kind == 'colored':
        assert spatial._colored_sweep(frame, sweep, tmp_path, compare_all=True) == expected
    elif kind == 'fixed':
        assert spatial._fixed_parameter_sweep(frame, sweep, tmp_path, compare_all=True) == expected
    else:
        assert time._time_sweep(frame, tmp_path, compare_all=True) == expected
    assert len(figures) == expected
    if kind == 'fixed':
        assert set(figures) == {Y_LABELS[metric] for metric in (
            'msfem_l2_error', 'msfem_h1_error', 'msfem_h1_semi_error',
            'corrector_l2_error', 'corrector_h1_semi_error', 'corrector_jump_l2')}
