"""Configuration and run expansion for whole-domain corrector experiments."""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np
import yaml

from experiments.solver_config import solver_parameters as _solver_parameters


KINDS = ("elliptic", "parabolic", "modified")
ALPHA = 1 / (16 * math.pi**2)
BETA = 1 / (4 * math.pi**2)


def _linspace(spec: dict, label: str) -> list[float]:
    if not isinstance(spec, dict) or not {"start", "stop", "num"} <= spec.keys():
        raise ValueError(f"{label} needs start, stop, and num")
    start, stop, num = spec["start"], spec["stop"], spec["num"]
    if (type(num) is not int or num < 1 or
            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, stop)) or
            start <= 0 or stop < start or (num == 1 and start != stop)):
        raise ValueError(f"Invalid {label} linspace")
    return [float(v) for v in np.linspace(start, stop, num)]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment YAML must be a mapping")
    name = config.get("experiment_name", path.stem)
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid experiment_name")
    if name != path.stem:
        raise ValueError("experiment_name must match the YAML filename")
    config["experiment_name"] = name
    if config.get("experiment") != "correctors":
        raise ValueError("experiment must be 'correctors' for experiments/correctors")
    problem = config.get("problem", {})
    if problem.get("name") != "periodic" or problem.get("epsilon") != 1.0:
        raise ValueError("This experiment requires the periodic problem with epsilon: 1.0")
    _linspace(config.get("R"), "R")
    delta = config.get("delta")
    if (type(delta) not in (int, float) or not math.isfinite(delta)
            or delta < 0):
        raise ValueError("delta must be finite and nonnegative")
    ppu = config.get("mesh", {}).get("points_per_unit")
    if type(ppu) is not int or ppu < 2:
        raise ValueError("mesh.points_per_unit must be an integer >= 2")
    kinds = config.get("corrector_kind")
    if not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds) or any(k not in KINDS for k in kinds):
        raise ValueError("corrector_kind must be a nonempty subset of elliptic, parabolic, modified")
    time = config.get("T")
    if time is None:
        if any(kind != "elliptic" for kind in kinds):
            raise ValueError("T needs a linspace or default_setting: true")
    else:
        if not isinstance(time, dict) or not time.get("default_setting", False) and not {"start", "stop", "num"} <= time.keys():
            raise ValueError("T needs a linspace or default_setting: true")
        if {"start", "stop", "num"} <= time.keys():
            _linspace(time, "T")
        if (time.get("default_setting", False) and delta == 0
                and any(kind != "elliptic" for kind in kinds)):
            raise ValueError("default T is zero when delta = 0")
    if "parabolic" in kinds:
        max_dt = config.get("parabolic_settings", {}).get("max_time_step")
        if not isinstance(max_dt, (int, float)) or not math.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("parabolic_settings.max_time_step must be positive")
    solver_parameters(config)
    return config


def solver_parameters(config: dict) -> dict:
    return _solver_parameters(config, default_modified_mass_lumped=True)


def core_size(R: float, delta: float) -> float:
    """Return L when K_R extends K_L by delta*L on every side."""
    return R / (1 + 2 * delta)


def default_time(R: float, delta: float) -> float:
    L = core_size(R, delta)
    return (R - L) / (2 * math.pi * math.sqrt(ALPHA * BETA))


def mesh_intervals(R: float, points_per_unit: int, delta: float) -> int:
    """Choose a near-uniform resolution and align common rational K_L boundaries."""
    target = max(math.ceil(points_per_unit * R), math.ceil(1 + 2 * delta))
    rational_delta = Fraction(str(delta)).limit_denominator(1000)
    if not math.isclose(float(rational_delta), delta, rel_tol=0, abs_tol=1e-12):
        return target
    ratio = 1 + 2 * rational_delta
    left_edge = (ratio - 1) / (2 * ratio)
    multiple = left_edge.denominator
    # Avoid a pathological mesh size for a highly resolved decimal delta.
    if multiple > 120:
        return target
    return multiple * math.ceil(target / multiple)


def expand_runs(config: dict) -> list[dict]:
    grouped_options = solver_parameters(config)
    radii = _linspace(config["R"], "R")
    kinds = config["corrector_kind"]
    has_timed_method = any(kind != "elliptic" for kind in kinds)
    time_spec = config.get("T") or {}
    grid_times = (_linspace(time_spec, "T")
                  if has_timed_method and {"start", "stop", "num"} <= time_spec.keys()
                  else [])
    delta = float(config["delta"])
    ppu = config["mesh"]["points_per_unit"]
    max_dt = config.get("parabolic_settings", {}).get("max_time_step")
    runs = []
    for R in radii:
        L = core_size(R, delta)
        n = mesh_intervals(R, ppu, delta)
        T_default = default_time(R, delta) if has_timed_method else None
        times = [(T, "grid") for T in grid_times]
        if has_timed_method and time_spec.get("default_setting", False):
            match = next((i for i, (T, _) in enumerate(times) if math.isclose(T, T_default, rel_tol=1e-12)), None)
            if match is None:
                times.append((T_default, "default"))
            else:
                times[match] = (times[match][0], "grid_default")
        for kind in kinds:
            method_times = [(None, "none")] if kind == "elliptic" else times
            for T, time_source in method_times:
                Nt = math.ceil(T / max_dt) if kind == "parabolic" else None
                params = dict(
                    experiment_name=config["experiment_name"], problem="periodic", epsilon=1.0,
                    R=R, L=L, delta=delta, n=n, h=R / n, points_per_unit=ppu,
                    corrector_kind=kind, T=T, T_default=T_default,
                    time_source=time_source, num_time_steps=Nt,
                    time_step=None if Nt is None else T / Nt,
                )
                if kind == "modified":
                    params["mass_lumped"] = grouped_options.get("modified", {}).get("mass_lumped", True)
                encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
                params["run_id"] = sha256(encoded.encode()).hexdigest()[:12]
                runs.append(params)
    return runs
