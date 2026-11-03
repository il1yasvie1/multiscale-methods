"""Read and expand the small YAML schema used by batch experiments."""

from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment YAML must contain a mapping")
    name = config.get("experiment_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("experiment_name must contain only letters, digits, '_' or '-'")
    if config.get("method") != "msfem":
        raise ValueError("Only method: msfem is currently supported")
    problem = config.get("problem")
    if not isinstance(problem, dict) or problem.get("name") not in (
        "constant", "nonconstant", "periodic"
    ):
        raise ValueError("problem.name must be constant, nonconstant or periodic")
    for key in ("N_H", "N_h"):
        if key not in config.get("mesh", {}):
            raise ValueError(f"mesh.{key} is required")
    return config


def validate_snapshot(previous: dict, current: dict) -> None:
    """Keep one experiment directory tied to one method and problem family."""
    if (previous.get("method") != current["method"] or
            previous.get("problem", {}).get("name") != current["problem"]["name"]):
        raise ValueError(
            "An existing experiment directory uses a different method or problem; "
            "choose a new experiment_name"
        )


def _choices(value, label: str) -> list:
    values = value if isinstance(value, list) else [value]
    if not values or any(item is None for item in values):
        raise ValueError(f"{label} must be a nonempty value or list")
    return values


def expand_runs(config: dict):
    problem = config["problem"]
    epsilons = _choices(problem["epsilon"], "problem.epsilon") if "epsilon" in problem else [None]
    if problem["name"] != "periodic" and epsilons != [None]:
        raise ValueError("epsilon is only supported for the periodic problem")
    if problem["name"] == "periodic" and any(
        value is None or float(value) <= 0 for value in epsilons
    ):
        raise ValueError("periodic problems require positive epsilon")
    mesh = config["mesh"]
    settings = config.get("parabolic_settings", [])
    if not isinstance(settings, list) or any(
        not isinstance(setting, dict) or
        not {"final_time", "num_time_steps"} <= setting.keys()
        for setting in settings
    ):
        raise ValueError(
            "parabolic_settings must be a list of mappings with "
            "final_time and num_time_steps"
        )
    for epsilon, N_H, N_h, delta, assembly, kind in product(
        epsilons,
        _choices(mesh["N_H"], "mesh.N_H"),
        _choices(mesh["N_h"], "mesh.N_h"),
        _choices(config["delta"], "delta"),
        _choices(config.get("assembly_type", "petrov_galerkin"), "assembly_type"),
        _choices(config.get("corrector_kind", "elliptic"), "corrector_kind"),
    ):
        if type(N_H) is not int or type(N_h) is not int or N_H <= 0 or N_h < N_H:
            raise ValueError("N_H and N_h must be positive integers with N_h >= N_H")
        ratio = N_h // N_H
        if N_h % N_H or ratio & (ratio - 1):
            raise ValueError("N_h/N_H must be a power of two")
        if float(delta) < 0:
            raise ValueError("delta must be nonnegative")
        if assembly not in ("petrov_galerkin", "galerkin"):
            raise ValueError(f"Unknown assembly_type: {assembly}")
        if kind not in ("elliptic", "parabolic"):
            raise ValueError(f"Unknown corrector_kind: {kind}")
        times = settings if kind == "parabolic" else [None]
        if not times:
            raise ValueError("parabolic_settings is required for parabolic runs")
        for setting in times:
            if setting is not None:
                T = setting["final_time"]
                Nt = setting["num_time_steps"]
                if float(T) <= 0 or type(Nt) is not int or Nt <= 0:
                    raise ValueError("final_time and num_time_steps must be positive")
            else:
                T = Nt = None
            params = dict(
                experiment_name=config["experiment_name"], method="msfem",
                problem=problem["name"], epsilon=epsilon,
                N_H=N_H, N_h=N_h, H=1 / N_H, h=1 / N_h,
                delta=float(delta), assembly_type=assembly,
                corrector_kind=kind, final_time=T, num_time_steps=Nt,
                time_step=None if T is None else T / Nt,
            )
            encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
            params["run_id"] = sha256(encoded.encode()).hexdigest()[:12]
            yield params
