"""Read and expand the small YAML schema used by batch experiments."""

from hashlib import sha256
from itertools import product
import json
from math import isfinite
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
        "constant", "nonconstant", "periodic", "random"
    ):
        raise ValueError("problem.name must be constant, nonconstant, periodic or random")
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
    random_settings = {}
    if problem["name"] == "random":
        model = problem.get("model", "lognormal")
        micro_size = problem.get("micro_size", 16)
        seed = problem.get("seed", 42)
        length = problem.get("correlation_length", 0.125)
        std = problem.get("log_std", 0.75)
        low, high = problem.get("low", 1.0), problem.get("high", 9.0)
        if (model not in ("lognormal", "checkerboard") or
                type(micro_size) is not int or micro_size < 2 or
                type(seed) is not int or
                not isinstance(length, (int, float)) or length <= 0 or
                not isinstance(std, (int, float)) or std < 0 or
                not isinstance(low, (int, float)) or
                not isinstance(high, (int, float)) or not 0 < low < high):
            raise ValueError("Invalid random problem parameters")
        random_settings = dict(random_model=model, micro_size=micro_size,
                               seed=seed, correlation_length=float(length),
                               log_std=float(std), low=float(low), high=float(high))
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
    modified_settings = config.get("modified_settings", [])
    if not isinstance(modified_settings, list) or any(
        not isinstance(setting, dict) or setting.keys() != {"final_time"}
        or type(setting["final_time"]) not in (int, float)
        or not isfinite(setting["final_time"]) or setting["final_time"] <= 0
        for setting in modified_settings
    ):
        raise ValueError("modified_settings must be a list of mappings with positive final_time")
    settings_by_kind = {"elliptic": [None], "parabolic": settings, "modified": modified_settings}
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
        if random_settings and N_h % random_settings["micro_size"]:
            raise ValueError("random micro_size must divide N_h so material edges align")
        if float(delta) < 0:
            raise ValueError("delta must be nonnegative")
        if assembly not in ("petrov_galerkin", "galerkin"):
            raise ValueError(f"Unknown assembly_type: {assembly}")
        if not isinstance(kind, str) or kind not in settings_by_kind:
            raise ValueError(f"Unknown corrector_kind: {kind}")
        times = settings_by_kind[kind]
        if not times:
            raise ValueError(f"{kind}_settings is required for {kind} runs")
        for setting in times:
            if kind == "parabolic":
                T = setting["final_time"]
                Nt = setting["num_time_steps"]
                if not isfinite(float(T)) or float(T) <= 0 or type(Nt) is not int or Nt <= 0:
                    raise ValueError("final_time and num_time_steps must be positive")
            elif kind == "modified":
                T, Nt = float(setting["final_time"]), None
            else:
                T = Nt = None
            params = dict(
                experiment_name=config["experiment_name"], method="msfem",
                problem=problem["name"], epsilon=epsilon,
                N_H=N_H, N_h=N_h, H=1 / N_H, h=1 / N_h,
                delta=float(delta), assembly_type=assembly,
                corrector_kind=kind, final_time=T, num_time_steps=Nt,
                time_step=None if Nt is None else T / Nt,
            )
            params.update(random_settings)
            encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
            params["run_id"] = sha256(encoded.encode()).hexdigest()[:12]
            yield params
