"""Validate method-specific solver settings without importing PDE libraries."""

from math import isfinite


KINDS = {"elliptic", "parabolic", "modified"}
KSP_DIAGNOSTICS = {"ksp_view", "ksp_monitor"}
MFN_DIAGNOSTICS = {"mfn_view", "mfn_monitor"}
_DEFAULT_KSP = {"ksp_type": "preonly", "pc_type": "lu"}


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _ksp_options(value, label: str) -> dict:
    options = {}
    for key, setting in _mapping(value, label).items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} needs nonempty PETSc option names")
        if key in KSP_DIAGNOSTICS:
            if type(setting) is not bool:
                raise ValueError(f"{label}.{key} must be boolean")
            if setting:
                # PETSc viewer flags need an empty option value, not "true".
                options[key] = None
        elif key == "ksp_error_if_not_converged":
            raise ValueError(f"{label}.{key} is always enabled by the solver")
        elif type(setting) in (str, int, bool) or (
            type(setting) is float and isfinite(setting)
        ):
            options[key] = setting
        else:
            raise ValueError(f"{label}.{key} must be a finite PETSc option value")
    return options


def _mfn_options(value, label: str) -> dict:
    options = {}
    for key, setting in _mapping(value, label).items():
        if key in MFN_DIAGNOSTICS:
            if type(setting) is not bool:
                raise ValueError(f"{label}.{key} must be boolean")
            if setting:
                options[key] = None
        elif key == "type" and isinstance(setting, str) and setting:
            options[key] = setting
        elif key == "tol" and type(setting) in (int, float) and isfinite(setting) and setting > 0:
            options[key] = float(setting)
        else:
            raise ValueError(f"{label}.{key} must be a supported MFN setting")
    return options


def solver_parameters(config: dict, *, default_modified_mass_lumped: bool) -> dict:
    """Return solver options grouped by corrector kind, with viewer flags normalized."""
    raw = _mapping(config.get("solver_parameters", {}), "solver_parameters")
    unknown = raw.keys() - KINDS
    if unknown:
        raise ValueError(f"Unknown solver_parameters groups: {sorted(unknown)}")

    normalized = {}
    if "elliptic" in raw:
        normalized["elliptic"] = _ksp_options(raw["elliptic"], "solver_parameters.elliptic")
    if "parabolic" in raw:
        parabolic = _mapping(raw["parabolic"], "solver_parameters.parabolic")
        unknown = parabolic.keys() - {"initial", "step"}
        if unknown:
            raise ValueError(f"Unknown parabolic solver groups: {sorted(unknown)}")
        normalized["parabolic"] = {
            name: _ksp_options(value, f"solver_parameters.parabolic.{name}")
            for name, value in parabolic.items()
        }
    if "modified" in raw:
        modified = _mapping(raw["modified"], "solver_parameters.modified")
        unknown = modified.keys() - {"mass", "mfn", "mass_lumped"}
        if unknown:
            raise ValueError(f"Unknown modified solver groups: {sorted(unknown)}")
        mass_lumped = modified.get("mass_lumped", default_modified_mass_lumped)
        if type(mass_lumped) is not bool:
            raise ValueError("solver_parameters.modified.mass_lumped must be boolean")
        mass = _ksp_options(modified.get("mass", {}), "solver_parameters.modified.mass")
        if mass_lumped and mass:
            raise ValueError("modified.mass options need mass_lumped: false")
        normalized["modified"] = {"mass_lumped": mass_lumped}
        if "mass" in modified:
            normalized["modified"]["mass"] = mass
        if "mfn" in modified:
            normalized["modified"]["mfn"] = _mfn_options(
                modified["mfn"], "solver_parameters.modified.mfn"
            )
    return normalized


def options_for_kind(grouped: dict, kind: str, *, default_modified_mass_lumped: bool) -> dict:
    """Select the options passed to one LocalCorrectors implementation."""
    selected = dict(grouped.get(kind, {}))
    if kind == "modified":
        selected.setdefault("mass_lumped", default_modified_mass_lumped)
    return selected


def numerical_solver_settings(config: dict, *, default_modified_mass_lumped: bool,
                              legacy_snapshot: bool = False) -> dict:
    """Changes here need a new experiment name; diagnostic flags do not."""
    raw = config.get("solver_parameters", {})
    if legacy_snapshot and isinstance(raw, dict) and raw.keys() <= (
        KSP_DIAGNOSTICS | MFN_DIAGNOSTICS
    ):
        return {}
    grouped = solver_parameters(config, default_modified_mass_lumped=default_modified_mass_lumped)

    def ksp_changes(options: dict) -> dict:
        changes = {key: value for key, value in options.items() if key not in KSP_DIAGNOSTICS}
        for key, default in _DEFAULT_KSP.items():
            if changes.get(key) == default:
                changes.pop(key)
        return changes

    changes = {}
    if "elliptic" in grouped:
        elliptic = ksp_changes(grouped["elliptic"])
        if elliptic:
            changes["elliptic"] = elliptic
    if "parabolic" in grouped:
        parabolic = {name: ksp_changes(options)
                     for name, options in grouped["parabolic"].items()}
        parabolic = {name: options for name, options in parabolic.items() if options}
        if parabolic:
            changes["parabolic"] = parabolic
    if "modified" in grouped:
        modified = grouped["modified"]
        mod_changes = {}
        if modified["mass_lumped"] != default_modified_mass_lumped:
            mod_changes["mass_lumped"] = modified["mass_lumped"]
        if "mass" in modified:
            mass = ksp_changes(modified["mass"])
            if mass:
                mod_changes["mass"] = mass
        if "mfn" in modified:
            mfn = {key: value for key, value in modified["mfn"].items()
                   if key not in MFN_DIAGNOSTICS}
            if mfn.get("type") == "krylov":
                mfn.pop("type")
            if mfn:
                mod_changes["mfn"] = mfn
        if mod_changes:
            changes["modified"] = mod_changes
    return changes
