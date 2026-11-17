"""Reusable multiscale components built on DOLFINx."""

from importlib import import_module

__all__ = [
    "MsFEM_Oversampling",
    "oversampling",
    "refine_with_meshtags",
    "LocalCorrectors",
    "EllipticCorrectors",
    "ParabolicCorrectors",
    "ModifiedEllipticCorrectors",
    "create_local_correctors",
]

_EXPORTS = {
    "MsFEM_Oversampling": (".msfem", "MsFEM_Oversampling"),
    "oversampling": (".patches", "oversampling"),
    "refine_with_meshtags": (".meshes", "refine_with_meshtags"),
    "LocalCorrectors": (".correctors", "LocalCorrectors"),
    "EllipticCorrectors": (".correctors", "EllipticCorrectors"),
    "ParabolicCorrectors": (".correctors", "ParabolicCorrectors"),
    "ModifiedEllipticCorrectors": (".correctors", "ModifiedEllipticCorrectors"),
    "create_local_correctors": (".correctors", "create_local_correctors"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
