"""Multiscale Finite Element Method."""

__all__ = ["MsFEM_Oversampling"]


def __getattr__(name):
    if name == "MsFEM_Oversampling":
        from .msfem import MsFEM_Oversampling
        return MsFEM_Oversampling
    raise AttributeError(name)
