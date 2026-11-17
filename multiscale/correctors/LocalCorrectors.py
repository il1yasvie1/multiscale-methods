import dolfinx.fem as fem
import dolfinx.mesh as mesh
import numpy as np


def _check_corrector_parameters(parameters: dict) -> str:
    kind = parameters.get("kind")
    expected = {
        "elliptic": {"kind"},
        "parabolic": {"kind", "final_time", "num_time_steps"},
        "modified": {"kind", "final_time"},
    }
    if not isinstance(kind, str) or kind not in expected:
        raise ValueError(f"Unknown corrector kind: {kind}")
    if parameters.keys() != expected[kind]:
        raise ValueError(f"Incorrect corrector parameters for {kind}: {sorted(parameters)}")
    if kind in {"parabolic", "modified"}:
        final_time = parameters["final_time"]
        if (
            isinstance(final_time, bool)
            or not isinstance(final_time, (int, float, np.integer, np.floating))
            or not np.isfinite(final_time)
            or final_time <= 0
        ):
            raise ValueError("final_time must be finite and positive")
    if kind in {"parabolic"}:
        steps = parameters["num_time_steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("num_time_steps must be a positive integer")
    return kind


class LocalCorrectors:
    def __init__(
        self,
        mesh: mesh.Mesh,
        element,
        coefficient,
        corrector_parameters: dict | None = None,
        solver_parameters: dict | None = None,
        quadrature_degree: int = 12,
    ):
        self.mesh = mesh
        self.gdim = mesh.geometry.dim
        self.element = element
        self.function_space = fem.functionspace(mesh, element)
        self.coefficient = coefficient
        self.corrector_parameters = dict(corrector_parameters or {})
        self.solver_parameters = dict(solver_parameters or {})
        self.quadrature_degree = quadrature_degree

    def solve(self):
        raise NotImplementedError


def create_local_correctors(
    mesh: mesh.Mesh,
    element,
    coefficient,
    corrector_parameters: dict | None = None,
    solver_parameters: dict | None = None,
    quadrature_degree: int = 12,
) -> LocalCorrectors:
    kind = _check_corrector_parameters(corrector_parameters)
    if kind == "elliptic":
        from .EllipticCorrectors import EllipticCorrectors
        cls = EllipticCorrectors
    elif kind == "parabolic":
        from .ParabolicCorrectors import ParabolicCorrectors
        cls = ParabolicCorrectors
    elif kind == "modified":
        from .ModifiedEllipticCorrectors import ModifiedEllipticCorrectors
        cls = ModifiedEllipticCorrectors
    else:
        raise NotImplementedError

    return cls(
        mesh,
        element,
        coefficient,
        corrector_parameters,
        solver_parameters,
        quadrature_degree,
    )
