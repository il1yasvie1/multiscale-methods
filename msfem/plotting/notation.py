"""Plot notation matching the definitions and CSV mapping in notes/notes.qmd."""

import math


# Keep the canonical TeX spelling used in the notes. Matplotlib's mathtext
# supports a smaller delimiter vocabulary; translate only at the display boundary.
NORM_NOTATIONS = {
    "l2": r"\lVert\cdot\rVert_{L^2}",
    "h1": r"\lVert\cdot\rVert_{H^1}",
    "h1_semi": r"\lvert\cdot\rvert_{H^1}",
}
NORM_NAMES = {"l2": r"$L^2$ error", "h1": r"$H^1$ error",
              "h1_semi": r"$H^1$ seminorm error"}
METRIC_SYMBOLS = {}
METRIC_NAMES = {}
for prefix, symbol, name in (("corrector", r"\chi", "Corrector"),
                             ("msfem", "u", "MsFEM solution")):
    for norm, notation in NORM_NOTATIONS.items():
        metric = f"{prefix}_{norm}_error"
        METRIC_SYMBOLS[metric] = rf"$E^\alpha_{{{symbol},{notation}}}$"
        METRIC_NAMES[metric] = f"{name} {NORM_NAMES[norm]}"
    metric = f"{prefix}_jump_l2"
    METRIC_SYMBOLS[metric] = rf"$J_{symbol}^\alpha$"
    METRIC_NAMES[metric] = f"{name} jump"

CORRECTOR_METRICS = tuple(metric for metric in METRIC_SYMBOLS if metric.startswith("corrector_"))
SOLUTION_METRICS = tuple(metric for metric in METRIC_SYMBOLS if metric.startswith("msfem_"))
SOLUTION_ERROR_METRICS = tuple(metric for metric in SOLUTION_METRICS if metric.endswith("_error"))
METRICS = (*CORRECTOR_METRICS, *SOLUTION_METRICS)
OVERVIEW_METRICS = tuple(metric for metric in METRICS if not metric.endswith("_h1_error"))


def _mathtext_label(tex):
    for canonical, supported in ((r"\lVert", r"\|"), (r"\rVert", r"\|"),
                                  (r"\lvert", "|"), (r"\rvert", "|")):
        tex = tex.replace(canonical, supported)
    return tex


Y_LABELS = {metric: _mathtext_label(symbol) for metric, symbol in METRIC_SYMBOLS.items()}
OVERVIEW_TITLES = {metric: METRIC_NAMES[metric] + "\n" + Y_LABELS[metric]
                   for metric in OVERVIEW_METRICS}

KIND_LABELS = {"elliptic": r"$\alpha=\mathrm{ell}$",
               "parabolic": r"$\alpha=\mathrm{par}$",
               "modified": r"$\alpha=\mathrm{mod}$"}
FEM_REFERENCE_STYLES = {
    "coarse_fem": dict(label=r"FEM $u^H$", color="0.25", linestyle="--"),
    "fine_fem": dict(label=r"FEM $u^h$", color="0.55", linestyle=":"),
}
PARAMETER_SYMBOLS = {"epsilon": r"\varepsilon", "H": "H", "h": "h",
                     "delta": r"\delta", "final_time": "T", "num_time_steps": "N_t"}
AXIS_LABELS = {name: f"${symbol}$" for name, symbol in PARAMETER_SYMBOLS.items()}
ASSEMBLY_SYMBOLS = {"petrov_galerkin": r"\mathrm{PG}", "galerkin": r"\mathrm{G}"}


def power_two_label(value):
    exponent = round(math.log2(float(value)))
    return rf"$2^{{{exponent}}}$" if math.isclose(float(value), 2.0**exponent) else f"{value:g}"


def parameter_expression(name, value):
    """A parameter equation without dollar delimiters, suitable for plot titles."""
    if name == "assembly_type":
        return ASSEMBLY_SYMBOLS[value]
    if name in ("N_H", "N_h"):
        symbol = "H" if name == "N_H" else "h"
        return rf"{symbol}=1/{int(value)}"
    return rf"{PARAMETER_SYMBOLS[name]}={value:g}"


def parameter_label(name, value, *, power_two=False):
    if power_two:
        return rf"${PARAMETER_SYMBOLS[name]}={power_two_label(value).strip('$')}$"
    return f"${parameter_expression(name, value)}$"
