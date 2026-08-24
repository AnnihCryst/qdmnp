"""Shared physical-parameter overrides for the native QD--MNP scripts.

Only explicitly supplied values replace ``make_default_params()``.  Lengths,
energies, linewidths and dipoles are converted from nm/eV/meV/Debye to the
atomic units used by the solver.  The native coupling is always geometric:
``orientation='long'`` means G=2 and ``orientation='trans'`` means G=-1.
"""

from __future__ import annotations

from qd_mnp_rational_fit import (
    DEBYE_C_M,
    make_params_with_overrides,
)

__all__ = ["DEBYE_C_M", "make_params_with_overrides"]
