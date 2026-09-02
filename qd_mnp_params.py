"""Shared physical-parameter overrides for the native QD--MNP scripts.

Only explicitly supplied values replace ``make_default_params()``.  Lengths,
energies, linewidths and dipoles are converted from nm/eV/meV/Debye to the
atomic units used by the solver.

The native coupling is always geometric.  ``qd_position`` places the QD centre
at ``r_D=(0,0,c+h)`` (``'tip'``) or ``r_D=(a+h,0,0)`` (``'equatorial'``) and
``field_polarization`` points the incident field along ``e_z``
(``'longitudinal'``) or ``e_x`` (``'transverse'``).  The two are independent,
and the dipole-tensor factor follows from them as ``G=3*(e_L.r_D_hat)**2-1``:
``G=2`` when the polarization is parallel to the QD direction and ``G=-1`` when
it is perpendicular.  ``orientation='long'``/``'trans'`` remains an alias of the
polarization, so the historical tip geometry keeps ``G=2``/``G=-1``.
"""

from __future__ import annotations

from qd_mnp_rational_fit import (
    DEBYE_C_M,
    make_params_with_overrides,
)

__all__ = ["DEBYE_C_M", "make_params_with_overrides"]
