"""Small deterministic fixtures shared by the physics-contract tests.

The zero-mode model deliberately bypasses the expensive material fit.  It keeps
the production Bloch RHS and solver, while isolating solver diagnostics and
pulse normalization from rational-fit accuracy.
"""

from types import SimpleNamespace

import numpy as np

from qd_mnp_rational_fit import (
    GaussianPulse,
    HybridQDPlasmonModel,
    HybridSystemParams,
    eV_to_au,
    fs_to_au,
    nm_to_au,
)


def make_zero_mode_model(*, eps_m: float = 1.0) -> HybridQDPlasmonModel:
    params = HybridSystemParams(
        c_au=float(nm_to_au(15.0)),
        a_au=float(nm_to_au(7.0)),
        R_au=float(nm_to_au(20.0)),
        G=2.0,
        eps_m=eps_m,
        d_au=1.0,
        omega0_au=float(eV_to_au(2.0)),
        gamma_au=1.0e-4,
        Gamma_au=1.0e-4,
        eps_qd=eps_m,
    )

    model = object.__new__(HybridQDPlasmonModel)
    model.params = params
    model.orientation = "long"
    model.n_modes = 0
    model.C = 0.0
    model.J = 0.0
    model.fit = SimpleNamespace(
        alpha_inf=0.0,
        strengths_au2=np.empty(0),
        omega_modes_au=np.empty(0),
        gamma_modes_au=np.empty(0),
    )
    return model


def make_test_pulse(*, e0_au: float = 2.0e-4) -> GaussianPulse:
    return GaussianPulse(
        E0_au=e0_au,
        omegaL_au=float(eV_to_au(2.0)),
        tau_au=float(fs_to_au(2.0)),
        tau_kind="fwhm_intensity",
    )
