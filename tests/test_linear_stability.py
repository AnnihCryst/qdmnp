"""Tests for the full coupled ground-state Jacobian guard."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from qd_mnp_rational_fit import dipole_si_to_au
from tests._fixtures import make_zero_mode_model


class LinearStabilityTests(unittest.TestCase):
    def test_uncoupled_damped_model_has_a_stable_full_jacobian(self) -> None:
        model = make_zero_mode_model()
        jacobian = model.linearized_ground_state_jacobian()
        diagnostics = model.linearized_ground_state_stability()

        self.assertEqual(jacobian.shape, (3, 3))
        self.assertEqual(diagnostics.poles_au.shape, (3,))
        self.assertTrue(diagnostics.stable)
        self.assertLess(diagnostics.spectral_abscissa_au, 0.0)

    def test_excessive_feedback_coupling_is_rejected(self) -> None:
        model = make_zero_mode_model()
        model.n_modes = 1
        model.params = replace(
            model.params,
            G=1000.0,
            d_au=float(dipole_si_to_au(7.5e-29)),
        )
        model.C = model.params.eps_m * model.params.a_au**2 * model.params.c_au / 3.0
        model.fit = SimpleNamespace(
            alpha_inf=5.226325952868758,
            strengths_au2=np.asarray([0.02468415]),
            omega_modes_au=np.asarray([0.09284675]),
            gamma_modes_au=np.asarray([0.03531425]),
        )

        diagnostics = model.linearized_ground_state_stability()
        self.assertFalse(diagnostics.stable)
        self.assertGreater(diagnostics.spectral_abscissa_au, diagnostics.tolerance_au)
        with self.assertRaisesRegex(RuntimeError, "Unstable|Jacobian"):
            model.assert_linearized_ground_state_stable()


if __name__ == "__main__":
    unittest.main()
