"""Basic numerical and density-matrix diagnostics returned by solve()."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from qd_mnp_rational_fit import (
    DEBYE_C_M,
    GaussianPulse,
    dipole_si_to_au,
    eV_to_au,
    field_si_to_au,
    fs_to_au,
)
from tests._fixtures import make_test_pulse, make_zero_mode_model


class SolverDiagnosticsTests(unittest.TestCase):
    def _solve(self, *, rtol: float = 1.0e-8, atol: float = 1.0e-10):
        model = make_zero_mode_model()
        pulse = make_test_pulse()
        t_span = (-8.0 * pulse.sigma_t_au, 8.0 * pulse.sigma_t_au)
        result = model.solve(
            pulse,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            t_span_au=t_span,
        )
        return model, result

    def test_solver_output_arrays_are_finite_and_aligned(self) -> None:
        model, result = self._solve()
        n_time = result.t_au.size

        self.assertTrue(result.solve_ivp_result.success)
        self.assertGreater(n_time, 2)
        self.assertTrue(np.all(np.diff(result.t_au) > 0.0))
        self.assertEqual(result.y.shape, (2 * model.n_modes + 3, n_time))
        for values in (
            result.t_au,
            result.y,
            result.mu_p_au,
            result.mu_d_au,
            result.mu_total_au,
            result.mu_dot_total_au,
        ):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.isfinite(result.absorbed_energy_j))
        self.assertTrue(np.isfinite(result.fluence_j_cm2))
        self.assertTrue(np.isfinite(result.peak_intensity_w_cm2))

    def test_reported_density_matrix_diagnostics_match_trajectory(self) -> None:
        model, result = self._solve()
        W, Q, P = result.y[2 * model.n_modes : 2 * model.n_modes + 3]
        radius = np.sqrt(W**2 + Q**2 + P**2)
        expected_max_radius = float(np.max(radius))
        expected_min_eigenvalue = float(np.min(0.5 * (1.0 - radius)))

        try:
            max_bloch_radius = result.max_bloch_radius
            min_density_eigenvalue = result.min_density_eigenvalue
        except AttributeError as exc:
            self.fail(f"HybridSolveResult physical diagnostics are missing: {exc}")

        self.assertAlmostEqual(max_bloch_radius, expected_max_radius, places=13)
        self.assertAlmostEqual(
            min_density_eigenvalue,
            expected_min_eigenvalue,
            places=13,
        )
        self.assertLessEqual(max_bloch_radius, 1.0 + 1.0e-8)
        self.assertGreaterEqual(min_density_eigenvalue, -5.0e-9)

    def test_tighter_solver_tolerances_do_not_change_final_bloch_state(self) -> None:
        model_loose, loose = self._solve(rtol=1.0e-7, atol=1.0e-9)
        model_tight, tight = self._solve(rtol=1.0e-10, atol=1.0e-12)
        loose_final = loose.y[2 * model_loose.n_modes :, -1]
        tight_final = tight.y[2 * model_tight.n_modes :, -1]
        np.testing.assert_allclose(loose_final, tight_final, rtol=2.0e-6, atol=2.0e-8)

    def test_solver_rejects_a_time_window_that_truncates_the_pulse(self) -> None:
        model = make_zero_mode_model()
        pulse = make_test_pulse()
        with self.assertRaisesRegex(ValueError, "truncates the incident pulse"):
            model.solve(
                pulse,
                method="DOP853",
                t_span_au=(-3.0 * pulse.sigma_t_au, 5.0 * pulse.sigma_t_au),
            )

    def test_solver_rejects_material_bloch_ball_violation(self) -> None:
        """The post-solve guard must catch more than floating-point overshoot.

        The model object is deliberately assembled without the constructor here:
        constructor validation has its own tests, while this test exercises the
        independent trajectory-level safety net.
        """
        model = make_zero_mode_model()
        model.params = replace(
            model.params,
            d_au=float(dipole_si_to_au(30.0 * DEBYE_C_M)),
            omega0_au=float(eV_to_au(2.042)),
            gamma_au=float(eV_to_au(0.00302)),
            Gamma_au=float(eV_to_au(0.00127)),
        )
        pulse = GaussianPulse(
            E0_au=float(field_si_to_au(1.0e7)),
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20.0)),
            tau_kind="fwhm_intensity",
        )

        with self.assertRaisesRegex(
            (RuntimeError, ValueError),
            "Bloch|density|physical|positiv",
        ):
            model.solve(
                pulse,
                method="DOP853",
                rtol=1.0e-9,
                atol=1.0e-11,
                t_span_au=(-5.0 * pulse.sigma_t_au, 5.0 * pulse.sigma_t_au),
            )

    def test_completed_passive_run_rejects_materially_negative_external_work(self) -> None:
        model = make_zero_mode_model()
        pulse = make_test_pulse()
        with patch(
            "qd_mnp_rational_fit.np.trapezoid",
            side_effect=[-1.0, 1.0],
        ):
            with self.assertRaisesRegex(RuntimeError, "negative.*external-field work"):
                model.solve(
                    pulse,
                    method="DOP853",
                    rtol=1.0e-8,
                    atol=1.0e-10,
                    t_span_au=(-8.0 * pulse.sigma_t_au, 8.0 * pulse.sigma_t_au),
                )


if __name__ == "__main__":
    unittest.main()
