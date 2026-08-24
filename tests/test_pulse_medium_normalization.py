"""Tests for electromagnetic intensity and fluence in a dielectric host."""

import unittest

import numpy as np
from scipy.constants import c as C_SI
from scipy.constants import epsilon_0

from qd_mnp_rational_fit import AU_TIME_S, field_au_to_si
from tests._fixtures import make_test_pulse, make_zero_mode_model


class PulseMediumNormalizationTests(unittest.TestCase):
    def test_zero_amplitude_is_rejected_before_fluence_division(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero|fluence"):
            make_test_pulse(e0_au=0.0)

    def test_legacy_default_remains_vacuum_normalization(self) -> None:
        """Calling the legacy no-argument API must retain eps_m=1 behavior."""
        pulse = make_test_pulse()
        e0_si = float(field_au_to_si(pulse.E0_au))

        expected_intensity = 0.5 * epsilon_0 * C_SI * e0_si**2 * 1.0e-4
        sigma_s = pulse.sigma_t_au * AU_TIME_S
        carrier_correction = np.exp(-(pulse.omegaL_au * pulse.sigma_t_au) ** 2)
        integral_e2 = (
            0.5
            * np.sqrt(np.pi)
            * sigma_s
            * e0_si**2
            * (1.0 + carrier_correction)
        )
        expected_fluence = epsilon_0 * C_SI * integral_e2 * 1.0e-4

        self.assertAlmostEqual(pulse.peak_intensity_w_cm2(), expected_intensity, places=12)
        self.assertAlmostEqual(pulse.fluence_j_cm2(), expected_fluence, places=12)

    def test_intensity_and_fluence_scale_with_refractive_index(self) -> None:
        """Both quantities must acquire n=sqrt(eps_m), not eps_m."""
        pulse = make_test_pulse()
        try:
            intensity_vacuum = pulse.peak_intensity_w_cm2(eps_m=1.0)
            intensity_medium = pulse.peak_intensity_w_cm2(eps_m=2.25)
            fluence_vacuum = pulse.fluence_j_cm2(eps_m=1.0)
            fluence_medium = pulse.fluence_j_cm2(eps_m=2.25)
        except TypeError as exc:
            self.fail(f"GaussianPulse medium normalization API is missing: {exc}")

        self.assertAlmostEqual(intensity_medium / intensity_vacuum, 1.5, places=13)
        self.assertAlmostEqual(fluence_medium / fluence_vacuum, 1.5, places=13)

    def test_nonpositive_medium_permittivity_is_rejected(self) -> None:
        pulse = make_test_pulse()
        for eps_m in (0.0, -1.0):
            with self.subTest(eps_m=eps_m):
                try:
                    with self.assertRaises(ValueError):
                        pulse.peak_intensity_w_cm2(eps_m=eps_m)
                    with self.assertRaises(ValueError):
                        pulse.fluence_j_cm2(eps_m=eps_m)
                except TypeError as exc:
                    self.fail(f"GaussianPulse medium validation API is missing: {exc}")

    def test_solver_reports_medium_normalized_pulse_observables(self) -> None:
        """The solver must pass params.eps_m through to both pulse methods."""
        pulse = make_test_pulse(e0_au=1.0e-5)
        vacuum_model = make_zero_mode_model(eps_m=1.0)
        medium_model = make_zero_mode_model(eps_m=2.25)
        t_span = (-8.0 * pulse.sigma_t_au, 8.0 * pulse.sigma_t_au)

        vacuum = vacuum_model.solve(
            pulse,
            method="DOP853",
            rtol=1.0e-8,
            atol=1.0e-10,
            t_span_au=t_span,
        )
        medium = medium_model.solve(
            pulse,
            method="DOP853",
            rtol=1.0e-8,
            atol=1.0e-10,
            t_span_au=t_span,
        )

        self.assertAlmostEqual(
            medium.peak_intensity_w_cm2 / vacuum.peak_intensity_w_cm2,
            1.5,
            places=12,
        )
        self.assertAlmostEqual(
            medium.fluence_j_cm2 / vacuum.fluence_j_cm2,
            1.5,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
