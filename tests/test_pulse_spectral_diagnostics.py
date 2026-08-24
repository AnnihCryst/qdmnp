"""Fast tests for pulse-tail selection and strict-QS spectral semantics."""

from types import SimpleNamespace
import unittest

import numpy as np

from qd_mnp_pulse_absorption_sweep import (
    response_tail_ratio,
    spectral_absorption_cross_section_cm2,
    spectral_cross_sections_cm2,
)
from tests._fixtures import make_test_pulse, make_zero_mode_model


class PulseSpectralDiagnosticsTests(unittest.TestCase):
    def test_recommended_tail_uses_slowest_coherence_rate(self) -> None:
        model = make_zero_mode_model()
        self.assertAlmostEqual(model.recommended_post_pulse_time_au(8.0), 8.0e4)

        model.fit.gamma_modes_au = np.array([1.0e-5])
        # Lorentz amplitudes decay at gamma/2, hence 8/(1e-5/2).
        self.assertAlmostEqual(model.recommended_post_pulse_time_au(8.0), 1.6e6)

    def test_recommended_tail_rejects_an_undamped_coherence(self) -> None:
        model = make_zero_mode_model()
        model.params = type(model.params)(
            model.params.c_au,
            model.params.a_au,
            model.params.R_au,
            model.params.G,
            model.params.eps_m,
            model.params.d_au,
            model.params.omega0_au,
            0.0,
            0.0,
            model.params.material,
        )
        with self.assertRaisesRegex(ValueError, "undamped|finite"):
            model.recommended_post_pulse_time_au()

    def test_spectral_sections_expose_work_scattering_and_optical_residual(self) -> None:
        pulse = make_test_pulse()
        t = np.linspace(-8.0 * pulse.sigma_t_au, 8.0 * pulse.sigma_t_au, 4001)
        # A quadrature response gives a non-zero imaginary effective alpha.
        mu = -pulse.envelope(t) * np.sin(pulse.omegaL_au * t)
        result = SimpleNamespace(t_au=t, mu_total_au=mu)

        sections = spectral_cross_sections_cm2(result, pulse, eps_m=2.25)
        self.assertAlmostEqual(
            float(sections.quasistatic_work_loss_cm2),
            float(
                sections.optical_theorem_residual_cm2
                + sections.rayleigh_scattering_estimate_cm2
            ),
            places=24,
        )
        # Historical dataclass fields and this badly named function remain
        # only to read/write schema-1 data.  In particular ``absorption`` is
        # the optical-theorem residual, while the deprecated function actually
        # returned the formal k Im(alpha) work-loss estimate.
        self.assertEqual(sections.extinction_cm2, sections.quasistatic_work_loss_cm2)
        self.assertEqual(
            sections.scattering_cm2,
            sections.rayleigh_scattering_estimate_cm2,
        )
        self.assertEqual(
            sections.absorption_cm2,
            sections.optical_theorem_residual_cm2,
        )
        with self.assertWarns(DeprecationWarning):
            legacy_value = spectral_absorption_cross_section_cm2(
                result, pulse, 2.25
            )
        self.assertEqual(legacy_value, float(sections.quasistatic_work_loss_cm2))

    def test_response_tail_ratio_detects_an_undecayed_tail(self) -> None:
        decayed = np.concatenate([np.ones(100), np.zeros(100)])
        ringing = np.ones(200)
        self.assertEqual(response_tail_ratio(decayed), 0.0)
        self.assertEqual(response_tail_ratio(ringing), 1.0)


if __name__ == "__main__":
    unittest.main()
