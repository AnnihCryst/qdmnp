"""Real time/frequency consistency tests for the production equations.

The one-mode material fit is intentionally used here only to make the test
fast.  It need not approximate tabulated gold with production accuracy: both
branches receive the same fitted transfer function, so the test isolates the
algebraic equivalence of the analytic weak-field model and the Maxwell--Bloch
time-domain realization, including local-field and Fourier conventions.
"""

from dataclasses import replace
import unittest

import numpy as np

from qd_mnp_linear_spectrum import linear_coupled_alpha_au
from qd_mnp_pulse_absorption_sweep import (
    spectral_cross_sections_cm2,
    spectral_effective_alpha_au,
)
from qd_mnp_rational_fit import (
    GaussianPulse,
    HybridQDPlasmonModel,
    eV_to_au,
    fs_to_au,
    make_default_params,
)


class TimeFrequencyRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        params = replace(
            make_default_params(),
            gamma_au=float(eV_to_au(0.020)),
            Gamma_au=float(eV_to_au(0.020)),
        )
        cls.model = HybridQDPlasmonModel(
            params,
            n_modes=1,
            # Material-fit accuracy is deliberately outside this algebraic
            # regression; production runs retain the strict default gates.
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            verbose=False,
        )
        cls.pulse = GaussianPulse(
            E0_au=1.0e-8,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(5.0)),
            tau_kind="fwhm_intensity",
        )
        cls.post_au = cls.model.recommended_post_pulse_time_au(decay_times=12.0)
        start_au = -10.0 * cls.pulse.sigma_t_au
        solve_kwargs = dict(method="DOP853", rtol=3.0e-10, atol=1.0e-12)
        cls.result_t = cls.model.solve(
            cls.pulse,
            t_span_au=(start_au, cls.post_au),
            **solve_kwargs,
        )
        cls.result_2t = cls.model.solve(
            cls.pulse,
            t_span_au=(start_au, 2.0 * cls.post_au),
            **solve_kwargs,
        )

    def test_weak_field_time_domain_matches_linear_spectrum(self) -> None:
        alpha_time = spectral_effective_alpha_au(
            self.result_t,
            self.pulse,
            self.model.params.eps_m,
        )
        alpha_linear = linear_coupled_alpha_au(
            self.model,
            np.asarray([2.042]),
        )[0][0]
        relative_error = abs(alpha_time - alpha_linear) / abs(alpha_linear)

        self.assertLess(relative_error, 1.0e-6)
        self.assertLess(
            self.result_t.diagnostics.excited_population_max,
            1.0e-7,
        )
        self.assertGreater(self.result_t.work_from_incident_field_j, 0.0)
        self.assertGreater(self.result_t.sigma_energy_transfer_cm2, 0.0)
        self.assertTrue(self.result_t.diagnostics.work_passivity_checked)
        self.assertTrue(
            self.result_t.diagnostics.work_nonnegative_within_tolerance
        )

    def test_observables_converge_when_post_window_is_doubled(self) -> None:
        alpha_t = spectral_effective_alpha_au(
            self.result_t,
            self.pulse,
            self.model.params.eps_m,
        )
        alpha_2t = spectral_effective_alpha_au(
            self.result_2t,
            self.pulse,
            self.model.params.eps_m,
        )
        self.assertLess(abs(alpha_t - alpha_2t) / abs(alpha_2t), 1.0e-6)

        sections_t = spectral_cross_sections_cm2(
            self.result_t,
            self.pulse,
            self.model.params.eps_m,
        )
        sections_2t = spectral_cross_sections_cm2(
            self.result_2t,
            self.pulse,
            self.model.params.eps_m,
        )
        np.testing.assert_allclose(
            [
                sections_t.extinction_cm2,
                sections_t.scattering_cm2,
                sections_t.absorption_cm2,
            ],
            [
                sections_2t.extinction_cm2,
                sections_2t.scattering_cm2,
                sections_2t.absorption_cm2,
            ],
            rtol=1.0e-5,
            atol=1.0e-24,
        )
        np.testing.assert_allclose(
            self.result_t.sigma_energy_transfer_cm2,
            self.result_2t.sigma_energy_transfer_cm2,
            rtol=1.0e-9,
            atol=1.0e-24,
        )

    def test_weak_time_frequency_equivalence_covers_orientation_and_local_field(self) -> None:
        cases = (
            ("trans", "effective_external", 1.0, 6.0),
            ("long", "bare_internal", 2.25, 6.0),
            ("trans", "bare_internal", 2.25, 6.0),
        )
        for orientation, convention, eps_m, eps_qd in cases:
            with self.subTest(
                orientation=orientation,
                convention=convention,
                eps_m=eps_m,
            ):
                params = replace(
                    make_default_params(orientation),
                    eps_m=eps_m,
                    eps_qd=eps_qd,
                    qd_dipole_convention=convention,
                    gamma_au=float(eV_to_au(0.020)),
                    Gamma_au=float(eV_to_au(0.020)),
                )
                model = HybridQDPlasmonModel(
                    params,
                    orientation=orientation,
                    n_modes=1,
                    max_fit_normalized_rms=None,
                    max_fit_pointwise_relative_error=None,
                    verbose=False,
                )
                if convention == "bare_internal":
                    self.assertNotEqual(params.qd_local_field_factor, 1.0)
                post_au = model.recommended_post_pulse_time_au(decay_times=12.0)
                result = model.solve(
                    self.pulse,
                    t_span_au=(-10.0 * self.pulse.sigma_t_au, post_au),
                    method="DOP853",
                    rtol=3.0e-10,
                    atol=1.0e-12,
                    # This regression proves algebraic equivalence using the
                    # same deliberately low-order transfer function on both
                    # sides; production N9 spectral gates are tested elsewhere.
                    spectral_window_policy="ignore",
                )
                alpha_time = spectral_effective_alpha_au(
                    result,
                    self.pulse,
                    params.eps_m,
                )
                alpha_linear = linear_coupled_alpha_au(
                    model,
                    np.asarray([2.042]),
                )[0][0]
                self.assertLess(
                    abs(alpha_time - alpha_linear) / abs(alpha_linear),
                    2.0e-6,
                )

    def test_strong_few_cycle_work_and_fourier_response_converge_with_half_step(self) -> None:
        pulse = GaussianPulse(
            E0_au=1.0e-3,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(2.0)),
            tau_kind="fwhm_intensity",
        )
        t_span = (
            -10.0 * pulse.sigma_t_au,
            self.model.recommended_post_pulse_time_au(decay_times=6.0),
        )
        solve_kwargs = dict(
            method="DOP853",
            rtol=1.0e-8,
            atol=1.0e-10,
            t_span_au=t_span,
            # This regression isolates time integration/quadrature.  A 2 fs
            # pulse is intentionally broader than the one-mode test fit.
            spectral_window_policy="ignore",
        )
        baseline = self.model.solve(pulse, **solve_kwargs)
        refined = self.model.solve(
            pulse,
            max_step_au=0.5 * baseline.diagnostics.max_step_limit_au,
            **solve_kwargs,
        )

        work_relative_change = abs(
            baseline.work_from_incident_field_j
            - refined.work_from_incident_field_j
        ) / abs(refined.work_from_incident_field_j)
        alpha_baseline = spectral_effective_alpha_au(
            baseline,
            pulse,
            self.model.params.eps_m,
        )
        alpha_refined = spectral_effective_alpha_au(
            refined,
            pulse,
            self.model.params.eps_m,
        )
        alpha_relative_change = abs(alpha_baseline - alpha_refined) / abs(
            alpha_refined
        )

        self.assertLess(work_relative_change, 1.0e-5)
        self.assertLess(alpha_relative_change, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
