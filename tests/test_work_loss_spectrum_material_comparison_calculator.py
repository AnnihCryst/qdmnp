"""Tests for the transient Shah-type work-loss spectrum calculator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from qdmnp.observables.calculate_work_loss_spectrum_material_comparison import (
    BRANCH_IDS,
    SCHEMA_NAME,
    _fourier_integral_grid,
    _spectrum_window_audit,
    analytic_incident_field_ft_au,
    calculate_payload,
    energy_grid_resolution_diagnostics,
    parse_args,
    spectral_effective_alpha_grid,
)
from qdmnp.observables.calculate_work_loss_fluence import (
    _audit_work_observable_window,
    spectral_effective_alpha_au,
)
from qdmnp.rational_fit import AU_ENERGY_J, GaussianPulse, au_to_eV, eV_to_au, fs_to_au
from qdmnp.observables.work_spectrum_metrics import delta_window_diagnostics


class WorkLossSpectrumFourierTests(unittest.TestCase):
    def test_fourier_quadrature_resolves_gaussian_wings_on_nonuniform_grid(self) -> None:
        pulse = GaussianPulse(
            E0_au=2e-5, omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20)), tau_kind="fwhm_intensity",
        )
        # An adaptive solver changes its step near the pulse. On this grid,
        # trapezoidal integration has a 0.74% error in the spectral wings.
        time = np.concatenate((
            np.arange(-10*pulse.sigma_t_au, 0.0, 3.0),
            np.arange(0.0, 10*pulse.sigma_t_au, 6.0),
        ))
        energy = np.linspace(1.90, 2.18, 101)
        actual = _fourier_integral_grid(time, pulse.field(time), energy)
        expected = analytic_incident_field_ft_au(pulse, energy)
        self.assertLess(np.max(np.abs(actual-expected)/np.abs(expected)), 1e-3)

    def test_work_window_uses_the_ode_accumulator_and_detects_late_work(self) -> None:
        pulse = GaussianPulse(
            E0_au=2e-5, omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20)), tau_kind="fwhm_intensity",
        )
        time = np.linspace(-10*pulse.sigma_t_au, 40*pulse.sigma_t_au, 2001)
        # A compact response with no Fourier tail after T/2. The independently
        # integrated work state must be used, even if sampled quadrature differs.
        mu = -pulse.field_dot(time)
        work = np.where(time > 8*pulse.sigma_t_au, 1e-8, 0.0)
        result = SimpleNamespace(
            t_au=time, mu_total_au=mu, mu_dot_total_au=np.gradient(mu, time),
            accumulated_work_au=work,
            sigma_energy_transfer_cm2=work[-1]*AU_ENERGY_J/pulse.fluence_j_cm2(eps_m=2.25),
        )
        def spectrum_audit():
            return _spectrum_window_audit(
                result, pulse, 2.25, np.linspace(2.0, 2.08, 9),
                minimum_incident_relative_amplitude=1e-3,
                max_incident_ft_pointwise_relative_error=1e-3,
                max_spectrum_window_relative_change=1e-3,
                max_energy_window_relative_change=1e-3,
            )
        audit = spectrum_audit()
        self.assertAlmostEqual(audit.energy_window_relative_change, 0.0)
        self.assertTrue(audit.spectrum_window_converged)
        self.assertTrue(_audit_work_observable_window(result, pulse, 2.25, 1e-3).accepted)

        # The window gate must still reject a real 1% increment after T/2.
        result.accumulated_work_au = np.where(time < 0.6*time[-1], 0.99*work, work)
        audit = spectrum_audit()
        self.assertAlmostEqual(audit.energy_window_relative_change, 0.01)
        self.assertFalse(audit.spectrum_window_converged)
        self.assertFalse(_audit_work_observable_window(result, pulse, 2.25, 1e-3).accepted)

    def test_metal_background_cannot_hide_unconverged_qd_contrast(self) -> None:
        bare = np.full(9, 1.0e-10)
        full = bare + 1.0e-14
        half = full + 1.0e-15
        # The total spectrum easily passes a 0.1% check; its contrast does not.
        self.assertLess(np.max(abs(full-half))/np.max(abs(full)), 1e-3)
        audit = delta_window_diagnostics(full, half, bare, np.ones(9, dtype=bool),
                                         relative_tolerance=1e-3)
        self.assertFalse(audit["delta_window_converged"])
        self.assertAlmostEqual(float(audit["delta_window_max_normalized_change"]), 0.1, places=9)

    def test_zero_contrast_and_unsupported_points_have_defined_window_status(self) -> None:
        bare = np.asarray([1.0e-10, 1.0e-10, np.nan])
        support = np.asarray([True, True, False])
        audit = delta_window_diagnostics(bare, bare, bare, support, relative_tolerance=.01)
        self.assertTrue(audit["delta_window_converged"])
        self.assertEqual(float(audit["delta_window_max_normalized_change"]), 0.0)
        changed = bare.copy()
        changed[0] += 1e-15
        audit = delta_window_diagnostics(bare, changed, bare, support, relative_tolerance=.01)
        self.assertFalse(audit["delta_window_converged"])

    def test_analytic_real_gaussian_transform_matches_quadrature(self) -> None:
        pulse = GaussianPulse(
            E0_au=2.3e-5,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20.0)),
            tau_kind="fwhm_intensity",
        )
        time = np.linspace(
            -10.0 * pulse.sigma_t_au,
            10.0 * pulse.sigma_t_au,
            24001,
        )
        energy = np.linspace(1.96, 2.12, 41)
        numerical = _fourier_integral_grid(time, pulse.field(time), energy)
        analytic = analytic_incident_field_ft_au(pulse, energy)
        relative = np.max(np.abs(numerical - analytic)) / np.max(np.abs(analytic))
        self.assertLess(relative, 2.0e-11)

    def test_fourier_sign_recovers_positive_imaginary_response(self) -> None:
        pulse = GaussianPulse(
            E0_au=1.4e-5,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20.0)),
            tau_kind="fwhm_intensity",
        )
        time = np.linspace(
            -10.0 * pulse.sigma_t_au,
            10.0 * pulse.sigma_t_au,
            24001,
        )
        energy = np.linspace(1.98, 2.10, 31)
        eps_m = 1.7
        real_part = 3.4
        slope = 0.23
        # For the +i omega t convention, FT[dE/dt] = -i omega FT[E].
        mu = eps_m * (
            real_part * pulse.field(time)
            - slope / pulse.omegaL_au * pulse.field_dot(time)
        )
        result = SimpleNamespace(t_au=time, mu_total_au=mu)
        alpha, *_ = spectral_effective_alpha_grid(
            result,
            pulse,
            eps_m,
            energy,
            minimum_incident_relative_amplitude=1.0e-3,
        )
        expected = real_part + 1j * slope * eV_to_au(energy) / pulse.omegaL_au
        np.testing.assert_allclose(alpha, expected, rtol=3.0e-10, atol=3.0e-10)
        self.assertTrue(np.all(alpha.imag > 0.0))

    def test_carrier_grid_value_matches_existing_scalar_observable_api(self) -> None:
        pulse = GaussianPulse(
            E0_au=1.7e-5,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20.0)),
            tau_kind="fwhm_intensity",
        )
        time = np.linspace(
            -10.0 * pulse.sigma_t_au,
            10.0 * pulse.sigma_t_au,
            24001,
        )
        mu = 1.8 * pulse.field(time) - 0.07 * pulse.field_dot(time)
        result = SimpleNamespace(t_au=time, mu_total_au=mu)
        scalar = spectral_effective_alpha_au(result, pulse, eps_m=1.4)
        vector, *_ = spectral_effective_alpha_grid(
            result,
            pulse,
            eps_m=1.4,
            energies_eV=np.asarray([au_to_eV(pulse.omegaL_au)]),
            minimum_incident_relative_amplitude=1.0e-3,
        )
        self.assertAlmostEqual(vector[0].real, scalar.real, places=13)
        self.assertAlmostEqual(vector[0].imag, scalar.imag, places=13)

    def test_nested_energy_certificate_detects_curvature(self) -> None:
        energy = np.linspace(1.9, 2.2, 101)
        smooth = (energy - 2.0) ** 2
        values = np.stack((smooth, 2.0 * smooth))[None, None, None, :, :]
        diagnostics = energy_grid_resolution_diagnostics(
            energy,
            values,
            max_midpoint_normalized_error=1.0e-3,
        )
        self.assertEqual(
            np.asarray(diagnostics["midpoint_normalized_error"]).shape,
            (1, 1, 1, 2),
        )
        self.assertTrue(diagnostics["all_accepted"])


class WorkLossSpectrumCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.args = parse_args(
            [
                "--output",
                "unused.npz",
                "--preset",
                "quick",
                "--channel",
                "axis_long",
                "--fluence-j-cm2",
                "1e-10",
                "1.1e-10",
                "--fluence-label",
                "weak-1",
                "weak-2",
                "--carrier-energy-ev",
                "2.042",
                "--pulse-tau-fs",
                "5",
                "--energy-min-ev",
                "2.00",
                "--energy-max-ev",
                "2.08",
                "--energy-points",
                "9",
                "--spatial-order-max",
                "1",
                "--multi-fit-modes",
                "2",
                "--modal-audit-points",
                "101",
                "--one-fit-accuracy-policy",
                "ignore",
                "--fit-quality-policy",
                "ignore",
                "--radiative-consistency-policy",
                "ignore",
                "--spatial-convergence-policy",
                "ignore",
                "--reduction-policy",
                "ignore",
                "--spectral-window-policy",
                "ignore",
                "--positivity-policy",
                "ignore",
                "--work-passivity-policy",
                "ignore",
                "--tail-policy",
                "ignore",
                "--observable-convergence-policy",
                "ignore",
                "--spectral-support-policy",
                "ignore",
                "--incident-ft-policy",
                "ignore",
                "--energy-grid-convergence-policy",
                "ignore",
                "--post-fs",
                "100",
                "--max-auto-tail-extensions",
                "0",
                "--rtol",
                "1e-6",
                "--atol",
                "1e-8",
                "--points-per-fastest-cycle",
                "8",
                "--max-spectrum-window-relative-change",
                "1",
                "--max-energy-window-relative-change",
                "1",
                "--max-incident-ft-pointwise-relative-error",
                "1",
                "--max-energy-grid-midpoint-normalized-error",
                "1",
                "--quiet",
            ]
        )
        cls.payload, cls.metadata = calculate_payload(
            cls.args,
            generator_path=Path(__file__).resolve().parents[1]
            / "src"
            / "observables"
            / "calculate_work_loss_spectrum_material_comparison.py",
        )

    def test_real_fqs_payload_has_expected_axes_and_identities(self) -> None:
        payload = self.payload
        self.assertEqual(payload["branch_id"].tolist(), BRANCH_IDS.tolist())
        self.assertEqual(payload["sigma_qs_work_cm2"].shape, (2, 1, 2, 9))
        self.assertEqual(payload["bare_mnp_sigma_qs_work_cm2"].shape, (2, 1, 9))
        self.assertEqual(payload["energy_grid_converged"].shape, (2, 1, 2))
        self.assertEqual(payload["delta_window_converged"].shape, (2, 1, 2))
        self.assertEqual(payload["delta_sigma_half_window_cm2"].shape, (2, 1, 2, 9))
        np.testing.assert_allclose(
            payload["delta_sigma_qs_work_cm2"],
            payload["sigma_qs_work_cm2"]
            - payload["bare_mnp_sigma_qs_work_cm2"][:, :, None, :],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            payload["actual_fluence_j_cm2"],
            np.broadcast_to(
                payload["selected_fluence_j_cm2"][None, None, :],
                payload["actual_fluence_j_cm2"].shape,
            ),
            rtol=2.0e-13,
            atol=0.0,
        )
        self.assertTrue(payload["spectrum_support_mask"].all())
        self.assertTrue(payload["fit_passive"].all())
        self.assertTrue(payload["coupled_stable"].all())
        self.assertTrue(np.isfinite(payload["alpha_eff_au3_real"]).all())
        self.assertTrue(np.isfinite(payload["alpha_eff_au3_imag"]).all())

    def test_artifact_metadata_preserves_inputs_constants_and_limits(self) -> None:
        metadata = self.metadata
        self.assertEqual(metadata["schema_name"], SCHEMA_NAME)
        self.assertEqual(metadata["multi_fit_mode_count"], 2)
        self.assertEqual(
            metadata["calculation_method"]["ode_solves"],
            "one per branch x channel x selected fluence",
        )
        self.assertFalse(
            metadata["model_scope"]["laser_propagation_direction_included"]
        )
        self.assertIn("vacuum_permittivity_f_m", metadata["physical_constants"])
        self.assertIn("terminology", metadata["observable_definitions"])
        self.assertIn("source_sha256", metadata["provenance"])
        for value in self.payload.values():
            self.assertFalse(np.asarray(value).dtype.hasobject)


class WorkLossSpectrumArgumentTests(unittest.TestCase):
    def test_carrier_must_lie_inside_saved_spectrum(self) -> None:
        args = parse_args(
            [
                "--fluence-j-cm2",
                "1e-9",
                "2e-9",
                "--energy-min-ev",
                "1.90",
                "--energy-max-ev",
                "2.00",
            ]
        )
        from qdmnp.observables.calculate_work_loss_spectrum_material_comparison import (
            _validate_args,
        )

        with self.assertRaisesRegex(ValueError, "carrier"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
