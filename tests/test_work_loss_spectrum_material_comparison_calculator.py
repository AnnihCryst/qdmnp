"""Tests for the transient Shah-type work-loss spectrum calculator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from article_observables.qd_mnp_calculate_work_loss_spectrum_material_comparison import (
    BRANCH_IDS,
    SCHEMA_NAME,
    _fourier_integral_grid,
    analytic_incident_field_ft_au,
    calculate_payload,
    energy_grid_resolution_diagnostics,
    parse_args,
    spectral_effective_alpha_grid,
)
from article_observables.qd_mnp_calculate_work_loss_fluence import (
    spectral_effective_alpha_au,
)
from qd_mnp_rational_fit import GaussianPulse, au_to_eV, eV_to_au, fs_to_au


class WorkLossSpectrumFourierTests(unittest.TestCase):
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
            / "article_observables"
            / "qd_mnp_calculate_work_loss_spectrum_material_comparison.py",
        )

    def test_real_fqs_payload_has_expected_axes_and_identities(self) -> None:
        payload = self.payload
        self.assertEqual(payload["branch_id"].tolist(), BRANCH_IDS.tolist())
        self.assertEqual(payload["sigma_qs_work_cm2"].shape, (2, 1, 2, 9))
        self.assertEqual(payload["bare_mnp_sigma_qs_work_cm2"].shape, (2, 1, 9))
        self.assertEqual(payload["energy_grid_converged"].shape, (2, 1, 2))
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
        from article_observables.qd_mnp_calculate_work_loss_spectrum_material_comparison import (
            _validate_args,
        )

        with self.assertRaisesRegex(ValueError, "carrier"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
