"""Fast physics contracts for the native quasistatic QD--MNP model.

Most tests avoid nonlinear optimization and check algebraic invariants.  One
production regression deliberately builds both canonical nine-mode fits and
audits them on an independent dense grid, because sparse material nodes alone
cannot exclude artificial between-node resonances.
"""

from dataclasses import replace
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from qd_mnp_linear_spectrum import (
    MODAL_OBSERVABLE_LOCAL_RELATIVE_ERROR_LIMIT,
    MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT,
    compute_spectrum,
    qd_linear_polarizability_au,
)
from qd_mnp_rational_fit import (
    GaussianPulse,
    HybridQDPlasmonModel,
    RationalLorentzFit,
    eV_to_au,
    fs_to_au,
    make_default_params,
    orientation_factor,
    sampled_positive_frequency_spectral_fraction,
)
from tests._fixtures import make_zero_mode_model


FIT_WINDOW_EV = (0.8, 3.0)


def _pulse(tau_fs: float) -> GaussianPulse:
    return GaussianPulse(
        E0_au=1.0e-8,
        omegaL_au=float(eV_to_au(2.042)),
        tau_au=float(fs_to_au(tau_fs)),
        tau_kind="fwhm_intensity",
    )


def _passive_lorentz_model() -> HybridQDPlasmonModel:
    """Build a deterministic modal response without running an optimizer."""
    fit = RationalLorentzFit(
        alpha_inf=0.35,
        strengths_au2=np.array([2.0e-4, 7.5e-4, 1.2e-3]),
        omega_modes_au=np.array([0.02, 0.08, 0.20]),
        gamma_modes_au=np.array([0.003, 0.010, 0.025]),
        energies_used_eV=np.array([0.5, 2.0, 5.0]),
        alpha_used=np.ones(3, dtype=complex),
        rms_alpha=0.0,
        rms_inv_alpha=0.0,
        cost=0.0,
        passive_on_fit_window=True,
        passive_for_all_positive_frequencies=True,
    )
    model = object.__new__(HybridQDPlasmonModel)
    model.n_modes = len(fit.strengths_au2)
    model.fit_window_eV = (1.0e-8, 1.0e3)
    model.fit = fit
    return model


class NativeOrientationTests(unittest.TestCase):
    def test_near_spherical_depolarization_factors_have_a_stable_limit(self) -> None:
        base = make_default_params()
        diagnostic_model = object.__new__(HybridQDPlasmonModel)
        diagnostic_model.params = replace(
            base,
            c_au=base.a_au * (1.0 + 1.0e-11),
        )

        longitudinal, transverse = diagnostic_model._depolarization_factors()

        self.assertAlmostEqual(longitudinal, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(
            longitudinal + 2.0 * transverse,
            1.0,
            places=15,
        )

    def test_orientation_fixes_the_only_two_allowed_dipole_factors(self) -> None:
        self.assertEqual(orientation_factor("long"), 2.0)
        self.assertEqual(orientation_factor("trans"), -1.0)
        self.assertEqual(make_default_params("long").G, 2.0)
        self.assertEqual(make_default_params("trans").G, -1.0)

    def test_model_rejects_orientation_and_g_conflict_before_fitting(self) -> None:
        longitudinal_params = make_default_params("long")
        with self.assertRaisesRegex(ValueError, "orientation.*requires G"):
            HybridQDPlasmonModel(
                longitudinal_params,
                orientation="trans",
                n_modes=1,
                verbose=False,
            )

    def test_stability_override_cannot_mix_orientation_and_g(self) -> None:
        model = HybridQDPlasmonModel(
            make_default_params("long"),
            orientation="long",
            n_modes=9,
            verbose=False,
        )
        with self.assertRaisesRegex(ValueError, "requires G=2"):
            model.linearized_ground_state_jacobian(g_factor=-1.0)

    def test_applicability_diagnostics_separate_gap_from_point_dipole_limit(self) -> None:
        model = HybridQDPlasmonModel(
            make_default_params(),
            n_modes=9,
            verbose=False,
        )
        diagnostics = model.dipole_applicability_diagnostics(energy_eV=2.042)

        self.assertGreater(model.params.axial_surface_gap_au, 0.0)
        self.assertAlmostEqual(
            diagnostics.mnp_size_to_separation_ratio,
            15.0 / 18.0,
            places=14,
        )
        self.assertTrue(diagnostics.quasistatic_guide_satisfied)
        self.assertFalse(diagnostics.point_dipole_guide_satisfied)
        self.assertTrue(diagnostics.near_field_coupling_guide_satisfied)

    def test_finite_qd_size_participates_in_point_dipole_applicability(self) -> None:
        base = make_default_params()
        diagnostic_model = object.__new__(HybridQDPlasmonModel)
        diagnostic_model.params = replace(
            base,
            c_au=base.R_au / 10.0,
            a_au=base.R_au / 10.0,
            qd_radius_au=0.8 * base.R_au,
        )
        diagnostics = diagnostic_model.dipole_applicability_diagnostics(
            energy_eV=2.042
        )

        self.assertTrue(diagnostics.mnp_point_dipole_guide_satisfied)
        self.assertFalse(diagnostics.qd_point_dipole_guide_satisfied)
        self.assertFalse(diagnostics.point_dipole_guide_satisfied)

    def test_near_field_coupling_checks_kR_not_only_particle_size_kc(self) -> None:
        base = make_default_params()
        diagnostic_model = object.__new__(HybridQDPlasmonModel)
        diagnostic_model.params = replace(base, R_au=base.R_au * (40.0 / 18.0))
        diagnostics = diagnostic_model.dipole_applicability_diagnostics(
            energy_eV=2.042
        )

        self.assertTrue(diagnostics.particle_quasistatic_guide_satisfied)
        self.assertFalse(diagnostics.near_field_coupling_guide_satisfied)
        self.assertFalse(diagnostics.quasistatic_guide_satisfied)


class QDLocalFieldTests(unittest.TestCase):
    def test_default_keeps_the_unsourced_legacy_dipole_normalization(self) -> None:
        params = make_default_params()
        self.assertEqual(params.qd_dipole_convention, "effective_external")
        self.assertEqual(params.qd_local_field_factor, 1.0)

    def test_dipole_convention_selects_screening_without_double_counting(self) -> None:
        eps_m = 2.25
        eps_qd = 6.0
        expected_l = 3.0 * eps_m / (eps_qd + 2.0 * eps_m)
        base = make_default_params()

        bare = replace(
            base,
            eps_m=eps_m,
            eps_qd=eps_qd,
            qd_dipole_convention="bare_internal",
        )
        effective = replace(
            bare,
            qd_dipole_convention="effective_external",
        )

        self.assertAlmostEqual(bare.qd_local_field_factor, expected_l, places=15)
        self.assertEqual(effective.qd_local_field_factor, 1.0)

    def test_linear_qd_polarizability_scales_as_local_field_squared(self) -> None:
        params = replace(
            make_default_params(),
            eps_m=2.25,
            eps_qd=6.0,
            qd_dipole_convention="bare_internal",
        )
        energies_eV = np.linspace(1.7, 2.3, 101)
        beta_unscreened = qd_linear_polarizability_au(
            energies_eV,
            params.d_au,
            params.omega0_au,
            params.Gamma_au,
            local_field_factor=1.0,
        )
        beta_screened = qd_linear_polarizability_au(
            energies_eV,
            params.d_au,
            params.omega0_au,
            params.Gamma_au,
            local_field_factor=params.qd_local_field_factor,
        )

        np.testing.assert_allclose(
            beta_screened,
            params.qd_local_field_factor**2 * beta_unscreened,
            rtol=5.0e-15,
            atol=0.0,
        )


class PulseFitWindowTests(unittest.TestCase):
    def test_default_five_fs_pulse_is_covered_but_one_fs_pulse_is_not(self) -> None:
        leakage_5fs = _pulse(5.0).spectral_leakage_fraction(FIT_WINDOW_EV)
        leakage_1fs = _pulse(1.0).spectral_leakage_fraction(FIT_WINDOW_EV)

        self.assertLess(leakage_5fs, 1.0e-3)
        self.assertGreater(leakage_1fs, 1.0e-3)
        self.assertAlmostEqual(
            _pulse(5.0).positive_frequency_spectral_fraction(FIT_WINDOW_EV)
            + leakage_5fs,
            1.0,
            places=15,
        )

    def test_solver_rejects_pulse_not_covered_by_modal_fit_window(self) -> None:
        model = make_zero_mode_model()
        model.fit_window_eV = FIT_WINDOW_EV

        with self.assertRaisesRegex(ValueError, "Pulse spectrum.*fit_window"):
            model.solve(
                _pulse(1.0),
                spectral_window_policy="raise",
                max_spectral_leakage=1.0e-3,
            )

    def test_sampled_response_audit_detects_generated_out_of_window_frequency(self) -> None:
        t_au = np.linspace(-4000.0, 4000.0, 4001)
        envelope = np.exp(-0.5 * (t_au / 600.0) ** 2)
        inside_signal = envelope * np.cos(float(eV_to_au(2.0)) * t_au)
        outside_signal = envelope * np.cos(float(eV_to_au(4.0)) * t_au)

        inside_fraction = sampled_positive_frequency_spectral_fraction(
            t_au,
            inside_signal,
            FIT_WINDOW_EV,
            highest_resolved_omega_au=float(eV_to_au(4.0)),
        )
        outside_fraction = sampled_positive_frequency_spectral_fraction(
            t_au,
            outside_signal,
            FIT_WINDOW_EV,
            highest_resolved_omega_au=float(eV_to_au(4.0)),
        )
        self.assertGreater(inside_fraction, 0.999)
        self.assertLess(outside_fraction, 1.0e-3)


class PassiveModalFitTests(unittest.TestCase):
    def test_linear_rows_export_local_and_global_modal_observable_verdicts(self) -> None:
        for orientation in ("long", "trans"):
            with (
                self.subTest(orientation=orientation),
                warnings.catch_warnings(),
                patch.object(HybridQDPlasmonModel, "print_fit_summary"),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                rows = compute_spectrum(
                    energy_min_ev=2.0,
                    energy_max_ev=2.08,
                    points=201,
                    n_modes=9,
                    fit_window_ev=FIT_WINDOW_EV,
                    weight_center_ev=None,
                    weight_sigma_ev=None,
                    c_nm=None,
                    a_nm=None,
                    r_nm=18.0,
                    qd_radius_nm=None,
                    g_factor=None,
                    eps_m=None,
                    d_debye=None,
                    omega0_ev=None,
                    gamma_population_mev=None,
                    gamma2_coherence_mev=None,
                    orientation=orientation,
                )

                first = rows[0]
                local_max = max(
                    max(
                        row["modal_vs_material_coupled_local_relative_error"]
                        for row in rows
                    ),
                    max(
                        row["modal_vs_material_bare_local_relative_error"]
                        for row in rows
                    ),
                )
                self.assertAlmostEqual(
                    first["modal_observable_local_relative_max_error"],
                    local_max,
                    places=14,
                )
                expected_verdict = bool(
                    first["modal_observable_global_scale_normalized_max_error"]
                    <= MODAL_OBSERVABLE_NORMALIZED_MAX_ERROR_LIMIT
                    and local_max <= MODAL_OBSERVABLE_LOCAL_RELATIVE_ERROR_LIMIT
                )
                self.assertTrue(expected_verdict)
                self.assertEqual(
                    first["modal_observable_converged"],
                    expected_verdict,
                )
                self.assertTrue(first["spectral_grid_resolved"])

    def test_linear_output_grid_must_resolve_gamma2_before_fitting(self) -> None:
        with patch.object(HybridQDPlasmonModel, "__init__") as constructor:
            with self.assertRaisesRegex(ValueError, "Gamma2/4.*at least"):
                compute_spectrum(
                    energy_min_ev=2.0,
                    energy_max_ev=2.08,
                    points=2,
                    n_modes=9,
                    fit_window_ev=FIT_WINDOW_EV,
                    weight_center_ev=None,
                    weight_sigma_ev=None,
                    c_nm=None,
                    a_nm=None,
                    r_nm=18.0,
                    qd_radius_nm=None,
                    g_factor=None,
                    eps_m=None,
                    d_debye=None,
                    omega0_ev=None,
                    gamma_population_mev=None,
                    gamma2_coherence_mev=None,
                    orientation="long",
                )
        constructor.assert_not_called()

    def test_fit_window_cannot_exceed_tabulated_material_data(self) -> None:
        params = make_default_params()
        with self.assertRaisesRegex(ValueError, "tabulated material-data"):
            HybridQDPlasmonModel(
                params,
                n_modes=9,
                fit_window_eV=(0.5, 3.0),
                verbose=False,
            )

    def test_modal_fit_requires_more_independent_data_than_fit_coordinates(self) -> None:
        params = make_default_params()
        with self.assertRaisesRegex(
            ValueError,
            "Too few independent tabulated points.*real constraints",
        ):
            HybridQDPlasmonModel(
                params,
                n_modes=11,
                fit_window_eV=FIT_WINDOW_EV,
                verbose=False,
            )

    def test_bundled_production_fits_pass_accuracy_and_passivity_gates(self) -> None:
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                model = HybridQDPlasmonModel(
                    make_default_params(orientation),
                    orientation=orientation,
                    n_modes=9,
                    alpha_objective_weight=1.0,
                    inv_alpha_objective_weight=1.2,
                    verbose=False,
                )
                self.assertLessEqual(model.fit.normalized_rms_alpha, 0.025)
                self.assertLessEqual(model.fit.normalized_rms_inv_alpha, 0.025)
                self.assertLessEqual(model.fit.max_normalized_alpha_error, 0.05)
                self.assertTrue(np.all(model.fit.strengths_au2 >= 0.0))
                self.assertEqual(model.fit.alpha_inf, model.physical_alpha_infinity)
                self.assertEqual(model.fit.alpha_inf, 0.0)
                self.assertLessEqual(
                    float(np.max(model.fit.omega_modes_au)),
                    float(eV_to_au(model.params.material.energy_eV[-1]))
                    * (1.0 + 1.0e-12),
                )
                self.assertGreaterEqual(float(np.min(model.alpha_tab.imag)), 0.0)
                self.assertTrue(model.fit.passive_for_all_positive_frequencies)

                # Recompute accuracy independently on an offset grid much
                # denser than both the JC table and the optimizer grid.  This
                # prevents a narrow between-node resonance from passing merely
                # because the stored fit metrics sampled the same nodes.
                edges = np.linspace(FIT_WINDOW_EV[0], FIT_WINDOW_EV[1], 10_002)
                audit_energies = 0.5 * (edges[:-1] + edges[1:])
                target = model.alpha_from_material(audit_energies)
                fitted = model.alpha_from_fit(audit_energies)
                self.assertGreaterEqual(float(np.min(target.imag)), 0.0)
                self.assertGreaterEqual(float(np.min(fitted.imag)), -1.0e-15)
                independent_nrms = float(
                    np.sqrt(np.mean(np.abs(fitted - target) ** 2))
                    / np.sqrt(np.mean(np.abs(target) ** 2))
                )
                independent_inv_nrms = float(
                    np.sqrt(np.mean(np.abs(1.0 / fitted - 1.0 / target) ** 2))
                    / np.sqrt(np.mean(np.abs(1.0 / target) ** 2))
                )
                independent_max_relative = float(
                    np.max(np.abs(fitted - target) / np.maximum(np.abs(target), 1e-15))
                )
                self.assertLessEqual(independent_nrms, 0.025)
                self.assertLessEqual(independent_inv_nrms, 0.025)
                self.assertLessEqual(independent_max_relative, 0.05)

    def test_default_time_step_resolves_the_fastest_modal_or_hybrid_frequency(self) -> None:
        model = HybridQDPlasmonModel(
            make_default_params("long"),
            orientation="long",
            n_modes=9,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        pulse = _pulse(5.0)
        result = model.solve(
            pulse,
            t_span_au=(
                -8.0 * pulse.sigma_t_au,
                float(fs_to_au(20.0)),
            ),
            method="DOP853",
            rtol=1.0e-8,
            atol=1.0e-10,
            spectral_window_policy="ignore",
        )
        fastest_expected = max(
            pulse.omegaL_au,
            model.params.omega0_au,
            float(np.max(model.fit.omega_modes_au)),
            float(np.max(np.abs(model.linear_stability.poles_au))),
        )
        expected_step_limit = 2.0 * np.pi / (20.0 * fastest_expected)
        self.assertAlmostEqual(
            result.diagnostics.integration_frequency_ceiling_au,
            fastest_expected,
            places=14,
        )
        self.assertLessEqual(
            result.diagnostics.max_step_limit_au,
            expected_step_limit * (1.0 + 1.0e-14),
        )
        self.assertLessEqual(
            result.diagnostics.max_step_au,
            result.diagnostics.max_step_limit_au * (1.0 + 1.0e-10),
        )

    def test_strong_field_time_step_also_resolves_the_peak_rabi_frequency(self) -> None:
        model = make_zero_mode_model()
        pulse = GaussianPulse(
            E0_au=1.0,
            omegaL_au=float(eV_to_au(2.0)),
            tau_au=float(fs_to_au(0.1)),
            tau_kind="fwhm_intensity",
        )
        result = model.solve(
            pulse,
            t_span_au=(-8.0 * pulse.sigma_t_au, 8.0 * pulse.sigma_t_au),
            method="DOP853",
            rtol=1.0e-8,
            atol=1.0e-10,
            spectral_window_policy="ignore",
        )

        expected_incident_rabi = (
            2.0
            * model.params.d_au
            * model.params.qd_local_field_factor
            * pulse.E0_au
        )
        self.assertAlmostEqual(
            result.diagnostics.incident_peak_rabi_frequency_au,
            expected_incident_rabi,
            places=14,
        )
        self.assertGreaterEqual(
            result.diagnostics.integration_frequency_ceiling_au,
            result.diagnostics.observed_peak_rabi_frequency_au,
        )
        self.assertLessEqual(
            result.diagnostics.max_step_limit_au,
            2.0 * np.pi / (20.0 * expected_incident_rabi),
        )

    def test_alpha_from_fit_rejects_uncontrolled_extrapolation(self) -> None:
        model = _passive_lorentz_model()
        model.fit_window_eV = FIT_WINDOW_EV

        with self.assertRaisesRegex(ValueError, "outside fit_window"):
            model.alpha_from_fit(np.array([0.79, 2.0]))
        with self.assertRaisesRegex(ValueError, "outside fit_window"):
            model.alpha_from_fit(np.array([2.0, 3.01]))

        extrapolated = model.alpha_from_fit(
            np.array([0.79, 3.01]),
            allow_extrapolation=True,
        )
        self.assertTrue(np.all(np.isfinite(extrapolated)))

    def test_positive_lorentz_residues_are_passive_on_dense_frequency_grid(self) -> None:
        model = _passive_lorentz_model()
        energies_eV = np.geomspace(
            model.fit_window_eV[0],
            model.fit_window_eV[1],
            25_001,
        )
        alpha = model.alpha_from_fit(energies_eV)

        self.assertTrue(np.all(model.fit.strengths_au2 >= 0.0))
        self.assertTrue(np.all(model.fit.omega_modes_au > 0.0))
        self.assertTrue(np.all(model.fit.gamma_modes_au > 0.0))
        self.assertGreaterEqual(float(np.min(alpha.imag)), -1.0e-15)
        self.assertTrue(model.fit.passive_for_all_positive_frequencies)
        self.assertFalse(model.fit.strengths_au2.flags.writeable)

    def test_fit_container_rejects_an_active_negative_residue(self) -> None:
        with self.assertRaisesRegex(ValueError, "f_k|strength"):
            RationalLorentzFit(
                alpha_inf=0.0,
                strengths_au2=np.asarray([-1.0e-4]),
                omega_modes_au=np.asarray([0.08]),
                gamma_modes_au=np.asarray([0.01]),
                energies_used_eV=np.asarray([2.0]),
                alpha_used=np.asarray([1.0j]),
                rms_alpha=0.0,
                rms_inv_alpha=0.0,
                cost=0.0,
            )

    def test_fit_container_rejects_complex_instantaneous_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha_inf.*real"):
            RationalLorentzFit(
                alpha_inf=0.2 - 0.1j,
                strengths_au2=np.asarray([1.0e-4]),
                omega_modes_au=np.asarray([0.08]),
                gamma_modes_au=np.asarray([0.01]),
                energies_used_eV=np.asarray([2.0]),
                alpha_used=np.asarray([1.0j]),
                rms_alpha=0.0,
                rms_inv_alpha=0.0,
                cost=0.0,
            )

    def test_weighted_fit_requires_a_complete_gaussian_weight_pair(self) -> None:
        params = make_default_params()
        with self.assertRaisesRegex(ValueError, "specified together"):
            HybridQDPlasmonModel(
                params,
                n_modes=9,
                weight_center_eV=2.1,
                weight_sigma_eV=None,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
