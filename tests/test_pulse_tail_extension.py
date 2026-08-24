"""Robust pulse-tail diagnostics and automatic post-pulse extension."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import qd_mnp_pulse_absorption_sweep as sweep
from qd_mnp_rational_fit import DipoleCrossSections, fs_to_au
from tests._fixtures import make_zero_mode_model


class ResponseTailRatioTests(unittest.TestCase):
    def test_simple_call_preserves_sample_based_schema1_behavior(self) -> None:
        values = np.linspace(-2.0, 3.0, 20)
        n_tail = 8  # max(8, ceil(5% * 20))
        expected = np.sqrt(np.mean(values[-n_tail:] ** 2)) / np.max(np.abs(values))
        self.assertEqual(sweep.response_tail_ratio(values), float(expected))

    def test_time_weighted_rms_is_independent_of_sample_clustering(self) -> None:
        sparse_t = np.array([0.0, 4.0, 8.0, 9.0, 10.0])
        clustered_t = np.array([0.0, 4.0, 8.0, 8.01, 8.02, 8.2, 9.7, 9.99, 10.0])
        sparse_mu = np.sqrt(sparse_t + 1.0)
        clustered_mu = np.sqrt(clustered_t + 1.0)

        # Over [8, 10], the time average of mu**2=t+1 is exactly 10.
        expected = np.sqrt(10.0 / 11.0)
        sparse_ratio = sweep.response_tail_ratio(
            sparse_mu,
            sparse_t,
            tail_fraction=0.2,
        )
        clustered_ratio = sweep.response_tail_ratio(
            clustered_mu,
            clustered_t,
            tail_fraction=0.2,
        )
        self.assertAlmostEqual(sparse_ratio, expected, places=14)
        self.assertAlmostEqual(clustered_ratio, expected, places=14)

    def test_component_max_detects_tail_hidden_in_total_dipole(self) -> None:
        t_au = np.linspace(0.0, 10.0, 101)
        mu_total = np.zeros_like(t_au)
        mu_total[0] = 1.0
        mu_p = np.ones_like(t_au)
        mu_d = -mu_p

        ratio = sweep.response_tail_ratio(mu_total, t_au, mu_p, mu_d)
        self.assertEqual(ratio, 1.0)


class ComputeSweepTailExtensionTests(unittest.TestCase):
    @staticmethod
    def _sections() -> DipoleCrossSections:
        return DipoleCrossSections(
            quasistatic_work_loss_cm2=np.asarray(2.0),
            rayleigh_scattering_estimate_cm2=np.asarray(0.5),
            optical_theorem_residual_cm2=np.asarray(1.5),
        )

    @staticmethod
    def _result(
        t_span_au: tuple[float, float],
        *,
        lingering_tail: bool,
        drive_leakage: float = 0.0,
        mnp_dipole_leakage: float | None = None,
    ):
        if mnp_dipole_leakage is None:
            mnp_dipole_leakage = drive_leakage
        t_au = np.linspace(t_span_au[0], t_span_au[1], 101)
        if lingering_tail:
            mu_p = np.ones_like(t_au)
        else:
            mu_p = np.zeros_like(t_au)
            mu_p[:50] = 1.0
        mu_d = np.zeros_like(t_au)
        mu_total = mu_p + mu_d
        diagnostics = SimpleNamespace(
            n_steps=t_au.size,
            nfev=10,
            solver_success=True,
            t_final_reached=True,
            pulse_spectral_fraction_in_fit_window=1.0,
            pulse_spectral_leakage=0.0,
            mnp_drive_spectral_fraction_in_fit_window=1.0 - drive_leakage,
            mnp_drive_spectral_leakage=drive_leakage,
            mnp_dipole_spectral_fraction_in_fit_window=1.0 - mnp_dipole_leakage,
            mnp_dipole_spectral_leakage=mnp_dipole_leakage,
            work_passivity_checked=True,
            work_passivity_tolerance_au=1.0e-12,
            work_nonnegative_within_tolerance=True,
        )
        return SimpleNamespace(
            t_au=t_au,
            y=np.vstack((-np.ones_like(t_au), np.zeros_like(t_au), np.zeros_like(t_au))),
            mu_p_au=mu_p,
            mu_d_au=mu_d,
            mu_total_au=mu_total,
            mu_dot_total_au=np.zeros_like(t_au),
            peak_intensity_w_cm2=1.0,
            fluence_j_cm2=1.0,
            sigma_energy_transfer_cm2=2.0,
            work_from_incident_field_j=2.0,
            max_bloch_radius=1.0,
            min_density_eigenvalue=0.0,
            diagnostics=diagnostics,
        )

    def _compute(
        self,
        model,
        *,
        post_fs: float | None,
        spectral_window_policy: str = "ignore",
    ):
        params = make_zero_mode_model().params
        if not hasattr(model, "linear_stability"):
            model.linear_stability = SimpleNamespace(stable=True, spectral_abscissa_au=-1.0e-4)
        if not hasattr(model, "fit"):
            model.fit = SimpleNamespace(
                alpha_inf=0.0,
                strengths_au2=np.asarray([], dtype=float),
                omega_modes_au=np.asarray([], dtype=float),
                gamma_modes_au=np.asarray([], dtype=float),
                normalized_rms_alpha=0.0,
                normalized_rms_inv_alpha=0.0,
                max_normalized_alpha_error=0.0,
                passive_for_all_positive_frequencies=True,
                nonnegative_imaginary_part_all_positive_frequencies=True,
            )
        model.n_modes = 0
        if not hasattr(model, "dipole_applicability_diagnostics"):
            model.dipole_applicability_diagnostics = lambda **kwargs: SimpleNamespace(
                medium_size_parameter_kc=0.1,
                medium_separation_parameter_kR=0.15,
                mnp_size_to_separation_ratio=0.2,
                qd_size_to_separation_ratio=0.05,
                particle_quasistatic_guide_satisfied=True,
                near_field_coupling_guide_satisfied=True,
                quasistatic_guide_satisfied=True,
                mnp_point_dipole_guide_satisfied=True,
                qd_point_dipole_guide_satisfied=True,
                point_dipole_guide_satisfied=True,
            )
        with (
            patch.object(sweep, "make_params_with_overrides", return_value=params) as make_params,
            patch.object(sweep, "HybridQDPlasmonModel", return_value=model),
            patch.object(sweep, "bare_mnp_spectral_cross_sections_cm2", return_value=self._sections()),
            patch.object(sweep, "spectral_cross_sections_cm2", return_value=self._sections()),
        ):
            rows, traces, returned_params = sweep.compute_sweep(
                tau_values_fs=[5.0],
                e0_values_v_m=np.asarray([1.0e5]),
                omega_l_ev=2.0,
                n_modes=0,
                fit_window_ev=(1.0, 3.0),
                weight_center_ev=None,
                weight_sigma_ev=None,
                method="DOP853",
                rtol=1.0e-8,
                atol=1.0e-10,
                pre_sigma=None,
                post_fs=post_fs,
                c_nm=None,
                a_nm=None,
                r_nm=None,
                g_factor=None,
                eps_m=None,
                d_debye=None,
                omega0_ev=None,
                gamma_population_mev=None,
                gamma2_coherence_mev=2.0,
                spectral_window_policy=spectral_window_policy,
            )
        self.assertIs(returned_params, params)
        self.assertEqual(make_params.call_args.kwargs["qd_radius_nm"], None)
        self.assertEqual(make_params.call_args.kwargs["gamma2_coherence_mev"], 2.0)
        self.assertEqual(make_params.call_args.kwargs["gamma_dephasing_mev"], None)
        return rows, traces

    def test_automatic_tail_doubles_and_resolves(self) -> None:
        solve_spans: list[tuple[float, float]] = []

        def solve(_pulse, **kwargs):
            t_span = kwargs["t_span_au"]
            solve_spans.append(t_span)
            return self._result(t_span, lingering_tail=len(solve_spans) == 1)

        model = SimpleNamespace(
            recommended_post_pulse_time_au=lambda decay_times: float(fs_to_au(10.0)),
            solve=solve,
        )
        rows, traces = self._compute(model, post_fs=None)

        self.assertEqual(len(solve_spans), 2)
        self.assertAlmostEqual(solve_spans[1][1], 2.0 * solve_spans[0][1])
        self.assertEqual(rows[0]["tail_extension_count"], 1)
        self.assertTrue(rows[0]["tail_below_tolerance"])
        self.assertEqual(rows[0]["response_tail_ratio"], 0.0)
        self.assertEqual(len(traces), 1)
        np.testing.assert_array_equal(traces[0]["t_au"], np.linspace(*solve_spans[-1], 101))

    def test_explicit_post_time_is_diagnostic_only(self) -> None:
        solve_spans: list[tuple[float, float]] = []

        def solve(_pulse, **kwargs):
            t_span = kwargs["t_span_au"]
            solve_spans.append(t_span)
            return self._result(t_span, lingering_tail=True)

        model = SimpleNamespace(
            recommended_post_pulse_time_au=lambda decay_times: float(fs_to_au(10.0)),
            solve=solve,
        )
        with self.assertWarnsRegex(RuntimeWarning, "explicit --post-fs is diagnostic only"):
            rows, _ = self._compute(model, post_fs=10.0)

        self.assertEqual(len(solve_spans), 1)
        self.assertEqual(rows[0]["tail_extension_count"], 0)
        self.assertFalse(rows[0]["tail_below_tolerance"])

    def test_unconverged_automatic_tail_is_not_returned_as_production_data(self) -> None:
        def solve(_pulse, **kwargs):
            return self._result(kwargs["t_span_au"], lingering_tail=True)

        model = SimpleNamespace(
            recommended_post_pulse_time_au=lambda decay_times: float(fs_to_au(10.0)),
            solve=solve,
        )
        with self.assertRaisesRegex(RuntimeError, "observables are not converged"):
            self._compute(model, post_fs=None)

    def test_drive_spectrum_verdict_waits_for_automatic_tail_convergence(self) -> None:
        solve_policies: list[str] = []

        def solve(_pulse, **kwargs):
            solve_policies.append(kwargs["spectral_window_policy"])
            return self._result(
                kwargs["t_span_au"],
                lingering_tail=len(solve_policies) == 1,
                drive_leakage=0.5,
            )

        model = SimpleNamespace(
            recommended_post_pulse_time_au=lambda decay_times: float(fs_to_au(10.0)),
            solve=solve,
        )
        with self.assertRaisesRegex(
            ValueError,
            "converged self-consistent field",
        ):
            self._compute(
                model,
                post_fs=None,
                spectral_window_policy="raise",
            )

        # The first truncated FFT is diagnostic only; the tail gets one chance
        # to converge before the final spectrum can invalidate production data.
        self.assertEqual(solve_policies, ["ignore", "ignore"])

    def test_mnp_output_spectrum_is_enforced_even_when_drive_is_covered(self) -> None:
        def solve(_pulse, **kwargs):
            return self._result(
                kwargs["t_span_au"],
                lingering_tail=False,
                drive_leakage=0.0,
                mnp_dipole_leakage=0.5,
            )

        model = SimpleNamespace(
            recommended_post_pulse_time_au=lambda decay_times: float(fs_to_au(10.0)),
            solve=solve,
        )
        with self.assertRaisesRegex(ValueError, "MNP dipole response"):
            self._compute(
                model,
                post_fs=None,
                spectral_window_policy="raise",
            )


if __name__ == "__main__":
    unittest.main()
