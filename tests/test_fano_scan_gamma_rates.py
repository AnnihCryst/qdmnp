"""Fast Fano-scan tests for the gamma1/Gamma2 physical contract."""

from unittest.mock import patch
from types import SimpleNamespace
import unittest

import numpy as np

import qd_mnp_fano_scan as fano


class _FakeModel:
    """Minimal passive MNP response that avoids the expensive rational fit."""

    C = 1.0

    def __init__(self, *args, **kwargs) -> None:
        self.n_modes = int(kwargs.get("n_modes", 1))
        self.fit = SimpleNamespace(
            alpha_inf=0.0,
            normalized_rms_alpha=0.01,
            normalized_rms_inv_alpha=0.01,
            max_normalized_alpha_error=0.02,
            passive_for_all_positive_frequencies=True,
            nonnegative_imaginary_part_all_positive_frequencies=True,
        )

    def alpha_from_fit(self, energies_ev: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(energies_ev).shape, 1.0j, dtype=complex)

    def alpha_from_material(self, energies_ev: np.ndarray) -> np.ndarray:
        return self.alpha_from_fit(energies_ev)

    def linearized_ground_state_stability(self, **kwargs):
        return SimpleNamespace(stable=True, spectral_abscissa_au=-1.0e-4)

    def dipole_applicability_diagnostics(self, **kwargs):
        return SimpleNamespace(
            medium_size_parameter_kc=0.1,
            particle_quasistatic_guide_satisfied=True,
            quasistatic_guide_satisfied=True,
            guide_threshold=0.3,
        )


class _UnstableFakeModel(_FakeModel):
    def linearized_ground_state_stability(self, **kwargs):
        return SimpleNamespace(stable=False, spectral_abscissa_au=1.0e-4)


class FanoScanGammaRateTests(unittest.TestCase):
    def _scan_kwargs(self, **gamma_kwargs):
        kwargs = dict(
            target_ev=2.042,
            window_ev=0.0004,
            grid_points=3,
            omega0_min_ev=2.042,
            omega0_max_ev=2.042,
            omega0_points=1,
            d_debye_values=[13.9],
            r_min_nm=20.0,
            r_max_nm=20.0,
            r_points=1,
            r_spacing="linear",
            fit_window_ev=(0.8, 3.0),
            weight_center_ev=None,
            weight_sigma_ev=None,
            eps_m=None,
            c_nm=None,
            a_nm=None,
            gamma_population_mev=3.02,
        )
        kwargs.update(gamma_kwargs)
        return kwargs

    def _scan(self, **gamma_kwargs):
        kwargs = self._scan_kwargs(**gamma_kwargs)
        with (
            patch.object(fano, "HybridQDPlasmonModel", _FakeModel),
            patch.object(
                fano,
                "qd_linear_polarizability_au",
                side_effect=lambda omega, *args: np.zeros_like(omega, dtype=complex),
            ),
        ):
            return fano.scan_candidates(**kwargs)

    def test_gamma2_below_half_population_width_is_rejected_before_fit(self) -> None:
        with patch.object(fano, "HybridQDPlasmonModel") as model_class:
            with self.assertRaisesRegex(ValueError, "Gamma2.*gamma1/2"):
                fano.scan_candidates(
                    **self._scan_kwargs(gamma2_coherence_mev_values=[1.27])
                )
            model_class.assert_not_called()

    def test_even_energy_grid_is_rejected_instead_of_mislabeling_a_nearby_point(self) -> None:
        kwargs = self._scan_kwargs(gamma2_coherence_mev_values=[2.78])
        kwargs["grid_points"] = 4
        with patch.object(fano, "HybridQDPlasmonModel") as model_class:
            with self.assertRaisesRegex(ValueError, "odd.*target_ev"):
                fano.scan_candidates(**kwargs)
        model_class.assert_not_called()

    def test_valid_gamma_pair_reaches_scan_and_is_saved_in_row(self) -> None:
        rows = self._scan(gamma2_coherence_mev_values=[2.78])

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["gamma_population_mev"], 3.02, places=12)
        self.assertEqual(rows[0]["gamma2_coherence_mev"], 2.78)
        self.assertAlmostEqual(rows[0]["gamma_pure_dephasing_mev"], 1.27, places=12)
        self.assertEqual(rows[0]["gamma_dephasing_mev"], 2.78)
        self.assertEqual(rows[0]["G"], 2.0)
        self.assertEqual(rows[0]["R_nm"], 20.0)
        self.assertGreater(rows[0]["surface_gap_nm"], 0.0)
        self.assertTrue(rows[0]["modal_observable_converged"])
        self.assertTrue(rows[0]["accepted_for_modal_numerical_ranking"])
        self.assertFalse(rows[0]["suppression_at_target"])
        self.assertFalse(rows[0]["material_reference_suppression_at_target"])
        self.assertFalse(
            rows[0]["suppression_at_target_confirmed_by_material_reference"]
        )
        self.assertFalse(rows[0]["accepted_for_fano_like_suppression_ranking"])
        self.assertFalse(rows[0]["quantitative_physical_applicability"])
        self.assertFalse(rows[0]["accepted_for_quantitative_ranking"])
        self.assertEqual(rows[0]["modal_observable_normalized_max_error"], 0.0)
        self.assertEqual(
            rows[0]["ratio_qs_work_loss_at_target"],
            rows[0]["ratio_qs_work_loss_material_reference_at_target"],
        )

    def test_underresolved_scan_is_saved_but_excluded_from_quantitative_top(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "scan step"):
            rows = self._scan(
                gamma2_coherence_mev_values=[2.78],
                window_ev=0.08,
                grid_points=3,
            )

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["scan_grid_resolved"])
        self.assertFalse(rows[0]["accepted_for_modal_numerical_ranking"])
        self.assertFalse(rows[0]["accepted_for_fano_like_suppression_ranking"])
        self.assertFalse(rows[0]["accepted_for_quantitative_ranking"])
        with patch("builtins.print") as mocked_print:
            fano.print_top(rows, 1)
        printed = " ".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn("No row", printed)
        self.assertNotIn("ratio@target=", printed)

    def test_scan_step_must_also_resolve_the_qd_coherence_width(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "Gamma2/4"):
            rows = self._scan(
                gamma2_coherence_mev_values=[0.05],
                gamma_population_mev=0.1,
            )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["scan_absolute_grid_resolved"])
        self.assertFalse(rows[0]["scan_gamma2_relative_grid_resolved"])
        self.assertFalse(rows[0]["scan_grid_resolved"])
        self.assertAlmostEqual(
            rows[0]["scan_max_energy_step_ev"],
            0.0000125,
            places=14,
        )

    def test_legacy_python_keyword_remains_an_alias(self) -> None:
        rows = self._scan(gamma_dephasing_mev_values=[2.78])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gamma2_coherence_mev"], 2.78)

    def test_conflicting_canonical_and_legacy_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self._scan(
                gamma2_coherence_mev_values=[2.78],
                gamma_dephasing_mev_values=[3.02],
            )

    def test_cli_forwards_population_width_and_canonical_gamma2_values(self) -> None:
        argv = [
            "qd_mnp_fano_scan.py",
            "--gamma-population-mev",
            "3.02",
            "--gamma2-coherence-mev-values",
            "1.51",
            "2.78",
        ]
        with patch("sys.argv", argv):
            args = fano.parse_args()

        self.assertEqual(args.gamma_population_mev, 3.02)
        self.assertEqual(args.gamma2_coherence_mev_values, [1.51, 2.78])

    def test_legacy_cli_option_maps_to_canonical_destination(self) -> None:
        with patch(
            "sys.argv",
            ["qd_mnp_fano_scan.py", "--gamma-dephasing-mev-values", "2.78"],
        ):
            args = fano.parse_args()

        self.assertEqual(args.gamma2_coherence_mev_values, [2.78])

    def test_conflicting_cli_lists_are_rejected(self) -> None:
        argv = [
            "qd_mnp_fano_scan.py",
            "--gamma2-coherence-mev-values",
            "2.78",
            "--gamma-dephasing-mev-values",
            "3.02",
        ]
        with patch("sys.argv", argv), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                fano.parse_args()

    def test_unstable_coupled_candidate_is_excluded(self) -> None:
        with (
            patch.object(fano, "HybridQDPlasmonModel", _UnstableFakeModel),
            patch.object(
                fano,
                "qd_linear_polarizability_au",
                side_effect=lambda omega, *args: np.zeros_like(omega, dtype=complex),
            ),
            self.assertWarnsRegex(RuntimeWarning, "Excluded.*Jacobian"),
        ):
            rows = fano.scan_candidates(
                **self._scan_kwargs(gamma2_coherence_mev_values=[2.78])
            )
        self.assertEqual(rows, [])

    def test_negative_coupled_qs_work_loss_is_reported_not_silently_filtered(self) -> None:
        grid_shape = (3,)
        with (
            patch.object(fano, "HybridQDPlasmonModel", _FakeModel),
            patch.object(
                fano,
                "qd_linear_polarizability_au",
                side_effect=lambda omega, *args: np.zeros_like(omega, dtype=complex),
            ),
            patch.object(
                fano,
                "quasistatic_work_loss_cross_section_cm2",
                side_effect=[
                    np.ones(grid_shape),
                    np.ones(grid_shape),
                    -np.ones(grid_shape),
                    -np.ones(grid_shape),
                ],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "negative QS work loss"):
                fano.scan_candidates(
                    **self._scan_kwargs(gamma2_coherence_mev_values=[2.78])
                )

    def test_negative_direct_material_reference_is_also_rejected(self) -> None:
        grid_shape = (3,)
        with (
            patch.object(fano, "HybridQDPlasmonModel", _FakeModel),
            patch.object(
                fano,
                "qd_linear_polarizability_au",
                side_effect=lambda omega, *args: np.zeros_like(omega, dtype=complex),
            ),
            patch.object(
                fano,
                "quasistatic_work_loss_cross_section_cm2",
                side_effect=[
                    np.ones(grid_shape),
                    np.ones(grid_shape),
                    np.ones(grid_shape),
                    -np.ones(grid_shape),
                ],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "direct-material.*non-negative"):
                fano.scan_candidates(
                    **self._scan_kwargs(gamma2_coherence_mev_values=[2.78])
                )


if __name__ == "__main__":
    unittest.main()
