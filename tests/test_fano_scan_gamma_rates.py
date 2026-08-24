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
        pass

    def alpha_from_fit(self, energies_ev: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(energies_ev).shape, 1.0j, dtype=complex)

    def linearized_ground_state_stability(self, **kwargs):
        return SimpleNamespace(stable=True, spectral_abscissa_au=-1.0e-4)


class _UnstableFakeModel(_FakeModel):
    def linearized_ground_state_stability(self, **kwargs):
        return SimpleNamespace(stable=False, spectral_abscissa_au=1.0e-4)


class FanoScanGammaRateTests(unittest.TestCase):
    def _scan_kwargs(self, **gamma_kwargs):
        kwargs = dict(
            target_ev=2.042,
            window_ev=0.001,
            grid_points=3,
            omega0_min_ev=2.042,
            omega0_max_ev=2.042,
            omega0_points=1,
            d_debye_values=[13.9],
            g_min=2.0,
            g_max=2.0,
            g_points=1,
            g_spacing="linear",
            fit_window_ev=(0.8, 3.0),
            weight_center_ev=None,
            weight_sigma_ev=None,
            eps_m=None,
            r_nm=None,
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

    def test_valid_gamma_pair_reaches_scan_and_is_saved_in_row(self) -> None:
        rows = self._scan(gamma2_coherence_mev_values=[2.78])

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["gamma_population_mev"], 3.02, places=12)
        self.assertEqual(rows[0]["gamma2_coherence_mev"], 2.78)
        self.assertAlmostEqual(rows[0]["gamma_pure_dephasing_mev"], 1.27, places=12)
        self.assertEqual(rows[0]["gamma_dephasing_mev"], 2.78)

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


if __name__ == "__main__":
    unittest.main()
