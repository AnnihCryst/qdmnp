"""Regression tests for first-lobe thresholds and auditable pulse diagnostics."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from qdmnp.observables import gap_metrics_common as gap
from qdmnp.observables.threshold_metrics import threshold_from_curve


@dataclass
class _LegacyDiagnostics:
    nfev: int = 17
    integration_frequency_ceiling_au: float = 0.1
    pulse_spectral_leakage: float = 0.0
    mnp_drive_spectral_leakage: float = 0.0
    mnp_dipole_spectral_leakage: float = 0.0


class ArticleTemporalCertificateTests(unittest.TestCase):
    def test_threshold_handles_plateaus_and_descending_left_boundary(self) -> None:
        fluence = np.arange(1.0, 7.0) ** 2
        # A plateau inside a still-rising lobe must not truncate the search.
        value, status, bracket = threshold_from_curve(fluence, [0.1, 0.3, 0.3, 0.6, 0.4, 0.8], 0.5)
        self.assertEqual(status, "resolved")
        self.assertEqual(bracket, (2, 3))
        self.assertTrue(np.isfinite(value))
        value, status, _ = threshold_from_curve(fluence, [0.4, 0.3, 0.4, 0.8, 0.6, 0.4], 0.5)
        self.assertEqual(status, "left_lobe_censored")
        self.assertTrue(np.isnan(value))

    def test_publication_spectral_policy_is_strict_and_explicit_override_survives(self) -> None:
        for preset, expected in (("publication", "raise"), ("quick", "warn")):
            args = gap.parse_threshold_calculation_args(["--output", "unused.npz", "--preset", preset])
            self.assertEqual(args.spectral_window_policy, expected)
        args = gap.parse_threshold_calculation_args(["--output", "unused.npz", "--spectral-window-policy", "warn"])
        self.assertEqual(args.spectral_window_policy, "warn")

    def _legacy_fixture(self):
        args = gap.parse_threshold_calculation_args(["--output", "unused.npz", "--preset", "quick", "--tail-policy", "ignore"])
        times = np.linspace(0.0, 100.0, 201)
        state = np.zeros((5, times.size))
        state[2] = -1.0
        # Q coherence survives although every emitted dipole is zero.
        state[3] = 0.01
        result = SimpleNamespace(
            y=state, t_au=times, mu_p_au=np.zeros_like(times),
            mu_d_au=np.zeros_like(times), mu_total_au=np.zeros_like(times),
            diagnostics=_LegacyDiagnostics(),
        )
        model = SimpleNamespace(n_modes=1, J=2.0, solve=Mock(return_value=result))
        # Time-step enforcement has its own tests; these fixtures deliberately
        # isolate tail/spectral diagnostics from integration.
        resolver = patch.object(gap, "solve_dd_with_resolution", return_value=result)
        resolver.start()
        self.addCleanup(resolver.stop)
        bundle = SimpleNamespace(dd_model=model, spec=SimpleNamespace(channel_id="axis_long"), gap_nm=1.0)
        return args, bundle, result

    def test_dd_tail_cannot_default_to_success_when_coherence_remains(self) -> None:
        args, bundle, _ = self._legacy_fixture()
        with patch.object(gap, "sampled_positive_frequency_spectral_fraction", return_value=1.0):
            _, diagnostic = gap._solve_population("dd", bundle, object(), (0.0, 100.0), args)
        self.assertFalse(diagnostic["tail_converged"])
        self.assertAlmostEqual(diagnostic["tail_ratio"], 1.0)
        self.assertFalse(diagnostic["solver_certificate"]["response_tail_converged"])
        self.assertEqual(diagnostic["solver_certificate"]["qd_source_spectral_leakage"], 0.0)

        args.tail_policy = "raise"
        with patch.object(gap, "sampled_positive_frequency_spectral_fraction", return_value=1.0):
            with self.assertRaisesRegex(RuntimeError, "DD response tail did not converge"):
                gap._solve_population("dd", bundle, object(), (0.0, 100.0), args)

    def test_dd_tail_tests_cancelled_components_separately(self) -> None:
        args, bundle, result = self._legacy_fixture()
        result.y[3] = 0.0
        result.mu_p_au[:] = 0.01
        result.mu_d_au[:] = -0.01
        with patch.object(gap, "sampled_positive_frequency_spectral_fraction", return_value=1.0):
            _, diagnostic = gap._solve_population("dd", bundle, object(), (0.0, 100.0), args)
        self.assertFalse(diagnostic["tail_converged"])
        self.assertAlmostEqual(diagnostic["tail_ratio"], 1.0)

    def test_dd_additional_spectral_certificate_obeys_policy(self) -> None:
        args, bundle, _ = self._legacy_fixture()
        args.spectral_window_policy = "raise"
        with patch.object(gap, "sampled_positive_frequency_spectral_fraction", return_value=0.5):
            with self.assertRaisesRegex(RuntimeError, "DD qd_source spectral leakage"):
                gap._solve_population("dd", bundle, object(), (0.0, 100.0), args)

    def test_certificate_serialization_never_fakes_missing_success(self) -> None:
        records = [
            {"model_id": "isolated", "diagnostic__pulse_spectral_leakage": 0.01},
            {"model_id": "dd", "diagnostic__pulse_spectral_leakage": 0.02, "diagnostic__work_nonnegative_within_tolerance": True},
            {"model_id": "fqs", "diagnostic__pulse_spectral_leakage": 0.03, "diagnostic__work_nonnegative_within_tolerance": False},
        ]
        payload = gap._solver_evaluation_payload(records)
        self.assertTrue(np.array_equal(payload["evaluation_diagnostic__work_nonnegative_within_tolerance__available"], [False, True, True]))
        self.assertTrue(np.array_equal(payload["evaluation_diagnostic__work_nonnegative_within_tolerance"], [False, True, False]))
        self.assertTrue(np.allclose(payload["evaluation_diagnostic__pulse_spectral_leakage"], [0.01, 0.02, 0.03]))
        self.assertFalse(any(value.dtype.hasobject for value in payload.values()))


if __name__ == "__main__":
    unittest.main()
