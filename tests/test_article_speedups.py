"""Speed-ups of the article chain that must leave every computed quantity unchanged."""

import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from article_observables import qd_mnp_gap_metrics_common as gm
from article_observables.qd_mnp_dd_pulse import solve_dd_with_resolution
from article_observables.qd_mnp_parallel import resolve_workers
from article_observables.qd_mnp_threshold_metrics import threshold_from_curve

AREA = 203.3  # isolated pulse area per sqrt(J/cm^2), production scenario
F_MIN, F_MAX = 5e-10, 2e-4


def curve(gain, *, amplitude=1.0, stretch=0.0):
    """Coherent two-level population; ``stretch`` makes the weak-field prediction too optimistic."""
    def population(fluence):
        theta = AREA * np.sqrt(gain * fluence) * (1.0 + stretch * np.sqrt(fluence / F_MAX))
        return amplitude * np.sin(theta / 2.0) ** 2
    return population


class BracketThresholdTests(unittest.TestCase):
    def run_search(self, population, max_area=0.35):
        args = argparse.Namespace(carrier_energy_ev=2.042, refine_threshold=True,
                                  threshold_root_xtol_sqrt_fluence=1e-14, threshold_root_rtol=1e-12)
        context = {"fluence_min": F_MIN, "fluence_max": F_MAX, "target": 0.5,
                   "isolated_population_at_min": float(np.sin(AREA*np.sqrt(F_MIN)/2)**2),
                   "area_per_sqrt_fluence": AREA, "max_area_step_rad": max_area, "t_span": (0.0, 1.0)}
        bundle = SimpleNamespace(spec=SimpleNamespace(channel_id="axis_long"), gap_nm=1.0)

        def solve(kind, bundle, fluence, t_span, args):
            return population(fluence), {"tail_converged": True, "tail_ratio": 0.0, "nfev": 1, "solver_certificate": {}}

        with patch.object(gm, "_pulse_for_fluence", side_effect=lambda fluence, *a: fluence), \
                patch.object(gm, "_solve_population", side_effect=solve):
            return gm._bracket_threshold("fqs", bundle, args, context)

    def test_strong_channel_matches_analytic_and_grid_threshold_with_few_solves(self):
        gain = 33.6
        outcome = self.run_search(curve(gain))
        exact = (np.pi / (2 * AREA * np.sqrt(gain))) ** 2
        self.assertEqual(outcome["status"], "resolved_refined")
        self.assertAlmostEqual(outcome["threshold"] / exact, 1.0, places=9)
        self.assertLessEqual(len(outcome["records"]), 25)
        grid = np.linspace(np.sqrt(F_MIN), np.sqrt(F_MAX), 257) ** 2
        grid_threshold, grid_status, _ = threshold_from_curve(grid, curve(gain)(grid), 0.5)
        self.assertEqual(grid_status, "resolved")
        self.assertAlmostEqual(outcome["threshold"] / grid_threshold, 1.0, places=2)
        self.assertLessEqual(outcome["max_observed_area_step_rad"], 0.7)

    def test_suppressed_channel_is_right_censored_without_scanning_the_grid(self):
        outcome = self.run_search(curve(0.07))
        self.assertEqual(outcome["status"], "right_censored")
        self.assertTrue(np.isnan(outcome["threshold"]))
        self.assertLessEqual(len(outcome["records"]), 12)

    def test_branch_that_turns_below_target_is_not_resolved(self):
        outcome = self.run_search(curve(33.6, amplitude=0.4))
        self.assertEqual(outcome["status"], "not_reached_first_lobe")

    def test_threshold_below_scan_is_left_censored(self):
        outcome = self.run_search(curve(3.0e6))
        self.assertEqual(outcome["status"], "left_censored")
        self.assertEqual(outcome["threshold"], F_MIN)

    def test_optimistic_prediction_is_caught_by_observed_area_audit(self):
        outcome = self.run_search(curve(4.0, stretch=6.0))
        self.assertEqual(outcome["status"], "resolved_refined")
        self.assertGreater(outcome["refinements"], 0)
        grid = np.linspace(np.sqrt(F_MIN), np.sqrt(F_MAX), 4097) ** 2
        grid_threshold, _, _ = threshold_from_curve(grid, curve(4.0, stretch=6.0)(grid), 0.5)
        self.assertAlmostEqual(outcome["threshold"] / grid_threshold, 1.0, places=3)


class ResolutionAdapterTests(unittest.TestCase):
    def test_floating_point_step_excess_is_not_an_unresolved_frequency(self):
        omega = 0.075
        pulse = SimpleNamespace(omegaL_au=omega, E0_au=1e-6)
        step = 2 * np.pi / (8 * omega)
        times = 1.0e5 + np.arange(4) * step
        times[-1] = times[-2] + step * (1 + 3e-12)  # rounding at t ~ 1e5 a.u.

        class Model:
            params = SimpleNamespace(omega0_au=omega, d_au=5.0, qd_local_field_factor=1.0)
            fit = SimpleNamespace(omega_modes_au=np.array([0.3]))
            linear_stability = SimpleNamespace(poles_au=np.array([0.4]))
            calls = []

            def solve(self, pulse, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(t_au=times, diagnostics=SimpleNamespace(integration_frequency_ceiling_au=omega))

        model = Model()
        solve_dd_with_resolution(model, pulse, points_per_fastest_cycle=8, step_frequency_policy="excited_band")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["step_frequency_policy"], "excited_band")
        self.assertEqual(model.calls[0]["points_per_fastest_cycle"], 8)
        # Legacy policy keeps the core's own 20-point minimum and all poles.
        legacy = Model()
        legacy.calls = []
        with self.assertRaises(RuntimeError):
            solve_dd_with_resolution(legacy, pulse, points_per_fastest_cycle=8)
        self.assertEqual(legacy.calls[0]["points_per_fastest_cycle"], 20.0)


class WorkerCountTests(unittest.TestCase):
    def test_worker_resolution(self):
        self.assertEqual(resolve_workers(1, 100), 1)
        self.assertEqual(resolve_workers(8, 3), 3)
        self.assertEqual(resolve_workers(4, 1), 1)
        self.assertGreaterEqual(resolve_workers(0, 1000), 1)


if __name__ == "__main__":
    unittest.main()
