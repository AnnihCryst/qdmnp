"""Nonlinear DD refinement without modifying the legacy solver equations."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from qdmnp.observables.dd_pulse import solve_dd_with_resolution


def make_model(*, observed_frequency=1.0):
    model = SimpleNamespace(
        params=SimpleNamespace(omega0_au=1.0, d_au=1.0, qd_local_field_factor=1.0),
        fit=SimpleNamespace(omega_modes_au=np.asarray([0.8])),
        linear_stability=SimpleNamespace(poles_au=np.asarray([-0.1 + 0.9j])),
    )

    def solve(pulse, *, max_step_au, **kwargs):
        # Behave like the core: resolve the observed Rabi frequency at its
        # fixed 20-point minimum, then report that observed ceiling.
        step = min(max_step_au, 2.0 * np.pi / (20.0 * observed_frequency))
        return SimpleNamespace(
            t_au=np.asarray([0.0, step, 2.0 * step]),
            diagnostics=SimpleNamespace(integration_frequency_ceiling_au=observed_frequency),
        )

    model.solve = Mock(side_effect=solve)
    return model


class DDResolutionTests(unittest.TestCase):
    def setUp(self):
        self.pulse = SimpleNamespace(omegaL_au=1.0, E0_au=0.1)

    def test_doubling_points_halves_dd_maximum_step(self):
        model = make_model()
        twenty = solve_dd_with_resolution(model, self.pulse, points_per_fastest_cycle=20)
        forty = solve_dd_with_resolution(model, self.pulse, points_per_fastest_cycle=40)
        self.assertAlmostEqual(np.max(np.diff(twenty.t_au)), 2 * np.max(np.diff(forty.t_au)))

    def test_observed_local_rabi_frequency_requires_rerun(self):
        model = make_model(observed_frequency=10.0)
        result = solve_dd_with_resolution(
            model, self.pulse, points_per_fastest_cycle=40, rtol=1e-9,
        )
        self.assertEqual(model.solve.call_count, 2)
        self.assertLessEqual(np.max(np.diff(result.t_au)), 2 * np.pi / (40 * 10.0))
        self.assertEqual(model.solve.call_args.kwargs["rtol"], 1e-9)

    def test_material_mode_and_explicit_step_are_respected(self):
        model = make_model()
        model.fit.omega_modes_au = np.asarray([100.0])
        result = solve_dd_with_resolution(model, self.pulse, points_per_fastest_cycle=40)
        self.assertLessEqual(np.max(np.diff(result.t_au)), 2 * np.pi / (40 * 100.0))
        result = solve_dd_with_resolution(
            model, self.pulse, points_per_fastest_cycle=40, max_step_au=1e-5,
        )
        self.assertLessEqual(np.max(np.diff(result.t_au)), 1e-5)

    def test_noncompliant_solver_is_rejected_after_bounded_reruns(self):
        model = make_model()
        model.solve.side_effect = lambda *a, **k: SimpleNamespace(
            t_au=np.asarray([0.0, 1.0]),
            diagnostics=SimpleNamespace(integration_frequency_ceiling_au=1.0),
        )
        with self.assertRaisesRegex(RuntimeError, "three reruns"):
            solve_dd_with_resolution(model, self.pulse, points_per_fastest_cycle=40)
        self.assertEqual(model.solve.call_count, 4)


if __name__ == "__main__":
    unittest.main()
