"""Tests for positive reduction of the dark full-QS spatial measure."""

import unittest

import numpy as np

from qd_mnp_modal_reduction import modal_measure_sha256, reduce_positive_dark_measure


def _bright_response(energies: np.ndarray) -> np.ndarray:
    # A passive one-pole material surrogate in the project's exp(-i wt)
    # convention.  Only algebraic transformation identities are tested here.
    omega = np.asarray(energies, dtype=float)
    return 0.25 + 0.8 / (1.7 - omega**2 - 0.12j * omega)


class PositiveDarkKernelReductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.L = np.asarray([0.2, 0.24, 0.31, 0.43, 0.57, 0.71])
        self.weights = np.asarray([2.0, 0.7, 0.4, 0.2, 0.08, 0.03])
        self.fit_H = _bright_response(np.linspace(0.4, 1.2, 321))
        self.audit_H = _bright_response(np.linspace(0.401, 1.199, 487))

    def test_full_node_limit_is_exact_and_preserves_positive_measure(self) -> None:
        reduction = reduce_positive_dark_measure(
            self.L,
            self.weights,
            bright_index=0,
            bright_susceptibility_fit=self.fit_H,
            bright_susceptibility_audit=self.audit_H,
            rms_tolerance=1.0e-15,
            max_tolerance=1.0e-14,
        )
        self.assertTrue(reduction.diagnostics.accepted)
        self.assertLessEqual(reduction.node_count, self.L.size - 1)
        self.assertTrue(np.all(reduction.weights_au_minus3 > 0.0))
        self.assertTrue(np.all((reduction.depolarization_nodes > 0.0)))
        self.assertTrue(np.all((reduction.depolarization_nodes < 1.0)))
        self.assertLess(reduction.diagnostics.total_weight_relative_error, 1.0e-15)
        self.assertLess(reduction.diagnostics.first_moment_relative_error, 1.0e-15)
        self.assertEqual(
            reduction.source_measure_sha256,
            modal_measure_sha256(self.L, self.weights, bright_index=0),
        )
        self.assertNotEqual(
            reduction.source_measure_sha256,
            modal_measure_sha256(
                self.L,
                np.asarray([2.0, 0.7, 0.4, 0.2, 0.08, 0.031]),
                bright_index=0,
            ),
        )

        exact = np.sum(
            self.weights[1:, None]
            * self.audit_H[None, :]
            / (1.0 + (self.L[1:] - self.L[0])[:, None] * self.audit_H[None, :]),
            axis=0,
        )
        np.testing.assert_allclose(
            reduction.evaluate_from_bright(self.audit_H),
            exact,
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    def test_adaptive_reduction_uses_independent_audit_grid(self) -> None:
        reduction = reduce_positive_dark_measure(
            self.L,
            self.weights,
            bright_index=0,
            bright_susceptibility_fit=self.fit_H,
            bright_susceptibility_audit=self.audit_H,
            rms_tolerance=2.0e-4,
            max_tolerance=1.0e-3,
        )
        diagnostics = reduction.diagnostics
        self.assertTrue(diagnostics.accepted)
        self.assertEqual(diagnostics.fit_grid_points, self.fit_H.size)
        self.assertEqual(diagnostics.audit_grid_points, self.audit_H.size)
        self.assertLess(reduction.node_count, self.L.size - 1)
        self.assertTrue(diagnostics.passive_on_audit_grid)

    def test_holdout_values_cannot_change_constructed_nodes(self) -> None:
        options = dict(
            bright_index=0,
            bright_susceptibility_fit=self.fit_H,
            rms_tolerance=2.0e-4,
            max_tolerance=1.0e-3,
            policy="ignore",
        )
        reference = reduce_positive_dark_measure(
            self.L,
            self.weights,
            bright_susceptibility_audit=self.audit_H,
            **options,
        )
        deliberately_different_holdout = 1.7 * self.audit_H + 0.4j
        changed_holdout = reduce_positive_dark_measure(
            self.L,
            self.weights,
            bright_susceptibility_audit=deliberately_different_holdout,
            **options,
        )
        self.assertEqual(
            changed_holdout.source_mode_indices,
            reference.source_mode_indices,
        )
        np.testing.assert_array_equal(
            changed_holdout.depolarization_nodes,
            reference.depolarization_nodes,
        )
        np.testing.assert_array_equal(
            changed_holdout.weights_au_minus3,
            reference.weights_au_minus3,
        )

    def test_strict_gate_rejects_too_few_nodes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            reduce_positive_dark_measure(
                self.L,
                self.weights,
                bright_index=0,
                bright_susceptibility_fit=self.fit_H,
                bright_susceptibility_audit=self.audit_H,
                rms_tolerance=1.0e-12,
                max_tolerance=1.0e-12,
                max_nodes=1,
            )

    def test_bright_only_measure_has_an_empty_valid_dark_reduction(self) -> None:
        reduction = reduce_positive_dark_measure(
            np.asarray([0.3]),
            np.asarray([2.0]),
            bright_index=0,
            bright_susceptibility_fit=self.fit_H,
            bright_susceptibility_audit=self.audit_H,
        )
        self.assertTrue(reduction.diagnostics.accepted)
        self.assertEqual(reduction.node_count, 0)
        self.assertEqual(reduction.diagnostics.original_dark_mode_count, 0)
        np.testing.assert_array_equal(
            reduction.evaluate_from_bright(self.audit_H),
            np.zeros_like(self.audit_H),
        )

    def test_invalid_or_nonpositive_modal_data_are_rejected(self) -> None:
        for L, weights in (
            (np.asarray([0.0, 0.3]), np.asarray([1.0, 1.0])),
            (np.asarray([0.2, 0.3]), np.asarray([1.0, -1.0])),
            (np.asarray([0.2]), np.asarray([1.0, 2.0])),
        ):
            with self.subTest(L=L, weights=weights):
                with self.assertRaises(ValueError):
                    reduce_positive_dark_measure(
                        L,
                        weights,
                        bright_index=0,
                        bright_susceptibility_fit=self.fit_H,
                        bright_susceptibility_audit=self.audit_H,
                    )


if __name__ == "__main__":
    unittest.main()
