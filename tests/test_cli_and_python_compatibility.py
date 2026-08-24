"""Compatibility checks for renamed Gamma2 and pulse-response APIs."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch
import unittest

import qd_mnp_fano_scan as fano
import qd_mnp_linear_spectrum as linear
import qd_mnp_plot_pulse_absorption_results as pulse_plot
import qd_mnp_pulse_absorption_sweep as pulse
import qd_mnp_rational_fit as core


class CliAndPythonCompatibilityTests(unittest.TestCase):
    def test_scalar_gamma2_cli_aliases_share_the_canonical_destination(self) -> None:
        for module in (core, linear, pulse):
            with self.subTest(module=module.__name__, option="canonical"):
                with patch("sys.argv", [module.__name__, "--gamma2-coherence-mev", "2.78"]):
                    self.assertEqual(module.parse_args().gamma2_coherence_mev, 2.78)
            with self.subTest(module=module.__name__, option="legacy"):
                with patch("sys.argv", [module.__name__, "--gamma-dephasing-mev", "2.78"]):
                    self.assertEqual(module.parse_args().gamma2_coherence_mev, 2.78)

    def test_qd_radius_is_optional_in_existing_compute_apis(self) -> None:
        for function in (linear.compute_spectrum, pulse.compute_sweep, fano.scan_candidates):
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters["qd_radius_nm"]
                self.assertIsNone(parameter.default)

    def test_conflicting_scalar_gamma2_cli_aliases_are_rejected(self) -> None:
        argv_tail = [
            "--gamma2-coherence-mev",
            "2.78",
            "--gamma-dephasing-mev",
            "3.02",
        ]
        for module in (core, linear, pulse):
            with self.subTest(module=module.__name__):
                with patch("sys.argv", [module.__name__, *argv_tail]), patch("sys.stderr"):
                    with self.assertRaises(SystemExit):
                        module.parse_args()

    def test_historical_plot_helper_warns_and_delegates(self) -> None:
        output = Path("unused.png")
        with patch.object(pulse_plot, "plot_pulse_response_sweep") as canonical:
            with self.assertWarns(DeprecationWarning):
                pulse_plot.plot_absorption_sweep(None, "fluence", output, False)
        canonical.assert_called_once_with(None, x_axis="fluence", output_path=output, show=False)


if __name__ == "__main__":
    unittest.main()
