"""Compatibility checks for renamed Gamma2 and pulse-response APIs."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import qd_mnp_fano_scan as fano
import qd_mnp_linear_spectrum as linear
import qd_mnp_plot_pulse_absorption_results as pulse_plot
import qd_mnp_pulse_absorption_sweep as pulse
import qd_mnp_rational_fit as core


class CliAndPythonCompatibilityTests(unittest.TestCase):
    def test_all_production_entrypoints_share_the_nine_mode_default(self) -> None:
        self.assertEqual(
            inspect.signature(core.HybridQDPlasmonModel).parameters[
                "n_modes"
            ].default,
            9,
        )
        with patch("sys.argv", [core.__name__]):
            core_args = core.parse_args()
        self.assertEqual(core_args.modes, [9])
        self.assertEqual(core_args.dynamics_n_modes, 9)

        for module in (linear, fano, pulse):
            with self.subTest(module=module.__name__), patch(
                "sys.argv",
                [module.__name__],
            ):
                self.assertEqual(module.parse_args().n_modes, 9)

        with patch("sys.argv", [linear.__name__]):
            linear_args = linear.parse_args()
        self.assertEqual(
            (linear_args.energy_min_ev, linear_args.energy_max_ev, linear_args.points),
            (2.0, 2.08, 201),
        )

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

    def test_rational_fit_artifact_forwards_geometry_to_parameters(self) -> None:
        """The CLI path must forward the QD position and the polarization."""

        cases = (
            (["--orientation", "trans"], "tip", "transverse"),
            (["--field-polarization", "transverse"], "tip", "transverse"),
            (
                ["--qd-position", "equatorial", "--field-polarization", "longitudinal"],
                "equatorial",
                "longitudinal",
            ),
            (["--qd-position", "equatorial"], "equatorial", "longitudinal"),
        )
        for argv_tail, expected_position, expected_polarization in cases:
            with self.subTest(argv_tail=argv_tail):
                with tempfile.TemporaryDirectory() as run_dir:
                    with patch(
                        "sys.argv",
                        [core.__name__, *argv_tail, "--run-dir", run_dir],
                    ):
                        args = core.parse_args()

                    with patch.object(
                        core,
                        "make_params_with_overrides",
                        side_effect=RuntimeError("stop after argument forwarding"),
                    ) as factory:
                        with self.assertRaisesRegex(RuntimeError, "argument forwarding"):
                            core.build_rational_fit_artifact(args)

                self.assertEqual(
                    factory.call_args.kwargs["qd_position"],
                    expected_position,
                )
                self.assertEqual(
                    factory.call_args.kwargs["field_polarization"],
                    expected_polarization,
                )

    def test_historical_plot_helper_warns_and_delegates(self) -> None:
        output = Path("unused.png")
        with patch.object(pulse_plot, "plot_pulse_response_sweep") as canonical:
            with self.assertWarns(DeprecationWarning):
                pulse_plot.plot_absorption_sweep(None, "fluence", output, False)
        canonical.assert_called_once_with(None, x_axis="fluence", output_path=output, show=False)


if __name__ == "__main__":
    unittest.main()
