"""Contracts for the Yan Fig. 2(b)/Fig. 3 full-QS validation driver."""

from dataclasses import replace
import unittest

import numpy as np

from literature_reproductions.yan2008_fig2b_fig3_full_qs import (
    GAMMA2_MEV,
    PAPER_RASTER_ANCHORS,
    TRANSITION_DIPOLE_DEBYE,
    calculate_spectral_curve,
    harmonic_state_from_modal_ode,
    incident_field_amplitude_au,
    johnson_christy_silver,
    make_profile_params,
    parse_args,
    validate_periodic_time_ode,
    yan_profiles,
)
from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
from qd_mnp_rational_fit import (
    HybridQDPlasmonModel,
    eV_to_au,
    make_default_params,
)
from qd_mnp_spheroid_green import SpheroidGreenInteraction


class YanMaterialAndPaperSpectrumTests(unittest.TestCase):
    def test_silver_table_is_ordered_and_covers_both_fig3_windows(self) -> None:
        material = johnson_christy_silver()
        self.assertEqual(material.energy_eV.size, 49)
        self.assertTrue(np.all(np.diff(material.energy_eV) > 0.0))
        self.assertLess(material.energy_eV[0], 3.1)
        self.assertGreater(material.energy_eV[-1], 3.5)
        epsilon = material.epsilon_at(np.asarray([3.1, 3.34, 3.5]))
        self.assertTrue(np.all(np.isfinite(epsilon)))
        self.assertTrue(np.all(epsilon.imag > 0.0))

    def test_parameter_completion_is_explicit_and_uses_bare_qd_dipole(self) -> None:
        au_profile, ag_profile = yan_profiles()
        for profile in (au_profile, ag_profile):
            params = make_profile_params(profile)
            self.assertEqual(params.qd_dipole_convention, "bare_internal")
            self.assertIn("not all restated", profile.completion_note)
        self.assertAlmostEqual(TRANSITION_DIPOLE_DEBYE, 31.22084, places=4)
        self.assertAlmostEqual(GAMMA2_MEV, 0.00219404, places=8)
        self.assertAlmostEqual(
            incident_field_amplitude_au(1.0, 1.0),
            5.33803e-9,
            delta=2.0e-14,
        )

    def test_fig2b_direct_material_reproduces_qualified_peak_locations(self) -> None:
        profile = yan_profiles()[0]
        params = make_profile_params(profile)
        kernel = SpheroidGreenInteraction.from_params(
            params, orientation="long", n_max=10
        )
        curves = {
            order: calculate_spectral_curve(
                profile,
                params,
                kernel,
                spatial_order=order,
                detuning_window_meV=(-0.5, 0.1),
                points=3001,
            )
            for order in (1, 10)
        }
        self.assertAlmostEqual(
            curves[1].material_peak_detuning_meV,
            PAPER_RASTER_ANCHORS["fig2b_N1"]["printed_peak_label_meV"],
            delta=0.001,
        )
        self.assertAlmostEqual(
            curves[10].material_peak_detuning_meV,
            PAPER_RASTER_ANCHORS["fig2b_N10"]["printed_peak_label_meV"],
            delta=0.020,
        )
        self.assertGreater(curves[1].material_peak_power_W, 3.0e-11)
        self.assertLess(curves[1].material_peak_power_W, 3.8e-11)
        self.assertGreater(curves[10].material_peak_power_W, 0.8e-11)
        self.assertLess(curves[10].material_peak_power_W, 1.2e-11)

    def test_fig3_completed_profile_gets_sign_reversal_but_not_paper_magnitudes(self) -> None:
        profile = yan_profiles()[1]
        params = make_profile_params(profile)
        kernel = SpheroidGreenInteraction.from_params(
            params, orientation="long", n_max=10
        )
        n1 = calculate_spectral_curve(
            profile,
            params,
            kernel,
            spatial_order=1,
            detuning_window_meV=(-2.0, 2.0),
            points=8001,
        )
        n10 = calculate_spectral_curve(
            profile,
            params,
            kernel,
            spatial_order=10,
            detuning_window_meV=(-10.0, 10.0),
            points=8001,
        )
        self.assertGreater(n1.material_peak_detuning_meV, 0.0)
        self.assertLess(n10.material_peak_detuning_meV, 0.0)
        # This is an intentional negative-result regression: the Fig. 3
        # caption omits enough parameters/material-processing detail that the
        # traceable completion must not be advertised as quantitative reuse.
        self.assertGreater(
            abs(
                n1.material_peak_detuning_meV
                - PAPER_RASTER_ANCHORS["fig3_N1"]["detuning_meV"]
            ),
            0.10,
        )
        self.assertGreater(
            abs(
                n10.material_peak_detuning_meV
                - PAPER_RASTER_ANCHORS["fig3_N10"]["detuning_meV"]
            ),
            2.0,
        )

    def test_cli_defaults_run_the_full_nonquick_validation(self) -> None:
        args = parse_args([])
        self.assertFalse(args.quick)
        self.assertFalse(args.skip_time_domain)
        self.assertFalse(args.no_plots)


class YanModalODEHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        params = replace(
            make_default_params("long"),
            gamma_au=float(eV_to_au(0.020)),
            Gamma_au=float(eV_to_au(0.020)),
        )
        bright = HybridQDPlasmonModel(
            params,
            orientation="long",
            n_modes=1,
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        cls.model = FullQSSpheroidPulseModel(
            bright,
            SpheroidGreenInteraction.from_params(
                params, orientation="long", n_max=2
            ),
            fit_quality_policy="ignore",
            spatial_convergence_policy="ignore",
            modal_audit_points=201,
        )

    def test_harmonic_state_space_solution_matches_public_A_B_K_ports(self) -> None:
        result = harmonic_state_from_modal_ode(self.model, 2.042)
        self.assertEqual(result.state_over_field.size, 6)
        self.assertLess(result.max_port_relative_error, 1.0e-9)

    def test_short_periodic_carrier_run_checks_alpha_power_and_weak_field(self) -> None:
        result = validate_periodic_time_ode(
            self.model,
            material_name="fixture",
            energy_eV=2.042,
            intensity_w_cm2=1.0,
            cycles=4,
            analysis_cycles=2,
            points_per_cycle=16,
        )
        self.assertLess(result.alpha_relative_error, 1.0e-5)
        self.assertLess(result.mean_power_relative_error, 1.0e-5)
        self.assertLess(result.accumulator_vs_quadrature_relative_error, 1.0e-5)
        self.assertLess(result.max_excited_population, 1.0e-7)


if __name__ == "__main__":
    unittest.main()
