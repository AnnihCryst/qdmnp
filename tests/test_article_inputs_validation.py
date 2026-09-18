"""Reject unusable article scenarios before fitting or launching ODE sweeps."""

import unittest

from article_observables.qd_mnp_article_inputs import load_inputs, validate_inputs


class ArticleInputsValidationTests(unittest.TestCase):
    def assert_invalid(self, section, key, value, message=None):
        config = load_inputs()
        config[section][key] = value
        with self.assertRaisesRegex(ValueError, message or key):
            validate_inputs(config)

    def test_production_and_smoke_inputs_both_validate(self):
        validate_inputs(load_inputs())
        validate_inputs(load_inputs(smoke=True))

    def test_counts_cannot_be_fractional_booleans_or_zero(self):
        for section, key in (
            ("pulse", "fluence_points"), ("pulse", "max_fluence_points"),
            ("spectrum", "points"), ("spectrum", "max_points"),
            ("material", "validation_modes"), ("numerics", "spatial_order"),
            ("numerics", "max_spatial_order"), ("numerics", "modal_audit_points"),
            ("numerics", "points_per_fastest_cycle"), ("work_spectrum", "points"),
            ("output", "dpi"),
        ):
            for value in (False, 3.5, 0):
                with self.subTest(section=section, key=key, value=value):
                    self.assert_invalid(section, key, value)
        self.assert_invalid("pulse", "max_fluence_extensions", 1.5)
        self.assert_invalid("pulse", "max_fluence_extensions", -1)

    def test_grid_caps_and_actual_native_audit_minimum(self):
        self.assert_invalid("spectrum", "max_points", 1001, "refinement limits")
        self.assert_invalid("pulse", "max_fluence_points", 5, "refinement limits")
        self.assert_invalid("numerics", "max_spatial_order", 40, "refinement limits")
        self.assert_invalid("numerics", "modal_audit_points", 100)
        self.assert_invalid("numerics", "points_per_fastest_cycle", 7)

    def test_mode_candidates_must_be_ordered_distinct_integers(self):
        for value in ([9, 9], [10, 9], [9, 10.5], [True, 9], [1], []):
            with self.subTest(value=value):
                self.assert_invalid("material", "mode_candidates", value, "Multi-mode")
        self.assert_invalid("material", "validation_modes", 12, "exceed")

    def test_more_interpolation_points_do_not_create_independent_Au_data(self):
        config = load_inputs()
        config["material"].update(fit_min_eV=1.8, fit_max_eV=2.5)
        config["spectrum"].update(points=100001, max_points=100001)
        with self.assertRaisesRegex(ValueError, "independent native Au samples"):
            validate_inputs(config)
        self.assert_invalid("material", "validation_modes", 14, "independent native Au samples")
        # A broader interval supplies enough actual tabulated observations.
        config = load_inputs()
        config["material"].update(fit_max_eV=6.6, validation_modes=14)
        validate_inputs(config)

    def test_fit_interval_cannot_extrapolate_the_material_table(self):
        self.assert_invalid("material", "fit_min_eV", .1, "native Au table")
        self.assert_invalid("material", "fit_max_eV", 7, "native Au table")

    def test_all_shape_variations_retain_a_prolate_nonintersecting_geometry(self):
        self.assert_invalid("validation", "shape_relative_offset", 1, "Shape sensitivity")
        self.assert_invalid("validation", "shape_relative_offset", .4, "Shape sensitivity")
        self.assert_invalid("validation", "gap_offset_nm", 1, "intersect")
        self.assert_invalid("geometry", "reference_surface_gap_nm", 0)
        config = load_inputs()
        config["geometry"].update(c_nm=10., a_nm=10.)
        config["validation"]["shape_relative_offset"] = 0
        validate_inputs(config)

    def test_rate_and_exciton_variations_remain_physical(self):
        for key in ("population_decay_relative_offset", "dephasing_relative_offset"):
            self.assert_invalid("validation", key, 1.1)
        self.assert_invalid("validation", "exciton_offset_eV", 2.042, "Exciton sensitivity")
        config = load_inputs()
        config["qd"]["pure_dephasing_energy_meV"] = 0
        # Lifetime broadening alone is mathematically allowed, but a T2 of ~4.9 ns
        # cannot decay before the common population read time of the article run.
        with self.assertRaisesRegex(ValueError, "population_read_fs"):
            validate_inputs(config)
        config["qd"]["population_decay_energy_neV"] = 0
        with self.assertRaisesRegex(ValueError, "Gamma2"):
            validate_inputs(config)
        config = load_inputs()
        config["qd"]["population_decay_energy_neV"] = 0
        config["validation"]["dephasing_relative_offset"] = 1
        with self.assertRaisesRegex(ValueError, "Gamma2"):
            validate_inputs(config)

    def test_work_spectrum_requires_complete_ordered_supported_interval(self):
        for key in ("min_eV", "post_fs", "max_delta_window_absolute_change_cm2"):
            config = load_inputs()
            del config["work_spectrum"][key]
            with self.assertRaisesRegex(ValueError, "Missing.*work_spectrum"):
                validate_inputs(config)
        self.assert_invalid("work_spectrum", "min_eV", 2.2, "interval")
        self.assert_invalid("work_spectrum", "max_eV", 4, "interval")
        self.assert_invalid("work_spectrum", "max_eV", 2., "carrier")
        self.assert_invalid("work_spectrum", "post_fs", 0)
        self.assert_invalid("work_spectrum", "max_delta_window_absolute_change_cm2", -1)
        config = load_inputs()
        config["work_spectrum"] = {"enabled": False}
        validate_inputs(config)

    def test_reject_nonfinite_non_numeric_and_boolean_physical_scales(self):
        for value in (float("nan"), float("inf"), -1., "15 nm", True):
            with self.subTest(value=value):
                self.assert_invalid("geometry", "c_nm", value)
        self.assert_invalid("geometry", "gaps_nm", [True, 10])
        self.assert_invalid("pulse", "carrier_scan_eV", [2.042, 2.002])
        self.assert_invalid("numerics", "tail_window_fraction", 1, "fraction")
        self.assert_invalid("numerics", "max_spectral_leakage", 1, "leakage")
        self.assert_invalid("numerics", "method", "typo", "method")

    def test_missing_tables_and_wrong_switches_have_actionable_errors(self):
        config = load_inputs()
        del config["numerics"]
        with self.assertRaisesRegex(ValueError, "numerics"):
            validate_inputs(config)
        self.assert_invalid("validation", "enabled", "false", "enabled")
        self.assert_invalid("output", "directory", " ", "directory")

    def test_disabled_sensitivity_does_not_reject_unused_variations(self):
        config = load_inputs()
        config["validation"].update(enabled=False, shape_relative_offset=.9)
        config["material"]["validation_modes"] = 14
        validate_inputs(config)


if __name__ == "__main__":
    unittest.main()
