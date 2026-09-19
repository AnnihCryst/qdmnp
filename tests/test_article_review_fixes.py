"""Regression tests for the review of the article chain (logic and consistency)."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_article as runner
from article_observables.qd_mnp_article_inputs import (
    load_inputs, model_error_estimates, quasistatic_gap_counts, validate_inputs,
)
from article_observables.qd_mnp_gap_metrics_common import dd_validity_assessment, dd_validity_distance
from article_observables.qd_mnp_spectral_features import extract_feature


def threshold_scan(channels, gaps, fqs, status=None, isolated=6e-5, isolated_status="resolved_refined"):
    fqs = np.asarray(fqs, float)
    values = np.stack([fqs, fqs])
    statuses = np.full(values.shape, "resolved_refined", dtype="U32")
    if status is not None:
        statuses[1] = np.asarray(status)
        statuses[0] = np.asarray(status)
    return {"channel_id": np.asarray(channels), "gap_nm": np.asarray(gaps, float),
            "threshold_fluence_j_cm2": values, "threshold_status": statuses,
            "isolated_threshold_fluence_j_cm2": np.asarray(isolated),
            "isolated_threshold_status": np.asarray(isolated_status)}


class CensoredControlTests(unittest.TestCase):
    channels = ["axis_long", "side_trans_radial", "axis_trans"]

    def scan(self, control_status="right_censored", control_value=np.nan, leader=2e-6):
        return threshold_scan(self.channels, [1.0], [[leader], [1e-5], [control_value]],
                              [["resolved_refined"], ["resolved_refined"], [control_status]])

    def test_censored_control_does_not_block_numerical_check(self):
        reference, candidate = self.scan(), self.scan(leader=2.02e-6)
        result = runner.compare_thresholds(reference, candidate, required_channels=self.channels[:2])
        self.assertTrue(result["resolved_pairs_complete"])
        self.assertFalse(result["all_pairs_resolved"])
        self.assertTrue(result["nonrequired_pairs_consistent"])
        self.assertAlmostEqual(result["max_relative_threshold_change"], 0.01, places=12)
        # Legacy default still requires every channel.
        self.assertFalse(runner.compare_thresholds(reference, candidate)["resolved_pairs_complete"])

    def test_control_crossing_the_scan_edge_is_reported_not_hidden(self):
        result = runner.compare_thresholds(self.scan(), self.scan("resolved_refined", 7e-4),
                                           required_channels=self.channels[:2])
        self.assertTrue(result["resolved_pairs_complete"])
        self.assertFalse(result["nonrequired_pairs_consistent"])

    def test_ranking_treats_right_censored_control_as_separated_lower_bound(self):
        config = load_inputs()
        master = {"channel_id": np.array(config["geometry"]["channels"]), "gap_nm": np.array([1.0, 2.0]),
                  "threshold_fluence_j_cm2": np.full((2, 5, 2), 4e-5),
                  "threshold_status": np.full((2, 5, 2), "resolved", dtype="U32")}
        master["threshold_fluence_j_cm2"][1, 0, 0] = 2e-6
        master["threshold_status"][1, 1, 0] = "right_censored"
        selection = {"best_channel": "axis_long", "best_gap_nm": 1.0, "resolved_threshold_selected": True,
                     "selection_allowed_by_declared_retardation_cutoffs": np.ones((5, 2), bool)}
        records = [{"label": label, "max_relative_threshold_change": .01}
                   for label in ("higher_N", "fluence_refined", "spatial_refined", "time_step_refined")]
        row = {"accepted": True, "resolved_pairs_complete": True, "best_configuration_unchanged": True,
               "candidate_own_thresholds": [[2e-6], [1e-5], [np.nan]],
               "candidate_status": [["resolved"], ["resolved"], ["right_censored"]]}
        for name in ("gap", "c", "a", "exciton", "gamma1", "dephasing"):
            for suffix in ("low", "high"):
                records.append({"label": name+"_"+suffix, **row})
        for energy in config["pulse"]["carrier_scan_eV"]:
            records.append({"label": "carrier_"+str(energy), "carrier_energy_eV": energy, **row})
        outcome = runner.assess_recommendation_ranking(master, selection, records, config, numerical_accepted=True)
        self.assertTrue(outcome["selected_candidate_separated_in_sensitivity_cases"])
        self.assertTrue(outcome["accepted"])
        gap_row = next(row for row in records if row["label"] == "gap_low")
        gap_row["candidate_status"] = [["resolved"], ["resolved"], ["left_censored"]]
        self.assertFalse(runner.assess_recommendation_ranking(master, selection, records, config,
                                                              numerical_accepted=True)["accepted"])


class FluenceRangeTests(unittest.TestCase):
    def data(self, status, isolated="resolved_refined"):
        config = load_inputs()
        channels = config["geometry"]["channels"]
        scan = threshold_scan(channels, [1.0, 2.0], np.full((5, 2), 1e-5), isolated_status=isolated)
        scan["threshold_status"][:] = "resolved_refined"
        for index, value in status.items():
            scan["threshold_status"][(slice(None),) + index] = value
        return config, scan

    def test_suppressing_channel_above_scan_does_not_trigger_full_rescan(self):
        config, scan = self.data({(1, 0): "right_censored", (4, 0): "not_reached_first_lobe"})
        self.assertEqual(runner.censoring_extension(scan, config), (False, False))

    def test_left_censoring_or_unresolved_reference_extends_range(self):
        config, scan = self.data({(2, 1): "left_censored"})
        self.assertEqual(runner.censoring_extension(scan, config), (True, False))
        config, scan = self.data({}, isolated="right_censored")
        self.assertEqual(runner.censoring_extension(scan, config), (False, True))
        config, scan = self.data({(ci, gi): "right_censored" for ci in range(5) for gi in range(2)})
        self.assertEqual(runner.censoring_extension(scan, config), (False, True))

    def test_grid_prediction_starts_strong_channels_on_passing_grid(self):
        config = load_inputs()
        weak = runner.predict_fluence_points(config, np.array([1.0, 0.5]), 65)
        strong = runner.predict_fluence_points(config, np.array([[33.6, 0.07]]), 65)
        self.assertEqual(weak["predicted_points"], 65)
        self.assertEqual(strong["predicted_points"], 129)
        self.assertLessEqual(strong["predicted_midpoint_error"], config["pulse"]["max_fluence_midpoint_population_error"])


class SelectionAndPreflightTests(unittest.TestCase):
    def _selection_data(self, config, gaps, best_profile):
        channels = config["geometry"]["channels"]
        values = np.full((2, 5, gaps.size), 4e-5)
        values[1, 0] = best_profile
        return {"channel_id": np.array(channels), "gap_nm": gaps, "threshold_fluence_j_cm2": values,
                "threshold_status": np.full(values.shape, "resolved_refined", dtype="U32"),
                "absolute_threshold_discrepancy_dd_vs_fqs": np.full((5, gaps.size), .5)}

    def test_locality_advisory_reports_but_never_excludes_a_closer_winner(self):
        config = load_inputs()
        config["geometry"]["locality_advisory_gap_nm"] = 2.0
        gaps = np.array([1.0, 2.0, 3.0])
        data = self._selection_data(config, gaps, [1e-6, 2e-6, 3e-6])
        chosen = runner.select_scenarios(data, config)
        # The advisory no longer filters: the 1 nm winner is selected and flagged.
        self.assertEqual((chosen["best_channel"], chosen["best_gap_nm"]), ("axis_long", 1.0))
        self.assertEqual(chosen["locality_advisory_gap_nm"], 2.0)
        self.assertTrue(chosen["best_gap_below_locality_advisory"])
        self.assertEqual(chosen["best_gap_optimum_kind"], "at_smallest_sampled_gap")
        self.assertAlmostEqual(chosen["threshold_gain_below_locality_advisory"], 2.0)
        self.assertTrue(chosen["best_gap_at_lower_admissible_bound"])
        self.assertEqual(chosen["far_gap_role"], "largest_admissible_gap_without_dd_agreement")
        self.assertEqual(chosen["threshold_dd_validity_status"], "not_established_within_validity")
        data["absolute_threshold_discrepancy_dd_vs_fqs"][0] = [.5, .05, .05]
        chosen = runner.select_scenarios(data, config)
        self.assertEqual(chosen["far_gap_role"], "dd_fqs_threshold_agreement")
        self.assertEqual(chosen["threshold_dd_validity_gap_nm"], 2.0)

    def test_interior_optimum_is_reported_as_the_model_s_own_answer(self):
        config = load_inputs()
        config["geometry"]["locality_advisory_gap_nm"] = 1.0
        gaps = np.array([0.5, 1.0, 2.0, 3.0])
        data = self._selection_data(config, gaps, [3e-6, 1e-6, 2e-6, 4e-6])
        chosen = runner.select_scenarios(data, config)
        self.assertEqual(chosen["best_gap_nm"], 1.0)
        self.assertEqual(chosen["best_gap_optimum_kind"], "interior")
        self.assertFalse(chosen["best_gap_below_locality_advisory"])
        # Going below the advisory costs rather than gains here.
        self.assertAlmostEqual(chosen["threshold_gain_below_locality_advisory"], 1/3)

    def test_preflight_gaps_stay_admissible_and_follow_the_dd_transition(self):
        config = load_inputs()
        gaps = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
        _, allowed = runner.retardation_selection_mask(config, config["geometry"]["channels"], gaps)
        delta = np.tile([3.0, 1.0, .5, .12, .05, .01, .001], (5, 1))
        chosen, _ = runner.choose_preflight_gaps(gaps, allowed, delta, .1)
        self.assertEqual(chosen, [1.0, 5.0, 10.0])
        self.assertEqual(runner.choose_preflight_gaps(gaps, allowed, None, .1)[0][1], 3.0)


class ValidationReuseTests(unittest.TestCase):
    def test_reference_scan_and_reference_carrier_are_not_solved_again(self):
        config = load_inputs(smoke=True)
        channels = config["geometry"]["channels"]
        master = threshold_scan(channels, [1.0, 10.0], np.full((5, 2), 1e-5))
        master["threshold_fluence_j_cm2"][1, 0, 0] = 2e-6
        with tempfile.TemporaryDirectory() as directory:
            run = runner.ArticleRun(config, Path(directory))
            path = Path(directory)/"master.npz"
            np.savez(path, metadata_json=np.asarray("{}"), **master)
            run.state.update(threshold_master=str(path), selection={
                "best_channel": "axis_long", "best_gap_nm": 1.0, "runner_up_channel": "side_long",
                "runner_up_gap_nm": 1.0, "control_channel": "axis_trans", "resolved_threshold_selected": True,
                "selection_allowed_by_declared_retardation_cutoffs": np.ones((5, 2), bool)})
            calls = []

            def fake_concurrent(specs, concurrency):
                outcomes = {}
                for label, module, arguments in specs:
                    calls.append(label)
                    target = Path(directory)/(label+".npz")
                    subset = runner.threshold_subset(master, arguments["channels"], arguments["gaps-nm"])
                    np.savez(target, metadata_json=np.asarray("{}"), **subset)
                    outcomes[label] = target
                return outcomes

            with patch.object(run, "steps_concurrently", side_effect=fake_concurrent), \
                    patch.object(run, "plot_validation"):
                run.validation()
        self.assertNotIn("validation_reference", calls)
        self.assertEqual(sorted(calls), ["check_carrier_2.022", "check_carrier_2.062"])
        summary = run.state["carrier_scan_summary"]
        self.assertEqual([row["threshold_J_cm2"] for row in summary["rows"]], [2e-6]*3)
        self.assertEqual(summary["best_tested_carrier_eV"], 2.022)
        self.assertFalse(summary["best_tested_carrier_is_interior"])


class QuasiStaticValidityTests(unittest.TestCase):
    def test_boundary_uses_only_admissible_gaps_and_reports_support(self):
        gaps = np.array([1., 2., 5., 10., 20., 50.])
        delta = np.array([.6, .5, .3, .2, .06, .004])
        valid = gaps <= 10
        self.assertEqual(dd_validity_distance(gaps, delta, .1), 20.0)
        self.assertEqual(dd_validity_assessment(gaps, delta, .1, valid)[1], "not_established_within_validity")
        delta[3] = .05
        self.assertEqual(dd_validity_assessment(gaps, delta, .1, valid), (10.0, "last_valid_point_only", 1))
        delta[2] = .08
        self.assertEqual(dd_validity_assessment(gaps, delta, .1, valid), (5.0, "established", 2))
        self.assertEqual(dd_validity_assessment(gaps, delta, .1, np.zeros(6, bool))[1], "no_valid_gaps")

    def test_production_grid_is_inside_declared_cutoff(self):
        config = load_inputs()
        counts = quasistatic_gap_counts(config)
        self.assertEqual(counts, {"axis": len(config["geometry"]["gaps_nm"]), "side": len(config["geometry"]["gaps_nm"])})
        config["geometry"]["gaps_nm"] = [0.5, 20.0, 50.0]
        with self.assertRaisesRegex(ValueError, "three gaps"):
            validate_inputs(config)
        # A grid that never reaches below the advisory cannot decide the question.
        config["geometry"]["gaps_nm"] = [1.0, 2.0, 3.0]
        with self.assertRaisesRegex(ValueError, "must sample below"):
            validate_inputs(config)


class ModelErrorAndFeatureTests(unittest.TestCase):
    def test_model_error_indicators_have_expected_signs(self):
        estimates = model_error_estimates(load_inputs())
        long = estimates["long"]
        self.assertLess(long["lspr_shift_meV"], 0)
        self.assertGreater(long["alpha_squared_ratio_retardation_at_carrier"], 1.1)
        ratios = [row["alpha_squared_ratio_at_carrier"] for row in long["surface_damping"]]
        self.assertTrue(all(r < 1 for r in ratios) and ratios[0] > ratios[-1])
        self.assertAlmostEqual(estimates["effective_path_length_nm"], 14.855, places=2)

    def test_peak_energy_is_not_quantized_to_grid_nodes(self):
        energy = np.linspace(1.85, 2.234, 4001)
        step = energy[1]-energy[0]
        for fraction in (.13, .37):
            center = 2.042 + fraction*step
            spectrum = 1/(1+((energy-center)/.00127)**2)
            feature = extract_feature(energy, spectrum, center_eV=2.042, half_window_eV=.17)
            self.assertLess(abs(feature.energy_eV-center), .01*step)


if __name__ == "__main__":
    unittest.main()
