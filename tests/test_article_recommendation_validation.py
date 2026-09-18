"""Independent-grid checks and conservative recommendation decisions."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from article_observables.qd_mnp_article_inputs import load_inputs
from run_article import (
    assess_recommendation_ranking, compare_spectral_refinement, compare_thresholds,
)


class RecommendationRankingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_inputs()
        # These tests exercise the gate that includes the carrier scan.
        self.config["validation"]["carrier_scan_in_ranking"] = True
        self.master = {
            "channel_id": np.array(self.config["geometry"]["channels"]),
            "gap_nm": np.array([1., 10.]),
            "threshold_fluence_j_cm2": np.full((2, 5, 2), 4e-5),
            "threshold_status": np.full((2, 5, 2), "resolved", dtype="U32"),
        }
        self.master["threshold_fluence_j_cm2"][1, 0, 0] = 1e-5
        self.selection = {
            "best_channel": "axis_long", "best_gap_nm": 1.,
            "resolved_threshold_selected": True,
            "selection_allowed_by_declared_retardation_cutoffs": np.ones((5, 2), bool),
        }
        self.records = [{"label": label, "max_relative_threshold_change": .01}
                        for label in ("higher_N", "fluence_refined", "spatial_refined", "time_step_refined")]
        for name in ("gap", "c", "a", "exciton", "gamma1", "dephasing"):
            for suffix in ("low", "high"):
                self.records.append({"label": name+"_"+suffix, "accepted": True,
                                     "resolved_pairs_complete": True,
                                     "candidate_thresholds": [[1e-5], [4e-5]],
                                     "best_configuration_unchanged": True})
        for energy in self.config["pulse"]["carrier_scan_eV"]:
            self.records.append({"label": "carrier_"+str(energy), "carrier_energy_eV": energy,
                                 "accepted": True, "resolved_pairs_complete": True,
                                 "candidate_thresholds": [[1e-5], [4e-5]],
                                 "best_configuration_unchanged": True})

    def assess(self, numerical_accepted=True):
        return assess_recommendation_ranking(self.master, self.selection, self.records,
                                             self.config, numerical_accepted=numerical_accepted)

    def test_separated_and_stable_candidate_passes_only_with_numerical_checks(self):
        self.assertTrue(self.assess()["accepted"])
        self.assertFalse(self.assess(numerical_accepted=False)["accepted"])

    def test_nearby_gap_in_same_channel_counts_as_indistinguishable_competitor(self):
        self.master["threshold_fluence_j_cm2"][1, 0, 1] = 1.05e-5
        outcome = self.assess()
        self.assertFalse(outcome["accepted"])
        self.assertFalse(outcome["selected_candidate_separated"])
        self.assertEqual(len(outcome["indistinguishable_grid_candidates"]), 2)

    def test_changed_or_missing_sensitivity_check_prevents_positive_verdict(self):
        self.records[-1]["best_configuration_unchanged"] = False
        self.assertFalse(self.assess()["accepted"])
        self.records.pop()
        self.assertFalse(self.assess()["sensitivity_checks_complete"])

    def test_carrier_scan_can_be_reported_without_gating_robustness(self):
        self.records[-1]["best_configuration_unchanged"] = False
        self.records[-1]["resolved_pairs_complete"] = False
        self.config["validation"]["carrier_scan_in_ranking"] = False
        self.assertTrue(self.assess()["accepted"])
        self.records = [row for row in self.records if "carrier_energy_eV" not in row]
        self.assertTrue(self.assess()["sensitivity_checks_complete"])

    def test_left_bound_can_conceal_better_candidate(self):
        self.master["threshold_status"][1, 1, 0] = "left_censored"
        self.master["threshold_fluence_j_cm2"][1, 1, 0] = 1e-8
        outcome = self.assess()
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["unresolved_potential_competitor_count"], 1)

    def test_sensitivity_can_keep_rank_but_destroy_numerical_separation(self):
        self.records[-1]["candidate_thresholds"] = [[1e-5], [1.01e-5]]
        outcome = self.assess()
        self.assertTrue(outcome["ranking_preserved_in_tested_sensitivity_cases"])
        self.assertFalse(outcome["selected_candidate_separated_in_sensitivity_cases"])
        self.assertFalse(outcome["accepted"])


class ArtifactRefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def save(self, label, payload):
        path = self.directory / (label+".npz")
        np.savez(path, metadata_json=np.array("{}"), **payload)
        return path

    def spectrum(self, count):
        energy = np.linspace(1.85, 2.234, count)
        curve = np.exp(-((energy-2.042)/.02)**2)
        return {
            "energy_eV":energy, "channel_id":np.array(["axis_long", "axis_trans"]),
            "material_model_id":np.array(["direct", "one", "multi"]), "surface_gap_nm":np.array(1.),
            "excitation_spectrum_au6":np.broadcast_to(curve,(3,2,count)).copy(),
            "feature_status":np.full((3,2),"ok",dtype="U32"),
            "competing_peak_count":np.zeros((3,2),int), "fwhm_eV":np.full((3,2),.04),
            "peak_energy_eV":np.full((3,2),2.042), "excitation_gain_optimized":np.ones((3,2)),
            "isolated_fwhm_eV":np.array(.04), "isolated_peak_energy_eV":np.array(2.042),
            "isolated_feature_status":np.array("ok"),
        }

    def test_refined_solve_must_agree_with_original_spectrum(self):
        coarse, fine = self.spectrum(201), self.spectrum(401)
        reference = self.save("coarse",coarse)
        limits = load_inputs()["spectrum"]
        self.assertTrue(compare_spectral_refinement(reference,self.save("fine",fine),limits)["accepted"])
        fine["excitation_spectrum_au6"][2] *= 1.2
        self.assertFalse(compare_spectral_refinement(reference,self.save("changed",fine),limits)["accepted"])

    def test_split_spectrum_compares_components_without_inventing_aggregate_width(self):
        coarse, fine = self.spectrum(201), self.spectrum(401)
        for data in (coarse,fine):
            data["feature_status"][:] = "split_or_ambiguous"
            data["competing_peak_count"][:] = 2
            data["fwhm_eV"][:] = np.nan
            data["peak_energy_eV"][:] = np.nan
        limits=load_inputs()["spectrum"]
        reference=self.save("coarse",coarse)
        result=compare_spectral_refinement(reference,self.save("fine",fine),limits)
        self.assertTrue(result["accepted"])
        self.assertIsNone(result["unique_peak_width_max_relative_change"])
        fine["competing_peak_count"][2,0]=3
        self.assertFalse(compare_spectral_refinement(reference,self.save("split_changed",fine),limits)["accepted"])

    def test_bare_threshold_and_change_of_gap_are_part_of_validation(self):
        baseline = {
            "channel_id":np.array(["axis_long"]), "gap_nm":np.array([1.,2.]),
            "threshold_fluence_j_cm2":np.array([[[1e-5,2e-5]],[[1e-5,2e-5]]]),
            "threshold_status":np.full((2,1,2),"resolved",dtype="U32"),
            "isolated_threshold_fluence_j_cm2":np.array(1e-5),
            "isolated_threshold_status":np.array("resolved"),
        }
        reference=self.save("reference",baseline)
        changed=deepcopy(baseline)
        changed["isolated_threshold_fluence_j_cm2"]=np.array(2e-5)
        changed["threshold_fluence_j_cm2"][1,0]=[2e-5,1e-5]
        result=compare_thresholds(reference,self.save("changed",changed))
        self.assertTrue(result["best_channel_unchanged"])
        self.assertFalse(result["best_configuration_unchanged"])
        self.assertEqual(result["isolated_relative_threshold_change"],1.)
        self.assertGreaterEqual(result["max_relative_threshold_change"],1.)


if __name__ == "__main__":
    unittest.main()
