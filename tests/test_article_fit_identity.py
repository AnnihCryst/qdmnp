"""Cross-artifact fit provenance, including the serializers' padding formats."""

from copy import deepcopy
import unittest

import numpy as np

from qdmnp.observables.article_fit_identity import compare_fit_coefficients
from qdmnp.observables.article_inputs import load_inputs, validate_inputs


# Four distinguishable material models, so swapping branch/channel axes cannot
# accidentally pass. These are positive Lorentz coefficients, not fitted data.
FITS = (
    dict(alpha_inf=.01, strengths_au2=[.11], omega_modes_au=[.21], gamma_modes_au=[.03]),
    dict(alpha_inf=.02, strengths_au2=[.12, .13], omega_modes_au=[.22, .23], gamma_modes_au=[.04, .05]),
    dict(alpha_inf=.06, strengths_au2=[.16], omega_modes_au=[.26], gamma_modes_au=[.08]),
    dict(alpha_inf=.07, strengths_au2=[.17, .18], omega_modes_au=[.27, .28], gamma_modes_au=[.09, .10]),
)
CHANNELS = np.array(["side_trans_tangential", "axis_long", "side_long", "side_trans_radial", "axis_trans"])
BRANCH_CHANNEL_MODELS = np.array([[2, 0, 0, 2, 2], [3, 1, 1, 3, 3]])


def packed(indices, *, padding=0., prefix="fit_", alpha_key="fit_alpha_inf"):
    """Serialize selected physical fits into a rectangular, padded NPZ table."""
    indices = np.asarray(indices)
    result = {alpha_key: np.empty(indices.shape)}
    for key in ("strengths_au2", "omega_modes_au", "gamma_modes_au"):
        result[prefix + key] = np.full((*indices.shape, 3), padding)
    for index in np.ndindex(indices.shape):
        fit = FITS[indices[index]]
        result[alpha_key][index] = fit["alpha_inf"]
        for key in ("strengths_au2", "omega_modes_au", "gamma_modes_au"):
            result[prefix + key][index][:len(fit[key])] = fit[key]
    return result


def material_reference():
    arrays = packed([[0, 1], [2, 3]], padding=np.nan, alpha_key="fit_alpha_inf_dimensionless")
    arrays["orientation_ids"] = np.array(["long", "trans"])
    return arrays


def article_layouts():
    """The actual axis/key conventions used by Fig.4, masters, Fig.5, Fig.6, S1."""
    fig4 = packed(BRANCH_CHANNEL_MODELS)
    fig4.update(channel_id=CHANNELS, fit_model_id=np.array(["one", "multi"]))
    gap_master = packed(np.repeat(BRANCH_CHANNEL_MODELS[1, :, None], 3, axis=1), padding=np.nan)
    gap_master.update(channel_id=CHANNELS, gap_nm=np.array([1., 2., 10.]))
    fig5 = {"hybrid_channel_id": CHANNELS}
    for branch, row in zip(("one", "multi"), BRANCH_CHANNEL_MODELS):
        fig5.update(packed(row, prefix=branch+"__fit_", alpha_key=branch+"__fit_alpha_inf"))
    work = packed(BRANCH_CHANNEL_MODELS, prefix="material_fit_", alpha_key="material_fit_alpha_inf_au3")
    work["channel_key"] = CHANNELS
    dynamics_metadata = {"model_by_channel": {
        channel: {"material_fit": deepcopy(FITS[fit_index])}
        for channel, fit_index in zip(CHANNELS, BRANCH_CHANNEL_MODELS[1])
    }}
    return {
        "fig4": (fig4, {}, 10), "gap_master": (gap_master, {}, 15),
        "fig5": (fig5, {}, 10), "dynamics": ({}, dynamics_metadata, 5),
        "work": (work, {}, 10),
    }


class ArticleFitIdentityTests(unittest.TestCase):
    def test_all_article_layouts_match_the_same_material_reference(self):
        for name, (arrays, metadata, expected_count) in article_layouts().items():
            with self.subTest(layout=name):
                certificate = compare_fit_coefficients(material_reference(), {}, arrays, metadata)
                self.assertTrue(certificate["accepted"])
                self.assertEqual(certificate["coefficient_sets_checked"], expected_count)
                self.assertEqual({row["orientation"] for row in certificate["fits"]}, {"long", "trans"})

    def test_N1_nan_and_zero_padding_have_identical_identity_hashes(self):
        reference = material_reference()
        arrays, metadata, _ = article_layouts()["fig4"]
        ref_check = compare_fit_coefficients(reference, {}, reference, {})
        check = compare_fit_coefficients(reference, {}, arrays, metadata)
        reference_hashes = {(row["orientation"], row["modes"]): row["sha256"] for row in ref_check["fits"]}
        self.assertEqual(len(reference_hashes), 4)
        for row in check["fits"]:
            self.assertEqual(row["sha256"], reference_hashes[row["orientation"], row["modes"]])
        self.assertEqual({row["modes"] for row in check["fits"]}, {1, 2})

    def test_changed_coefficient_in_each_layout_is_rejected(self):
        changes = {
            "fig4": ("fit_omega_modes_au", (1, 3, 1)),
            "gap_master": ("fit_gamma_modes_au", (4, 2, 0)),
            "fig5": ("one__fit_strengths_au2", (0, 0)),
            "work": ("material_fit_alpha_inf_au3", (0, 0)),
        }
        layouts = article_layouts()
        for name, (key, index) in changes.items():
            arrays, metadata, _ = layouts[name]
            arrays[key][index] += .001
            with self.subTest(layout=name), self.assertRaisesRegex(ValueError, "Different Lorentz coefficients"):
                compare_fit_coefficients(material_reference(), {}, arrays, metadata)
        arrays, metadata, _ = layouts["dynamics"]
        metadata["model_by_channel"]["axis_long"]["material_fit"]["alpha_inf"] += .001
        with self.assertRaisesRegex(ValueError, "Different Lorentz coefficients"):
            compare_fit_coefficients(material_reference(), {}, arrays, metadata)

    def test_assigning_transverse_fit_to_longitudinal_channel_is_rejected(self):
        arrays, metadata, _ = article_layouts()["gap_master"]
        arrays["channel_id"] = CHANNELS.copy()
        arrays["channel_id"][0], arrays["channel_id"][1] = arrays["channel_id"][1], arrays["channel_id"][0]
        with self.assertRaisesRegex(ValueError, "Different Lorentz coefficients"):
            compare_fit_coefficients(material_reference(), {}, arrays, metadata)

    def test_missing_material_identity_cannot_receive_a_certificate(self):
        with self.assertRaisesRegex(ValueError, "no recognized"):
            compare_fit_coefficients(material_reference(), {}, {}, {})
        arrays, metadata, _ = article_layouts()["fig4"]
        arrays["fit_omega_modes_au"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "Missing finite"):
            compare_fit_coefficients(material_reference(), {}, arrays, metadata)


class WeakFieldInputValidationTests(unittest.TestCase):
    def test_refinement_count_is_a_nonnegative_integer(self):
        for value in (-1, 2.5, True):
            config = load_inputs()
            config["numerics"]["max_weak_field_refinements"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_weak_field_refinements"):
                validate_inputs(config)
        config = load_inputs()
        config["numerics"]["max_weak_field_refinements"] = 0
        validate_inputs(config)

    def test_weak_field_bounds_are_finite_nonnegative_numbers(self):
        for key in ("weak_field_max_population", "weak_field_relative_tolerance"):
            for value in (-1, float("nan"), float("inf"), "0.1", True):
                config = load_inputs()
                config["numerics"][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    validate_inputs(config)
        config = load_inputs()
        config["numerics"]["weak_field_max_population"] = 1.1
        with self.assertRaisesRegex(ValueError, "weak_field_max_population"):
            validate_inputs(config)
        config = load_inputs()
        config["numerics"].update(weak_field_max_population=0., weak_field_relative_tolerance=0.)
        validate_inputs(config)  # Exact bounds are valid, though rarely attainable.


if __name__ == "__main__":
    unittest.main()
