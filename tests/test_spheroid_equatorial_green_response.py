"""Contracts for the full-QS equatorial prolate-spheroid Green kernel."""

from dataclasses import replace
import unittest

import numpy as np

from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
from qd_mnp_rational_fit import (
    HybridQDPlasmonModel,
    eV_to_au,
    make_default_params,
)
from qd_mnp_spheroid_equatorial import (
    MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE,
    EquatorialSpheroidGeometry,
    EquatorialSpheroidGreenInteraction,
)


CHANNELS = (
    ("long", None),
    ("trans", "radial"),
    ("trans", "tangential"),
)


def _geometry(
    orientation: str,
    alignment: str | None,
    *,
    a: float = 1.0,
    c: float = 2.0,
    R: float = 3.0,
    eps_m: float = 1.0,
) -> EquatorialSpheroidGeometry:
    return EquatorialSpheroidGeometry(
        a,
        c,
        R,
        eps_m,
        orientation=orientation,
        side_transverse_alignment=alignment,
    )


class EquatorialGeometryTests(unittest.TestCase):
    def test_geometry_uses_the_equatorial_radius_for_overlap_and_gap(self) -> None:
        geometry = EquatorialSpheroidGeometry(
            2.0,
            5.0,
            2.7,
            1.0,
            qd_radius_au=0.5,
        )
        self.assertAlmostEqual(geometry.surface_gap_au, 0.2)
        self.assertAlmostEqual(
            geometry.xi_qd,
            np.sqrt(geometry.R_au**2 + geometry.focal_length_au**2)
            / geometry.focal_length_au,
        )
        with self.assertRaisesRegex(ValueError, "side QD"):
            replace(geometry, R_au=2.5)

    def test_only_the_three_mirror_symmetric_side_channels_are_accepted(self) -> None:
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                geometry = _geometry(orientation, alignment)
                self.assertIn(
                    geometry.channel,
                    {"long", "transverse_radial", "transverse_tangential"},
                )
        with self.assertRaisesRegex(ValueError, "requires"):
            _geometry("trans", None)
        with self.assertRaisesRegex(ValueError, "applies only"):
            _geometry("long", "radial")

    def test_from_params_requires_and_preserves_the_shared_side_geometry(self) -> None:
        axis = make_default_params("long")
        with self.assertRaisesRegex(ValueError, "qd_placement='side'"):
            EquatorialSpheroidGeometry.from_params(axis, orientation="long")

        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                params = make_default_params(
                    orientation,
                    qd_placement="side",
                    side_transverse_alignment=alignment,
                )
                geometry = EquatorialSpheroidGeometry.from_params(
                    params,
                    orientation=orientation,
                )
                self.assertEqual(geometry.orientation, orientation)
                self.assertEqual(geometry.side_transverse_alignment, alignment)
                self.assertEqual(geometry.surface_gap_au, params.surface_gap_au)

    def test_uniform_length_rescaling_has_A_B_K_dimensions(self) -> None:
        epsilon = -8.0 + 1.1j
        scale = 3.7
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                base_geometry = _geometry(orientation, alignment, eps_m=1.5)
                scaled_geometry = replace(
                    base_geometry,
                    a_au=scale * base_geometry.a_au,
                    c_au=scale * base_geometry.c_au,
                    R_au=scale * base_geometry.R_au,
                )
                base = EquatorialSpheroidGreenInteraction(
                    base_geometry,
                    n_max=24,
                ).response_from_epsilon(epsilon)
                scaled = EquatorialSpheroidGreenInteraction(
                    scaled_geometry,
                    n_max=24,
                ).response_from_epsilon(epsilon)
                np.testing.assert_allclose(scaled.A_au3, scale**3 * base.A_au3)
                np.testing.assert_allclose(scaled.B, base.B, rtol=2.0e-13)
                np.testing.assert_allclose(
                    scaled.K_au_minus3,
                    base.K_au_minus3 / scale**3,
                    rtol=3.0e-13,
                )


class EquatorialModeTests(unittest.TestCase):
    def test_mode_metadata_encodes_the_three_selection_rules(self) -> None:
        expected = {
            ("long", None): (
                (1, 0, "cos"),
                (2, 1, "cos"),
                (3, 0, "cos"),
                (3, 2, "cos"),
                (4, 1, "cos"),
                (4, 3, "cos"),
            ),
            ("trans", "radial"): (
                (1, 1, "cos"),
                (2, 0, "cos"),
                (2, 2, "cos"),
                (3, 1, "cos"),
                (3, 3, "cos"),
                (4, 0, "cos"),
                (4, 2, "cos"),
                (4, 4, "cos"),
            ),
            ("trans", "tangential"): (
                (1, 1, "sin"),
                (2, 2, "sin"),
                (3, 1, "sin"),
                (3, 3, "sin"),
                (4, 2, "sin"),
                (4, 4, "sin"),
            ),
        }
        for channel, modes in expected.items():
            with self.subTest(channel=channel):
                kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(*channel),
                    n_max=4,
                )
                self.assertEqual(kernel.modes, modes)
                self.assertEqual(kernel.mode_count, len(modes))
                self.assertEqual(kernel.bright_mode_index, 0)

    def test_spatial_order_guard_and_truncation_are_strict(self) -> None:
        geometry = _geometry("long", None)
        for invalid in (True, 2.0, np.float64(2.0)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "integer"):
                    EquatorialSpheroidGreenInteraction(geometry, n_max=invalid)
        with self.assertRaisesRegex(
            ValueError,
            "MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE",
        ):
            EquatorialSpheroidGreenInteraction(
                geometry,
                n_max=MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE + 1,
            )

        response = EquatorialSpheroidGreenInteraction(
            geometry,
            n_max=6,
        ).response_from_epsilon(-8.0 + 1.0j)
        truncated = response.truncate(np.int64(3))
        self.assertEqual(truncated.n_max, 3)
        self.assertTrue(np.all(truncated.mode_degrees <= 3))
        np.testing.assert_allclose(
            truncated.K_au_minus3,
            np.sum(response.K_by_degree_au_minus3[:3], axis=0),
            rtol=0.0,
            atol=0.0,
        )
        for invalid in (True, 1.0, 0, 7):
            with self.subTest(truncate_invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must lie"):
                    response.truncate(invalid)

    def test_response_from_modal_susceptibility_preserves_exact_normalization(self) -> None:
        kernel = EquatorialSpheroidGreenInteraction(
            _geometry("trans", "radial"),
            n_max=5,
        )
        modal = (
            np.arange(1, kernel.mode_count + 1, dtype=float)[:, None]
            * np.asarray([1.0 + 0.1j, 0.3 + 0.4j])[None, :]
        )
        response = kernel.response_from_modal_susceptibility(modal)
        np.testing.assert_allclose(response.A_au3, kernel.C_au3 * modal[0])
        np.testing.assert_allclose(
            response.B,
            kernel.bright_source_coupling_au_minus3 * response.A_au3,
        )
        np.testing.assert_allclose(
            response.K_by_mode_au_minus3,
            kernel.reaction_weight_by_mode_au_minus3[:, None] * modal,
        )
        np.testing.assert_allclose(
            response.K_au_minus3,
            np.sum(response.K_by_mode_au_minus3, axis=0),
            rtol=2.0e-16,
        )

    def test_tail_mass_cannot_cancel_between_azimuthal_modes(self) -> None:
        kernel = EquatorialSpheroidGreenInteraction(
            _geometry("trans", "radial"),
            n_max=6,
        )
        tail_mask = kernel.mode_degrees == kernel.n_max
        self.assertEqual(int(np.count_nonzero(tail_mask)), 4)
        desired_K_by_mode = np.zeros(kernel.mode_count, dtype=complex)
        desired_K_by_mode[tail_mask] = np.asarray(
            [1.0, -1.0, 1.0, -1.0 + 1.0e-6],
            dtype=complex,
        )
        modal = (
            desired_K_by_mode
            / kernel.reaction_weight_by_mode_au_minus3
        )
        response = kernel.response_from_modal_susceptibility(modal)
        denominator = max(
            abs(complex(response.K_au_minus3)),
            1.0e-14 * float(np.sum(np.abs(desired_K_by_mode))),
        )
        expected = float(
            np.sum(np.abs(desired_K_by_mode[tail_mask])) / denominator
        )
        actual = float(response.relative_tail_block(block_size=1))
        self.assertAlmostEqual(actual, expected, places=9)
        self.assertGreater(actual, 1.0e6)

    def test_one_degree_convergence_compares_with_the_empty_reaction_sum(self) -> None:
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                response = EquatorialSpheroidGreenInteraction(
                    _geometry(orientation, alignment),
                    n_max=1,
                ).response_from_epsilon(np.asarray([-8.0 + 1.0j, 2.0 + 0.1j]))
                np.testing.assert_allclose(
                    response.relative_half_order_change(),
                    np.ones(2),
                    rtol=2.0e-15,
                )


class EquatorialSphereTests(unittest.TestCase):
    def test_exact_sphere_collapses_each_degree_to_radial_or_tangential_weight(self) -> None:
        a = 1.7
        R = 4.3
        eps_m = 1.9
        n_max = 12
        degree = np.arange(1, n_max + 1, dtype=float)
        radial_scale = a ** (2.0 * degree + 1.0) / R ** (
            2.0 * degree + 4.0
        )
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(
                        orientation,
                        alignment,
                        a=a,
                        c=a,
                        R=R,
                        eps_m=eps_m,
                    ),
                    n_max=n_max,
                )
                if alignment == "radial":
                    expected = (
                        degree * (degree + 1.0) ** 2
                        / (2.0 * degree + 1.0)
                        * radial_scale
                        / eps_m
                    )
                    expected_lambda = 2.0 / (eps_m * R**3)
                else:
                    expected = (
                        degree**2
                        * (degree + 1.0)
                        / (2.0 * (2.0 * degree + 1.0))
                        * radial_scale
                        / eps_m
                    )
                    expected_lambda = -1.0 / (eps_m * R**3)
                by_degree = np.asarray(
                    [
                        np.sum(
                            kernel.reaction_weight_by_mode_au_minus3[
                                kernel.mode_degrees == n
                            ]
                        )
                        for n in range(1, n_max + 1)
                    ]
                )
                np.testing.assert_allclose(by_degree, expected, rtol=2.0e-14)
                np.testing.assert_allclose(
                    kernel.depolarization_by_mode,
                    kernel.mode_degrees / (2.0 * kernel.mode_degrees + 1.0),
                    rtol=2.0e-15,
                )
                self.assertAlmostEqual(
                    kernel.bright_source_coupling_au_minus3,
                    expected_lambda,
                    places=15,
                )

    def test_nearly_spherical_n80_kernel_is_finite_and_continuous(self) -> None:
        epsilon = -8.0 + 1.0j
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                sphere = EquatorialSpheroidGreenInteraction(
                    _geometry(orientation, alignment, a=1.0, c=1.0),
                    n_max=80,
                ).response_from_epsilon(epsilon)
                near_kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(orientation, alignment, a=1.0, c=1.0001),
                    n_max=80,
                )
                near = near_kernel.response_from_epsilon(epsilon)
                self.assertTrue(
                    np.all(
                        np.isfinite(
                            near_kernel.reaction_weight_by_mode_au_minus3
                        )
                    )
                )
                np.testing.assert_allclose(near.A_au3, sphere.A_au3, rtol=3.0e-4)
                np.testing.assert_allclose(near.B, sphere.B, rtol=3.0e-4)
                np.testing.assert_allclose(
                    near.K_au_minus3,
                    sphere.K_au_minus3,
                    rtol=3.0e-4,
                )


class EquatorialResponseTests(unittest.TestCase):
    def test_low_order_kernel_plugs_into_the_full_qs_time_model(self) -> None:
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                params = replace(
                    make_default_params(
                        orientation,
                        qd_placement="side",
                        side_transverse_alignment=alignment,
                    ),
                    gamma_au=float(eV_to_au(0.020)),
                    Gamma_au=float(eV_to_au(0.020)),
                )
                bright = HybridQDPlasmonModel(
                    params,
                    orientation=orientation,
                    n_modes=1,
                    max_fit_normalized_rms=None,
                    max_fit_pointwise_relative_error=None,
                    radiative_consistency_policy="ignore",
                    verbose=False,
                )
                kernel = EquatorialSpheroidGreenInteraction.from_params(
                    params,
                    orientation=orientation,
                    n_max=3,
                )
                model = FullQSSpheroidPulseModel(
                    bright,
                    kernel,
                    fit_quality_policy="ignore",
                    spatial_convergence_policy="ignore",
                    modal_audit_points=201,
                )
                response = model.frequency_response_from_fit(np.asarray([2.042]))
                self.assertEqual(model.n_spatial_modes, kernel.mode_count)
                np.testing.assert_allclose(
                    response.K_bright_au_minus3,
                    response.B**2 / response.A_au3,
                    rtol=5.0e-14,
                )

    def test_bright_A_uses_longitudinal_or_transverse_ellipsoid_factor(self) -> None:
        geometry_values = dict(a=1.0, c=2.0, R=3.0, eps_m=1.4)
        eccentricity = np.sqrt(1.0 - geometry_values["a"] ** 2 / geometry_values["c"] ** 2)
        L_long = (
            (1.0 - eccentricity**2)
            * (np.arctanh(eccentricity) - eccentricity)
            / eccentricity**3
        )
        L_trans = 0.5 * (1.0 - L_long)
        epsilon = np.asarray([-8.0 + 1.0j, 2.3 + 0.2j])
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(orientation, alignment, **geometry_values),
                    n_max=8,
                )
                response = kernel.response_from_epsilon(epsilon)
                L = L_long if orientation == "long" else L_trans
                delta = epsilon - geometry_values["eps_m"]
                expected = kernel.C_au3 * delta / (
                    geometry_values["eps_m"] + L * delta
                )
                self.assertAlmostEqual(
                    kernel.depolarization_by_mode[0],
                    L,
                    places=14,
                )
                np.testing.assert_allclose(response.A_au3, expected, rtol=4.0e-15)

    def test_bright_reaction_is_B_squared_over_A_without_conjugation(self) -> None:
        epsilon = np.asarray([-8.0 + 1.0j, 2.0 + 0.1j])
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                response = EquatorialSpheroidGreenInteraction(
                    _geometry(orientation, alignment),
                    n_max=30,
                ).response_from_epsilon(epsilon)
                np.testing.assert_allclose(
                    response.K_bright_au_minus3,
                    response.B**2 / response.A_au3,
                    rtol=4.0e-15,
                )
                self.assertGreater(
                    float(
                        np.max(
                            np.abs(
                                response.K_bright_au_minus3
                                - np.abs(response.B) ** 2 / response.A_au3
                            )
                        )
                    ),
                    0.0,
                )

    def test_low_order_normalization_fixture(self) -> None:
        # Independently evaluated from the Neumann coefficients and direct
        # associated-Legendre boundary matching at a=1, c=2, R=3, eps_m=1.
        expected = {
            ("long", None): {
                "lambda": -0.028466915703969135,
                "A": 0.568785878292379 + 0.04839496755888997j,
                "B": -0.016191579650957202 - 0.0013776554619952417j,
                "K": 0.0010134323182851089 + 0.00007859901801009585j,
                "K1": 0.00046092433303790073 + 0.000039217601905731204j,
            },
            ("trans", "radial"): {
                "lambda": 0.06234598028445339,
                "A": 0.4727117985331664 + 0.03335180005535544j,
                "B": 0.029471680471577295 + 0.002079350668702222j,
                "K": 0.0029390642102553777 + 0.00020877413249384172j,
                "K1": 0.0018374408096306677 + 0.00012963915579537367j,
            },
            ("trans", "tangential"): {
                "lambda": -0.03387906458048425,
                "A": 0.4727117985331664 + 0.03335180005535544j,
                "B": -0.016015033550462007 - 0.0011299277879507853j,
                "K": 0.0008233422837185503 + 0.0000574553614794206j,
                "K1": 0.0005425743559147243 + 0.00003828089649926837j,
            },
        }
        for channel, reference in expected.items():
            with self.subTest(channel=channel):
                kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(*channel),
                    n_max=80,
                )
                response = kernel.response_from_epsilon(2.0 + 0.1j)
                self.assertAlmostEqual(
                    kernel.bright_source_coupling_au_minus3,
                    reference["lambda"],
                    places=15,
                )
                self.assertTrue(np.isclose(response.A_au3, reference["A"], rtol=3e-14))
                self.assertTrue(np.isclose(response.B, reference["B"], rtol=3e-14))
                self.assertTrue(np.isclose(response.K_au_minus3, reference["K"], rtol=3e-13))
                self.assertTrue(
                    np.isclose(response.K_bright_au_minus3, reference["K1"], rtol=3e-14)
                )

    def test_no_contrast_zeroes_all_channels_and_passive_loss_is_nonnegative(self) -> None:
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(orientation, alignment, eps_m=2.25),
                    n_max=32,
                )
                zero = kernel.response_from_epsilon(2.25 + 0.0j)
                self.assertEqual(complex(zero.A_au3), 0.0j)
                self.assertEqual(complex(zero.B), 0.0j)
                self.assertEqual(complex(zero.K_au_minus3), 0.0j)
                lossy = kernel.response_from_epsilon(
                    np.asarray([-8.0 + 1.0j, 3.0 + 0.2j])
                )
                self.assertGreaterEqual(float(np.min(lossy.A_au3.imag)), -1.0e-15)
                self.assertGreaterEqual(
                    float(np.min(lossy.K_by_mode_au_minus3.imag)),
                    -1.0e-18,
                )

    def test_far_field_recovers_the_correct_side_point_dipole_factor(self) -> None:
        epsilon = -10.0 + 1.0j
        R = 1.0e4
        factors = {
            ("long", None): -1.0,
            ("trans", "radial"): 2.0,
            ("trans", "tangential"): -1.0,
        }
        kernels = {}
        for channel, factor in factors.items():
            with self.subTest(channel=channel):
                kernel = EquatorialSpheroidGreenInteraction(
                    _geometry(*channel, R=R),
                    n_max=12,
                )
                response = kernel.response_from_epsilon(epsilon)
                expected_lambda = factor / R**3
                self.assertTrue(
                    np.isclose(
                        kernel.bright_source_coupling_au_minus3,
                        expected_lambda,
                        rtol=2.0e-7,
                    )
                )
                np.testing.assert_allclose(
                    response.K_au_minus3,
                    response.A_au3 * expected_lambda**2,
                    rtol=8.0e-7,
                )
                kernels[channel] = kernel
        self.assertAlmostEqual(
            kernels[("trans", "radial")].bright_source_coupling_au_minus3
            + kernels[("trans", "tangential")].bright_source_coupling_au_minus3,
            -kernels[("long", None)].bright_source_coupling_au_minus3,
            places=27,
        )

    def test_default_n80_response_is_finite_and_converged(self) -> None:
        for orientation, alignment in CHANNELS:
            with self.subTest(orientation=orientation, alignment=alignment):
                params = make_default_params(
                    orientation,
                    qd_placement="side",
                    side_transverse_alignment=alignment,
                )
                kernel = EquatorialSpheroidGreenInteraction.from_params(
                    params,
                    orientation=orientation,
                    n_max=80,
                )
                response = kernel.response_from_material(
                    params.material,
                    np.asarray([2.042]),
                )
                self.assertIn(kernel.mode_count, {1640, 1680})
                self.assertTrue(np.all(np.isfinite(response.K_by_mode_au_minus3)))
                np.testing.assert_allclose(
                    response.K_au_minus3,
                    np.sum(response.K_by_degree_au_minus3, axis=0),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertLess(float(response.relative_tail_block()[0]), 1.0e-20)


if __name__ == "__main__":
    unittest.main()
