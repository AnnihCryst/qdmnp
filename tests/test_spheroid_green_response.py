"""Algebraic and asymptotic contracts for the full-QS spheroid kernel."""

from dataclasses import replace
import unittest

import numpy as np

from qd_mnp_linear_spectrum import linear_coupled_alpha_au
from qd_mnp_rational_fit import HybridQDPlasmonModel, make_default_params, nm_to_au
from qd_mnp_spheroid_green import (
    LegacyDipoleInteraction,
    MAX_SUPPORTED_SPATIAL_DEGREE,
    ProlateSpheroidGeometry,
    SpheroidGreenInteraction,
    qd_linear_polarizability_from_params,
    solve_linear_hybrid_response,
)


def _material_only_legacy_model(
    orientation: str,
    *,
    qd_placement: str = "axis",
    side_transverse_alignment: str | None = None,
) -> HybridQDPlasmonModel:
    """Construct the material-response part of the old API without fitting."""

    params = make_default_params(
        orientation,
        qd_placement=qd_placement,
        side_transverse_alignment=side_transverse_alignment,
    )
    model = object.__new__(HybridQDPlasmonModel)
    model.params = params
    model.orientation = orientation
    model.L_long, model.L_trans = model._depolarization_factors()
    model.L = model.L_long if orientation == "long" else model.L_trans
    model.C = params.eps_m * params.a_au**2 * params.c_au / 3.0
    model.J = params.G / (params.eps_m * params.R_au**3)
    return model


class SpheroidGeometryTests(unittest.TestCase):
    def test_invalid_or_intersecting_geometry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "prolate or spherical"):
            ProlateSpheroidGeometry(2.0, 1.0, 3.0, 1.0)
        with self.assertRaisesRegex(ValueError, "strictly outside"):
            ProlateSpheroidGeometry(
                1.0,
                2.0,
                2.1,
                1.0,
                qd_radius_au=0.2,
            )

    def test_exact_sphere_is_an_explicit_supported_geometry(self) -> None:
        geometry = ProlateSpheroidGeometry(1.0, 1.0, 3.0, 1.5)
        self.assertTrue(np.isinf(geometry.xi_surface))
        self.assertTrue(np.isinf(geometry.xi_qd))
        self.assertEqual(geometry.focal_length_au, 0.0)

    def test_length_rescaling_has_the_required_A_B_K_dimensions(self) -> None:
        eps = -8.0 + 1.2j
        scale = 3.7
        base_geometry = ProlateSpheroidGeometry(7.0, 15.0, 20.0, 1.5)
        scaled_geometry = ProlateSpheroidGeometry(
            scale * 7.0,
            scale * 15.0,
            scale * 20.0,
            1.5,
        )
        base = SpheroidGreenInteraction(base_geometry, n_max=40).response_from_epsilon(eps)
        scaled = SpheroidGreenInteraction(scaled_geometry, n_max=40).response_from_epsilon(eps)

        np.testing.assert_allclose(scaled.A_au3, scale**3 * base.A_au3, rtol=2e-13)
        np.testing.assert_allclose(scaled.B, base.B, rtol=2e-13)
        np.testing.assert_allclose(
            scaled.K_au_minus3,
            base.K_au_minus3 / scale**3,
            rtol=2e-13,
        )


class SpheroidModeResponseTests(unittest.TestCase):
    def test_exact_spherical_branch_matches_analytic_multipole_weights(self) -> None:
        a = 1.7
        R = 4.3
        eps_m = 1.9
        n_max = 16
        degree = np.arange(1, n_max + 1, dtype=float)
        radial_scale = a ** (2.0 * degree + 1.0) / R ** (
            2.0 * degree + 4.0
        )
        epsilon_particle = np.asarray([-8.0 + 1.0j, 2.4 + 0.2j])

        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                kernel = SpheroidGreenInteraction(
                    ProlateSpheroidGeometry(
                        a,
                        a,
                        R,
                        eps_m,
                        orientation=orientation,
                    ),
                    n_max=n_max,
                )
                if orientation == "long":
                    angular_weight = degree * (degree + 1.0) ** 2 / (
                        2.0 * degree + 1.0
                    )
                else:
                    angular_weight = degree**2 * (degree + 1.0) / (
                        2.0 * (2.0 * degree + 1.0)
                    )
                expected_weight = angular_weight * radial_scale / eps_m
                np.testing.assert_allclose(
                    kernel.depolarization_by_degree,
                    degree / (2.0 * degree + 1.0),
                    rtol=2.0e-15,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    kernel.reaction_weight_by_degree_au_minus3,
                    expected_weight,
                    rtol=1.2e-14,
                    atol=0.0,
                )

                response = kernel.response_from_epsilon(epsilon_particle)
                delta = epsilon_particle - eps_m
                expected_modal = delta[None, :] / (
                    eps_m
                    + (degree / (2.0 * degree + 1.0))[:, None]
                    * delta[None, :]
                )
                np.testing.assert_allclose(
                    response.K_by_degree_au_minus3,
                    expected_weight[:, None] * expected_modal,
                    rtol=9.0e-15,
                    atol=0.0,
                )
                self.assertEqual(response.eps_m, eps_m)

    def test_nearly_spherical_n80_response_is_finite_and_continuous(self) -> None:
        epsilon_particle = -8.0 + 1.0j
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                sphere = SpheroidGreenInteraction(
                    ProlateSpheroidGeometry(
                        1.0,
                        1.0,
                        3.0,
                        1.0,
                        orientation=orientation,
                    ),
                    n_max=80,
                ).response_from_epsilon(epsilon_particle)
                near_kernel = SpheroidGreenInteraction(
                    ProlateSpheroidGeometry(
                        1.0,
                        1.0001,
                        3.0,
                        1.0,
                        orientation=orientation,
                    ),
                    n_max=80,
                )
                near = near_kernel.response_from_epsilon(epsilon_particle)

                self.assertTrue(
                    np.all(
                        np.isfinite(
                            near_kernel.reaction_weight_by_degree_au_minus3
                        )
                    )
                )
                self.assertTrue(
                    np.all(
                        np.isfinite(
                            near_kernel.log_abs_geometric_factor_by_degree
                        )
                    )
                )
                self.assertTrue(np.all(np.isfinite(near.K_by_degree_au_minus3)))
                np.testing.assert_allclose(
                    near.A_au3,
                    sphere.A_au3,
                    rtol=5.0e-4,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    near.B,
                    sphere.B,
                    rtol=5.0e-4,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    near.K_au_minus3,
                    sphere.K_au_minus3,
                    rtol=5.0e-4,
                    atol=0.0,
                )

    def test_guarded_n512_kernel_remains_finite_and_converged(self) -> None:
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                kernel = SpheroidGreenInteraction(
                    ProlateSpheroidGeometry(
                        7.0,
                        15.0,
                        18.0,
                        1.5,
                        orientation=orientation,
                    ),
                    n_max=MAX_SUPPORTED_SPATIAL_DEGREE,
                )
                response = kernel.response_from_epsilon(-8.0 + 1.0j)
                self.assertTrue(
                    np.all(
                        np.isfinite(kernel.reaction_weight_by_degree_au_minus3)
                    )
                )
                self.assertTrue(
                    np.all(
                        np.isfinite(
                            kernel.log_abs_geometric_factor_by_degree
                        )
                    )
                )
                self.assertTrue(np.all(np.isfinite(response.K_by_degree_au_minus3)))
                self.assertLess(float(response.relative_tail_block()), 1.0e-50)

    def test_spatial_order_guard_and_truncation_require_integers(self) -> None:
        geometry = ProlateSpheroidGeometry(1.0, 2.0, 3.0, 1.0)
        for invalid in (True, 2.0, np.float64(2.0)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "integer"):
                    SpheroidGreenInteraction(geometry, n_max=invalid)
        with self.assertRaisesRegex(ValueError, "MAX_SUPPORTED_SPATIAL_DEGREE"):
            SpheroidGreenInteraction(
                geometry,
                n_max=MAX_SUPPORTED_SPATIAL_DEGREE + 1,
            )

        response = SpheroidGreenInteraction(
            geometry,
            n_max=4,
        ).response_from_epsilon(-8.0 + 1.0j)
        for invalid in (True, 1.0, np.float64(1.0)):
            with self.subTest(truncate_invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must lie"):
                    response.truncate(invalid)
        self.assertEqual(response.truncate(np.int64(2)).n_max, 2)

    def test_one_mode_half_order_change_compares_against_empty_sum(self) -> None:
        response = SpheroidGreenInteraction(
            ProlateSpheroidGeometry(1.0, 2.0, 3.0, 1.0),
            n_max=1,
        ).response_from_epsilon(np.asarray([-8.0 + 1.0j, 2.0 + 0.1j]))
        np.testing.assert_allclose(
            response.relative_half_order_change(),
            np.ones(2),
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_frequency_inputs_reject_empty_and_multidimensional_arrays(self) -> None:
        params = make_default_params("long")
        kernel = SpheroidGreenInteraction.from_params(
            params,
            orientation="long",
            n_max=4,
        )
        invalid_inputs = (np.asarray([]), np.ones((1, 2)))
        for invalid in invalid_inputs:
            with self.subTest(kind="epsilon", shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "scalar or a non-empty"):
                    kernel.response_from_epsilon(invalid)
            with self.subTest(kind="energy", shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "scalar or a non-empty"):
                    kernel.response_from_material(params.material, invalid)
            with self.subTest(kind="qd", shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "scalar or a non-empty"):
                    qd_linear_polarizability_from_params(params, invalid)

        legacy = LegacyDipoleInteraction(_material_only_legacy_model("long"))
        for invalid in invalid_inputs:
            with self.subTest(kind="legacy", shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "scalar or a non-empty"):
                    legacy.frequency_response(invalid, mnp_response="material")

    def test_independent_low_order_normalization_fixture(self) -> None:
        # Values were obtained from an independent direct spheroidal-harmonic
        # expansion of the Coulomb dipole potential and boundary conditions.
        expected = {
            "long": {
                "L1": 0.1735639975339643,
                "A": 0.5687858782923798 + 0.0483949675588900j,
                "B": 0.0681559064901257 + 0.00579902386718691j,
                "K": 0.02524783919643397 + 0.00198065221777002j,
                "K1": 0.00816691793304847 + 0.000694879643658856j,
            },
            "trans": {
                "L1": 0.4132180012330179,
                "A": 0.4727117985331669 + 0.0333518000553555j,
                "B": -0.02832181877856347 - 0.00199822310345906j,
                "K": 0.00659716346642222 + 0.000467238722488700j,
                "K1": 0.001696859315580444 + 0.000119720541756978j,
            },
        }
        for orientation, reference in expected.items():
            with self.subTest(orientation=orientation):
                kernel = SpheroidGreenInteraction(
                    ProlateSpheroidGeometry(
                        1.0,
                        2.0,
                        3.0,
                        1.0,
                        orientation=orientation,
                    ),
                    n_max=100,
                )
                response = kernel.response_from_epsilon(2.0 + 0.1j)
                self.assertTrue(
                    np.isclose(
                        kernel.depolarization_by_degree[0],
                        reference["L1"],
                        rtol=2.0e-13,
                    )
                )
                self.assertTrue(np.isclose(response.A_au3, reference["A"], rtol=2e-13))
                self.assertTrue(np.isclose(response.B, reference["B"], rtol=2e-13))
                self.assertTrue(
                    np.isclose(response.K_au_minus3, reference["K"], rtol=2e-13)
                )
                self.assertTrue(
                    np.isclose(response.K_bright_au_minus3, reference["K1"], rtol=2e-13)
                )

    def test_no_dielectric_contrast_zeroes_every_response_channel(self) -> None:
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                geometry = ProlateSpheroidGeometry(
                    7.0,
                    15.0,
                    18.0,
                    2.25,
                    orientation=orientation,
                )
                response = SpheroidGreenInteraction(
                    geometry,
                    n_max=32,
                ).response_from_epsilon(2.25 + 0.0j)
                self.assertEqual(complex(response.A_au3), 0.0j)
                self.assertEqual(complex(response.B), 0.0j)
                self.assertEqual(complex(response.K_au_minus3), 0.0j)
                np.testing.assert_array_equal(
                    response.K_by_degree_au_minus3,
                    np.zeros(32, dtype=complex),
                )

    def test_uniform_field_A_matches_existing_ellipsoid_polarizability(self) -> None:
        epsilon_particle = np.asarray([-12.0 + 0.8j, -3.0 + 2.0j, 4.0 + 0.1j])
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                model = _material_only_legacy_model(orientation)
                kernel = SpheroidGreenInteraction.from_params(
                    model.params,
                    orientation=orientation,
                    n_max=8,
                )
                response = kernel.response_from_epsilon(epsilon_particle)
                expected = model.C * (
                    (epsilon_particle - model.params.eps_m)
                    / (
                        model.params.eps_m
                        + model.L * (epsilon_particle - model.params.eps_m)
                    )
                )
                np.testing.assert_allclose(response.A_au3, expected, rtol=2e-13)
                self.assertAlmostEqual(
                    kernel.depolarization_by_degree[0],
                    model.L,
                    places=12,
                )

    def test_bright_reaction_is_B_squared_over_A_without_conjugation(self) -> None:
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                params = make_default_params(orientation)
                response = SpheroidGreenInteraction.from_params(
                    params,
                    orientation=orientation,
                    n_max=50,
                ).response_from_material(params.material, np.asarray([1.8, 2.042, 2.4]))
                np.testing.assert_allclose(
                    response.K_bright_au_minus3,
                    response.B**2 / response.A_au3,
                    rtol=3e-13,
                    atol=0.0,
                )
                # A conjugated magnitude square would violate reciprocity for
                # a lossy material and must not accidentally replace B**2.
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

    def test_full_reaction_equals_the_exported_modal_sum_and_is_converged(self) -> None:
        params = make_default_params("long")
        response = SpheroidGreenInteraction.from_params(
            params,
            orientation="long",
            n_max=80,
        ).response_from_material(params.material, np.asarray([2.042]))
        np.testing.assert_allclose(
            response.K_au_minus3,
            np.sum(response.K_by_degree_au_minus3, axis=0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertLess(float(response.relative_half_order_change()[0]), 2.0e-8)

    def test_each_passive_material_mode_has_nonnegative_reaction_loss(self) -> None:
        energies = np.linspace(1.0, 3.0, 41)
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                params = make_default_params(orientation)
                response = SpheroidGreenInteraction.from_params(
                    params,
                    orientation=orientation,
                    n_max=40,
                ).response_from_material(params.material, energies)
                self.assertGreaterEqual(
                    float(np.min(response.A_au3.imag)),
                    -1.0e-10,
                )
                self.assertGreaterEqual(
                    float(np.min(response.K_by_degree_au_minus3.imag)),
                    -1.0e-18,
                )

    def test_far_field_recovers_the_legacy_point_dipole_channels(self) -> None:
        epsilon_particle = -10.0 + 1.0j
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                params = replace(
                    make_default_params(orientation),
                    R_au=float(nm_to_au(300.0)),
                )
                legacy_model = _material_only_legacy_model(orientation)
                legacy_model.params = params
                legacy_model.C = params.eps_m * params.a_au**2 * params.c_au / 3.0
                legacy_model.J = params.G / (params.eps_m * params.R_au**3)
                legacy_A = legacy_model.C * (
                    (epsilon_particle - params.eps_m)
                    / (params.eps_m + legacy_model.L * (epsilon_particle - params.eps_m))
                )
                legacy_B = legacy_A * legacy_model.J
                legacy_K = legacy_A * legacy_model.J**2

                exact = SpheroidGreenInteraction.from_params(
                    params,
                    orientation=orientation,
                    n_max=20,
                ).response_from_epsilon(epsilon_particle)
                self.assertTrue(np.isclose(exact.B / legacy_B, 1.0, rtol=3.0e-3))
                self.assertTrue(
                    np.isclose(exact.K_au_minus3 / legacy_K, 1.0, rtol=1.2e-2)
                )


class GenericLinearResponseTests(unittest.TestCase):
    def test_legacy_adapter_uses_the_selected_side_geometry_factor(self) -> None:
        configurations = (
            ("long", None),
            ("trans", "radial"),
            ("trans", "tangential"),
        )
        energies = np.asarray([1.9, 2.042, 2.2])
        for orientation, alignment in configurations:
            with self.subTest(orientation=orientation, alignment=alignment):
                model = _material_only_legacy_model(
                    orientation,
                    qd_placement="side",
                    side_transverse_alignment=alignment,
                )
                response = LegacyDipoleInteraction(model).frequency_response(
                    energies,
                    mnp_response="material",
                )
                expected_A = model.C * model.alpha_from_material(energies)
                np.testing.assert_allclose(
                    response.A_au3,
                    expected_A,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    response.B,
                    expected_A * model.J,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    response.K_au_minus3,
                    expected_A * model.J**2,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    response.K_au_minus3,
                    response.B * model.J,
                    rtol=5.0e-16,
                    atol=0.0,
                )

    def test_legacy_adapter_supports_a_sphere_and_exports_consistent_metadata(self) -> None:
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                model = _material_only_legacy_model(orientation)
                params = replace(model.params, c_au=model.params.a_au)
                model.params = params
                model.L_long = model.L_trans = 1.0 / 3.0
                model.L = 1.0 / 3.0
                model.C = params.eps_m * params.a_au**3 / 3.0
                model.J = params.G / (params.eps_m * params.R_au**3)

                response = LegacyDipoleInteraction(model).frequency_response(
                    np.asarray([1.8, 2.2]),
                    mnp_response="material",
                )
                expected_modal = response.A_au3 / model.C
                expected_weight = model.C * model.J**2
                np.testing.assert_allclose(
                    response.modal_susceptibility_by_degree[0],
                    expected_modal,
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    response.reaction_weight_by_degree_au_minus3,
                    np.asarray([expected_weight]),
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    response.K_by_degree_au_minus3,
                    response.reaction_weight_by_degree_au_minus3[:, None]
                    * response.modal_susceptibility_by_degree,
                    rtol=2.0e-16,
                    atol=0.0,
                )
                self.assertAlmostEqual(
                    response.depolarization_by_degree[0],
                    1.0 / 3.0,
                    places=15,
                )
                self.assertEqual(response.eps_m, params.eps_m)
                self.assertEqual(response.truncate(1).model, "legacy")

    def test_legacy_adapter_reproduces_the_existing_linear_api(self) -> None:
        energies = np.linspace(2.0, 2.08, 101)
        for orientation in ("long", "trans"):
            with self.subTest(orientation=orientation):
                model = _material_only_legacy_model(orientation)
                old_alpha, _, _, _ = linear_coupled_alpha_au(
                    model,
                    energies,
                    mnp_response="material",
                )
                interaction = LegacyDipoleInteraction(model).frequency_response(
                    energies,
                    mnp_response="material",
                )
                beta = qd_linear_polarizability_from_params(model.params, energies)
                generic = solve_linear_hybrid_response(
                    interaction,
                    beta,
                    eps_m=model.params.eps_m,
                )
                np.testing.assert_allclose(
                    generic.alpha_effective_au3,
                    old_alpha,
                    rtol=3.0e-15,
                    atol=0.0,
                )

    def test_generic_solution_satisfies_both_coupled_equations(self) -> None:
        params = make_default_params("long")
        energies = np.asarray([1.9, 2.042, 2.2])
        response = SpheroidGreenInteraction.from_params(
            params,
            orientation="long",
            n_max=60,
        ).response_from_material(params.material, energies)
        beta = qd_linear_polarizability_from_params(params, energies)
        solved = solve_linear_hybrid_response(response, beta, eps_m=params.eps_m)

        with self.assertRaisesRegex(ValueError, "must match"):
            solve_linear_hybrid_response(
                response,
                beta,
                eps_m=1.1 * params.eps_m,
            )

        np.testing.assert_allclose(
            solved.mnp_dipole_over_field_au3,
            response.A_au3 + response.B * solved.qd_dipole_over_field_au3,
            rtol=2e-15,
        )
        np.testing.assert_allclose(
            solved.qd_dipole_over_field_au3,
            beta
            * (
                1.0
                + response.B
                + response.K_au_minus3 * solved.qd_dipole_over_field_au3
            ),
            rtol=2e-15,
        )


if __name__ == "__main__":
    unittest.main()
