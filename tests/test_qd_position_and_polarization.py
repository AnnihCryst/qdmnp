"""Contract tests for the independent QD position and field polarization.

The MNP is the axisymmetric prolate spheroid ``(x**2+y**2)/a**2+z**2/c**2=1``
with ``c>a``.  The QD sits either at ``r_D=(0,0,c+h)`` on the long axis (tip) or
at ``r_D=(a+h,0,0)`` beside the particle (equatorial), and the incident field is
polarized either along ``e_z`` (longitudinal) or along ``e_x`` (transverse).
The two choices are independent, so all four combinations must be supported.

The exact spheroidal kernel is checked against a short direct evaluation of the
same modal formulas with SciPy's associated Legendre functions, against the
sphere limit, and against the point-dipole limit at large separation.
"""

from dataclasses import replace
from math import factorial
import unittest

import numpy as np
from scipy.special import assoc_legendre_p_all, lqmn

from qd_mnp_full_qs_model import FullQSSpheroidPulseModel
from qd_mnp_rational_fit import (
    GaussianPulse,
    HybridQDPlasmonModel,
    au_to_nm,
    eV_to_au,
    field_polarization_unit_vector,
    fs_to_au,
    geometric_coupling_factor,
    make_default_params,
    make_params_with_overrides,
    qd_position_unit_vector,
    resolve_field_polarization,
)
from qd_mnp_spheroid_green import (
    ProlateSpheroidGeometry,
    SpheroidGreenInteraction,
    legacy_dipole_response_from_A,
)

POSITIONS = ("tip", "equatorial")
POLARIZATIONS = ("longitudinal", "transverse")
EXPECTED_G = {
    ("tip", "longitudinal"): 2.0,
    ("tip", "transverse"): -1.0,
    ("equatorial", "longitudinal"): -1.0,
    ("equatorial", "transverse"): 2.0,
}


def _hobson_radial(n_max: int, x: float):
    """P_n^m, P_n^m', Q_n^m, Q_n^m' for x>1 as [n, m] tables."""

    values = assoc_legendre_p_all(n_max, n_max, x, branch_cut=3, diff_n=1)
    Q, Q_prime = lqmn(n_max, n_max, x)
    return (
        values[0][:, : n_max + 1],
        values[1][:, : n_max + 1],
        Q.T[:, : n_max + 1],
        Q_prime.T[:, : n_max + 1],
    )


def _ferrers_without_condon_shortley(n_max: int, eta: float):
    """Ferrers P_n^m(eta) and its eta-derivative without the (-1)**m phase."""

    values = assoc_legendre_p_all(n_max, n_max, eta, branch_cut=2, diff_n=1)
    phase = np.asarray([(-1.0) ** m for m in range(n_max + 1)])
    return values[0][:, : n_max + 1] * phase, values[1][:, : n_max + 1] * phase


def reference_modes(geometry: ProlateSpheroidGeometry, n_max: int):
    """Directly evaluated (n, m, L_nm, w_nm) for the analytic modal kernel.

    ``K = sum_nm w_nm*chi_nm`` with ``chi_nm`` the modal susceptibility and

    ``w_nm = |A_nm|*|g_nm|*(e_d.grad u_nm(r_D))**2/(eps_m*f)``,
    ``A_nm = (-1)**m*(2-delta_m0)*(2n+1)*((n-m)!/(n+m)!)**2``,
    ``g_nm = -L_nm*P_n^m(xi_0)/Q_n^m(xi_0)``.

    This mirrors the production kernel's algebra but uses SciPy's functions
    directly instead of the scaled log recurrences, so it is an independent
    numerical check of those recurrences at moderate degree.
    """

    focal = geometry.focal_length_au
    xi_surface = geometry.xi_surface
    xi_qd = geometry.xi_qd
    P, P_prime, Q, Q_prime = _hobson_radial(n_max, xi_surface)
    _, _, Q_qd, Q_prime_qd = _hobson_radial(n_max, xi_qd)
    ferrers, ferrers_prime = _ferrers_without_condon_shortley(n_max, geometry.eta_qd)
    radial = geometry.field_polarization == "transverse"

    modes = []
    for degree in range(1, n_max + 1):
        for order in range(0, degree + 1):
            abs_Q = ((-1.0) ** order) * Q[degree, order]
            abs_Q_prime = ((-1.0) ** (order + 1)) * Q_prime[degree, order]
            depolarization = (abs_Q * P_prime[degree, order]) / (
                abs_Q * P_prime[degree, order] + P[degree, order] * abs_Q_prime
            )
            abs_geometric = depolarization * P[degree, order] / abs_Q
            abs_expansion = (
                (1.0 if order == 0 else 2.0)
                * (2 * degree + 1)
                * (factorial(degree - order) / factorial(degree + order)) ** 2
            )
            if geometry.qd_position == "tip":
                if radial and order != 1:
                    continue
                if not radial and order != 0:
                    continue
                axial_derivative = Q_prime_qd[degree, 0] / focal
                derivative = (
                    0.5 * degree * (degree + 1) * axial_derivative
                    if radial
                    else axial_derivative
                )
            else:
                if (degree - order) % 2 != (0 if radial else 1):
                    continue
                if radial:
                    derivative = (
                        np.sqrt(xi_qd * xi_qd - 1.0)
                        / (focal * xi_qd)
                        * Q_prime_qd[degree, order]
                        * ferrers[degree, order]
                    )
                else:
                    derivative = (
                        Q_qd[degree, order]
                        * ferrers_prime[degree, order]
                        / (focal * xi_qd)
                    )
            weight = (
                abs_expansion
                * abs_geometric
                * derivative**2
                / (geometry.eps_m * focal)
            )
            modes.append((degree, order, depolarization, weight))
    return modes


def reference_bright_coupling(geometry: ProlateSpheroidGeometry) -> float:
    """Field at the QD per unit MNP dipole, projected on the QD dipole."""

    focal = geometry.focal_length_au
    xi_qd = geometry.xi_qd
    _, _, Q_qd, Q_prime_qd = _hobson_radial(1, xi_qd)
    ferrers, ferrers_prime = _ferrers_without_condon_shortley(1, geometry.eta_qd)
    if geometry.field_polarization == "longitudinal":
        if geometry.qd_position == "tip":
            derivative = Q_prime_qd[1, 0] / focal
        else:
            derivative = Q_qd[1, 0] * ferrers_prime[1, 0] / (focal * xi_qd)
        return float(-3.0 * derivative / (geometry.eps_m * focal**2))
    if geometry.qd_position == "tip":
        derivative = Q_prime_qd[1, 0] / focal
    else:
        derivative = (
            np.sqrt(xi_qd * xi_qd - 1.0)
            / (focal * xi_qd)
            * Q_prime_qd[1, 1]
            * ferrers[1, 1]
        )
    return float(1.5 * derivative / (geometry.eps_m * focal**2))


class GeometryPrimitiveTests(unittest.TestCase):
    def test_position_and_polarization_vectors(self) -> None:
        np.testing.assert_allclose(qd_position_unit_vector("tip"), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            qd_position_unit_vector("equatorial"), [1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            field_polarization_unit_vector("longitudinal"), [0.0, 0.0, 1.0]
        )
        np.testing.assert_allclose(
            field_polarization_unit_vector("transverse"), [1.0, 0.0, 0.0]
        )

    def test_all_four_combinations_have_the_documented_dipole_factor(self) -> None:
        for (position, polarization), expected in EXPECTED_G.items():
            with self.subTest(position=position, polarization=polarization):
                self.assertEqual(
                    geometric_coupling_factor(position, polarization), expected
                )

    def test_invalid_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "qd_position"):
            geometric_coupling_factor("pole", "longitudinal")
        with self.assertRaisesRegex(ValueError, "field_polarization"):
            geometric_coupling_factor("tip", "sideways")

    def test_orientation_alias_round_trips(self) -> None:
        self.assertEqual(resolve_field_polarization("long", None), "longitudinal")
        self.assertEqual(resolve_field_polarization("trans", None), "transverse")
        self.assertEqual(resolve_field_polarization(None, "transverse"), "transverse")
        self.assertEqual(resolve_field_polarization("trans", "transverse"), "transverse")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            resolve_field_polarization("long", "transverse")


class ParameterGeometryTests(unittest.TestCase):
    def test_parameters_expose_the_actual_position_vector(self) -> None:
        for position in POSITIONS:
            params = make_params_with_overrides(r_nm=12.0, qd_position=position)
            expected = 12.0 * np.asarray(
                [0.0, 0.0, 1.0] if position == "tip" else [1.0, 0.0, 0.0]
            )
            np.testing.assert_allclose(
                au_to_nm(params.qd_position_vector_au), expected, atol=1e-9
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(params.qd_position_vector_au)),
                float(params.R_au),
            )

    def test_surface_gap_uses_the_semiaxis_along_the_qd_direction(self) -> None:
        tip = make_params_with_overrides(
            c_nm=15.0, a_nm=7.0, r_nm=20.0, qd_radius_nm=2.0, qd_position="tip"
        )
        equatorial = replace(tip, qd_position="equatorial")
        self.assertAlmostEqual(float(au_to_nm(tip.surface_gap_au)), 3.0)
        self.assertAlmostEqual(float(au_to_nm(equatorial.surface_gap_au)), 11.0)
        self.assertAlmostEqual(float(au_to_nm(tip.mnp_directional_semiaxis_au)), 15.0)
        self.assertAlmostEqual(
            float(au_to_nm(equatorial.mnp_directional_semiaxis_au)), 7.0
        )

    def test_equatorial_qd_may_sit_closer_than_the_long_semiaxis(self) -> None:
        # R=10 nm overlaps a 15 nm long semiaxis but clears the 7 nm short one.
        def build(position):
            return make_params_with_overrides(
                c_nm=15.0,
                a_nm=7.0,
                r_nm=10.0,
                qd_radius_nm=2.0,
                qd_position=position,
            )

        beside = build("equatorial")
        self.assertAlmostEqual(float(au_to_nm(beside.surface_gap_au)), 1.0)
        HybridQDPlasmonModel(
            beside,
            n_modes=1,
            radiative_consistency_policy="ignore",
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            verbose=False,
        )
        self.assertAlmostEqual(float(au_to_nm(build("tip").surface_gap_au)), -7.0)
        with self.assertRaisesRegex(ValueError, "surface gap"):
            HybridQDPlasmonModel(
                build("tip"),
                n_modes=1,
                radiative_consistency_policy="ignore",
                max_fit_normalized_rms=None,
                max_fit_pointwise_relative_error=None,
                verbose=False,
            )

    def test_default_parameters_carry_a_consistent_dipole_factor(self) -> None:
        for position in POSITIONS:
            for polarization in POLARIZATIONS:
                with self.subTest(position=position, polarization=polarization):
                    params = make_default_params(
                        qd_position=position, field_polarization=polarization
                    )
                    self.assertEqual(params.G, EXPECTED_G[(position, polarization)])
                    self.assertEqual(params.geometric_coupling_factor, params.G)

    def test_hand_edited_g_that_contradicts_the_geometry_is_rejected(self) -> None:
        params = replace(
            make_default_params(qd_position="equatorial", field_polarization="transverse"),
            G=-1.0,
        )
        with self.assertRaisesRegex(ValueError, "geometric value"):
            HybridQDPlasmonModel(params, n_modes=1, verbose=False)


class NativeCouplingTests(unittest.TestCase):
    """The point-dipole core must use r_D and e_L, not a tuned coefficient."""

    def _model(self, position, polarization, r_nm=12.0):
        # A short particle keeps the same R legal at both QD positions, so the
        # tip/equatorial comparison changes only the geometry, not R.
        params = make_params_with_overrides(
            c_nm=8.0,
            a_nm=5.0,
            r_nm=r_nm,
            qd_radius_nm=0.0,
            qd_position=position,
            field_polarization=polarization,
        )
        # J and L are pure geometry; the modal fit is irrelevant here.
        return HybridQDPlasmonModel(
            params,
            n_modes=1,
            radiative_consistency_policy="ignore",
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            verbose=False,
        )

    def test_coupling_follows_the_geometric_factor_and_the_actual_distance(self) -> None:
        for position in POSITIONS:
            for polarization in POLARIZATIONS:
                with self.subTest(position=position, polarization=polarization):
                    model = self._model(position, polarization)
                    expected = EXPECTED_G[(position, polarization)] / (
                        model.params.eps_m * model.params.R_au**3
                    )
                    self.assertAlmostEqual(model.J / expected, 1.0, places=12)

    def test_polarizability_branch_depends_only_on_the_polarization(self) -> None:
        for polarization in POLARIZATIONS:
            tip = self._model("tip", polarization)
            equatorial = self._model("equatorial", polarization)
            with self.subTest(polarization=polarization):
                self.assertEqual(tip.L, equatorial.L)
                expected = tip.L_long if polarization == "longitudinal" else tip.L_trans
                self.assertEqual(tip.L, expected)
        self.assertNotEqual(
            self._model("tip", "longitudinal").L,
            self._model("tip", "transverse").L,
        )

    def test_the_two_choices_are_genuinely_independent(self) -> None:
        """Equal G with different plasmon branches, and vice versa."""

        parallel_long = self._model("tip", "longitudinal")
        parallel_trans = self._model("equatorial", "transverse")
        self.assertAlmostEqual(parallel_long.J, parallel_trans.J, places=14)
        self.assertNotEqual(parallel_long.L, parallel_trans.L)

        same_branch_tip = self._model("tip", "longitudinal")
        same_branch_equatorial = self._model("equatorial", "longitudinal")
        self.assertEqual(same_branch_tip.L, same_branch_equatorial.L)
        self.assertAlmostEqual(
            same_branch_equatorial.J / same_branch_tip.J, -0.5, places=12
        )


class SpheroidKernelGeometryTests(unittest.TestCase):
    def _geometry(self, position, polarization, *, a=7.0, c=15.0, r=None, eps_m=1.4):
        if r is None:
            r = 18.0 if position == "tip" else 11.0
        return ProlateSpheroidGeometry(
            a_au=a,
            c_au=c,
            R_au=r,
            eps_m=eps_m,
            qd_position=position,
            field_polarization=polarization,
        )

    def test_qd_coordinates_match_the_position_vector(self) -> None:
        tip = self._geometry("tip", "longitudinal")
        equatorial = self._geometry("equatorial", "longitudinal")
        focal = tip.focal_length_au
        self.assertAlmostEqual(tip.eta_qd, 1.0)
        self.assertAlmostEqual(tip.xi_qd, tip.R_au / focal)
        self.assertAlmostEqual(equatorial.eta_qd, 0.0)
        # x = f*sqrt(xi**2-1) must reproduce R for the equatorial QD.
        self.assertAlmostEqual(
            focal * np.sqrt(equatorial.xi_qd**2 - 1.0), equatorial.R_au
        )
        np.testing.assert_allclose(
            equatorial.qd_position_vector_au, [equatorial.R_au, 0.0, 0.0]
        )

    def test_modes_match_the_direct_reference_evaluation(self) -> None:
        n_max = 16
        for position in POSITIONS:
            for polarization in POLARIZATIONS:
                with self.subTest(position=position, polarization=polarization):
                    geometry = self._geometry(position, polarization)
                    kernel = SpheroidGreenInteraction(geometry, n_max=n_max)
                    reference = reference_modes(geometry, n_max)
                    self.assertEqual(
                        list(zip(kernel.degrees.tolist(), kernel.azimuthal_orders.tolist())),
                        [(degree, order) for degree, order, _, _ in reference],
                    )
                    np.testing.assert_allclose(
                        kernel.depolarization_by_degree,
                        [value for _, _, value, _ in reference],
                        rtol=1e-12,
                    )
                    np.testing.assert_allclose(
                        kernel.reaction_weight_by_degree_au_minus3,
                        [value for _, _, _, value in reference],
                        rtol=1e-11,
                    )
                    self.assertAlmostEqual(
                        kernel.bright_source_coupling_au_minus3
                        / reference_bright_coupling(geometry),
                        1.0,
                        places=11,
                    )

    def test_equatorial_families_have_the_expected_parity_and_ordering(self) -> None:
        n_max = 9
        for polarization, parity in (("transverse", 0), ("longitudinal", 1)):
            with self.subTest(polarization=polarization):
                kernel = SpheroidGreenInteraction(
                    self._geometry("equatorial", polarization), n_max=n_max
                )
                degrees = kernel.degrees
                orders = kernel.azimuthal_orders
                self.assertTrue(np.all((degrees - orders) % 2 == parity))
                self.assertTrue(np.all(orders <= degrees))
                self.assertGreater(kernel.mode_count, n_max)
                labels = list(zip(degrees.tolist(), orders.tolist()))
                self.assertEqual(labels, sorted(labels))
                # The bright n=1 harmonic of the incident polarization is first.
                self.assertEqual(labels[0], (1, 0 if parity else 1))

    def test_axial_kernels_keep_a_single_azimuthal_order(self) -> None:
        for polarization, order in (("longitudinal", 0), ("transverse", 1)):
            kernel = SpheroidGreenInteraction(
                self._geometry("tip", polarization), n_max=12
            )
            with self.subTest(polarization=polarization):
                self.assertEqual(kernel.mode_count, 12)
                np.testing.assert_array_equal(kernel.degrees, np.arange(1, 13))
                np.testing.assert_array_equal(
                    kernel.azimuthal_orders, np.full(12, order)
                )

    def test_bright_reciprocity_holds_for_every_combination(self) -> None:
        for position in POSITIONS:
            for polarization in POLARIZATIONS:
                with self.subTest(position=position, polarization=polarization):
                    geometry = self._geometry(position, polarization)
                    kernel = SpheroidGreenInteraction(geometry, n_max=20)
                    C = geometry.eps_m * geometry.a_au**2 * geometry.c_au / 3.0
                    self.assertAlmostEqual(
                        kernel.reaction_weight_by_degree_au_minus3[0]
                        / (C * kernel.bright_source_coupling_au_minus3**2),
                        1.0,
                        places=11,
                    )

    def test_bright_depolarization_depends_only_on_the_polarization(self) -> None:
        for polarization in POLARIZATIONS:
            values = {
                position: SpheroidGreenInteraction(
                    self._geometry(position, polarization), n_max=6
                ).depolarization_by_degree[0]
                for position in POSITIONS
            }
            with self.subTest(polarization=polarization):
                self.assertAlmostEqual(values["tip"], values["equatorial"], places=13)

    def test_bright_coupling_sign_tracks_the_dipole_tensor_factor(self) -> None:
        for position in POSITIONS:
            for polarization in POLARIZATIONS:
                with self.subTest(position=position, polarization=polarization):
                    kernel = SpheroidGreenInteraction(
                        self._geometry(position, polarization), n_max=4
                    )
                    self.assertEqual(
                        np.sign(kernel.bright_source_coupling_au_minus3),
                        np.sign(EXPECTED_G[(position, polarization)]),
                    )

    def test_sphere_makes_the_two_positions_equivalent_at_equal_g(self) -> None:
        """A sphere has no axis, so only the dipole/radius angle matters."""

        for equatorial_polarization, tip_polarization in (
            ("transverse", "longitudinal"),
            ("longitudinal", "transverse"),
        ):
            with self.subTest(polarization=equatorial_polarization):
                equatorial = SpheroidGreenInteraction(
                    self._geometry(
                        "equatorial", equatorial_polarization, a=6.0, c=6.0, r=10.0
                    ),
                    n_max=14,
                )
                tip = SpheroidGreenInteraction(
                    self._geometry("tip", tip_polarization, a=6.0, c=6.0, r=10.0),
                    n_max=14,
                )
                np.testing.assert_allclose(
                    equatorial.reaction_weight_by_degree_au_minus3,
                    tip.reaction_weight_by_degree_au_minus3,
                    rtol=0.0,
                )
                self.assertEqual(
                    equatorial.bright_source_coupling_au_minus3,
                    tip.bright_source_coupling_au_minus3,
                )

    def test_weakly_prolate_equatorial_kernel_approaches_the_sphere(self) -> None:
        n_max = 8
        previous = None
        for aspect_ratio in (1.01, 1.003, 1.001):
            prolate = SpheroidGreenInteraction(
                self._geometry(
                    "equatorial", "transverse", a=7.0, c=7.0 * aspect_ratio, r=11.0
                ),
                n_max=n_max,
            )
            sphere = SpheroidGreenInteraction(
                self._geometry("equatorial", "transverse", a=7.0, c=7.0, r=11.0),
                n_max=n_max,
            )
            # Rotations mix azimuthal orders but preserve the spatial degree,
            # so the sphere limit is a per-degree statement.
            summed = np.zeros(n_max)
            for degree, weight in zip(
                prolate.degrees, prolate.reaction_weight_by_degree_au_minus3
            ):
                summed[degree - 1] += weight
            error = float(
                np.max(
                    np.abs(summed / sphere.reaction_weight_by_degree_au_minus3 - 1.0)
                )
            )
            self.assertLess(error, 4.0 * (aspect_ratio - 1.0))
            if previous is not None:
                self.assertLess(error, previous)
            previous = error

    def test_far_field_reaction_approaches_the_point_dipole_limit(self) -> None:
        """The exact kernel must decay onto A*J**2 like the quadrupole term.

        The leading correction to a central point dipole is O((c/R)**2), so
        doubling the separation must shrink the discrepancy roughly fourfold
        for every position/polarization pair.
        """

        epsilon = np.asarray([-8.0 + 1.2j])
        for position in POSITIONS:
            for polarization in POLARIZATIONS:
                previous = None
                for separation in (120.0, 240.0, 480.0):
                    geometry = self._geometry(position, polarization, r=separation)
                    kernel = SpheroidGreenInteraction(geometry, n_max=12)
                    full = kernel.response_from_epsilon(epsilon)
                    legacy = legacy_dipole_response_from_A(full.A_au3, geometry)
                    error = float(
                        np.abs(full.K_au_minus3[0] / legacy.K_au_minus3[0] - 1.0)
                    )
                    with self.subTest(
                        position=position,
                        polarization=polarization,
                        separation=separation,
                    ):
                        if previous is not None:
                            self.assertLess(3.0, previous / error)
                            self.assertLess(previous / error, 5.0)
                    previous = error
                with self.subTest(position=position, polarization=polarization):
                    self.assertLess(previous, 0.005)

    def test_truncation_keeps_whole_degrees(self) -> None:
        kernel = SpheroidGreenInteraction(
            self._geometry("equatorial", "transverse"), n_max=10
        )
        response = kernel.response_from_epsilon(np.asarray([-8.0 + 1.2j]))
        truncated = response.truncate(4)
        self.assertEqual(int(truncated.degrees[-1]), 4)
        np.testing.assert_array_equal(
            truncated.degrees, response.degrees[response.degrees <= 4]
        )
        np.testing.assert_array_equal(
            truncated.azimuthal_orders,
            response.azimuthal_orders[response.degrees <= 4],
        )
        self.assertEqual(truncated.qd_position, "equatorial")
        np.testing.assert_allclose(
            truncated.K_au_minus3,
            np.sum(response.K_by_degree_au_minus3[response.degrees <= 4], axis=0),
        )

    def test_geometry_rejects_a_qd_inside_the_particle(self) -> None:
        with self.assertRaisesRegex(ValueError, "a_au"):
            ProlateSpheroidGeometry(
                a_au=7.0,
                c_au=15.0,
                R_au=6.5,
                eps_m=1.0,
                qd_position="equatorial",
            )
        # The same centre distance is legal beside the particle but not at the tip.
        ProlateSpheroidGeometry(
            a_au=7.0, c_au=15.0, R_au=9.0, eps_m=1.0, qd_position="equatorial"
        )
        with self.assertRaisesRegex(ValueError, "c_au"):
            ProlateSpheroidGeometry(
                a_au=7.0, c_au=15.0, R_au=9.0, eps_m=1.0, qd_position="tip"
            )

    def test_geometry_from_params_rejects_a_contradicting_override(self) -> None:
        params = make_params_with_overrides(
            r_nm=11.0, qd_position="equatorial", field_polarization="transverse"
        )
        geometry = ProlateSpheroidGeometry.from_params(params)
        self.assertEqual(geometry.qd_position, "equatorial")
        self.assertEqual(geometry.field_polarization, "transverse")
        self.assertEqual(geometry.orientation, "trans")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            ProlateSpheroidGeometry.from_params(params, qd_position="tip")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            ProlateSpheroidGeometry.from_params(params, orientation="long")


class FullQSGeometryTests(unittest.TestCase):
    def _build(self, position, polarization, n_max):
        params = make_params_with_overrides(
            r_nm=18.0 if position == "tip" else 12.0,
            qd_position=position,
            field_polarization=polarization,
        )
        legacy = HybridQDPlasmonModel(
            params,
            n_modes=9,
            radiative_consistency_policy="ignore",
            verbose=False,
        )
        kernel = SpheroidGreenInteraction.from_params(params, n_max=n_max)
        return params, legacy, kernel

    def test_equatorial_full_qs_model_carries_every_azimuthal_mode(self) -> None:
        params, legacy, kernel = self._build("equatorial", "transverse", 12)
        model = FullQSSpheroidPulseModel(
            legacy,
            kernel,
            spatial_convergence_policy="ignore",
            fit_quality_policy="ignore",
        )
        self.assertEqual(model.qd_position, "equatorial")
        self.assertEqual(model.field_polarization, "transverse")
        self.assertEqual(model.n_spatial_modes, kernel.mode_count)
        self.assertGreater(model.n_spatial_modes, kernel.n_max)
        self.assertEqual(
            model.state_size,
            2 * kernel.mode_count * legacy.n_modes + 4,
        )
        response = model.frequency_response_from_fit(np.asarray([2.042]))
        self.assertEqual(response.qd_position, "equatorial")
        np.testing.assert_array_equal(response.degrees, kernel.degrees)
        np.testing.assert_array_equal(
            response.azimuthal_orders, kernel.azimuthal_orders
        )

    def test_equatorial_time_derivative_is_finite_and_uses_the_bright_channel(
        self,
    ) -> None:
        params, legacy, kernel = self._build("equatorial", "longitudinal", 10)
        model = FullQSSpheroidPulseModel(
            legacy,
            kernel,
            spatial_convergence_policy="ignore",
            fit_quality_policy="ignore",
        )
        pulse = GaussianPulse(
            E0_au=1.0e-5,
            omegaL_au=float(eV_to_au(2.042)),
            tau_au=float(fs_to_au(20.0)),
            tau_kind="fwhm_intensity",
        )
        state = model.initial_state()
        state[model.P_index] = 1.0e-3
        derivative = model.rhs(0.0, state, pulse)
        self.assertEqual(derivative.shape, (model.state_size,))
        self.assertTrue(np.all(np.isfinite(derivative)))
        # Only the bright mode is driven by the incident field; every other
        # (n, m) mode is driven by the QD dipole alone.
        velocities = derivative[1 : model.mode_state_count : 2]
        self.assertGreater(
            float(np.max(np.abs(velocities[: legacy.n_modes]))),
            0.0,
        )

    def test_mismatched_kernel_geometry_is_rejected(self) -> None:
        # A short particle keeps one centre distance legal at both QD
        # positions, so only the position/polarization labels differ.
        params = make_params_with_overrides(
            c_nm=8.0,
            a_nm=7.0,
            r_nm=12.0,
            qd_radius_nm=2.0,
            qd_position="equatorial",
            field_polarization="transverse",
        )
        legacy = HybridQDPlasmonModel(
            params,
            n_modes=9,
            radiative_consistency_policy="ignore",
            max_fit_normalized_rms=None,
            max_fit_pointwise_relative_error=None,
            verbose=False,
        )
        for override, message in (
            ({"qd_position": "tip"}, "QD positions"),
            ({"field_polarization": "longitudinal"}, "incident"),
        ):
            mismatched = SpheroidGreenInteraction(
                ProlateSpheroidGeometry(
                    a_au=params.a_au,
                    c_au=params.c_au,
                    R_au=params.R_au,
                    eps_m=params.eps_m,
                    qd_radius_au=params.qd_radius_au,
                    **{
                        "qd_position": "equatorial",
                        "field_polarization": "transverse",
                        **override,
                    },
                ),
                n_max=8,
            )
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    FullQSSpheroidPulseModel(
                        legacy,
                        mismatched,
                        spatial_convergence_policy="ignore",
                        fit_quality_policy="ignore",
                    )

    def test_weak_field_full_qs_reduces_to_the_native_core_at_large_gap(self) -> None:
        """Far from the particle both backends must agree on the QD feedback."""

        energies = np.asarray([2.042])
        for polarization in POLARIZATIONS:
            params = make_params_with_overrides(
                r_nm=400.0,
                qd_position="equatorial",
                field_polarization=polarization,
            )
            legacy = HybridQDPlasmonModel(
                params,
                n_modes=9,
                radiative_consistency_policy="ignore",
                verbose=False,
            )
            kernel = SpheroidGreenInteraction.from_params(params, n_max=10)
            full = kernel.response_from_material(params.material, energies)
            point = legacy_dipole_response_from_A(
                legacy.C * np.asarray(legacy.alpha_from_material(energies)),
                ProlateSpheroidGeometry.from_params(params),
            )
            # The residual is the O((c/R)**2) quadrupole correction, which is
            # 1.4e-3 for c=15 nm at R=400 nm.
            tolerance = 3.0 * (params.c_au / params.R_au) ** 2
            with self.subTest(polarization=polarization):
                self.assertLess(
                    float(np.abs(full.B[0] / point.B[0] - 1.0)), tolerance
                )
                self.assertLess(
                    float(
                        np.abs(full.K_au_minus3[0] / point.K_au_minus3[0] - 1.0)
                    ),
                    tolerance,
                )


if __name__ == "__main__":
    unittest.main()
