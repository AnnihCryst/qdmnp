"""Independent sphere contracts for the Cartesian surface-charge BEM."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from qd_mnp_bem_validation import (
    build_affine_icosphere,
    mesh_sha256,
    run_nested_bem_convergence,
    solve_spheroid_bem,
)
from qd_mnp_spheroid_equatorial import (
    EquatorialSpheroidGeometry,
    EquatorialSpheroidGreenInteraction,
)
from qd_mnp_spheroid_green import ProlateSpheroidGeometry, SpheroidGreenInteraction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "spheroid_bem_v1.json"


def _decode_complex(value: list[float]) -> complex:
    return complex(float(value[0]), float(value[1]))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sphere_exact_longitudinal(
    *,
    radius: float,
    distance: float,
    eps_m: float,
    epsilon_particle: complex,
    n_max: int = 400,
) -> tuple[complex, complex, complex]:
    """Elementary dielectric-sphere A, B and axial reaction-field series."""

    contrast = epsilon_particle - eps_m
    A = eps_m * radius**3 * contrast / (epsilon_particle + 2.0 * eps_m)
    B = 2.0 * A / (eps_m * distance**3)
    degree = np.arange(1, n_max + 1, dtype=float)
    denominator = degree * epsilon_particle + (degree + 1.0) * eps_m
    # Keep the geometric factor as (a/R)^(2n+1)/R^3.  Computing the
    # numerator and denominator powers separately overflows long before their
    # rapidly decaying ratio becomes numerically negligible.
    terms = (
        degree
        * (degree + 1.0) ** 2
        * (radius / distance) ** (2.0 * degree + 1.0)
        * contrast
        / (eps_m * distance**3 * denominator)
    )
    return complex(A), complex(B), complex(np.sum(terms))


class AffineIcosphereTests(unittest.TestCase):
    def test_mesh_is_deterministic_nested_and_outward(self) -> None:
        first = build_affine_icosphere(1.0, 1.7, subdivision_level=2)
        second = build_affine_icosphere(1.0, 1.7, subdivision_level=2)
        self.assertEqual(first.vertex_count, 162)
        self.assertEqual(first.panel_count, 320)
        np.testing.assert_array_equal(first.vertices_au, second.vertices_au)
        np.testing.assert_array_equal(first.faces, second.faces)
        self.assertFalse(first.vertices_au.flags.writeable)
        ellipsoid_gradient = first.centroids_au / np.asarray([1.0, 1.0, 1.7**2])
        self.assertTrue(
            np.all(np.einsum("ij,ij->i", first.outward_normals, ellipsoid_gradient) > 0.0)
        )
        coarser = build_affine_icosphere(1.0, 1.7, subdivision_level=1)
        self.assertEqual(first.panel_count, 4 * coarser.panel_count)
        self.assertLess(first.max_edge_au, coarser.max_edge_au)

    def test_invalid_qd_and_dense_memory_guard_are_explicit(self) -> None:
        common = dict(
            a_au=1.0,
            c_au=1.5,
            polarization=(0.0, 0.0, 1.0),
            eps_m=1.2,
            epsilon_particle=-5.0 + 0.5j,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            solve_spheroid_bem(qd_position_au=(0.0, 0.0, 1.4), **common)
        with self.assertRaisesRegex(ValueError, "Dense BEM mesh"):
            solve_spheroid_bem(
                qd_position_au=(0.0, 0.0, 3.0),
                subdivision_level=2,
                max_dense_panels=100,
                **common,
            )


class SphereBEMValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.radius = 1.0
        cls.distance = 3.0
        # Lossy metallic contrast, but not so close to the sphere dipole pole
        # that a deliberately first-order live-CI mesh becomes misleading.
        cls.eps_m = 1.0
        cls.epsilon_particle = -8.0 + 1.0j
        cls.exact = _sphere_exact_longitudinal(
            radius=cls.radius,
            distance=cls.distance,
            eps_m=cls.eps_m,
            epsilon_particle=cls.epsilon_particle,
        )
        cls.convergence = run_nested_bem_convergence(
            a_au=cls.radius,
            c_au=cls.radius,
            qd_position_au=(0.0, 0.0, cls.distance),
            polarization=(0.0, 0.0, 2.0),
            eps_m=cls.eps_m,
            epsilon_particle=cls.epsilon_particle,
            subdivision_levels=(1, 2, 3),
        )

    def test_uniform_sphere_polarizability_converges_to_elementary_value(self) -> None:
        exact_A = self.exact[0]
        errors = [
            abs(response.observables.A_au3 - exact_A) / abs(exact_A)
            for response in self.convergence.responses
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], 0.08)
        extrapolated_error = (
            abs(self.convergence.extrapolated.A_au3 - exact_A) / abs(exact_A)
        )
        self.assertLess(extrapolated_error, errors[-1])

    def test_two_independent_B_extractions_converge_and_obey_reciprocity(self) -> None:
        exact_B = self.exact[1]
        for field_name in ("B_field", "B_dipole"):
            errors = [
                abs(getattr(response.observables, field_name) - exact_B) / abs(exact_B)
                for response in self.convergence.responses
            ]
            self.assertGreater(errors[0], errors[-1])
            self.assertLess(errors[-1], 0.10)
        finest = self.convergence.responses[-1]
        self.assertLess(finest.diagnostics.reciprocity_relative_error, 0.025)

    def test_full_reaction_field_converges_to_sphere_multipole_series(self) -> None:
        exact_K = self.exact[2]
        errors = [
            abs(response.observables.K_au_minus3 - exact_K) / abs(exact_K)
            for response in self.convergence.responses
        ]
        self.assertGreater(errors[0], errors[-1])
        self.assertLess(errors[-1], 0.15)
        self.assertLess(
            abs(self.convergence.extrapolated.K_au_minus3 - exact_K) / abs(exact_K),
            errors[-1],
        )

    def test_linear_solve_charge_and_resolution_diagnostics_are_exposed(self) -> None:
        finest = self.convergence.responses[-1].diagnostics
        self.assertEqual(finest.panel_count, 1280)
        self.assertLess(finest.relative_residual_uniform, 2.0e-12)
        self.assertLess(finest.relative_residual_point_dipole, 2.0e-12)
        self.assertLess(finest.relative_net_charge_uniform, 2.0e-12)
        self.assertLess(finest.relative_net_charge_point_dipole, 2.0e-3)
        self.assertGreater(finest.minimum_qd_surface_distance_au, 1.9)
        self.assertGreater(finest.qd_distance_over_max_edge, 5.0)
        self.assertEqual(finest.quadrature_rule, "constant_panel_centroid")

    def test_extrapolation_declares_model_and_conservative_uncertainty(self) -> None:
        convergence = self.convergence
        self.assertEqual(convergence.correction_orders, (1.0, 2.0))
        self.assertEqual(convergence.extrapolation_levels, (1, 2, 3))
        self.assertGreater(convergence.extrapolation_h_scale_au, 0.0)
        pairs = (
            (
                convergence.extrapolated.A_au3,
                convergence.lower_order_extrapolated.A_au3,
                convergence.estimated_absolute_uncertainty.A,
            ),
            (
                convergence.extrapolated.B_field,
                convergence.lower_order_extrapolated.B_field,
                convergence.estimated_absolute_uncertainty.B_field,
            ),
            (
                convergence.extrapolated.B_dipole,
                convergence.lower_order_extrapolated.B_dipole,
                convergence.estimated_absolute_uncertainty.B_dipole,
            ),
            (
                convergence.extrapolated.K_au_minus3,
                convergence.lower_order_extrapolated.K_au_minus3,
                convergence.estimated_absolute_uncertainty.K,
            ),
        )
        for extrapolated, lower_order, uncertainty in pairs:
            self.assertGreaterEqual(
                uncertainty,
                abs(extrapolated - lower_order) * (1.0 - 2.0e-14),
            )

    def test_invalid_extrapolation_policy_is_rejected_before_solving(self) -> None:
        common = dict(
            a_au=1.0,
            c_au=1.0,
            qd_position_au=(0.0, 0.0, 3.0),
            polarization=(0.0, 0.0, 1.0),
            eps_m=1.0,
            epsilon_particle=-8.0 + 1.0j,
            subdivision_levels=(1, 2, 3),
        )
        with self.assertRaisesRegex(ValueError, "correction_terms"):
            run_nested_bem_convergence(correction_terms=0, **common)
        with self.assertRaisesRegex(ValueError, "extrapolation_level_count"):
            run_nested_bem_convergence(
                correction_terms=2,
                extrapolation_level_count=2,
                **common,
            )

    def test_zero_contrast_has_exactly_zero_induced_response(self) -> None:
        response = solve_spheroid_bem(
            a_au=1.0,
            c_au=1.4,
            qd_position_au=(2.5, 0.0, 0.0),
            polarization=(0.0, 0.0, 1.0),
            eps_m=1.7,
            epsilon_particle=1.7,
            subdivision_level=1,
        )
        self.assertTrue(response.diagnostics.zero_contrast)
        self.assertIsNone(response.diagnostics.bie_lambda)
        self.assertEqual(response.observables, type(response.observables)(0j, 0j, 0j, 0j))

    def test_nonzero_side_geometry_is_a_genuine_three_dimensional_solve(self) -> None:
        response = solve_spheroid_bem(
            a_au=1.0,
            c_au=1.7,
            qd_position_au=(2.5, 0.0, 0.0),
            # The dipole is along the long z axis while the QD lies on the
            # equatorial x axis: this cannot be represented by the native
            # on-axis long/trans scalar geometry.
            polarization=(0.0, 0.0, 1.0),
            eps_m=1.0,
            epsilon_particle=-8.0 + 1.0j,
            subdivision_level=2,
        )
        values = np.asarray(
            [
                response.observables.A_au3,
                response.observables.B_field,
                response.observables.B_dipole,
                response.observables.K_au_minus3,
            ]
        )
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(abs(response.observables.K_au_minus3), 0.0)
        self.assertLess(response.diagnostics.reciprocity_relative_error, 0.02)
        self.assertGreater(response.diagnostics.minimum_qd_surface_distance_au, 0.0)


class ImmutableBEMFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_provenance_is_complete_and_generator_is_independent(self) -> None:
        fixture = self.fixture
        self.assertEqual(fixture["schema"], "qdmnp.independent_spheroid_bem_fixture")
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(fixture["mesh_levels"], [1, 2, 3, 4])
        provenance = fixture["provenance"]
        solver_path = PROJECT_ROOT / provenance["solver_module"]
        generator_path = PROJECT_ROOT / provenance["generator"]
        self.assertEqual(provenance["solver_sha256"], _file_sha256(solver_path))
        self.assertEqual(provenance["generator_sha256"], _file_sha256(generator_path))

        for path in (solver_path, generator_path):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported_modules.append(node.module)
            self.assertFalse(
                any(module.startswith("qd_mnp_spheroid") for module in imported_modules),
                f"{path.name} must remain independent of the analytic kernels.",
            )

    def test_mesh_hashes_and_panel_counts_reproduce_without_solving(self) -> None:
        seen: set[tuple[float, float, int]] = set()
        for case in self.fixture["cases"]:
            for level_data in case["raw_levels"]:
                key = (
                    float(case["a_au"]),
                    float(case["c_au"]),
                    int(level_data["subdivision_level"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                mesh = build_affine_icosphere(*key[:2], subdivision_level=key[2])
                self.assertEqual(mesh.panel_count, level_data["panel_count"])
                self.assertEqual(mesh.vertex_count, level_data["vertex_count"])
                self.assertEqual(mesh_sha256(mesh), level_data["mesh_sha256"])

    def test_fixture_covers_all_approved_geometries(self) -> None:
        channels = {
            (
                case["regime"],
                case["placement"],
                case["orientation"],
                case["side_transverse_alignment"],
            )
            for case in self.fixture["cases"]
        }
        required = {
            ("ordinary", "axis", "long", None),
            ("ordinary", "axis", "trans", None),
            ("ordinary", "side", "long", None),
            ("ordinary", "side", "trans", "radial"),
            ("ordinary", "side", "trans", "tangential"),
            ("small_gap", "side", "trans", "radial"),
        }
        self.assertTrue(required.issubset(channels))
        self.assertTrue(
            any(case["a_au"] == case["c_au"] for case in self.fixture["cases"])
        )
        self.assertTrue(
            any(case["c_au"] > case["a_au"] for case in self.fixture["cases"])
        )

    def test_analytic_kernels_meet_fixed_bem_tolerances_and_uncertainties(self) -> None:
        fixture_to_quantity = {
            "A": "A_au3",
            "B_field": "B_field",
            "B_dipole": "B_dipole",
            "K": "K_au_minus3",
        }
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                position = np.asarray(case["qd_position_au"], dtype=float)
                R_au = float(np.linalg.norm(position))
                if case["placement"] == "axis":
                    kernel = SpheroidGreenInteraction(
                        ProlateSpheroidGeometry(
                            a_au=case["a_au"],
                            c_au=case["c_au"],
                            R_au=R_au,
                            eps_m=case["eps_m"],
                            orientation=case["orientation"],
                        ),
                        n_max=80,
                    )
                else:
                    kernel = EquatorialSpheroidGreenInteraction(
                        EquatorialSpheroidGeometry(
                            a_au=case["a_au"],
                            c_au=case["c_au"],
                            R_au=R_au,
                            eps_m=case["eps_m"],
                            orientation=case["orientation"],
                            side_transverse_alignment=(
                                case["side_transverse_alignment"]
                            ),
                        ),
                        n_max=80,
                    )
                response = kernel.response_from_epsilon(
                    _decode_complex(case["epsilon_particle"])
                )
                exact = {
                    "A": complex(np.asarray(response.A_au3).item()),
                    "B_field": complex(np.asarray(response.B).item()),
                    "B_dipole": complex(np.asarray(response.B).item()),
                    "K": complex(np.asarray(response.K_au_minus3).item()),
                }
                sigma_multiplier = case["acceptance_relative"][
                    "uncertainty_sigma_multiplier"
                ]
                for quantity, exact_value in exact.items():
                    fixture_value = _decode_complex(
                        case["extrapolated"][fixture_to_quantity[quantity]]
                    )
                    absolute_error = abs(fixture_value - exact_value)
                    relative_error = absolute_error / abs(exact_value)
                    self.assertLessEqual(
                        relative_error,
                        case["acceptance_relative"][quantity],
                        f"{case['id']} {quantity} exceeds the fixed relative gate.",
                    )
                    self.assertLessEqual(
                        absolute_error,
                        sigma_multiplier
                        * case["estimated_absolute_uncertainty"][quantity],
                        f"{case['id']} {quantity} lies outside three BEM uncertainties.",
                    )

    def test_fixture_algebra_diagnostics_are_resolved(self) -> None:
        for case in self.fixture["cases"]:
            finest = case["raw_levels"][-1]
            with self.subTest(case=case["id"]):
                self.assertLess(finest["relative_residual_uniform"], 2.0e-12)
                self.assertLess(finest["relative_residual_point_dipole"], 2.0e-12)
                self.assertLess(finest["relative_net_charge_uniform"], 2.0e-12)
                # Constant-panel collocation does not impose the zero-total-
                # charge constraint explicitly.  The worst axial prolate case
                # decreases from 20% to 1.27% over levels 1--4; 1.5% is a
                # declared discretization diagnostic, not an analytic gate.
                self.assertLess(finest["relative_net_charge_point_dipole"], 0.015)
                self.assertLess(finest["reciprocity_relative_error"], 0.006)
                self.assertGreater(finest["qd_distance_over_max_edge"], 4.0)


if __name__ == "__main__":
    unittest.main()
