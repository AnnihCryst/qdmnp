"""Physical-domain validation for relaxation rates and particle geometry."""

import unittest
from unittest.mock import patch

import numpy as np

from qd_mnp_rational_fit import (
    HybridQDPlasmonModel,
    HybridSystemParams,
    RationalLorentzFit,
    au_to_nm,
    homogeneous_radiative_decay_rate_au,
    make_default_params,
    nm_to_au,
)
from qd_mnp_params import make_params_with_overrides


def _dummy_fit() -> RationalLorentzFit:
    return RationalLorentzFit(
        alpha_inf=0.0,
        strengths_au2=np.array([1.0e-4]),
        omega_modes_au=np.array([0.08]),
        gamma_modes_au=np.array([0.01]),
        energies_used_eV=np.array([2.0]),
        alpha_used=np.array([1.0j]),
        rms_alpha=0.0,
        rms_inv_alpha=0.0,
        cost=0.0,
    )


def _construct_model_without_fitting(params: HybridSystemParams) -> HybridQDPlasmonModel:
    with patch.object(HybridQDPlasmonModel, "_fit_rational_alpha", return_value=_dummy_fit()):
        return HybridQDPlasmonModel(
            params,
            n_modes=1,
            radiative_consistency_policy="ignore",
            verbose=False,
        )


class ParameterValidationTests(unittest.TestCase):
    def test_default_fixture_has_one_nanometre_surface_gap(self) -> None:
        params = make_default_params()
        try:
            qd_radius_au = params.qd_radius_au
        except AttributeError as exc:
            self.fail(f"HybridSystemParams.qd_radius_au is missing: {exc}")

        gap_nm = float(au_to_nm(params.R_au - params.c_au - qd_radius_au))
        self.assertAlmostEqual(float(au_to_nm(params.R_au)), 18.0, places=12)
        self.assertAlmostEqual(float(au_to_nm(qd_radius_au)), 2.0, places=12)
        self.assertAlmostEqual(gap_nm, 1.0, places=12)
        self.assertEqual(params.qd_placement, "axis")
        self.assertIsNone(params.side_transverse_alignment)
        self.assertEqual(params.surface_gap_au, params.axial_surface_gap_au)

    def test_default_records_legacy_radiative_rate_inconsistency(self) -> None:
        params = make_default_params()
        expected = homogeneous_radiative_decay_rate_au(
            params.qd_external_dipole_au,
            params.omega0_au,
            params.eps_m,
        )
        self.assertEqual(params.homogeneous_radiative_decay_au, expected)
        self.assertLess(params.gamma_au, expected)
        self.assertFalse(
            params.radiative_rate_diagnostics.homogeneous_host_consistent
        )

    def test_strict_policy_rejects_gamma1_below_homogeneous_reference_rate(self) -> None:
        base = make_default_params()
        invalid = HybridSystemParams(
            c_au=base.c_au,
            a_au=base.a_au,
            R_au=base.R_au,
            G=base.G,
            eps_m=base.eps_m,
            d_au=base.d_au,
            omega0_au=base.omega0_au,
            gamma_au=0.5 * base.homogeneous_radiative_decay_au,
            Gamma_au=base.Gamma_au,
            material=base.material,
            qd_radius_au=base.qd_radius_au,
            eps_qd=base.eps_qd,
            qd_dipole_convention=base.qd_dipole_convention,
        )
        with patch.object(
            HybridQDPlasmonModel,
            "_fit_rational_alpha",
            return_value=_dummy_fit(),
        ):
            with self.assertRaisesRegex(ValueError, "gamma1/gamma_rad"):
                HybridQDPlasmonModel(
                    invalid,
                    n_modes=1,
                    radiative_consistency_policy="raise",
                    verbose=False,
                )

    def test_legacy_parameter_construction_defaults_to_point_qd(self) -> None:
        """Adding qd_radius_au must not break callers using the old signature."""
        base = make_default_params()
        params = HybridSystemParams(
            c_au=base.c_au,
            a_au=base.a_au,
            R_au=base.R_au,
            G=base.G,
            eps_m=base.eps_m,
            d_au=base.d_au,
            omega0_au=base.omega0_au,
            gamma_au=base.gamma_au,
            Gamma_au=base.Gamma_au,
            material=base.material,
        )
        try:
            qd_radius_au = params.qd_radius_au
        except AttributeError as exc:
            self.fail(f"Backward-compatible qd_radius_au default is missing: {exc}")
        self.assertEqual(qd_radius_au, 0.0)

    def test_legacy_positional_material_argument_is_preserved(self) -> None:
        base = make_default_params()
        params = HybridSystemParams(
            base.c_au,
            base.a_au,
            base.R_au,
            base.G,
            base.eps_m,
            base.d_au,
            base.omega0_au,
            base.gamma_au,
            base.Gamma_au,
            base.material,
        )
        self.assertIs(params.material, base.material)
        self.assertEqual(params.qd_radius_au, 0.0)
        self.assertEqual(params.qd_placement, "axis")
        self.assertIsNone(params.side_transverse_alignment)

    def test_model_rejects_nonpositive_surface_gap(self) -> None:
        base = make_default_params()
        try:
            overlapping = HybridSystemParams(
                c_au=base.c_au,
                a_au=base.a_au,
                R_au=base.c_au + float(nm_to_au(1.0)),
                G=base.G,
                eps_m=base.eps_m,
                d_au=base.d_au,
                omega0_au=base.omega0_au,
                gamma_au=base.gamma_au,
                Gamma_au=base.Gamma_au,
                material=base.material,
                qd_radius_au=float(nm_to_au(2.0)),
            )
        except TypeError as exc:
            self.fail(f"Finite-QD geometry API is missing: {exc}")

        with self.assertRaisesRegex(ValueError, "gap|R.*c|overlap"):
            _construct_model_without_fitting(overlapping)

    def test_model_accepts_strictly_positive_surface_gap(self) -> None:
        params = make_default_params()
        model = _construct_model_without_fitting(params)
        self.assertIs(model.params, params)

    def test_side_placement_uses_equatorial_radius_for_gap_and_overlap(self) -> None:
        params = make_params_with_overrides(
            r_nm=10.0,
            orientation="long",
            qd_placement="side",
        )

        self.assertLess(params.axial_surface_gap_au, 0.0)
        self.assertAlmostEqual(float(au_to_nm(params.surface_gap_au)), 1.0, places=12)
        self.assertEqual(params.G, -1.0)
        model = _construct_model_without_fitting(params)
        self.assertIs(model.params, params)

    def test_side_placement_rejects_overlap_against_equatorial_radius(self) -> None:
        params = make_params_with_overrides(
            r_nm=8.5,
            orientation="long",
            qd_placement="side",
        )
        with self.assertRaisesRegex(ValueError, "R > a.*qd_placement='side'"):
            _construct_model_without_fitting(params)

    def test_invalid_placement_alignment_combinations_are_rejected(self) -> None:
        invalid = (
            {"qd_placement": "invalid"},
            {
                "qd_placement": "axis",
                "side_transverse_alignment": "radial",
            },
            {
                "orientation": "long",
                "qd_placement": "side",
                "side_transverse_alignment": "tangential",
            },
            {
                "orientation": "trans",
                "qd_placement": "side",
            },
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_default_params(**kwargs)

    def test_qd_radius_override_reaches_canonical_parameters(self) -> None:
        params = make_params_with_overrides(qd_radius_nm=3.0, r_nm=19.5)
        self.assertAlmostEqual(float(au_to_nm(params.qd_radius_au)), 3.0, places=12)
        self.assertAlmostEqual(float(au_to_nm(params.axial_surface_gap_au)), 1.5, places=12)
        _construct_model_without_fitting(params)

    def test_model_rejects_gamma2_below_half_population_decay(self) -> None:
        base = make_default_params()
        kwargs = dict(
            c_au=base.c_au,
            a_au=base.a_au,
            R_au=base.R_au,
            G=base.G,
            eps_m=base.eps_m,
            d_au=base.d_au,
            omega0_au=base.omega0_au,
            gamma_au=2.0e-4,
            Gamma_au=np.nextafter(1.0e-4, 0.0),
            material=base.material,
        )
        if hasattr(base, "qd_radius_au"):
            kwargs["qd_radius_au"] = base.qd_radius_au
        invalid = HybridSystemParams(**kwargs)

        with self.assertRaisesRegex(ValueError, "Gamma|gamma|dephas|coherence"):
            _construct_model_without_fitting(invalid)

    def test_gamma2_equal_to_half_population_decay_is_allowed(self) -> None:
        base = make_default_params()
        kwargs = dict(
            c_au=base.c_au,
            a_au=base.a_au,
            R_au=base.R_au,
            G=base.G,
            eps_m=base.eps_m,
            d_au=base.d_au,
            omega0_au=base.omega0_au,
            gamma_au=2.0e-4,
            Gamma_au=1.0e-4,
            material=base.material,
        )
        if hasattr(base, "qd_radius_au"):
            kwargs["qd_radius_au"] = base.qd_radius_au
        boundary = HybridSystemParams(**kwargs)
        model = _construct_model_without_fitting(boundary)
        self.assertIs(model.params, boundary)


if __name__ == "__main__":
    unittest.main()
