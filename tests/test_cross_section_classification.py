"""Semantics of optical estimates in the strict-quasistatic model.

The native model supplies an undressed electrostatic polarizability
``alpha_eff = p/(eps_m E_inc)``.  Therefore the three returned quantities are

    quasistatic work-loss estimate = k/eps0 Im(alpha_eff),
    Rayleigh scattering estimate  = k**4/(6*pi*eps0**2) |alpha_eff|**2,
    optical-theorem residual       = work loss - scattering.

The last quantity is deliberately *not* called material absorption: without a
radiation-reaction dressing of ``alpha_eff`` these terms are not an exact
optical-theorem energy partition.  Historical ``extinction``/``absorption``
names are tested below only as schema-compatibility aliases.
"""

import unittest

import numpy as np
from scipy.constants import c as C_SI
from scipy.constants import epsilon_0

import qd_mnp_linear_spectrum as spectrum
from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M,
    AU_FIELD_V_M,
    AU_TIME_S,
    dipole_cross_sections_cm2,
    eV_to_au,
    quasistatic_dipole_cross_section_estimates_cm2,
)


class CrossSectionClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.alpha_au = np.array([100.0 + 10.0j, 120.0 + 20.0j])
        self.omega_au = np.asarray(eV_to_au(np.array([1.8, 2.2])))
        self.eps_m = 2.25

    def _expected_terms(self) -> tuple[np.ndarray, np.ndarray]:
        alpha_si = self.alpha_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)
        omega_si = self.omega_au / AU_TIME_S
        k_si = np.sqrt(self.eps_m) * omega_si / C_SI
        work_loss_cm2 = (k_si / epsilon_0) * alpha_si.imag * 1.0e4
        scattering_cm2 = (
            k_si**4
            / (6.0 * np.pi * epsilon_0**2)
            * np.abs(alpha_si) ** 2
            * 1.0e4
        )
        return work_loss_cm2, scattering_cm2

    def test_quasistatic_work_loss_matches_formal_k_im_alpha(self) -> None:
        sections = quasistatic_dipole_cross_section_estimates_cm2(
            self.alpha_au,
            self.omega_au,
            self.eps_m,
        )
        expected_work_loss, _ = self._expected_terms()

        np.testing.assert_allclose(
            sections.quasistatic_work_loss_cm2,
            expected_work_loss,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_rayleigh_estimate_matches_coherent_dipole_radiation_formula(self) -> None:
        sections = quasistatic_dipole_cross_section_estimates_cm2(
            self.alpha_au,
            self.omega_au,
            self.eps_m,
        )
        _, expected_scattering = self._expected_terms()

        np.testing.assert_allclose(
            sections.rayleigh_scattering_estimate_cm2,
            expected_scattering,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_optical_theorem_residual_is_difference_not_material_absorption(self) -> None:
        sections = quasistatic_dipole_cross_section_estimates_cm2(
            self.alpha_au,
            self.omega_au,
            self.eps_m,
        )
        np.testing.assert_allclose(
            sections.optical_theorem_residual_cm2,
            sections.quasistatic_work_loss_cm2
            - sections.rayleigh_scattering_estimate_cm2,
            rtol=2.0e-15,
            atol=0.0,
        )

        # An undressed electrostatic alpha need not obey the radiative optical
        # theorem.  A negative residual is consequently a diagnostic result,
        # not a negative physical material-absorption cross section.
        deliberately_undressed = quasistatic_dipole_cross_section_estimates_cm2(
            1.0e8 + 1.0e-12j,
            float(self.omega_au[0]),
            self.eps_m,
        )
        self.assertLess(
            float(deliberately_undressed.optical_theorem_residual_cm2),
            0.0,
        )

    def test_historical_fields_are_schema_compatibility_aliases(self) -> None:
        sections = quasistatic_dipole_cross_section_estimates_cm2(
            self.alpha_au,
            self.omega_au,
            self.eps_m,
        )
        np.testing.assert_array_equal(
            sections.extinction_cm2,
            sections.quasistatic_work_loss_cm2,
        )
        np.testing.assert_array_equal(
            sections.scattering_cm2,
            sections.rayleigh_scattering_estimate_cm2,
        )
        np.testing.assert_array_equal(
            sections.absorption_cm2,
            sections.optical_theorem_residual_cm2,
        )

        legacy_sections = dipole_cross_sections_cm2(
            self.alpha_au,
            self.omega_au,
            self.eps_m,
        )
        np.testing.assert_array_equal(
            legacy_sections.extinction_cm2,
            sections.quasistatic_work_loss_cm2,
        )
        np.testing.assert_array_equal(
            legacy_sections.absorption_cm2,
            sections.optical_theorem_residual_cm2,
        )

    def test_legacy_linear_spectrum_helpers_preserve_old_schema(self) -> None:
        canonical = quasistatic_dipole_cross_section_estimates_cm2(
            self.alpha_au,
            self.omega_au,
            self.eps_m,
        )
        np.testing.assert_allclose(
            spectrum.quasistatic_work_loss_cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            ),
            canonical.quasistatic_work_loss_cm2,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            spectrum.rayleigh_scattering_estimate_cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            ),
            canonical.rayleigh_scattering_estimate_cm2,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            spectrum.optical_theorem_residual_cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            ),
            canonical.optical_theorem_residual_cm2,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            spectrum.extinction_cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            ),
            canonical.quasistatic_work_loss_cm2,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            spectrum.scattering_cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            ),
            canonical.rayleigh_scattering_estimate_cm2,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            spectrum.absorption_cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            ),
            canonical.optical_theorem_residual_cm2,
            rtol=0.0,
            atol=0.0,
        )
        with self.assertWarns(DeprecationWarning):
            ambiguous_legacy = spectrum.cross_section_cm2(
                self.alpha_au, self.omega_au, self.eps_m
            )
        np.testing.assert_allclose(
            ambiguous_legacy,
            canonical.quasistatic_work_loss_cm2,
            rtol=0.0,
            atol=0.0,
        )

    def test_nonpositive_medium_permittivity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quasistatic_dipole_cross_section_estimates_cm2(
                self.alpha_au,
                self.omega_au,
                0.0,
            )

    def test_passive_work_loss_guard_rejects_a_physical_negative_branch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-negative QS work loss"):
            spectrum.enforce_passive_work_loss_cm2(
                np.asarray([1.0, -0.1]),
                label="test response",
            )

    def test_passive_work_loss_guard_clips_roundoff_with_warning(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "roundoff"):
            checked = spectrum.enforce_passive_work_loss_cm2(
                np.asarray([1.0, -1.0e-12]),
                label="test response",
            )
        np.testing.assert_array_equal(checked, np.asarray([1.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
