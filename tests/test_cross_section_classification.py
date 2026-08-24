"""Dipole cross-section classification in the code's alpha_eff convention.

The frequency-domain functions receive alpha_eff = p/(eps_m E_inc).  In this
convention the homogeneous-medium dipole formulas are

    sigma_ext = k/eps0 Im(alpha_eff)
    sigma_sca = k**4/(6*pi*eps0**2) |alpha_eff|**2
    sigma_abs = sigma_ext - sigma_sca.
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
    eV_to_au,
)


class CrossSectionClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.alpha_au = np.array([100.0 + 10.0j, 120.0 + 20.0j])
        self.omega_au = np.asarray(eV_to_au(np.array([1.8, 2.2])))
        self.eps_m = 2.25

    def _required_function(self, name: str):
        fn = getattr(spectrum, name, None)
        self.assertIsNotNone(fn, f"qd_mnp_linear_spectrum.{name} is required")
        return fn

    def test_extinction_matches_optical_theorem_formula(self) -> None:
        extinction = self._required_function("extinction_cross_section_cm2")
        alpha_si = self.alpha_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)
        omega_si = self.omega_au / AU_TIME_S
        k_si = np.sqrt(self.eps_m) * omega_si / C_SI
        expected = (k_si / epsilon_0) * alpha_si.imag * 1.0e4

        np.testing.assert_allclose(
            extinction(self.alpha_au, self.omega_au, self.eps_m),
            expected,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_legacy_cross_section_name_remains_extinction_alias(self) -> None:
        extinction = self._required_function("extinction_cross_section_cm2")
        np.testing.assert_allclose(
            spectrum.cross_section_cm2(self.alpha_au, self.omega_au, self.eps_m),
            extinction(self.alpha_au, self.omega_au, self.eps_m),
            rtol=0.0,
            atol=0.0,
        )

    def test_scattering_matches_dipole_radiation_formula(self) -> None:
        scattering = self._required_function("scattering_cross_section_cm2")
        alpha_si = self.alpha_au * (AU_DIPOLE_C_M / AU_FIELD_V_M)
        omega_si = self.omega_au / AU_TIME_S
        k_si = np.sqrt(self.eps_m) * omega_si / C_SI
        expected = (
            k_si**4
            / (6.0 * np.pi * epsilon_0**2)
            * np.abs(alpha_si) ** 2
            * 1.0e4
        )

        np.testing.assert_allclose(
            scattering(self.alpha_au, self.omega_au, self.eps_m),
            expected,
            rtol=2.0e-15,
            atol=0.0,
        )

    def test_absorption_is_extinction_minus_scattering(self) -> None:
        extinction = self._required_function("extinction_cross_section_cm2")
        scattering = self._required_function("scattering_cross_section_cm2")
        absorption = self._required_function("absorption_cross_section_cm2")

        ext = extinction(self.alpha_au, self.omega_au, self.eps_m)
        sca = scattering(self.alpha_au, self.omega_au, self.eps_m)
        abs_ = absorption(self.alpha_au, self.omega_au, self.eps_m)
        np.testing.assert_allclose(abs_, ext - sca, rtol=2.0e-15, atol=0.0)
        self.assertTrue(np.all(ext > 0.0))
        self.assertTrue(np.all(sca > 0.0))
        self.assertTrue(np.all(abs_ > 0.0))

    def test_nonpositive_medium_permittivity_is_rejected(self) -> None:
        for name in (
            "extinction_cross_section_cm2",
            "scattering_cross_section_cm2",
            "absorption_cross_section_cm2",
        ):
            fn = self._required_function(name)
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    fn(self.alpha_au, self.omega_au, 0.0)


if __name__ == "__main__":
    unittest.main()
