"""Contracts for immutable material data and the canonical Gamma2 API."""

import unittest

import numpy as np

from qd_mnp_params import make_params_with_overrides as make_shared_params
from qd_mnp_rational_fit import (
    HybridSolveResult,
    MaterialDispersion,
    au_to_eV,
    make_params_with_overrides as make_core_params,
)


class MaterialDispersionStorageTests(unittest.TestCase):
    def test_list_inputs_are_stored_as_immutable_contiguous_arrays(self) -> None:
        material = MaterialDispersion(
            energy_eV=[1.0, 2.0],
            n=[1.5, 1.6],
            k=[0.1, 0.2],
        )

        for values in (material.energy_eV, material.n, material.k):
            self.assertIsInstance(values, np.ndarray)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
        np.testing.assert_allclose(material.epsilon, (np.array([1.5, 1.6]) + 1j * np.array([0.1, 0.2])) ** 2)

    def test_external_mutation_cannot_change_stored_material(self) -> None:
        energy = np.array([1.0, 2.0])
        n = np.array([1.5, 1.6])
        k = np.array([0.1, 0.2])
        material = MaterialDispersion(energy, n, k)

        energy[0] = 9.0
        n[0] = 9.0
        k[0] = 9.0
        self.assertEqual(material.energy_eV[0], 1.0)
        self.assertEqual(material.n[0], 1.5)
        self.assertEqual(material.k[0], 0.1)
        with self.assertRaises(ValueError):
            material.n[0] = 2.0

    def test_linear_interpolation_is_explicit_and_does_not_extrapolate(self) -> None:
        material = MaterialDispersion(
            energy_eV=[1.0, 2.0, 4.0],
            n=[1.0, 2.0, 4.0],
            k=[0.2, 0.4, 0.8],
        )
        n_mid, k_mid = material.optical_constants_at(np.asarray([1.5, 3.0]))
        np.testing.assert_allclose(n_mid, [1.5, 3.0])
        np.testing.assert_allclose(k_mid, [0.3, 0.6])
        np.testing.assert_allclose(
            material.epsilon_at(material.energy_eV),
            material.epsilon,
        )
        with self.assertRaisesRegex(ValueError, "outside the tabulated"):
            material.epsilon_at(np.asarray([0.99, 2.0]))


class Gamma2OverrideApiTests(unittest.TestCase):
    @staticmethod
    def _gamma2_mev(params) -> float:
        return float(au_to_eV(params.Gamma_au) * 1000.0)

    def test_canonical_name_is_supported_by_both_factories(self) -> None:
        for factory in (make_core_params, make_shared_params):
            with self.subTest(factory=factory.__module__):
                self.assertAlmostEqual(self._gamma2_mev(factory(gamma2_coherence_mev=1.75)), 1.75)

    def test_equal_canonical_and_legacy_values_are_accepted(self) -> None:
        for factory in (make_core_params, make_shared_params):
            with self.subTest(factory=factory.__module__):
                params = factory(gamma2_coherence_mev=1.75, gamma_dephasing_mev=1.75)
                self.assertAlmostEqual(self._gamma2_mev(params), 1.75)

    def test_conflicting_canonical_and_legacy_values_are_rejected(self) -> None:
        for factory in (make_core_params, make_shared_params):
            with self.subTest(factory=factory.__module__):
                with self.assertRaisesRegex(ValueError, "gamma2_coherence_mev|agree"):
                    factory(gamma2_coherence_mev=1.75, gamma_dephasing_mev=2.0)


class CompatibilityAliasWarningTests(unittest.TestCase):
    def test_mislabeled_result_properties_warn(self) -> None:
        result = object.__new__(HybridSolveResult)
        object.__setattr__(result, "sigma_energy_transfer_cm2", 2.5)
        object.__setattr__(result, "work_from_incident_field_j", 3.5)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(result.sigma_abs_cm2, 2.5)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(result.absorbed_energy_j, 3.5)


if __name__ == "__main__":
    unittest.main()
