"""Fresh-fit provenance, eV conversion and rejection of cached source fits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qdmnp.observables.article_fit_cache import material_fit_cache
from qdmnp.observables.article_fit_seed import native_material_seed
from qdmnp.rational_fit import (
    AU_ENERGY_EV, HybridQDPlasmonModel, RationalLorentzFit,
    make_params_with_overrides,
)


class NativeMaterialSeedTests(unittest.TestCase):
    modes = np.array([[2., 1.2, .2], [4., 2.5, .4]])
    kwargs = {
        "c_nm": 11.875, "a_nm": 4.5, "eps_m": 2.25, "n_modes": 2,
        "fit_window_eV": (.8, 3.5),
        "fit_refinement": {"focus_center_eV": 2.042},
    }

    @classmethod
    def inaccurate_native_fit(cls, model):
        """Passive, slightly inaccurate source: valid guess, invalid final fit."""
        return RationalLorentzFit(
            alpha_inf=model.physical_alpha_infinity,
            strengths_au2=cls.modes[:, 0] / AU_ENERGY_EV**2,
            omega_modes_au=cls.modes[:, 1] / AU_ENERGY_EV,
            gamma_modes_au=cls.modes[:, 2] / AU_ENERGY_EV,
            energies_used_eV=np.array([.8, 3.5]),
            alpha_used=np.array([1.+.1j, 2.+.2j]),
            rms_alpha=.1, rms_inv_alpha=.1, cost=.01,
            normalized_rms_alpha=.024, normalized_rms_inv_alpha=.023,
            max_normalized_alpha_error=.050184,
        )

    def test_fresh_guess_units_identity_and_input_are_preserved(self):
        arguments = deepcopy(self.kwargs)
        arguments["fit_refinement"]["initial_modes_eV"] = {
            "long": [[8., 1.1, .3], [9., 2.4, .5]],
        }
        before = deepcopy(arguments)
        seen = []

        def native(model):
            seen.append(model)
            self.assertIsNone(model.fit_refinement.initial_modes_eV)
            self.assertIsNone(model.max_fit_normalized_rms)
            self.assertIsNone(model.max_fit_pointwise_relative_error)
            return self.inaccurate_native_fit(model)

        with patch.object(HybridQDPlasmonModel, "_fit_rational_alpha", native):
            modes, receipt = native_material_seed(**arguments)
            modes2, receipt2 = native_material_seed(**self.kwargs)
        self.assertEqual(arguments, before)
        self.assertEqual(modes, modes2)
        self.assertEqual(receipt, receipt2)
        np.testing.assert_allclose(modes, self.modes, rtol=1e-15, atol=0)
        self.assertEqual(receipt["geometry_nm"], {"c": 11.875, "a": 4.5, "b": 4.5})
        self.assertEqual(receipt["native_fit_identity"]["refinement"]["initial_modes_eV"], None)
        self.assertFalse(receipt["native_source_meets_default_accuracy"])
        self.assertTrue(receipt["destination_requires_independent_refit_and_acceptance"])
        self.assertFalse(receipt["acceptance_limits_relaxed"])
        self.assertEqual(receipt["role"], "initial_guess_only")
        encoded = np.r_[seen[0].fit.alpha_inf, np.array(modes).ravel()].astype("<f8")
        self.assertEqual(receipt["coefficient_sha256"], hashlib.sha256(encoded.tobytes()).hexdigest())
        self.assertEqual(json.loads(json.dumps(receipt))["orientation"], "long")

    def test_current_cache_reuses_source_without_accepting_it_for_calculations(self):
        seen = []

        def native(model):
            seen.append(model)
            return self.inaccurate_native_fit(model)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(list(directory.iterdir()), [])
            with patch.object(HybridQDPlasmonModel, "_fit_rational_alpha", native):
                with material_fit_cache(directory):
                    first = native_material_seed(**self.kwargs)
                    self.assertEqual(first, native_material_seed(**self.kwargs))
                    params = make_params_with_overrides(
                        c_nm=11.875, a_nm=4.5, r_nm=26.375,
                        eps_m=2.25, orientation="long",
                    )
                    with self.assertRaisesRegex(RuntimeError, "pointwise accuracy"):
                        HybridQDPlasmonModel(
                            params, orientation="long", n_modes=2,
                            fit_window_eV=(.8, 3.5),
                            fit_refinement={"focus_center_eV": 2.042},
                            max_fit_normalized_rms=.025,
                            max_fit_pointwise_relative_error=.05,
                            radiative_consistency_policy="ignore", verbose=False,
                        )
            self.assertEqual(len(seen), 1)
            self.assertEqual(len(list(directory.glob("*.npz"))), 1)
            with patch.object(HybridQDPlasmonModel, "_fit_rational_alpha",
                              side_effect=AssertionError("must reuse current run cache")):
                with material_fit_cache(directory):
                    self.assertEqual(first, native_material_seed(**self.kwargs))

    def test_invalid_requests_do_not_invoke_fitter(self):
        for extra in ({"n_modes": 1.5}, {"n_modes": True}, {"n_modes": 0},
                      {"orientation": "invalid"}, {"fit_refinement": "bad"}):
            with self.subTest(extra=extra):
                with patch.object(HybridQDPlasmonModel, "_fit_rational_alpha",
                                  side_effect=AssertionError("invalid input reached fitter")):
                    with self.assertRaises(ValueError):
                        native_material_seed(**(self.kwargs | extra))


if __name__ == "__main__":
    unittest.main()
