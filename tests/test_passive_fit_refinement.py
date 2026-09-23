"""Independent checks for opt-in passive fitting, units, and cache rejection."""
import argparse
from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from qdmnp.passive_fit import PassiveFitRefinement, lorentz_values_jacobian, positive_lorentz_candidate
from qdmnp.rational_fit import HybridQDPlasmonModel, make_default_params
from qdmnp.observables.article_fit_cache import material_fit_cache, fit_key
from qdmnp.observables.fit_options import parse_fit_refinement


class PassiveRefinementTests(unittest.TestCase):
    def test_complex_jacobian_matches_central_differences(self):
        energy = np.linspace(.8,3.5,123)
        u = np.r_[[2.,4.],np.log([1.2,2.5]),np.log([.2,.4])]
        _,jac = lorentz_values_jacobian(energy,u,-.2)
        for k in range(u.size):
            step=1e-6*max(1,abs(u[k])); offset=np.zeros_like(u);offset[k]=step
            numeric=(lorentz_values_jacobian(energy,u+offset,-.2)[0]-lorentz_values_jacobian(energy,u-offset,-.2)[0])/(2*step)
            np.testing.assert_allclose(jac[:,k],numeric,rtol=2e-7,atol=1e-8)

    def test_recovers_synthetic_passive_response_on_independent_grid(self):
        energy=np.linspace(.8,3.5,301)
        u=np.r_[[2.,4.],np.log([1.2,2.5]),np.log([.2,.4])]
        target,_=lorentz_values_jacobian(energy,u,-.2)
        fitted=positive_lorentz_candidate(energy,target,-.2,2,omega_bounds=(.28,26.4),
            gamma_bounds=(.008,7.),strength_max=1e5,nrms_limit=.001,pointwise_limit=.003)
        audit=np.linspace(.80031,3.4997,1507)
        actual,_=lorentz_values_jacobian(audit,fitted,-.2)
        expected,_=lorentz_values_jacobian(audit,u,-.2)
        self.assertLess(np.max(abs((actual-expected)/expected)),1e-5)
        self.assertTrue(np.all(fitted[:2]>=0))
        # Positive harmonic dissipation outside the fitted band does not imply
        # that extrapolated gold optical constants are experimentally validated.
        broad,_=lorentz_values_jacobian(np.geomspace(.001,100,1001),fitted,-.2)
        self.assertGreaterEqual(broad.imag.min(),0)

    def test_refinement_options_reject_invalid_units_ranges_and_unknown_fields(self):
        for text in ('[]','{"focus_center_eV":-2}', '{"focus_center_eV":2,"focus_half_width_eV":0}',
                     '{"focus_center_eV":2,"focus_relative_error":1}',
                     '{"focus_center_eV":2,"unknown":1}', '{"focus_center_eV":true}'):
            with self.subTest(text=text),self.assertRaises(argparse.ArgumentTypeError):parse_fit_refinement(text)
        with self.assertRaisesRegex(ValueError,'inside fit_window'):
            HybridQDPlasmonModel(make_default_params(),fit_refinement={'focus_center_eV':3.1},verbose=False)

    def test_seeded_minimax_recovers_passive_response_on_independent_grid(self):
        energy = np.linspace(.8, 3.5, 211)
        exact = np.r_[[2., 4.], np.log([1.2, 2.5]), np.log([.2, .4])]
        target, _ = lorentz_values_jacobian(energy, exact, -.2)
        fitted = positive_lorentz_candidate(
            energy, target, -.2, 2, omega_bounds=(.28, 26.4),
            gamma_bounds=(.008, 7.), strength_max=1e5,
            nrms_limit=.001, pointwise_limit=.003,
            initial_modes_eV=((1.8, 1.17, .24), (4.2, 2.56, .36)),
        )
        audit = np.linspace(.8013, 3.4991, 997)
        actual, _ = lorentz_values_jacobian(audit, fitted, -.2)
        expected, _ = lorentz_values_jacobian(audit, exact, -.2)
        self.assertLess(np.max(abs((actual-expected)/expected)), 1e-5)
        self.assertTrue(np.all(fitted[:2] >= 0))
        broad, _ = lorentz_values_jacobian(np.geomspace(.001, 100, 1001), fitted, -.2)
        self.assertGreaterEqual(broad.imag.min(), 0)

    def test_initial_guess_validation_and_cache_identity(self):
        for seed in ({}, {'bad': [[1, 2, .1]]}, {'long': [[-1, 2, .1]]},
                     {'long': [[1, 0, .1]]}, {'long': [[1, 2, float('nan')]]},
                     {'long': [[1, 2]]}):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                PassiveFitRefinement(2.042, initial_modes_eV=seed)
        model = HybridQDPlasmonModel(make_default_params(), n_modes=1, verbose=False,
            max_fit_normalized_rms=None, max_fit_pointwise_relative_error=None,
            radiative_consistency_policy='ignore')
        model.fit_refinement = PassiveFitRefinement(2.042)
        original_key = fit_key(model)
        model.fit_refinement = PassiveFitRefinement(2.042, initial_modes_eV={'long': [[1., 2., .1]]})
        self.assertNotEqual(original_key, fit_key(model))
        energy = np.linspace(.8, 3.5, 31)
        with self.assertRaisesRegex(ValueError, 'exactly n_modes'):
            positive_lorentz_candidate(energy, np.ones(31, complex), -.2, 2,
                omega_bounds=(.28, 26.4), gamma_bounds=(.008, 7.), strength_max=1e5,
                initial_modes_eV=[[1., 2., .1]])
        with self.assertRaisesRegex(ValueError, 'inside the declared bounds'):
            positive_lorentz_candidate(energy, np.ones(31, complex), -.2, 1,
                omega_bounds=(.28, 26.4), gamma_bounds=(.008, 7.), strength_max=1e5,
                initial_modes_eV=[[1., 200., .1]])

    def test_cache_preserves_rejected_fit_without_accepting_or_repeating_it(self):
        seen=[];original=HybridQDPlasmonModel._fit_rational_alpha
        def recording(model):
            seen.append(model)
            self.assertIsNone(model.max_fit_normalized_rms)
            return original(model)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(HybridQDPlasmonModel,'_fit_rational_alpha',recording):
                with material_fit_cache(Path(directory)):
                    for _ in range(2):
                        with self.assertRaisesRegex(RuntimeError,'Cached native material fit misses'):
                            HybridQDPlasmonModel(make_default_params(),n_modes=1,verbose=False,radiative_consistency_policy='ignore')
            self.assertEqual(len(seen),1)
            self.assertEqual(seen[0].max_fit_normalized_rms,.025)
            self.assertEqual(len(list(Path(directory).glob('*.npz'))),1)
            with patch.object(HybridQDPlasmonModel,'_fit_rational_alpha',side_effect=AssertionError('must use disk')):
                with material_fit_cache(Path(directory)):
                    with self.assertRaisesRegex(RuntimeError,'Cached native material fit misses'):
                        HybridQDPlasmonModel(make_default_params(),n_modes=1,verbose=False,radiative_consistency_policy='ignore')

    def test_cache_identity_includes_numerical_refinement(self):
        model=HybridQDPlasmonModel(make_default_params(),n_modes=1,verbose=False,
            max_fit_normalized_rms=None,max_fit_pointwise_relative_error=None,radiative_consistency_policy='ignore')
        original_key=fit_key(model)
        model.fit_refinement=PassiveFitRefinement(2.042)
        self.assertNotEqual(original_key,fit_key(model))
        self.assertEqual(fit_key(model)['refinement'],asdict(model.fit_refinement))

if __name__=='__main__':unittest.main()
