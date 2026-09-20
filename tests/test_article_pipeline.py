"""Units, selection, restart integrity and native CLI integration of the runner."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qdmnp import pipeline as runner
from qdmnp.observables.article_inputs import load_inputs, physical_arguments, unit_audit, validate_inputs
from qdmnp.observables.article_fit_cache import material_fit_cache, fit_key
from qdmnp.rational_fit import HybridQDPlasmonModel, make_params_with_overrides


class ArticleInputTests(unittest.TestCase):
    def test_shah_single_particle_geometry_and_rate_units(self):
        config = load_inputs()
        values = unit_audit(config)
        args = physical_arguments(config)
        self.assertEqual(values["reference_tip_center_distance_nm"], 18)
        self.assertEqual(values["reference_tip_surface_gap_nm"], 1)
        self.assertAlmostEqual(args["gamma-population-mev"], .000268)
        self.assertAlmostEqual(args["gamma2-coherence-mev"], 1.270134)
        self.assertAlmostEqual(values["SI"]["T1_ns"], 2.45601476, places=7)
        self.assertAlmostEqual(values["SI"]["T2_fs"], 518.22245, places=4)
        self.assertAlmostEqual(values["SI"]["qd_dipole_C_m"], 4.6365396e-29, delta=1e-36)
        self.assertAlmostEqual(values["reference_field_V_m"], 4857396.425, places=2)
        self.assertAlmostEqual(values["reference_peak_intensity_W_cm2"], 4697186.3935, places=2)
        self.assertAlmostEqual(values["SI"]["pulse_intensity_spectral_fwhm_eV"], .09124755, places=7)
        self.assertLess(values["own_population_decay_fraction_at_read"], config["pulse"]["max_population_decay_fraction"])
        self.assertNotIn("coupling_g_mev", args)

    def test_reject_double_screening_and_invalid_gap_sensitivity(self):
        config = load_inputs()
        config["qd"]["dipole_convention"] = "bare_internal"
        with self.assertRaisesRegex(ValueError, "screening"):
            validate_inputs(config)
        config = load_inputs()
        config["validation"]["gap_offset_nm"] = 2
        with self.assertRaisesRegex(ValueError, "intersect"):
            validate_inputs(config)

    def test_smoke_does_not_change_physical_inputs(self):
        full, smoke = load_inputs(), load_inputs(smoke=True)
        self.assertEqual(physical_arguments(full),physical_arguments(smoke))
        self.assertEqual(smoke["pulse"]["intensity_fwhm_fs"],20)
        self.assertTrue(smoke["smoke"])


class PipelineSelectionTests(unittest.TestCase):
    def fixture(self):
        cfg = load_inputs()
        channels = cfg["geometry"]["channels"]
        gaps = np.array([1.,10.,50.])
        values = np.full((2,5,3),2e-5)
        values[1,0,0] = 1e-7 # censored bound, must never win
        values[1,1,2] = 2e-7 # outside declared kR selection range
        values[1,2,1] = 3e-6 # valid winner
        status = np.full(values.shape,"resolved",dtype="U32")
        status[1,0,0]="left_censored"
        return cfg, {"channel_id":np.array(channels),"gap_nm":gaps,"threshold_fluence_j_cm2":values,
                     "threshold_status":status,"absolute_threshold_discrepancy_dd_vs_fqs":np.full((5,3),.2)}

    def test_selection_rejects_bounds_and_large_kR(self):
        cfg,data=self.fixture()
        chosen=runner.select_scenarios(data,cfg)
        self.assertEqual(chosen["best_channel"],"side_long")
        self.assertEqual(chosen["best_gap_nm"],10)
        self.assertEqual(chosen["control_channel"],"side_trans_radial")
        self.assertTrue(chosen["resolved_threshold_selected"])
        self.assertFalse(chosen["far_point_satisfies_threshold_DD_tolerance"])

    def test_no_resolved_threshold_is_an_illustration_not_an_optimum(self):
        cfg,data=self.fixture()
        data["threshold_status"][:]='right_censored'
        chosen=runner.select_scenarios(data,cfg)
        self.assertTrue(chosen["selection_is_illustration_only"])
        self.assertFalse(chosen["resolved_threshold_selected"])


class PipelineRestartTests(unittest.TestCase):
    def test_resume_verifies_files_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=load_inputs(smoke=True)
            run=runner.ArticleRun(cfg,Path(directory))
            def fake_module(*args,**kwargs):
                target=Path(runner.sys.argv[-1])
                target.write_bytes(b'completed artifact')
            with patch.object(runner.runpy,"run_module",side_effect=fake_module) as call:
                output=run.step('test','fake.module',{'value':2})
                again=runner.ArticleRun(cfg,Path(directory))
                self.assertEqual(again.step('test','fake.module',{'value':2}),output)
                self.assertEqual(call.call_count,1)
                output.write_bytes(b'changed')
                with self.assertRaisesRegex(runner.StageError,'changed'):
                    again.step('test','fake.module',{'value':2})
            cfg["qd"]["effective_dipole_debye"]+=1
            with self.assertRaisesRegex(ValueError,'changed'):
                runner.ArticleRun(cfg,Path(directory))

    def test_threshold_master_accepts_scalar_text_status_and_routes_both_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            run=runner.ArticleRun(load_inputs(smoke=True),Path(directory))
            path=Path(directory)/'threshold.npz'
            np.savez(path,metadata_json=np.asarray('{}'),
                     isolated_threshold_status=np.asarray('resolved'),
                     threshold_status=np.full((2,5,2),'resolved'))
            run.threshold_calculation=lambda *a,**k:path
            run.audit_fit_identity=lambda *a,**k:None
            with patch.object(run,'plot') as plot, patch.object(run,'step',return_value=path):
                run.thresholds()
            self.assertEqual(run.state['threshold_master'],str(path))
            self.assertEqual([call.args[0] for call in plot.call_args_list],['fig03a','fig03b'])


class FitCacheTests(unittest.TestCase):
    def test_cache_preserves_fit_and_recomputes_geometric_coupling(self):
        with tempfile.TemporaryDirectory() as directory:
            original=HybridQDPlasmonModel._fit_rational_alpha
            kwargs=dict(n_modes=1,verbose=False,max_fit_normalized_rms=None,max_fit_pointwise_relative_error=None,radiative_consistency_policy='ignore')
            with material_fit_cache(Path(directory)):
                near=HybridQDPlasmonModel(make_params_with_overrides(r_nm=18),**kwargs)
                far=HybridQDPlasmonModel(make_params_with_overrides(r_nm=30),**kwargs)
                self.assertEqual(fit_key(near),fit_key(far))
                self.assertNotEqual(near.J,far.J)
                np.testing.assert_array_equal(near.fit.strengths_au2,far.fit.strengths_au2)
                far.fit.strengths_au2[0]=999
                self.assertNotEqual(near.fit.strengths_au2[0],999)
            self.assertIs(HybridQDPlasmonModel._fit_rational_alpha,original)
            with patch.object(HybridQDPlasmonModel,"_fit_rational_alpha",side_effect=AssertionError('must use disk cache')):
                with material_fit_cache(Path(directory)):
                    loaded=HybridQDPlasmonModel(make_params_with_overrides(r_nm=30),**kwargs)
                    np.testing.assert_array_equal(near.fit.strengths_au2,loaded.fit.strengths_au2)


class PipelineCLITests(unittest.TestCase):
    def test_generated_spectral_and_temporal_commands_match_real_parsers(self):
        cfg=load_inputs(smoke=True)
        with tempfile.TemporaryDirectory() as directory:
            run=runner.ArticleRun(cfg,Path(directory))
            calls=[]
            def capture(label,module,arguments=(),**kwargs):
                calls.append((module, runner.flags(arguments) if isinstance(arguments,dict) else list(arguments)))
                return Path(directory)/'unused.npz'
            run.step=capture
            run.material_spectrum('spectrum',1)
            run.threshold_calculation('threshold')
            run.state['selection']={'best_channel':'axis_long','control_channel':'axis_trans','best_gap_nm':1.,'far_gap_nm':10.}
            run.state['selected_fluences']={'linear':1e-9,'threshold_or_illustration':1e-6,'nonlinear_sample':2e-6,'threshold_status':'resolved'}
            run.state['fluence_material_artifact']=str(Path(directory)/'source.npz')
            run.plot=lambda *a,**k: None
            run.audit_fit_identity=lambda *a,**k: None
            run.dynamics()
            run.work_spectrum()
            for module,argv in calls:
                captured=[]
                class Captured(Exception): pass
                def parse(parser,*a,**k):
                    captured.append(parser)
                    raise Captured
                with patch.object(argparse.ArgumentParser,'parse_args',parse), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    try: runner.runpy.run_module(module,run_name='__main__')
                    except Captured: pass
                self.assertEqual(len(captured),1,module)
                # Actual argparse validation checks flags, choices, types, and
                # required inputs, without executing the numerical solver.
                captured[0].parse_args([*argv,'--output',str(Path(directory)/'check.npz')])


if __name__=='__main__':
    unittest.main()
