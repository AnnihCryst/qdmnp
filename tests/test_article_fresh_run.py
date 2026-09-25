"""Archive-free numerical fallbacks and their integration into the main runner."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qdmnp import pipeline as runner
from qdmnp.observables.article_inputs import load_inputs, validate_inputs
from qdmnp.observables.plot_fixed_and_pulse_gain import gain_curves


class FreshRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cfg = load_inputs(runner.ROOT/'inputs/ARTICLE_INPUTS_c12p5_a4p5.toml')
        self.cfg['numerics']['workers'] = 1
        self.run = runner.ArticleRun(self.cfg, self.directory)
        self.limits = {'max-modal-normalized-rms': .12,
                       'max-modal-relative-error': .25, 'spatial-convergence-rtol': 1e-4}

    def test_fallback_refits_target_and_routes_its_seed_to_both_calculators(self):
        cfg = deepcopy(self.cfg)
        cfg['geometry']['a_nm'] *= 1.05
        initial = deepcopy(cfg)
        triples = tuple((1., 1.+i*.1, .1) for i in range(13))
        calls = []

        def spectrum(label, gap, **kwargs):
            calls.append(deepcopy(kwargs))
            if not kwargs['config']['material']['refinement'].get('initial_modes_eV'):
                raise runner.StageError('Native fit misses requested pointwise accuracy')
            return self.directory/'accepted_shape.npz'

        with patch.object(self.run, 'material_spectrum', side_effect=spectrum), \
                patch('qdmnp.observables.article_fit_seed.native_material_seed',
                      return_value=(triples, {'role': 'initial_guess_only'})) as seed:
            result, count, overrides, record = self.run.shape_validation_candidate('a_high', cfg, [.5], self.limits)
        self.assertEqual(cfg, initial)
        self.assertEqual([x['count'] for x in calls], [12, 13, 13])
        self.assertEqual(count, 13)
        self.assertEqual(seed.call_args.kwargs['c_nm'], 11.875)
        self.assertEqual(seed.call_args.kwargs['a_nm'], 4.5)
        self.assertEqual(result['geometry']['a_nm'], 4.5*1.05)
        self.assertEqual(record['material_seed']['role'], 'initial_guess_only')
        self.assertEqual(record['spatial_order_used'], 160)
        self.assertEqual(len(record['refinement_attempts']), 2)
        self.assertEqual(record['relaxed_numerical_gates'], self.limits)
        self.assertEqual(overrides['spatial-order-max'], 160)
        expected = {'long': [list(row) for row in triples]}
        arguments = self.run.threshold_arguments(config=result, count=count)
        self.assertEqual(json.loads(arguments['fit-refinement'])['initial_modes_eV'], expected)
        self.assertEqual(arguments['max-bright-fit-pointwise-relative-error'], .05)
        with patch.object(self.run, 'step') as step:
            self.run.material_spectrum('shape', .5, count=count, config=result, overrides=overrides)
        arguments = step.call_args.args[2]
        self.assertEqual(json.loads(arguments['fit-refinement'])['initial_modes_eV'], expected)
        self.assertEqual(arguments['a-nm'], 4.5*1.05)
        self.assertEqual(arguments['spatial-order-max'], 160)
        from qdmnp.observables.calculate_excitation_spectrum_material_comparison import parse_args
        parsed = parse_args([*runner.flags(arguments), '--output', str(self.directory/'unused.npz')])
        self.assertEqual(parsed.fit_refinement['initial_modes_eV'], {'long': triples})
        self.assertEqual(parsed.multi_fit_modes, 13)

    def test_native_success_does_not_generate_seed_and_failed_fallback_stays_failed(self):
        with patch.object(self.run, 'material_spectrum', return_value=self.directory/'shape.npz'), \
                patch('qdmnp.observables.article_fit_seed.native_material_seed') as seed:
            _, count, _, record = self.run.shape_validation_candidate('c_high', self.cfg, [.5], self.limits)
            seed.assert_not_called()
            self.assertEqual(count, 12)
            self.assertEqual(record['material_refinement'], 'native')
        self.run.config['validation']['shape_fit_fallback'] = 'none'
        with patch.object(self.run, 'material_spectrum', side_effect=runner.StageError('fit inaccurate')), \
                patch('qdmnp.observables.article_fit_seed.native_material_seed') as seed:
            with self.assertRaises(runner.StageError):
                self.run.shape_validation_candidate('c_low', self.cfg, [.5], self.limits)
            seed.assert_not_called()
        self.run.config['validation']['shape_fit_fallback'] = 'native_neighbor_minimax'
        with patch.object(self.run, 'material_spectrum', side_effect=runner.StageError('fit inaccurate')), \
                patch('qdmnp.observables.article_fit_seed.native_material_seed',
                      return_value=(((1., 2., .1),)*13, {})):
            with self.assertRaisesRegex(runner.StageError, 'fit inaccurate'):
                self.run.shape_validation_candidate('c_low', self.cfg, [.5], self.limits)

    def test_spatial_retry_preserves_gates_and_does_not_retry_other_failures(self):
        arguments = self.run.threshold_arguments(gaps=[.3], channels=['side_long', 'side_trans_radial'])
        item = {'label': 'gap_low', 'extra': {}, 'spec': ('check_gap_low', 'fake', arguments)}
        with patch.object(self.run, 'step', return_value=self.directory/'refined.npz') as step:
            result = self.run.refine_sensitivity_spatial_failure(item, runner.StageError('Spatial series did not converge'))
        self.assertEqual(result, self.directory/'refined.npz')
        refined = step.call_args.args[2]
        self.assertEqual(refined, arguments | {'spatial-order-max': 160})
        self.assertEqual(arguments['spatial-order-max'], 80)
        self.assertEqual(item['extra']['spatial_order_used'], 160)
        for error in ('tail did not converge', 'material fit rejected'):
            with patch.object(self.run, 'step') as step:
                failure = runner.StageError(error)
                self.assertIs(self.run.refine_sensitivity_spatial_failure(item, failure), failure)
                step.assert_not_called()
        with patch.object(self.run, 'step', side_effect=runner.StageError('spatial still fails')) as step:
            self.assertIsInstance(self.run.refine_sensitivity_spatial_failure(item, runner.StageError('spatial')), runner.StageError)
            self.assertEqual(step.call_count, 1)

    def test_main_validation_merges_repaired_cases_in_native_ranking(self):
        channels, gaps = self.cfg['geometry']['channels'], np.asarray(self.cfg['geometry']['gaps_nm'])
        fqs = np.broadcast_to(np.array([5., 8., 1., 2., 9.])[:, None]*1e-5, (5, len(gaps))).copy()
        fqs *= np.linspace(1., 3., len(gaps))
        master = dict(channel_id=np.array(channels), gap_nm=gaps,
            threshold_fluence_j_cm2=np.stack([fqs, fqs]),
            threshold_status=np.full((2, 5, len(gaps)), 'resolved_refined'),
            isolated_threshold_fluence_j_cm2=np.asarray(6e-5), isolated_threshold_status=np.asarray('resolved_refined'))
        path = self.directory/'master.npz'
        np.savez(path, metadata_json=np.asarray('{}'), **master)
        self.run.state.update(threshold_master=str(path), selection={
            'best_channel': 'side_long', 'runner_up_channel': 'side_trans_radial', 'control_channel': 'axis_long',
            'best_gap_nm': .5, 'runner_up_gap_nm': .5, 'resolved_threshold_selected': True,
            'selection_allowed_by_declared_retardation_cutoffs': np.ones((5, len(gaps)), bool)})

        def spectrum(label, gap, **kwargs):
            if label in ('check_shape_c_low', 'check_shape_a_high'):
                if not kwargs['config']['material']['refinement'].get('initial_modes_eV'):
                    raise runner.StageError('fit inaccurate')
            return self.directory/(label+'.npz')

        def solve(label, module, args):
            if label == 'check_gap_low' and args['spatial-order-max'] == 80:
                raise runner.StageError('spatial series did not converge')
            data = runner.threshold_subset(master, args['channels'], [.5])
            data['gap_nm'] = np.array(args['gaps-nm'])
            target = self.directory/(label+'.npz')
            np.savez(target, metadata_json=np.asarray('{}'), **data)
            return target

        def concurrent(specs, concurrency):
            outcomes = {}
            for label, module, args in specs:
                try:
                    outcomes[label] = solve(label, module, args)
                except runner.StageError as error:
                    outcomes[label] = error
            return outcomes

        with patch.object(self.run, 'material_spectrum', side_effect=spectrum), \
                patch.object(self.run, 'steps_concurrently', side_effect=concurrent), \
                patch.object(self.run, 'step', side_effect=solve), patch.object(self.run, 'plot_validation'), \
                patch.object(runner, 'compare_spectral_refinement', return_value={'accepted': True}), \
                patch('qdmnp.observables.article_fit_seed.native_material_seed',
                      return_value=(((1., 2., .1),)*13, {'role': 'initial_guess_only'})):
            self.run.validation()
        records = {row['label']: row for row in self.run.state['validation']}
        self.assertTrue(all(row['accepted'] for row in records.values()))
        self.assertEqual(records['gap_low']['spatial_order_used'], 160)
        for label in ('c_low', 'a_high'):
            self.assertEqual(records[label]['material_refinement'], 'native_neighbor_minimax')
        self.assertTrue(self.run.state['ranking_validation']['accepted'])
        saved = json.loads((self.directory/'ranking_validation.json').read_text())
        self.assertTrue(saved['accepted'])
        self.assertEqual(self.run.spatial_order, 80)
        self.assertNotIn('initial_modes_eV', self.run.config['material']['refinement'])

    def test_shape_policy_requires_supported_mode_and_refinement(self):
        config = deepcopy(self.cfg)
        config['validation']['shape_fit_fallback'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'shape_fit_fallback'):
            validate_inputs(config)
        config['validation']['shape_fit_fallback'] = 'native_neighbor_minimax'
        config['material'].pop('refinement')
        with self.assertRaisesRegex(ValueError, 'refinement'):
            validate_inputs(config)

    def test_diagnostic_plots_use_only_native_artifacts(self):
        common = dict(model_id=np.array(['dd', 'fqs']), channel_id=np.array(['side_long', 'axis_long']), gap_nm=np.array([.5, 1.]))
        spectral = common | {'excitation_gain_at_reference_energy': np.ones((2, 2, 2))}
        thresholds = common | {'weak_field_population_gain': np.full((2, 2, 2), 2.)}
        gain_curves(spectral, thresholds)
        with self.assertRaisesRegex(ValueError, 'gap_nm'):
            gain_curves(spectral, thresholds | {'gap_nm': np.array([.5, 2.])})
        for name, arrays in [('spectral', spectral), ('thresholds', thresholds)]:
            np.savez(self.directory/(name+'.npz'), **arrays)
        self.run.state.update(spectral_master=str(self.directory/'spectral.npz'),
                              threshold_master=str(self.directory/'thresholds.npz'), selection={'best_gap_nm': .5})
        self.run.article_diagnostics()
        for role in ('tip_side_fields', 'fixed_and_pulse_gain'):
            output = self.directory/self.run.manifest['figures'][role]
            self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(len(self.run.manifest['steps']), 2)
        with patch.object(runner.runpy, 'run_module', side_effect=AssertionError('must reuse verified figures')):
            self.run.article_diagnostics()


if __name__ == '__main__':
    unittest.main()
