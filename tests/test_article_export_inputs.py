"""Input routing/provenance of the exporter, without checking article contents."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/prepare_c12p5_a4p5_manuscript.py'
spec = importlib.util.spec_from_file_location('article_export', SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportInputTests(unittest.TestCase):
    def test_native_s1_requires_only_its_recorded_npz_and_enforces_its_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            origin = Path(directory)
            output = origin/'native_s1.npz'
            np.savez(output, energy_eV=np.array([2., 2.1]))
            record = {'status': 'complete', 'output': str(output), 'sha256': export.sha(output)}
            manifest = {'steps': {'supp01_work_spectrum:good': record,
                                 'supp01_work_spectrum:bad': {'status': 'failed'}}}
            seen = []

            def verify(path, digest=None):
                path = Path(path)
                self.assertEqual(path, output)
                self.assertEqual(export.sha(path), digest)
                seen.append(path)
                return path

            arrays, description = export.load_work_spectrum(manifest, origin, verify)
            np.testing.assert_array_equal(arrays['energy_eV'], [2., 2.1])
            self.assertIn('native', description)
            self.assertEqual(seen, [output])
            self.assertEqual(list(origin.iterdir()), [output])
            # An old supplemental directory cannot override the fresh native step.
            (origin/'s1_repaired').mkdir()
            (origin/'s1_repaired/rerun_receipt.json').write_text('invalid old metadata')
            export.load_work_spectrum(manifest, origin, verify)
            output.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                export.load_work_spectrum(manifest, origin, verify)

    def test_recorded_accepted_shape_wins_over_multiple_completed_attempts(self):
        old, accepted = 'shape_old.npz', 'shape_accepted.npz'
        records = [{'status': 'complete', 'output': path, 'sha256': path} for path in (old, accepted)]
        manifest = {'steps': {f'check_shape_c_low:{i}': row for i, row in enumerate(records)}}
        row = {'label': 'c_low', 'shape_spectrum_artifact': accepted}
        self.assertIs(export.shape_artifact(row, manifest), records[1])
        with self.assertRaisesRegex(RuntimeError, 'absent/ambiguous'):
            export.shape_artifact(row | {'shape_spectrum_artifact': 'unrecorded.npz'}, manifest)
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            export.completed_artifact(manifest, 'check_shape_c_low')

    def test_new_source_identity_rejects_drift_and_legacy_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'model.py'
            source.write_text('current code')
            manifest = {'identity': {'source_sha256': {'model.py': export.sha(source)}}}
            with patch.object(export, 'ROOT', root):
                checked = export.check_source_identity(manifest, lambda p, d: p)
                self.assertTrue(checked['accepted'])
                source.write_text('changed code')
                with self.assertRaisesRegex(RuntimeError, 'changed since'):
                    export.check_source_identity(manifest, lambda p, d: p)
                self.assertFalse(export.check_source_identity(manifest, lambda p, d: p, historical=True)['accepted'])

    def test_sensitivity_exports_dynamic_nominal_values_channel_order_and_missing_status(self):
        threshold = {'channel_id': np.array(['side_trans_radial', 'side_long']),
                     'gap_nm': np.array([.5]),
                     'threshold_fluence_j_cm2': np.array([[[9e-6], [7e-6]], [[24e-6], [18e-6]]])}
        validation = [{'label': 'c_low', 'accepted': False,
                       'channels': ['side_trans_radial', 'side_long'], 'gaps_nm': [.5],
                       'candidate_thresholds': [[None], [11e-6]],
                       'candidate_status': [['right_censored'], ['resolved_refined']]}]
        with tempfile.TemporaryDirectory() as directory:
            tables = Path(directory)
            export.sensitivity_table(validation, threshold, {'best_gap_nm': .5}, tables)
            content = (tables/'sensitivity_table.tex').read_text(encoding='utf-8')
            self.assertIn('18,000 & 24,000', content)
            with (tables/'sensitivity.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = {r['label']: r for r in csv.DictReader(stream)}
            self.assertAlmostEqual(float(rows['c_low']['side_long_uJ_cm2']), 11.)
            self.assertEqual(rows['c_low']['side_radial_uJ_cm2'], '')
            self.assertEqual(rows['c_low']['side_radial_status'], 'right_censored')
            self.assertEqual(rows['gap_low']['side_long_status'], 'not_performed')


if __name__ == '__main__':
    unittest.main()
