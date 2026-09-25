"""Independent bare-field formula and archive-free figure generation."""
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from qdmnp.observables.article_field_diagnostic import (
    build_field_diagnostic, main, uniform_field_ratio,
)
from qdmnp.rational_fit import DEFAULT_AU_MATERIAL, make_params_with_overrides
from qdmnp.spheroid_green import SpheroidGreenInteraction
from qdmnp.spheroid_equatorial import EquatorialSpheroidGreenInteraction


class ArticleFieldDiagnosticTests(unittest.TestCase):
    def test_confocal_potential_matches_independent_spheroidal_bright_field(self):
        # Three shapes, distances and energies; both independent spatial kernels.
        for c_nm, a_nm, distance in ((12.5, 4.5, 2.5), (15., 10., 3.), (11.875, 4.5, 2.3)):
            for placement, factory in (("axis", SpheroidGreenInteraction),
                                       ("side", EquatorialSpheroidGreenInteraction)):
                params = make_params_with_overrides(c_nm=c_nm, a_nm=a_nm, eps_m=2.25,
                    r_nm=(c_nm if placement == "axis" else a_nm)+distance,
                    orientation="long", qd_placement=placement)
                kernel = factory.from_params(params, orientation="long", n_max=8)
                energies = np.array([1.768625, 2.042, 2.2])
                epsilon = DEFAULT_AU_MATERIAL.epsilon_at(energies)
                expected = 1 + kernel.response_from_epsilon(epsilon).B
                actual = uniform_field_ratio(c_nm, a_nm, distance, epsilon, 2.25, placement)
                with self.subTest(c_nm=c_nm, a_nm=a_nm, placement=placement):
                    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)

    def test_sphere_limit_and_absent_dielectric_contrast(self):
        epsilon, medium, radius, distance = -9.+1.5j, 2.25, 7., 3.
        dipole = radius**3/(radius+distance)**3*(epsilon-medium)/(epsilon+2*medium)
        for placement, angular in (("axis", 2), ("side", -1)):
            self.assertAlmostEqual(uniform_field_ratio(radius, radius, distance, epsilon, medium, placement),
                                   1+angular*dipole, places=12)
            self.assertEqual(uniform_field_ratio(12.5, 4.5, 0., medium, medium, placement), 1.)

    def test_fresh_arrays_preserve_surface_vs_qd_centre_distinction(self):
        data = build_field_diagnostic(12.5, 4.5, 2.25, 2.042, 2., .5)
        info = data["interpretation"]
        self.assertEqual(info["centre_distance_from_surface_nm"], 2.5)
        self.assertGreater(info["surface_field_gain"]["axis"], info["surface_field_gain"]["side"])
        self.assertLess(info["centre_field_gain"]["axis"], info["centre_field_gain"]["side"])
        self.assertLess(info["first_profile_crossing_nm"], 2.5)
        self.assertEqual(data["spectrum_field_gain"].shape, (2, 1201))
        self.assertTrue(np.all(np.isfinite(data["profile_field_gain"])))

    def test_cli_writes_png_in_empty_directory_without_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"figures"/"tip_side_fields.png"
            result = main(["--c-nm", "12.5", "--a-nm", "4.5", "--eps-m", "2.25",
                           "--carrier-energy-ev", "2.042", "--qd-radius-nm", "2", "--gap-nm", ".5",
                           "--dpi", "60", "--output", str(output)])
            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual([p for p in Path(directory).rglob("*") if p.is_file()], [output])


if __name__ == "__main__":
    unittest.main()
