"""Exact tight-gap accuracy of the analytic full-QS kernels.

The independent surface-charge BEM fixture reaches a point-QD-to-surface
distance of 0.5 a.  The article's recommended configuration sits at 0.25 a
(a = 10 nm, r_QD = 2 nm, g = 0.5 nm), and the BEM cannot sharpen the check
there: at that gap its own reciprocity -- B extracted from the field channel
versus from the dipole channel, equal analytically -- is already violated by
8-12 %, which is larger than its disagreement with the analytic kernels.

In the sphere limit c = a the reaction field has an elementary closed form that
converges geometrically, so it is an exact reference at ANY gap with no
discretization error.  It is written from the classical image series and uses
no spheroidal harmonic, depolarization factor or project physics module, so
agreement tests precisely the high-order machinery that small gaps exercise.

These tests therefore pin down what the BEM cannot: that the kernels stay exact
at the gaps the article actually recommends, and that the production spatial
order is high enough there.
"""

import unittest

import numpy as np

from qd_mnp_spheroid_equatorial import (
    EquatorialSpheroidGeometry,
    EquatorialSpheroidGreenInteraction,
)
from qd_mnp_spheroid_green import ProlateSpheroidGeometry, SpheroidGreenInteraction


EPS_M = 1.0
EPS_P = -8.0 + 1.0j
RADIUS = 1.0

# Production geometry maps onto these normalized gaps: the QD point stands
# r_QD + g from the surface, so with a = 10 nm and r_QD = 2 nm the recommended
# g = 0.5 nm is 0.25 a and even contact (g = 0) is still 0.20 a.
PRODUCTION_GAP = 0.25
CONTACT_GAP = 0.20
PRODUCTION_ORDER = 160


def sphere_exact(distance, n_max=2000):
    """Elementary dielectric-sphere A, B and radial reaction-field series."""
    contrast = EPS_P - EPS_M
    A = EPS_M * RADIUS**3 * contrast / (EPS_P + 2.0 * EPS_M)
    B = 2.0 * A / (EPS_M * distance**3)
    degree = np.arange(1, n_max + 1, dtype=float)
    denominator = degree * EPS_P + (degree + 1.0) * EPS_M
    # Keep (a/R)^(2n+1)/R^3 together: separate powers overflow long before the
    # decaying ratio stops mattering.
    terms = (
        degree
        * (degree + 1.0) ** 2
        * (RADIUS / distance) ** (2.0 * degree + 1.0)
        * contrast
        / (EPS_M * distance**3 * denominator)
    )
    return complex(A), complex(B), complex(np.sum(terms))


def axis_response(distance, n_max):
    return SpheroidGreenInteraction(
        ProlateSpheroidGeometry(
            a_au=RADIUS, c_au=RADIUS, R_au=distance, eps_m=EPS_M, orientation="long"
        ),
        n_max=n_max,
    ).response_from_epsilon(EPS_P)


def equatorial_response(distance, n_max):
    return EquatorialSpheroidGreenInteraction(
        EquatorialSpheroidGeometry(
            a_au=RADIUS,
            c_au=RADIUS,
            R_au=distance,
            eps_m=EPS_M,
            orientation="trans",
            side_transverse_alignment="radial",
        ),
        n_max=n_max,
    ).response_from_epsilon(EPS_P)


def scalar(value):
    return complex(np.asarray(value).item())


def ports(response):
    return (
        scalar(response.A_au3),
        scalar(response.B),
        scalar(response.K_au_minus3),
    )


class SphereLimitTightGapTests(unittest.TestCase):
    def test_both_kernels_are_exact_at_the_recommended_and_contact_gaps(self):
        for gap in (1.0, 0.5, PRODUCTION_GAP, CONTACT_GAP):
            distance = RADIUS + gap
            exact = sphere_exact(distance)
            for name, builder in (("axis", axis_response), ("equatorial", equatorial_response)):
                with self.subTest(gap=gap, kernel=name):
                    computed = ports(builder(distance, PRODUCTION_ORDER))
                    for label, value, reference in zip("ABK", computed, exact):
                        relative = abs(value - reference) / abs(reference)
                        self.assertLess(
                            relative,
                            1e-12,
                            f"{name} kernel {label} at gap {gap}: relative error {relative:.3g}",
                        )

    def test_the_two_independent_kernels_agree_with_each_other(self):
        # For a sphere the axis and equatorial placements are the same physical
        # configuration, so the two implementations must coincide.
        for gap in (1.0, PRODUCTION_GAP, CONTACT_GAP):
            with self.subTest(gap=gap):
                distance = RADIUS + gap
                axis = ports(axis_response(distance, PRODUCTION_ORDER))
                equatorial = ports(equatorial_response(distance, PRODUCTION_ORDER))
                for label, left, right in zip("ABK", axis, equatorial):
                    self.assertLess(
                        abs(left - right) / abs(left),
                        1e-12,
                        f"kernels disagree on {label} at gap {gap}",
                    )

    def test_the_production_order_is_what_the_recommended_gap_requires(self):
        # The truncation error of the reaction-field series grows as the gap
        # closes. Order 80 -- the previous ceiling -- is already adequate at the
        # recommended gap, but degrades quickly below it, which is why the
        # ceiling was raised. This test fails if the requirement moves.
        distance = RADIUS + PRODUCTION_GAP
        reference = sphere_exact(distance)[2]
        errors = {
            order: abs(scalar(axis_response(distance, order).K_au_minus3) - reference)
            / abs(reference)
            for order in (80, PRODUCTION_ORDER)
        }
        self.assertLess(errors[80], 1e-10)
        self.assertLess(errors[PRODUCTION_ORDER], 1e-14)
        self.assertLess(errors[PRODUCTION_ORDER], errors[80])

        # Well below contact the series is genuinely harder and order 80 fails a
        # gate the production order still passes.
        tight = RADIUS + 0.05
        tight_reference = sphere_exact(tight)[2]
        coarse = abs(scalar(axis_response(tight, 80).K_au_minus3) - tight_reference) / abs(
            tight_reference
        )
        fine = abs(
            scalar(axis_response(tight, PRODUCTION_ORDER).K_au_minus3) - tight_reference
        ) / abs(tight_reference)
        self.assertGreater(coarse, 1e-3)
        self.assertLess(fine, 1e-3)


if __name__ == "__main__":
    unittest.main()
