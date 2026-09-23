"""Shared, explicitly recorded opt-in options for passive material refinement."""
import argparse
from dataclasses import asdict
import json
from qdmnp.passive_fit import PassiveFitRefinement


def parse_fit_refinement(text):
    try:
        values = json.loads(text)
        if not isinstance(values, dict):
            raise ValueError('expected a JSON object')
        return asdict(PassiveFitRefinement(**values))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f'Invalid passive-fit refinement: {exc}') from exc


def add_fit_refinement_argument(parser):
    parser.add_argument('--fit-refinement', type=parse_fit_refinement, default=None,
        help='JSON numerical options for positive-Lorentz refinement: focus_center_eV, '
             'focus_half_width_eV, focus_relative_error, pole_bound_factor, '
             'optional initial_modes_eV (long/trans lists of [strength_eV2, energy_eV, damping_eV]). '
             'Global accuracy/passivity gates still apply; omitted retains legacy fitting.')
