"""Offline generator for immutable independent spheroid-BEM fixtures.

This program intentionally imports only the Cartesian BEM validator.  In
particular, it never imports either spheroidal-harmonic implementation.  The
checked-in JSON is therefore a numerical reference that can be compared with
the analytic full-QS kernels without circularly generating its expected data.

The default level-4 mesh contains 5120 panels and allocates roughly 0.4 GiB
for one dense complex matrix.  Fixture regeneration is an offline operation,
not part of the live unit-test suite.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import scipy

import qd_mnp_bem_validation as bem


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "tests" / "fixtures" / "spheroid_bem_v1.json"
DEFAULT_LEVELS = (1, 2, 3, 4)


CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "sphere_axis_long_ordinary",
        "regime": "ordinary",
        "a_au": 1.0,
        "c_au": 1.0,
        "qd_position_au": (0.0, 0.0, 3.0),
        "polarization": (0.0, 0.0, 1.0),
        "placement": "axis",
        "orientation": "long",
        "side_transverse_alignment": None,
    },
    {
        "id": "prolate_axis_long_ordinary",
        "regime": "ordinary",
        "a_au": 1.0,
        "c_au": 1.5,
        "qd_position_au": (0.0, 0.0, 2.5),
        "polarization": (0.0, 0.0, 1.0),
        "placement": "axis",
        "orientation": "long",
        "side_transverse_alignment": None,
    },
    {
        "id": "prolate_axis_trans_ordinary",
        "regime": "ordinary",
        "a_au": 1.0,
        "c_au": 1.5,
        "qd_position_au": (0.0, 0.0, 2.5),
        "polarization": (1.0, 0.0, 0.0),
        "placement": "axis",
        "orientation": "trans",
        "side_transverse_alignment": None,
    },
    {
        "id": "prolate_side_long_ordinary",
        "regime": "ordinary",
        "a_au": 1.0,
        "c_au": 1.5,
        "qd_position_au": (2.0, 0.0, 0.0),
        "polarization": (0.0, 0.0, 1.0),
        "placement": "side",
        "orientation": "long",
        "side_transverse_alignment": None,
    },
    {
        "id": "prolate_side_trans_radial_ordinary",
        "regime": "ordinary",
        "a_au": 1.0,
        "c_au": 1.5,
        "qd_position_au": (2.0, 0.0, 0.0),
        "polarization": (1.0, 0.0, 0.0),
        "placement": "side",
        "orientation": "trans",
        "side_transverse_alignment": "radial",
    },
    {
        "id": "prolate_side_trans_tangential_ordinary",
        "regime": "ordinary",
        "a_au": 1.0,
        "c_au": 1.5,
        "qd_position_au": (2.0, 0.0, 0.0),
        "polarization": (0.0, 1.0, 0.0),
        "placement": "side",
        "orientation": "trans",
        "side_transverse_alignment": "tangential",
    },
    {
        "id": "prolate_side_trans_radial_small_gap",
        "regime": "small_gap",
        "a_au": 1.0,
        "c_au": 1.5,
        "qd_position_au": (1.5, 0.0, 0.0),
        "polarization": (1.0, 0.0, 0.0),
        "placement": "side",
        "orientation": "trans",
        "side_transverse_alignment": "radial",
    },
)


def _complex(value: complex) -> list[float]:
    number = complex(value)
    return [float(number.real), float(number.imag)]


def _observables(value: bem.BEMObservables) -> dict[str, list[float]]:
    return {
        "A_au3": _complex(value.A_au3),
        "B_field": _complex(value.B_field),
        "B_dipole": _complex(value.B_dipole),
        "K_au_minus3": _complex(value.K_au_minus3),
    }


def _errors(value: bem.BEMObservableErrors) -> dict[str, float]:
    return {
        "A": value.A,
        "B_field": value.B_field,
        "B_dipole": value.B_dipole,
        "K": value.K,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _surface_gap(case: dict[str, Any]) -> float:
    position = np.asarray(case["qd_position_au"], dtype=float)
    if case["placement"] == "axis":
        return float(abs(position[2]) - float(case["c_au"]))
    return float(np.hypot(position[0], position[1]) - float(case["a_au"]))


def _generate_case(
    specification: dict[str, Any],
    *,
    levels: tuple[int, ...],
    max_dense_panels: int,
) -> dict[str, Any]:
    eps_m = 1.0
    epsilon_particle = -8.0 + 1.0j
    convergence = bem.run_nested_bem_convergence(
        a_au=specification["a_au"],
        c_au=specification["c_au"],
        qd_position_au=specification["qd_position_au"],
        polarization=specification["polarization"],
        eps_m=eps_m,
        epsilon_particle=epsilon_particle,
        subdivision_levels=levels,
        assumed_order=1.0,
        correction_terms=2,
        extrapolation_level_count=3,
        max_dense_panels=max_dense_panels,
    )
    raw_levels = []
    for response in convergence.responses:
        diagnostics = response.diagnostics
        raw_levels.append(
            {
                "subdivision_level": response.mesh.subdivision_level,
                "panel_count": response.mesh.panel_count,
                "vertex_count": response.mesh.vertex_count,
                "max_edge_au": response.mesh.max_edge_au,
                "mesh_sha256": bem.mesh_sha256(response.mesh),
                "observables": _observables(response.observables),
                "relative_residual_uniform": diagnostics.relative_residual_uniform,
                "relative_residual_point_dipole": (
                    diagnostics.relative_residual_point_dipole
                ),
                "relative_net_charge_uniform": diagnostics.relative_net_charge_uniform,
                "relative_net_charge_point_dipole": (
                    diagnostics.relative_net_charge_point_dipole
                ),
                "reciprocity_relative_error": diagnostics.reciprocity_relative_error,
                "minimum_qd_surface_distance_au": (
                    diagnostics.minimum_qd_surface_distance_au
                ),
                "qd_distance_over_max_edge": diagnostics.qd_distance_over_max_edge,
            }
        )

    result = dict(specification)
    result.update(
        {
            "surface_gap_au": _surface_gap(specification),
            "eps_m": eps_m,
            "epsilon_particle": _complex(epsilon_particle),
            "acceptance_relative": {
                "A": 0.005,
                "B_field": 0.005,
                "B_dipole": 0.005,
                "K": 0.02 if specification["regime"] == "small_gap" else 0.01,
                "uncertainty_sigma_multiplier": 3.0,
            },
            "extrapolation_levels": list(convergence.extrapolation_levels),
            "correction_orders": list(convergence.correction_orders),
            "extrapolation_h_scale_au": convergence.extrapolation_h_scale_au,
            "uncertainty_method": convergence.uncertainty_method,
            "extrapolated": _observables(convergence.extrapolated),
            "lower_order_extrapolated": _observables(
                convergence.lower_order_extrapolated
            ),
            "estimated_absolute_uncertainty": _errors(
                convergence.estimated_absolute_uncertainty
            ),
            "estimated_relative_uncertainty": _errors(
                convergence.estimated_relative_uncertainty
            ),
            "finest_relative_change": _errors(convergence.finest_relative_change),
            "extrapolation_relative_residual": _errors(
                convergence.extrapolation_relative_residual
            ),
            "raw_levels": raw_levels,
        }
    )
    return result


def generate_fixture(
    *,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    max_dense_panels: int = bem.MAX_DENSE_PANELS,
) -> dict[str, Any]:
    if len(levels) < 3:
        raise ValueError("Fixture generation requires at least three mesh levels.")
    module_path = Path(bem.__file__).resolve()
    generator_path = Path(__file__).resolve()
    return {
        "schema": "qdmnp.independent_spheroid_bem_fixture",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "generator": generator_path.name,
            "generator_sha256": _file_sha256(generator_path),
            "solver_module": module_path.name,
            "solver_sha256": _file_sha256(module_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "equation": "[Lambda I + K*] sigma = n dot E_inc",
            "surface_mesh": "affine icosphere; flat triangular panels",
            "density_basis": "piecewise constant",
            "collocation": "triangle centroid",
            "quadrature": "constant-panel centroid Nystrom",
            "extrapolation": "X(h)=X0+c1*h+c2*h^2 on finest three levels",
            "uncertainty": (
                "max(two-term versus leading-only X0 shift, absolute fit residual)"
            ),
            "independence": (
                "No spheroidal harmonics, depolarization factors, or analytic "
                "full-QS response are imported by this generator or its solver."
            ),
            "scope_limit": (
                "The small-gap fixture has gap/a=0.5. Smaller gaps require "
                "additional local refinement or near-singular quadrature and "
                "are not certified by this fixture set."
            ),
        },
        "mesh_levels": list(levels),
        "cases": [
            _generate_case(
                case,
                levels=levels,
                max_dense_panels=max_dense_panels,
            )
            for case in CASES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Strictly increasing affine-icosphere subdivision levels.",
    )
    parser.add_argument(
        "--max-dense-panels",
        type=int,
        default=bem.MAX_DENSE_PANELS,
        help="Explicit memory guard passed to the dense solver.",
    )
    args = parser.parse_args()
    levels = tuple(args.levels)
    fixture = generate_fixture(
        levels=levels,
        max_dense_panels=args.max_dense_panels,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output} ({len(fixture['cases'])} cases).")


if __name__ == "__main__":
    main()
