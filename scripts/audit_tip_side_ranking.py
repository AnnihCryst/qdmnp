"""Audit nominal tip/side ranking using an independent ellipsoid potential and QS-BEM.

This diagnostic does not modify the pulse model, fits, article inputs or results.
The independent potential uses one-dimensional ellipsoidal integrals, not the
project's spheroidal harmonics. BEM solves the Cartesian boundary equation.
"""
from qdmnp.pipeline import ROOT, read_npz, sha, write_json

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from qdmnp.observables.article_inputs import physical_arguments
from qdmnp.rational_fit import AU_LENGTH_M, make_params_with_overrides
from qdmnp.spheroid_green import SpheroidGreenInteraction, qd_linear_polarizability_from_params
from qdmnp.spheroid_equatorial import EquatorialSpheroidGreenInteraction


def uniform_field_ratio(c, a, distance_from_surface, epsilon, eps_m, placement):
    """Exact exterior Ez/E0 for E0 parallel to c; all lengths in the same unit.

    Phi/E0 = -z + (a*a*c/2)*chi*z*I(lambda),
    I(lambda) = integral_lambda^infinity ds/[(a*a+s)*(c*c+s)**(3/2)],
    chi = (epsilon-eps_m)/[eps_m+Lz*(epsilon-eps_m)].
    """
    def integral(lower):
        return quad(lambda s: 1/((a*a+s)*(c*c+s)**1.5), lower, np.inf,
                    epsabs=1e-14, epsrel=1e-12)[0]
    volume_factor = a*a*c/2
    Lz = volume_factor*integral(0)
    chi = (epsilon-eps_m)/(eps_m+Lz*(epsilon-eps_m))
    R = (c if placement == "axis" else a)+distance_from_surface
    lam = R*R-(c*c if placement == "axis" else a*a)
    derivative = (2/((a*a+lam)*R)-integral(lam)) if placement == "axis" else -integral(lam)
    return 1+volume_factor*chi*derivative


def pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bem", action="store_true")
    args = parser.parse_args()
    origin = args.run_directory.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = origin/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["identity"]["config"]
    common = {k.replace("-", "_"): v for k, v in physical_arguments(config).items()}
    g = config["geometry"]
    gap = manifest["state"]["selection"]["best_gap_nm"]
    energy = config["pulse"]["carrier_energy_eV"]
    threshold_path = Path(manifest["state"]["threshold_master"])
    expected = next(r["sha256"] for r in manifest["steps"].values()
                    if r.get("status") == "complete" and r.get("output") == str(threshold_path))
    if sha(threshold_path) != expected:
        raise ValueError("Threshold artifact differs from the original manifest.")
    thresholds, _ = read_npz(threshold_path)
    gi = list(thresholds["gap_nm"]).index(gap)
    report = {"scope": "nominal local-QS model; no assertion of full-wave or finite-QD validity",
              "original_manifest_sha256": sha(manifest_path), "threshold_sha256": expected,
              "c_nm": g["c_nm"], "a_nm": g["a_nm"], "gap_nm": gap,
              "qd_radius_nm": g["qd_radius_nm"], "energy_eV": energy,
              "eps_m": common["eps_m"], "channels": {}}
    nm_per_au = AU_LENGTH_M*1e9
    for placement in ("axis", "side"):
        channel = placement+"_long"
        R = (g["c_nm"] if placement == "axis" else g["a_nm"])+g["qd_radius_nm"]+gap
        params = make_params_with_overrides(**common, r_nm=R, orientation="long", qd_placement=placement)
        factory = SpheroidGreenInteraction if placement == "axis" else EquatorialSpheroidGreenInteraction
        kernel = factory.from_params(params, orientation="long", n_max=80)
        epsilon = complex(params.material.epsilon_at(energy))
        response = kernel.response_from_epsilon(epsilon)
        B, K = complex(response.B), complex(response.K_au_minus3)
        independent = uniform_field_ratio(g["c_nm"], g["a_nm"], g["qd_radius_nm"]+gap,
                                          epsilon, params.eps_m, placement)
        mismatch = abs(independent-(1+B))/abs(1+B)
        assert mismatch < 1e-10
        beta = complex(qd_linear_polarizability_from_params(params, energy))
        ci = list(thresholds["channel_id"]).index(channel)
        row = {"center_distance_nm": R, "epsilon": pair(epsilon), "B": pair(B),
               "E_over_E0": pair(independent), "bare_field_intensity_gain": abs(independent)**2,
               "scattered_field_intensity_gain": abs(B)**2,
               "integral_vs_kernel_relative_difference": mismatch,
               "K_au_minus3": pair(K), "K_nm_minus3": pair(K/nm_per_au**3),
               "linear_reaction_denominator": pair(1-beta*K),
               "monochromatic_qd_response_gain": abs((1+B)/(1-beta*K))**2,
               "threshold_J_cm2": float(thresholds["threshold_fluence_j_cm2"][1, ci, gi]),
               "weak_pulse_population_gain": float(thresholds["weak_field_population_gain"][1, ci, gi])}
        report["channels"][channel] = row
        print(channel, json.dumps(row), flush=True)
        write_json(args.output/"audit.json", report)
        if args.bem:
            sys.path.insert(0, str(ROOT))
            import qd_mnp_bem_validation as bem
            position = [0, 0, params.R_au] if placement == "axis" else [params.R_au, 0, 0]
            print(f"Independent QS-BEM: {channel}, 320/1280/5120 panels", flush=True)
            audit = bem.run_nested_bem_convergence(
                a_au=params.a_au, c_au=params.c_au, qd_position_au=position,
                polarization=[0, 0, 1], eps_m=params.eps_m, epsilon_particle=epsilon,
                subdivision_levels=(2, 3, 4),
            )
            row["bem"] = {
                "raw": [{"panels": r.mesh.panel_count,
                         "B": pair(r.observables.B_field), "K": pair(r.observables.K_au_minus3),
                         "bare_field_intensity_gain": abs(1+r.observables.B_field)**2,
                         "reciprocity_relative_error": r.diagnostics.reciprocity_relative_error,
                         "relative_residual_uniform": r.diagnostics.relative_residual_uniform,
                         "relative_residual_dipole": r.diagnostics.relative_residual_point_dipole,
                         "point_distance_over_max_edge": r.diagnostics.qd_distance_over_max_edge}
                        for r in audit.responses],
                "extrapolated_B": pair(audit.extrapolated.B_field),
                "extrapolated_K": pair(audit.extrapolated.K_au_minus3),
                "extrapolated_field_gain": abs(1+audit.extrapolated.B_field)**2,
                "B_relative_difference": abs(audit.extrapolated.B_field-B)/abs(B),
                "K_relative_difference": abs(audit.extrapolated.K_au_minus3-K)/abs(K),
                "estimated_relative_uncertainty": asdict(audit.estimated_relative_uncertainty),
            }
            print(channel, "BEM:", json.dumps(row["bem"]), flush=True)
            write_json(args.output/"audit.json", report)

    # Figures use only the independent potential; no fitted dispersion or kernel.
    energies = np.linspace(1.60, 2.20, 1201)
    distances = np.linspace(0, 10, 1001)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for placement, label in (("axis", "tip"), ("side", "side, E parallel to c")):
        spectrum = uniform_field_ratio(g["c_nm"], g["a_nm"], g["qd_radius_nm"]+gap,
                                       params.material.epsilon_at(energies), params.eps_m, placement)
        profile = [uniform_field_ratio(g["c_nm"], g["a_nm"], d, epsilon, params.eps_m, placement)
                   for d in distances]
        axes[0].semilogy(energies, abs(spectrum)**2, label=label)
        axes[1].semilogy(distances, np.abs(profile)**2, label=label)
    axes[0].axvline(energy, color="black", ls=":", label="fixed carrier")
    axes[1].axvline(g["qd_radius_nm"]+gap, color="black", ls=":", label="QD centre")
    axes[0].set(xlabel="Photon energy (eV)", ylabel="Bare-MNP total field |E/E0|^2")
    axes[1].set(xlabel="Distance of observation point from surface (nm)", ylabel="|E/E0|^2 at fixed carrier")
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.legend()
    fig.savefig(args.output/"tip_side_fields.png", dpi=200)
    plt.close(fig)
    report["source_sha256"] = {str(p): sha(p) for p in [Path(__file__), ROOT/"qd_mnp_bem_validation.py",
        ROOT/"src/qdmnp/spheroid_green.py", ROOT/"src/qdmnp/spheroid_equatorial.py", ROOT/"src/qdmnp/rational_fit.py"]}
    write_json(args.output/"audit.json", report)
    print("Saved independent ranking audit:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
