#!/usr/bin/env python
"""Recheck the saved article run without fitting materials or solving ODEs.

Writes numerical evidence for docs/MANUSCRIPT_RESULTS_AUDIT.md.  This checks
artifact integrity and independently reconstructs the published observables;
it is not a physical validation or a replacement for the production run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def audit(directory, manuscript, extra_figure=None):
    report = json.loads((directory / "article_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    state = report["results"]
    checks = {}

    def check(name, actual, expected, rtol=2e-12, atol=0):
        checks[name] = bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))

    def artifact(prefix):
        # Use completed manifest steps, excluding failed refinement attempts.
        matches = [directory / "data" / Path(v["output"]).name
                   for k, v in manifest["steps"].items()
                   if k.split(":")[0] == prefix and v["status"] == "complete"
                   and Path(v["output"]).suffix == ".npz"]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed {prefix} artifact, got {matches}")
        with np.load(matches[0], allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    source_mismatches = [name for name, digest in manifest["identity"]["source_sha256"].items()
                         if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    output_mismatches = []
    completed = 0
    for step in manifest["steps"].values():
        if step["status"] != "complete":
            continue
        path = Path(step["output"])
        local = directory / path.parent.name / path.name
        completed += 1
        if not local.is_file() or sha256(local) != step["sha256"]:
            output_mismatches.append(str(local))
    checks["recorded_sources_match"] = not source_mismatches
    checks["completed_outputs_match"] = not output_mismatches
    figures = {}
    for name, relative in report["figures"].items():
        target = manuscript / "figures" / (name + ".png")
        figures[name] = target.is_file() and sha256(directory / relative) == sha256(target)
    if extra_figure is not None:
        figures["fig02e_rebuilt"] = sha256(extra_figure) == sha256(manuscript / "figures/fig02e.png")
    checks["manuscript_figures_match"] = all(figures.values())

    z = artifact("fig02_master")
    meta = json.loads(z["metadata_json"].item())
    params, units = meta["reference_physical_parameters"], meta["physical_constants"]
    e = z["energy_eV"]
    w = e / units["atomic_energy_eV"]
    w0 = params["omega0_ev"] / units["atomic_energy_eV"]
    gamma2 = params["gamma2_coherence_mev"] * 1e-3 / units["atomic_energy_eV"]
    d = params["qd_external_dipole_debye"] * units["debye_C_m"] / units["atomic_dipole_C_m"]
    beta = 2 * d**2 * w0 / (w0**2 + (gamma2 - 1j*w)**2)
    transfer = beta * (1 + z["interaction_B"]) / (1 - beta*z["interaction_K_au_minus3"])
    spectrum = np.abs(transfer)**2
    check("linear_QD_transfer", transfer, z["qd_dipole_over_field_au3"])
    check("linear_QD_spectrum", spectrum, z["qd_excitation_spectrum"])
    check("isolated_QD_spectrum", abs(beta)**2, z["isolated_qd_spectrum"])
    args = meta["requested_arguments"]
    center, half = args["feature_center_ev"], args["feature_half_window_ev"]
    mask = abs(e-center) <= half
    dd_window, fqs_window = spectrum[0][..., mask], spectrum[1][..., mask]
    delta = np.sqrt(np.trapezoid((dd_window-fqs_window)**2, e[mask], axis=-1)
                    / np.trapezoid(fqs_window**2, e[mask], axis=-1))
    check("spectral_discrepancy", delta, z["spectral_l2_relative_dd_vs_fqs"])
    optimized = spectrum[...,mask].max(axis=-1) / (abs(beta[mask])**2).max()
    fixed = np.array([np.interp(center, e, row) for row in spectrum.reshape(-1,e.size)])
    fixed = fixed.reshape(spectrum.shape[:-1]) / np.interp(center, e, abs(beta)**2)
    check("optimized_gain", optimized, z["excitation_gain_optimized"])
    check("fixed_gain", fixed, z["excitation_gain_at_reference_energy"])
    gaps, channels = z["gap_nm"], z["channel_id"].tolist()
    power = {}
    for ci, channel in enumerate(channels):
        radius = z["center_distance_nm"][ci]
        use = gaps >= 3
        slope, intercept = np.polyfit(np.log(radius[use]), np.log(delta[ci,use]), 1)
        fitted = np.exp(intercept)*radius[use]**slope
        power[channel] = dict(exponent=-slope, max_relative_residual=max(abs(fitted/delta[ci,use]-1)),
                              extrapolated_R_nm=np.exp((np.log(.1)-intercept)/slope))
    spectral = dict(channels=channels, gaps_nm=gaps, fixed_gain=fixed, optimized_gain=optimized,
                    delta_spec=delta, shift_over_gamma0=z["resonance_shift_over_gamma0"],
                    width_over_gamma0=z["spectral_width_over_gamma0"], gamma0_eV=z["isolated_fwhm_eV"],
                    dd_boundary_nm=z["dd_validity_gap_nm"], tail_power_fits=power)

    t = artifact("fig03_master")
    thresholds = t["threshold_fluence_j_cm2"]
    f0 = t["isolated_threshold_fluence_j_cm2"].item()
    check("threshold_ratio", thresholds/f0, t["threshold_fluence_ratio_to_isolated"])
    delta_f = abs(thresholds[0]/thresholds[1]-1)
    check("threshold_discrepancy", delta_f, t["absolute_threshold_discrepancy_dd_vs_fqs"])
    material = artifact("fig01_material")
    material_metrics = {key: material[key] for key in
                        ("nrms_alpha", "nrms_inverse_alpha", "max_normalized_alpha_error",
                         "max_normalized_inverse_alpha_error")}
    fluence = artifact("fig05_material_fluence")
    ratio = fluence["threshold_fluence_j_cm2"][0,1:]/fluence["threshold_fluence_j_cm2"][1,1:]
    check("material_threshold_ratio", ratio, fluence["hybrid_threshold_ratio_one_to_multi"])
    dynamics = {}
    for tag in ("near", "far"):
        data = artifact("fig06_dynamics_"+tag)
        dynamics[tag] = dict(zip(data["channel_ids"].tolist(), data["population_final"].tolist()))

    work = artifact("supp01_work_spectrum")
    check("work_spectrum_subtraction", work["sigma_qs_work_cm2"]-work["bare_mnp_sigma_qs_work_cm2"][:,:,None,:],
          work["delta_sigma_qs_work_cm2"], rtol=1e-8, atol=1e-26)
    extrema = []
    for bi, branch in enumerate(("one", "multi")):
        for fi, f in enumerate(work["selected_fluence_j_cm2"]):
            curve = work["delta_sigma_qs_work_cm2"][bi,0,fi]
            peak = int(np.argmax(curve))
            extrema.append(dict(branch=branch, fluence_J_cm2=f, minimum_cm2=curve.min(),
                                maximum_cm2=curve.max(), max_absolute_cm2=abs(curve).max(),
                                peak_to_peak_cm2=np.ptp(curve), maximum_energy_eV=work["energy_eV"][peak],
                                bare_at_maximum_cm2=work["bare_mnp_sigma_qs_work_cm2"][bi,0,peak]))
    return clean(dict(scope="Saved-run integrity and algebraic postprocessing; no new ODE solves",
                      accepted=all(checks.values()), checks=checks,
                      provenance=dict(source_count=len(manifest["identity"]["source_sha256"]),
                                      source_mismatches=source_mismatches, completed_output_count=completed,
                                      output_mismatches=output_mismatches, figures=figures,
                                      failed_attempts={k:v.get("error") for k,v in manifest["steps"].items()
                                                       if v["status"]=="failed"}),
                      material=material_metrics, spectral=spectral,
                      thresholds=dict(isolated_J_cm2=f0, by_model_channel_gap_J_cm2=thresholds,
                                      status=t["threshold_status"], delta_F=delta_f,
                                      best_gain=f0/thresholds[1,0,0]),
                      material_threshold_ratios=ratio,
                      max_population_difference_N1_N12=np.max(abs(fluence["population_one_minus_multi"])),
                      dynamics=dynamics, work_spectrum_extrema=extrema,
                      numerical_validation_accepted=state["numerical_validation_accepted"],
                      validation=state["validation"], ranking=state["ranking_validation"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/article_shah_single")
    parser.add_argument("--manuscript", type=Path, default=ROOT/"manuscript")
    parser.add_argument("--rebuilt-fixed-gain", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"results/manuscript_audit/evidence.json")
    args = parser.parse_args()
    evidence = audit(args.results, args.manuscript, args.rebuilt_fixed_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(accepted=evidence["accepted"], checks=evidence["checks"], output=str(args.output)), indent=2))
    return 0 if evidence["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
