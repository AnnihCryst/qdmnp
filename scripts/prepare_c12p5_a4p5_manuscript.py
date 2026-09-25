"""Verify saved article artifacts and export tables/figures, without new ODE runs.

Run from the repository root after the ordinary article pipeline completes.
No archived diagnostics or test logs are required. ``--sensitivity`` explicitly
selects the legacy supplemental-repair workflow for an older saved run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qdmnp.rational_fit import AU_ENERGY_EV, AU_DIPOLE_C_M, DEBYE_C_M

ROOT = Path(__file__).resolve().parents[1]
CHANNELS_RU = ["Торец, продольный", "Торец, поперечный", "Бок, продольный",
               "Бок, радиальный", "Бок, тангенциальный"]
SENSITIVITY_LABELS = {
    "gap_low": r"$g-\Delta g$", "gap_high": r"$g+\Delta g$",
    "c_low": r"$c-\Delta c$", "c_high": r"$c+\Delta c$",
    "a_low": r"$a=b-\Delta a$", "a_high": r"$a=b+\Delta a$",
    "exciton_low": r"$E_q-\Delta E_q$", "exciton_high": r"$E_q+\Delta E_q$",
    "gamma1_low": r"$\gamma_1-\Delta\gamma_1$", "gamma1_high": r"$\gamma_1+\Delta\gamma_1$",
    "dephasing_low": r"$\gamma_\varphi-\Delta\gamma_\varphi$",
    "dephasing_high": r"$\gamma_\varphi+\Delta\gamma_\varphi$",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def completed_artifact(manifest, role):
    """Select a recorded success, never a failed attempt or an unrecorded file."""
    records = [r for k, r in manifest["steps"].items()
               if k.startswith(role+":") and r.get("status") == "complete"
               and Path(r["output"]).suffix == ".npz"]
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one completed NPZ for {role}; got {len(records)}")
    return records[0]


def load_work_spectrum(manifest, origin, verify, *, historical=False, figures=None):
    """Native runs use their S1 step; old repairs require explicit selection."""
    receipt_path = origin/"s1_repaired/rerun_receipt.json"
    if historical and receipt_path.is_file():
        receipt = json.loads(verify(receipt_path).read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or receipt["original_manifest"]["sha256"] != sha(origin/"manifest.json"):
            raise RuntimeError("S1 repair is incomplete or belongs to a different run")
        for path, digest in receipt["artifacts"].items():
            verify(path, digest)
        source = verify(origin/"s1_repaired/supp01_work_spectrum.npz")
        if figures is not None:
            shutil.copyfile(verify(origin/"s1_repaired/supp01.png"), figures/"supp01.png")
        description = "explicit historical s1_repaired receipt"
    else:
        record = completed_artifact(manifest, "supp01_work_spectrum")
        source = verify(record["output"], record["sha256"])
        description = "native completed supp01_work_spectrum manifest step"
    with np.load(source, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}, description


def check_source_identity(manifest, verify, *, historical=False):
    """A new run must match the code used to compute it; archives report drift."""
    mismatches = []
    recorded = manifest["identity"]["source_sha256"]
    if not recorded:
        raise RuntimeError("Run has no recorded calculation-source hashes")
    for relative, digest in recorded.items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            mismatches.append(relative)
        else:
            verify(path, digest)
    if mismatches and not historical:
        raise RuntimeError("Calculation sources changed since this run: "+", ".join(mismatches))
    return {"accepted": not mismatches, "historical_archive": historical,
            "recorded_source_count": len(recorded), "mismatched_sources": mismatches}


def shape_artifact(row, manifest):
    """Use the accepted artifact recorded by validation, not a directory glob."""
    path = row.get("shape_spectrum_artifact")
    if path is None:
        paths = row.get("shape_spectrum_artifacts", [])
        path = paths[0] if paths else None
    if path is None:
        return completed_artifact(manifest, "check_shape_"+row["label"])
    matches = [r for r in manifest["steps"].values()
               if r.get("status") == "complete" and Path(r["output"]).resolve() == Path(path).resolve()]
    if len(matches) != 1:
        raise RuntimeError("Shape validation artifact is absent/ambiguous in manifest: "+str(path))
    return matches[0]


def audit_material_fit(data):
    """Independent between-node reconstruction from the saved table and poles."""
    request = json.loads(str(data["metadata_json"].item()))["requested_arguments"]
    c_nm, a_nm, eps_m = request["c_nm"], request["a_nm"], request["eps_m"]
    low, high = request["fit_min_ev"], request["fit_max_ev"]
    audit_e = np.linspace(low+(high-low)*1e-5, high-(high-low)*1e-5, 8003)
    eps = (np.interp(audit_e, data["material_energy_eV"], data["material_n"])
           +1j*np.interp(audit_e, data["material_energy_eV"], data["material_k"]))**2
    ecc = np.sqrt(1-(a_nm/c_nm)**2)
    lz = (1-ecc**2)/(2*ecc**3)*(np.log((1+ecc)/(1-ecc))-2*ecc)
    channels = list(data["channel_id"].astype(str))
    result = {}
    for orientation, dep, channel in (("long", lz, "axis_long"), ("trans", (1-lz)/2, "axis_trans")):
        ci = channels.index(channel)
        mask = data["fit_mode_mask"][1, ci].astype(bool)
        target = (eps-eps_m)/(eps_m+dep*(eps-eps_m))
        fitted = data["fit_alpha_inf"][1, ci]+np.sum(data["fit_strengths_eV2"][1, ci, mask]/
            (data["fit_omega_modes_eV"][1, ci, mask]**2-audit_e[:, None]**2
             -1j*audit_e[:, None]*data["fit_gamma_modes_eV"][1, ci, mask]), axis=1)
        metrics = dict(max_relative=float(np.max(abs((fitted-target)/target))),
                       nrms=float(np.linalg.norm(fitted-target)/np.linalg.norm(target)),
                       inverse_nrms=float(np.linalg.norm(1/fitted-1/target)/np.linalg.norm(1/target)),
                       min_imag=float(fitted.imag.min()), points=len(audit_e))
        metrics["accepted"] = bool(metrics["max_relative"] <= request["max_bright_fit_pointwise_relative_error"]
            and max(metrics["nrms"], metrics["inverse_nrms"]) <= request["max_bright_fit_normalized_rms"]
            and metrics["min_imag"] >= 0)
        result[orientation] = metrics
    return result


def sensitivity_table(validation, threshold, selection, tables):
    """Export measured values/statuses, including unresolved or absent checks."""
    channels = ["side_long", "side_trans_radial"]
    nominal_channels = list(threshold["channel_id"].astype(str))
    nominal_gap = float(selection["best_gap_nm"])
    gi = int(np.flatnonzero(np.isclose(threshold["gap_nm"], nominal_gap, rtol=0, atol=1e-12))[0])
    nominal = [float(threshold["threshold_fluence_j_cm2"][1, nominal_channels.index(c), gi])*1e6
               for c in channels]

    def formatted(value):
        return "---" if value is None or not np.isfinite(value) else f"{value:.3f}".replace(".", ",")

    lines = [r"\begin{tabular}{lrr}\toprule",
             r"Вариация & Боковой продольный & Боковой радиальный\\\midrule",
             "Номинальные параметры & "+" & ".join(map(formatted, nominal))+r"\\"]
    by_label = {r["label"]: r for r in validation}
    rows = []
    for label, title in SENSITIVITY_LABELS.items():
        row = by_label.get(label, {})
        values, statuses = [], []
        row_channels = row.get("channels", channels)
        row_gaps = np.asarray(row.get("gaps_nm", [nominal_gap]), dtype=float)
        gap_index = int(np.argmin(abs(row_gaps-nominal_gap)))
        for channel in channels:
            index = row_channels.index(channel) if channel in row_channels else None
            raw = row.get("candidate_thresholds")
            value = None if raw is None or index is None else raw[index][gap_index]
            status = row.get("candidate_status")
            statuses.append("not_performed" if status is None or index is None else status[index][gap_index])
            values.append(None if value is None else float(value)*1e6)
        lines.append(title+" & "+" & ".join(map(formatted, values))+r"\\")
        rows.append(dict(label=label, accepted=row.get("accepted", False),
                         side_long_uJ_cm2=values[0], side_radial_uJ_cm2=values[1],
                         side_long_status=statuses[0], side_radial_status=statuses[1],
                         error=row.get("error")))
    lines.extend([r"\bottomrule\end{tabular}", ""])
    (tables/"sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")
    with (tables/"sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/article_single_mnp_c12p5_a4p5")
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript_c12p5_a4p5")
    parser.add_argument("--sensitivity", type=Path,
                        help="Explicit legacy repair supplement; not needed for a new complete run")
    args = parser.parse_args()
    origin, dest = args.run.resolve(), args.output.resolve()
    figures, tables = dest / "figures", dest / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    manifest_path = origin / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks, sources = {}, {}

    def verify(path, digest=None):
        path = Path(path)
        actual = sha(path)
        if digest is not None and actual != digest:
            raise RuntimeError(f"Artifact hash mismatch: {path}")
        try:
            source_name = str(path.resolve().relative_to(ROOT))
        except ValueError:
            source_name = str(path.resolve())
        sources[source_name] = actual
        return path

    complete = [r for r in manifest["steps"].values() if r.get("status") == "complete"]
    for record in complete:
        verify(record["output"], record["sha256"])
    verify(manifest_path)
    checks["original_completed_artifacts_verified"] = len(complete)
    if args.sensitivity is None and manifest.get("status") != "complete":
        raise RuntimeError("Ordinary export requires a complete run; use --sensitivity only for a legacy repaired archive")
    checks["calculation_source_identity"] = check_source_identity(
        manifest, verify, historical=args.sensitivity is not None)

    def load(role):
        record = completed_artifact(manifest, role)
        with np.load(verify(record["output"], record["sha256"]), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def close(label, a, b, **kwargs):
        np.testing.assert_allclose(a, b, **kwargs)
        checks[label] = True

    material, spec, threshold = load("fig01_material"), load("fig02_master"), load("fig03_master")
    comparison, near, far = load("fig05_material_fluence"), load("fig06_dynamics_near"), load("fig06_dynamics_far")
    e, gaps = spec["energy_eV"], spec["gap_nm"]
    close("geometry_gaps_consistent", gaps, threshold["gap_nm"], atol=0, rtol=0)
    config = manifest["identity"]["config"]
    assert config["geometry"]["c_nm"] == 12.5 and config["geometry"]["a_nm"] == 4.5
    checks["requested_geometry"] = True
    # Independent two-level susceptibility in the declared external-dipole convention.
    qd = config["qd"]
    qd_energy = qd["transition_energy_eV"]
    eps_m = config["medium"]["relative_permittivity"]
    d = qd["effective_dipole_debye"] * DEBYE_C_M / AU_DIPOLE_C_M
    gamma_eV = qd["pure_dephasing_energy_meV"]*1e-3 + qd["population_decay_energy_neV"]*1e-9/2
    w0, w, gamma = qd_energy/AU_ENERGY_EV, e/AU_ENERGY_EV, gamma_eV/AU_ENERGY_EV
    beta = 2*d*d*w0/(w0*w0+(gamma-1j*w)**2)
    response = beta*(1+spec["interaction_B"])/(1-beta*spec["interaction_K_au_minus3"])
    close("linear_transfer_from_B_K", response, spec["qd_dipole_over_field_au3"], rtol=2e-8, atol=1e-8)
    close("linear_spectrum_from_transfer", abs(response)**2, spec["qd_excitation_spectrum"], rtol=4e-8)
    feature_mask = abs(e-config["spectrum"]["feature_center_eV"]) <= config["spectrum"]["feature_half_window_eV"]
    dd, fqs = spec["qd_excitation_spectrum"][0][...,feature_mask], spec["qd_excitation_spectrum"][1][...,feature_mask]
    delta = np.sqrt(np.trapezoid((dd-fqs)**2, e[feature_mask], axis=-1)
                    / np.trapezoid(fqs**2, e[feature_mask], axis=-1))
    close("spectral_discrepancy_from_unnormalized_spectra", delta, spec["spectral_l2_relative_dd_vs_fqs"], rtol=1e-12)
    ft = threshold["threshold_fluence_j_cm2"]
    close("threshold_efficiency", threshold["isolated_threshold_fluence_j_cm2"]/ft,
          threshold["fluence_efficiency_isolated_over_hybrid"], rtol=1e-12, equal_nan=True)
    close("threshold_discrepancy", abs(ft[0]/ft[1]-1), threshold["absolute_threshold_discrepancy_dd_vs_fqs"], rtol=1e-12, equal_nan=True)
    for name, dynamics in (("near", near), ("far", far)):
        close(name+"_density_from_W", (1+dynamics["W"])/2, dynamics["rho22"], rtol=0, atol=1e-15)
        close(name+"_final_population", dynamics["rho22"][:, -1], dynamics["population_final"], rtol=0, atol=1e-15)

    rows = []
    for mi, model in enumerate(spec["model_id"]):
        for ci, channel in enumerate(spec["channel_id"]):
            for gi, gap in enumerate(gaps):
                rows.append(dict(model=model, channel=channel, gap_nm=gap,
                    gain_fixed=spec["excitation_gain_at_reference_energy"][mi,ci,gi],
                    gain_optimized=spec["excitation_gain_optimized"][mi,ci,gi],
                    shift_over_gamma0=spec["resonance_shift_over_gamma0"][mi,ci,gi],
                    width_over_gamma0=spec["spectral_width_over_gamma0"][mi,ci,gi],
                    threshold_uJ_cm2=ft[mi,ci,gi]*1e6,
                    threshold_status=threshold["threshold_status"][mi,ci,gi],
                    weak_pulse_population_gain=threshold["weak_field_population_gain"][mi,ci,gi],
                    delta_spectrum=delta[ci,gi],
                    delta_threshold=threshold["absolute_threshold_discrepancy_dd_vs_fqs"][ci,gi]))
    with (tables/"all_channels_and_gaps.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(clean(rows))

    for role, relative in manifest["figures"].items():
        source = verify(origin/relative)
        shutil.copyfile(source, figures/(role+".png"))
    s1, checks["work_spectrum_source"] = load_work_spectrum(
        manifest, origin, verify, historical=args.sensitivity is not None, figures=figures)
    close("work_spectrum_background_subtraction", s1["sigma_qs_work_cm2"]-s1["bare_mnp_sigma_qs_work_cm2"][:,:,None,:],
          s1["delta_sigma_qs_work_cm2"], rtol=0, atol=1e-25)
    assert np.all(s1["work_nonnegative_within_tolerance"])
    checks["S1_integrated_work_nonnegative"] = True
    # Independently reconstruct the linear response on the FINAL S1 energy grid.
    # Use its saved N12 coefficients and both unreduced and reduced spatial
    # weights. This does not depend on the earlier diagnostic time trace.
    selection = manifest["state"]["selection"]
    s1_channel = str(s1["channel_key"][0])
    ci = list(spec["channel_id"]).index(s1_channel)
    gi = int(np.flatnonzero(np.isclose(gaps, selection["best_gap_nm"], rtol=0, atol=1e-12))[0])
    orientation_index = 0 if s1_channel.endswith("long") else 1
    sw = s1["energy_eV"]/AU_ENERGY_EV
    h = s1["material_fit_alpha_inf_dimensionless"][1,0]+np.sum(
        s1["material_fit_strengths_au2"][1,0]/
        (s1["material_fit_omega_modes_au"][1,0]**2-sw[:,None]**2
         -1j*sw[:,None]*s1["material_fit_gamma_modes_au"][1,0]),axis=-1)
    interaction_j = spec["interaction_B"][1,ci,gi]/spec["interaction_A_au3"][1,ci,gi]
    np.testing.assert_allclose(interaction_j,interaction_j[0],rtol=1e-12)
    a_fit = material["polarizability_scale_au3"][orientation_index]*h
    b_fit = a_fit*interaction_j[0]
    s_beta = 2*d*d*w0/(w0*w0+(gamma-1j*sw)**2)
    numerical_alpha = s1["alpha_eff_au3_real"][1,0,0]+1j*s1["alpha_eff_au3_imag"][1,0,0]
    final_s1_linear_audit = {}
    for prefix in ("spatial", "dynamic_spatial"):
        start, end = threshold[prefix+"_bundle_offsets"][ci*len(gaps)+gi:ci*len(gaps)+gi+2]
        key = "spatial_mode" if prefix == "spatial" else prefix
        dep = threshold[key+"_depolarization"][start:end]
        weights = threshold[key+"_reaction_weight_au_minus3"][start:end]
        k_fit = np.sum(weights*h[:,None]/(1+(dep-material["depolarization_factor"][orientation_index])*h[:,None]),axis=-1)
        reference_alpha = (a_fit+s_beta*(1+b_fit)**2/(1-s_beta*k_fit))/eps_m
        metrics = dict(points=len(sw), spatial_modes=len(weights),
            complex_max_normalized=float(np.max(abs(numerical_alpha-reference_alpha))/np.max(abs(reference_alpha))),
            complex_nrms=float(np.linalg.norm(numerical_alpha-reference_alpha)/np.linalg.norm(reference_alpha)),
            imag_max_normalized=float(np.max(abs(numerical_alpha.imag-reference_alpha.imag))/np.max(abs(reference_alpha.imag))))
        assert metrics["imag_max_normalized"] < .001
        final_s1_linear_audit[prefix] = metrics
    checks["final_S1_weak_spectrum_vs_independent_linear_solution"] = True
    mask = abs(s1["energy_eV"]-qd_energy) <= .01
    narrow = s1["delta_sigma_qs_work_cm2"][:,:,:,mask]

    validation = manifest["state"]["validation"]
    ranking = manifest["state"]["ranking_validation"]
    independent_fit_audit = {}
    if args.sensitivity:
        supplement = args.sensitivity.resolve()
        sr = json.loads(verify(supplement/"rerun_receipt.json").read_text(encoding="utf-8"))
        assert sr["original_manifest"]["sha256"] == sha(manifest_path)
        assert sr["status"] == "complete"
        for path, digest in sr["artifacts"].items():
            verify(path, digest)
        supplement_manifest = json.loads((supplement/"manifest.json").read_text(encoding="utf-8"))
        for relative, digest in supplement_manifest["identity"]["source_sha256"].items():
            verify(ROOT/relative, digest)
        verify(ROOT/"scripts/rerun_article_sensitivity.py",sr["script_sha256"])
        checks["current_calculation_sources_match_final_supplement"] = True
        inherited = sr.get("inherited_validation")
        if inherited:
            previous = Path(inherited["path"]).parent/"rerun_receipt.json"
            pr = json.loads(verify(previous, inherited["receipt_sha256"]).read_text(encoding="utf-8"))
            for path, digest in pr["artifacts"].items():
                verify(path, digest)
        validation = json.loads((supplement/"validation.json").read_text(encoding="utf-8"))
        ranking = json.loads((supplement/"ranking_validation.json").read_text(encoding="utf-8"))
        for name in ("validation_sensitivity.png", "validation_carrier.png"):
            shutil.copyfile(verify(supplement/"figures"/name), figures/name)
    sensitivity_table(validation, threshold, selection, tables)
    by_label = {r["label"]: r for r in validation}
    for label in ("c_low", "c_high", "a_low", "a_high"):
        row = by_label.get(label)
        if row is None or not row.get("accepted"):
            independent_fit_audit[label] = {"status": "not_performed", "reason": "No accepted shape validation"}
            continue
        # The legacy receipt can inherit shape spectra from a prior supplement.
        shape_manifest = manifest
        if args.sensitivity is not None:
            shape_manifest_path = Path(row["artifact"]).parent.parent/"manifest.json"
            shape_manifest = json.loads(verify(shape_manifest_path).read_text(encoding="utf-8"))
        record = shape_artifact(row, shape_manifest)
        with np.load(verify(record["output"], record["sha256"]), allow_pickle=False) as data:
            independent_fit_audit[label] = audit_material_fit(data)
    completed_fit_audits = [metrics for case in independent_fit_audit.values()
                           for metrics in case.values() if isinstance(metrics, dict) and "accepted" in metrics]
    checks["new_material_fits_independent_8003_point_grid"] = {
        "performed_orientations": len(completed_fit_audits),
        "accepted": bool(completed_fit_audits) and all(m["accepted"] for m in completed_fit_audits)}

    plt.rcParams.update({"font.size":10, "axes.grid":True, "grid.alpha":.2})
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for ci, label in enumerate(CHANNELS_RU):
        axs[0].plot(gaps, spec["excitation_gain_at_reference_energy"][1,ci], 'o-', ms=3, label=label)
        axs[1].plot(gaps, threshold["weak_field_population_gain"][1,ci], 'o-', ms=3, label=label)
    for ax in axs:
        ax.axhline(1, color='k', lw=.7, ls='--'); ax.set_xlabel("Зазор g, нм"); ax.set_yscale('log')
    axs[0].set_ylabel(r"$G_{\rm fixed}$ при 2,042 эВ")
    axs[1].set_ylabel("Усиление слабой импульсной заселённости")
    axs[0].legend(fontsize=8)
    fig.savefig(figures/"fixed_and_pulse_gain.pdf"); plt.close(fig)
    # Optional archival diagnostics are never prerequisites of a fresh export.
    controls = {"units.json": json.loads(verify(origin/"units.json").read_text(encoding="utf-8"))}
    for name in ("s1_repaired/validation_summary.json", "s1_repaired/weak_linear_audit.json",
                 "tip_side_audit/audit.json", "tip_side_audit/identity_checks.json",
                 "tip_side_audit/frequency_fit_check.json", "tip_side_audit/threshold_comparison.json",
                 "tip_side_audit/threshold_receipt.json"):
        path = origin/name
        controls[name] = (json.loads(verify(path).read_text(encoding="utf-8"))
                          if args.sensitivity is not None and path.is_file()
                          else {"status": "not_performed", "scope": "optional historical diagnostic"})
    checks["regression_tests"] = {"status": "not_performed_by_exporter",
                                  "reason": "Run the current test suite separately; no historical count is reused"}
    weak_audit = controls["s1_repaired/weak_linear_audit.json"]
    if "trace_sha256" in weak_audit:
        verify(origin/"s1_fixed/weak_trace_diagnostic.npz", weak_audit["trace_sha256"])
        weak_audit["audited_trace"] = "s1_fixed/weak_trace_diagnostic.npz"
    if "tip_side_fields" in manifest["figures"]:
        from qdmnp.observables.article_field_diagnostic import build_field_diagnostic
        geometry = config["geometry"]
        interp = build_field_diagnostic(c_nm=geometry["c_nm"], a_nm=geometry["a_nm"],
            eps_m=eps_m, carrier_energy_eV=config["pulse"]["carrier_energy_eV"],
            qd_radius_nm=geometry["qd_radius_nm"], gap_nm=selection["best_gap_nm"])["interpretation"]
    elif args.sensitivity is not None and (origin/"tip_side_audit/interpretation_numbers.json").is_file():
        interp = json.loads(verify(origin/"tip_side_audit/interpretation_numbers.json").read_text(encoding="utf-8"))
        shutil.copyfile(verify(origin/"tip_side_audit/tip_side_fields.png"), figures/"tip_side_fields.png")
    else:
        interp = {"status": "not_performed", "reason": "No recorded field-diagnostic figure"}
    preflight = {}
    preflight_roles = sorted({key.split(":", 1)[0] for key, record in manifest["steps"].items()
                              if key.startswith("preflight_g") and record.get("status") == "complete"})
    for role in preflight_roles:
        gap = role.removeprefix("preflight_g")
        data = load(role)
        preflight[gap] = {k:data[k] for k in ("observable_fit_accepted", "observable_spectrum_nrms_error_vs_direct",
            "observable_shift_error_over_isolated_fwhm", "observable_width_relative_error_vs_direct",
            "observable_gain_relative_error_vs_direct")}
    sensitivity_rows = [{k:r.get(k) for k in ("label","accepted","candidate_thresholds","candidate_status","error")} for r in validation]
    evidence = dict(
        scope=("Saved c12.5 a4.5 article results; explicitly selected historical supplement"
               if args.sensitivity else "Fresh c12.5 a4.5 native article run; no historical diagnostics required"),
        original_run_status=manifest["status"], original_manifest_sha256=sha(manifest_path),
        checks=checks, config=config,
        material={k:material[k] for k in ("orientation_ids","branch_ids","depolarization_factor","nrms_alpha","nrms_inverse_alpha","lspr_peak_energy_eV","lspr_fwhm_eV","lspr_status")},
        channels=spec["channel_id"], gaps_nm=gaps, isolated_linewidth_eV=spec["isolated_fwhm_eV"],
        near_gap_rows=[r for r in rows if r["gap_nm"]==selection["best_gap_nm"]],
        isolated_threshold_uJ_cm2=float(threshold["isolated_threshold_fluence_j_cm2"]*1e6),
        dd_spectral_boundary={k:spec[k] for k in ("dd_validity_gap_nm","dd_validity_status","dd_validity_supporting_points")},
        dd_threshold_boundary={k:threshold[k] for k in ("dd_validity_gap_nm","dd_validity_status","dd_validity_supporting_points")},
        material_thresholds_uJ_cm2=comparison["threshold_fluence_j_cm2"]*1e6,
        material_threshold_status=comparison["threshold_status"],
        near_population=near["population_final"], far_population=far["population_final"],
        dynamics_channels=near["channel_ids"],
        work_spectrum_delta_min_cm2=narrow.min(axis=-1),work_spectrum_delta_max_cm2=narrow.max(axis=-1),
        work_j=s1["work_from_incident_field_j"], work_effective_area_cm2=s1["sigma_energy_transfer_cm2"],
        field_interpretation=interp, sensitivity=sensitivity_rows, ranking=ranking,
        independent_refitted_material_audit=independent_fit_audit,
        final_s1_linear_audit=final_s1_linear_audit,
        controls=controls, preflight_observables=preflight,
        source_sha256=sources, preparation_script_sha256=sha(__file__),
    )
    (tables/"evidence.json").write_text(json.dumps(clean(evidence), ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(dest), "checks":checks, "ranking_accepted":ranking["accepted"],
                      "sources":len(sources)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
