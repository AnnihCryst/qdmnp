r"""Run the single-MNP article chain: inputs -> native kernels -> NPZ -> figures.

    .venv\Scripts\python.exe run_article.py
    .venv\Scripts\python.exe run_article.py --dry-run
    .venv\Scripts\python.exe run_article.py --smoke

No equations are copied here. The existing article calculators call the DD
and FQS kernels; the runner handles dependencies, selection and validation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time
import traceback

# Small least-squares matrices otherwise incur excessive BLAS thread overhead.
# Respect an explicit user choice; apply before importing NumPy/SciPy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import scipy
from scipy.constants import c, hbar, elementary_charge

from article_observables.qd_mnp_article_inputs import (
    DEFAULT_INPUT, ROOT, load_inputs, physical_arguments, unit_audit,
)
from article_observables.qd_mnp_article_fit_cache import material_fit_cache
from article_observables.qd_mnp_article_fit_identity import compare_fit_coefficients
from article_observables.qd_mnp_threshold_metrics import resolved_threshold_mask


PREFIX = "article_observables.qd_mnp_"
KERNELS = {
    "DD_time": "qd_mnp_rational_fit.HybridQDPlasmonModel",
    "FQS_time": "qd_mnp_full_qs_model.FullQSSpheroidPulseModel",
    "FQS_tip": "qd_mnp_spheroid_green.SpheroidGreenInteraction",
    "FQS_side": "qd_mnp_spheroid_equatorial.EquatorialSpheroidGreenInteraction",
}
STAGES = (
    "material", "spectra", "preflight", "thresholds", "selection",
    "validation", "material_effect", "dynamics", "work_spectrum", "report",
)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(clean_json(document), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flags(arguments: dict) -> list[str]:
    result = []
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        result.append("--" + key)
        if value is not True:
            result.extend(str(x) for x in (value if isinstance(value, (list, tuple, np.ndarray)) else [value]))
    return result


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files if name != "metadata_json"}, json.loads(str(archive["metadata_json"].item()))


class StageError(RuntimeError):
    pass


@contextmanager
def diagnostic_figure_label(enabled):
    """Mark every standalone smoke figure at save time, including legacy plots."""
    if not enabled:
        yield
        return
    from matplotlib.figure import Figure
    original = Figure.savefig
    def save(figure, *args, **kwargs):
        label = figure.text(.5, .005, "SMOKE: technical check; unconverged data",
                            ha="center", va="bottom", fontsize=9, color="darkred",
                            bbox={"facecolor":"white", "edgecolor":"darkred", "alpha":.9})
        try:
            return original(figure, *args, **kwargs)
        finally:
            label.remove()
    Figure.savefig = save
    try:
        yield
    finally:
        Figure.savefig = original


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.terminal.write(value)

    def flush(self):
        self.log.flush()
        self.terminal.flush()


class ArticleRun:
    def __init__(self, config: dict, directory: Path, *, stop_after: str | None = None):
        self.config = deepcopy(config)
        self.directory = directory.resolve()
        self.stop_after = stop_after
        self.directory.mkdir(parents=True, exist_ok=True)
        sources = sorted([ROOT / "run_article.py", *ROOT.glob("qd_mnp*.py"), * (ROOT / "article_observables").glob("*.py")])
        self.identity = {"config": config, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                         "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
        self.fingerprint = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.manifest_path = self.directory / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["fingerprint"] != self.fingerprint:
                raise ValueError("Inputs, source code or library versions changed. Use a NEW --output directory; existing results are preserved.")
        else:
            self.manifest = {"fingerprint": self.fingerprint, "identity": self.identity, "kernels": KERNELS,
                             "smoke": config["smoke"], "status": "in_progress", "steps": {}, "figures": {}}
        self.state = self.manifest.setdefault("state", {})
        self.audit = unit_audit(config)
        self.material_n = config["material"]["mode_candidates"][0]
        self.spatial_order = config["numerics"]["spatial_order"]
        self.energy_points = config["spectrum"]["points"]
        self.fluence_points = config["pulse"]["fluence_points"]
        write_json(self.directory / "resolved_inputs.json", self.identity)
        write_json(self.directory / "units.json", self.audit)
        self.save()

    @property
    def policy(self):
        return "warn" if self.config["smoke"] else "raise"

    def save(self):
        write_json(self.manifest_path, self.manifest)

    def step(self, label, module, arguments=(), *, suffix=".npz", dependencies=()):
        arguments = flags(arguments) if isinstance(arguments, dict) else list(map(str, arguments))
        signature = hashlib.sha256(json.dumps([module, arguments, [(str(p), sha(Path(p))) for p in dependencies]], sort_keys=True).encode()).hexdigest()
        path = self.directory / ("figures" if suffix == ".png" else "data") / f"{label}_{signature[:10]}{suffix}"
        key = f"{label}:{signature}"
        record = self.manifest["steps"].get(key, {})
        # E.g. the carrier scan includes its reference carrier: reuse the same
        # completed calculation even when its presentation label is different.
        if not record and suffix == ".npz":
            for previous_key, previous in self.manifest["steps"].items():
                if previous_key.endswith(":"+signature) and previous.get("status") == "complete":
                    original = Path(previous["output"])
                    if not original.exists() or sha(original) != previous["sha256"]:
                        raise StageError(f"Completed artifact is missing or changed: {original}")
                    print(f"Reuse identical calculation: {label}", flush=True)
                    return original
        if record.get("status") == "complete":
            if not path.exists() or record.get("sha256") != sha(path):
                raise StageError(f"Completed artifact is missing or changed: {path}")
            print(f"Resume: {label}", flush=True)
            return path
        if path.exists():
            raise StageError(f"Uncertified/unrecorded output already exists: {path}; preserve it and use a new output directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        log = self.directory / "logs" / f"{label}_{signature[:10]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = [module, *arguments, "--output", str(path)]
        self.manifest["steps"][key] = {"status": "running", "command": [sys.executable, "-m", module, *argv[1:]], "output": str(path), "log": str(log)}
        self.save()
        started = time.perf_counter()
        old_argv = sys.argv
        print(f"\nRunning {label} -> {path.name}", flush=True)
        try:
            with log.open("w", encoding="utf-8") as stream, redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                sys.argv = argv
                try:
                    runpy.run_module(module, run_name="__main__")
                except SystemExit as exc:
                    if exc.code not in (None, 0):
                        raise StageError(f"CLI exited with {exc.code}") from exc
            if not path.exists():
                raise StageError(f"Stage did not create its output: {path}")
        except (Exception, KeyboardInterrupt) as exc:
            self.manifest["steps"][key].update(status="failed", error=str(exc), elapsed_s=time.perf_counter()-started)
            self.save()
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise StageError(f"{label}: {exc}; see {log}") from exc
        finally:
            sys.argv = old_argv
        self.manifest["steps"][key].update(status="complete", sha256=sha(path), elapsed_s=time.perf_counter()-started)
        if suffix == ".png":
            self.manifest["figures"][label] = str(path.relative_to(self.directory))
        self.save()
        return path

    def common(self, config=None):
        return physical_arguments(config or self.config)

    def fit_args(self, *, mode_key="material-fit-modes", count=None):
        m, n = self.config["material"], self.config["numerics"]
        return {mode_key: count or self.material_n, "fit-min-ev": m["fit_min_eV"], "fit-max-ev": m["fit_max_eV"],
                "fit-refinement": json.dumps(m["refinement"], sort_keys=True) if m.get("refinement") else None,
                "max-bright-fit-normalized-rms": m["max_bright_nrms"], "max-bright-fit-pointwise-relative-error": m["max_bright_pointwise_error"],
                "spatial-order-max": self.spatial_order, "spatial-convergence-rtol": n["spatial_rtol"],
                "modal-audit-points": n["modal_audit_points"], "max-modal-normalized-rms": n["max_modal_nrms"],
                "max-modal-relative-error": n["max_modal_relative_error"], "spatial-convergence-policy": self.policy,
                "fit-quality-policy": self.policy, "radiative-consistency-policy": "warn"}

    def spectral_args(self):
        s = self.config["spectrum"]
        return {"energy-min-ev": s["min_eV"], "energy-max-ev": s["max_eV"], "energy-points": self.energy_points,
                "feature-center-ev": s["feature_center_eV"], "feature-half-window-ev": s["feature_half_window_eV"],
                "energy-resolution-policy": self.policy, "max-spectral-coarsening-change": s["max_coarsening_change"],
                "max-spectral-window-change": s["max_window_change"]}

    def temporal_args(self, *, dynamics=False, config=None):
        cfg = config or self.config
        p, n = cfg["pulse"], cfg["numerics"]
        result = {"pulse-tau-fs": p["intensity_fwhm_fs"], "pulse-tau-kind": "fwhm_intensity", "post-fs": p["population_read_fs"],
                  "method": n["method"], "rtol": n["rtol"], "atol": n["atol"], "points-per-fastest-cycle": n["points_per_fastest_cycle"],
                  "spectral-window-policy": self.policy, "max-spectral-leakage": n["max_spectral_leakage"],
                  "positivity-policy": "raise", "work-passivity-policy": "raise", "population-decay-policy": self.policy,
                  "max-population-decay-fraction-at-read": p["max_population_decay_fraction"],
                  "bright-fit-quality-policy": self.policy}
        result.update({"response-tail-policy" if dynamics else "tail-policy": self.policy,
                       "response-tail-tolerance" if dynamics else "tail-ratio-tolerance": n["tail_ratio_tolerance"],
                       "response-tail-window-fraction" if dynamics else "tail-window-fraction": n["tail_window_fraction"]})
        return result

    def plot(self, label, kind, data, extra=None):
        extra = dict(extra or {})
        if self.config["smoke"] and kind in {"excitation_spectrum_material_comparison", "excitation_fluence_material_comparison", "population_dynamics", "work_loss_spectrum_material_comparison"}:
            extra["allow-unconverged"] = True
        argv = (["--input", str(data)] if kind == "excitation_spectrum_material_comparison" else [str(data)])
        return self.step(label, PREFIX + "plot_" + kind, [*argv, *flags(extra), "--dpi", str(self.config["output"]["dpi"])], suffix=".png", dependencies=[data])

    def audit_fit_identity(self, path):
        reference, ref_meta = read_npz(self.state["material_artifact"])
        arrays, metadata = read_npz(path)
        result = compare_fit_coefficients(reference, ref_meta, arrays, metadata)
        self.state.setdefault("material_coefficient_identity", {})[str(path)] = result

    def material(self):
        m, g = self.config["material"], self.config["geometry"]
        errors = []
        for count in m["mode_candidates"]:
            if count < self.state.get("material_modes", count):
                continue  # Earlier rejected candidates need not be fitted again on resume.
            try:
                data = self.step("fig01_material", PREFIX+"calculate_material_dispersion_comparison", {
                    "c-nm": g["c_nm"], "a-nm": g["a_nm"], "eps-m": self.config["medium"]["relative_permittivity"],
                    "fit-refinement": json.dumps(m["refinement"], sort_keys=True) if m.get("refinement") else None,
                    "fit-min-ev": m["fit_min_eV"], "fit-max-ev": m["fit_max_eV"], "energy-min-ev": m["fit_min_eV"], "energy-max-ev": m["fit_max_eV"],
                    "one-modes": 1, "multi-modes": count, "seed": m["seed"], "max-normalized-rms": m["max_bright_nrms"], "max-pointwise-relative-error": m["max_bright_pointwise_error"]})
                self.material_n = count
                self.state.update(material_modes=count, material_artifact=str(data))
                self.plot("fig01_material", "material_dispersion_comparison", data)
                return
            except StageError as exc:
                if not any(word in str(exc).lower() for word in ("fit", "accuracy", "tabulated points")):
                    raise
                errors.append(str(exc))
        raise StageError("No configured material representation passed: " + "\n".join(errors))

    def spectra(self):
        while True:
            try:
                master = self.step("fig02_master", PREFIX+"calculate_excitation_gain_gap", {
                    "preset": "publication", **self.common(), **self.fit_args(), **self.spectral_args(),
                    "gaps-nm": self.config["geometry"]["gaps_nm"], "channels": self.config["geometry"]["channels"],
                    "max-energy-step-over-gamma0": self.config["spectrum"]["max_step_over_width"],
                    "spectral-observable": "linear_qd_response", "dd-tolerance": self.config["numerics"]["dd_tolerance"]})
                break
            except StageError as exc:
                if not self.refine_spectral_failure(exc):
                    raise
        self.state["spectral_master"] = str(master)
        for letter, metric in zip("abcd", ("excitation_gain", "resonance_shift", "spectral_width", "model_discrepancy")):
            data = master if letter == "a" else self.step("fig02"+letter+"_data", PREFIX+"calculate_"+metric+"_gap", {"source-artifact": master}, dependencies=[master])
            self.plot("fig02"+letter, metric+"_gap", data)
        self.state.update(energy_points=self.energy_points, spatial_order=self.spatial_order)

    def refine_spectral_failure(self, exc):
        message = str(exc).lower()
        if any(x in message for x in ("sampling", "coarsening", "energy-grid", "spectral grid")) and self.energy_points < self.config["spectrum"]["max_points"]:
            self.energy_points = min(2*self.energy_points-1, self.config["spectrum"]["max_points"])
            return True
        if "spatial" in message and self.spatial_order < self.config["numerics"]["max_spatial_order"]:
            self.spatial_order = min(2*self.spatial_order, self.config["numerics"]["max_spatial_order"])
            return True
        return False

    def material_spectrum(self, label, gap, *, count=None, config=None):
        s = self.config["spectrum"]
        fit = self.fit_args(mode_key="multi-fit-modes", count=count)
        # This calculator checks multi-mode fit accuracy unconditionally;
        # only its ONE-mode diagnostic branch has an accuracy policy switch.
        fit.pop("fit-quality-policy")
        return self.step(label, PREFIX+"calculate_excitation_spectrum_material_comparison", {
            "preset": "publication", **self.common(config), **fit, **self.spectral_args(),
            "channels": self.config["geometry"]["channels"], "gap-nm": gap,
            "max-energy-step-over-isolated-fwhm": s["max_step_over_width"], "observable-fit-policy": self.policy,
            "max-observable-spectrum-nrms": s["max_observable_nrms"], "max-observable-shift-over-isolated-fwhm": s["max_observable_shift_over_gamma0"],
            "max-observable-width-relative-error": s["max_observable_width_relative_error"], "max-observable-gain-relative-error": s["max_observable_gain_relative_error"]})

    def preflight(self):
        gaps = self.config["geometry"]["gaps_nm"]
        preflight_gaps = sorted(set([gaps[0], gaps[len(gaps)//2], gaps[-1]]))
        while True:
            try:
                paths = [self.material_spectrum("preflight_g"+str(g), g) for g in preflight_gaps]
                self.state.update(preflight=[str(p) for p in paths], material_modes=self.material_n,
                                  spatial_order=self.spatial_order, energy_points=self.energy_points)
                for path in paths:
                    self.audit_fit_identity(path)
                return
            except StageError as exc:
                if self.refine_spectral_failure(exc):
                    continue
                candidates = self.config["material"]["mode_candidates"]
                larger = [x for x in candidates if x > self.material_n]
                if not larger or not any(x in str(exc).lower() for x in ("fit", "modal", "accuracy", "observable")):
                    raise
                self.material_n = larger[0]
                # The material figure must describe the SAME accepted representation.
                old = self.config["material"]["mode_candidates"]
                self.config["material"]["mode_candidates"] = larger
                try:
                    self.material()
                finally:
                    self.config["material"]["mode_candidates"] = old

    def threshold_calculation(self, label, *, config=None, gaps=None, channels=None, count=None, points=None):
        cfg = config or self.config
        p = cfg["pulse"]
        return self.step(label, PREFIX+"calculate_threshold_fluence_gap", {
            "preset": "publication", **self.common(cfg), **self.fit_args(count=count), **self.temporal_args(config=cfg),
            "gaps-nm": gaps if gaps is not None else cfg["geometry"]["gaps_nm"],
            "channels": channels if channels is not None else cfg["geometry"]["channels"],
            "carrier-energy-ev": p["carrier_energy_eV"], "fluence-min-j-cm2": p["fluence_min_J_cm2"],
            "fluence-max-j-cm2": p["fluence_max_J_cm2"], "fluence-points": points or self.fluence_points,
            "target-population": p["target_population"], "reference-fluence-j-cm2": np.sqrt(p["fluence_min_J_cm2"]*p["fluence_max_J_cm2"]),
            "no-refine-threshold": not p["refine_threshold"],
            "threshold-root-xtol-sqrt-fluence": p["threshold_root_xtol_sqrt_J_cm2"],
            "threshold-root-rtol": p["threshold_root_rtol"],
            "max-fluence-grid-midpoint-error": p["max_fluence_midpoint_population_error"],
            "max-isolated-pulse-area-step-rad": p["max_isolated_pulse_area_step_rad"],
            "fluence-grid-convergence-policy": self.policy, "dd-tolerance": cfg["numerics"]["dd_tolerance"]})

    def thresholds(self):
        extensions = 0
        while True:
            try:
                path = self.threshold_calculation("fig03_master")
            except StageError as exc:
                if "fluence grid" in str(exc).lower() and self.fluence_points < self.config["pulse"]["max_fluence_points"]:
                    self.fluence_points = min(2*self.fluence_points-1, self.config["pulse"]["max_fluence_points"])
                    continue
                raise
            data, _ = read_npz(path)
            statuses = np.concatenate(([str(data["isolated_threshold_status"])], data["threshold_status"].ravel()))
            if self.config["smoke"] or not np.any(np.isin(statuses, ["left_censored", "right_censored", "left_lobe_censored"])) or extensions >= self.config["pulse"]["max_fluence_extensions"]:
                break
            if np.any(np.isin(statuses, ["left_censored", "left_lobe_censored"])):
                self.config["pulse"]["fluence_min_J_cm2"] /= 10
            if np.any(statuses == "right_censored"):
                self.config["pulse"]["fluence_max_J_cm2"] *= 2
            extensions += 1
        self.state.update(threshold_master=str(path), fluence_points=self.fluence_points,
                          fluence_min=self.config["pulse"]["fluence_min_J_cm2"], fluence_max=self.config["pulse"]["fluence_max_J_cm2"])
        self.audit_fit_identity(path)
        self.plot("fig03a", "threshold_fluence_gap", path)
        derived = self.step("fig03b_data", PREFIX+"calculate_model_discrepancy_gap", {"source-artifact": path}, dependencies=[path])
        self.plot("fig03b", "model_discrepancy_gap", derived)

    def selection(self):
        data, _ = read_npz(self.state["threshold_master"])
        self.state["selection"] = select_scenarios(data, self.config)
        write_json(self.directory / "selection.json", self.state["selection"])

    def validation(self):
        selection = self.state["selection"]
        channels = list(dict.fromkeys([selection["best_channel"], selection["runner_up_channel"], selection["control_channel"]]))
        gaps = sorted(set([selection["best_gap_nm"], selection["runner_up_gap_nm"]]))
        baseline = self.threshold_calculation("validation_reference", gaps=gaps, channels=channels)
        records = []

        def check(label, cfg=None, **kwargs):
            try:
                artifact = self.threshold_calculation("check_"+label, config=cfg, gaps=kwargs.pop("gaps", gaps), channels=channels, **kwargs)
                result = compare_thresholds(baseline, artifact)
                record = {"label": label, "artifact": str(artifact), **result}
            except StageError as exc:
                # Preserve a failed scientific check as a negative outcome. Never
                # label the recommendation certified just because other plots exist.
                record = {"label": label, "accepted": False, "error": str(exc)}
            records.append(record)
            return record

        for carrier in self.config["pulse"]["carrier_scan_eV"]:
            cfg = deepcopy(self.config)
            cfg["pulse"]["carrier_energy_eV"] = carrier
            record = check("carrier_"+str(carrier), cfg)
            record["carrier_energy_eV"] = carrier
        if self.config["validation"]["enabled"]:
            # Certify the higher-N linear response before its nonlinear check.
            higher = self.config["material"]["validation_modes"]
            try:
                for gap in gaps:
                    self.material_spectrum("check_higher_N_spectrum", gap, count=higher)
                check("higher_N", count=higher)
            except StageError as exc:
                records.append({"label": "higher_N", "accepted": False, "error": str(exc)})
            check("fluence_refined", points=2*self.fluence_points-1)
            original_order = self.spatial_order
            self.spatial_order *= 2
            try:
                check("spatial_refined")
            finally:
                self.spatial_order = original_order
            cfg = deepcopy(self.config)
            cfg["numerics"].update(rtol=cfg["numerics"]["rtol"]/10, atol=cfg["numerics"]["atol"]/10,
                                   points_per_fastest_cycle=2*cfg["numerics"]["points_per_fastest_cycle"])
            check("time_step_refined", cfg)
            # Direct frequency refinement for final candidates is independent of
            # the coarsening checks in the saved original spectra.
            original_points = self.energy_points
            try:
                for gap in gaps:
                    reference = self.material_spectrum("check_energy_reference", gap)
                    self.energy_points = 2*original_points-1
                    path = self.material_spectrum("check_energy_refined", gap)
                    result = compare_spectral_refinement(reference, path, self.config["spectrum"])
                    records.append({"label": "energy_refined", "artifact": str(path),
                                    "reference_artifact": str(reference), "gap_nm": gap, **result})
                    self.energy_points = original_points
            except StageError as exc:
                records.append({"label": "energy_refined", "accepted": False, "error": str(exc)})
            finally:
                self.energy_points = original_points
            v = self.config["validation"]
            for sign, suffix in ((-1, "low"), (1, "high")):
                check("gap_"+suffix, gaps=[g+sign*v["gap_offset_nm"] for g in gaps])
                for parameter, section, key, amount, relative in (
                    ("c", "geometry", "c_nm", v["shape_relative_offset"], True),
                    ("a", "geometry", "a_nm", v["shape_relative_offset"], True),
                    ("exciton", "qd", "transition_energy_eV", v["exciton_offset_eV"], False),
                    ("gamma1", "qd", "population_decay_energy_neV", v["population_decay_relative_offset"], True),
                    ("dephasing", "qd", "pure_dephasing_energy_meV", v["dephasing_relative_offset"], True),
                ):
                    cfg = deepcopy(self.config)
                    cfg[section][key] = cfg[section][key]*(1+sign*amount) if relative else cfg[section][key]+sign*amount
                    if parameter in ("c", "a"):
                        try:
                            for gap in gaps:
                                self.material_spectrum("check_shape_"+parameter+"_"+suffix, gap, config=cfg)
                        except StageError as exc:
                            records.append({"label": parameter+"_"+suffix, "accepted": False, "error": str(exc)})
                            continue
                    check(parameter+"_"+suffix, cfg)
        numerical_labels = {"higher_N", "fluence_refined", "spatial_refined", "time_step_refined", "energy_refined"}
        tolerance = self.config["numerics"]["threshold_relative_tolerance"]
        for row in records:
            if row["label"] in numerical_labels and "max_relative_threshold_change" in row:
                row["accepted"] = row["resolved_pairs_complete"] and row["max_relative_threshold_change"] <= tolerance
        self.state["validation"] = records
        present_labels = {row["label"] for row in records}
        self.state["numerical_validation_accepted"] = bool(self.config["validation"]["enabled"]
            and numerical_labels <= present_labels and all(
                row.get("accepted", False) for row in records if row["label"] in numerical_labels))
        master, _ = read_npz(self.state["threshold_master"])
        self.state["ranking_validation"] = assess_recommendation_ranking(
            master, selection, records, self.config,
            numerical_accepted=self.state["numerical_validation_accepted"],
        )
        write_json(self.directory / "validation.json", records)
        write_json(self.directory / "ranking_validation.json", self.state["ranking_validation"])
        self.plot_validation(records)

    def plot_validation(self, records):
        import csv
        import matplotlib.pyplot as plt
        columns = ("label", "accepted", "resolved_pairs_complete", "max_relative_threshold_change",
                   "best_channel_unchanged", "best_configuration_unchanged", "error")
        with (self.directory / "validation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        sensitivity = [row for row in records if row["label"].endswith(("_low", "_high"))]
        if sensitivity:
            fig, ax = plt.subplots(figsize=(10, 5))
            values = [row.get("max_relative_threshold_change", np.nan) for row in sensitivity]
            values = [value if np.isfinite(value) else np.nan for value in values]
            ax.barh([row["label"] for row in sensitivity], values,
                    color=["tab:blue" if row.get("best_configuration_unchanged", False)
                           and row.get("resolved_pairs_complete", False) else "tab:orange" for row in sensitivity])
            ax.set_xlabel("Maximum relative change in resolved threshold (dimensionless)")
            ax.set_title("Input sensitivity: orange = changed or unresolved ranking")
            ax.grid(axis="x", alpha=.3)
            fig.tight_layout()
            path = self.directory / "figures" / "validation_sensitivity.png"
            fig.savefig(path, dpi=self.config["output"]["dpi"])
            plt.close(fig)
            self.manifest["figures"]["validation_sensitivity"] = str(path.relative_to(self.directory))
        carriers = sorted([r for r in records if "carrier_energy_eV" in r and "candidate_thresholds" in r], key=lambda r: r["carrier_energy_eV"])
        if not carriers:
            return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        first = carriers[0]
        for ci, channel in enumerate(first["channels"]):
            for gi, gap in enumerate(first["gaps_nm"]):
                values = np.array([r["candidate_thresholds"][ci][gi] for r in carriers], float)
                bare = np.array([r["isolated_threshold"] for r in carriers], float)
                energies = [r["carrier_energy_eV"] for r in carriers]
                label = f"{channel}, g={gap:g} nm"
                axes[0].plot(energies, values, "o-", label=label)
                axes[1].plot(energies, np.divide(values, bare, out=np.full_like(values,np.nan), where=bare>0), "o-", label=label)
        axes[0].set_ylabel(r"$F_\eta$ (J/cm$^2$)")
        axes[1].set_ylabel(r"$F_\eta/F_{\eta,0}$")
        for ax in axes:
            ax.set_xlabel("Carrier energy (eV)")
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        if self.config["smoke"]:
            fig.suptitle("SMOKE: diagnostic calculation; not converged article data")
        fig.tight_layout()
        path = self.directory / "figures" / "validation_carrier.png"
        fig.savefig(path, dpi=self.config["output"]["dpi"])
        plt.close(fig)
        self.manifest["figures"]["validation_carrier"] = str(path.relative_to(self.directory))

    def material_effect(self):
        selection = self.state["selection"]
        gap = selection["best_gap_nm"]
        path = self.material_spectrum("fig04_material_excitation", gap)
        self.plot("fig04", "excitation_spectrum_material_comparison", path,
                  {"channels": [selection["best_channel"], selection["control_channel"]]})
        if selection["runner_up_gap_nm"] != gap:
            self.material_spectrum("fig04_runner_up_check", selection["runner_up_gap_nm"])
        p = self.config["pulse"]
        data = self.step("fig05_material_fluence", PREFIX+"calculate_excitation_fluence_material_comparison", {
            "preset": "publication", **self.common(), **self.fit_args(), **self.temporal_args(),
            "gap-nm": gap, "pulse-energy-ev": p["carrier_energy_eV"],
            "fluence-min-j-cm2": p["fluence_min_J_cm2"], "fluence-max-j-cm2": p["fluence_max_J_cm2"],
            "points": self.fluence_points, "grid-scale": "sqrt", "target-population": p["target_population"],
            "max-fluence-grid-midpoint-error": p["max_fluence_midpoint_population_error"],
            "max-isolated-pulse-area-step-rad": p["max_isolated_pulse_area_step_rad"],
            "fluence-grid-convergence-policy": self.policy})
        self.plot("fig05", "excitation_fluence_material_comparison", data,
                  {"channels": [selection["best_channel"], selection["control_channel"]], "target-population": p["target_population"]})
        self.state["fluence_material_artifact"] = str(data)
        self.audit_fit_identity(path)
        self.audit_fit_identity(data)
        arrays, _ = read_npz(data)
        ci = list(arrays["channel_id"].astype(str)).index(selection["best_channel"])
        curve = arrays["p_exc_read"][1, ci]
        fluence = arrays["fluence_j_cm2"]
        threshold = float(arrays["threshold_fluence_j_cm2"][1, ci])
        status = str(arrays["threshold_status"][1, ci])
        first_descent = np.flatnonzero(np.diff(curve) < 0)
        peak = int(first_descent[0]) if first_descent.size else int(np.argmax(curve))
        if not bool(resolved_threshold_mask(status)):
            threshold = float(fluence[max(1, peak)])
        nonlinear = float(fluence[max(1, peak)])
        if nonlinear <= threshold:
            # Pick a measured point, labelled explicitly as a sampled nonlinear
            # example rather than inventing an unsolved first-lobe population.
            nonlinear = float(fluence[min(np.searchsorted(fluence, threshold, side="right"), len(fluence)-1)])
        self.state["selected_fluences"] = {"linear": float(fluence[0]), "threshold_or_illustration": threshold,
                                           "nonlinear_sample": nonlinear, "threshold_status": status}
        self.certify_weak_field()
        master, _ = read_npz(self.state["threshold_master"])
        mc = list(master["channel_id"].astype(str)).index(selection["best_channel"])
        mg = int(np.flatnonzero(np.isclose(master["gap_nm"], gap))[0])
        reference = float(master["threshold_fluence_j_cm2"][1, mc, mg])
        self.state["fig03_fig05_threshold_relative_difference"] = abs(threshold/reference-1) if bool(resolved_threshold_mask(status)) and np.isfinite(reference) and reference>0 else None

    def certify_weak_field(self):
        """Check small population AND the P/F plateau using two solved pulses."""
        selected, n = self.state["selection"], self.config["numerics"]
        fluence = self.state["selected_fluences"]["linear"]
        certificate = {}
        for attempt in range(n["max_weak_field_refinements"] + 1):
            paths = []
            populations, maxima = [], []
            for scale in (1., .5):
                path = self.step("check_weak_field", PREFIX+"calculate_population_dynamics", {
                    **self.common(), **self.fit_args(), **self.temporal_args(dynamics=True),
                    "channels": [selected["best_channel"]], "gap-nm": selected["best_gap_nm"],
                    "fluence-j-cm2": fluence*scale,
                    "pulse-energy-ev": self.config["pulse"]["carrier_energy_eV"]})
                arrays, _ = read_npz(path)
                populations.append(np.asarray(arrays["population_final"], float)/(fluence*scale))
                maxima.append(float(np.max(arrays["population_max"])))
                paths.append(str(path))
            relative = float(np.max(np.abs(populations[0]-populations[1]) /
                                    np.maximum(np.abs(populations[1]), np.finfo(float).tiny)))
            accepted = (all(np.all(np.isfinite(p) & (p > 0)) for p in populations)
                        and relative <= n["weak_field_relative_tolerance"]
                        and max(maxima) <= n["weak_field_max_population"])
            certificate = {"accepted": bool(accepted), "fluence_J_cm2": fluence,
                           "max_population": max(maxima), "max_relative_change_P_over_F": relative,
                           "artifacts": paths, "includes_isolated_QD": True}
            if accepted:
                break
            if attempt < n["max_weak_field_refinements"]:
                fluence /= 10
        self.state["weak_field_check"] = certificate
        self.state["selected_fluences"]["linear"] = fluence
        if not certificate["accepted"] and not self.config["smoke"]:
            raise StageError("No weak-field P/F plateau found within configured refinements; inspect weak_field_check.")

    def dynamics(self):
        selected = self.state["selection"]
        fluence = self.state["selected_fluences"]["threshold_or_illustration"]
        for name, gap in (("near", selected["best_gap_nm"]), ("far", selected["far_gap_nm"])):
            data = self.step("fig06_dynamics_"+name, PREFIX+"calculate_population_dynamics", {
                **self.common(), **self.fit_args(), **self.temporal_args(dynamics=True),
                "include-dd": True, "channels": [selected["best_channel"], selected["control_channel"]],
                "gap-nm": gap, "fluence-j-cm2": fluence, "pulse-energy-ev": self.config["pulse"]["carrier_energy_eV"]})
            self.audit_fit_identity(data)
            self.plot("fig06_"+name, "population_dynamics", data)
            self.plot("fig06_"+name+"_pulse_zoom", "population_dynamics", data, {"x-min-fs": -60, "x-max-fs": 300})

    def work_spectrum(self):
        w = self.config["work_spectrum"]
        if not w["enabled"]:
            return
        f, selected = self.state["selected_fluences"], self.state["selection"]
        material = self.config["material"]
        fit = self.fit_args(mode_key="multi-fit-modes")
        fit.pop("fit-min-ev")
        fit.pop("fit-max-ev")
        temporal = self.temporal_args()
        for key in ("population-decay-policy", "max-population-decay-fraction-at-read", "bright-fit-quality-policy"):
            temporal.pop(key)
        temporal["post-fs"] = w["post_fs"]
        # These are recorded diagnostic policies in --smoke only.
        args = {"preset": "publication", **self.common(), **fit, **temporal,
                "fit-window-ev": [material["fit_min_eV"],material["fit_max_eV"]],
                "channel": selected["best_channel"], "gap-nm": selected["best_gap_nm"],
                "carrier-energy-ev": self.config["pulse"]["carrier_energy_eV"],
                "fluence-j-cm2": [f["linear"], f["threshold_or_illustration"], f["nonlinear_sample"]],
                "fluence-label": ["weak_field" if self.state.get("weak_field_check", {}).get("accepted") else "low_fluence_unverified",
                                  "threshold" if bool(resolved_threshold_mask(f["threshold_status"])) else "illustration", "nonlinear_sample"],
                "selection-source-artifact": self.state["fluence_material_artifact"],
                "energy-min-ev": w["min_eV"], "energy-max-ev": w["max_eV"], "energy-points": w["points"],
                "max-delta-window-relative-change": w["max_delta_window_relative_change"],
                "max-delta-window-absolute-change-cm2": w["max_delta_window_absolute_change_cm2"],
                "observable-convergence-policy": self.policy, "energy-grid-convergence-policy": self.policy,
                "spectral-support-policy": self.policy, "incident-ft-policy": self.policy}
        path = self.step("supp01_work_spectrum", PREFIX+"calculate_work_loss_spectrum_material_comparison", args,
                         dependencies=[Path(self.state["fluence_material_artifact"])])
        self.audit_fit_identity(path)
        self.plot("supp01", "work_loss_spectrum_material_comparison", path)

    def report(self):
        verdict = bool(not self.config["smoke"] and self.state.get("numerical_validation_accepted", False)
                       and self.state["selection"]["resolved_threshold_selected"]
                       and self.state.get("ranking_validation", {}).get("accepted", False))
        difference = self.state.get("fig03_fig05_threshold_relative_difference")
        verdict = bool(verdict and difference is not None and difference <= self.config["numerics"]["threshold_relative_tolerance"])
        report = {"scope": "conditional single-MNP local-QS recommendations, not Shah dimer reproduction or PL quantum yield",
                  "numerically_checked_recommendation": verdict, "smoke": self.config["smoke"],
                  "units": self.audit, "results": self.state, "figures": self.manifest["figures"]}
        write_json(self.directory / "article_results.json", report)
        self.manifest["recommendation_accepted"] = verdict
        banner = "SMOKE: техническая проверка цепочки, не данные для статьи." if self.config["smoke"] else (
            "Проверки численной сходимости и устойчивости выбранного кандидата на заданных сетках пройдены; глобальный оптимум и физическая применимость не удостоверены." if verdict else
            "Есть неразрешённые проверки или пороги. Графики не удостоверяют однозначную рекомендацию; см. article_results.json и validation.json.")
        figures = "\n".join(f'<figure><figcaption>{html.escape(name)}</figcaption><a href="{html.escape(path)}"><img src="{html.escape(path)}" loading="lazy"></a></figure>' for name,path in self.manifest["figures"].items())
        page = f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>QD–MNP article results</title><style>body{{font:16px sans-serif;max-width:1200px;margin:2em auto}}img{{max-width:100%}}figure{{margin:2em 0}}aside{{padding:1em;background:#fff2cf}}</style><h1>КТ и одна золотая МНЧ</h1><aside>{banner}</aside><p>Входы: resolved_inputs.json. Единицы: units.json. Численные результаты: article_results.json. Журнал: manifest.json и logs/.</p>{figures}</html>'
        (self.directory / "index.html").write_text(page, encoding="utf-8")
        print(f"\nResults and figures: {self.directory / 'index.html'}", flush=True)

    def run(self):
        # Replaying orchestration on resume reuses verified artifacts and rebuilds
        # dynamic choices deterministically; no mutable state is blindly trusted.
        with material_fit_cache(self.directory / "material_fit_cache"), diagnostic_figure_label(self.config["smoke"]):
            for stage in STAGES:
                getattr(self, stage)()
                self.save()
                if self.stop_after == stage:
                    self.manifest["status"] = "stopped_after_"+stage
                    self.save()
                    return
        self.manifest["status"] = "complete_diagnostic" if self.config["smoke"] else "complete"
        self.save()


def select_scenarios(data: dict, config: dict) -> dict:
    channels, gaps = list(data["channel_id"].astype(str)), np.asarray(data["gap_nm"], float)
    thresholds = np.asarray(data["threshold_fluence_j_cm2"][1], float)
    resolved = resolved_threshold_mask(data["threshold_status"][1]) & np.isfinite(thresholds) & (thresholds>0)
    g, v = config["geometry"], config["validation"]
    upper_energy = max(config["spectrum"]["max_eV"], max(config["pulse"]["carrier_scan_eV"]))
    k = np.sqrt(config["medium"]["relative_permittivity"])*upper_energy*elementary_charge/(hbar*c)*1e-9
    radii = np.array([g["c_nm"] if ch.startswith("axis") else g["a_nm"] for ch in channels])
    kR = k*(radii[:,None]+g["qd_radius_nm"]+gaps[None,:])
    allowed = (kR<=v["max_k_R_for_selection"]) & (k*g["c_nm"]<=v["max_k_c_for_selection"])
    eligible = resolved & allowed
    candidates = sorted(((float(thresholds[ci,gi]),ci,gi) for ci,gi in zip(*np.where(eligible))))
    if candidates:
        _, ci, gi = candidates[0]
        runner = next((entry for entry in candidates if entry[1]!=ci), candidates[0])
        _, rci, rgi = runner
    else:
        # An illustration is still needed for the material/time comparison.
        # It is explicitly NOT an inferred optimum or a resolved threshold.
        ci = channels.index("axis_long")
        gi = int(np.argmin(np.abs(gaps-g["reference_surface_gap_nm"])))
        rci, rgi = channels.index("side_long"), gi
    controls = {"axis_long":"axis_trans", "axis_trans":"axis_long", "side_long":"side_trans_radial",
                "side_trans_radial":"side_long", "side_trans_tangential":"side_trans_radial"}
    usable_gaps = gaps[allowed[ci]]
    far = float(usable_gaps[-1]) if usable_gaps.size else float(gaps[-1])
    discrepancy = np.asarray(data["absolute_threshold_discrepancy_dd_vs_fqs"][ci],float)
    valid_discrepancy = discrepancy[np.isclose(gaps, far)]
    return {"best_channel": channels[ci], "best_gap_nm": float(gaps[gi]), "runner_up_channel":channels[rci],
            "runner_up_gap_nm":float(gaps[rgi]), "control_channel":controls[channels[ci]], "far_gap_nm":far,
            "resolved_threshold_selected": bool(candidates), "selection_is_illustration_only":not bool(candidates),
            "far_point_satisfies_threshold_DD_tolerance": bool(valid_discrepancy.size and np.isfinite(valid_discrepancy[0]) and valid_discrepancy[0]<=config["numerics"]["dd_tolerance"]),
            "k_c_at_upper_study_energy":float(k*g["c_nm"]), "k_R_by_channel_gap":kR,
            "selection_allowed_by_declared_retardation_cutoffs":allowed,
            "physical_scope":"conditional point-QD/local-QS result; declared k cutoffs do not validate finite-QD or nonlocal physics"}


def compare_thresholds(reference_path, candidate_path):
    reference,_ = read_npz(reference_path)
    candidate,_ = read_npz(candidate_path)
    if not np.array_equal(reference["channel_id"], candidate["channel_id"]) or reference["threshold_status"].shape != candidate["threshold_status"].shape:
        raise ValueError("Threshold comparison requires matching channel and gap-index ordering.")
    left, right = reference["threshold_fluence_j_cm2"][1], candidate["threshold_fluence_j_cm2"][1]
    valid = resolved_threshold_mask(reference["threshold_status"][1]) & resolved_threshold_mask(candidate["threshold_status"][1])
    valid &= np.isfinite(left) & np.isfinite(right) & (left>0) & (right>0)
    difference = np.abs(right[valid]/left[valid]-1)
    masked = np.where(valid,right,np.nan)
    isolated = float(candidate["isolated_threshold_fluence_j_cm2"])
    isolated_reference = float(reference["isolated_threshold_fluence_j_cm2"])
    isolated_valid = bool(resolved_threshold_mask(str(candidate["isolated_threshold_status"]))
                          and resolved_threshold_mask(str(reference["isolated_threshold_status"]))
                          and np.isfinite(isolated) and np.isfinite(isolated_reference)
                          and isolated > 0 and isolated_reference > 0)
    isolated_change = abs(isolated/isolated_reference-1) if isolated_valid else float("inf")
    if not isolated_valid:
        isolated = np.nan
    before_best = np.unravel_index(np.nanargmin(np.where(valid,left,np.nan)),left.shape) if np.any(valid) else None
    after_best = np.unravel_index(np.nanargmin(masked),right.shape) if np.any(valid) else None
    return {"accepted":bool(np.any(valid)), "resolved_pairs_complete":bool(np.all(valid) and isolated_valid),
            "max_relative_threshold_change":max(float(np.max(difference)), isolated_change) if difference.size else float("inf"),
            "isolated_relative_threshold_change":isolated_change,
            "channels":candidate["channel_id"].astype(str).tolist(), "gaps_nm":candidate["gap_nm"].tolist(),
            "candidate_thresholds":masked.tolist(), "isolated_threshold":isolated,
            "candidate_status":candidate["threshold_status"][1].astype(str).tolist(),
            "best_channel_unchanged":bool(before_best is not None and before_best[0]==after_best[0]),
            "best_configuration_unchanged":bool(before_best is not None and before_best==after_best)}


def compare_spectral_refinement(reference_path, candidate_path, limits):
    """Compare actual coarse/fine spectra; a successful second solve is insufficient."""
    left, _ = read_npz(reference_path)
    right, _ = read_npz(candidate_path)
    for key in ("channel_id", "material_model_id", "surface_gap_nm"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError("Spectral refinement must preserve channels, material branches and gap.")
    coarse, fine = left["energy_eV"], right["energy_eV"]
    if fine.size <= coarse.size or not np.allclose(coarse[[0,-1]], fine[[0,-1]], rtol=0, atol=1e-12):
        raise ValueError("Spectral refinement must increase point count on the same energy interval.")
    branches = [list(left["material_model_id"].astype(str)).index(name) for name in ("direct", "multi")]
    band = np.abs(fine-limits["feature_center_eV"]) <= limits["feature_half_window_eV"]
    spectra_left = left["excitation_spectrum_au6"][branches]
    spectra_right = right["excitation_spectrum_au6"][branches]
    interpolated = np.array([np.interp(fine, coarse, curve) for curve in spectra_left.reshape(-1, coarse.size)]).reshape(spectra_right.shape)
    denominator = np.sqrt(np.mean(spectra_right[...,band]**2, axis=-1))
    nrms = np.sqrt(np.mean((interpolated[...,band]-spectra_right[...,band])**2, axis=-1)) / np.maximum(denominator, np.finfo(float).tiny)
    status_left, status_right = left["feature_status"][branches], right["feature_status"][branches]
    unique = (status_left=="ok") & (status_right=="ok")
    split = (status_left=="split_or_ambiguous") & (status_right=="split_or_ambiguous")
    split &= left["competing_peak_count"][branches] == right["competing_peak_count"][branches]
    width0 = float(left["isolated_fwhm_eV"])
    shift = np.abs(left["peak_energy_eV"][branches]-right["peak_energy_eV"][branches])/width0
    width = np.abs(right["fwhm_eV"][branches]/left["fwhm_eV"][branches]-1)
    gain = np.abs(right["excitation_gain_optimized"][branches]/left["excitation_gain_optimized"][branches]-1)
    feature_ok = split | (unique & (shift<=limits["max_observable_shift_over_gamma0"])
                         & (width<=limits["max_observable_width_relative_error"]))
    isolated_ok = bool(str(left["isolated_feature_status"])==str(right["isolated_feature_status"])=="ok"
                       and np.isfinite(width0) and width0>0
                       and abs(float(right["isolated_fwhm_eV"])/width0-1)<=limits["max_observable_width_relative_error"]
                       and abs(float(right["isolated_peak_energy_eV"])-float(left["isolated_peak_energy_eV"]))/width0<=limits["max_observable_shift_over_gamma0"])
    accepted = bool(isolated_ok and np.all(feature_ok) and np.all(np.isfinite(nrms))
                    and np.all(nrms<=limits["max_coarsening_change"])
                    and np.all(gain<=limits["max_observable_gain_relative_error"]))
    return {"accepted":accepted, "spectrum_refinement_max_nrms":float(np.max(nrms)),
            "gain_refinement_max_relative_change":float(np.max(gain)),
            "unique_peak_shift_max_over_isolated_fwhm":float(np.max(shift[unique])) if np.any(unique) else None,
            "unique_peak_width_max_relative_change":float(np.max(width[unique])) if np.any(unique) else None,
            "feature_classification_preserved":bool(np.all(unique|split)), "isolated_feature_accepted":isolated_ok}


def assess_recommendation_ranking(master, selection, records, config, *, numerical_accepted):
    """Conservative grid-candidate check, not a confidence interval or global optimum."""
    numerical = {"higher_N", "fluence_refined", "spatial_refined", "time_step_refined"}
    changes = [float(row["max_relative_threshold_change"]) for row in records
               if row["label"] in numerical and np.isfinite(row.get("max_relative_threshold_change", np.inf))]
    allowance = max([config["numerics"]["threshold_relative_tolerance"], *changes])
    channels = list(master["channel_id"].astype(str))
    gaps = np.asarray(master["gap_nm"], float)
    values = np.asarray(master["threshold_fluence_j_cm2"][1], float)
    statuses = master["threshold_status"][1].astype(str)
    allowed = np.asarray(selection["selection_allowed_by_declared_retardation_cutoffs"], bool)
    resolved = resolved_threshold_mask(statuses) & np.isfinite(values) & (values>0)
    ci = channels.index(selection["best_channel"])
    gi = int(np.flatnonzero(np.isclose(gaps, selection["best_gap_nm"]))[0])
    winner = values[ci,gi]
    winner_resolved = bool(selection["resolved_threshold_selected"] and resolved[ci,gi] and allowed[ci,gi])
    # A right bound or a missed first-lobe target cannot beat a resolved target
    # inside this same scan. Left bounds and failed curves can conceal a winner.
    uncertain = allowed & ~resolved & ~np.isin(statuses, ("right_censored", "not_reached_first_lobe"))
    alternatives = allowed & resolved
    alternatives[ci,gi] = False
    separated = bool(winner_resolved and not np.any(uncertain) and allowance<1
                     and np.all(values[alternatives]*(1-allowance)>winner*(1+allowance)))
    equivalent = []
    if winner_resolved:
        for aci, agi in zip(*np.where(allowed & resolved & (values*(1-allowance)<=winner*(1+allowance)))):
            equivalent.append({"channel":channels[aci], "gap_nm":float(gaps[agi]),
                               "threshold_J_cm2":float(values[aci,agi])})
    expected_sensitivity = {name+"_"+suffix for name in ("gap", "c", "a", "exciton", "gamma1", "dephasing") for suffix in ("low", "high")}
    sensitivity = [row for row in records if row["label"] in expected_sensitivity or "carrier_energy_eV" in row]
    present = {row["label"] for row in sensitivity}
    carrier_values = {row["carrier_energy_eV"] for row in sensitivity if "carrier_energy_eV" in row}
    sensitivity_complete = bool(expected_sensitivity<=present and set(config["pulse"]["carrier_scan_eV"])<=carrier_values)
    ranking_preserved = bool(sensitivity_complete and all(row.get("accepted", False)
        and row.get("resolved_pairs_complete", False) and row.get("best_configuration_unchanged", False) for row in sensitivity))

    def sensitivity_separated(row):
        candidates = np.asarray(row.get("candidate_thresholds", []), float).ravel()
        if not candidates.size or not np.all(np.isfinite(candidates)) or np.any(candidates<=0):
            return False
        ordered = np.sort(candidates)
        return bool(allowance<1 and (ordered.size==1 or ordered[0]*(1+allowance)<ordered[1]*(1-allowance)))

    separated_sensitivity = bool(sensitivity_complete and all(sensitivity_separated(row) for row in sensitivity))
    return {"accepted":bool(config["validation"]["enabled"] and numerical_accepted and separated and ranking_preserved and separated_sensitivity),
            "selected_candidate_separated":separated, "sensitivity_checks_complete":sensitivity_complete,
            "ranking_preserved_in_tested_sensitivity_cases":ranking_preserved,
            "selected_candidate_separated_in_sensitivity_cases":separated_sensitivity,
            "relative_numerical_allowance":allowance,
            "allowance_definition":"max(declared threshold tolerance, observed refinement changes); conservative resolution allowance, not a statistical confidence interval",
            "indistinguishable_grid_candidates":equivalent,
            "unresolved_potential_competitor_count":int(np.count_nonzero(uncertain)),
            "scope":"configured discrete candidates and one-parameter sensitivity cases only; no global or ensemble optimum"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Exercise the whole chain on small, explicitly unconverged grids.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and show SI/a.u. conversions without any solve or file changes.")
    parser.add_argument("--stop-after", choices=STAGES, help="Stop after this stage; the same command without this option resumes.")
    args = parser.parse_args(argv)
    config = load_inputs(args.config, smoke=args.smoke)
    audit = unit_audit(config)
    if args.dry_run:
        print(json.dumps(clean_json({"stages":STAGES,"kernels":KERNELS,"unit_audit":audit,
                                    "base_threshold_grid_solve_count": (1+2*len(config["geometry"]["channels"])*len(config["geometry"]["gaps_nm"]))*config["pulse"]["fluence_points"],
                                    "count_excludes":"root refinement, preflight, material comparisons and validation repeats"}),ensure_ascii=False,indent=2))
        return 0
    directory = args.output or ROOT / (config["output"]["directory"] + ("_smoke" if args.smoke else ""))
    run = ArticleRun(config,directory,stop_after=args.stop_after)
    try:
        run.run()
    except (Exception,KeyboardInterrupt) as exc:
        run.manifest.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed", error=str(exc))
        run.save()
        print(f"\nRun stopped. Completed artifacts are preserved in {run.directory}.\n{exc}",file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
