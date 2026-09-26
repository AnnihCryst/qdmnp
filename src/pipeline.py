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
import subprocess
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

from qdmnp.observables.article_inputs import (
    DEFAULT_INPUT, ROOT, center_distance_nm, load_inputs, medium_wavenumber_per_nm,
    physical_arguments, unit_audit, upper_study_energy_eV,
)
from qdmnp.observables.article_fit_cache import material_fit_cache
from qdmnp.observables.article_fit_identity import compare_fit_coefficients
from qdmnp.observables.threshold_metrics import resolved_threshold_mask
from qdmnp.observables.gap_metrics_common import dd_validity_assessment
from qdmnp.observables.parallel import fit_material_job, process_pool, resolve_workers
from qdmnp.spheroid_equatorial import MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE
from qdmnp.spheroid_green import MAX_SUPPORTED_SPATIAL_DEGREE


def spatial_order_limit(channels) -> int:
    """Largest spatial order the analytic kernels of these channels support."""
    side = any(str(channel).startswith("side") for channel in channels)
    return MAX_SUPPORTED_EQUATORIAL_SPATIAL_DEGREE if side else MAX_SUPPORTED_SPATIAL_DEGREE

RIGHT_CENSORED = ("right_censored", "not_reached_first_lobe")
LEFT_CENSORED = ("left_censored", "left_lobe_censored")


PREFIX = "qdmnp.observables."
KERNELS = {
    "DD_time": "qdmnp.rational_fit.HybridQDPlasmonModel",
    "FQS_time": "qdmnp.full_qs_model.FullQSSpheroidPulseModel",
    "FQS_tip": "qdmnp.spheroid_green.SpheroidGreenInteraction",
    "FQS_side": "qdmnp.spheroid_equatorial.EquatorialSpheroidGreenInteraction",
}
STAGES = (
    "material", "spectra", "preflight", "thresholds", "selection",
    "validation", "material_effect", "dynamics", "work_spectrum", "article_diagnostics", "report",
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
        package = ROOT / "src"
        sources = sorted([ROOT / "scripts" / "run_article.py", *package.glob("*.py"),
                          *(package / "observables").glob("*.py")])
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

    def _prepare_step(self, label, module, arguments=(), *, suffix=".npz", dependencies=()):
        """Resume/reuse rules shared by serial and concurrent steps: ("done", path) or ("pending", info)."""
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
                    return "done", original
        if record.get("status") == "complete":
            if not path.exists() or record.get("sha256") != sha(path):
                raise StageError(f"Completed artifact is missing or changed: {path}")
            print(f"Resume: {label}", flush=True)
            return "done", path
        if path.exists():
            raise StageError(f"Uncertified/unrecorded output already exists: {path}; preserve it and use a new output directory.")
        path.parent.mkdir(parents=True, exist_ok=True)
        log = self.directory / "logs" / f"{label}_{signature[:10]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        return "pending", {"label": label, "module": module, "arguments": arguments, "path": path, "key": key, "log": log}

    def step(self, label, module, arguments=(), *, suffix=".npz", dependencies=()):
        state, prepared = self._prepare_step(label, module, arguments, suffix=suffix, dependencies=dependencies)
        if state == "done":
            return prepared
        arguments, path, key, log = prepared["arguments"], prepared["path"], prepared["key"], prepared["log"]
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

    def steps_concurrently(self, specs, concurrency):
        """Run independent calculator steps as parallel subprocesses.

        The resume, reuse and manifest rules are those of step(). Returns
        {label: path or StageError}; one failed step does not stop the others.
        """
        outcomes, pending = {}, []
        for label, module, arguments in specs:
            try:
                state, prepared = self._prepare_step(label, module, arguments)
            except StageError as exc:
                outcomes[label] = exc
                continue
            if state == "done":
                outcomes[label] = prepared
            else:
                pending.append(prepared)
        if concurrency <= 1 or len(pending) <= 1:
            for prepared in pending:
                try:
                    outcomes[prepared["label"]] = self.step(prepared["label"], prepared["module"], prepared["arguments"])
                except StageError as exc:
                    outcomes[prepared["label"]] = exc
            return outcomes
        queue, running = list(pending), []
        try:
            while queue or running:
                while queue and len(running) < concurrency:
                    prepared = queue.pop(0)
                    command = [sys.executable, "-m", prepared["module"], *prepared["arguments"], "--output", str(prepared["path"])]
                    self.manifest["steps"][prepared["key"]] = {"status": "running", "command": command,
                                                               "output": str(prepared["path"]), "log": str(prepared["log"])}
                    self.save()
                    stream = prepared["log"].open("w", encoding="utf-8")
                    # The child inherits QDMNP_MATERIAL_FIT_CACHE and single-thread BLAS settings.
                    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    running.append((prepared, process, stream, time.perf_counter()))
                    print(f"Running {prepared['label']} -> {prepared['path'].name} (concurrent)", flush=True)
                time.sleep(1.0)
                still_running = []
                for prepared, process, stream, started in running:
                    code = process.poll()
                    if code is None:
                        still_running.append((prepared, process, stream, started))
                        continue
                    stream.close()
                    elapsed = time.perf_counter() - started
                    record = self.manifest["steps"][prepared["key"]]
                    if code == 0 and prepared["path"].exists():
                        record.update(status="complete", sha256=sha(prepared["path"]), elapsed_s=elapsed)
                        outcomes[prepared["label"]] = prepared["path"]
                        print(f"Finished {prepared['label']} in {elapsed:.0f} s", flush=True)
                    else:
                        message = f"CLI exited with {code}" if code else f"Stage did not create its output: {prepared['path']}"
                        tail = prepared["log"].read_text(encoding="utf-8", errors="replace")[-1500:]
                        record.update(status="failed", error=message, elapsed_s=elapsed)
                        outcomes[prepared["label"]] = StageError(f"{prepared['label']}: {message}; see {prepared['log']}\n{tail}")
                        print(f"Failed {prepared['label']} after {elapsed:.0f} s", flush=True)
                    self.save()
                running = still_running
        except KeyboardInterrupt:
            for prepared, process, stream, started in running:
                process.terminate()
                stream.close()
                self.manifest["steps"][prepared["key"]].update(status="failed", error="interrupted",
                                                               elapsed_s=time.perf_counter()-started)
            self.save()
            raise
        return outcomes

    def prefit_materials(self):
        """Fill the material-fit cache for every shape and pole count of the run in parallel.

        The fits are identical to those the calculators would compute one after
        another (same cache key); only the waiting time changes.
        """
        if self.config["smoke"]:
            return
        g, m, v = self.config["geometry"], self.config["material"], self.config["validation"]
        window, refinement = (m["fit_min_eV"], m["fit_max_eV"]), m.get("refinement")
        eps_m = self.config["medium"]["relative_permittivity"]
        combinations = [(g["c_nm"], g["a_nm"], n) for n in sorted({1, *m["mode_candidates"]})]
        if v["enabled"]:
            combinations.append((g["c_nm"], g["a_nm"], m["validation_modes"]))
            amount = v["shape_relative_offset"]
            for sign in (-1, 1):
                # Same arithmetic as validation(): cfg[key]*(1+sign*amount).
                for c_nm, a_nm in ((g["c_nm"]*(1+sign*amount), g["a_nm"]), (g["c_nm"], g["a_nm"]*(1+sign*amount))):
                    combinations += [(c_nm, a_nm, n) for n in sorted({1, *v["shape_check_mode_candidates"]})]
        jobs = [(c_nm, a_nm, eps_m, orientation, n, window, refinement)
                for c_nm, a_nm, n in combinations for orientation in ("long", "trans")]
        workers = resolve_workers(self.config["numerics"]["workers"], len(jobs))
        if workers <= 1:
            return
        started = time.perf_counter()
        print(f"Prefitting {len(jobs)} material representations on {workers} processes ...", flush=True)
        with process_pool(workers) as pool:
            fitted = list(pool.map(fit_material_job, jobs))
        self.state["prefit_materials"] = {"count": len(fitted), "elapsed_s": time.perf_counter()-started,
                                          "fits": [dict(zip(("c_nm", "a_nm", "orientation", "n_modes", "nrms_alpha",
                                                             "nrms_inverse_alpha", "max_alpha_error"), row)) for row in fitted]}
        print(f"Prefit finished in {time.perf_counter()-started:.0f} s", flush=True)

    def common(self, config=None):
        return physical_arguments(config or self.config)

    def fit_args(self, *, mode_key="material-fit-modes", count=None, config=None):
        cfg = config or self.config
        m, n = cfg["material"], cfg["numerics"]
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
                  "step-frequency-policy": n["step_frequency_policy"], "dark-reduction": n["dark_reduction"],
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
                    "spectral-observable": "linear_qd_response", "dd-tolerance": self.config["numerics"]["dd_tolerance"],
                    **self.retardation_args()})
                break
            except StageError as exc:
                if not self.refine_spectral_failure(exc):
                    raise
        self.state["spectral_master"] = str(master)
        for letter, metric in zip("abcd", ("excitation_gain", "resonance_shift", "spectral_width", "model_discrepancy")):
            data = master if letter == "a" else self.step("fig02"+letter+"_data", PREFIX+"calculate_"+metric+"_gap", {"source-artifact": master}, dependencies=[master])
            self.plot("fig02"+letter, metric+"_gap", data)
        self.state.update(energy_points=self.energy_points, spatial_order=self.spatial_order)

    def retardation_args(self, config=None):
        cfg = config or self.config
        return {"max-k-r": cfg["validation"]["max_k_R_for_selection"],
                "retardation-energy-ev": upper_study_energy_eV(cfg)}

    def refine_spectral_failure(self, exc):
        message = str(exc).lower()
        if any(x in message for x in ("sampling", "coarsening", "energy-grid", "spectral grid")) and self.energy_points < self.config["spectrum"]["max_points"]:
            self.energy_points = min(2*self.energy_points-1, self.config["spectrum"]["max_points"])
            return True
        ceiling = min(self.config["numerics"]["max_spatial_order"], spatial_order_limit(self.config["geometry"]["channels"]))
        if "spatial" in message and self.spatial_order < ceiling:
            self.spatial_order = min(2*self.spatial_order, ceiling)
            return True
        return False

    def material_spectrum(self, label, gap, *, count=None, config=None, overrides=None):
        s = self.config["spectrum"]
        fit = self.fit_args(mode_key="multi-fit-modes", count=count, config=config)
        # This calculator checks multi-mode fit accuracy unconditionally;
        # only its ONE-mode diagnostic branch has an accuracy policy switch.
        fit.pop("fit-quality-policy")
        return self.step(label, PREFIX+"calculate_excitation_spectrum_material_comparison", {
            "preset": "publication", **self.common(config), **fit, **self.spectral_args(),
            "channels": self.config["geometry"]["channels"], "gap-nm": gap,
            "max-energy-step-over-isolated-fwhm": s["max_step_over_width"], "observable-fit-policy": self.policy,
            "max-observable-spectrum-nrms": s["max_observable_nrms"], "max-observable-shift-over-isolated-fwhm": s["max_observable_shift_over_gamma0"],
            "max-observable-width-relative-error": s["max_observable_width_relative_error"], "max-observable-gain-relative-error": s["max_observable_gain_relative_error"],
            **(overrides or {})})

    def preflight(self):
        gaps = np.asarray(self.config["geometry"]["gaps_nm"], float)
        _, allowed = retardation_selection_mask(self.config, self.config["geometry"]["channels"], gaps)
        discrepancy = None
        if self.state.get("spectral_master"):
            spectral, _ = read_npz(self.state["spectral_master"])
            discrepancy = spectral.get("spectral_l2_relative_dd_vs_fqs")
        preflight_gaps, rule = choose_preflight_gaps(gaps, allowed, discrepancy, self.config["numerics"]["dd_tolerance"])
        self.state["preflight_gap_choice"] = {"gaps_nm": preflight_gaps, "rule": rule}
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
        return self.step(label, PREFIX+"calculate_threshold_fluence_gap",
                         self.threshold_arguments(config=config, gaps=gaps, channels=channels, count=count, points=points))

    def threshold_arguments(self, *, config=None, gaps=None, channels=None, count=None, points=None):
        cfg = config or self.config
        p = cfg["pulse"]
        return {
            "preset": "publication", **self.common(cfg), **self.fit_args(count=count, config=cfg), **self.temporal_args(config=cfg),
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
            "fluence-grid-convergence-policy": self.policy, "dd-tolerance": cfg["numerics"]["dd_tolerance"],
            "threshold-search": cfg["numerics"]["threshold_search"],
            "bracket-max-area-step-rad": cfg["numerics"]["bracket_max_area_step_rad"],
            "workers": cfg["numerics"]["workers"],
            **self.retardation_args(cfg)}

    def thresholds(self):
        extensions = 0
        # The grid prediction only matters when every hybrid curve uses the grid.
        if self.state.get("spectral_master") and not self.config["smoke"] and self.config["numerics"]["threshold_search"] == "grid":
            spectral, _ = read_npz(self.state["spectral_master"])
            prediction = predict_fluence_points(self.config, spectral.get("excitation_gain_optimized"), self.fluence_points)
            self.state["fluence_grid_prediction"] = prediction
            # A failed grid audit discards every solve of the whole master; start
            # from the nested grid predicted to pass instead of paying for it.
            self.fluence_points = max(self.fluence_points, prediction["predicted_points"])
        while True:
            try:
                path = self.threshold_calculation("fig03_master")
            except StageError as exc:
                if "fluence grid" in str(exc).lower() and self.fluence_points < self.config["pulse"]["max_fluence_points"]:
                    self.fluence_points = min(2*self.fluence_points-1, self.config["pulse"]["max_fluence_points"])
                    continue
                raise
            data, _ = read_npz(path)
            if self.config["smoke"]:
                break
            extend_min, extend_max = censoring_extension(data, self.config)
            self.state["fluence_range_decision"] = {"extend_min": extend_min, "extend_max": extend_max, "extensions": extensions}
            if not (extend_min or extend_max) or extensions >= self.config["pulse"]["max_fluence_extensions"]:
                break
            if extend_min:
                self.config["pulse"]["fluence_min_J_cm2"] /= 10
            if extend_max:
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

    def shape_validation_candidate(self, label, config, gaps, overrides):
        """Certify a varied shape, refining numerics without weakening its gates.

        A seed is computed from the native lower-c fit in THIS run. It is only
        an optimizer starting point; the target shape must pass the ordinary
        material and all-channel final-spectrum checks again.
        """
        v = self.config["validation"]
        attempts = []
        selected_overrides = dict(overrides)
        ceiling = min(self.config["numerics"]["max_spatial_order"],
                      spatial_order_limit(self.config["geometry"]["channels"]))

        def certify(cfg, count, refinement):
            while True:
                order = selected_overrides.get("spatial-order-max", self.spatial_order)
                try:
                    paths = [self.material_spectrum("check_shape_"+label, gap, count=count,
                                 config=cfg, overrides=selected_overrides) for gap in gaps]
                    return cfg, count, dict(selected_overrides), {
                        "material_modes_used": count, "spatial_order_used": order,
                        "shape_spectrum_artifact": str(paths[0]),
                        "shape_spectrum_artifacts": [str(p) for p in paths],
                        "material_refinement": refinement, "refinement_attempts": list(attempts),
                        "relaxed_numerical_gates": dict(overrides),
                        "gate_scope": "Unchanged shape-specific limits from configured inputs",
                    }
                except StageError as exc:
                    attempts.append({"material_modes": count, "spatial_order": order,
                                     "refinement": refinement, "error": str(exc)})
                    if "spatial" in str(exc).lower() and order < ceiling:
                        selected_overrides["spatial-order-max"] = min(2*order, ceiling)
                        continue
                    raise

        for count in v["shape_check_mode_candidates"]:
            try:
                return certify(config, count, "native")
            except StageError as exc:
                if not any(word in str(exc).lower() for word in ("fit", "accuracy", "modal", "observable")):
                    raise
        if v.get("shape_fit_fallback", "none") != "native_neighbor_minimax":
            raise StageError(" | ".join(row["error"] for row in attempts))
        from qdmnp.observables.article_fit_seed import native_material_seed
        g, m = self.config["geometry"], self.config["material"]
        count = max(v["shape_check_mode_candidates"])
        modes, provenance = native_material_seed(
            c_nm=g["c_nm"]*(1-v["shape_relative_offset"]), a_nm=g["a_nm"],
            eps_m=self.config["medium"]["relative_permittivity"], n_modes=count,
            fit_window_eV=(m["fit_min_eV"], m["fit_max_eV"]),
            fit_refinement=m.get("refinement"), orientation="long",
        )
        cfg = deepcopy(config)
        cfg["material"]["refinement"]["initial_modes_eV"] = {"long": modes}
        # The independently validated repair uses the finest declared spatial
        # order, keeping material-fit error distinct from series truncation.
        selected_overrides["spatial-order-max"] = ceiling
        candidate = certify(cfg, count, "native_neighbor_minimax")
        candidate[3]["material_seed"] = provenance
        return candidate

    def refine_sensitivity_spatial_failure(self, item, outcome):
        """Retry only a parameter-variation spatial rejection, within its limit."""
        if not item["label"].endswith(("_low", "_high")):
            return outcome
        label, module, arguments = item["spec"]
        arguments = dict(arguments)
        ceiling = min(self.config["numerics"]["max_spatial_order"],
                      spatial_order_limit(arguments["channels"]))
        while isinstance(outcome, StageError) and "spatial" in str(outcome).lower():
            order = arguments["spatial-order-max"]
            if order >= ceiling:
                break
            item["extra"].setdefault("spatial_refinement_attempts", []).append(
                {"spatial_order": order, "error": str(outcome)})
            arguments["spatial-order-max"] = min(2*order, ceiling)
            item["extra"]["spatial_order_used"] = arguments["spatial-order-max"]
            try:
                outcome = self.step(label, module, arguments)
            except StageError as exc:
                outcome = exc
        return outcome

    def validation(self):
        selection = self.state["selection"]
        channels = list(dict.fromkeys([selection["best_channel"], selection["runner_up_channel"], selection["control_channel"]]))
        # The recommendation depends on resolved thresholds of the leader and the
        # nearest competitor. The control is a contrast: a strongly suppressing
        # control is expected to lie above the scanned fluence range, and that
        # censoring must not invalidate the numerical checks of the leader.
        required = list(dict.fromkeys([selection["best_channel"], selection["runner_up_channel"]]))
        gaps = sorted(set([selection["best_gap_nm"], selection["runner_up_gap_nm"]]))
        master, _ = read_npz(self.state["threshold_master"])
        # The same inputs and grid were already solved in the master scan.
        baseline = threshold_subset(master, channels, gaps)
        # Frequency-domain checks run now (seconds, they gate some threshold checks);
        # all threshold checks are planned in order and then solved concurrently.
        planned = []

        def check(label, cfg=None, overrides=None, extra=None, **kwargs):
            check_gaps = kwargs.pop("gaps", gaps)
            arguments = {**self.threshold_arguments(config=cfg, gaps=check_gaps, channels=channels, **kwargs),
                         **(overrides or {})}
            item = {"label": label, "gaps": check_gaps, "extra": extra or {},
                    "spec": ("check_"+label, PREFIX+"calculate_threshold_fluence_gap", arguments)}
            planned.append(item)
            return item

        for carrier in self.config["pulse"]["carrier_scan_eV"]:
            if np.isclose(carrier, self.config["pulse"]["carrier_energy_eV"], rtol=0, atol=1e-12):
                # The reference carrier is the master scan itself.
                item = {"label": "carrier_"+str(carrier), "record": {
                    "label": "carrier_"+str(carrier), "artifact": self.state["threshold_master"], "derived_from_master": True,
                    **compare_thresholds(baseline, baseline, required_channels=required)}}
                planned.append(item)
            else:
                cfg = deepcopy(self.config)
                cfg["pulse"]["carrier_energy_eV"] = carrier
                item = check("carrier_"+str(carrier), cfg)
            item["carrier_energy_eV"] = carrier
        if self.config["validation"]["enabled"]:
            # Certify the higher-N linear response before its nonlinear check.
            higher = self.config["material"]["validation_modes"]
            try:
                for gap in gaps:
                    self.material_spectrum("check_higher_N_spectrum", gap, count=higher)
                check("higher_N", count=higher)
            except StageError as exc:
                planned.append({"label": "higher_N", "error": str(exc)})
            check("fluence_refined", points=2*self.fluence_points-1)
            original_order = self.spatial_order
            limit = spatial_order_limit(channels)
            # When the kernel ceiling prevents doubling, compare with half the
            # order instead: the threshold change
            # between n/2 and n bounds the remaining spatial error at n. The n/2 model
            # only reports its own half-order certificate; the production order keeps
            # the strict one in the master scan.
            refined = 2*original_order <= limit
            self.spatial_order = 2*original_order if refined else max(1, original_order//2)
            try:
                check("spatial_refined",
                      overrides=None if refined else {"spatial-convergence-policy": "warn"},
                      extra={"spatial_order_check": {"production_order": original_order, "check_order": self.spatial_order,
                                                     "direction": "refined" if refined else "coarsened",
                                                     "kernel_order_limit": limit}})
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
                    planned.append({"label": "energy_refined", "record": {
                        "label": "energy_refined", "artifact": str(path), "reference_artifact": str(reference),
                        "gap_nm": gap, **result}})
                    self.energy_points = original_points
            except StageError as exc:
                planned.append({"label": "energy_refined", "error": str(exc)})
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
                        # Preserve the declared shape-specific limits. A rejected
                        # material representation may use the configured numerical
                        # fallback, but must pass the same final-spectrum checks.
                        relaxed = {"max-modal-normalized-rms": v["shape_check_max_modal_nrms"],
                                   "max-modal-relative-error": v["shape_check_max_modal_relative_error"],
                                   "spatial-convergence-rtol": v["shape_check_spatial_rtol"]}
                        label = parameter+"_"+suffix
                        try:
                            cfg, used, overrides, details = self.shape_validation_candidate(label, cfg, gaps, relaxed)
                        except (StageError, RuntimeError, ValueError) as exc:
                            planned.append({"label": label, "error": str(exc)})
                            continue
                        check(label, cfg, count=used, overrides=overrides, extra=details)
                        continue
                    check(parameter+"_"+suffix, cfg)

        specs = [item["spec"] for item in planned if "spec" in item]
        # Each check already uses up to 2*channels*gaps worker processes; run as many
        # checks side by side as the CPU budget allows.
        per_check = 2*len(channels)*len(gaps)
        budget = resolve_workers(self.config["numerics"]["workers"], 10**6)
        concurrency = max(1, budget // max(1, per_check)) if self.config["numerics"]["workers"] != 1 else 1
        outcomes = self.steps_concurrently(specs, concurrency)
        records = []
        for item in planned:
            if "record" in item:
                record = item["record"]
            elif "error" in item:
                record = {"label": item["label"], "accepted": False, "error": item["error"]}
            else:
                outcome = self.refine_sensitivity_spatial_failure(item, outcomes[item["spec"][0]])
                if isinstance(outcome, Exception):
                    # Preserve a failed scientific check as a negative outcome. Never
                    # label the recommendation certified just because other plots exist.
                    record = {"label": item["label"], "accepted": False, "error": str(outcome)}
                else:
                    record = {"label": item["label"], "artifact": str(outcome),
                              **compare_thresholds(baseline, outcome, required_channels=required)}
                    below = [g for g in item["gaps"] if g < self.config["geometry"]["locality_advisory_gap_nm"]]
                    if below:
                        record["gaps_below_locality_advisory_gap_nm"] = below
            if "carrier_energy_eV" in item:
                record["carrier_energy_eV"] = item["carrier_energy_eV"]
            record.update(item.get("extra", {}))
            records.append(record)
        numerical_labels = {"higher_N", "fluence_refined", "spatial_refined", "time_step_refined", "energy_refined"}
        tolerance = self.config["numerics"]["threshold_relative_tolerance"]
        for row in records:
            if row["label"] in numerical_labels and "max_relative_threshold_change" in row:
                row["accepted"] = row["resolved_pairs_complete"] and row["max_relative_threshold_change"] <= tolerance
        self.state["validation"] = records
        self.state["carrier_scan_summary"] = summarize_carrier_scan(records, selection)
        present_labels = {row["label"] for row in records}
        self.state["numerical_validation_accepted"] = bool(self.config["validation"]["enabled"]
            and numerical_labels <= present_labels and all(
                row.get("accepted", False) for row in records if row["label"] in numerical_labels))
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
        points = self.fluence_points
        if self.state.get("spectral_master") and not self.config["smoke"]:
            # Fig. 5 plots whole grid curves; start from the grid predicted to pass the
            # midpoint audit (a failed audit discards every solve of both branches).
            spectral, _ = read_npz(self.state["spectral_master"])
            prediction = predict_fluence_points(self.config, spectral.get("excitation_gain_optimized"), points)
            self.state["fig05_fluence_grid_prediction"] = prediction
            points = max(points, prediction["predicted_points"])
        while True:
            try:
                data = self.step("fig05_material_fluence", PREFIX+"calculate_excitation_fluence_material_comparison", {
                    "preset": "publication", **self.common(), **self.fit_args(), **self.temporal_args(),
                    "gap-nm": gap, "pulse-energy-ev": p["carrier_energy_eV"],
                    "fluence-min-j-cm2": p["fluence_min_J_cm2"], "fluence-max-j-cm2": p["fluence_max_J_cm2"],
                    "points": points, "grid-scale": "sqrt", "target-population": p["target_population"],
                    "max-fluence-grid-midpoint-error": p["max_fluence_midpoint_population_error"],
                    "max-isolated-pulse-area-step-rad": p["max_isolated_pulse_area_step_rad"],
                    "fluence-grid-convergence-policy": self.policy, "workers": self.config["numerics"]["workers"]})
                break
            except StageError as exc:
                # The full P_exc(F) curves are plotted, so the grid audit covers the
                # whole range; refine like the threshold master instead of stopping.
                if "grid" in str(exc).lower() and "fluence" in str(exc).lower() and points < p["max_fluence_points"]:
                    points = min(2*points-1, p["max_fluence_points"])
                    continue
                raise
        self.state["fig05_fluence_points"] = points
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
        for key in ("population-decay-policy", "max-population-decay-fraction-at-read", "bright-fit-quality-policy", "dark-reduction"):
            temporal.pop(key)
        temporal["post-fs"] = w["post_fs"]
        # S1 Fourier-transforms the trajectory: keep 20 samples per excited-band cycle
        # (0.1 fs) even when the population observables use the coarser setting.
        temporal["points-per-fastest-cycle"] = max(20, temporal["points-per-fastest-cycle"])
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
        while True:
            try:
                path = self.step("supp01_work_spectrum", PREFIX+"calculate_work_loss_spectrum_material_comparison", args,
                                 dependencies=[Path(self.state["fluence_material_artifact"])])
                break
            except StageError as exc:
                # A narrow hybrid feature needs a denser energy grid; refine like Fig. 2.
                if "energy grid" in str(exc).lower() and args["energy-points"] < w["max_points"]:
                    args["energy-points"] = min(2*args["energy-points"]-1, w["max_points"])
                    continue
                raise
        self.state["supp01_energy_points"] = args["energy-points"]
        self.state["work_spectrum_artifact"] = str(path)
        self.audit_fit_identity(path)
        self.plot("supp01", "work_loss_spectrum_material_comparison", path)

    def article_diagnostics(self):
        """Regenerate the additional field and gain plots from this run alone."""
        cfg, selection = self.config, self.state["selection"]
        self.step("tip_side_fields", PREFIX+"article_field_diagnostic", {
            "c-nm": cfg["geometry"]["c_nm"], "a-nm": cfg["geometry"]["a_nm"],
            "eps-m": cfg["medium"]["relative_permittivity"],
            "carrier-energy-ev": cfg["pulse"]["carrier_energy_eV"],
            "energy-min-ev": min(1.6, cfg["pulse"]["carrier_energy_eV"]),
            "energy-max-ev": max(2.2, cfg["pulse"]["carrier_energy_eV"]),
            "qd-radius-nm": cfg["geometry"]["qd_radius_nm"],
            "gap-nm": selection["best_gap_nm"], "dpi": cfg["output"]["dpi"],
        }, suffix=".png")
        spectral, thresholds = Path(self.state["spectral_master"]), Path(self.state["threshold_master"])
        self.step("fixed_and_pulse_gain", PREFIX+"plot_fixed_and_pulse_gain", {
            "spectral-artifact": spectral, "threshold-artifact": thresholds,
            "dpi": cfg["output"]["dpi"],
        }, suffix=".png", dependencies=[spectral, thresholds])

    def report(self):
        verdict = bool(not self.config["smoke"] and self.state.get("numerical_validation_accepted", False)
                       and self.state["selection"]["resolved_threshold_selected"]
                       and self.state.get("ranking_validation", {}).get("accepted", False))
        difference = self.state.get("fig03_fig05_threshold_relative_difference")
        verdict = bool(verdict and difference is not None and difference <= self.config["numerics"]["threshold_relative_tolerance"])
        report = {"scope": "conditional single-MNP local-QS recommendations, not Shah dimer reproduction or PL quantum yield",
                  "numerically_checked_recommendation": verdict, "smoke": self.config["smoke"],
                  "interpretation": recommendation_notes(self.state, self.audit),
                  "units": self.audit, "results": self.state, "figures": self.manifest["figures"]}
        write_json(self.directory / "article_results.json", report)
        self.manifest["recommendation_accepted"] = verdict
        banner = "SMOKE: техническая проверка цепочки, не данные для статьи." if self.config["smoke"] else (
            "Проверки численной сходимости и устойчивости выбранного кандидата на заданных сетках пройдены; глобальный оптимум и физическая применимость не удостоверены." if verdict else
            "Есть неразрешённые проверки или пороги. Графики не удостоверяют однозначную рекомендацию; см. article_results.json и validation.json.")
        notes = "".join(f"<li>{html.escape(note)}</li>" for note in report["interpretation"])
        figures = "\n".join(f'<figure><figcaption>{html.escape(name)}</figcaption><a href="{html.escape(path)}"><img src="{html.escape(path)}" loading="lazy"></a></figure>' for name,path in self.manifest["figures"].items())
        page = f'<!doctype html><html lang="ru"><meta charset="utf-8"><title>QD–MNP article results</title><style>body{{font:16px sans-serif;max-width:1200px;margin:2em auto}}img{{max-width:100%}}figure{{margin:2em 0}}aside{{padding:1em;background:#fff2cf}}</style><h1>КТ и одна золотая МНЧ</h1><aside>{banner}</aside><p>Входы: resolved_inputs.json. Единицы: units.json. Численные результаты: article_results.json. Журнал: manifest.json и logs/.</p><h2>Условия интерпретации</h2><ul>{notes}</ul>{figures}</html>'
        (self.directory / "index.html").write_text(page, encoding="utf-8")
        print(f"\nResults and figures: {self.directory / 'index.html'}", flush=True)

    def run(self):
        # Replaying orchestration on resume reuses verified artifacts and rebuilds
        # dynamic choices deterministically; no mutable state is blindly trusted.
        with material_fit_cache(self.directory / "material_fit_cache"), diagnostic_figure_label(self.config["smoke"]):
            self.prefit_materials()
            for stage in STAGES:
                getattr(self, stage)()
                self.save()
                if self.stop_after == stage:
                    self.manifest["status"] = "stopped_after_"+stage
                    self.save()
                    return
        self.manifest["status"] = "complete_diagnostic" if self.config["smoke"] else "complete"
        self.save()


def retardation_selection_mask(config: dict, channels, gaps) -> tuple[np.ndarray, np.ndarray]:
    """k_m R per channel/gap and the admissible set from the k_m R and k_m c cutoffs.

    Admissibility is decided only by the quasi-static cutoffs, which bound the
    model's own numerical validity. The declared locality gap
    (geometry.locality_advisory_gap_nm) is reported next to the result and never
    removes a gap from the search, so an optimum is found by the model rather
    than fixed by the assumption.
    """
    g, v = config["geometry"], config["validation"]
    gaps = np.asarray(gaps, float)
    k = medium_wavenumber_per_nm(config, upper_study_energy_eV(config))
    kR = np.array([[k*center_distance_nm(config, str(ch), gap) for gap in gaps] for ch in channels], float)
    allowed = (kR <= v["max_k_R_for_selection"]) & (k*g["c_nm"] <= v["max_k_c_for_selection"])
    return kR, allowed


def select_scenarios(data: dict, config: dict) -> dict:
    channels, gaps = list(data["channel_id"].astype(str)), np.asarray(data["gap_nm"], float)
    thresholds = np.asarray(data["threshold_fluence_j_cm2"][1], float)
    statuses = np.asarray(data["threshold_status"][1]).astype(str)
    resolved = resolved_threshold_mask(statuses) & np.isfinite(thresholds) & (thresholds>0)
    g = config["geometry"]
    k = medium_wavenumber_per_nm(config, upper_study_energy_eV(config))
    kR, allowed = retardation_selection_mask(config, channels, gaps)
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
    far_agrees = bool(valid_discrepancy.size and np.isfinite(valid_discrepancy[0]) and valid_discrepancy[0]<=config["numerics"]["dd_tolerance"])
    boundary, boundary_status, support = dd_validity_assessment(gaps, discrepancy, config["numerics"]["dd_tolerance"], allowed[ci])
    control = controls[channels[ci]]
    control_index = channels.index(control)
    best_gap = float(gaps[gi])
    # An interior optimum is the model's own answer; an optimum at either end of
    # the sampled admissible range only bounds it and must be reported as such.
    usable = np.flatnonzero(allowed[ci] & resolved[ci])
    if not candidates or usable.size < 3:
        optimum_kind = "not_determined"
    elif gi == usable[0]:
        optimum_kind = "at_smallest_sampled_gap"
    elif gi == usable[-1]:
        optimum_kind = "at_largest_sampled_gap"
    else:
        optimum_kind = "interior"
    # How much of the optimum needs the region the locality assumption doubts.
    advisory = float(g["locality_advisory_gap_nm"])
    above = [j for j in usable if gaps[j] >= advisory]
    gain_below = (float(thresholds[ci, min(above)]/thresholds[ci, usable[0]])
                  if above and usable.size and gaps[usable[0]] < advisory else None)
    return {"best_channel": channels[ci], "best_gap_nm": best_gap, "runner_up_channel":channels[rci],
            "runner_up_gap_nm":float(gaps[rgi]), "control_channel":control, "far_gap_nm":far,
            "resolved_threshold_selected": bool(candidates), "selection_is_illustration_only":not bool(candidates),
            "far_point_satisfies_threshold_DD_tolerance": far_agrees,
            # The largest admissible gap agrees whenever any admissible agreement
            # suffix exists; otherwise Fig. 6 "far" is labelled as a contrast only.
            "far_gap_role": "dd_fqs_threshold_agreement" if far_agrees else "largest_admissible_gap_without_dd_agreement",
            "threshold_dd_validity_gap_nm": boundary, "threshold_dd_validity_status": boundary_status,
            "threshold_dd_validity_supporting_points": support,
            "locality_advisory_gap_nm": float(g["locality_advisory_gap_nm"]),
            "threshold_gain_below_locality_advisory": gain_below,
            "best_gap_below_locality_advisory": bool(best_gap < g["locality_advisory_gap_nm"]),
            "best_gap_at_lower_admissible_bound": bool(candidates) and bool(usable_gaps.size) and np.isclose(best_gap, usable_gaps[0]),
            "best_gap_optimum_kind": optimum_kind,
            "best_channel_gap_profile": [{"gap_nm": float(gap), "threshold_J_cm2": float(thresholds[ci, j]) if resolved[ci, j] else None,
                                          "status": statuses[ci, j], "admissible": bool(allowed[ci, j])} for j, gap in enumerate(gaps)],
            "control_threshold_status_at_best_gap": statuses[control_index, gi],
            "point_qd_field_variation_lower_bound_at_best": 3*g["qd_radius_nm"]/center_distance_nm(config, channels[ci], best_gap),
            "k_c_at_upper_study_energy":float(k*g["c_nm"]), "k_R_by_channel_gap":kR,
            "selection_allowed_by_declared_retardation_cutoffs":allowed,
            "physical_scope":"conditional point-QD/local-QS result; declared k cutoffs do not validate finite-QD or nonlocal physics"}


def choose_preflight_gaps(gaps, allowed, discrepancy, tolerance):
    """Near, transition and far gaps inside the admissible range for every channel.

    near/far are the extreme gaps admissible for all channels; the transition gap
    is the intermediate gap whose worst-channel spectral DD/FQS discrepancy is
    closest (logarithmically) to the DD tolerance.
    """
    gaps = np.asarray(gaps, float)
    common = gaps[np.all(np.asarray(allowed, bool), axis=0)]
    if not common.size:
        raise StageError("No gap is admissible for every channel; revise gaps_nm or the k_m R cutoff.")
    near, far, middle = float(common[0]), float(common[-1]), common[1:-1]
    if not middle.size:
        return sorted({near, far}), "fewer than three admissible gaps: near and far only"
    if discrepancy is None:
        transition = float(middle[middle.size//2])
        rule = "transition = middle admissible gap (no spectral DD/FQS discrepancy available)"
    else:
        delta = np.asarray(discrepancy, float)
        worst = np.array([np.nanmax(np.where(np.isfinite(delta[:, j]), delta[:, j], np.inf))
                          for j in np.flatnonzero(np.isin(gaps, middle))])
        score = np.abs(np.log(np.maximum(worst, np.finfo(float).tiny)/tolerance))
        transition = float(middle[int(np.argmin(score))])
        rule = "transition = admissible gap whose worst-channel delta_spec is closest to the DD tolerance"
    return [near, transition, far], rule


def predict_fluence_points(config, excitation_gain, start_points):
    """Smallest nested sqrt(F) grid (n -> 2n-1) predicted to pass the midpoint audit.

    Weak-field population of the strongest hybrid scales as G * theta_0^2, so its
    pulse-area step is sqrt(G) times the isolated step. For sin^2(theta/2) the
    audited midpoint error is bounded by (2*dtheta)^2/8 * 1/2 = dtheta^2/4.
    The prediction only chooses the starting grid; the saved audit still decides.
    """
    p, audit = config["pulse"], unit_audit(config)
    si = audit["SI"]
    area_reference = (si["qd_dipole_C_m"]*audit["reference_field_V_m"]*np.sqrt(2*np.pi)
                      * si["pulse_sigma_field_fs"]*1e-15/hbar)
    area_per_sqrt_fluence = area_reference/np.sqrt(audit["reference_fluence_J_cm2"])
    gain = np.asarray(excitation_gain if excitation_gain is not None else [1.0], float)
    gain_max = float(np.nanmax(gain[np.isfinite(gain)])) if np.any(np.isfinite(gain)) else 1.0
    gain_max = max(gain_max, 1.0)
    limit = min(0.9*2*np.sqrt(p["max_fluence_midpoint_population_error"]), p["max_isolated_pulse_area_step_rad"]*np.sqrt(gain_max))
    span = np.sqrt(p["fluence_max_J_cm2"]) - np.sqrt(p["fluence_min_J_cm2"])
    points = int(start_points)
    while True:
        step = area_per_sqrt_fluence*span/(points-1)*np.sqrt(gain_max)
        if step <= limit or points >= p["max_fluence_points"]:
            break
        points = min(2*points-1, p["max_fluence_points"])
    return {"predicted_points": points, "max_weak_field_gain": gain_max,
            "predicted_hybrid_area_step_rad": float(step), "predicted_midpoint_error": float(step**2/4),
            "rule": "dtheta_hybrid = dtheta_isolated*sqrt(max G_exc); midpoint error <= dtheta^2/4; audit remains authoritative"}


def censoring_extension(data: dict, config: dict) -> tuple[bool, bool]:
    """Decide whether a censored threshold can change the recommendation.

    A threshold below the scan (left censoring) can conceal a better candidate, so
    the lower bound is extended. A hybrid threshold above the scan exceeds the
    isolated threshold and cannot win; extending the upper bound is needed only
    when the isolated reference or every admissible FQS threshold is unresolved.
    """
    statuses = np.asarray(data["threshold_status"]).astype(str)
    isolated = str(data["isolated_threshold_status"])
    extend_min = bool(isolated in LEFT_CENSORED or np.any(np.isin(statuses, LEFT_CENSORED)))
    channels, gaps = list(np.asarray(data["channel_id"]).astype(str)), np.asarray(data["gap_nm"], float)
    _, allowed = retardation_selection_mask(config, channels, gaps)
    any_resolved = bool(np.any(resolved_threshold_mask(statuses[1]) & allowed))
    extend_max = bool(isolated in RIGHT_CENSORED or not any_resolved)
    return extend_min, extend_max


def threshold_subset(master: dict, channels, gaps) -> dict:
    """Slice a threshold master to the channel/gap ordering of a validation check."""
    master_channels = list(np.asarray(master["channel_id"]).astype(str))
    ci = [master_channels.index(ch) for ch in channels]
    master_gaps = np.asarray(master["gap_nm"], float)
    gi = [int(np.flatnonzero(np.isclose(master_gaps, gap, rtol=0, atol=1e-12))[0]) for gap in gaps]
    return {"channel_id": np.asarray(channels), "gap_nm": np.asarray(gaps, float),
            "threshold_fluence_j_cm2": np.asarray(master["threshold_fluence_j_cm2"])[:, ci][:, :, gi],
            "threshold_status": np.asarray(master["threshold_status"])[:, ci][:, :, gi],
            "isolated_threshold_fluence_j_cm2": np.asarray(master["isolated_threshold_fluence_j_cm2"]),
            "isolated_threshold_status": np.asarray(master["isolated_threshold_status"])}


def summarize_carrier_scan(records, selection) -> dict:
    """Best tested carrier for the selected configuration; no post-hoc figure change."""
    rows = []
    for row in sorted((r for r in records if "carrier_energy_eV" in r and "candidate_own_thresholds" in r),
                      key=lambda r: r["carrier_energy_eV"]):
        ci = row["channels"].index(selection["best_channel"])
        gi = [float(g) for g in row["gaps_nm"]].index(float(selection["best_gap_nm"]))
        value = row["candidate_own_thresholds"][ci][gi]
        value = float(value) if value is not None and np.isfinite(value) else None
        isolated = row.get("isolated_threshold")
        isolated = float(isolated) if isolated is not None and np.isfinite(isolated) and isolated > 0 else None
        rows.append({"carrier_energy_eV": row["carrier_energy_eV"], "threshold_J_cm2": value,
                     "isolated_threshold_J_cm2": isolated,
                     "ratio_to_isolated": value/isolated if value is not None and isolated is not None else None})
    resolved = [r for r in rows if r["threshold_J_cm2"] is not None]
    if not resolved:
        return {"rows": rows, "best_tested_carrier_eV": None, "note": "no resolved threshold in the carrier scan"}
    best = min(resolved, key=lambda r: r["threshold_J_cm2"])
    ratios = [r for r in resolved if r["ratio_to_isolated"] is not None]
    best_ratio = min(ratios, key=lambda r: r["ratio_to_isolated"]) if ratios else None
    energies = [r["carrier_energy_eV"] for r in rows]
    return {"rows": rows, "best_tested_carrier_eV": best["carrier_energy_eV"],
            "best_tested_carrier_is_interior": best["carrier_energy_eV"] not in (min(energies), max(energies)),
            "largest_relative_gain_carrier_eV": None if best_ratio is None else best_ratio["carrier_energy_eV"],
            "note": "figures keep the pre-registered carrier; an edge optimum only bounds the tested range"}


def recommendation_notes(state: dict, audit: dict) -> list[str]:
    """Plain statements that must accompany any recommendation drawn from this run."""
    notes = []
    selection = state.get("selection", {})
    ranking = state.get("ranking_validation", {})
    span = ranking.get("recommended_gap_range_nm")
    if ranking.get("recommendation_is_a_range") and span:
        notes.append(f"Порог насыщается: зазоры {span[0]}–{span[1]} нм неразличимы в пределах численного допуска "
                     f"{ranking.get('relative_numerical_allowance')}, поэтому рекомендуется диапазон, а не одна точка; "
                     "всё вне этого диапазона отделено от победителя.")
    kind, best = selection.get("best_gap_optimum_kind"), selection.get("best_gap_nm")
    advisory = selection.get("locality_advisory_gap_nm")
    if kind == "interior":
        notes.append(f"Оптимальный зазор {best} нм — внутренний оптимум просканированного допустимого диапазона: "
                     "он найден моделью, а не задан нижней границей сетки.")
    elif kind == "at_smallest_sampled_gap":
        notes.append(f"Лучший зазор {best} нм совпадает с наименьшим просканированным допустимым зазором: "
                     "это краевой, а не внутренний оптимум; сетку зазоров нужно продлить вниз.")
    elif kind == "at_largest_sampled_gap":
        notes.append(f"Лучший зазор {best} нм совпадает с наибольшим просканированным допустимым зазором: "
                     "это краевой, а не внутренний оптимум; сетку зазоров нужно продлить вверх.")
    gain = selection.get("threshold_gain_below_locality_advisory")
    if gain is not None:
        notes.append(f"Ниже объявленного порога локальности {advisory} нм порог падает ещё лишь в {gain:.2f} раза "
                     f"(от {advisory} нм до наименьшего просканированного зазора): профиль монотонный, но насыщающийся, "
                     "поэтому краевой характер оптимума не означает заметного выигрыша от дальнейшего сближения.")
    if selection.get("best_gap_below_locality_advisory"):
        notes.append(f"Выбранный зазор {best} нм меньше объявленного порога локальности "
                     f"{advisory} нм: ниже него локальный континуальный отклик золота, отсутствие туннелирования "
                     "и точечная КТ без лигандной оболочки не защитимы; значение остаётся расчётным, а не рекомендуемым.")
    if selection.get("far_gap_role") == "largest_admissible_gap_without_dd_agreement":
        notes.append("Дальний случай рис. 6 — наибольший допустимый зазор без согласия DD/FQS по порогу; "
                     "граница применимости DD в допустимой по k_m R области не установлена.")
    bound = selection.get("point_qd_field_variation_lower_bound_at_best")
    if bound is not None:
        notes.append(f"Точечная КТ: r_QD|∇E|/|E| ≥ {bound:.2f} уже для дипольного поля у выбранного зазора (не ≪ 1).")
    carrier = state.get("carrier_scan_summary", {})
    if carrier.get("best_tested_carrier_eV") is not None:
        edge = "" if carrier.get("best_tested_carrier_is_interior") else " (на краю сетки несущих — только граница)"
        notes.append(f"Лучшая проверенная несущая: {carrier['best_tested_carrier_eV']} эВ{edge}; рисунки построены при заранее заданной несущей.")
    estimates = audit.get("model_error_estimates", {})
    if "long" in estimates:
        long = estimates["long"]
        damping = long["surface_damping"]
        notes.append(
            "Не учтённые моделью эффекты (оценка для продольной поляризуемости на несущей): запаздывание "
            f"×{long['alpha_squared_ratio_retardation_at_carrier']:.2f} для |α|², сдвиг LSPR {long['lspr_shift_meV']:.0f} мэВ; "
            f"поверхностное затухание ×{damping[-1]['alpha_squared_ratio_at_carrier']:.2f}…×{damping[0]['alpha_squared_ratio_at_carrier']:.2f}. "
            "Абсолютные усиления и пороги имеют систематическую неопределённость этого порядка; сравнение DD/FQS — нет.")
    return notes


def _threshold_arrays(source):
    return source if isinstance(source, dict) else read_npz(source)[0]


def compare_thresholds(reference_path, candidate_path, required_channels=None):
    """Compare FQS thresholds of a check with its reference on identical channel/gap ordering.

    ``required_channels`` must be resolved in both scans (default: every channel).
    Other channels (the physical control) are compared when resolved and otherwise
    only need a consistent censoring class; they never block the numerical check.
    """
    reference = _threshold_arrays(reference_path)
    candidate = _threshold_arrays(candidate_path)
    if not np.array_equal(np.asarray(reference["channel_id"]).astype(str), np.asarray(candidate["channel_id"]).astype(str)) or np.shape(reference["threshold_status"]) != np.shape(candidate["threshold_status"]):
        raise ValueError("Threshold comparison requires matching channel and gap-index ordering.")
    channels = list(np.asarray(candidate["channel_id"]).astype(str))
    required = np.isin(channels, channels if required_channels is None else list(required_channels))
    left, right = np.asarray(reference["threshold_fluence_j_cm2"][1], float), np.asarray(candidate["threshold_fluence_j_cm2"][1], float)
    left_status = np.asarray(reference["threshold_status"][1]).astype(str)
    right_status = np.asarray(candidate["threshold_status"][1]).astype(str)
    left_ok = resolved_threshold_mask(left_status) & np.isfinite(left) & (left>0)
    right_ok = resolved_threshold_mask(right_status) & np.isfinite(right) & (right>0)
    valid = left_ok & right_ok
    required_pairs = np.broadcast_to(required[:, None], valid.shape)
    difference = np.abs(right[valid & required_pairs]/left[valid & required_pairs]-1)
    optional = ~required_pairs
    optional_difference = np.abs(right[valid & optional]/left[valid & optional]-1)
    same_class = (np.isin(left_status, RIGHT_CENSORED) & np.isin(right_status, RIGHT_CENSORED)) | (np.isin(left_status, LEFT_CENSORED) & np.isin(right_status, LEFT_CENSORED))
    optional_consistent = bool(np.all((valid | same_class)[optional]))
    masked = np.where(valid,right,np.nan)
    own = np.where(right_ok, right, np.nan)
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
    return {"accepted":bool(np.any(valid & required_pairs)),
            "resolved_pairs_complete":bool(np.all(valid[required_pairs]) and isolated_valid),
            "all_pairs_resolved":bool(np.all(valid) and isolated_valid),
            "required_channels":[ch for ch, flag in zip(channels, required) if flag],
            "max_relative_threshold_change":max(float(np.max(difference)), isolated_change) if difference.size else float("inf"),
            "isolated_relative_threshold_change":isolated_change,
            "nonrequired_max_relative_threshold_change":float(np.max(optional_difference)) if optional_difference.size else None,
            "nonrequired_pairs_consistent":optional_consistent,
            "channels":channels, "gaps_nm":np.asarray(candidate["gap_nm"], float).tolist(),
            "candidate_thresholds":masked.tolist(), "candidate_own_thresholds":own.tolist(), "isolated_threshold":isolated,
            "candidate_status":right_status.tolist(),
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
    # Candidates the scan cannot tell apart from the winner form a resolution
    # plateau. Where the threshold saturates there is no unique best gap, and the
    # defensible claim is the plateau itself rather than an arbitrary point in
    # it; the claim stays safe as long as everything OUTSIDE the plateau is worse
    # than the winner beyond the allowance. Requiring separation from the
    # plateau's own edge instead would be unsatisfiable on any smooth profile
    # once the grid is fine enough.
    plateau = allowed & resolved & (values*(1-allowance)<=winner*(1+allowance))
    equivalent = [{"channel":channels[aci], "gap_nm":float(gaps[agi]), "threshold_J_cm2":float(values[aci,agi])}
                  for aci, agi in zip(*np.where(plateau))] if winner_resolved else []
    plateau_gaps = sorted(float(gaps[agi]) for aci, agi in zip(*np.where(plateau)) if aci == ci)
    indices = sorted(int(agi) for aci, agi in zip(*np.where(plateau)) if aci == ci)
    # A plateau that jumps channels or skips a gap is not one flat region.
    plateau_is_one_region = bool(winner_resolved and np.all(np.where(plateau)[0] == ci)
                                 and indices == list(range(indices[0], indices[-1]+1)) and gi in indices)
    outside = allowed & resolved & ~plateau
    separated = bool(winner_resolved and plateau_is_one_region and not np.any(uncertain) and allowance<1
                     and np.all(values[outside]*(1-allowance)>winner*(1+allowance)))
    expected_sensitivity = {name+"_"+suffix for name in ("gap", "c", "a", "exciton", "gamma1", "dephasing") for suffix in ("low", "high")}
    # Laser detuning is controllable: by default the carrier scan informs the choice
    # of E_L (carrier_scan_summary) while uncontrolled sample inputs gate robustness.
    carriers_gate = bool(config["validation"].get("carrier_scan_in_ranking", True))
    sensitivity = [row for row in records if row["label"] in expected_sensitivity
                   or (carriers_gate and "carrier_energy_eV" in row)]
    present = {row["label"] for row in sensitivity}
    carrier_values = {row["carrier_energy_eV"] for row in sensitivity if "carrier_energy_eV" in row}
    sensitivity_complete = bool(expected_sensitivity<=present
                                and (not carriers_gate or set(config["pulse"]["carrier_scan_eV"])<=carrier_values))
    ranking_preserved = bool(sensitivity_complete and all(row.get("accepted", False)
        and row.get("resolved_pairs_complete", False) and row.get("best_configuration_unchanged", False) for row in sensitivity))

    def sensitivity_separated(row):
        values = np.asarray(row.get("candidate_own_thresholds", row.get("candidate_thresholds", [])), float)
        status = np.asarray(row.get("candidate_status", []), dtype=str)
        if status.shape == values.shape and values.size:
            # A threshold above the scanned range is a lower bound larger than the
            # scan maximum: it is separated from any resolved winner inside the scan.
            above = np.isin(status, RIGHT_CENSORED)
            resolved = resolved_threshold_mask(status) & np.isfinite(values) & (values>0)
            if not np.all(resolved | above) or not np.any(resolved):
                return False
            candidates = np.where(above, np.inf, values).ravel()
        else:
            candidates = values.ravel()
            if not candidates.size or not np.all(np.isfinite(candidates)) or np.any(candidates<=0):
                return False
        ordered = np.sort(candidates)
        return bool(allowance<1 and np.isfinite(ordered[0]) and (ordered.size==1 or ordered[0]*(1+allowance)<ordered[1]*(1-allowance)))

    separated_sensitivity = bool(sensitivity_complete and all(sensitivity_separated(row) for row in sensitivity))
    return {"accepted":bool(config["validation"]["enabled"] and numerical_accepted and separated and ranking_preserved and separated_sensitivity),
            "selected_candidate_separated":separated, "sensitivity_checks_complete":sensitivity_complete,
            "ranking_preserved_in_tested_sensitivity_cases":ranking_preserved,
            "selected_candidate_separated_in_sensitivity_cases":separated_sensitivity,
            "relative_numerical_allowance":allowance,
            "allowance_definition":"max(declared threshold tolerance, observed refinement changes); conservative resolution allowance, not a statistical confidence interval",
            "indistinguishable_grid_candidates":equivalent,
            "recommended_gap_range_nm":[plateau_gaps[0], plateau_gaps[-1]] if plateau_gaps else None,
            "recommendation_is_a_range":bool(len(plateau_gaps)>1),
            "plateau_is_one_contiguous_region_on_the_best_channel":plateau_is_one_region,
            "separation_definition":"every candidate outside the resolution plateau is worse than the winner by more than the allowance",
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
