"""Opt-in, lossless memoization of the native fitter during an article run.

The mathematical model files are unmodified. A context manager temporarily
wraps their deterministic material fit, restoring the method on exit. Geometry
coupling, stability, spatial convergence and every ODE are still evaluated.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path

import numpy as np

from qd_mnp_rational_fit import HybridQDPlasmonModel, RationalLorentzFit


def fit_key(model) -> dict:
    material = model.params.material
    source = Path(__file__).resolve().parents[1] / "qd_mnp_rational_fit.py"
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "orientation": model.orientation, "L": float(model.L),
        "aspect": float(model.params.c_au / model.params.a_au), "eps_m": float(model.params.eps_m),
        "n_modes": int(model.n_modes), "window": list(model.fit_window_eV),
        "weight_center": model.weight_center_eV, "weight_sigma": model.weight_sigma_eV,
        "alpha_weight": model.alpha_objective_weight, "inverse_weight": model.inv_alpha_objective_weight,
        "seed": model.seed,
        # The fitter uses min(default gate, requested gate) for early exit.
        "optimizer_nrms": min(0.025, model.max_fit_normalized_rms or 0.025),
        "optimizer_pointwise": min(0.05, model.max_fit_pointwise_relative_error or 0.05),
        "material_sha256": hashlib.sha256(
            np.stack([material.energy_eV, material.n, material.k]).astype("<f8").tobytes()
        ).hexdigest(),
    }


def check_requested_accuracy(model, fit):
    if model.max_fit_normalized_rms is not None and max(fit.normalized_rms_alpha, fit.normalized_rms_inv_alpha) > model.max_fit_normalized_rms:
        raise RuntimeError("Cached native material fit misses requested NRMS accuracy; increase n_modes.")
    if model.max_fit_pointwise_relative_error is not None and fit.max_normalized_alpha_error > model.max_fit_pointwise_relative_error:
        raise RuntimeError("Cached native material fit misses requested pointwise accuracy; increase n_modes.")


@contextmanager
def material_fit_cache(directory: Path):
    """Reuse fits only for identical fitter inputs; use no pickle files."""
    directory.mkdir(parents=True, exist_ok=True)
    original = HybridQDPlasmonModel._fit_rational_alpha
    memory = {}

    def cached(model):
        key_json = json.dumps(fit_key(model), sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(key_json.encode()).hexdigest()
        path, receipt = directory / f"{digest}.npz", directory / f"{digest}.json"
        if digest not in memory:
            if path.exists() and receipt.exists():
                record = json.loads(receipt.read_text(encoding="utf-8"))
                if record.get("key") != key_json or record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise RuntimeError(f"Material-fit cache integrity failure: {path}")
                with np.load(path, allow_pickle=False) as archive:
                    values = {field.name: np.array(archive[field.name], copy=True) for field in fields(RationalLorentzFit)}
                values = {name: value.item() if value.ndim == 0 else value for name, value in values.items()}
                memory[digest] = RationalLorentzFit(**values)
            else:
                print(f"Fitting material N={model.n_modes}, {model.orientation}; subsequent identical fits will be reused.", flush=True)
                memory[digest] = original(model)
                temporary = path.with_suffix(".pending.npz")
                np.savez_compressed(temporary, **asdict(memory[digest]))
                temporary.replace(path)
                receipt.write_text(json.dumps({"key": key_json, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
        result = deepcopy(memory[digest])
        check_requested_accuracy(model, result)
        return result

    HybridQDPlasmonModel._fit_rational_alpha = cached
    try:
        yield
    finally:
        HybridQDPlasmonModel._fit_rational_alpha = original
