"""Fresh native material fits used only as deterministic optimizer guesses.

This helper takes physical/numerical inputs, never an archived NPZ. It uses
the current article run's material-fit cache when that context is active. A
source fit may miss the accuracy gates: its coefficients initialize a new fit,
whose caller must enforce the normal material and observable acceptance gates.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib

import numpy as np

from qdmnp.observables.article_fit_cache import fit_key
from qdmnp.passive_fit import PassiveFitRefinement
from qdmnp.rational_fit import (
    AU_ENERGY_EV, HybridQDPlasmonModel, make_params_with_overrides,
)


def native_material_seed(
    *, c_nm: float, a_nm: float, eps_m: float, n_modes: int,
    fit_window_eV: tuple[float, float],
    fit_refinement: PassiveFitRefinement | dict | None,
    orientation: str = "long",
) -> tuple[tuple[tuple[float, float, float], ...], dict]:
    """Return native Lorentz triples in eV and a JSON-serializable provenance.

    Each triple contains strength (eV**2), energy (eV), and damping (eV).
    Existing ``initial_modes_eV`` are removed from a copy of the refinement,
    so generation starts from the native deterministic fitter. Removing the
    model's accuracy rejection does not alter its default optimizer goals
    (NRMS 0.025, maximum relative error 0.05), or accept this source fit for an
    observable calculation. ``alpha_inf`` is recorded but not transferred:
    the destination model must use its own physical high-frequency limit.
    """
    if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
        raise ValueError("n_modes must be a positive integer.")
    if orientation not in ("long", "trans"):
        raise ValueError("orientation must be long or trans.")
    if isinstance(fit_refinement, PassiveFitRefinement):
        refinement = asdict(fit_refinement)
    elif isinstance(fit_refinement, dict):
        refinement = dict(fit_refinement)
    elif fit_refinement is None:
        refinement = None
    else:
        raise ValueError("fit_refinement must be PassiveFitRefinement, a dict, or None.")
    if refinement is not None:
        refinement.pop("initial_modes_eV", None)
        refinement = PassiveFitRefinement(**refinement)
    # Like parallel.fit_material_job: material fitting is independent of QD
    # distance and dipole. A distant placeholder satisfies model validation.
    params = make_params_with_overrides(
        c_nm=c_nm, a_nm=a_nm, r_nm=c_nm + a_nm + 10.0,
        eps_m=eps_m, orientation=orientation,
    )
    model = HybridQDPlasmonModel(
        params, orientation=orientation, n_modes=int(n_modes),
        fit_window_eV=tuple(fit_window_eV), fit_refinement=refinement,
        max_fit_normalized_rms=None, max_fit_pointwise_relative_error=None,
        radiative_consistency_policy="ignore", verbose=False,
    )
    fit = model.fit
    triples = np.column_stack((
        fit.strengths_au2 * AU_ENERGY_EV**2,
        fit.omega_modes_au * AU_ENERGY_EV,
        fit.gamma_modes_au * AU_ENERGY_EV,
    ))
    # Explicit byte order/shape convention makes the coefficient receipt
    # stable across platforms and independent of compressed-NPZ timestamps.
    coefficients = np.r_[fit.alpha_inf, triples.ravel()].astype("<f8")
    errors = {
        "normalized_rms_alpha": float(fit.normalized_rms_alpha),
        "normalized_rms_inv_alpha": float(fit.normalized_rms_inv_alpha),
        "max_normalized_alpha_error": float(fit.max_normalized_alpha_error),
    }
    if not np.all(np.isfinite(list(errors.values()))):
        raise ValueError("Native material seed has non-finite error diagnostics.")
    provenance = {
        "role": "initial_guess_only",
        "source": "native_fit_from_current_run_inputs",
        "geometry_nm": {"c": float(c_nm), "a": float(a_nm), "b": float(a_nm)},
        "orientation": orientation,
        "n_modes": int(n_modes),
        "coefficient_units": ["eV^2", "eV", "eV"],
        "alpha_inf": float(fit.alpha_inf),
        "coefficient_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
        "coefficient_hash_encoding": "little-endian float64: alpha_inf, row-major (strength_eV2, energy_eV, damping_eV)",
        "native_fit_identity": fit_key(model),
        "native_fit_errors": errors,
        "native_optimizer_goals": {"max_nrms": 0.025, "max_pointwise_relative_error": 0.05},
        "native_source_meets_default_accuracy": bool(
            max(errors["normalized_rms_alpha"], errors["normalized_rms_inv_alpha"]) <= .025
            and errors["max_normalized_alpha_error"] <= .05
        ),
        "acceptance_limits_relaxed": False,
        "destination_requires_independent_refit_and_acceptance": True,
    }
    return tuple(tuple(float(value) for value in row) for row in triples), provenance
