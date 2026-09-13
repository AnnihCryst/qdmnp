"""Compare saved Lorentz coefficients, independently of solver/cache reuse."""

import hashlib

import numpy as np


def _orientation(channel):
    return "long" if str(channel).endswith("_long") else "trans"


def _entry(alpha, strengths, omega, gamma):
    strengths, omega, gamma = (np.asarray(v, dtype=float) for v in (strengths, omega, gamma))
    # Artifact padding is either NaN or zeros; physical poles have omega,gamma>0.
    valid = np.isfinite(strengths) & np.isfinite(omega) & np.isfinite(gamma) & (omega > 0) & (gamma > 0)
    if not np.any(valid) or not np.isfinite(alpha):
        raise ValueError("Missing finite material coefficients.")
    values = np.r_[float(alpha), strengths[valid], omega[valid], gamma[valid]]
    return int(np.sum(valid)), values


def coefficients(arrays, metadata):
    """Yield (orientation, N, coefficient vector) for article NPZ layouts."""
    if "orientation_ids" in arrays:
        for oi, orientation in enumerate(arrays["orientation_ids"].astype(str)):
            for bi in range(arrays["fit_alpha_inf_dimensionless"].shape[1]):
                count, values = _entry(arrays["fit_alpha_inf_dimensionless"][oi, bi],
                    *(arrays["fit_"+k][oi, bi] for k in ("strengths_au2", "omega_modes_au", "gamma_modes_au")))
                yield orientation, count, values
    elif "one__fit_alpha_inf" in arrays:
        for branch in ("one", "multi"):
            for ci, channel in enumerate(arrays["hybrid_channel_id"].astype(str)):
                count, values = _entry(arrays[branch+"__fit_alpha_inf"][ci],
                    *(arrays[branch+"__fit_"+k][ci] for k in ("strengths_au2", "omega_modes_au", "gamma_modes_au")))
                yield _orientation(channel), count, values
    elif "fit_alpha_inf" in arrays:
        channels = arrays["channel_id"].astype(str)
        alpha = arrays["fit_alpha_inf"]
        for index in np.ndindex(alpha.shape):
            # Fig.4: (branch, channel); gap masters: (channel, gap).
            ci = index[1] if "fit_model_id" in arrays else index[0]
            count, values = _entry(alpha[index],
                *(arrays["fit_"+k][index] for k in ("strengths_au2", "omega_modes_au", "gamma_modes_au")))
            yield _orientation(channels[ci]), count, values
    elif "material_fit_alpha_inf_au3" in arrays:
        # Despite the historical key suffix, this stores dimensionless fit.alpha_inf.
        for bi, ci in np.ndindex(arrays["material_fit_alpha_inf_au3"].shape):
            count, values = _entry(arrays["material_fit_alpha_inf_au3"][bi, ci],
                *(arrays["material_fit_"+k][bi, ci] for k in ("strengths_au2", "omega_modes_au", "gamma_modes_au")))
            yield _orientation(arrays["channel_key"][ci]), count, values
    else:
        for channel, document in metadata.get("model_by_channel", {}).items():
            fit = document.get("material_fit")
            if fit:
                count, values = _entry(fit["alpha_inf"], *(fit[k] for k in ("strengths_au2", "omega_modes_au", "gamma_modes_au")))
                yield _orientation(channel), count, values


def compare_fit_coefficients(reference_arrays, reference_metadata, arrays, metadata):
    reference = {(orientation, count): values for orientation, count, values in coefficients(reference_arrays, reference_metadata)}
    records = []
    for orientation, count, values in coefficients(arrays, metadata):
        expected = reference.get((orientation, count))
        if expected is None or not np.array_equal(values, expected):
            raise ValueError(f"Different Lorentz coefficients: orientation={orientation}, N={count}.")
        records.append({"orientation": orientation, "modes": count,
                        "sha256": hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()})
    if not records:
        raise ValueError("Artifact has no recognized saved Lorentz coefficients to compare.")
    return {"accepted": True, "coefficient_sets_checked": len(records), "fits": records}
