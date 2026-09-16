"""Unit-audited input for the single-MNP article workflow (SI at its boundary)."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tomllib

import numpy as np
from scipy.constants import c, epsilon_0, hbar, elementary_charge
from qd_mnp_passive_fit import PassiveFitRefinement

from qd_mnp_rational_fit import (
    AU_DIPOLE_C_M, AU_LENGTH_M, AU_TIME_S, AU_ENERGY_EV, DEBYE_C_M,
    DEFAULT_AU_MATERIAL, GaussianPulse, eV_to_au, fs_to_au,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "article_observables" / "ARTICLE_INPUTS.toml"
CHANNELS = ("axis_long", "axis_trans", "side_long", "side_trans_radial", "side_trans_tangential")


def load_inputs(path: Path = DEFAULT_INPUT, *, smoke: bool = False) -> dict:
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    validate_inputs(config)
    config = deepcopy(config)
    if smoke:
        # Same physical system; reduced numerical accuracy is NEVER a paper result.
        config["geometry"]["gaps_nm"] = [1.0, 10.0]
        config["material"].update(mode_candidates=[3], validation_modes=4,
                                  max_bright_nrms=2.0, max_bright_pointwise_error=5.0)
        config["spectrum"].update(points=101, max_points=101)
        config["pulse"].update(fluence_points=5, max_fluence_points=5,
                               max_fluence_extensions=0, population_read_fs=150.0,
                               refine_threshold=False,
                               carrier_scan_eV=[2.022, 2.042, 2.062])
        config["numerics"].update(spatial_order=2, max_spatial_order=2,
                                  modal_audit_points=101, rtol=1e-6, atol=1e-8,
                                  points_per_fastest_cycle=8,
                                  max_modal_nrms=2.0, max_modal_relative_error=5.0)
        config["work_spectrum"].update(points=51, post_fs=150.0)
        config["validation"]["enabled"] = False
        config["output"]["dpi"] = 90
    config["smoke"] = smoke
    return config


def validate_inputs(config: dict) -> None:
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported ARTICLE_INPUTS schema_version.")
    required = {
        "geometry": "mnp_count c_nm a_nm qd_radius_nm reference_surface_gap_nm gaps_nm channels",
        "qd": "transition_energy_eV effective_dipole_debye population_decay_energy_neV pure_dephasing_energy_meV dipole_convention background_relative_permittivity",
        "medium": "relative_permittivity",
        "pulse": "carrier_energy_eV intensity_fwhm_fs target_population fluence_min_J_cm2 fluence_max_J_cm2 fluence_points max_fluence_points max_fluence_extensions carrier_scan_eV population_read_fs max_population_decay_fraction refine_threshold threshold_root_xtol_sqrt_J_cm2 threshold_root_rtol max_fluence_midpoint_population_error max_isolated_pulse_area_step_rad",
        "material": "table fit_min_eV fit_max_eV mode_candidates validation_modes seed max_bright_nrms max_bright_pointwise_error",
        "spectrum": "min_eV max_eV feature_center_eV feature_half_window_eV points max_points max_step_over_width max_coarsening_change max_window_change max_observable_nrms max_observable_shift_over_gamma0 max_observable_width_relative_error max_observable_gain_relative_error",
        "numerics": "spatial_order max_spatial_order spatial_rtol modal_audit_points max_modal_nrms max_modal_relative_error method rtol atol points_per_fastest_cycle tail_ratio_tolerance tail_window_fraction max_spectral_leakage dd_tolerance threshold_relative_tolerance weak_field_max_population weak_field_relative_tolerance max_weak_field_refinements",
        "validation": "enabled gap_offset_nm shape_relative_offset exciton_offset_eV population_decay_relative_offset dephasing_relative_offset max_k_c_for_selection max_k_R_for_selection",
        "work_spectrum": "enabled",
        "output": "directory dpi",
    }
    for name, keys in required.items():
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing or invalid [{name}] input table.")
        missing = set(keys.split()) - config[name].keys()
        if missing:
            raise ValueError(f"Missing [{name}] inputs: {', '.join(sorted(missing))}.")
    for name in ("validation", "work_spectrum"):
        if type(config[name]["enabled"]) is not bool:
            raise ValueError(f"{name}.enabled must be true or false.")
    if type(config["pulse"]["refine_threshold"]) is not bool:
        raise ValueError("pulse.refine_threshold must be true or false.")

    def number(section, key, *, positive=False):
        value = config[section][key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            bound = "positive" if positive else "nonnegative"
            raise ValueError(f"{section}.{key} must be a finite {bound} number.")
        return value

    def integer(section, key, minimum):
        value = config[section][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}.")
        return value

    for key in ("threshold_root_xtol_sqrt_J_cm2", "threshold_root_rtol",
                "max_fluence_midpoint_population_error", "max_isolated_pulse_area_step_rad"):
        number("pulse", key, positive=True)

    for section, keys in {
        "geometry": "c_nm a_nm qd_radius_nm reference_surface_gap_nm",
        "qd": "transition_energy_eV effective_dipole_debye background_relative_permittivity",
        "medium": "relative_permittivity",
        "pulse": "carrier_energy_eV intensity_fwhm_fs target_population fluence_min_J_cm2 fluence_max_J_cm2 population_read_fs max_population_decay_fraction",
        "material": "fit_min_eV fit_max_eV max_bright_nrms max_bright_pointwise_error",
        "spectrum": "min_eV max_eV feature_center_eV feature_half_window_eV max_step_over_width max_coarsening_change max_window_change max_observable_nrms max_observable_shift_over_gamma0 max_observable_width_relative_error max_observable_gain_relative_error",
        "numerics": "spatial_rtol max_modal_nrms max_modal_relative_error rtol atol tail_ratio_tolerance tail_window_fraction dd_tolerance threshold_relative_tolerance",
        "validation": "max_k_c_for_selection max_k_R_for_selection",
    }.items():
        for key in keys.split():
            number(section, key, positive=True)
    for section, keys in {
        "qd": "population_decay_energy_neV pure_dephasing_energy_meV",
        "numerics": "max_spectral_leakage weak_field_max_population weak_field_relative_tolerance",
        "validation": "gap_offset_nm shape_relative_offset exciton_offset_eV population_decay_relative_offset dephasing_relative_offset",
    }.items():
        for key in keys.split():
            number(section, key)
    for section, key, minimum in (
        ("geometry", "mnp_count", 1), ("material", "seed", 0),
        ("material", "validation_modes", 2), ("spectrum", "points", 5),
        ("spectrum", "max_points", 5), ("pulse", "fluence_points", 3),
        ("pulse", "max_fluence_points", 3), ("pulse", "max_fluence_extensions", 0),
        ("numerics", "spatial_order", 1), ("numerics", "max_spatial_order", 1),
        ("numerics", "modal_audit_points", 101), ("numerics", "points_per_fastest_cycle", 8),
        ("numerics", "max_weak_field_refinements", 0),
        ("output", "dpi", 1),
    ):
        integer(section, key, minimum)
    g, q, p, m, s, n = (config[k] for k in ("geometry", "qd", "pulse", "material", "spectrum", "numerics"))
    if m.get("refinement") is not None:
        refinement = PassiveFitRefinement(**m["refinement"])
        if not (m["fit_min_eV"] < refinement.focus_center_eV-refinement.focus_half_width_eV
                < refinement.focus_center_eV+refinement.focus_half_width_eV < m["fit_max_eV"]):
            raise ValueError("The material refinement focus must lie inside the fit window.")
    if g["mnp_count"] != 1:
        raise ValueError("This workflow contains exactly one MNP.")
    if not 0 < g["a_nm"] <= g["c_nm"] or g["qd_radius_nm"] <= 0:
        raise ValueError("Require c_nm >= a_nm > 0 and qd_radius_nm > 0.")
    if q["dipole_convention"] != "effective_external":
        raise ValueError("The Shah fitted dipole uses effective_external; do not apply a second screening factor.")
    if m["table"] != "native_gold_johnson_christy" or m["seed"] != 12345:
        raise ValueError("Current article APIs share the native Au table and seed 12345.")
    for name, values in (("gaps_nm", g["gaps_nm"]), ("carrier_scan_eV", p["carrier_scan_eV"])):
        if not isinstance(values, list) or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
            raise ValueError(f"{name} must be a list of numbers.")
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or not arr.size or not np.all(np.isfinite(arr)) or np.any(arr <= 0) or np.any(np.diff(arr) <= 0):
            raise ValueError(f"{name} must be finite, positive and strictly increasing.")
    if (not isinstance(g["channels"], list) or any(not isinstance(x, str) for x in g["channels"])
            or set(g["channels"]) != set(CHANNELS) or len(g["channels"]) != len(CHANNELS)):
        raise ValueError("The article workflow requires all five distinct geometric channels.")
    if q["pure_dephasing_energy_meV"] + .5e-6*q["population_decay_energy_neV"] <= 0:
        raise ValueError("The total coherence rate Gamma2 must be positive for the article linewidth metrics.")
    if not 0 < p["target_population"] < 1 or not 0 < p["max_population_decay_fraction"] < 1:
        raise ValueError("Population target and allowed decay fraction must lie in (0,1).")
    if not 0 < p["fluence_min_J_cm2"] < p["fluence_max_J_cm2"]:
        raise ValueError("Invalid fluence interval in J/cm^2.")
    if not 0 < m["fit_min_eV"] < s["min_eV"] < s["max_eV"] < m["fit_max_eV"]:
        raise ValueError("The spectral grid must lie inside the material fit window.")
    if not s["min_eV"] < s["feature_center_eV"] < s["max_eV"]:
        raise ValueError("Feature center must lie inside the energy grid.")
    if (s["max_points"] < s["points"] or p["max_fluence_points"] < p["fluence_points"]
            or n["max_spatial_order"] < n["spatial_order"]):
        raise ValueError("Invalid grid refinement limits.")
    if (not isinstance(m["mode_candidates"], list) or not m["mode_candidates"]
            or any(type(x) is not int or x < 2 for x in m["mode_candidates"])):
        raise ValueError("Multi-mode candidates must be integers >=2.")
    if any(right <= left for left, right in zip(m["mode_candidates"], m["mode_candidates"][1:])):
        raise ValueError("Multi-mode candidates must be strictly increasing.")
    if m["validation_modes"] <= max(m["mode_candidates"]):
        raise ValueError("validation_modes must exceed the production candidates.")
    energies = DEFAULT_AU_MATERIAL.energy_eV
    if m["fit_min_eV"] < energies[0] or m["fit_max_eV"] > energies[-1]:
        raise ValueError("The material fit window must lie inside the native Au table.")
    sample_count = int(np.count_nonzero((energies >= m["fit_min_eV"]) & (energies <= m["fit_max_eV"])))
    largest_mode_count = m["validation_modes"] if config["validation"]["enabled"] else max(m["mode_candidates"])
    # Match the kernel's identifiability rule: 3N+1 real fit coordinates,
    # four extra real constraints, and at least five independent complex nodes.
    if sample_count < max(5, math.ceil((3*largest_mode_count + 5)/2)):
        raise ValueError(f"Too few independent native Au samples ({sample_count}) for {largest_mode_count} modes.")
    if n["method"] not in ("DOP853", "RK45", "Radau", "BDF", "LSODA"):
        raise ValueError("Unsupported numerics.method for the article APIs.")
    if not 0 < n["tail_window_fraction"] < 1 or n["max_spectral_leakage"] >= 1:
        raise ValueError("Tail-window fraction must lie in (0,1), and spectral leakage in [0,1).")
    if n["weak_field_max_population"] > 1:
        raise ValueError("weak_field_max_population cannot exceed the population bound 1.")
    for value in [q["transition_energy_eV"], p["carrier_energy_eV"], *p["carrier_scan_eV"]]:
        if not m["fit_min_eV"] < value < m["fit_max_eV"]:
            raise ValueError("Exciton and carrier energies must lie inside the material fit window.")
    v = config["validation"]
    if v["enabled"]:
        if v["gap_offset_nm"] >= min(*g["gaps_nm"], g["reference_surface_gap_nm"]):
            raise ValueError("Gap sensitivity would intersect the metal/QD surfaces.")
        shape = v["shape_relative_offset"]
        if shape >= 1 or g["c_nm"]*(1-shape) < g["a_nm"] or g["a_nm"]*(1+shape) > g["c_nm"]:
            raise ValueError("Shape sensitivity must preserve c_nm >= a_nm > 0 for each varied geometry.")
        if not (m["fit_min_eV"] < q["transition_energy_eV"] - v["exciton_offset_eV"]
                <= q["transition_energy_eV"] + v["exciton_offset_eV"] < m["fit_max_eV"]):
            raise ValueError("Exciton sensitivity must stay inside the positive material fit window.")
        for key in ("population_decay_relative_offset", "dephasing_relative_offset"):
            if v[key] > 1:
                raise ValueError(f"{key} cannot exceed 1: the low-rate sensitivity would be negative.")
        gamma1_half = .5e-6*q["population_decay_energy_neV"]
        if (q["pure_dephasing_energy_meV"] + gamma1_half*(1-v["population_decay_relative_offset"]) <= 0
                or q["pure_dephasing_energy_meV"]*(1-v["dephasing_relative_offset"]) + gamma1_half <= 0):
            raise ValueError("Rate sensitivities must retain a positive total coherence rate Gamma2.")
    w = config["work_spectrum"]
    if w["enabled"]:
        missing = set("min_eV max_eV points post_fs max_delta_window_relative_change max_delta_window_absolute_change_cm2".split()) - w.keys()
        if missing:
            raise ValueError(f"Missing [work_spectrum] inputs: {', '.join(sorted(missing))}.")
        for key in ("min_eV", "max_eV", "post_fs", "max_delta_window_relative_change"):
            number("work_spectrum", key, positive=True)
        number("work_spectrum", "max_delta_window_absolute_change_cm2")
        integer("work_spectrum", "points", 5)
        if not m["fit_min_eV"] <= w["min_eV"] < w["max_eV"] <= m["fit_max_eV"]:
            raise ValueError("The work-spectrum interval must lie inside the material fit window.")
        if not w["min_eV"] <= p["carrier_energy_eV"] <= w["max_eV"]:
            raise ValueError("The pulse carrier must lie inside the work-spectrum interval.")
    if not isinstance(config["output"]["directory"], str) or not config["output"]["directory"].strip():
        raise ValueError("output.directory must be a nonempty path string.")


def physical_arguments(config: dict) -> dict:
    """Arguments in nm/eV/meV/D for EXISTING APIs; they convert once to a.u."""
    g, q = config["geometry"], config["qd"]
    gamma1_meV = q["population_decay_energy_neV"] * 1e-6
    return {
        "c-nm": g["c_nm"], "a-nm": g["a_nm"], "qd-radius-nm": g["qd_radius_nm"],
        "eps-m": config["medium"]["relative_permittivity"],
        "eps-qd": q["background_relative_permittivity"],
        "d-debye": q["effective_dipole_debye"], "omega0-ev": q["transition_energy_eV"],
        "gamma-population-mev": gamma1_meV,
        "gamma2-coherence-mev": q["pure_dephasing_energy_meV"] + gamma1_meV / 2,
        "qd-dipole-convention": q["dipole_convention"],
    }


def unit_audit(config: dict) -> dict:
    """Independent SI identities plus native pulse round-trip, saved in every run."""
    args = physical_arguments(config)
    g, q, p = config["geometry"], config["qd"], config["pulse"]
    energy = q["transition_energy_eV"] * elementary_charge
    d_si = q["effective_dipole_debye"] * DEBYE_C_M
    gamma1 = q["population_decay_energy_neV"] * 1e-9 * elementary_charge / hbar
    gamma_phi = q["pure_dephasing_energy_meV"] * 1e-3 * elementary_charge / hbar
    gamma2 = gamma_phi + gamma1 / 2
    tau_s = p["intensity_fwhm_fs"] * 1e-15
    sigma_s = tau_s / (2 * math.sqrt(math.log(2)))
    omegaL = p["carrier_energy_eV"] * elementary_charge / hbar
    refractive_index = math.sqrt(config["medium"]["relative_permittivity"])
    fluence = 1e-7  # audit reference, NOT a prescribed scientific fluence
    # Integral epsilon0*n*c*E(t)^2 dt, E(t)=E0 exp(-t^2/2sigma^2) cos(omega*t).
    integral_factor_s = math.sqrt(math.pi) * sigma_s * (1 + math.exp(-(omegaL*sigma_s)**2))
    e0_si = math.sqrt(2 * fluence * 1e4 / (refractive_index * epsilon_0 * c * integral_factor_s))
    reference = GaussianPulse(E0_au=1.0, omegaL_au=float(eV_to_au(p["carrier_energy_eV"])),
                              tau_au=float(fs_to_au(p["intensity_fwhm_fs"])), tau_kind="fwhm_intensity")
    amplitude_au = math.sqrt(fluence / reference.fluence_j_cm2(eps_m=refractive_index**2))
    from qd_mnp_rational_fit import AU_FIELD_V_M
    if not math.isclose(amplitude_au * AU_FIELD_V_M, e0_si, rel_tol=2e-10):
        raise ValueError("SI/native incident-fluence conversion disagrees.")
    r_tip = g["c_nm"] + g["qd_radius_nm"] + g["reference_surface_gap_nm"]
    k = refractive_index * omegaL / c
    return {
        "source": config["source_url"], "scope": "one-MNP adaptation; intrinsic QD decay, no imported dimer Purcell rate",
        "api_arguments": args,
        "definitions": {"gamma2": "total coherence rate = gamma_phi + gamma1/2", "rates": "energy/hbar, angular frequency; no factor 2*pi",
                        "fluence": "incident J/cm^2; 1 J/cm^2 = 1e4 J/m^2", "pulse_width": "intensity FWHM, not field FWHM or sigma"},
        "SI": {"c_m": g["c_nm"]*1e-9, "a_m": g["a_nm"]*1e-9, "qd_radius_m": g["qd_radius_nm"]*1e-9,
               "qd_dipole_C_m": d_si, "exciton_energy_J": energy, "omega0_rad_s": energy/hbar,
               "gamma1_s_inverse": gamma1, "gamma_phi_s_inverse": gamma_phi, "Gamma2_s_inverse": gamma2,
               "T1_ns": 1e9/gamma1 if gamma1 else None, "T2_fs": 1e15/gamma2,
               "pulse_sigma_field_fs": sigma_s*1e15, "pulse_intensity_spectral_fwhm_eV": 4*math.log(2)*hbar/tau_s/elementary_charge},
        "atomic_units": {"c_bohr": g["c_nm"]*1e-9/AU_LENGTH_M, "d_e_bohr": d_si/AU_DIPOLE_C_M,
                         "omega0": q["transition_energy_eV"]/AU_ENERGY_EV, "gamma1": gamma1*AU_TIME_S, "Gamma2": gamma2*AU_TIME_S},
        "reference_tip_center_distance_nm": r_tip,
        "reference_tip_surface_gap_nm": r_tip-g["c_nm"]-g["qd_radius_nm"],
        "reference_fluence_J_cm2": fluence, "reference_field_V_m": e0_si,
        "reference_peak_intensity_W_cm2": 0.5*refractive_index*epsilon_0*c*e0_si**2/1e4,
        "own_population_decay_fraction_at_read": -math.expm1(-gamma1*p["population_read_fs"]*1e-15),
        "k_c_at_carrier": k*g["c_nm"]*1e-9,
        "k_R_tip_at_carrier": [k*(g["c_nm"]+g["qd_radius_nm"]+gap)*1e-9 for gap in g["gaps_nm"]],
        "applicability": "point QD/local electrostatics only; finite-QD field variation, nonlocality, and ensemble effects are not certified",
    }
