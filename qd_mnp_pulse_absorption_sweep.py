"""Нелинейный импульсный расчет энергетического и спектрального отклика КТ-МНЧ.
Скрипт решает временную задачу из ``qd_mnp_rational_fit.py`` для сетки
амплитуд и длительностей лазерного импульса. На выходе он сохраняет CSV и
график с работой внешнего поля на диполе, делённой на fluence, и эффективной
спектральной экстинкцией, полученной из Фурье-компоненты временного отклика.
Работать с ним лучше после проверки слабополевого спектра в
``qd_mnp_linear_spectrum.py``. Физические параметры настраиваются аргументами
``--omega0-ev``, ``--gamma2-coherence-mev``, ``--d-debye``, ``--G``,
``--r-nm`` и ``--eps-m``. По умолчанию послепульсный интервал сначала
оценивается по самой медленной несвязанной скорости затухания, а затем
автоматически расширяется по фактическому остаточному дипольному отклику;
``--post-fs`` нужен только для явной проверки сходимости.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qd_mnp_rational_fit import (
    DipoleCrossSections,
    GaussianPulse,
    HybridQDPlasmonModel,
    SCHEMA_VERSION,
    au_to_fs,
    dipole_cross_sections_cm2,
    eV_to_au,
    field_si_to_au,
    fs_to_au,
    params_to_physical_dict,
    timestamped_run_dir,
    write_json,
)
from qd_mnp_params import make_params_with_overrides


TAIL_RATIO_TOL = 1e-3
TAIL_WINDOW_FRACTION = 0.05
MAX_AUTO_TAIL_EXTENSIONS = 3


def spectral_cross_sections_cm2(
    result,
    pulse: GaussianPulse,
    eps_m: float,
    omega_eval_au: float | None = None,
) -> DipoleCrossSections:
    """Fourier-domain dipole cross sections at one angular frequency.

    This follows the pulsed-response convention used by Shah et al.:

        alpha(omega) = int exp(-i omega t) mu(t) dt
                       / [eps_m int exp(-i omega t) E(t) dt]
        sigma_ext(omega) = k / eps0 * Im alpha(omega)

    The solver stores real fields written as cos(omega t). With this real-time
    convention the positive-frequency response compatible with p = alpha E is
    selected by exp(+i omega t); using exp(-i omega t) would return the complex
    conjugate and reverse the sign of Im alpha. The time integrals are evaluated
    in atomic units. The ratio is an atomic polarizability and is converted to
    SI before applying the cross-section formula.
    """
    omega_au = pulse.omegaL_au if omega_eval_au is None else float(omega_eval_au)
    phase = np.exp(1j * omega_au * result.t_au)

    mu_omega_au = np.trapezoid(result.mu_total_au * phase, result.t_au)
    e_omega_au = np.trapezoid(pulse.field(result.t_au) * phase, result.t_au)
    if abs(e_omega_au) < 1e-30:
        nan = np.asarray(np.nan)
        return DipoleCrossSections(nan, nan, nan)

    alpha_au = mu_omega_au / (eps_m * e_omega_au)
    return dipole_cross_sections_cm2(alpha_au, omega_au, eps_m)


def spectral_absorption_cross_section_cm2(
    result,
    pulse: GaussianPulse,
    eps_m: float,
    omega_eval_au: float | None = None,
) -> float:
    """Schema-1 compatibility alias; the old function returned extinction."""
    warnings.warn(
        "spectral_absorption_cross_section_cm2() is a deprecated schema-1 name; "
        "use spectral_cross_sections_cm2(...).extinction_cm2 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    sections = spectral_cross_sections_cm2(result, pulse, eps_m, omega_eval_au)
    return float(sections.extinction_cm2)


def isolated_qd_pulse_area(pulse: GaussianPulse, d_au: float) -> float:
    """Envelope pulse area for an isolated resonant two-level QD."""
    return float(d_au * pulse.E0_au * np.sqrt(2.0 * np.pi) * pulse.sigma_t_au)


def bare_mnp_spectral_cross_sections_cm2(
    model: HybridQDPlasmonModel,
    omega_ev: float,
    eps_m: float,
) -> DipoleCrossSections:
    """Bare-MNP dipole cross sections from the fitted polarizability."""
    omega_au = float(eV_to_au(omega_ev))
    alpha_mnp_au = model.C * model.alpha_from_fit(np.array([omega_ev]))[0] / eps_m
    return dipole_cross_sections_cm2(alpha_mnp_au, omega_au, eps_m)


def bare_mnp_spectral_cross_section_cm2(
    model: HybridQDPlasmonModel,
    omega_ev: float,
    eps_m: float,
) -> float:
    """Schema-1 compatibility alias; the old function returned extinction."""
    warnings.warn(
        "bare_mnp_spectral_cross_section_cm2() is a deprecated schema-1 name; "
        "use bare_mnp_spectral_cross_sections_cm2(...).extinction_cm2 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return float(bare_mnp_spectral_cross_sections_cm2(model, omega_ev, eps_m).extinction_cm2)


def response_tail_ratio(
    mu_total_au: np.ndarray,
    t_au: np.ndarray | None = None,
    mu_p_au: np.ndarray | None = None,
    mu_d_au: np.ndarray | None = None,
    *,
    tail_fraction: float = TAIL_WINDOW_FRACTION,
) -> float:
    """Return a normalized residual-dipole metric near the final time.

    ``response_tail_ratio(mu_total_au)`` preserves the schema-1 behavior: an
    unweighted RMS over the final 5% of samples, normalized by the peak total
    dipole.  When ``t_au`` is supplied, the RMS is integrated over the final
    ``tail_fraction`` of the *time interval*, so adaptive solver sampling cannot
    bias the metric.  In that mode the returned value is the maximum individual
    ratio for ``mu_total_au``, ``mu_p_au`` and ``mu_d_au`` (when supplied), which
    also detects a slowly decaying component hidden by dipole cancellation.
    """
    values = np.asarray(mu_total_au, dtype=float)
    if t_au is None:
        if mu_p_au is not None or mu_d_au is not None:
            raise ValueError('t_au is required when component dipoles are supplied.')
        if values.size == 0:
            return np.nan
        peak = float(np.max(np.abs(values)))
        if peak == 0.0:
            return 0.0
        n_tail = max(8, int(np.ceil(TAIL_WINDOW_FRACTION * values.size)))
        n_tail = min(n_tail, values.size)
        return float(np.sqrt(np.mean(values[-n_tail:] ** 2)) / peak)

    times = np.asarray(t_au, dtype=float)
    if not np.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
        raise ValueError('tail_fraction must be finite and lie in (0, 1].')
    if times.ndim != 1 or times.size < 2:
        raise ValueError('t_au must be a one-dimensional array with at least two samples.')
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError('t_au must contain finite, strictly increasing values.')

    responses = [values]
    responses.extend(
        np.asarray(component, dtype=float)
        for component in (mu_p_au, mu_d_au)
        if component is not None
    )
    if any(response.ndim != 1 or response.size != times.size for response in responses):
        raise ValueError('Each dipole response must be one-dimensional and aligned with t_au.')
    if any(np.any(~np.isfinite(response)) for response in responses):
        raise ValueError('Dipole responses must contain only finite values.')

    window_start = float(times[-1] - tail_fraction * (times[-1] - times[0]))
    start_index = int(np.searchsorted(times, window_start, side='left'))
    ratios: list[float] = []
    for response in responses:
        peak = float(np.max(np.abs(response)))
        if peak == 0.0:
            ratios.append(0.0)
            continue

        if times[start_index] == window_start:
            tail_times = times[start_index:]
            tail_values = response[start_index:]
        else:
            left = start_index - 1
            weight = (window_start - times[left]) / (times[start_index] - times[left])
            value_at_start = response[left] + weight * (response[start_index] - response[left])
            tail_times = np.concatenate(([window_start], times[start_index:]))
            tail_values = np.concatenate(([value_at_start], response[start_index:]))

        duration = float(tail_times[-1] - tail_times[0])
        mean_square = float(np.trapezoid(tail_values**2, tail_times) / duration)
        ratios.append(float(np.sqrt(max(mean_square, 0.0)) / peak))

    return max(ratios)


def compute_sweep(
    *,
    tau_values_fs: list[float],
    e0_values_v_m: np.ndarray,
    omega_l_ev: float,
    n_modes: int,
    fit_window_ev: tuple[float, float],
    weight_center_ev: float | None,
    weight_sigma_ev: float | None,
    method: str,
    rtol: float,
    atol: float,
    pre_sigma: float | None,
    post_fs: float | None,
    c_nm: float | None,
    a_nm: float | None,
    r_nm: float | None,
    qd_radius_nm: float | None = None,
    g_factor: float | None,
    eps_m: float | None,
    d_debye: float | None,
    omega0_ev: float | None,
    gamma_population_mev: float | None,
    gamma2_coherence_mev: float | None = None,
    gamma_dephasing_mev: float | None = None,
) -> tuple[list[dict[str, float]], list[dict[str, object]], object]:
    tau_array = np.asarray(tau_values_fs, dtype=float)
    e0_array = np.asarray(e0_values_v_m, dtype=float)
    if tau_array.ndim != 1 or tau_array.size == 0 or np.any(~np.isfinite(tau_array)) or np.any(tau_array <= 0.0):
        raise ValueError('tau_values_fs must be a non-empty one-dimensional sequence of positive finite values.')
    if e0_array.ndim != 1 or e0_array.size == 0 or np.any(~np.isfinite(e0_array)) or np.any(e0_array <= 0.0):
        raise ValueError('e0_values_v_m must be a non-empty one-dimensional array of positive finite values.')
    if not np.isfinite(omega_l_ev) or omega_l_ev <= 0.0:
        raise ValueError('omega_l_ev must be finite and positive.')
    if pre_sigma is not None and (not np.isfinite(pre_sigma) or pre_sigma <= 0.0):
        raise ValueError('pre_sigma must be finite and positive when explicitly specified.')
    if post_fs is not None and (not np.isfinite(post_fs) or post_fs <= 0.0):
        raise ValueError('post_fs must be finite and positive when explicitly specified.')

    params = make_params_with_overrides(
        c_nm=c_nm,
        a_nm=a_nm,
        r_nm=r_nm,
        qd_radius_nm=qd_radius_nm,
        g_factor=g_factor,
        eps_m=eps_m,
        d_debye=d_debye,
        omega0_ev=omega0_ev,
        gamma_population_mev=gamma_population_mev,
        gamma2_coherence_mev=gamma2_coherence_mev,
        gamma_dephasing_mev=gamma_dephasing_mev,
    )
    model = HybridQDPlasmonModel(
        params,
        orientation="long",
        n_modes=n_modes,
        fit_window_eV=fit_window_ev,
        weight_center_eV=weight_center_ev,
        weight_sigma_eV=weight_sigma_ev,
        alpha_objective_weight=1.0,
        inv_alpha_objective_weight=1.2,
        verbose=True,
    )

    rows: list[dict[str, float]] = []
    traces: list[dict[str, object]] = []
    omega_l_au = float(eV_to_au(omega_l_ev))
    bare_sections = bare_mnp_spectral_cross_sections_cm2(model, omega_l_ev, params.eps_m)
    bare_ext = float(bare_sections.extinction_cm2)
    bare_sca = float(bare_sections.scattering_cm2)
    bare_abs = float(bare_sections.absorption_cm2)
    auto_tail_fs = float(au_to_fs(model.recommended_post_pulse_time_au(decay_times=8.0)))

    for tau_index, tau_fs in enumerate(tau_array):
        for e0_index, e0_v_m in enumerate(e0_array):
            pulse = GaussianPulse(
                E0_au=float(field_si_to_au(e0_v_m)),
                omegaL_au=omega_l_au,
                tau_au=float(fs_to_au(tau_fs)),
                tau_kind="fwhm_intensity",
            )
            effective_pre_sigma = 8.0 if pre_sigma is None else float(pre_sigma)
            if effective_pre_sigma <= 0.0:
                raise ValueError('pre_sigma must be positive.')
            pulse_sigma_fs = float(au_to_fs(pulse.sigma_t_au))
            if post_fs is None:
                effective_post_fs = max(auto_tail_fs, 8.0 * pulse_sigma_fs)
            elif post_fs > 0.0:
                effective_post_fs = float(post_fs)
            else:
                raise ValueError('post_fs must be positive when explicitly specified.')
            automatic_post_fs = post_fs is None
            tail_extension_count = 0
            while True:
                t_span_au = (
                    -effective_pre_sigma * pulse.sigma_t_au,
                    float(fs_to_au(effective_post_fs)),
                )
                result = model.solve(pulse, method=method, rtol=rtol, atol=atol, t_span_au=t_span_au)
                tail_ratio = response_tail_ratio(
                    result.mu_total_au,
                    result.t_au,
                    result.mu_p_au,
                    result.mu_d_au,
                )
                tail_below_tolerance = bool(
                    np.isfinite(tail_ratio) and tail_ratio <= TAIL_RATIO_TOL
                )
                if (
                    not automatic_post_fs
                    or tail_below_tolerance
                    or tail_extension_count >= MAX_AUTO_TAIL_EXTENSIONS
                ):
                    break
                effective_post_fs *= 2.0
                tail_extension_count += 1

            spectral_sections = spectral_cross_sections_cm2(
                result,
                pulse,
                eps_m=params.eps_m,
                omega_eval_au=omega_l_au,
            )
            spectral_ext = float(spectral_sections.extinction_cm2)
            spectral_sca = float(spectral_sections.scattering_cm2)
            spectral_abs = float(spectral_sections.absorption_cm2)
            W, Q, P = result.y[2 * n_modes : 2 * n_modes + 3]
            bloch_radius = np.sqrt(W**2 + Q**2 + P**2)
            traces.append(
                {
                    "tau_index": int(tau_index),
                    "e0_index": int(e0_index),
                    "t_au": result.t_au,
                    "t_fs": au_to_fs(result.t_au),
                    "e_field_au": pulse.field(result.t_au),
                    "mu_p_au": result.mu_p_au,
                    "mu_d_au": result.mu_d_au,
                    "mu_total_au": result.mu_total_au,
                    "mu_dot_total_au": result.mu_dot_total_au,
                    "W": W,
                    "Q": Q,
                    "P": P,
                    "excited_population": 0.5 * (W + 1.0),
                    "bloch_radius": bloch_radius,
                }
            )

            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "tau_fwhm_intensity_fs": float(tau_fs),
                    "e0_v_m": float(e0_v_m),
                    "omega_l_ev": float(omega_l_ev),
                    "linearized_ground_state_stable": bool(model.linear_stability.stable),
                    "linearized_ground_state_spectral_abscissa_au": float(
                        model.linear_stability.spectral_abscissa_au
                    ),
                    "peak_intensity_w_cm2": float(result.peak_intensity_w_cm2),
                    "fluence_j_cm2": float(result.fluence_j_cm2),
                    "pulse_area_isolated_qd": isolated_qd_pulse_area(pulse, params.d_au),
                    "sigma_energy_transfer_cm2": float(result.sigma_energy_transfer_cm2),
                    "sigma_spectral_ext_cm2": spectral_ext,
                    "sigma_spectral_sca_cm2": spectral_sca,
                    "sigma_spectral_abs_cm2": spectral_abs,
                    "sigma_bare_mnp_ext_cm2": bare_ext,
                    "sigma_bare_mnp_sca_cm2": bare_sca,
                    "sigma_bare_mnp_abs_cm2": bare_abs,
                    "delta_sigma_spectral_ext_cm2": spectral_ext - bare_ext,
                    "delta_sigma_spectral_sca_cm2": spectral_sca - bare_sca,
                    "delta_sigma_spectral_abs_cm2": spectral_abs - bare_abs,
                    "work_from_incident_field_j": float(result.work_from_incident_field_j),
                    "post_fs_effective": effective_post_fs,
                    "response_tail_ratio": tail_ratio,
                    "tail_below_tolerance": tail_below_tolerance,
                    "tail_extension_count": tail_extension_count,
                    "max_bloch_radius": result.max_bloch_radius,
                    "min_density_eigenvalue": result.min_density_eigenvalue,
                    "solver_n_steps": int(result.diagnostics.n_steps),
                    "solver_nfev": int(result.diagnostics.nfev),
                    "solver_success": bool(result.diagnostics.solver_success),
                    "t_final_reached": bool(result.diagnostics.t_final_reached),
                    # Schema-1 compatibility aliases. Spectral aliases mean extinction.
                    "sigma_energy_cm2": float(result.sigma_energy_transfer_cm2),
                    "sigma_spectral_cm2": spectral_ext,
                    "sigma_bare_mnp_cm2": bare_ext,
                    "sigma_spectral_minus_bare_cm2": spectral_ext - bare_ext,
                    "absorbed_energy_j": float(result.work_from_incident_field_j),
                }
            )

    unconverged = sum(not bool(row["tail_below_tolerance"]) for row in rows)
    if unconverged:
        if post_fs is None:
            advice = (
                f'automatic doubling reached its limit of {MAX_AUTO_TAIL_EXTENSIONS} extension(s); '
                'inspect the response or repeat with a larger explicit --post-fs.'
            )
        else:
            advice = 'the explicit --post-fs is diagnostic only; increase it and verify convergence.'
        warnings.warn(
            f'{unconverged} pulse response(s) retain a tail ratio above {TAIL_RATIO_TOL:g}; '
            + advice,
            RuntimeWarning,
            stacklevel=2,
        )

    optical_balance_failures = sum(
        np.isfinite(row["sigma_spectral_abs_cm2"])
        and row["sigma_spectral_abs_cm2"]
        < -1e-9
        * max(
            abs(row["sigma_spectral_ext_cm2"]),
            abs(row["sigma_spectral_sca_cm2"]),
            1e-30,
        )
        for row in rows
    )
    if optical_balance_failures:
        warnings.warn(
            f'{optical_balance_failures} effective pulsed response(s) have '
            'sigma_ext-sigma_sca < 0 beyond numerical tolerance. This quantity '
            'cannot be interpreted as material absorption until radiative correction, '
            'nonlinear frequency conversion, Fourier-window convergence and fit error '
            'have been checked.',
            RuntimeWarning,
            stacklevel=2,
        )

    return rows, traces, params


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweeps(rows: list[dict[str, float]], x_axis: str, output_path: Path | None, show: bool) -> None:
    tau_values = sorted({row["tau_fwhm_intensity_fs"] for row in rows})
    x_key = {
        "fluence": "fluence_j_cm2",
        "intensity": "peak_intensity_w_cm2",
        "pulse_area": "pulse_area_isolated_qd",
    }[x_axis]
    x_label = {
        "fluence": r"Fluence, J/cm$^2$",
        "intensity": r"Peak intensity, W/cm$^2$",
        "pulse_area": r"Isolated-QD pulse area",
    }[x_axis]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    sigma_bare = rows[0].get("sigma_bare_mnp_ext_cm2")
    for tau_fs in tau_values:
        group = [row for row in rows if row["tau_fwhm_intensity_fs"] == tau_fs]
        group.sort(key=lambda row: row[x_key])
        x = np.array([row[x_key] for row in group])
        sigma_energy = np.array([row["sigma_energy_transfer_cm2"] for row in group])
        sigma_spectral = np.array([row["sigma_spectral_ext_cm2"] for row in group])

        label = f"{tau_fs:g} fs"
        axes[0].plot(x, sigma_energy, marker="o", ms=4, lw=1.8, label=label)
        axes[1].plot(x, sigma_spectral, marker="s", ms=4, lw=1.8, label=label)

    axes[0].set_ylabel(r"$\sigma_E = W_{ext}/\mathcal{F}$, cm$^2$")
    axes[1].set_ylabel(r"$\sigma_{ext,eff}(\omega_L)$, cm$^2$")
    axes[1].set_xlabel(x_label)
    if sigma_bare is not None:
        axes[1].axhline(
            sigma_bare,
            color="0.25",
            lw=1.4,
            ls=":",
            label="bare MNP",
        )

    if x_axis in {"fluence", "intensity"}:
        axes[1].set_xscale("log")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Pulse FWHM")

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _summary_grid(rows: list[dict[str, float]], key: str, n_tau: int, n_e0: int) -> np.ndarray:
    return np.asarray([row[key] for row in rows]).reshape(n_tau, n_e0)


def _concatenate_traces(traces: list[dict[str, object]], key: str) -> np.ndarray:
    if not traces:
        return np.asarray([], dtype=float)
    return np.concatenate([np.asarray(trace[key]) for trace in traces])


def save_artifact_run(
    *,
    rows: list[dict[str, float]],
    traces: list[dict[str, object]],
    params,
    args: argparse.Namespace,
    e0_values_v_m: np.ndarray,
    run_dir: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    n_tau = len(args.tau_fs)
    n_e0 = len(e0_values_v_m)
    tau_values = np.asarray(args.tau_fs, dtype=float)

    trace_lengths = np.asarray([len(np.asarray(trace["t_au"])) for trace in traces], dtype=np.int64)
    trace_offsets = np.zeros_like(trace_lengths)
    if len(trace_lengths) > 1:
        trace_offsets[1:] = np.cumsum(trace_lengths[:-1])

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "script": "qd_mnp_pulse_absorption_sweep.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "physical": params_to_physical_dict(params, orientation="long"),
        "fit": {
            "n_modes": int(args.n_modes),
            "fit_window_ev": [float(args.fit_min_ev), float(args.fit_max_ev)],
            "weight_center_ev": None if args.weight_center_ev is None else float(args.weight_center_ev),
            "weight_sigma_ev": None if args.weight_sigma_ev is None else float(args.weight_sigma_ev),
            "alpha_objective_weight": 1.0,
            "inv_alpha_objective_weight": 1.2,
        },
        "sweep": {
            "tau_fs": [float(x) for x in tau_values],
            "e0_values_v_m": [float(x) for x in e0_values_v_m],
            "omega_l_ev": float(args.omega_l_ev),
            "pre_sigma": None if args.pre_sigma is None else float(args.pre_sigma),
            "post_fs": None if args.post_fs is None else float(args.post_fs),
            "post_fs_policy": (
                f"automatic 8-amplitude-decay tail with up to {MAX_AUTO_TAIL_EXTENSIONS} doublings"
                if args.post_fs is None
                else "explicit diagnostic-only tail"
            ),
            "max_auto_tail_extensions": MAX_AUTO_TAIL_EXTENSIONS,
            "x_axis": args.x_axis,
        },
        "solver": {
            "method": args.method,
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "linearized_ground_state_stability": {
                "stable": None if not rows else bool(rows[0].get("linearized_ground_state_stable", True)),
                "spectral_abscissa_au": (
                    None
                    if not rows or "linearized_ground_state_spectral_abscissa_au" not in rows[0]
                    else float(rows[0]["linearized_ground_state_spectral_abscissa_au"])
                ),
            },
        },
        "observables": {
            "sigma_energy_transfer_cm2": "external-field work divided by incident fluence",
            "sigma_spectral_ext_cm2": "effective pulsed dipole extinction at omega_l",
            "sigma_spectral_sca_cm2": "effective pulsed dipole scattering at omega_l",
            "sigma_spectral_abs_cm2": (
                "diagnostic sigma_spectral_ext_cm2 - sigma_spectral_sca_cm2; "
                "negative values are not clipped and are not material absorption"
            ),
            "tail_ratio_tolerance": TAIL_RATIO_TOL,
            "tail_metric": (
                f"maximum component-wise time-weighted RMS over the final "
                f"{TAIL_WINDOW_FRACTION:g} fraction of elapsed time, normalized by each component peak"
            ),
            "legacy_aliases": {
                "sigma_energy_cm2": "sigma_energy_transfer_cm2",
                "sigma_spectral_cm2": "sigma_spectral_ext_cm2",
                "sigma_bare_mnp_cm2": "sigma_bare_mnp_ext_cm2",
                "sigma_spectral_minus_bare_cm2": "delta_sigma_spectral_ext_cm2",
                "absorbed_energy_j": "work_from_incident_field_j",
            },
        },
    }

    np.savez_compressed(
        run_dir / "data.npz",
        tau_fs_grid=np.repeat(tau_values[:, None], n_e0, axis=1),
        e0_v_m_grid=np.repeat(e0_values_v_m[None, :], n_tau, axis=0),
        peak_intensity_w_cm2=_summary_grid(rows, "peak_intensity_w_cm2", n_tau, n_e0),
        linearized_ground_state_stable=np.asarray(
            [row.get("linearized_ground_state_stable", True) for row in rows], dtype=bool
        ).reshape(n_tau, n_e0),
        linearized_ground_state_spectral_abscissa_au=np.asarray(
            [row.get("linearized_ground_state_spectral_abscissa_au", np.nan) for row in rows], dtype=float
        ).reshape(n_tau, n_e0),
        fluence_j_cm2=_summary_grid(rows, "fluence_j_cm2", n_tau, n_e0),
        pulse_area_isolated_qd=_summary_grid(rows, "pulse_area_isolated_qd", n_tau, n_e0),
        sigma_energy_transfer_cm2=_summary_grid(rows, "sigma_energy_transfer_cm2", n_tau, n_e0),
        sigma_spectral_ext_cm2=_summary_grid(rows, "sigma_spectral_ext_cm2", n_tau, n_e0),
        sigma_spectral_sca_cm2=_summary_grid(rows, "sigma_spectral_sca_cm2", n_tau, n_e0),
        sigma_spectral_abs_cm2=_summary_grid(rows, "sigma_spectral_abs_cm2", n_tau, n_e0),
        sigma_bare_mnp_ext_cm2=_summary_grid(rows, "sigma_bare_mnp_ext_cm2", n_tau, n_e0),
        sigma_bare_mnp_sca_cm2=_summary_grid(rows, "sigma_bare_mnp_sca_cm2", n_tau, n_e0),
        sigma_bare_mnp_abs_cm2=_summary_grid(rows, "sigma_bare_mnp_abs_cm2", n_tau, n_e0),
        delta_sigma_spectral_ext_cm2=_summary_grid(rows, "delta_sigma_spectral_ext_cm2", n_tau, n_e0),
        delta_sigma_spectral_sca_cm2=_summary_grid(rows, "delta_sigma_spectral_sca_cm2", n_tau, n_e0),
        delta_sigma_spectral_abs_cm2=_summary_grid(rows, "delta_sigma_spectral_abs_cm2", n_tau, n_e0),
        work_from_incident_field_j=_summary_grid(rows, "work_from_incident_field_j", n_tau, n_e0),
        post_fs_effective=_summary_grid(rows, "post_fs_effective", n_tau, n_e0),
        response_tail_ratio=_summary_grid(rows, "response_tail_ratio", n_tau, n_e0),
        tail_below_tolerance=_summary_grid(rows, "tail_below_tolerance", n_tau, n_e0).astype(bool),
        # Rows assembled by schema-1 callers predate automatic tail extension.
        tail_extension_count=np.asarray(
            [row.get("tail_extension_count", 0) for row in rows], dtype=np.int64
        ).reshape(n_tau, n_e0),
        max_bloch_radius=_summary_grid(rows, "max_bloch_radius", n_tau, n_e0),
        min_density_eigenvalue=_summary_grid(rows, "min_density_eigenvalue", n_tau, n_e0),
        solver_nfev=_summary_grid(rows, "solver_nfev", n_tau, n_e0).astype(np.int64),
        t_final_reached=_summary_grid(rows, "t_final_reached", n_tau, n_e0).astype(bool),
        # Schema-1 compatibility aliases.
        sigma_energy_cm2=_summary_grid(rows, "sigma_energy_cm2", n_tau, n_e0),
        sigma_spectral_cm2=_summary_grid(rows, "sigma_spectral_cm2", n_tau, n_e0),
        sigma_bare_mnp_cm2=_summary_grid(rows, "sigma_bare_mnp_cm2", n_tau, n_e0),
        sigma_spectral_minus_bare_cm2=_summary_grid(rows, "sigma_spectral_minus_bare_cm2", n_tau, n_e0),
        absorbed_energy_j=_summary_grid(rows, "absorbed_energy_j", n_tau, n_e0),
        solver_n_steps=_summary_grid(rows, "solver_n_steps", n_tau, n_e0).astype(np.int64),
        solver_success=_summary_grid(rows, "solver_success", n_tau, n_e0).astype(bool),
        trace_offsets=trace_offsets,
        trace_lengths=trace_lengths,
        trace_tau_index=np.asarray([trace["tau_index"] for trace in traces], dtype=np.int64),
        trace_e0_index=np.asarray([trace["e0_index"] for trace in traces], dtype=np.int64),
        trace_t_au=_concatenate_traces(traces, "t_au"),
        trace_t_fs=_concatenate_traces(traces, "t_fs"),
        trace_e_field_au=_concatenate_traces(traces, "e_field_au"),
        trace_mu_p_au=_concatenate_traces(traces, "mu_p_au"),
        trace_mu_d_au=_concatenate_traces(traces, "mu_d_au"),
        trace_mu_total_au=_concatenate_traces(traces, "mu_total_au"),
        trace_mu_dot_total_au=_concatenate_traces(traces, "mu_dot_total_au"),
        trace_W=_concatenate_traces(traces, "W"),
        trace_Q=_concatenate_traces(traces, "Q"),
        trace_P=_concatenate_traces(traces, "P"),
        trace_excited_population=_concatenate_traces(traces, "excited_population"),
        trace_bloch_radius=_concatenate_traces(traces, "bloch_radius"),
    )
    write_json(run_dir / "params.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare external-field energy transfer and effective Fourier dipole cross sections."
    )
    parser.add_argument("--tau-fs", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    parser.add_argument("--e0-min", type=float, default=1e5, help="Minimum field amplitude, V/m.")
    parser.add_argument("--e0-max", type=float, default=1e9, help="Maximum field amplitude, V/m.")
    parser.add_argument("--points", type=int, default=18)
    parser.add_argument("--omega-l-ev", type=float, default=2.042)
    parser.add_argument("--n-modes", type=int, default=4)
    parser.add_argument("--fit-min-ev", type=float, default=0.8)
    parser.add_argument("--fit-max-ev", type=float, default=3.0)
    parser.add_argument("--weight-center-ev", type=float, default=2.35)
    parser.add_argument("--weight-sigma-ev", type=float, default=0.30)
    parser.add_argument("--c-nm", type=float, default=None)
    parser.add_argument("--a-nm", type=float, default=None)
    parser.add_argument("--r-nm", type=float, default=None)
    parser.add_argument("--qd-radius-nm", type=float, default=None)
    parser.add_argument("--G", dest="g_factor", type=float, default=None)
    parser.add_argument("--eps-m", type=float, default=None)
    parser.add_argument("--d-debye", type=float, default=None)
    parser.add_argument("--omega0-ev", type=float, default=None)
    parser.add_argument("--gamma-population-mev", type=float, default=None)
    parser.add_argument(
        "--gamma2-coherence-mev",
        dest="gamma2_coherence_mev",
        metavar="GAMMA2_MEV",
        type=float,
        default=None,
        help="Total coherence HWHM hbar*Gamma2 in meV; requires Gamma2 >= gamma1/2.",
    )
    parser.add_argument(
        "--gamma-dephasing-mev",
        dest="gamma_dephasing_mev",
        metavar="GAMMA2_MEV",
        type=float,
        default=None,
        help="Deprecated alias for --gamma2-coherence-mev.",
    )
    parser.add_argument("--method", choices=["Radau", "BDF", "LSODA", "RK45", "DOP853"], default="Radau")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--pre-sigma",
        type=float,
        default=8.0,
        help="Start time in pulse sigmas.",
    )
    parser.add_argument(
        "--post-fs",
        type=float,
        default=None,
        help="Absolute final time after the pulse center in fs. By default an initial "
        "8-e-fold estimate is doubled until the component-wise dipole tail converges.",
    )
    parser.add_argument("--x-axis", choices=["fluence", "intensity", "pulse_area"], default="fluence")
    parser.add_argument("--csv", type=Path, default=Path("results/absorption_sections_sweep.csv"))
    parser.add_argument("--figure", type=Path, default=Path("results/absorption_sections_sweep.png"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-save-figure", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if (
        args.gamma2_coherence_mev is not None
        and args.gamma_dephasing_mev is not None
        and args.gamma2_coherence_mev != args.gamma_dephasing_mev
    ):
        parser.error("--gamma2-coherence-mev and --gamma-dephasing-mev must agree when both are supplied.")
    if args.gamma2_coherence_mev is None:
        args.gamma2_coherence_mev = args.gamma_dephasing_mev
    return args


def main() -> None:
    args = parse_args()
    if args.points < 1:
        raise ValueError('--points must be at least 1.')
    if (
        not np.isfinite(args.e0_min)
        or not np.isfinite(args.e0_max)
        or args.e0_min <= 0.0
        or args.e0_max < args.e0_min
    ):
        raise ValueError('--e0-min and --e0-max must be positive, finite and ordered.')
    e0_values = np.logspace(np.log10(args.e0_min), np.log10(args.e0_max), args.points)
    rows, traces, params = compute_sweep(
        tau_values_fs=args.tau_fs,
        e0_values_v_m=e0_values,
        omega_l_ev=args.omega_l_ev,
        n_modes=args.n_modes,
        fit_window_ev=(args.fit_min_ev, args.fit_max_ev),
        weight_center_ev=args.weight_center_ev,
        weight_sigma_ev=args.weight_sigma_ev,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        pre_sigma=args.pre_sigma,
        post_fs=args.post_fs,
        c_nm=args.c_nm,
        a_nm=args.a_nm,
        r_nm=args.r_nm,
        qd_radius_nm=args.qd_radius_nm,
        g_factor=args.g_factor,
        eps_m=args.eps_m,
        d_debye=args.d_debye,
        omega0_ev=args.omega0_ev,
        gamma_population_mev=args.gamma_population_mev,
        gamma2_coherence_mev=args.gamma2_coherence_mev,
        gamma_dephasing_mev=None,
    )
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")

    run_dir = args.run_dir if args.run_dir is not None else timestamped_run_dir("results/pulse_absorption_sweeps")
    run_dir = Path(run_dir)
    save_artifact_run(rows=rows, traces=traces, params=params, args=args, e0_values_v_m=e0_values, run_dir=run_dir)
    plot_sweeps(rows, args.x_axis, run_dir / "absorption_sweep.png", show=False)
    print(f"Wrote pulse-sweep artifact run to {run_dir}")

    figure_path = None if args.no_save_figure else args.figure
    plot_sweeps(rows, args.x_axis, figure_path, show=not args.no_show)


if __name__ == "__main__":
    main()
