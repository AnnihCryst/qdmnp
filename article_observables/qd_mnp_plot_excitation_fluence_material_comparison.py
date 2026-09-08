"""Plot a saved one-vs-multi material excitation-fluence comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from article_observables.qd_mnp_material_modes_artifact import (
    load_npz_artifact,
    save_figure,
)


SCHEMA_NAME = "qd_mnp.material_excitation_fluence_comparison"
REQUIRED_ARRAYS = {
    "branch_id",
    "branch_material_mode_count",
    "channel_id",
    "fluence_j_cm2",
    "p_exc_read",
    "p_exc_max",
    "read_time_fs",
    "target_population",
    "threshold_fluence_j_cm2",
    "hybrid_threshold_ratio_one_to_multi",
}


def _first_threshold(
    fluence: np.ndarray,
    population: np.ndarray,
    target: float,
) -> tuple[float, str]:
    if population[0] >= target:
        return float(fluence[0]), "left_censored"
    for index in range(fluence.size - 1):
        y0 = float(population[index])
        y1 = float(population[index + 1])
        if y0 < target <= y1 and y1 > y0:
            root = np.sqrt(fluence[index]) + (
                (target - y0)
                / (y1 - y0)
                * (np.sqrt(fluence[index + 1]) - np.sqrt(fluence[index]))
            )
            return float(root**2), "interpolated"
    return float("nan"), "not_reached"


def _branch_payload(payload: dict[str, np.ndarray], branch: str) -> dict[str, np.ndarray]:
    prefix = f"{branch}__"
    return {
        key[len(prefix) :]: np.asarray(value)
        for key, value in payload.items()
        if key.startswith(prefix)
    }


def _certificate_failures(
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
    branch: str,
) -> list[str]:
    data = _branch_payload(payload, branch)
    branch_metadata = metadata.get("branch_metadata", {}).get(branch, {})
    settings = branch_metadata.get("resolved_settings", {})
    failures: list[str] = []

    required_true = (
        "bare_diagnostic__solver_success",
        "bare_diagnostic__t_final_reached",
        "bare_diagnostic__state_is_finite",
        "bare_diagnostic__response_tail_converged",
        "full_diagnostic__solver_success",
        "full_diagnostic__t_final_reached",
        "full_diagnostic__state_is_finite",
        "full_diagnostic__response_tail_converged",
        "full_diagnostic__work_nonnegative_within_tolerance",
    )
    for key in required_true:
        values = np.asarray(data.get(key, np.asarray(False)))
        if values.dtype.kind != "b" or not np.all(values):
            failures.append(f"{branch}:{key}")

    try:
        positivity_tolerance = float(settings["positivity_tolerance"])
        decay_limit = float(settings["max_population_decay_fraction_at_read"])
        leakage_limit = float(settings["max_spectral_leakage"])
    except (KeyError, TypeError, ValueError):
        failures.append(f"{branch}:missing numerical limits")
        positivity_tolerance = decay_limit = leakage_limit = float("nan")

    for key in (
        "bare_diagnostic__min_density_eigenvalue",
        "full_diagnostic__min_density_eigenvalue",
    ):
        values = np.asarray(data.get(key, np.asarray(np.nan)), dtype=float)
        if not (
            np.isfinite(positivity_tolerance)
            and np.all(np.isfinite(values))
            and np.all(values >= -positivity_tolerance)
        ):
            failures.append(f"{branch}:{key}")

    decay = np.asarray(data.get("population_decay_fraction_at_read", np.asarray(np.nan)))
    if not (
        np.isfinite(decay_limit)
        and np.all(np.isfinite(decay))
        and np.all(decay <= decay_limit)
    ):
        failures.append(f"{branch}:population decay at read time")

    leakage_keys = (
        "bare_diagnostic__pulse_spectral_leakage",
        "full_diagnostic__pulse_spectral_leakage",
        "full_diagnostic__qd_source_spectral_leakage",
        "full_diagnostic__mnp_drive_spectral_leakage",
        "full_diagnostic__mnp_dipole_spectral_leakage",
        "full_diagnostic__mnp_field_spectral_leakage",
    )
    for key in leakage_keys:
        values = np.asarray(data.get(key, np.asarray(np.nan)), dtype=float)
        if not (
            np.isfinite(leakage_limit)
            and np.all(np.isfinite(values))
            and np.all(values <= leakage_limit)
        ):
            failures.append(f"{branch}:{key}")

    grid = np.asarray(data.get("fluence_grid_converged_by_channel", np.asarray(False)))
    if grid.dtype.kind != "b" or not np.all(grid):
        failures.append(f"{branch}:fluence-grid resolution")

    channels = branch_metadata.get("channels", [])
    for channel in channels[1:]:
        channel_id = channel.get("channel_id", "unknown")
        material = channel.get("material_fit", {})
        modal = channel.get("modal_transform_diagnostics", {})
        spatial = channel.get("spatial_convergence_diagnostics", {})
        stability = channel.get("coupled_stability_diagnostics", {})
        reduction = channel.get("dark_reduction")
        if material.get("passive_for_all_positive_frequencies") is not True:
            failures.append(f"{branch}:{channel_id}:material passivity")
        if modal.get("passive_on_audit_grid") is not True:
            failures.append(f"{branch}:{channel_id}:modal passivity")
        if spatial.get("accepted") is not True:
            failures.append(f"{branch}:{channel_id}:spatial convergence")
        if stability.get("stable") is not True:
            failures.append(f"{branch}:{channel_id}:coupled stability")
        if reduction is not None and not (
            reduction.get("diagnostics", {}).get("accepted") is True
            and reduction.get("current_transfer_reaudit", {}).get("accepted") is True
        ):
            failures.append(f"{branch}:{channel_id}:dark reduction")
        if branch == "multi":
            try:
                fit_ok = (
                    max(
                        float(material["normalized_rms_alpha"]),
                        float(material["normalized_rms_inv_alpha"]),
                    )
                    <= float(material["max_fit_normalized_rms_gate"])
                    and float(material["max_normalized_alpha_error"])
                    <= float(material["max_fit_pointwise_relative_error_gate"])
                )
            except (KeyError, TypeError, ValueError):
                fit_ok = False
            if not fit_ok:
                failures.append(f"{branch}:{channel_id}:production material fit")
            if modal.get("accepted") is not True:
                failures.append(f"{branch}:{channel_id}:production modal transform")
    return failures


def load_comparison_artifact(
    path: str | Path,
    *,
    allow_unconverged: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    payload, metadata = load_npz_artifact(
        path,
        schema_name=SCHEMA_NAME,
        required_arrays=REQUIRED_ARRAYS,
    )
    branches = np.asarray(payload["branch_id"]).astype(str)
    channels = np.asarray(payload["channel_id"]).astype(str)
    fluence = np.asarray(payload["fluence_j_cm2"], dtype=float)
    p_read = np.asarray(payload["p_exc_read"], dtype=float)
    p_max = np.asarray(payload["p_exc_max"], dtype=float)
    read_time = np.asarray(payload["read_time_fs"], dtype=float)
    if not np.array_equal(branches, np.asarray(["one", "multi"])):
        raise ValueError("branch_id must be exactly ['one', 'multi'].")
    if channels.ndim != 1 or channels.size < 2 or channels[0] != "bare_qd":
        raise ValueError("channel_id must be one-dimensional and start with bare_qd.")
    if np.unique(channels).size != channels.size:
        raise ValueError("channel_id values must be unique.")
    if (
        fluence.ndim != 1
        or fluence.size < 3
        or np.any(~np.isfinite(fluence))
        or np.any(fluence <= 0.0)
        or np.any(np.diff(fluence) <= 0.0)
    ):
        raise ValueError("fluence_j_cm2 must be finite, positive and increasing.")
    expected = (2, channels.size, fluence.size)
    if p_read.shape != expected or p_max.shape != expected:
        raise ValueError(f"Population arrays must have shape {expected}.")
    if not (
        np.all(np.isfinite(p_read))
        and np.all(np.isfinite(p_max))
        and np.all(p_read >= -1.0e-6)
        and np.all(p_read <= 1.0 + 1.0e-6)
        and np.all(p_max + 1.0e-6 >= p_read)
    ):
        raise ValueError("Saved populations violate physical bounds.")
    if (
        read_time.shape != fluence.shape
        or np.any(~np.isfinite(read_time))
        or np.any(read_time <= 0.0)
        or not np.allclose(
            read_time,
            read_time[0],
            rtol=0.0,
            atol=1.0e-10 * max(abs(float(read_time[0])), 1.0),
        )
    ):
        raise ValueError("All curves must share one physical read time.")
    if not np.allclose(p_read[0, 0], p_read[1, 0], rtol=2.0e-9, atol=2.0e-11):
        raise ValueError("The two branches contain different isolated-QD controls.")
    if not np.allclose(p_max[0, 0], p_max[1, 0], rtol=2.0e-9, atol=2.0e-11):
        raise ValueError(
            "The two branches contain different isolated-QD maximum populations."
        )

    failures = [
        *_certificate_failures(payload, metadata, "one"),
        *_certificate_failures(payload, metadata, "multi"),
    ]
    failures = list(dict.fromkeys(failures))
    if failures and not allow_unconverged:
        raise ValueError(
            "Refusing to plot an uncertified material comparison; failed: "
            + ", ".join(failures)
            + ". Use --allow-unconverged only for a watermarked diagnostic plot."
        )
    return payload, metadata, failures


def plot_comparison(
    payload: dict[str, np.ndarray],
    metadata: dict[str, Any],
    output: str | Path,
    *,
    channel_filter: list[str] | None = None,
    target_population: float | None = None,
    include_maximum: bool = False,
    linear_x: bool = False,
    diagnostic_failures: list[str] | None = None,
    dpi: int = 250,
    show: bool = False,
) -> Path:
    branches = np.asarray(payload["branch_id"]).astype(str)
    modes = np.asarray(payload["branch_material_mode_count"], dtype=int)
    channel_ids = np.asarray(payload["channel_id"]).astype(str)
    fluence = np.asarray(payload["fluence_j_cm2"], dtype=float)
    population = np.asarray(payload["p_exc_read"], dtype=float)
    population_max = np.asarray(payload["p_exc_max"], dtype=float)
    if channel_filter is None:
        requested = [name for name in ("axis_long", "axis_trans") if name in channel_ids]
        if not requested:
            requested = list(channel_ids[1:3])
    else:
        requested = list(dict.fromkeys(channel_filter))
    unknown = [name for name in requested if name not in channel_ids or name == "bare_qd"]
    if unknown:
        raise ValueError("Unknown/non-hybrid --channel value(s): " + ", ".join(unknown))
    if not requested:
        raise ValueError("At least one hybrid channel is required.")
    indices = [int(np.flatnonzero(channel_ids == name)[0]) for name in requested]

    target = (
        float(np.asarray(payload["target_population"]))
        if target_population is None
        else float(target_population)
    )
    if not 0.0 < target < 1.0:
        raise ValueError("target_population must lie strictly between 0 and 1.")

    figure, axes = plt.subplots(
        2,
        len(indices),
        figsize=(6.1 * len(indices), 7.2),
        sharex="col",
        squeeze=False,
        gridspec_kw={"height_ratios": (2.1, 1.0)},
    )
    colors = {"one": "#D55E00", "multi": "#0072B2"}
    styles = {"one": "--", "multi": "-"}
    labels_by_id = {
        item.get("channel_id"): item.get("label", item.get("channel_id"))
        for item in metadata.get("branch_metadata", {})
        .get("multi", {})
        .get("channels", [])
    }
    for column, channel_index in enumerate(indices):
        upper = axes[0, column]
        lower = axes[1, column]
        upper.plot(
            fluence,
            population[0, 0],
            color="0.3",
            lw=1.4,
            ls=":",
            label="isolated QD",
        )
        for branch_index, branch in enumerate(branches):
            label = f"{branch}: N={modes[branch_index]}"
            upper.plot(
                fluence,
                population[branch_index, channel_index],
                color=colors[branch],
                ls=styles[branch],
                lw=2.0,
                label=label,
            )
            if include_maximum:
                upper.plot(
                    fluence,
                    population_max[branch_index, channel_index],
                    color=colors[branch],
                    ls=styles[branch],
                    lw=0.9,
                    alpha=0.4,
                )
            crossing, status = _first_threshold(
                fluence,
                population[branch_index, channel_index],
                target,
            )
            if np.isfinite(crossing):
                upper.plot(
                    crossing,
                    target,
                    marker="o" if status == "interpolated" else "<",
                    color=colors[branch],
                    ms=6,
                )
        difference = population[0, channel_index] - population[1, channel_index]
        lower.plot(fluence, difference, color="#6A3D9A", lw=1.8)
        lower.axhline(0.0, color="0.35", lw=0.9, ls=":")
        upper.axhline(target, color="0.45", lw=1.0, ls=":")
        one_threshold, _ = _first_threshold(fluence, population[0, channel_index], target)
        multi_threshold, _ = _first_threshold(fluence, population[1, channel_index], target)
        ratio_text = "threshold unavailable"
        if np.isfinite(one_threshold) and np.isfinite(multi_threshold) and multi_threshold > 0.0:
            ratio_text = rf"$\mathcal{{F}}_\eta^{{(1)}}/\mathcal{{F}}_\eta^{{(N)}}={one_threshold / multi_threshold:.3g}$"
        upper.text(0.03, 0.04, ratio_text, transform=upper.transAxes, fontsize=9)
        upper.set_title(labels_by_id.get(channel_ids[channel_index], channel_ids[channel_index]))
        upper.set_ylim(-0.025, 1.025)
        upper.grid(True, which="both", alpha=0.25)
        lower.grid(True, which="both", alpha=0.25)
        lower.set_xlabel(r"Fluence $\mathcal{F}$, J cm$^{-2}$")
        if not linear_x:
            upper.set_xscale("log")
            lower.set_xscale("log")
        if column == 0:
            upper.set_ylabel(r"Post-pulse $P_{\rm exc}$")
            lower.set_ylabel(r"$P_{\rm exc}^{(1)}-P_{\rm exc}^{(N)}$")
        upper.legend(fontsize=8)

    figure.suptitle(
        "Material-dispersion sensitivity of nonlinear exciton excitation\n"
        + rf"common $t_{{\rm read}}={float(np.asarray(payload['read_time_fs'])[0]):.3g}$ fs",
        fontsize=12,
    )
    failures = diagnostic_failures or []
    if failures:
        figure.text(
            0.5,
            0.5,
            "DIAGNOSTIC - UNCONVERGED",
            ha="center",
            va="center",
            rotation=25,
            fontsize=28,
            color="crimson",
            alpha=0.18,
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save_figure(figure, output, dpi=dpi, show=show)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/article/excitation_fluence_material_comparison.png"),
    )
    parser.add_argument("--channels", nargs="+")
    parser.add_argument("--target-population", type=float)
    parser.add_argument("--include-maximum", action="store_true")
    parser.add_argument("--linear-x", action="store_true")
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    payload, metadata, failures = load_comparison_artifact(
        args.artifact,
        allow_unconverged=args.allow_unconverged,
    )
    output = plot_comparison(
        payload,
        metadata,
        args.output,
        channel_filter=args.channels,
        target_population=args.target_population,
        include_maximum=args.include_maximum,
        linear_x=args.linear_x,
        diagnostic_failures=failures,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Saved material-mode excitation-fluence figure: {output}")
    return output


if __name__ == "__main__":
    main()
