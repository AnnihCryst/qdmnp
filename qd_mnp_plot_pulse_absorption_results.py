"""Восстановление графиков pulse-sweep расчета из data.npz и params.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


X_KEYS = {
    "fluence": "fluence_j_cm2",
    "intensity": "peak_intensity_w_cm2",
    "pulse_area": "pulse_area_isolated_qd",
}

X_LABELS = {
    "fluence": r"Fluence, J/cm$^2$",
    "intensity": r"Peak intensity, W/cm$^2$",
    "pulse_area": r"Isolated-QD pulse area",
}


def plot_absorption_sweep(data, x_axis: str, output_path: Path, show: bool) -> None:
    x_key = X_KEYS[x_axis]
    tau_grid = data["tau_fs_grid"]
    x_grid = data[x_key]
    sigma_energy = data["sigma_energy_cm2"]
    sigma_spectral = data["sigma_spectral_cm2"]
    sigma_bare = float(np.ravel(data["sigma_bare_mnp_cm2"])[0])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for tau_index, tau_fs in enumerate(tau_grid[:, 0]):
        order = np.argsort(x_grid[tau_index])
        x = x_grid[tau_index, order]
        label = f"{tau_fs:g} fs"
        axes[0].plot(x, sigma_energy[tau_index, order], marker="o", ms=4, lw=1.8, label=label)
        axes[1].plot(x, sigma_spectral[tau_index, order], marker="s", ms=4, lw=1.8, label=label)

    axes[0].set_ylabel(r"$\sigma_E = W_{abs}/\mathcal{F}$, cm$^2$")
    axes[1].set_ylabel(r"$\sigma_{abs}(\omega_L)$, cm$^2$")
    axes[1].set_xlabel(X_LABELS[x_axis])
    axes[1].axhline(sigma_bare, color="0.25", lw=1.4, ls=":", label="bare MNP")
    if x_axis in {"fluence", "intensity"}:
        axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Pulse FWHM")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _trace_slice(data, trace_index: int) -> slice:
    offsets = data["trace_offsets"]
    lengths = data["trace_lengths"]
    if trace_index < 0 or trace_index >= len(offsets):
        raise IndexError(f"trace-index must be in [0, {len(offsets) - 1}]")
    start = int(offsets[trace_index])
    stop = start + int(lengths[trace_index])
    return slice(start, stop)


def plot_trace(data, trace_index: int, output_path: Path, show: bool) -> None:
    sl = _trace_slice(data, trace_index)
    tau_index = int(data["trace_tau_index"][trace_index])
    e0_index = int(data["trace_e0_index"][trace_index])
    tau_fs = float(data["tau_fs_grid"][tau_index, e0_index])
    e0_v_m = float(data["e0_v_m_grid"][tau_index, e0_index])
    t_fs = data["trace_t_fs"][sl]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(t_fs, data["trace_e_field_au"][sl], color="0.35", label="E(t)")
    axes[0].set_ylabel("Field (a.u.)")
    axes[1].plot(t_fs, data["trace_mu_p_au"][sl], label="mu_p(t)")
    axes[1].set_ylabel("MNP dipole (a.u.)")
    axes[2].plot(t_fs, data["trace_mu_d_au"][sl], label="mu_d(t)")
    axes[2].set_ylabel("QD dipole (a.u.)")
    axes[3].plot(t_fs, data["trace_mu_total_au"][sl], label="mu_total(t)")
    axes[3].set_ylabel("Total dipole (a.u.)")
    axes[3].set_xlabel("Time (fs)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"trace {trace_index}: tau={tau_fs:g} fs, E0={e0_v_m:.3e} V/m")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild pulse absorption plots from saved NPZ/JSON artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--x-axis", choices=["auto", "fluence", "intensity", "pulse_area"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trace-index", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir if args.output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "params.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    x_axis = metadata.get("sweep", {}).get("x_axis", "fluence") if args.x_axis == "auto" else args.x_axis

    with np.load(run_dir / "data.npz") as data:
        plot_absorption_sweep(
            data,
            x_axis=x_axis,
            output_path=output_dir / "absorption_sweep.png",
            show=not args.no_show,
        )
        if args.trace_index is not None:
            plot_trace(
                data,
                trace_index=args.trace_index,
                output_path=output_dir / f"trace_{args.trace_index}.png",
                show=not args.no_show,
            )

    print(f"Rebuilt pulse absorption plots in {output_dir}")


if __name__ == "__main__":
    main()
