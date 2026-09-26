"""Compare two distinct gains using this run's saved spectral and pulse arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def gain_curves(spectral, thresholds):
    """Require matching coordinates instead of silently mixing different scans."""
    for key in ("model_id", "channel_id", "gap_nm"):
        if not np.array_equal(spectral[key], thresholds[key]):
            raise ValueError(f"Spectral and pulse coordinates differ: {key}")
    fixed = np.asarray(spectral["excitation_gain_at_reference_energy"], float)
    pulse = np.asarray(thresholds["weak_field_population_gain"], float)
    expected = (len(spectral["model_id"]), len(spectral["channel_id"]), len(spectral["gap_nm"]))
    if fixed.shape != expected or pulse.shape != expected:
        raise ValueError("Gain arrays must follow model, channel, gap coordinates.")
    return fixed, pulse


def plot_gain_comparison(spectral, thresholds, output, *, dpi=180):
    import matplotlib.pyplot as plt

    fixed, pulse = gain_curves(spectral, thresholds)
    models = list(spectral["model_id"].astype(str))
    index = models.index("fqs")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    try:
        for ci, label in enumerate(spectral["channel_id"].astype(str)):
            for axis, values in zip(axes, (fixed, pulse)):
                axis.plot(spectral["gap_nm"], values[index, ci], "o-", ms=3, label=label)
        for axis in axes:
            axis.axhline(1, color="black", lw=.7, ls="--")
            axis.set(xlabel="Surface gap (nm)", yscale="log")
            axis.grid(True, alpha=.2)
        axes[0].set_ylabel("QD spectral gain at the reference energy")
        axes[1].set_ylabel("Weak-pulse population gain")
        axes[0].legend(fontsize=8)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi)
    finally:
        plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-artifact", type=Path, required=True)
    parser.add_argument("--threshold-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    with np.load(args.spectral_artifact, allow_pickle=False) as spectral, \
            np.load(args.threshold_artifact, allow_pickle=False) as thresholds:
        return plot_gain_comparison(spectral, thresholds, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
