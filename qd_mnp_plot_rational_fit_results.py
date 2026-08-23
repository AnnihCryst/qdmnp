"""Восстановление графиков rational-fit расчета из data.npz и params.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qd_mnp_rational_fit import (
    plot_fit_diagnostics_from_data,
    plot_time_dynamics_from_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild rational-fit plots from saved NPZ/JSON artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir if args.output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "params.json").open("r", encoding="utf-8") as f:
        json.load(f)

    with np.load(run_dir / "data.npz") as data:
        plot_fit_diagnostics_from_data(
            data,
            output_dir / "fit_diagnostics.png",
            show=not args.no_show,
        )
        plot_time_dynamics_from_data(
            data,
            output_dir / "time_dynamics.png",
            show=not args.no_show,
        )

    print(f"Rebuilt rational-fit plots in {output_dir}")


if __name__ == "__main__":
    main()
