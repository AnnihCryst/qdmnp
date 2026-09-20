"""Calculate DD/FQS excitation-gain metrics versus QD--MNP surface gap."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from qdmnp.observables.gap_metrics_common import run_spectral_calculation


def main(argv: list[str] | None = None) -> Path:
    return run_spectral_calculation("excitation_gain", Path(__file__), argv)


if __name__ == "__main__":
    main()
