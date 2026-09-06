"""Calculate nonlinear DD/FQS threshold fluence versus QD--MNP surface gap."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from article_observables.qd_mnp_gap_metrics_common import run_threshold_calculation


def main(argv: list[str] | None = None) -> Path:
    return run_threshold_calculation(Path(__file__), argv)


if __name__ == "__main__":
    main()
