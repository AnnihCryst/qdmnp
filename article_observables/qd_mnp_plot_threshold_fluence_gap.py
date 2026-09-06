"""Plot threshold-fluence ratio from a saved DD/FQS NPZ artifact."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from article_observables.qd_mnp_gap_metrics_common import run_gap_plot


def main(argv: list[str] | None = None) -> Path:
    return run_gap_plot("threshold_fluence", argv)


if __name__ == "__main__":
    main()
