"""Plot DD--FQS discrepancy and its validity tolerance from a saved NPZ."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from qdmnp.observables.gap_metrics_common import run_gap_plot


def main(argv: list[str] | None = None) -> Path:
    return run_gap_plot("model_discrepancy", argv)


if __name__ == "__main__":
    main()
