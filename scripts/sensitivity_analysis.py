"""Print the sensitivity analysis (RQ4) and write it to data/sensitivity_report.csv.

    uv run python scripts/sensitivity_analysis.py --perturbation 0.2
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sensitivity import SensitivityReport, run_sensitivity_analysis  # noqa: E402

CSV_OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "sensitivity_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity analysis of the scheduler constants.")
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.2,
        help="Relative change applied to each constant (0.2 = +/-20 %%).",
    )
    parser.add_argument(
        "--influence-threshold",
        type=float,
        default=1.0,
        help="Interval shift in days above which a constant counts as influential.",
    )
    return parser.parse_args()


def print_report(report: SensitivityReport) -> None:
    print("=" * 78)
    print(f"Sensitivity analysis, +/-{report.perturbation * 100:.0f} %")
    print(f"Mean Spearman rho: {report.order_preserved_ratio:.3f}")
    print("-" * 78)
    print(
        f"{'Constant':<26}{'nominal':>12}{'max|delta|':>12}"
        f"{'elasticity':>12}{'reclass.':>10}  status"
    )
    print("-" * 78)
    for factor in report.factors:
        status = "INFLUENTIAL" if factor.is_influential else "low"
        print(
            f"{factor.name:<26}{factor.nominal:>12.3f}{factor.max_abs_delta:>12.3f}"
            f"{factor.mean_elasticity:>12.3f}{factor.reclassification_rate:>10.3f}  {status}"
        )
    print("-" * 78)
    print(report.summary)
    print("=" * 78)


def write_csv(report: SensitivityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(factor) for factor in report.factors]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Windows consoles default to cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    report = run_sensitivity_analysis(
        perturbation=args.perturbation,
        influence_threshold=args.influence_threshold,
    )
    print_report(report)
    write_csv(report, CSV_OUTPUT_PATH)
    print(f"\nCSV written to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
