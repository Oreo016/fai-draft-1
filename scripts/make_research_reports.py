"""Regenerate CSV, LaTeX, PDF, and PNG reports from stored study results."""

import argparse
import json
from pathlib import Path

from voc_sar.research_experiments.common import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run_directory / "summary.json").read_text(encoding="utf-8"))
    report_directory = Path(summary["report_directory"])
    write_reports(args.run_directory, report_directory)
    print(f"Reports: {report_directory.resolve()}")


if __name__ == "__main__":
    main()
