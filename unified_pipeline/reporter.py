"""Crash-safe CSV reporter.

Writes one row to metrics.csv immediately after every question.
If the run crashes at question 17, questions 1-16 are already saved.
"""

import csv
from pathlib import Path


class Reporter:
    def __init__(self, output_path: str):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def write_row(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)
