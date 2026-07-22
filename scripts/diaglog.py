"""Shared append-with-flush logger for the diagnosis tests."""
import os
import csv

CSV = "/src/gs25058/cr_RNE/covor_slam/diagnosis_results.csv"
FIELDS = ["test", "config", "robot", "metric", "value", "time_s", "note"]


def log_rows(rows):
    new = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())
