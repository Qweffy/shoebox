"""Read benchmark results for the dashboard Benchmark page."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRow:
    system: str
    n: int
    overall_acc: float
    vendor_acc: float
    date_acc: float
    total_acc: float
    mean_latency_s: float
    cost_per_1k: float

    @property
    def is_local(self) -> bool:
        return self.system.startswith(("local", "regex"))


@dataclass(frozen=True)
class BenchmarkData:
    rows: list[BenchmarkRow]
    best_local: BenchmarkRow | None
    best_cloud: BenchmarkRow | None
    headline_relative: int | None
    has_data: bool


def load_benchmark(path: Path) -> BenchmarkData:
    if not path.exists():
        return BenchmarkData([], None, None, None, has_data=False)
    rows: list[BenchmarkRow] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                BenchmarkRow(
                    system=row["system"],
                    n=int(row["n"]),
                    overall_acc=float(row["overall_acc"]),
                    vendor_acc=float(row["vendor_acc"]),
                    date_acc=float(row["date_acc"]),
                    total_acc=float(row["total_acc"]),
                    mean_latency_s=float(row["mean_latency_s"]),
                    cost_per_1k=float(row["cost_per_1k_usd_est"]),
                )
            )
    rows.sort(key=lambda r: r.overall_acc, reverse=True)
    best_local = max(
        (r for r in rows if r.system.startswith("local")),
        key=lambda r: r.overall_acc,
        default=None,
    )
    best_cloud = max(
        (r for r in rows if r.system.endswith("vision")),
        key=lambda r: r.overall_acc,
        default=None,
    )
    relative: int | None = None
    if best_local is not None and best_cloud is not None and best_cloud.overall_acc > 0:
        relative = round(best_local.overall_acc / best_cloud.overall_acc * 100)
    return BenchmarkData(rows, best_local, best_cloud, relative, has_data=bool(rows))
