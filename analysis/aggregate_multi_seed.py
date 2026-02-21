#!/usr/bin/env python3
"""
Aggregate per-model metrics across multiple seed runs.

Usage examples:
  python analysis/aggregate_multi_seed.py --glob "CONFRONTO_METRICHE_seed*.csv"
  python analysis/aggregate_multi_seed.py --inputs seed42:CONFRONTO_METRICHE.csv \
        seed43:/path/to/CONFRONTO_METRICHE_seed43.csv

The script expects CSV files with at least the columns:
    Modello, CLIPScore, BLEU-1, METEOR, ROUGE-L, Punteggio_Composito

For each model it computes mean and standard deviation (sample stdev, 0 if <2 samples)
for the metrics above and writes a summary CSV plus an optional LaTeX table.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, List, Sequence, Tuple

METRIC_COLUMNS = [
    "CLIPScore",
    "BLEU-1",
    "METEOR",
    "ROUGE-L",
    "Punteggio_Composito",
]

SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        help="Glob pattern to discover CSV files (e.g., 'CONFRONTO_METRICHE_seed*.csv')",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        help=(
            "Explicit file list in the form seed:PATH or PATH (seed inferred from filename). "
            "Example: seed42:./CONFRONTO_METRICHE.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="multi_seed_summary.csv",
        help="Path to write the aggregated CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--latex",
        help="Optional path to write a LaTeX tabular summarising the stats.",
    )
    return parser.parse_args()


def discover_files(pattern: str | None, inputs: Sequence[str] | None) -> List[Tuple[int, Path]]:
    pairs: List[Tuple[int, Path]] = []

    if pattern:
        for path in Path().glob(pattern):
            seed = infer_seed(path)
            pairs.append((seed, path))

    if inputs:
        for item in inputs:
            if ":" in item:
                seed_str, raw_path = item.split(":", 1)
                try:
                    seed = int(seed_str.replace("seed", ""))
                except ValueError as exc:
                    raise ValueError(f"Cannot parse seed from '{seed_str}'") from exc
                path = Path(raw_path).expanduser().resolve()
            else:
                path = Path(item).expanduser().resolve()
                seed = infer_seed(path)
            pairs.append((seed, path))

    if not pairs:
        raise SystemExit("No input files provided. Use --glob and/or --inputs.")

    return pairs


def infer_seed(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot infer seed from filename '{path}'. Provide entries as seedX:/path/to/file."
    )


def read_records(files: Sequence[Tuple[int, Path]]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for seed, path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [col for col in ("Modello", *METRIC_COLUMNS) if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"File {path} missing columns: {missing}")
            for row in reader:
                row = dict(row)
                row["Seed"] = seed
                records.append(row)
    return records


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Cannot convert value '{value}' to float") from exc


def aggregate(records: Sequence[Dict[str, str]]):
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, int] = defaultdict(int)

    for row in records:
        model = row["Modello"]
        counts[model] += 1
        for metric in METRIC_COLUMNS:
            grouped[model][metric].append(to_float(row[metric]))

    summary = []
    for model, metric_map in grouped.items():
        entry = {"Modello": model, "Samples": counts[model]}
        for metric, values in metric_map.items():
            mean = fmean(values)
            std = stdev(values) if len(values) > 1 else 0.0
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
        summary.append(entry)

    return sorted(summary, key=lambda e: e["Modello"])


def write_csv(summary: Sequence[Dict[str, float]], path: Path):
    if not summary:
        raise SystemExit("No summary rows to write.")
    fieldnames = ["Modello", "Samples"] + [
        f"{metric}_{suffix}" for metric in METRIC_COLUMNS for suffix in ("mean", "std")
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_latex(summary: Sequence[Dict[str, float]], path: Path):
    headers = [
        "Modello",
        "$n$",
        "CLIP$_{\mu\pm\sigma}$",
        "BLEU$_1{}_{\mu\pm\sigma}$",
        "METEOR$_{\mu\pm\sigma}$",
        "ROUGE$_L{}_{\mu\pm\sigma}$",
        "Comp.$_{\mu\pm\sigma}$",
    ]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", " & ".join(headers) + " \\\", "\\midrule"]
    fmt = "{mean:.2f}\\pm{std:.2f}"
    for row in summary:
        line = " & ".join(
            [
                row["Modello"],
                str(row["Samples"]),
                fmt.format(mean=row["CLIPScore_mean"], std=row["CLIPScore_std"]),
                fmt.format(mean=row["BLEU-1_mean"], std=row["BLEU-1_std"]),
                fmt.format(mean=row["METEOR_mean"], std=row["METEOR_std"]),
                fmt.format(mean=row["ROUGE-L_mean"], std=row["ROUGE-L_std"]),
                fmt.format(mean=row["Punteggio_Composito_mean"], std=row["Punteggio_Composito_std"]),
            ]
        ) + " \\\
"
        lines.append(line)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    files = discover_files(args.glob, args.inputs)
    records = read_records(files)
    summary = aggregate(records)
    write_csv(summary, Path(args.output))
    if args.latex:
        write_latex(summary, Path(args.latex))
    print(f"Aggregated {len(records)} rows across {len(files)} files -> {args.output}")


if __name__ == "__main__":
    main()
