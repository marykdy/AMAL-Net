#!/usr/bin/env python3
"""Aggregate eval_test_*.json files into mean ± std tables.

Example:
    python3 summarize.py --runs-dir runs
    python3 summarize.py --runs-dir runs --ckpt best_miou
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

MODEL_ORDER = ("amal", "coma", "uctransnet", "nnunet2d")
MODEL_DISPLAY = {
    "amal": "AMAL-Net",
    "coma": "COMA-Net",
    "uctransnet": "UCTransNet",
    "nnunet2d": "nnU-Net 2D (architecture)",
}
DATASET_ORDER = ("liver", "pancreas", "lung")
METRICS = ("tumor_dice", "organ_dice", "mean_dice", "miou", "tumor_iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize comparison results")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--ckpt", choices=("best_tumor_dice", "best_miou"),
                        default="best_tumor_dice")
    parser.add_argument("--baseline", default="amal",
                        help="Model key for paired deltas; empty string disables.")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    return parser.parse_args()


def fmt(value, digits=4):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def mean_std(values: list[float]):
    if not values:
        return None, None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    # run_one.sh writes eval_test_best_tumor.json (not ...best_tumor_dice.json).
    suffix = "best_tumor" if args.ckpt == "best_tumor_dice" else "best_miou"
    pattern = f"*/eval_test_{suffix}.json"
    files = sorted(runs_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {runs_dir / pattern}")

    # (dataset, model) -> list of records
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_seed: dict[tuple[str, str, int], dict] = {}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] skip {path}: {exc}")
            continue
        model = payload.get("model")
        dataset = payload.get("dataset")
        seed = payload.get("seed")
        metrics = payload.get("metrics", {})
        if not model or not dataset or seed is None or not metrics:
            print(f"[warn] skip {path}: missing model/dataset/seed/metrics")
            continue
        record = {
            "model": model,
            "model_impl": payload.get("model_impl", "provided"),
            "dataset": dataset,
            "seed": int(seed),
            "path": str(path),
            "tumor_dice": metrics.get("dice_tumor"),
            "organ_dice": metrics.get("dice_organ"),
            "mean_dice": metrics.get("mean_dice"),
            "miou": metrics.get("mean_iou"),
            "tumor_iou": metrics.get("iou_tumor"),
        }
        grouped[(dataset, model)].append(record)
        by_seed[(dataset, model, int(seed))] = record

    datasets = [d for d in DATASET_ORDER if any(k[0] == d for k in grouped)]
    datasets += sorted({k[0] for k in grouped} - set(datasets))
    models = [m for m in MODEL_ORDER if any(k[1] == m for k in grouped)]
    models += sorted({k[1] for k in grouped} - set(models))

    def label_for(model: str, records: list[dict]) -> str:
        label = MODEL_DISPLAY.get(model, model)
        impls = sorted({r.get("model_impl", "provided") for r in records})
        if model == "uctransnet" and len(impls) == 1 and impls[0] in ("local", "official"):
            label += f" ({impls[0]})"
        elif len(impls) > 1:
            label += " (mixed)"
        return label

    rows = []
    for dataset in datasets:
        for model in models:
            records = grouped.get((dataset, model), [])
            if not records:
                continue
            impls = sorted({r.get("model_impl", "provided") for r in records})
            row = {
                "dataset": dataset,
                "model": model,
                "model_impl": impls[0] if len(impls) == 1 else "mixed",
                "n": len(records),
            }
            for metric in METRICS:
                values = [r[metric] for r in records if r[metric] is not None]
                mean, std = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            rows.append(row)

    # Markdown output
    md_lines = [f"# Comparison results ({args.ckpt} on test)", ""]
    md_lines.append("Mean ± sample standard deviation across seeds; "
                    "checkpoints selected on validation only.")
    md_lines.append("")
    for dataset in datasets:
        md_lines.append(f"## {dataset.capitalize()}")
        md_lines.append("")
        md_lines.append("| Model | n | Tumor Dice | Organ Dice | Mean Dice | mIoU | Tumor IoU |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for model in models:
            row = next((r for r in rows if r["dataset"] == dataset and r["model"] == model), None)
            if row is None:
                continue
            def cell(metric):
                mean = row.get(f"{metric}_mean")
                std = row.get(f"{metric}_std")
                if mean is None:
                    return "—"
                return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"
            records = grouped.get((dataset, model), [])
            md_lines.append(
                f"| {label_for(model, records)} | {row['n']} | "
                f"{cell('tumor_dice')} | {cell('organ_dice')} | "
                f"{cell('mean_dice')} | {cell('miou')} | {cell('tumor_iou')} |"
            )
        md_lines.append("")

    # Paired deltas vs baseline
    if args.baseline:
        baseline_records = [r for (d, m), rs in grouped.items()
                            if m == args.baseline for r in rs]
        md_lines.append(
            f"## Paired tumor-Dice differences vs "
            f"{label_for(args.baseline, baseline_records)}"
        )
        md_lines.append("")
        md_lines.append("| Dataset | Model | n pairs | Δ Tumor Dice |")
        md_lines.append("|---|---|---:|---:|")
        for dataset in datasets:
            for model in models:
                if model == args.baseline:
                    continue
                deltas = []
                for seed in sorted({k[2] for k in by_seed if k[0] == dataset}):
                    base = by_seed.get((dataset, args.baseline, seed))
                    cur = by_seed.get((dataset, model, seed))
                    if base and cur and base["tumor_dice"] is not None and cur["tumor_dice"] is not None:
                        deltas.append(cur["tumor_dice"] - base["tumor_dice"])
                if deltas:
                    md_lines.append(
                        f"| {dataset} | {label_for(model, grouped.get((dataset, model), []))} | "
                        f"{len(deltas)} | {statistics.mean(deltas):+.4f} |"
                    )
        md_lines.append("")

    md_text = "\n".join(md_lines)
    print(md_text)

    out_md = Path(args.out_md) if args.out_md else runs_dir / f"summary_{args.ckpt}.md"
    out_csv = Path(args.out_csv) if args.out_csv else runs_dir / f"summary_{args.ckpt}.csv"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")

    fieldnames = ["dataset", "model", "model_impl", "n"]
    for metric in METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_std"]
    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[saved] {out_md}\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
