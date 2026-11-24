#!/usr/bin/env python3
"""Launch a manifest of (model, dataset, seed) jobs across the available GPUs.

Example:
    python3 run_matrix.py --manifest manifests/main.tsv \
        --gpus 0,1,2,3,4,5 --uctransnet-repo ./UCTransNet

Each job calls run_one.sh, which trains then evaluates the two val-selected
checkpoints on the held-out test split.  Existing completed runs are skipped
unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "runs"
MODEL_KEYS = ("amal", "coma", "uctransnet", "nnunet2d")
DATASETS = ("liver", "pancreas", "lung")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the comparison matrix")
    parser.add_argument("--manifest", default=str(HERE / "manifests" / "main.tsv"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-parallel", type=int, default=0,
                        help="0 = one job per GPU in --gpus")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--uctransnet-repo", default=os.environ.get("UCTRANSNET_REPO") or None)
    parser.add_argument("--uctransnet-impl", choices=("auto", "official", "local"),
                        default=os.environ.get("UCTRANSNET_IMPL", "auto"))
    parser.add_argument("--uctransnet-base-ch", type=int, default=None)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT") or None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--tumor-oversample", type=float, default=None)
    parser.add_argument("--aug", choices=("standard", "none"), default=None)
    parser.add_argument("--nnunet-impl", choices=("plain", "dna"), default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to train.py (continue from last.pth).")
    parser.add_argument("--hd95", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[tuple[str, str, int]]:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    jobs: list[tuple[str, str, int]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0].lower() == "model":
            continue
        if len(parts) < 3:
            raise SystemExit(f"{path}:{lineno}: expected 'model dataset seed', got {raw!r}")
        model, dataset, seed = parts[0].lower(), parts[1].lower(), int(parts[2])
        if model not in MODEL_KEYS:
            raise SystemExit(f"{path}:{lineno}: unknown model {model!r}")
        if dataset not in DATASETS:
            raise SystemExit(f"{path}:{lineno}: unknown dataset {dataset!r}")
        jobs.append((model, dataset, seed))
    if not jobs:
        raise SystemExit(f"No jobs in {path}")
    return jobs


def build_command(args: argparse.Namespace, model: str, dataset: str, seed: int) -> list[str]:
    cmd = ["bash", str(HERE / "run_one.sh"), model, dataset, str(seed)]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.limit_batches is not None:
        cmd += ["--limit-batches", str(args.limit_batches)]
    if args.limit_val_batches is not None:
        cmd += ["--limit-val-batches", str(args.limit_val_batches)]
    if args.tumor_oversample is not None:
        cmd += ["--tumor-oversample", str(args.tumor_oversample)]
    if args.aug is not None:
        cmd += ["--aug", args.aug]
    if args.resume:
        cmd += ["--resume"]
    return cmd


def build_env(args: argparse.Namespace, gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["RUNS_DIR"] = str(args.runs_dir)
    if args.uctransnet_repo:
        env["UCTRANSNET_REPO"] = str(args.uctransnet_repo)
    if args.uctransnet_impl:
        env["UCTRANSNET_IMPL"] = str(args.uctransnet_impl)
    if args.uctransnet_base_ch is not None:
        env["UCTRANSNET_BASE_CH"] = str(args.uctransnet_base_ch)
    if args.data_root:
        env["DATA_ROOT"] = str(args.data_root)
    if args.image_size is not None:
        env["IMAGE_SIZE"] = str(args.image_size)
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.num_workers is not None:
        env["NUM_WORKERS"] = str(args.num_workers)
    if args.nnunet_impl:
        env["NNUNET_IMPL"] = str(args.nnunet_impl)
    if args.hd95:
        env["HD95"] = "1"
    return env


def main() -> None:
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")
    # Fail early instead of launching jobs that will hit CUDA_VISIBLE_DEVICES=4
    # on a four-GPU machine.  Skipped automatically in the CPU-only container.
    try:
        import torch
        visible = torch.cuda.device_count()
        if visible:
            invalid = [g for g in gpus if g.isdigit() and int(g) >= visible]
            if invalid:
                raise SystemExit(
                    f"--gpus includes {invalid}, but torch sees only {visible} "
                    f"GPU(s). Use --gpus 0,1,2,3 (or the visible set)."
                )
    except ImportError:
        pass
    max_parallel = args.max_parallel or len(gpus)
    max_parallel = min(max_parallel, len(gpus))

    jobs = read_manifest(Path(args.manifest))
    runs_dir = Path(args.runs_dir)
    log_dir = runs_dir / "_matrix"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, str, int]] = []
    skipped: list[tuple[str, str, int]] = []
    for job in jobs:
        model, dataset, seed = job
        done = runs_dir / f"{dataset}_{model}_s{seed}" / "eval_test_best_tumor.json"
        if done.is_file() and not args.force:
            skipped.append(job)
        else:
            pending.append(job)

    print(f"[matrix] {len(jobs)} jobs, {len(skipped)} already complete, "
          f"{len(pending)} to run, GPUs={gpus}, max_parallel={max_parallel}", flush=True)

    if args.dry_run:
        for model, dataset, seed in pending:
            gpu = gpus[0]
            env = build_env(args, gpu)
            cmd = build_command(args, model, dataset, seed)
            print(f"[dry-run] CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
            print(f"[dry-run]   env DATA_ROOT={env.get('DATA_ROOT')} "
                  f"UCTRANSNET_REPO={env.get('UCTRANSNET_REPO')} "
                  f"UCTRANSNET_IMPL={env.get('UCTRANSNET_IMPL')} "
                  f"IMAGE_SIZE={env.get('IMAGE_SIZE')} BATCH_SIZE={env.get('BATCH_SIZE')}")
        return

    gpu_load = {gpu: 0 for gpu in gpus}
    running: list[tuple[subprocess.Popen, tuple[str, str, int], str, Path, object]] = []
    failed: list[tuple[tuple[str, str, int], int]] = []
    completed: list[tuple[str, str, int]] = []
    pending_iter = iter(pending)

    while True:
        while len(running) < max_parallel:
            try:
                model, dataset, seed = next(pending_iter)
            except StopIteration:
                break
            gpu = min(gpus, key=lambda g: (gpu_load[g], int(g) if g.isdigit() else 0))
            gpu_load[gpu] += 1
            cmd = build_command(args, model, dataset, seed)
            env = build_env(args, gpu)
            log_path = log_dir / f"{dataset}_{model}_s{seed}.out"
            log_handle = open(log_path, "w", encoding="utf-8")
            print(f"[matrix] start {dataset}_{model}_s{seed} on GPU {gpu} -> {log_path}",
                  flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=log_handle,
                                    stderr=subprocess.STDOUT, cwd=str(HERE))
            running.append((proc, (model, dataset, seed), gpu, log_path, log_handle))

        if not running:
            break

        time.sleep(5)
        still_running = []
        for proc, job, gpu, log_path, log_handle in running:
            code = proc.poll()
            if code is None:
                still_running.append((proc, job, gpu, log_path, log_handle))
                continue
            gpu_load[gpu] -= 1
            try:
                log_handle.close()
            except Exception:
                pass
            if code == 0:
                completed.append(job)
                print(f"[matrix] done  {job[1]}_{job[0]}_s{job[2]} (GPU {gpu})", flush=True)
            else:
                failed.append((job, code))
                print(f"[matrix] FAIL  {job[1]}_{job[0]}_s{job[2]} (GPU {gpu}) "
                      f"exit={code}; see {log_path}", flush=True)
        running = still_running

    print(f"[matrix] completed={len(completed)} skipped={len(skipped)} "
          f"failed={len(failed)}", flush=True)
    if failed:
        for job, code in failed:
            print(f"[matrix] failed: {job[1]}_{job[0]}_s{job[2]} exit={code}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
