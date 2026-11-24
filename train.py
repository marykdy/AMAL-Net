#!/usr/bin/env python3
"""Unified training entry point for the AMAL-Net comparison.

One script, four model definitions, three datasets, identical data pipeline and
optimisation protocol.  The model files (AMALNet_Main.py, COMANet_Model.py,
UCTransNet_Official.py, nnUNet2D_Model.py) are imported unchanged.

Example:
    CUDA_VISIBLE_DEVICES=0 python3 train.py \
        --model amal --dataset liver --seed 42 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - depends on the runtime container
    raise SystemExit(
        f"Missing Python dependency: {exc}. Run this inside the GPU container "
        "(e.g. docker exec -it amal-gpu bash), not the CPU-only segmary container."
    )

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    DEFAULT_RUNS_DIR,
    NUM_CLASSES,
    SegMetrics,
    build_train_loader,
    build_eval_loader,
    cosine_warmup_lr,
    count_parameters,
    environment_default,
    save_json,
    seed_everything,
)
from models import MODEL_DISPLAY, MODEL_KEYS, build_model, criterion_total, forward_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified AMAL-Net / COMA-Net / UCTransNet / nnU-Net-2D training"
    )
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--dataset", required=True, choices=("liver", "pancreas", "lung"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int,
                        default=int(environment_default("BATCH_SIZE", 8)))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int,
                        default=int(environment_default("IMAGE_SIZE", 256)))
    parser.add_argument("--num-workers", type=int,
                        default=int(environment_default("NUM_WORKERS", 8)))
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=60,
                        help="Early-stop after this many epochs without val tumor-Dice improvement.")
    parser.add_argument("--tumor-oversample", type=float, default=3.0)
    parser.add_argument("--aug", choices=("standard", "none"), default="standard")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--out-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT") or None,
                        help="Override the dataset root; default data/<dataset>_3way.")
    parser.add_argument("--uctransnet-repo", default=os.environ.get("UCTRANSNET_REPO") or None)
    parser.add_argument("--uctransnet-impl", choices=("auto", "official", "local"),
                        default=os.environ.get("UCTRANSNET_IMPL", "auto"),
                        help="auto: official checkout if present, otherwise the "
                             "project-local reimplementation.")
    parser.add_argument("--uctransnet-base-ch", type=int,
                        default=int(environment_default("UCTRANSNET_BASE_CH", 64)))
    parser.add_argument("--nnunet-impl", choices=("plain", "dna"),
                        default=os.environ.get("NNUNET_IMPL", "plain"))
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--enlargement", type=float, default=2.0)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--num-prototypes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--channel-gate", action="store_true", default=False)
    # Ablation toggles (paper Table 3 + reviewer-requested ablations).
    parser.add_argument("--grid", default="3x3",
                        help="Stage-2 patch grid as HxW (e.g. 2x2, 3x3, 4x4)")
    parser.add_argument("--routing", choices=("learned_topk", "random_topk", "all"),
                        default="learned_topk",
                        help="Stage-2 patch routing: learned top-k, random k, or all patches")
    parser.add_argument("--use-scores", dest="use_scores", action="store_true", default=True)
    parser.add_argument("--no-scores", dest="use_scores", action="store_false",
                        help="Remove the patch importance scorer (routing must not be learned_topk)")
    parser.add_argument("--score-modulation", dest="score_modulation", action="store_true",
                        default=True)
    parser.add_argument("--no-score-modulation", dest="score_modulation", action="store_false",
                        help="Disable score-weighted fusion of refined patches")
    parser.add_argument("--stage2", dest="stage2", action="store_true", default=True)
    parser.add_argument("--no-stage2", dest="stage2", action="store_false",
                        help="Stage-1 only (single-stage baseline)")
    parser.add_argument("--patch-refinement", dest="patch_refinement", action="store_true",
                        default=True)
    parser.add_argument("--no-patch-refinement", dest="patch_refinement", action="store_false",
                        help="Stage-2 without sparse patch refinement")
    parser.add_argument("--foreground-branch", dest="foreground_branch", action="store_true",
                        default=True)
    parser.add_argument("--no-foreground-branch", dest="foreground_branch", action="store_false",
                        help="Remove the Stage-1 foreground branch")
    parser.add_argument("--background-branch", dest="background_branch", action="store_true",
                        default=True)
    parser.add_argument("--no-background-branch", dest="background_branch", action="store_false",
                        help="Remove the Stage-1 background branch")
    parser.add_argument("--deep-supervision", dest="deep_supervision", action="store_true",
                        default=True)
    parser.add_argument("--no-deep-supervision", dest="deep_supervision", action="store_false",
                        help="Disable deep supervision")
    parser.add_argument("--cluster-attention", dest="cluster_attention", action="store_true",
                        default=True)
    parser.add_argument("--no-cluster-attention", dest="cluster_attention", action="store_false",
                        help="Remove cluster attention from the foreground decoder")
    parser.add_argument("--density-attention", dest="density_attention", action="store_true",
                        default=True)
    parser.add_argument("--no-density-attention", dest="density_attention", action="store_false",
                        help="Remove density attention from the background decoder")
    parser.add_argument("--channel-spatial-attention", dest="channel_spatial_attention",
                        action="store_true", default=True)
    parser.add_argument("--no-channel-spatial-attention", dest="channel_spatial_attention",
                        action="store_false",
                        help="Remove the CBAM channel-spatial attention from fusion modules")
    parser.add_argument("--skip-attention", dest="skip_attention", action="store_true",
                        default=True)
    parser.add_argument("--no-skip-attention", dest="skip_attention", action="store_false",
                        help="Remove attention-gated skip fusion")
    parser.add_argument("--residual", dest="residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="residual", action="store_false",
                        help="Remove residual connections in decoder blocks")
    parser.add_argument("--score-rank-weight", type=float, default=0.0)
    parser.add_argument("--score-div-weight", type=float, default=0.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--warmup-all-epochs", type=int, default=0,
                        help="AMAL-Net only: use all patches for the first N epochs.")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="Smoke-test only: limit training batches per epoch.")
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def amp_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_state_dict_compat(model: nn.Module, path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state dict")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return ckpt


@torch.no_grad()
def evaluate_split(model: nn.Module, loader, criterion, device, args,
                   model_name: str, max_batches: int = 0) -> dict:
    model.eval()
    metrics = SegMetrics(NUM_CLASSES, device)
    total_loss, n_batches = 0.0, 0
    for step, (images, labels, _names) in enumerate(loader, 1):
        if max_batches and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_autocast(args.amp):
            output = forward_model(model, model_name, images, epoch=10 ** 9, args=args)
            loss = criterion_total(criterion(output, labels))
        metrics.update(output["stage2_final"].detach().argmax(1), labels)
        total_loss += float(loss.detach())
        n_batches += 1
    result = metrics.compute()
    result["loss"] = total_loss / max(n_batches, 1)
    return result


def save_checkpoint(path: Path, model: nn.Module, optimizer, scaler, epoch: int,
                    best_tumor_dice: float, best_miou: float,
                    best_epoch: int, args: argparse.Namespace,
                    history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_tumor_dice": best_tumor_dice,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "args": vars(args),
        "history": history,
    }, path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Run this inside the GPU container "
            "(e.g. docker exec -it amal-gpu bash) or on a GPU host."
        )
    device = torch.device("cuda")
    seed_everything(args.seed)

    run_dir = Path(args.out_dir) / f"{args.dataset}_{args.model}_s{args.seed}"
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "args.json", vars(args))
    print(f"[run] {MODEL_DISPLAY[args.model]} | dataset={args.dataset} | seed={args.seed}",
          flush=True)
    print(f"[run] output={run_dir}", flush=True)
    print(f"[run] cuda={torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)",
          flush=True)

    train_loader, train_ds = build_train_loader(
        args.dataset,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tumor_oversample=args.tumor_oversample,
        aug=args.aug,
    )
    val_loader, val_ds = build_eval_loader(
        args.dataset,
        split="val",
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[data] train={len(train_ds):,}  val={len(val_ds):,}", flush=True)

    model, criterion, meta = build_model(args.model, args)
    save_json(run_dir / "model_meta.json", meta)
    model = model.to(device)
    criterion = criterion.to(device)
    n_params = count_parameters(model)
    impl = meta.get("impl") if isinstance(meta, dict) else None
    print(f"[model] {MODEL_DISPLAY[args.model]} params={n_params:,} ({n_params / 1e6:.2f}M)"
          + (f" impl={impl}" if impl else ""), flush=True)

    if args.dry_run:
        images, labels, _ = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)
        with amp_autocast(args.amp):
            output = forward_model(model, args.model, images, epoch=1, args=args)
            loss = criterion_total(criterion(output, labels))
        print(f"[dry-run] images={tuple(images.shape)} labels={tuple(labels.shape)} "
              f"logits={tuple(output['stage2_final'].shape)} loss={float(loss):.4f}",
              flush=True)
        return

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scaler = make_grad_scaler(args.amp)

    start_epoch = 1
    best_tumor_dice = -1.0
    best_miou = -1.0
    best_epoch = 0
    history: list = []
    last_path = ckpt_dir / "last.pth"
    if args.resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        load_state_dict_compat(model, last_path)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_tumor_dice = float(ckpt.get("best_tumor_dice", -1.0))
        best_miou = float(ckpt.get("best_miou", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        history = ckpt.get("history", [])
        print(f"[resume] epoch={start_epoch - 1} best_tumor={best_tumor_dice:.4f}",
              flush=True)

    max_val_batches = args.limit_val_batches or (10 if args.limit_batches else 0)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = cosine_warmup_lr(epoch - 1, args.epochs, args.warmup_epochs,
                              args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        train_metrics = SegMetrics(NUM_CLASSES, device)
        total_loss, n_ok, n_skipped = 0.0, 0, 0
        n_steps = len(train_loader)
        if args.limit_batches:
            n_steps = min(n_steps, args.limit_batches)
        log_interval = max(1, n_steps // 5)

        for step, (images, labels, _names) in enumerate(train_loader, 1):
            if args.limit_batches and step > args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(args.amp):
                output = forward_model(model, args.model, images, epoch=epoch, args=args)
                loss = criterion_total(criterion(output, labels))
            if not torch.isfinite(loss):
                n_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach())
            n_ok += 1
            train_metrics.update(output["stage2_final"].detach().argmax(1), labels)
            if step % log_interval == 0 or step == n_steps:
                print(f"[epoch {epoch:03d}] step {step}/{n_steps} "
                      f"loss={float(loss.detach()):.4f} lr={lr:.2e}", flush=True)

        train_result = train_metrics.compute()
        train_result.pop("confusion_matrix", None)
        train_result["loss"] = total_loss / max(n_ok, 1)
        train_result["skipped_batches"] = n_skipped
        train_result["minutes"] = (time.time() - t0) / 60.0

        entry = {"epoch": epoch, "lr": lr, "train": train_result}
        should_validate = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        if should_validate:
            val_result = evaluate_split(model, val_loader, criterion, device, args,
                                        args.model, max_batches=max_val_batches)
            entry["val"] = {k: v for k, v in val_result.items() if k != "confusion_matrix"}
            print(f"[epoch {epoch:03d}] val mIoU={val_result['mean_iou']:.4f} "
                  f"organ={val_result['dice_organ']:.4f} tumor={val_result['dice_tumor']:.4f} "
                  f"loss={val_result['loss']:.4f}", flush=True)

            improved = False
            if val_result["dice_tumor"] > best_tumor_dice:
                best_tumor_dice = val_result["dice_tumor"]
                best_epoch = epoch
                improved = True
                save_checkpoint(ckpt_dir / "best_tumor_dice.pth", model, optimizer,
                                scaler, epoch, best_tumor_dice, best_miou,
                                best_epoch, args, history + [entry])
                print(f"[checkpoint] best_tumor_dice={best_tumor_dice:.4f} "
                      f"at epoch {epoch}", flush=True)
            if val_result["mean_iou"] > best_miou:
                best_miou = val_result["mean_iou"]
                save_checkpoint(ckpt_dir / "best_miou.pth", model, optimizer,
                                scaler, epoch, best_tumor_dice, best_miou,
                                best_epoch, args, history + [entry])
                print(f"[checkpoint] best_miou={best_miou:.4f} at epoch {epoch}",
                      flush=True)
            if not improved and args.patience > 0 and epoch - best_epoch >= args.patience:
                print(f"[early-stop] no tumor-Dice improvement for "
                      f"{epoch - best_epoch} epochs", flush=True)
                history.append(entry)
                break

        history.append(entry)
        save_json(run_dir / "metrics.json", {"history": history})
        save_checkpoint(last_path, model, optimizer, scaler, epoch,
                        best_tumor_dice, best_miou, best_epoch, args, history)

    save_json(run_dir / "metrics.json", {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_tumor_dice": best_tumor_dice,
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "history": history,
    })
    print(f"[done] best_tumor_dice={best_tumor_dice:.4f} best_miou={best_miou:.4f} "
          f"best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
