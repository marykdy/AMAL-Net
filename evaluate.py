#!/usr/bin/env python3
"""Evaluate one checkpoint on the held-out split with the common metric code.

Example:
    python3 evaluate.py --model amal --dataset liver \
        --checkpoint runs/liver_amal_s42/ckpt/best_tumor_dice.pth \
        --split test --out runs/liver_amal_s42/eval_test_best_tumor.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - depends on the runtime container
    raise SystemExit(
        f"Missing Python dependency: {exc}. Run this inside the GPU container "
        "(e.g. docker exec -it amal-gpu bash), not the CPU-only segmary container."
    )

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    NUM_CLASSES,
    SegMetrics,
    build_eval_loader,
    environment_default,
    save_json,
    seed_everything,
)
from models import MODEL_KEYS, build_model, forward_model  # noqa: E402

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a modelreview checkpoint")
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--dataset", required=True, choices=("liver", "pancreas", "lung"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=42,
                        help="Only recorded in the output; evaluation is deterministic.")
    parser.add_argument("--batch-size", type=int,
                        default=int(environment_default("BATCH_SIZE", 8)))
    parser.add_argument("--image-size", type=int,
                        default=int(environment_default("IMAGE_SIZE", 256)))
    parser.add_argument("--num-workers", type=int,
                        default=int(environment_default("NUM_WORKERS", 8)))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT") or None)
    parser.add_argument("--uctransnet-repo", default=os.environ.get("UCTRANSNET_REPO") or None)
    parser.add_argument("--uctransnet-impl", choices=("auto", "official", "local"),
                        default=os.environ.get("UCTRANSNET_IMPL", "auto"))
    parser.add_argument("--uctransnet-base-ch", type=int,
                        default=int(environment_default("UCTRANSNET_BASE_CH", 64)))
    parser.add_argument("--nnunet-impl", choices=("plain", "dna"),
                        default=os.environ.get("NNUNET_IMPL", "plain"))
    parser.add_argument("--base-channels", type=int,
                        default=int(environment_default("BASE_CHANNELS", 32)))
    parser.add_argument("--top-k", type=int,
                        default=int(environment_default("TOP_K", 4)))
    parser.add_argument("--enlargement", type=float,
                        default=float(environment_default("ENLARGEMENT", 2.0)))
    parser.add_argument("--halo", type=int,
                        default=int(environment_default("HALO", 4)))
    parser.add_argument("--num-prototypes", type=int,
                        default=int(environment_default("NUM_PROTOTYPES", 4)))
    parser.add_argument("--dropout", type=float,
                        default=float(environment_default("DROPOUT", 0.05)))
    parser.add_argument("--channel-gate", action="store_true",
                        default=os.environ.get("CHANNEL_GATE", "0") == "1")
    # Ablation toggles mirroring train.py; normally loaded from args.json.
    parser.add_argument("--grid", default="3x3")
    parser.add_argument("--routing", choices=("learned_topk", "random_topk", "all"),
                        default="learned_topk")
    parser.add_argument("--use-scores", dest="use_scores", action="store_true", default=True)
    parser.add_argument("--no-scores", dest="use_scores", action="store_false")
    parser.add_argument("--score-modulation", dest="score_modulation", action="store_true",
                        default=True)
    parser.add_argument("--no-score-modulation", dest="score_modulation", action="store_false")
    parser.add_argument("--stage2", dest="stage2", action="store_true", default=True)
    parser.add_argument("--no-stage2", dest="stage2", action="store_false")
    parser.add_argument("--patch-refinement", dest="patch_refinement", action="store_true",
                        default=True)
    parser.add_argument("--no-patch-refinement", dest="patch_refinement", action="store_false")
    parser.add_argument("--foreground-branch", dest="foreground_branch", action="store_true",
                        default=True)
    parser.add_argument("--no-foreground-branch", dest="foreground_branch", action="store_false")
    parser.add_argument("--background-branch", dest="background_branch", action="store_true",
                        default=True)
    parser.add_argument("--no-background-branch", dest="background_branch", action="store_false")
    parser.add_argument("--deep-supervision", dest="deep_supervision", action="store_true",
                        default=True)
    parser.add_argument("--no-deep-supervision", dest="deep_supervision", action="store_false")
    parser.add_argument("--cluster-attention", dest="cluster_attention", action="store_true",
                        default=True)
    parser.add_argument("--no-cluster-attention", dest="cluster_attention", action="store_false")
    parser.add_argument("--density-attention", dest="density_attention", action="store_true",
                        default=True)
    parser.add_argument("--no-density-attention", dest="density_attention", action="store_false")
    parser.add_argument("--channel-spatial-attention", dest="channel_spatial_attention",
                        action="store_true", default=True)
    parser.add_argument("--no-channel-spatial-attention", dest="channel_spatial_attention",
                        action="store_false")
    parser.add_argument("--skip-attention", dest="skip_attention", action="store_true",
                        default=True)
    parser.add_argument("--no-skip-attention", dest="skip_attention", action="store_false")
    parser.add_argument("--residual", dest="residual", action="store_true", default=True)
    parser.add_argument("--no-residual", dest="residual", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--hd95", action="store_true",
                        help="Also compute tumor HD95/ASSD (requires scipy).")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def amp_autocast(enabled: bool):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def load_state_dict(model: torch.nn.Module, path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state dict")
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return ckpt


def surface_distances(pred: np.ndarray, gt: np.ndarray):
    """HD95 and ASSD for one binary 2-D mask pair; None if either is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    struct = np.ones((3, 3), dtype=bool)
    ps = pred & ~binary_erosion(pred, structure=struct, border_value=0)
    gs = gt & ~binary_erosion(gt, structure=struct, border_value=0)
    if ps.sum() == 0 or gs.sum() == 0:
        return None
    dt_gt = distance_transform_edt(~gs)
    dt_pred = distance_transform_edt(~ps)
    d = np.concatenate([dt_gt[ps], dt_pred[gs]])
    return float(np.percentile(d, 95)), float(np.mean(d))


def apply_saved_args(args: argparse.Namespace) -> None:
    """Overlay model-config fields from the run's args.json so evaluation
    always rebuilds the exact architecture that was trained (ablation runs
    included)."""
    base = Path(args.checkpoint).parent
    aj = base / "args.json"
    if not aj.is_file():
        # args.json lives in the run root, not in ckpt/.
        aj = base.parent / "args.json"
    if not aj.is_file():
        return
    saved = json.loads(aj.read_text())
    skip = {"model", "dataset", "checkpoint", "split", "out", "data_root"}
    for k, v in saved.items():
        if k in skip or not hasattr(args, k):
            continue
        setattr(args, k, v)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    apply_saved_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run inside the GPU container.")
    device = torch.device("cuda")
    seed_everything(args.seed)

    model, _criterion, meta = build_model(args.model, args)
    model = model.to(device)
    print(f"[eval] model={args.model} impl={meta.get('impl', 'provided')}", flush=True)
    load_state_dict(model, args.checkpoint)
    model.eval()

    loader, ds = build_eval_loader(
        args.dataset,
        split=args.split,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[eval] {args.model} {args.dataset}/{args.split}: {len(ds):,} slices", flush=True)

    metrics = SegMetrics(NUM_CLASSES, device)
    use_hd95 = bool(args.hd95)
    if use_hd95 and not HAS_SCIPY:
        print("[warn] scipy not available; skipping HD95/ASSD", flush=True)
        use_hd95 = False
    hd95_values, assd_values = [], []
    n_slices = 0

    for images, labels, _names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_autocast(args.amp):
            output = forward_model(model, args.model, images, epoch=10 ** 9, args=args)
        logits = output["stage2_final"]
        pred = logits.argmax(1)
        metrics.update(pred, labels)
        n_slices += int(images.shape[0])

        if use_hd95:
            pred_np = pred.detach().cpu().numpy()
            gt_np = labels.detach().cpu().numpy()
            for i in range(pred_np.shape[0]):
                pair = surface_distances(pred_np[i] == 2, gt_np[i] == 2)
                if pair is not None:
                    hd95_values.append(pair[0])
                    assd_values.append(pair[1])

    result = metrics.compute()
    payload = {
        "model": args.model,
        "model_impl": meta.get("impl", "provided"),
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "image_size": args.image_size,
        "n_slices": n_slices,
        "metrics": result,
    }
    if use_hd95:
        payload["hd95"] = {
            "tumor_hd95_mean": float(np.mean(hd95_values)) if hd95_values else None,
            "tumor_assd_mean": float(np.mean(assd_values)) if assd_values else None,
            "n_slices_with_both_tumor": len(hd95_values),
        }

    out = Path(args.out) if args.out else (
        Path(args.checkpoint).parent.parent
        / f"eval_{args.split}_{Path(args.checkpoint).stem}.json"
    )
    save_json(out, payload)
    m = result
    print(f"[eval] mIoU={m['mean_iou']:.4f} meanDice={m['mean_dice']:.4f} "
          f"organ={m['dice_organ']:.4f} tumor={m['dice_tumor']:.4f}", flush=True)
    if use_hd95:
        print(f"[eval] tumor HD95={payload['hd95']['tumor_hd95_mean']} "
              f"ASSD={payload['hd95']['tumor_assd_mean']} "
              f"(n={payload['hd95']['n_slices_with_both_tumor']})", flush=True)
    print(f"[eval] saved {out}", flush=True)


if __name__ == "__main__":
    main()
