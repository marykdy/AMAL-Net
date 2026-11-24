"""Unified model factory for the four comparison methods.

The four model definition files are imported unchanged.  No architecture or
loss file is modified here.

Model keys used on the command line / in manifests:
    amal         -> AMALNet_Main.AMALNet + AMALLoss
    amal_unified -> ourunified.ImprovedTwoStageSegmentation + AMALUnifiedLoss
    coma         -> COMANet_Model.COMANetPaperReproduction + COMANetLoss
    uctransnet   -> UCTransNet_Official.UCTransNetOfficial + StandardSegLoss
    nnunet2d     -> nnUNet2D_Model.NNUNet2DArchitecture + StandardSegLoss
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import AMALNet_Main as amal  # noqa: E402
import COMANet_Model as coma  # noqa: E402
import nnUNet2D_Model as nnunet  # noqa: E402
import UCTransNet_Official as uct  # noqa: E402
from baselines_adapters import (  # noqa: E402
    UCTransNetLocalAdapter,
    LOCAL_UCTRANSNET_AVAILABLE,
    uctransnet_local_description,
)

# ---------------------------------------------------------------------------
# Optional zoo of the original paper's comparison models.  The definitions in
# the livercode checkout are imported unchanged so TransUNet / Swin-UNet /
# TransFuse / MedNeXt / VM-UNet use exactly the architectures from the
# original experiments, now trained by the common-protocol pipeline here.
# ---------------------------------------------------------------------------
COMPARISON_ZOO_KEYS = ("transunet", "swin_unet", "transfuse", "mednext", "vm_unet")
_COMPARISON_ZOO_PY = Path(
    os.environ.get(
        "COMPARISON_MODELS_PY",
        "./comparison_models.py",
    )
)
compzoo = None
if _COMPARISON_ZOO_PY.is_file():
    import importlib.util

    _spec = importlib.util.spec_from_file_location("_comparison_models", _COMPARISON_ZOO_PY)
    compzoo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(compzoo)

# ---------------------------------------------------------------------------
# amal_unified: the pre-revision architecture (ImprovedTwoStageSegmentation
# from livercode model/ourunified.py) — UCTransNet-style channel transformer,
# all-nine-patch soft importance modulation, diversity/correlation score
# supervision.  Loaded verbatim so the architecture itself is compared, not a
# re-implementation of it.
# ---------------------------------------------------------------------------
_OURUNIFIED_PY = Path(
    os.environ.get(
        "OURUNIFIED_MODELS_PY",
        "./ourunified.py",
    )
)
uni = None
if _OURUNIFIED_PY.is_file():
    _spec = importlib.util.spec_from_file_location("_amal_unified", _OURUNIFIED_PY)
    uni = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(uni)

MODEL_KEYS = ("amal", "coma", "uctransnet", "nnunet2d")
if uni is not None:
    MODEL_KEYS += ("amal_unified",)
if compzoo is not None:
    MODEL_KEYS += COMPARISON_ZOO_KEYS
MODEL_DISPLAY = {
    "amal": "AMAL-Net",
    "coma": "COMA-Net",
    "uctransnet": "UCTransNet",
    "nnunet2d": "nnU-Net 2D (architecture)",
    "amal_unified": "AMAL-Net (unified/CCT)",
    "transunet": "TransUNet",
    "swin_unet": "Swin-UNet",
    "transfuse": "TransFuse",
    "mednext": "MedNeXt",
    "vm_unet": "VM-UNet",
}


class _DNAWrapper(nn.Module):
    """Normalise dynamic-network-architectures output to the common dict form."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> dict:
        output = self.network(x)
        if isinstance(output, (list, tuple)):
            return {"stage2_final": output[0], "deep_supervision": list(output[1:])}
        return {"stage2_final": output}


class AMALUnifiedLoss(nn.Module):
    """Common-protocol loss family for the unified architecture.

    Same family as the AMAL loss (CE + Focal + Dice + Tversky 0.3/0.7, deep
    supervision, Stage-1 priors) with no class weights and no label
    smoothing, plus the two patch-score supervision terms that the unified
    architecture requires (diversity + correlation).  The background prior
    uses the complementary target (Eq. (11), bg_target='bg'), not the
    original code's buggy foreground target.  Returns a dict with 'total'.
    """

    def __init__(self, num_classes=3, focal_gamma=2.0, dice_weight=0.5,
                 deep_sup_weights=(0.3, 0.2), patch_diversity_weight=0.05,
                 patch_correlation_weight=0.10, tversky_alpha=0.3,
                 tversky_beta=0.7, eps=1e-7):
        super().__init__()
        self.nc = num_classes
        self.eps = eps
        self.gamma = focal_gamma
        self.dw = dice_weight
        self.ds_weights = list(deep_sup_weights or [])
        self.div_w = patch_diversity_weight
        self.corr_w = patch_correlation_weight
        self.tv_alpha = tversky_alpha
        self.tv_beta = tversky_beta

    def _focal(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

    def _dice(self, pred, target, smooth=1.0):
        ps = F.softmax(pred, dim=1).clamp(self.eps, 1 - self.eps)
        oh = F.one_hot(target, self.nc).permute(0, 3, 1, 2).float()
        inter = (ps * oh).sum(dim=(2, 3))
        union = ps.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        d = (2 * inter + smooth) / (union + smooth)
        return 1 - d.mean()

    def _tversky(self, pred, target, smooth=1.0):
        ps = F.softmax(pred, dim=1).clamp(self.eps, 1 - self.eps)
        oh = F.one_hot(target, self.nc).permute(0, 3, 1, 2).float()
        tp = (ps * oh).sum(dim=(2, 3))
        fp = (ps * (1 - oh)).sum(dim=(2, 3))
        fn = ((1 - ps) * oh).sum(dim=(2, 3))
        ti = (tp + smooth) / (tp + self.tv_alpha * fp + self.tv_beta * fn + smooth)
        return 1 - ti.mean()

    def _seg_loss(self, pred, target):
        ce = F.cross_entropy(pred, target)
        focal = self._focal(pred, target)
        dice = self._dice(pred, target)
        tversky = self._tversky(pred, target)
        dw = self.dw
        loss = (1 - dw) * 0.5 * (ce + focal) + dw * 0.6 * dice + dw * 0.4 * tversky
        return loss, {'ce': ce.item(), 'focal': focal.item(),
                      'dice': dice.item(), 'tversky': tversky.item()}

    def _patch_importance_loss(self, imp_logit, labels, patch_hw=None):
        B = imp_logit.shape[0]
        imp = torch.sigmoid(imp_logit)                   # [B, 9] in (0,1)
        var_per_sample = imp.var(dim=1)                  # [B]
        diversity_loss = -var_per_sample.mean()

        H, W = labels.shape[1:]
        if patch_hw is None:
            ph, pw = H // 3, W // 3
        else:
            ph, pw = int(patch_hw[0]), int(patch_hw[1])
        Hp, Wp = 3 * ph, 3 * pw
        if Hp != H or Wp != W:
            labels_p = F.pad(labels.unsqueeze(1).float(),
                             (0, Wp - W, 0, Hp - H), mode='replicate').squeeze(1).long()
        else:
            labels_p = labels

        fg_ratio = torch.zeros(B, 9, device=labels.device)
        idx = 0
        for i in range(3):
            for j in range(3):
                patch_lbl = labels_p[:, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                organ = (patch_lbl == 1).float().mean(dim=(1, 2))
                tumor = (patch_lbl == 2).float().mean(dim=(1, 2))
                fg_ratio[:, idx] = organ + 3.0 * tumor
                idx += 1
        fg_max = fg_ratio.max(dim=1, keepdim=True)[0].clamp(min=self.eps)
        fg_target = fg_ratio / fg_max                    # [B, 9] in [0,1]
        correlation_loss = F.mse_loss(imp, fg_target)

        total = self.div_w * diversity_loss + self.corr_w * correlation_loss
        return total, {
            'patch_div': diversity_loss.item(),
            'patch_corr': correlation_loss.item(),
            'imp_var': var_per_sample.mean().item(),
        }

    def forward(self, outputs, labels):
        main_loss, detail = self._seg_loss(outputs['stage2_final'], labels)
        fg = (labels > 0).long()
        bg = 1 - fg
        aux = 0.5 * (F.cross_entropy(outputs['stage1_fg'], fg) +
                     F.cross_entropy(outputs['stage1_bg'], bg))
        total = main_loss + 0.15 * aux

        ds_info = outputs.get('deep_supervision', {})
        for key, w in zip(['ds_d3', 'ds_d2'], self.ds_weights):
            if key in ds_info:
                ds_pred = ds_info[key]
                ds_target = F.interpolate(
                    labels.unsqueeze(1).float(), size=ds_pred.shape[2:],
                    mode='nearest').squeeze(1).long()
                total = total + w * (F.cross_entropy(ds_pred, ds_target)
                                     + self._dice(ds_pred, ds_target))

        if 'importance_logit' in outputs:
            pi_loss, pi_detail = self._patch_importance_loss(
                outputs['importance_logit'], labels,
                patch_hw=outputs.get('patch_hw'))
            total = total + pi_loss
            detail.update(pi_detail)

        detail['total'] = total
        detail['aux'] = aux.item()
        return detail


class AMALProLoss(nn.Module):
    """AMAL-Loss + score ranking / diversity terms for the top-k scorer.

    Diagnostics on the liver checkpoints showed that 50-75% of samples have
    all nine patch scores nearly identical (std < 0.02), so top-k is close to
    random.  These two terms are training-only; the model architecture and
    top-k inference path are unchanged.
    """

    def __init__(self, base: nn.Module, rank_weight: float = 0.0,
                 div_weight: float = 0.0, margin: float = 0.1):
        super().__init__()
        self.base = base
        self.rank_weight = float(rank_weight)
        self.div_weight = float(div_weight)
        self.margin = float(margin)

    def forward(self, output, target):
        terms = self.base(output, target)
        total = terms["total"]
        if (self.rank_weight or self.div_weight) and "importance_logit" in output and "grid_meta" in output:
            scores = output["importance_logit"].float()
            patch_labels = amal.label_patches(target, output["grid_meta"], ignore_index=255)
            ratio = (patch_labels == 2).float().mean(dim=(-1, -2))  # [B,9]
            if self.rank_weight:
                diff = ratio[:, :, None] - ratio[:, None, :]
                mask = diff.abs() > 1e-6
                if mask.any():
                    score_diff = scores[:, :, None] - scores[:, None, :]
                    rank = F.relu(self.margin - torch.sign(diff) * score_diff)[mask].mean()
                    total = total + self.rank_weight * rank
                    terms["score_rank"] = rank.detach()
            if self.div_weight:
                prob = torch.sigmoid(scores)
                div = -prob.std(dim=1, unbiased=False).mean()
                total = total + self.div_weight * div
                terms["score_div"] = div.detach()
        terms["total"] = total
        return terms


def build_model(model_name: str, args: Any) -> tuple[nn.Module, nn.Module, dict]:
    """Build a model and its native/common criterion.

    Returns (model, criterion, metadata).
    """
    model_name = model_name.lower()
    if model_name not in MODEL_KEYS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_KEYS}")

    if model_name == "amal":
        # Keep the model definition and its paper defaults intact.  Only the
        # data-dependent fields (channels/classes/label ids) are set here.
        # Ablation toggles are read from args so reviewer-requested variants
        # (routing / stage2 / branches / attention / deep supervision) train
        # and evaluate with the exact same config saved in args.json.
        grid_str = str(getattr(args, "grid", "3x3")).lower()
        try:
            gh, gw = (int(v) for v in grid_str.split("x"))
        except ValueError:
            raise ValueError(f"Invalid grid {grid_str!r}; expected HxW like 3x3")
        cfg = amal.AMALConfig(
            in_channels=3,
            num_classes=3,
            lesion_ids=(2,),
            foreground_ids=(1, 2),
            base_channels=getattr(args, "base_channels", 32),
            grid=(gh, gw),
            top_k=getattr(args, "top_k", 4),
            routing=str(getattr(args, "routing", "learned_topk")),
            use_scores=bool(getattr(args, "use_scores", True)),
            score_modulation=bool(getattr(args, "score_modulation", True)),
            score_floor=0.25,
            enlargement=getattr(args, "enlargement", 2.0),
            halo=getattr(args, "halo", 4),
            num_prototypes=getattr(args, "num_prototypes", 4),
            temperature_init=0.5,
            stage2=bool(getattr(args, "stage2", True)),
            patch_refinement=bool(getattr(args, "patch_refinement", True)),
            foreground_branch=bool(getattr(args, "foreground_branch", True)),
            background_branch=bool(getattr(args, "background_branch", True)),
            cluster_attention=bool(getattr(args, "cluster_attention", True)),
            density_attention=bool(getattr(args, "density_attention", True)),
            deep_supervision=bool(getattr(args, "deep_supervision", True)),
            channel_spatial_attention=bool(getattr(args, "channel_spatial_attention", True)),
            skip_attention=bool(getattr(args, "skip_attention", True)),
            residual=bool(getattr(args, "residual", True)),
            channel_gate=bool(getattr(args, "channel_gate", False)),
            dropout=getattr(args, "dropout", 0.05),
            patch_chunk_size=0,
        )
        model = amal.AMALNet(cfg)
        loss_cfg = amal.LossConfig(
            tversky_alpha=float(getattr(args, "tversky_alpha", 0.3)),
            tversky_beta=float(getattr(args, "tversky_beta", 0.7)),
        )
        criterion = amal.AMALLoss(cfg, loss_cfg)
        rank_w = float(getattr(args, "score_rank_weight", 0.0) or 0.0)
        div_w = float(getattr(args, "score_div_weight", 0.0) or 0.0)
        if rank_w or div_w:
            criterion = AMALProLoss(criterion, rank_weight=rank_w, div_weight=div_w)
        return model, criterion, {
            "name": model_name,
            "is_amal": True,
            "config": cfg,
            "score_rank_weight": rank_w,
            "score_div_weight": div_w,
        }

    if model_name == "amal_unified":
        if uni is None:
            raise SystemExit(
                "amal_unified unavailable: unified architecture source not "
                f"found at {_OURUNIFIED_PY}. Set OURUNIFIED_MODELS_PY to the "
                "livercode model/ourunified.py path."
            )
        # Loaded verbatim; only the constructor defaults the pre-revision
        # training used (soft_bg pancreas runs) are pinned here so the
        # architecture matches those runs exactly.
        model = uni.ImprovedTwoStageSegmentation(
            in_channels=3,
            base_channels=32,
            num_classes=3,
            deep_supervision=True,
            imp_modulation_gamma=0.3,
            use_channel_transformer=True,
            use_patch_refinement=True,
        )
        criterion = AMALUnifiedLoss(
            num_classes=3,
            tversky_alpha=float(getattr(args, "tversky_alpha", 0.3)),
            tversky_beta=float(getattr(args, "tversky_beta", 0.7)),
        )
        return model, criterion, {
            "name": model_name,
            "is_amal": False,
            "impl": "amal_unified",
        }

    if model_name == "coma":
        model = coma.COMANetPaperReproduction(
            in_channels=3, num_classes=3, imagenet_weights=None
        )
        criterion = coma.COMANetLoss(foreground_ids=(1, 2))
        return model, criterion, {"name": model_name, "is_amal": False}

    if model_name == "uctransnet":
        repo = (
            getattr(args, "uctransnet_repo", None)
            or os.environ.get("UCTRANSNET_REPO")
        )
        impl = (
            getattr(args, "uctransnet_impl", None)
            or os.environ.get("UCTRANSNET_IMPL", "auto")
        ).lower()
        repo_ok = bool(repo) and (
            Path(repo).resolve() / "nets" / "UCTransNet.py"
        ).is_file() and (
            Path(repo).resolve() / "nets" / "CTrans.py"
        ).is_file()

        if impl == "auto":
            impl = "official" if repo_ok else "local"

        if impl == "official":
            if not repo_ok:
                raise SystemExit(
                    "UCTransNet --uctransnet-impl official requires the authors' "
                    "checkout. Pass --uctransnet-repo /path/to/UCTransNet or set "
                    "UCTRANSNET_REPO. Expected files: <repo>/nets/UCTransNet.py "
                    "and <repo>/nets/CTrans.py"
                )
            # Some official checkouts use absolute `from nets.CTrans import ...`
            # rather than a package-relative import.  Adding the checkout root
            # to sys.path is harmless for relative imports and fixes absolute ones.
            repo_root = str(Path(repo).resolve())
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            model = uct.UCTransNetOfficial(
                repo_dir=repo,
                in_channels=3,
                num_classes=3,
                image_size=int(getattr(args, "image_size", 256)),
            )
            criterion = amal.StandardSegLoss(amal.LossConfig(), deep_supervision=False)
            return model, criterion, {
                "name": model_name,
                "is_amal": False,
                "impl": "official",
                "repo": str(repo),
            }

        if impl == "local":
            if not LOCAL_UCTRANSNET_AVAILABLE:
                raise SystemExit(
                    "UCTransNet local implementation is unavailable: "
                    + uctransnet_local_description()
                )
            model = UCTransNetLocalAdapter(
                in_channels=3,
                num_classes=3,
                base_ch=getattr(args, "uctransnet_base_ch", 64),
            )
            criterion = amal.StandardSegLoss(amal.LossConfig(), deep_supervision=False)
            return model, criterion, {
                "name": model_name,
                "is_amal": False,
                "impl": "local",
                "local_source": uctransnet_local_description(),
            }

        raise SystemExit(
            f"Unknown UCTransNet implementation {impl!r}; choose auto/official/local"
        )

    if model_name in COMPARISON_ZOO_KEYS:
        if compzoo is None:
            raise SystemExit(
                "Comparison model definitions not found at "
                f"{_COMPARISON_ZOO_PY}. Set COMPARISON_MODELS_PY to the "
                "livercode model/comparison_models.py path."
            )
        cfg = dict(compzoo.MODEL_CONFIGS[model_name])
        model = compzoo.get_model(
            model_name, in_channels=3, num_classes=3, **cfg
        )
        loss_cfg = amal.LossConfig(
            tversky_alpha=float(getattr(args, "tversky_alpha", 0.3)),
            tversky_beta=float(getattr(args, "tversky_beta", 0.7)),
        )
        criterion = amal.StandardSegLoss(loss_cfg, deep_supervision=False)
        return model, criterion, {
            "name": model_name,
            "is_amal": False,
            "impl": "livercode-comparison",
        }

    # nnunet2d
    impl = getattr(args, "nnunet_impl", None) or os.environ.get("NNUNET_IMPL", "plain")
    if impl == "dna":
        network = nnunet.official_plain_architecture(
            in_channels=3,
            num_classes=3,
            features=(32, 64, 128, 256, 320, 320),
            deep_supervision=True,
        )
        model = _DNAWrapper(network)
    else:
        model = nnunet.NNUNet2DArchitecture(
            in_channels=3,
            num_classes=3,
            features=(32, 64, 128, 256, 320, 320),
            deep_supervision=True,
        )
    criterion = amal.StandardSegLoss(amal.LossConfig(), deep_supervision=True)
    return model, criterion, {"name": model_name, "is_amal": False, "impl": impl}


def _normalise_output(output) -> dict:
    if isinstance(output, dict):
        value = output.get("stage2_final")
        if isinstance(value, (list, tuple)):
            normalised = dict(output)
            normalised["stage2_final"] = value[0]
            normalised["deep_supervision"] = list(value[1:])
            return normalised
        return output
    if isinstance(output, (list, tuple)):
        return {"stage2_final": output[0], "deep_supervision": list(output[1:])}
    return {"stage2_final": output}


def forward_model(model: nn.Module, model_name: str, images: torch.Tensor,
                  epoch: int, args: Any) -> dict:
    """Run a forward pass and normalise to a dict with 'stage2_final'."""
    if model_name == "amal":
        warmup_all = int(getattr(args, "warmup_all_epochs", 0)) > 0 and epoch <= int(
            getattr(args, "warmup_all_epochs", 0)
        )
        return _normalise_output(model(images, warmup_all=warmup_all))
    return _normalise_output(model(images))


def criterion_total(loss_output) -> torch.Tensor:
    if isinstance(loss_output, dict):
        if "total" not in loss_output:
            raise KeyError(f"Loss dict has no 'total' key: {sorted(loss_output)}")
        return loss_output["total"]
    return loss_output
