"""Shared data, metrics and utility code for the modelreview comparison.

This module deliberately only *reads* the existing patient-level 3-way splits:
    <data-root>/{liver,pancreas,lung}_3way
No dataset file is modified or written.

Label convention in all three splits (verified from the masks):
    0 = background, 1 = organ (liver / pancreas / lung), 2 = tumor
"""
from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Fixed protocol constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 3
CLASS_NAMES = ("background", "organ", "tumor")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
DEFAULT_RUNS_DIR = Path(os.environ.get("RUNS_DIR", "./runs"))
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep the same determinism/cudnn policy as the existing training scripts.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def dataset_root(dataset: str, data_root: Optional[str] = None) -> Path:
    """Return the 3-way split root for a dataset."""
    if data_root:
        return Path(data_root)
    return DEFAULT_DATA_ROOT / f"{dataset}_3way"


def _index_files(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("._")
    ]
    mapping: dict[str, Path] = {}
    for path in files:
        if path.stem in mapping:
            raise ValueError(
                f"Duplicate stem {path.stem!r} in {directory}: "
                f"{mapping[path.stem].name} and {path.name}"
            )
        mapping[path.stem] = path
    return mapping


class SegDataset(Dataset):
    """Image/mask dataset paired by filename stem, never by sorted position.

    The original liver bug paired images and masks by sorted index.  Here the
    stems must be identical, which is true for the corrected *_3way splits.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = 256,
        augment: bool = False,
        aug: str = "standard",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.aug = aug

        images = _index_files(self.image_dir, IMAGE_SUFFIXES)
        masks = _index_files(self.mask_dir, MASK_SUFFIXES)
        if set(images) != set(masks):
            missing_masks = sorted(set(images) - set(masks))[:5]
            missing_images = sorted(set(masks) - set(images))[:5]
            raise ValueError(
                f"Image/mask stem mismatch in {self.image_dir} vs {self.mask_dir}. "
                f"images without masks={missing_masks}, masks without images={missing_images}"
            )
        self.items = [(images[stem], masks[stem]) for stem in sorted(images)]
        if not self.items:
            raise ValueError(f"No image/mask pairs found in {self.image_dir}")
        self._tumor_flags: Optional[list[bool]] = None
        # Validate a sample once at construction.  Doing np.unique on every
        # slice in every epoch would dominate the data-loading cost.
        for _, mask_path in self.items[: min(32, len(self.items))]:
            values = set(np.unique(np.array(Image.open(mask_path).convert("L"))).tolist())
            unexpected = values - {0, 1, 2}
            if unexpected:
                raise ValueError(
                    f"Mask {mask_path} contains unexpected labels {sorted(unexpected)}; "
                    "expected only 0/1/2. Refusing to silently remap the dataset."
                )

    def __len__(self) -> int:
        return len(self.items)

    def _load_pair(self, index: int) -> tuple[Image.Image, Image.Image]:
        image_path, mask_path = self.items[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Joint geometric augmentation; photometric augmentation only on image.

        This is intentionally a shared, moderate policy for all four models:
        horizontal flip, small rotation, mild colour jitter and mild blur.
        It is NOT the strong CutMix/elastic recipe used for the submitted
        AMAL-Net runs; a common policy is required for a fair comparison.
        """
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            angle = random.uniform(-10.0, 10.0)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        if random.random() < 0.3:
            image = transforms.ColorJitter(0.10, 0.10, 0.10, 0.02)(image)
        if random.random() < 0.10:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        return image, mask

    def _to_tensor(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR,
        )(image)
        mask = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )(mask)
        image_t = transforms.ToTensor()(image)
        image_t = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(image_t)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t

    def __getitem__(self, index: int):
        image, mask = self._load_pair(index)
        if self.augment and self.aug != "none":
            image, mask = self._augment(image, mask)
        image_t, mask_t = self._to_tensor(image, mask)
        return image_t, mask_t, self.items[index][0].name

    def scan_tumor_flags(self) -> list[bool]:
        """Return whether each mask contains at least one tumor pixel (class 2)."""
        if self._tumor_flags is None:
            def _check(path: Path) -> bool:
                arr = np.array(Image.open(path).convert("L"))
                values = set(np.unique(arr).tolist())
                unexpected = values - {0, 1, 2}
                if unexpected:
                    raise ValueError(
                        f"Mask {path} contains unexpected labels {sorted(unexpected)}; "
                        "expected only 0/1/2."
                    )
                return bool((arr == 2).any())

            with ThreadPoolExecutor(max_workers=8) as pool:
                self._tumor_flags = list(pool.map(_check, [m for _, m in self.items]))
        return self._tumor_flags


def build_train_loader(
    dataset: str,
    data_root: Optional[str] = None,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 8,
    tumor_oversample: float = 3.0,
    aug: str = "standard",
) -> tuple[DataLoader, SegDataset]:
    root = dataset_root(dataset, data_root)
    ds = SegDataset(root / "train_images", root / "train_masks",
                    image_size=image_size, augment=True, aug=aug)
    if tumor_oversample and tumor_oversample > 1.0:
        flags = ds.scan_tumor_flags()
        weights = [float(tumor_oversample) if flag else 1.0 for flag in flags]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(ds),
            replacement=True,
        )
        print(f"[data] tumor oversampling: {sum(flags)}/{len(flags)} positive, "
              f"weight={tumor_oversample}", flush=True)
        loader = DataLoader(
            ds, batch_size=batch_size, sampler=sampler, drop_last=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    else:
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    return loader, ds


def build_eval_loader(
    dataset: str,
    split: str = "val",
    data_root: Optional[str] = None,
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 8,
) -> tuple[DataLoader, SegDataset]:
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train/val/test")
    root = dataset_root(dataset, data_root)
    ds = SegDataset(root / f"{split}_images", root / f"{split}_masks",
                    image_size=image_size, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return loader, ds


class SegMetrics:
    """Pixel confusion-matrix metrics (same definitions as the revision evaluator)."""

    def __init__(self, num_classes: int = NUM_CLASSES, device: str | torch.device = "cuda"):
        self.num_classes = int(num_classes)
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    def reset(self) -> None:
        self.cm.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.dim() == 4:
            pred = pred.argmax(1)
        valid = (target >= 0) & (target < self.num_classes)
        idx = self.num_classes * target[valid] + pred[valid]
        self.cm += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict:
        cm = self.cm.detach().cpu().float().numpy()
        dice, iou = [], []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            dice.append(float(2 * tp / (2 * tp + fp + fn + 1e-10)))
            iou.append(float(tp / (tp + fp + fn + 1e-10)))
        return {
            "mean_dice": float(np.mean(dice)),
            "mean_iou": float(np.mean(iou)),
            "dice_bg": dice[0],
            "dice_organ": dice[1],
            "dice_tumor": dice[2],
            "iou_bg": iou[0],
            "iou_organ": iou[1],
            "iou_tumor": iou[2],
            "confusion_matrix": cm.tolist(),
            "n_pixels": int(cm.sum()),
        }


def cosine_warmup_lr(epoch: int, total: int, warmup: int, base_lr: float,
                     min_lr: float) -> float:
    """epoch is 0-based."""
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def environment_default(name: str, default):
    value = os.environ.get(name)
    return default if value in (None, "") else value
