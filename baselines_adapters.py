"""Adapters that make the existing project baselines usable in this harness.

The user-provided files in this folder are the primary definitions:
    AMALNet_Main.py
    COMANet_Model.py
    nnUNet2D_Model.py
    UCTransNet_Official.py

UCTransNet_Official.py is only a loader for the authors' checkout.  The
project already contains a self-contained UCTransNet reimplementation at
`<livercode>/model/comparison_models.py`.  This adapter exposes that local
implementation through the same dict-output interface, so the comparison can
run even when the official checkout is not available.

The local implementation is labelled `local` everywhere (train log, eval JSON,
README); it is never silently presented as the official code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch.nn as nn

LIVERCODE_ROOT = Path(os.environ.get("LIVERCODE_ROOT", "."))

try:
    if str(LIVERCODE_ROOT) not in sys.path:
        sys.path.insert(0, str(LIVERCODE_ROOT))
    from model.comparison_models import UCTransNet as _LocalUCTransNet
    LOCAL_UCTRANSNET_AVAILABLE = True
    LOCAL_UCTRANSNET_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the runtime mount
    _LocalUCTransNet = None
    LOCAL_UCTRANSNET_AVAILABLE = False
    LOCAL_UCTRANSNET_ERROR = str(exc)


class UCTransNetLocalAdapter(nn.Module):
    """Project-local UCTransNet implementation, normalised to dict output."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_ch: int = 64):
        super().__init__()
        if _LocalUCTransNet is None:
            raise ImportError(
                "Local UCTransNet is unavailable: "
                f"{LOCAL_UCTRANSNET_ERROR}. Expected {LIVERCODE_ROOT}/model/comparison_models.py "
                "or set LIVERCODE_ROOT."
            )
        self.network = _LocalUCTransNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=base_ch,
        )

    def forward(self, x):
        return {"stage2_final": self.network(x)}


def uctransnet_local_description() -> str:
    if LOCAL_UCTRANSNET_AVAILABLE:
        return f"local reimplementation at {LIVERCODE_ROOT}/model/comparison_models.py"
    return f"unavailable ({LOCAL_UCTRANSNET_ERROR})"
