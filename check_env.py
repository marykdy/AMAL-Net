#!/usr/bin/env python3
"""Pre-flight check for the modelreview comparison.

Run this inside the GPU container before launching the matrix:
    python3 check_env.py --uctransnet-repo ./UCTransNet

It checks: Python/torch/CUDA, model imports, model-file checksums, the three
3-way splits and their label convention, and the official UCTransNet checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_PARENT = Path(os.environ.get("DATA_ROOT", "./data"))
DATASETS = ("liver", "pancreas", "lung")
MODEL_FILES = ("AMALNet_Main.py", "COMANet_Model.py",
               "nnUNet2D_Model.py", "UCTransNet_Official.py")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment check")
    parser.add_argument("--uctransnet-repo", default=os.environ.get("UCTRANSNET_REPO") or None)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT") or None,
                        help="Parent containing {liver,pancreas,lung}_3way")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if the official UCTransNet checkout is missing.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checksums() -> list[str]:
    checksum_file = HERE / "model_checksums.txt"
    problems: list[str] = []
    if not checksum_file.is_file():
        print("[WARN] model_checksums.txt missing; cannot verify model files")
        return problems
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        expected, name = parts[0], parts[-1]
        path = HERE / name
        if not path.is_file():
            problems.append(f"missing model file: {name}")
            continue
        actual = sha256(path)
        if actual != expected:
            problems.append(f"checksum changed: {name}")
            print(f"[WARN] checksum changed: {name}")
        else:
            print(f"[OK]   checksum: {name}")
    return problems


def check_data(data_root: str | None) -> list[str]:
    problems: list[str] = []
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        print(f"[FAIL] PIL/numpy unavailable: {exc}")
        return ["PIL/numpy unavailable"]

    sys.path.insert(0, str(HERE))
    try:
        from common import SegDataset
    except Exception as exc:
        print(f"[FAIL] cannot import common.py: {exc}")
        return ["common.py import failed"]

    parent = Path(data_root) if data_root else DEFAULT_DATA_PARENT
    for dataset in DATASETS:
        root = parent / f"{dataset}_3way"
        print(f"\n[data] {dataset}: {root}")
        if not root.is_dir():
            problems.append(f"{dataset}: missing {root}")
            print(f"[FAIL] missing root")
            continue
        manifest = root / "split_manifest.json"
        if manifest.is_file():
            import json
            info = json.loads(manifest.read_text(encoding="utf-8"))
            overlap = info.get("splits", {}).get("overlap", {})
            print(f"[OK]   manifest overlap={overlap}")
        else:
            print("[WARN] split_manifest.json missing")
        for split in SPLITS:
            try:
                ds = SegDataset(root / f"{split}_images", root / f"{split}_masks",
                                image_size=256, augment=False)
                # Force validation of a few masks; __init__ alone does not read them.
                for i in range(min(5, len(ds))):
                    _img, _mask, _name = ds[i]
                print(f"[OK]   {split}: {len(ds):,} pairs, labels 0/1/2")
            except Exception as exc:
                problems.append(f"{dataset}/{split}: {exc}")
                print(f"[FAIL] {split}: {exc}")
    return problems


def check_uctransnet(repo: str | None) -> bool:
    if not repo:
        print("[INFO] official UCTransNet checkout not provided; "
              "auto will use the project-local reimplementation if available")
        return False
    root = Path(repo)
    ok = (root / "nets" / "UCTransNet.py").is_file() and (root / "nets" / "CTrans.py").is_file()
    if ok:
        print(f"[OK]   UCTransNet official checkout: {root}")
    else:
        print(f"[WARN] UCTransNet checkout missing nets/UCTransNet.py or nets/CTrans.py: {root}")
    return ok


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("modelreview environment check")
    print("=" * 78)
    print(f"python: {sys.version.split()[0]}  executable={sys.executable}")
    print(f"cwd:    {Path.cwd()}")
    print(f"root:   {HERE}")

    torch_ok = False
    try:
        import torch
        torch_ok = True
        print(f"[OK]   torch {torch.__version__}")
        print(f"[OK]   torchvision import: ", end="")
        import torchvision
        print(torchvision.__version__)
        print(f"[OK]   cuda available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"       GPU {i}: {props.name} {props.total_memory / 1024**3:.1f} GiB")
    except Exception as exc:
        print(f"[FAIL] torch/torchvision unavailable: {exc}")

    for name in ("PIL", "numpy", "scipy", "einops"):
        print(f"[{'OK' if importlib.util.find_spec(name) else 'WARN'}]   package: {name}")

    print("\n[model files]")
    checksum_problems = check_checksums()
    missing_model_files = [p for p in checksum_problems if p.startswith("missing model file")]
    model_import_ok = False
    if torch_ok:
        sys.path.insert(0, str(HERE))
        try:
            import AMALNet_Main  # noqa: F401
            import COMANet_Model  # noqa: F401
            import nnUNet2D_Model  # noqa: F401
            import UCTransNet_Official  # noqa: F401
            model_import_ok = True
            print("[OK]   all four model modules import")
        except Exception as exc:
            print(f"[FAIL] model import: {exc}")

    print("\n[data]")
    data_problems = check_data(args.data_root)

    print("\n[UCTransNet]")
    uct_ok = check_uctransnet(args.uctransnet_repo)
    local_uct_ok = False
    if torch_ok:
        sys.path.insert(0, str(HERE))
        try:
            from baselines_adapters import (  # noqa: F401
                LOCAL_UCTRANSNET_AVAILABLE,
                uctransnet_local_description,
            )
            local_uct_ok = bool(LOCAL_UCTRANSNET_AVAILABLE)
            print(f"[{'OK' if local_uct_ok else 'WARN'}]   local UCTransNet: "
                  f"{uctransnet_local_description()}")
        except Exception as exc:
            print(f"[WARN] local UCTransNet adapter unavailable: {exc}")
    print(f"[INFO] UCTransNet ready: official={uct_ok} local={local_uct_ok} "
          f"-> auto will use {'official' if uct_ok else ('local' if local_uct_ok else 'nothing')}")
    print(f"[INFO] nnunetv2={bool(importlib.util.find_spec('nnunetv2'))} "
          f"dynamic_network_architectures={bool(importlib.util.find_spec('dynamic_network_architectures'))}")

    print("\n[commands]")
    print("  cd <repo-root>")
    print("  python3 check_env.py --uctransnet-repo ./UCTransNet")
    print("  python3 run_matrix.py --manifest manifests/quick.tsv --gpus 0,1,2,3,4,5 \\")
    print("      --uctransnet-repo ./UCTransNet")

    critical = (not torch_ok) or (not model_import_ok) or bool(missing_model_files) or bool(data_problems)
    if critical:
        print("\n[RESULT] critical problems found; fix the [FAIL] lines above.")
        sys.exit(1)
    if args.strict and not (uct_ok or local_uct_ok):
        print("\n[RESULT] --strict: no UCTransNet implementation available "
              "(official checkout and local reimplementation both missing).")
        sys.exit(1)
    print("\n[RESULT] ready (see warnings above).")


if __name__ == "__main__":
    main()
