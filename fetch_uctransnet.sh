#!/usr/bin/env bash
# Optional helper: fetch the authors' UCTransNet checkout used by
# UCTransNet_Official.py.  Run on a machine with network access, or copy an
# existing checkout to the same destination.
set -euo pipefail

DEST="${1:-./UCTransNet}"
REPO="${UCTRANSNET_GIT_URL:-https://github.com/McGregorWwww/UCTransNet.git}"

if [[ -d "$DEST/nets" ]]; then
    echo "[fetch] already present: $DEST"
    exit 0
fi

git clone --depth 1 "$REPO" "$DEST"
echo "[fetch] cloned into $DEST"
echo "[fetch] verifying required files:"
ls -l "$DEST/nets/UCTransNet.py" "$DEST/nets/CTrans.py"
