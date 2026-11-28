#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <hf-username> <space-name>"
  echo "Example: $0 omsrajput977 deadlock-doom"
  exit 1
fi

HF_USER="$1"
SPACE_NAME="$2"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
TMP_DIR="$(mktemp -d /tmp/deadlock_hf_space_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Staging Hugging Face Space bundle in $TMP_DIR ..."
cp -R docker env maps submission training "$TMP_DIR/"
cp .gitattributes "$TMP_DIR/"
cp deploy/hf_space/Dockerfile "$TMP_DIR/Dockerfile"
cp deploy/hf_space/README.md "$TMP_DIR/README.md"

cd "$TMP_DIR"
git init -b main
git lfs install
git lfs track "*.pt"
git add .
git commit -m "Deploy DeadLock live DOOM web demo"
git remote add space "$SPACE_URL"

echo "Pushing to $SPACE_URL (enter your Hugging Face username & Access Token when prompted)..."
git push --force space main

echo ""
echo "Deployed! Your live public URL will be ready once the Space finishes building:"
echo "  https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
echo "  Direct app URL: https://${HF_USER}-${SPACE_NAME}.hf.space"
