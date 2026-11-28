#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <hf-username> <space-name> [--docker]"
  echo "Example: $0 OmRajput977 deadlock-doom"
  exit 1
fi

HF_USER="$1"
SPACE_NAME="$2"
MODE="${3:-gradio}"
SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
TMP_DIR="$(mktemp -d /tmp/deadlock_hf_space_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Staging Hugging Face Space (${MODE}) bundle in $TMP_DIR ..."
cp -R env maps submission training "$TMP_DIR/"
find "$TMP_DIR" -name "__pycache__" -type d -exec rm -rf {} +
cp .gitattributes "$TMP_DIR/"

if [ "$MODE" = "--docker" ]; then
  cp -R docker "$TMP_DIR/"
  cp deploy/hf_space/Dockerfile "$TMP_DIR/Dockerfile"
  cp deploy/hf_space/README.md "$TMP_DIR/README.md"
else
  cp deploy/hf_gradio/README.md "$TMP_DIR/README.md"
  cp deploy/hf_gradio/packages.txt "$TMP_DIR/packages.txt"
  cp deploy/hf_gradio/requirements.txt "$TMP_DIR/requirements.txt"
  cp deploy/hf_gradio/app.py "$TMP_DIR/app.py"
fi

cd "$TMP_DIR"
git init -b main
git lfs install
git lfs track "*.pt"
git add .
git commit -m "Deploy DeadLock live DOOM Gradio app"
git remote add space "$SPACE_URL"

echo "Pushing to $SPACE_URL (use your HF username and Write Access Token as password)..."
git push --force space main

echo ""
echo "Deployed! Your live public URL will be ready once the Space finishes building:"
echo "  https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
echo "  Direct app URL: https://${HF_USER}-${SPACE_NAME}.hf.space"
