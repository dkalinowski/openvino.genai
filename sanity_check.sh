#!/bin/bash
set -e

BUILD_DIR="/home/dkal/openvino.genai/build"
VENV_DIR="/home/dkal/openvino.genai/.venv"
MODEL_DIR="$HOME/model_server/demos/common/export_models/models/vlm_models_with_export_models/Qwen/Qwen3-VL-4B-Instruct_fp32"
SAMPLE="samples/python/visual_language_chat/visual_language_chat.py"
IMAGE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/common/static/images/zebra.jpeg"
IMAGE_PATH="/tmp/zebra.jpeg"
PROMPT="What is on the image?"
JOBS=32

echo "=== Rebuilding ==="
cmake --build "$BUILD_DIR" --parallel "$JOBS"

echo ""
echo "=== Downloading image ==="
curl -sL "$IMAGE_URL" -o "$IMAGE_PATH"

echo "=== Running sample ==="
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$BUILD_DIR:$PYTHONPATH"
echo "$PROMPT" | python "$BUILD_DIR/../$SAMPLE" "$MODEL_DIR" "$IMAGE_PATH"
echo ""
echo "=== Done ==="
