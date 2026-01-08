# Vision Encoder Sample

This sample demonstrates the usage of the `VisionEncoder` public API to encode images into embeddings that can be used with Visual Language Models (VLMs).

## Overview

The `VisionEncoder` class provides a simple interface to:
1. Load a vision encoder model from a directory
2. Encode images into embeddings

## Prerequisites

- OpenVINO GenAI library
- OpenCV (for image loading)
- A VLM model with vision encoder (e.g., LLaVA, MiniCPM-V, InternVL)

## Model Directory Structure

The model directory should contain:
- `config.json` - Model configuration with model type information
- `openvino_vision_embeddings_model.xml` - Vision encoder model
- `preprocessor_config.json` - Image preprocessing configuration

## Building

```bash
cmake -B build
cmake --build build --target vision_encoder_sample
```

## Usage

```bash
./vision_encoder_sample <MODEL_DIR> <IMAGE_FILE>
```

### Arguments
- `MODEL_DIR` - Path to the VLM model directory
- `IMAGE_FILE` - Path to input image (JPEG, PNG, etc.)

### Example

```bash
./vision_encoder_sample /path/to/llava-model /path/to/image.jpg
```

## Output

The sample outputs:
- Embeddings tensor shape
- Resized source size
- Original image size
- Number of image tokens
- Patches grid (if applicable)

## Code Walkthrough

### Creating the VisionEncoder

```cpp
#include <openvino/genai/visual_language/vision_encoder.hpp>

// Create VisionEncoder - model type is auto-detected from config.json
ov::genai::VisionEncoder encoder(model_dir, "CPU");
```

### Encoding an Image

```cpp
// Load image as ov::Tensor in NHWC format (uint8)
ov::Tensor image = load_image(image_path);

// Encode to get embeddings
ov::genai::EncodedImage encoded = encoder.encode(image);

// Access the embeddings
ov::Tensor embeddings = encoded.resized_source;
```

### EncodedImage Structure

The `EncodedImage` structure contains:
- `resized_source` - The actual embeddings tensor
- `resized_source_size` - Size after resizing (divided by patch size)
- `original_image_size` - Original input image dimensions
- `num_image_tokens` - Number of tokens to be added to the prompt
- `patches_grid` - Grid dimensions for models that use patching

## See Also

- [VLMPipeline](../visual_language_chat/) - Full Visual Language Model pipeline
- [OpenVINO GenAI Documentation](https://docs.openvino.ai/nightly/openvino-genai-guide.html)
