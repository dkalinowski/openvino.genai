# InputsEmbedder Sample

This sample demonstrates how to use the `InputsEmbedder` public API to compute combined text and vision embeddings for Visual Language Models (VLMs).

## Overview

The `InputsEmbedder` class provides a unified interface to:
1. Encode images using a vision encoder
2. Tokenize text prompts
3. Combine text and image embeddings into a single tensor

This is useful when you want fine-grained control over the embedding process or need to integrate with custom language model inference.

## Features

- Load VLM models (vision encoder + text embeddings model)
- Encode images into embeddings
- Combine text and image embeddings
- Support for different target devices (CPU, GPU, etc.)
- PIMPL pattern for ABI stability

## Prerequisites

- OpenVINO GenAI library
- A VLM model directory containing:
  - `config.json` - VLM configuration
  - `openvino_text_embeddings_model.xml` - Text embeddings model
  - `openvino_vision_embeddings_model.xml` - Vision encoder model
  - Other model-specific files

## Usage

```bash
./inputs_embedder_sample <model_dir> [device]
```

### Arguments

- `model_dir` - Directory containing the VLM model files
- `device` - (optional) Target device for inference (default: CPU)

### Example

```bash
# Using CPU (default)
./inputs_embedder_sample ./vlm_model

# Using GPU
./inputs_embedder_sample ./vlm_model GPU
```

## Code Example

```cpp
#include "openvino/genai/visual_language/inputs_embedder.hpp"

// Create InputsEmbedder from model directory
ov::genai::InputsEmbedder inputs_embedder(model_dir, "CPU");

// Load and encode images
std::vector<ov::Tensor> images = {load_image("photo.jpg")};
auto encoded_images = inputs_embedder.encode_images(images);

// Create prompt with image placeholder
std::string prompt = "<image>\nDescribe this image.";

// Get combined embeddings
ov::Tensor inputs_embeds = inputs_embedder.get_inputs_embeds(prompt, encoded_images);

// Use embeddings with language model for generation...
```

## API Reference

### Constructors

```cpp
// Load from model directory
InputsEmbedder(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& device_config = {});

// Use pre-loaded components
InputsEmbedder(
    const Tokenizer& tokenizer,
    const VisionEncoder& vision_encoder,
    const EmbeddingsModel& embeddings_model,
    const std::filesystem::path& config_dir_path);
```

### Methods

#### `encode_images`

```cpp
std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images);
```

Encodes images into embeddings using the vision encoder.

- **Input**: Vector of image tensors with shape `[1, H, W, C]` or `[H, W, C]` in NHWC layout
- **Output**: Vector of `EncodedImage` structures containing the embeddings

#### `get_inputs_embeds`

```cpp
ov::Tensor get_inputs_embeds(
    const std::string& prompt,
    const std::vector<EncodedImage>& encoded_images);
```

Computes combined text and image embeddings.

- **Input**: Text prompt (may contain image placeholders) and pre-encoded images
- **Output**: Combined embeddings tensor with shape `[batch, sequence_length, hidden_size]`

#### `get_tokenizer`

```cpp
Tokenizer get_tokenizer() const;
```

Returns the tokenizer used by this InputsEmbedder.

## Notes

- The `InputsEmbedder` uses the PIMPL pattern to hide implementation details
- This API is designed to be ABI-stable across library versions
- Image placeholders in prompts (e.g., `<image>`) are model-specific
