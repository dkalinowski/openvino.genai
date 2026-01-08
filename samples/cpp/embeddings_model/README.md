# EmbeddingsModel Sample

This sample demonstrates how to use the `EmbeddingsModel` public API to compute text embeddings from token IDs.

## Overview

The `EmbeddingsModel` class provides a simple interface to load and use text embedding models. It uses the PIMPL (Pointer to Implementation) pattern to hide internal details and maintain ABI stability.

## Features

- Load text embeddings model from a model directory
- Compute embeddings for input token IDs
- Support for different target devices (CPU, GPU, etc.)
- Clean, simple API

## Prerequisites

- OpenVINO GenAI library
- A VLM model directory containing `openvino_text_embeddings_model.xml`

## Usage

```bash
./embeddings_model_sample <model_dir> [device]
```

### Arguments

- `model_dir` - Directory containing the text embeddings model (e.g., from a VLM model)
- `device` - (optional) Target device for inference (default: CPU)

### Example

```bash
# Using CPU (default)
./embeddings_model_sample ./vlm_model_dir

# Using GPU
./embeddings_model_sample ./vlm_model_dir GPU
```

## Code Example

```cpp
#include "openvino/genai/visual_language/embeddings_model.hpp"

// Load the embeddings model
ov::genai::EmbeddingsModel embeddings_model(model_dir, "CPU");

// Create input token IDs tensor
ov::Tensor input_ids(ov::element::i64, {batch_size, sequence_length});
// ... fill with token IDs ...

// Compute embeddings
ov::Tensor embeddings = embeddings_model.infer(input_ids);

// Use the embeddings (shape: [batch_size, sequence_length, hidden_size])
```

## API Reference

### Constructor

```cpp
EmbeddingsModel(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& properties = {});
```

Creates an EmbeddingsModel from a model directory.

### Methods

#### `infer`

```cpp
ov::Tensor infer(const ov::Tensor& input_ids);
```

Computes embeddings for the given input token IDs.

- **Input**: Token IDs tensor with shape `[batch_size, sequence_length]`
- **Output**: Embeddings tensor with shape `[batch_size, sequence_length, hidden_size]`

## Notes

- The model directory should contain the `openvino_text_embeddings_model.xml` file and associated weights
- The embedding scale factor is automatically read from the model's config.json if present
- This API is designed to be ABI-stable across library versions
