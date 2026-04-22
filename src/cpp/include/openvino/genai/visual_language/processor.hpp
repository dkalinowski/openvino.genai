// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief Structured inputs for VLM generation produced by VLMProcessor.
/// Holds the merged text + vision embeddings ready to be consumed by
/// VLMPipeline::generate().
struct OPENVINO_GENAI_EXPORTS Embeddings {
    /// Merged embeddings of text tokens and vision features.
    /// Shape: [1, sequence_length, hidden_size].
    ov::Tensor inputs_embeds;
};

/// @brief A processor for Visual Language Models that handles vision
/// encoding, text tokenization, and embedding merging.
///
/// Analogous to transformers's AutoProcessor — combines an image
/// processor (VisionEncoder) with a tokenizer/embedder to produce
/// Embeddings that a VLMPipeline can consume.
class OPENVINO_GENAI_EXPORTS VLMProcessor {
public:
    /// @brief Construct a processor from a directory with vision encoder,
    /// text embeddings model, tokenizer, and configs.
    /// @param models_path Directory with the exported model files.
    /// @param device Inference device for vision encoder and embeddings model.
    /// @param properties Device configuration properties.
    VLMProcessor(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    ~VLMProcessor();

    /// @brief Prepare Embeddings for VLM generation from a text prompt
    /// and optional images / videos.
    /// @param prompt Pre-formatted text prompt. Chat template is not
    ///               applied here.
    /// @param images RGB image tensors.
    /// @param videos Video frame tensors.
    Embeddings embed(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images = {},
        const std::vector<ov::Tensor>& videos = {}
    );

    /// @brief Get the underlying tokenizer.
    Tokenizer get_tokenizer() const;

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
    friend class VLMPipeline;
};

}  // namespace ov::genai
