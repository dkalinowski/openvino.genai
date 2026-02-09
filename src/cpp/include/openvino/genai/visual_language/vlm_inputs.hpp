// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/// @brief Structured inputs for VLM generation, produced by VLMProcessor.
///
/// Contains merged text+vision embeddings, attention mask, and optional
/// model-specific tensors (position IDs, token type IDs). This struct is
/// the single intermediate between VLMProcessor::prepare() and
/// VLMPipeline::generate() / ContinuousBatchingPipeline::add_request().
///
/// Analogous to HuggingFace's BatchFeature returned by a Processor.
struct OPENVINO_GENAI_EXPORTS VLMInputs {
    /// @brief Merged embeddings of text tokens and projected vision features.
    /// Shape: [1, sequence_length, hidden_size].
    /// The vision encoder outputs have already been projected into
    /// the LLM embedding space and spliced at <image>/<video> placeholder positions.
    ov::Tensor inputs_embeds;

    /// @brief Attention mask for the merged sequence.
    /// Shape: [1, sequence_length]. Values: 1 = attend, 0 = ignore.
    ov::Tensor attention_mask;

    /// @brief Position IDs for models requiring explicit positioning
    /// (e.g., Qwen2-VL uses 3D MROPE position IDs).
    /// May be std::nullopt for models that derive positions from attention_mask.
    std::optional<ov::Tensor> position_ids;

    /// @brief Token type IDs for models that distinguish token types
    /// (e.g., Phi-4-MM).
    /// Shape: [1, sequence_length]. May be std::nullopt.
    std::optional<ov::Tensor> token_type_ids;

    /// @brief Rope delta value, used by some models (e.g., Qwen2-VL).
    std::optional<int64_t> rope_delta;

    /// @brief Check whether vision content was encoded into these inputs.
    bool has_vision_content() const {
        return inputs_embeds && inputs_embeds.get_size() > 0;
    }
};

} // namespace ov::genai
